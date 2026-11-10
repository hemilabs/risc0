// Copyright 2026 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! rv32im eval_check on the Metal HAL, and the rest of `CircuitHal`.
//!
//! R10 (PLAN 4.2): eval_check runs the baseline `eval_check` kernel (R6's translation of CUDA
//! `eval_check_0..3.cu` + `eval_check_kernel.cu` by `cuda2metal.py eval`). The host side mirrors
//! `hal/cuda.rs::eval_check` / `risc0_circuit_rv32im_cuda_eval_check`:
//! - the 458 poly_mix powers are computed on the CPU (`map_pow(poly_mix, POLY_MIX_POWERS)`) and
//!   copied into the HAL-owned `ScratchBuffers::poly_mix` buffer (CUDA copies them into
//!   `__constant__` memory; 7328 B is above the 4 KB `set_bytes` limit, so here it is a
//!   `const device FpExt*` binding);
//! - one dispatch of `ceil(domain / 256)` threadgroups of 256 threads (CUDA `EVAL_CHECK_THREADS`),
//!   then the batch is closed with a sync (end of a circuit phase, PLAN 2.6);
//! - `eval_check_dep` stays the default no-op (one command queue, PLAN 2.0; a second queue is R14).
//!
//! The R7 CPU path (the `hal/cpu.rs` loop over `risc0_circuit_rv32im_cpu_poly_fp` on host views of
//! the shared buffers) is kept as a debugging fallback, selected with
//! `RISC0_METAL_RV32IM_CPU_PHASES=eval` (a comma list of `witgen,accum,eval`).

use rayon::prelude::*;
use risc0_circuit_rv32im_sys::risc0_circuit_rv32im_cpu_poly_fp;
use risc0_core::{
    field::{map_pow, Elem, ExtElem as _, RootsOfUnity as _},
    scope,
};
use risc0_zkp::{
    core::log2_ceil,
    hal::{
        metal::{as_bytes, BufferImpl, Grid, KernelArg, MetalHal, MetalHash},
        AccumPreflight, Buffer as _, CircuitHal,
    },
    INV_RATE,
};

use super::{
    count, cpu_phase, grow, host_slice, phase_mark, MetalCircuitHal, Phase, CPU_PHASES_ENV,
};
use crate::{
    prove::{GLOBAL_MIX, GLOBAL_OUT},
    zirgen::{
        circuit::{ExtVal, Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CODE, REGISTER_GROUP_DATA},
        info::{NUM_POLY_MIX_POWERS, POLY_MIX_POWERS},
    },
};

/// Whether [CPU_PHASES_ENV] selects the CPU fallback for eval_check (read on every call).
fn eval_on_cpu() -> bool {
    cpu_phase("eval")
}

/// Threads per threadgroup: CUDA `EVAL_CHECK_THREADS` (eval_check_kernel.cu), which is also the
/// `[[max_total_threads_per_threadgroup]]` of the translated kernel.
const EVAL_CHECK_THREADS: u64 = 256;

// The baseline kernel's `poly_mix` table has `kNumPolyMixPows` entries (eval_check.cuh), and
// evalfast is generated for the same count (EF_NPC); a circuit change must regenerate both.
const _: () = assert!(NUM_POLY_MIX_POWERS == 458);
const _: () = assert!(std::mem::size_of::<ExtVal>() == 4 * std::mem::size_of::<u32>());
const _: () = assert!(std::mem::size_of::<Val>() == std::mem::size_of::<u32>());

impl<MH: MetalHash> MetalCircuitHal<MH> {
    /// eval_check on the GPU: the baseline `eval_check` kernel (R6 ABI, eval_check.metal):
    /// `0 check, 1 ctrl, 2 data, 3 accum, 4 mix, 5 out` (buffers), `6 rou` (`constant Fp&`),
    /// `7 po2`, `8 domain` (`constant uint&`), `9 poly_mix` (`const device FpExt*`, 458 entries).
    pub(crate) fn eval_check_metal(
        &self,
        check: &BufferImpl<Val>,
        groups: &[&BufferImpl<Val>],
        globals: &[&BufferImpl<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) -> anyhow::Result<()> {
        scope!("metal_eval_check");
        self.ctx.require(Phase::EvalCheck)?;

        let accum = groups[REGISTER_GROUP_ACCUM];
        let ctrl = groups[REGISTER_GROUP_CODE];
        let data = groups[REGISTER_GROUP_DATA];
        let mix = globals[GLOBAL_MIX];
        let out = globals[GLOBAL_OUT];

        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        anyhow::ensure!(
            check.size() == ExtVal::EXT_SIZE * domain,
            "eval_check: check has {} elements, expected {}",
            check.size(),
            ExtVal::EXT_SIZE * domain
        );
        anyhow::ensure!(
            domain <= u32::MAX as usize,
            "eval_check: domain {domain} does not fit u32"
        );
        let rou = Val::ROU_FWD[po2 + EXP_PO2];
        tracing::debug!(
            "eval_check (Metal): steps {steps}, domain {domain}, po2 {po2}, check {}, ctrl {}, \
             data {}, accum {}, mix {}, out {}",
            check.size(),
            ctrl.size(),
            data.size(),
            accum.size(),
            mix.size(),
            out.size()
        );

        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);
        assert_eq!(poly_mix_pows.len(), NUM_POLY_MIX_POWERS);
        let pm_buf = {
            let mut s = self.ctx.scratch();
            grow(&self.hal, &mut s.poly_mix, "poly_mix", NUM_POLY_MIX_POWERS)
        };
        // view_mut flushes and waits if a batch is open, so a previous eval_check that read this
        // buffer has finished before it is overwritten.
        pm_buf.view_mut(|dst| dst[..NUM_POLY_MIX_POWERS].copy_from_slice(&poly_mix_pows));

        let rou = [rou];
        let args = [
            check.as_arg(),
            ctrl.as_arg(),
            data.as_arg(),
            accum.as_arg(),
            mix.as_arg(),
            out.as_arg(),
            KernelArg::Bytes(as_bytes(&rou)),
            KernelArg::Integer(po2 as u32),
            KernelArg::Integer(domain as u32),
            pm_buf.as_arg(),
        ];
        let groups_x = (domain as u64).div_ceil(EVAL_CHECK_THREADS);
        self.hal.dispatch(
            self.ctx.pipeline("eval_check"),
            &args,
            Grid::Groups {
                groups: metal::MTLSize::new(groups_x, 1, 1),
                threads: metal::MTLSize::new(EVAL_CHECK_THREADS, 1, 1),
            },
            &[],
        );
        // End of the phase: close the batch and surface a command-buffer error (PLAN 2.6, 2.9).
        self.hal.try_sync()?;
        tracing::debug!(
            "eval_check (Metal) done: currentAllocatedSize {} B, process peak {} B",
            self.hal.allocated_size(),
            self.hal.peak_allocated_size()
        );
        Ok(())
    }

    /// eval_check on the CPU circuit kernels over host views of the Metal buffers (R7's
    /// delegation; the loop of `hal/cpu.rs`). Debugging fallback, see [CPU_PHASES_ENV].
    pub(crate) fn eval_check_cpu(
        &self,
        check: &BufferImpl<Val>,
        groups: &[&BufferImpl<Val>],
        globals: &[&BufferImpl<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) -> anyhow::Result<()> {
        scope!("metal_eval_check(cpu)");
        self.sync()?;

        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        tracing::debug!("eval_check: domain {domain} (CPU kernels over Metal buffers)");
        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);

        // SAFETY: synced, and nothing is encoded until this call returns.
        let (data, accum, mix, out) = unsafe {
            (
                host_slice(groups[REGISTER_GROUP_DATA]),
                host_slice(groups[REGISTER_GROUP_ACCUM]),
                host_slice(globals[GLOBAL_MIX]),
                host_slice(globals[GLOBAL_OUT]),
            )
        };
        assert_eq!(check.size(), ExtVal::EXT_SIZE * domain);
        let check_ptr = check.as_mut_ptr() as usize;
        let poly_mix_pows = poly_mix_pows.as_slice();
        let args: &[&[Val]] = &[accum, data, out, mix];

        (0..domain).into_par_iter().for_each(|cycle| {
            let args: Vec<*const Val> = args.iter().map(|x| (*x).as_ptr()).collect();
            let mut tot = ExtVal::ZERO;
            unsafe {
                risc0_circuit_rv32im_cpu_poly_fp(
                    cycle,
                    domain,
                    poly_mix_pows.as_ptr(),
                    args.as_ptr(),
                    &mut tot,
                )
            };
            let x = Val::ROU_FWD[po2 + EXP_PO2].pow(cycle);
            // The 3 is the coset shift of the evaluation domain (as hal/cpu.rs).
            let y = (Val::new(3) * x).pow(1 << po2);
            let ret = tot * (y - Val::new(1)).inv();
            let check = check_ptr as *mut Val;
            for i in 0..ExtVal::EXT_SIZE {
                // SAFETY: in bounds (asserted above); each cycle writes disjoint cells.
                unsafe { *check.add(i * domain + cycle) = ret.elems()[i] };
            }
        });
        Ok(())
    }
}

impl<MH: MetalHash> CircuitHal<MetalHal<MH>> for MetalCircuitHal<MH> {
    fn eval_check(
        &self,
        check: &BufferImpl<Val>,
        groups: &[&BufferImpl<Val>],
        globals: &[&BufferImpl<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) {
        count(Phase::EvalCheck);
        phase_mark(&self.hal, "eval_check start");
        let res = if eval_on_cpu() {
            self.eval_check_cpu(check, groups, globals, poly_mix, po2, steps)
        } else {
            self.eval_check_metal(check, groups, globals, poly_mix, po2, steps)
        };
        phase_mark(&self.hal, "eval_check end");
        // CircuitHal::eval_check cannot return an error (CUDA's ffi_wrap panics the same way).
        if let Err(e) = res {
            panic!(
                "rv32im Metal eval_check: {e:#} (set {CPU_PHASES_ENV}=eval to run it on the CPU)"
            );
        }
    }

    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        _ctrl: &BufferImpl<Val>,
        _io: &BufferImpl<Val>,
        _data: &BufferImpl<Val>,
        _mix: &BufferImpl<Val>,
        _accum: &BufferImpl<Val>,
        _steps: usize,
    ) {
        // As hal/cuda.rs: rv32im accumulates in CircuitAccumulator::step_accum (PLAN 8.4).
    }
}
