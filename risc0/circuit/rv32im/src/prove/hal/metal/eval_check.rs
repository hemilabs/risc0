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
//! R12 adds the evalfast kernels (`eval_check_ef.metal`: rv32im-sys
//! `kernels/cuda/evalfast/regen.py --metal`, the CUDA evalfast chain after remat_lu.py converted
//! by to_metal.py). The host mirrors `risc0_circuit_rv32im_cuda_eval_check` of
//! `eval_check_ef.cu`, except that CUDA's one-thread `ef_prep`/`ef_prep_n` kernels are replaced by
//! a CPU table builder ([ef_table], PLAN 4.2): the 627-entry poly_mix table (458 powers, 107
//! products, 62 hoisted out/mix terms from the generated `rk_hoist`), its NBETA-folded copy and
//! the 4 vanishing quotients go into one shared buffer; then `ef_eval_split` (even threadgroups:
//! main chain into `check`, odd: LogUp chain into the scratch partials) and `ef_combine`
//! (`check = (check + partial) * quotient`) in one batch. [EVAL_KERNEL_ENV] selects the kernel
//! (`evalfast`, the default, `evalfast_r12` or `baseline`); all are bit-exact with the CPU (tests).
//!
//! R15e (docs/metal-port/notes/R15e.md): evalfast reads its poly_mix table as `const device`
//! memory instead of `constant` and computes the NBETA-folded rows from the pm rows instead of
//! loading them (~25 % less split-kernel time at po2 20). `evalfast_r12` runs R12's form
//! (`ef_eval_split_r12`, the same generated TU compiled with `-DEF_R12_FORM`) for A/B runs. The
//! PSOs of the kernels that are not the default (`eval_check`, ~60 s cold, and
//! `ef_eval_split_r12`) are built on first use ([lazy_pipeline]) instead of at library load.
//!
//! The R7 CPU path (the `hal/cpu.rs` loop over `risc0_circuit_rv32im_cpu_poly_fp` on host views of
//! the shared buffers) is kept as a debugging fallback, selected with
//! `RISC0_METAL_RV32IM_CPU_PHASES=eval` (a comma list of `witgen,accum,eval`).

use rayon::prelude::*;
use risc0_circuit_rv32im_sys::{metal_evalfast as ef, risc0_circuit_rv32im_cpu_poly_fp};
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

/// Selects the GPU eval_check kernel: `evalfast` (default), `evalfast_r12` (evalfast with R12's
/// poly_mix table forms, the A/B switch of R15e's table change) or `baseline` (R10's translation
/// of CUDA eval_check_0..3.cu, the oracle). Read on every call; unknown values are an error.
pub(crate) const EVAL_KERNEL_ENV: &str = "RISC0_METAL_RV32IM_EVAL";

/// The GPU eval_check kernels ([EVAL_KERNEL_ENV]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EvalKernel {
    Baseline,
    Evalfast,
    /// evalfast compiled with `-DEF_R12_FORM` (`constant` table, NBETA rows loaded).
    EvalfastR12,
}

impl EvalKernel {
    /// The default when [EVAL_KERNEL_ENV] is unset: evalfast (bit-exact with the baseline and the
    /// CPU on every test, ~3x faster at po2 20; docs/metal-port/notes/R12.md).
    pub(crate) const DEFAULT: Self = Self::Evalfast;

    pub(crate) fn parse(v: &str) -> anyhow::Result<Self> {
        match v.trim().to_ascii_lowercase().as_str() {
            "" => Ok(Self::DEFAULT),
            "evalfast" | "fast" | "ef" => Ok(Self::Evalfast),
            "evalfast_r12" | "evalfast-r12" | "r12" => Ok(Self::EvalfastR12),
            "baseline" | "base" => Ok(Self::Baseline),
            _ => anyhow::bail!(
                "{EVAL_KERNEL_ENV}={v:?}: expected `evalfast`, `evalfast_r12` or `baseline`"
            ),
        }
    }

    /// The kernel [EVAL_KERNEL_ENV] selects now.
    pub(crate) fn current() -> anyhow::Result<Self> {
        match std::env::var(EVAL_KERNEL_ENV) {
            Ok(v) => Self::parse(&v),
            Err(_) => Ok(Self::DEFAULT),
        }
    }
}

/// evalfast threadgroup size: `EF_THREADS` of eval_check_ef.metal (its
/// `[[max_total_threads_per_threadgroup]]`, CUDA's default 128). R15e: 64 is within noise of 128,
/// 256 and 1024 are slower (notes/R15e.md).
const EF_THREADS: u64 = 128;

/// The PSO of an rv32im kernel that is not in `METAL_KERNEL_NAMES` (so not built when the
/// library loads): built from `lib` on first use and kept for the process (R15e: the baseline
/// `eval_check` PSO takes ~60 s cold and is not needed unless [EVAL_KERNEL_ENV] selects it).
/// One library per process (`Rv32imKernels` is process-wide), so the cache is keyed by name.
/// Errors are returned, not cached.
pub(crate) fn lazy_pipeline(
    lib: &metal::LibraryRef,
    name: &'static str,
) -> anyhow::Result<&'static metal::ComputePipelineState> {
    use std::{
        collections::HashMap,
        sync::{Mutex, OnceLock},
    };
    static PSOS: OnceLock<Mutex<HashMap<&'static str, &'static metal::ComputePipelineState>>> =
        OnceLock::new();
    // Held while a PSO builds, so two threads never build the same one.
    let mut psos = PSOS.get_or_init(Default::default).lock().unwrap();
    if let Some(pso) = psos.get(name) {
        return Ok(pso);
    }
    let t = std::time::Instant::now();
    let function = lib
        .get_function(name, None)
        .map_err(|e| anyhow::anyhow!("rv32im metallib function {name}: {e}"))?;
    let pso = lib
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("compute pipeline for {name}: {e}"))?;
    tracing::info!(
        "rv32im Metal: built the {name} PSO on first use in {:?}",
        t.elapsed()
    );
    let pso: &'static metal::ComputePipelineState = Box::leak(Box::new(pso));
    psos.insert(name, pso);
    Ok(pso)
}
/// The pipelines of one evalfast build and the threadgroup size its `ef_eval_split` was compiled
/// for (`EF_THREADS`).
pub(crate) struct EfKernels<'a> {
    pub split: &'a metal::ComputePipelineStateRef,
    pub combine: &'a metal::ComputePipelineStateRef,
    pub threads: u64,
}

/// ef_combine's threadgroup size (its `[[max_total_threads_per_threadgroup]]`).
const EF_COMBINE_THREADS: u64 = 256;
/// Entries of the evalfast table buffer: pm (EF_NPM), pmn (EF_NPM), the quotients (1).
pub(crate) const EF_TABLE_LEN: usize = 2 * ef::EF_NPM + 1;

const _: () = assert!(ef::EF_NPC == NUM_POLY_MIX_POWERS);
const _: () = assert!(ef::EF_NPM == 627);

/// The evalfast table buffer contents (CUDA `ef_prep` + `ef_prep_n` on the host; the kernel reads
/// each FpExt as 4 raw words, which is the in-memory form of [ExtVal]):
/// - `[0, EF_NPM)` pm: the poly_mix powers, `RK_QN` products `pm[RK_QA[i]] * pm[RK_QB[i]]` and the
///   `RK_HN` terms of the generated `rk_hoist` over `out` / `mix`;
/// - `[EF_NPM, 2 EF_NPM)` pmn: `(pm.1, pm.2, pm.3) * (P - 11), 0` (NBETA-folded rows);
/// - `2 EF_NPM`: the vanishing quotients `1 / ((3 rou^c)^(2^po2) - 1)`, `c = 0..4`.
pub(crate) fn ef_table(
    poly_mix_pows: &[ExtVal],
    out: &[Val],
    mix: &[Val],
    rou: Val,
    po2: usize,
) -> Vec<ExtVal> {
    assert_eq!(poly_mix_pows.len(), ef::EF_NPC);
    let mut tab = vec![ExtVal::ZERO; EF_TABLE_LEN];
    tab[..ef::EF_NPC].copy_from_slice(poly_mix_pows);
    for i in 0..ef::RK_QN {
        tab[ef::EF_NPC + i] =
            poly_mix_pows[ef::RK_QA[i] as usize] * poly_mix_pows[ef::RK_QB[i] as usize];
    }
    let (head, hoisted) = tab.split_at_mut(ef::EF_NPC + ef::RK_QN);
    ef::rk_hoist(head, out, mix, &mut hoisted[..ef::RK_HN]);
    let nbeta = Val::new(risc0_core::field::baby_bear::P - 11);
    for i in 0..ef::EF_NPM {
        let e = tab[i].elems();
        tab[ef::EF_NPM + i] = ExtVal::new(e[1] * nbeta, e[2] * nbeta, e[3] * nbeta, Val::ZERO);
    }
    let q: Vec<Val> = (0..4)
        .map(|c| {
            let x = Val::new(3) * rou.pow(c);
            (x.pow(1 << po2) - Val::ONE).inv()
        })
        .collect();
    tab[2 * ef::EF_NPM] = ExtVal::new(q[0], q[1], q[2], q[3]);
    tab
}

// The baseline kernel's `poly_mix` table has `kNumPolyMixPows` entries (eval_check.cuh), and
// evalfast is generated for the same count (EF_NPC); a circuit change must regenerate both.
const _: () = assert!(NUM_POLY_MIX_POWERS == 458);
const _: () = assert!(std::mem::size_of::<ExtVal>() == 4 * std::mem::size_of::<u32>());
const _: () = assert!(std::mem::size_of::<Val>() == std::mem::size_of::<u32>());

impl<MH: MetalHash> MetalCircuitHal<MH> {
    /// eval_check on the GPU with the kernel [EVAL_KERNEL_ENV] selects (evalfast by default).
    pub(crate) fn eval_check_metal(
        &self,
        check: &BufferImpl<Val>,
        groups: &[&BufferImpl<Val>],
        globals: &[&BufferImpl<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) -> anyhow::Result<()> {
        match EvalKernel::current()? {
            EvalKernel::Evalfast => {
                self.eval_check_evalfast(check, groups, globals, poly_mix, po2, steps)
            }
            EvalKernel::EvalfastR12 => {
                self.eval_check_evalfast_r12(check, groups, globals, poly_mix, po2, steps)
            }
            EvalKernel::Baseline => {
                self.eval_check_baseline(check, groups, globals, poly_mix, po2, steps)
            }
        }
    }

    /// eval_check on the GPU: the evalfast kernels (eval_check_ef.metal, R12). ABI:
    /// `ef_eval_split`: `0 check, 1 data, 2 accum, 3 mix, 4 out, 5 tab` ([ef_table]),
    /// `6 scratch` (4 * domain LogUp partials), `7 domain` (`constant uint&`), threadgroups of
    /// [EF_THREADS], two per row block (even: main chain, odd: LogUp); `ef_combine`:
    /// `0 check, 1 scratch, 2 tab, 3 domain`, one thread per row.
    pub(crate) fn eval_check_evalfast(
        &self,
        check: &BufferImpl<Val>,
        groups: &[&BufferImpl<Val>],
        globals: &[&BufferImpl<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) -> anyhow::Result<()> {
        scope!("metal_eval_check_evalfast");
        self.ctx.require(Phase::EvalCheck)?;
        let k = EfKernels {
            split: self.ctx.pipeline("ef_eval_split"),
            combine: self.ctx.pipeline("ef_combine"),
            threads: EF_THREADS,
        };
        self.eval_check_ef_with(&k, check, groups, globals, poly_mix, po2, steps, None)
    }

    /// [Self::eval_check_evalfast] with R12's table forms (`ef_eval_split_r12`, built on first
    /// use; same ABI and results).
    pub(crate) fn eval_check_evalfast_r12(
        &self,
        check: &BufferImpl<Val>,
        groups: &[&BufferImpl<Val>],
        globals: &[&BufferImpl<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) -> anyhow::Result<()> {
        scope!("metal_eval_check_evalfast_r12");
        self.ctx.require(Phase::EvalCheck)?;
        let k = EfKernels {
            split: lazy_pipeline(self.ctx.library(), "ef_eval_split_r12")?,
            combine: self.ctx.pipeline("ef_combine"),
            threads: EF_THREADS,
        };
        self.eval_check_ef_with(&k, check, groups, globals, poly_mix, po2, steps, None)
    }

    /// [Self::eval_check_evalfast] with the given pipelines. With `times`, each dispatch is
    /// committed and waited for on its own and its wall time (encode + GPU + wait) is stored as
    /// `[split, combine]` (benchmarks only).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eval_check_ef_with(
        &self,
        k: &EfKernels<'_>,
        check: &BufferImpl<Val>,
        groups: &[&BufferImpl<Val>],
        globals: &[&BufferImpl<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
        mut times: Option<&mut [std::time::Duration; 2]>,
    ) -> anyhow::Result<()> {
        let accum = groups[REGISTER_GROUP_ACCUM];
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
        tracing::debug!(
            "eval_check (Metal evalfast): steps {steps}, domain {domain}, po2 {po2}, data {}, \
             accum {}, mix {}, out {}",
            data.size(),
            accum.size(),
            mix.size(),
            out.size()
        );
        let (split, combine, threads) = (k.split, k.combine, k.threads);
        anyhow::ensure!(
            split.max_total_threads_per_threadgroup() >= threads
                && combine.max_total_threads_per_threadgroup() >= EF_COMBINE_THREADS,
            "eval_check: evalfast PSO thread limits {} / {} below {threads} / \
             {EF_COMBINE_THREADS}",
            split.max_total_threads_per_threadgroup(),
            combine.max_total_threads_per_threadgroup()
        );

        // The table (CUDA ef_prep on one GPU thread): built on the CPU from the powers and the
        // host views of out / mix (view() flushes and waits if a batch is open).
        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);
        assert_eq!(poly_mix_pows.len(), NUM_POLY_MIX_POWERS);
        let mut out_h = Vec::new();
        let mut mix_h = Vec::new();
        out.view(|v| out_h.extend_from_slice(v));
        mix.view(|v| mix_h.extend_from_slice(v));
        let tab = ef_table(
            &poly_mix_pows,
            &out_h,
            &mix_h,
            Val::ROU_FWD[po2 + EXP_PO2],
            po2,
        );
        let (tab_buf, scratch) = {
            let mut s = self.ctx.scratch();
            (
                grow(&self.hal, &mut s.poly_mix, "poly_mix", EF_TABLE_LEN),
                grow(
                    &self.hal,
                    &mut s.scratch,
                    "scratch",
                    ExtVal::EXT_SIZE * domain,
                ),
            )
        };
        // view_mut flushes and waits if a batch is open, so a previous eval_check that read this
        // buffer has finished before it is overwritten.
        tab_buf.view_mut(|dst| dst[..EF_TABLE_LEN].copy_from_slice(&tab));

        let rows = (domain as u64).div_ceil(threads);
        if times.is_some() {
            self.hal.try_sync()?;
        }
        let t0 = std::time::Instant::now();
        self.hal.dispatch(
            split,
            &[
                check.as_arg(),
                data.as_arg(),
                accum.as_arg(),
                mix.as_arg(),
                out.as_arg(),
                tab_buf.as_arg(),
                scratch.as_arg(),
                KernelArg::Integer(domain as u32),
            ],
            Grid::Groups {
                groups: metal::MTLSize::new(rows * 2, 1, 1),
                threads: metal::MTLSize::new(threads, 1, 1),
            },
            &[],
        );
        if let Some(t) = times.as_deref_mut() {
            self.hal.try_sync()?;
            t[0] = t0.elapsed();
        }
        let t1 = std::time::Instant::now();
        self.hal.dispatch(
            combine,
            &[
                check.as_arg(),
                scratch.as_arg(),
                tab_buf.as_arg(),
                KernelArg::Integer(domain as u32),
            ],
            Grid::Groups {
                groups: metal::MTLSize::new((domain as u64).div_ceil(EF_COMBINE_THREADS), 1, 1),
                threads: metal::MTLSize::new(EF_COMBINE_THREADS, 1, 1),
            },
            &[],
        );
        // End of the phase: close the batch and surface a command-buffer error (PLAN 2.6, 2.9).
        self.hal.try_sync()?;
        if let Some(t) = times {
            t[1] = t1.elapsed();
        }
        tracing::debug!(
            "eval_check (Metal evalfast) done: currentAllocatedSize {} B, process peak {} B",
            self.hal.allocated_size(),
            self.hal.peak_allocated_size()
        );
        Ok(())
    }

    /// eval_check on the GPU: the baseline `eval_check` kernel (R6 ABI, eval_check.metal):
    /// `0 check, 1 ctrl, 2 data, 3 accum, 4 mix, 5 out` (buffers), `6 rou` (`constant Fp&`),
    /// `7 po2`, `8 domain` (`constant uint&`), `9 poly_mix` (`const device FpExt*`, 458 entries).
    pub(crate) fn eval_check_baseline(
        &self,
        check: &BufferImpl<Val>,
        groups: &[&BufferImpl<Val>],
        globals: &[&BufferImpl<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) -> anyhow::Result<()> {
        scope!("metal_eval_check_baseline");
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
        let pso = lazy_pipeline(self.ctx.library(), "eval_check")?;
        self.hal.dispatch(
            pso,
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

#[cfg(test)]
mod ef_tests {
    //! R12: the evalfast kernels against the baseline kernel and the CPU (random canonical
    //! inputs; `Val::random` is uniform in [0, P)), the table builder and the kernel selection.
    //! The tests call `eval_check_evalfast` / `eval_check_baseline` directly and never set
    //! [EVAL_KERNEL_ENV], so they need no process-wide test lock (PLAN 8.7).

    use std::sync::Arc;

    use risc0_core::field::baby_bear::BabyBear;
    use risc0_zkp::{
        adapter::CircuitInfo as _,
        core::hash::sha::Sha256HashSuite,
        hal::{cpu::CpuHal, metal::MetalHalSha256, Hal},
    };
    use test_log::test;

    use super::*;
    use crate::{
        prove::hal::cpu::CpuCircuitHal,
        zirgen::{taps::TAPSET, CircuitImpl},
    };

    fn random_vals(n: usize) -> Vec<Val> {
        let mut rng = rand::rng();
        (0..n).map(|_| Val::random(&mut rng)).collect()
    }

    struct Inputs {
        code: Vec<Val>,
        data: Vec<Val>,
        accum: Vec<Val>,
        mix: Vec<Val>,
        out: Vec<Val>,
        poly_mix: ExtVal,
    }

    impl Inputs {
        fn random(po2: usize) -> Self {
            let domain = (1 << po2) * INV_RATE;
            Self {
                code: random_vals(TAPSET.group_size(REGISTER_GROUP_CODE) * domain),
                data: random_vals(TAPSET.group_size(REGISTER_GROUP_DATA) * domain),
                accum: random_vals(TAPSET.group_size(REGISTER_GROUP_ACCUM) * domain),
                mix: random_vals(CircuitImpl::MIX_SIZE),
                out: random_vals(CircuitImpl::OUTPUT_SIZE),
                poly_mix: ExtVal::random(&mut rand::rng()),
            }
        }

        fn bytes(&self) -> usize {
            4 * (self.code.len()
                + self.data.len()
                + self.accum.len()
                + self.mix.len()
                + self.out.len())
        }
    }

    /// `check` from `CpuCircuitHal::eval_check` (the reference of R10's tests).
    fn cpu_check(inp: &Inputs, po2: usize) -> Vec<Val> {
        let steps = 1 << po2;
        let hal: CpuHal<BabyBear> = CpuHal::new(Sha256HashSuite::new_suite());
        let check = hal.alloc_elem("check", ExtVal::EXT_SIZE * steps * INV_RATE);
        let [code, data, accum, mix, out] = [&inp.code, &inp.data, &inp.accum, &inp.mix, &inp.out]
            .map(|b| hal.copy_from_elem("in", b));
        CpuCircuitHal.eval_check(
            &check,
            &[&accum, &code, &data],
            &[&mix, &out],
            inp.poly_mix,
            po2,
            steps,
        );
        check.to_vec()
    }

    #[derive(Clone, Copy, Debug)]
    enum Which {
        Evalfast,
        EvalfastR12,
        Baseline,
    }

    /// Runs one GPU eval_check `reps` times over the same Metal buffers; returns every `check`,
    /// the GPU wall time of each run and the peak `currentAllocatedSize` during the runs.
    fn gpu_checks(
        c: &MetalCircuitHal<risc0_zkp::hal::metal::MetalHashSha256>,
        inp: &Inputs,
        po2: usize,
        which: Which,
        reps: usize,
    ) -> (Vec<Vec<Val>>, Vec<std::time::Duration>, u64) {
        let hal = &c.hal;
        let steps = 1 << po2;
        let domain = steps * INV_RATE;
        let [code, data, accum, mix, out] = [&inp.code, &inp.data, &inp.accum, &inp.mix, &inp.out]
            .map(|b| hal.copy_from_elem("in", b));
        let groups = [&accum, &code, &data];
        let globals = [&mix, &out];
        let mut res = Vec::new();
        let mut times = Vec::new();
        hal.sync();
        let mut peak = 0;
        for _ in 0..reps {
            // A fresh check buffer filled with garbage, so a row the kernel skips shows up.
            let check = hal.alloc_elem("check", ExtVal::EXT_SIZE * domain);
            check.view_mut(|v| v.fill(Val::new_raw(0x7fff_ffff)));
            let t = std::time::Instant::now();
            match which {
                Which::Evalfast => {
                    c.eval_check_evalfast(&check, &groups, &globals, inp.poly_mix, po2, steps)
                }
                Which::EvalfastR12 => {
                    c.eval_check_evalfast_r12(&check, &groups, &globals, inp.poly_mix, po2, steps)
                }
                Which::Baseline => {
                    c.eval_check_baseline(&check, &groups, &globals, inp.poly_mix, po2, steps)
                }
            }
            .unwrap();
            times.push(t.elapsed());
            peak = peak.max(hal.peak_allocated_size().max(hal.allocated_size()));
            res.push(check.to_vec());
        }
        (res, times, peak)
    }

    fn mismatches(a: &[Val], b: &[Val]) -> usize {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).filter(|(x, y)| x != y).count()
    }

    /// evalfast == evalfast_r12 == baseline == CPU, bit for bit over all `4 * domain` elements,
    /// `reps` evalfast runs on the same buffers each equal.
    fn evalfast_matches(po2: usize, reps: usize) {
        let inp = Inputs::random(po2);
        let t = std::time::Instant::now();
        let cpu = cpu_check(&inp, po2);
        let cpu_time = t.elapsed();
        let hal = Arc::new(MetalHalSha256::new());
        let c = MetalCircuitHal::new(hal.clone()).unwrap();
        let (fast, fast_t, fast_peak) = gpu_checks(&c, &inp, po2, Which::Evalfast, reps);
        let (base, base_t, base_peak) = gpu_checks(&c, &inp, po2, Which::Baseline, 1);
        let (r12, r12_t, _) = gpu_checks(&c, &inp, po2, Which::EvalfastR12, 1);
        let n = mismatches(&cpu, &r12[0]);
        assert_eq!(
            n,
            0,
            "po2 {po2}: evalfast_r12 vs CPU: {n} of {} differ",
            cpu.len()
        );
        for (i, f) in fast.iter().enumerate() {
            let n = mismatches(&cpu, f);
            assert_eq!(
                n,
                0,
                "po2 {po2} rep {i}: evalfast vs CPU: {n} of {} differ",
                cpu.len()
            );
        }
        let n = mismatches(&cpu, &base[0]);
        assert_eq!(
            n,
            0,
            "po2 {po2}: baseline vs CPU: {n} of {} differ",
            cpu.len()
        );
        println!(
            "R12 eval_check po2 {po2}: {} elements bit-exact (evalfast x{reps} == evalfast_r12 == \
             baseline == CPU); evalfast {fast_t:?}, evalfast_r12 {r12_t:?}, baseline {base_t:?}, \
             CPU HAL {cpu_time:?}; inputs {} B, peak \
             currentAllocatedSize evalfast {fast_peak} B / baseline {base_peak} B",
            cpu.len(),
            inp.bytes(),
        );
    }

    #[test]
    fn evalfast_matches_cpu_po2_4() {
        evalfast_matches(4, 3);
    }

    #[test]
    fn evalfast_matches_cpu_po2_10() {
        evalfast_matches(10, 3);
    }

    #[test]
    fn evalfast_matches_cpu_po2_16() {
        evalfast_matches(16, 3);
    }

    /// po2 20: ~5.3 GB of inputs and a ~1 min CPU reference (run explicitly; R12 note).
    #[test]
    #[ignore]
    fn evalfast_matches_cpu_po2_20() {
        evalfast_matches(20, 2);
    }

    /// Repeated evalfast runs on one random po2 `R12_EF_PO2` (default 18) input against the
    /// baseline, `R12_EF_REPS` times (default 20): the nondeterminism check R11 asked for (its
    /// witgen kernel failed most repeated calls on big inputs). Also prints the GPU times.
    #[test]
    #[ignore]
    fn evalfast_repeat_bit_exact() {
        let po2: usize = std::env::var("R12_EF_PO2").map_or(18, |v| v.parse().unwrap());
        let reps: usize = std::env::var("R12_EF_REPS").map_or(20, |v| v.parse().unwrap());
        let inp = Inputs::random(po2);
        let hal = Arc::new(MetalHalSha256::new());
        let c = MetalCircuitHal::new(hal.clone()).unwrap();
        // R17 audit: `R12_BASE_REPS` (default 2) baseline runs, each compared with the first.
        let base_reps: usize = std::env::var("R12_BASE_REPS").map_or(2, |v| v.parse().unwrap());
        let (base, base_t, _) = gpu_checks(&c, &inp, po2, Which::Baseline, base_reps.max(2));
        let base_bad: Vec<(usize, usize)> = base
            .iter()
            .enumerate()
            .map(|(i, b)| (i, mismatches(&base[0], b)))
            .filter(|(_, n)| *n > 0)
            .collect();
        println!(
            "R17AUDIT rv32im eval_check baseline po2 {po2}: {}/{} runs differ from run 0 {base_bad:?}",
            base_bad.len(),
            base.len()
        );
        assert!(base_bad.is_empty(), "baseline not deterministic");
        let (fast, fast_t, peak) = gpu_checks(&c, &inp, po2, Which::Evalfast, reps);
        let bad: Vec<(usize, usize)> = fast
            .iter()
            .enumerate()
            .map(|(i, f)| (i, mismatches(&base[0], f)))
            .filter(|(_, n)| *n > 0)
            .collect();
        println!(
            "R12 evalfast repeat po2 {po2}: {} of {reps} runs differ from the baseline {bad:?}; \
             evalfast {fast_t:?}; baseline {base_t:?}; peak {peak} B",
            bad.len()
        );
        assert!(bad.is_empty());
    }

    /// The table builder: products, NBETA-folded rows and quotients as CUDA ef_prep/ef_prep_n
    /// compute them; the hoisted terms are checked end to end by the kernel tests.
    #[test]
    fn ef_table_layout() {
        let pm = map_pow(ExtVal::random(&mut rand::rng()), POLY_MIX_POWERS);
        let out = random_vals(CircuitImpl::OUTPUT_SIZE);
        let mix = random_vals(CircuitImpl::MIX_SIZE);
        let po2 = 7;
        let rou = Val::ROU_FWD[po2 + 2];
        let tab = ef_table(&pm, &out, &mix, rou, po2);
        assert_eq!(tab.len(), EF_TABLE_LEN);
        assert_eq!(&tab[..ef::EF_NPC], &pm[..]);
        for i in 0..ef::RK_QN {
            let (a, b) = (ef::RK_QA[i] as usize, ef::RK_QB[i] as usize);
            assert_eq!(tab[ef::EF_NPC + i], pm[a] * pm[b]);
        }
        let nbeta = Val::ZERO - Val::new(11);
        for i in 0..ef::EF_NPM {
            let e = tab[i].elems();
            let n = tab[ef::EF_NPM + i].elems();
            assert_eq!(n, &[e[1] * nbeta, e[2] * nbeta, e[3] * nbeta, Val::ZERO]);
        }
        let q = tab[2 * ef::EF_NPM].elems();
        for (cyc, qc) in q.iter().enumerate() {
            // hal/cpu.rs: 1 / ((3 * rou^cycle)^(2^po2) - 1) depends on cycle mod 4 only.
            for k in [0usize, 4, 8, 400] {
                let y = (Val::new(3) * rou.pow(cyc + k)).pow(1 << po2);
                assert_eq!(*qc, (y - Val::ONE).inv(), "cycle {}", cyc + k);
            }
        }
        // rk_hoist reads out and mix: other values change the hoisted terms (and only them).
        for (out2, mix2) in [
            (random_vals(out.len()), mix.clone()),
            (out.clone(), random_vals(mix.len())),
        ] {
            let tab2 = ef_table(&pm, &out2, &mix2, rou, po2);
            let hoisted = ef::EF_NPC + ef::RK_QN..ef::EF_NPM;
            assert_ne!(&tab[hoisted.clone()], &tab2[hoisted.clone()]);
            assert_eq!(&tab[..hoisted.start], &tab2[..hoisted.start]);
        }
    }

    /// PSO creation time per eval_check kernel (`R12_PSO_KERNELS`, default the three), on a fresh
    /// library from `METAL_LIB`. Cold when run with `MTL_SHADER_CACHE_SIZE=0` (K5's method: this
    /// process neither reads nor writes the OS shader cache), warm otherwise.
    #[test]
    #[ignore]
    fn eval_check_pso_times() {
        let names = std::env::var("R12_PSO_KERNELS")
            .unwrap_or_else(|_| "ef_combine,ef_eval_split,eval_check".into());
        let device = metal::Device::system_default().unwrap();
        let lib = device
            .new_library_with_data(risc0_circuit_rv32im_sys::METAL_LIB)
            .unwrap();
        for name in names.split(',') {
            let f = lib.get_function(name, None).unwrap();
            let t = std::time::Instant::now();
            let pso = device.new_compute_pipeline_state_with_function(&f).unwrap();
            println!(
                "R12 PSO {name}: {:?} (MTL_SHADER_CACHE_SIZE={:?}); maxTotalThreadsPerThreadgroup \
                 {}, threadExecutionWidth {}, staticThreadgroupMemoryLength {}",
                t.elapsed(),
                std::env::var("MTL_SHADER_CACHE_SIZE").ok(),
                pso.max_total_threads_per_threadgroup(),
                pso.thread_execution_width(),
                pso.static_threadgroup_memory_length()
            );
        }
    }

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name).map_or(default, |v| v.parse().unwrap())
    }

    fn median_min_max(v: &[f64]) -> (f64, f64, f64) {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (s[s.len() / 2], s[0], s[s.len() - 1])
    }

    fn load_average() -> String {
        std::process::Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    }

    /// R15e: interleaved A/B of evalfast builds on one random input. `R15E_VARIANTS` is a comma
    /// list of `name=path[:threads[:split kernel]]` (`path` = `builtin` for the embedded
    /// METAL_LIB; `threads` is the `EF_THREADS` the build was compiled with, default 128; the
    /// split kernel defaults to `ef_eval_split`, e.g. `r12=builtin:128:ef_eval_split_r12`),
    /// `R15E_PO2` (default 20),
    /// `R15E_PAIRS` rounds (default 7; each round runs every variant once, in list order).
    /// Every run is compared bit for bit with the first run of the embedded kernel, and that one
    /// with the CPU when `R15E_CPU=1`. Prints the PSO properties and the median / min / max wall
    /// time (encode + GPU + wait) of each dispatch.
    #[test]
    #[ignore]
    fn ef_variant_bench() {
        let po2 = env_usize("R15E_PO2", 20);
        let pairs = env_usize("R15E_PAIRS", 7);
        let spec = std::env::var("R15E_VARIANTS").unwrap_or_else(|_| "builtin=builtin".into());
        let hal = Arc::new(MetalHalSha256::new());
        let c = MetalCircuitHal::new(hal.clone()).unwrap();
        let mut vars = Vec::new();
        for item in spec.split(',') {
            let (name, rest) = item.split_once('=').expect("name=path[:threads[:kernel]]");
            let mut parts = rest.split(':');
            let path = parts.next().unwrap();
            let threads = parts
                .next()
                .map_or(EF_THREADS, |t| t.parse::<u64>().unwrap());
            let split = parts.next().unwrap_or("ef_eval_split").to_string();
            let bytes = if path == "builtin" {
                risc0_circuit_rv32im_sys::METAL_LIB.to_vec()
            } else {
                std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"))
            };
            let t = std::time::Instant::now();
            let lib = hal
                .try_load_library(&bytes, &[split.as_str(), "ef_combine"])
                .unwrap();
            let pso = lib.pipeline(&split);
            println!(
                "R15E variant {name}: {path} threads {threads}; PSOs {:?}; {split} \
                 maxTotalThreadsPerThreadgroup {} threadExecutionWidth {} \
                 staticThreadgroupMemoryLength {}",
                t.elapsed(),
                pso.max_total_threads_per_threadgroup(),
                pso.thread_execution_width(),
                pso.static_threadgroup_memory_length()
            );
            vars.push((name.to_string(), lib, threads, split));
        }
        let inp = Inputs::random(po2);
        let steps = 1 << po2;
        let domain = steps * INV_RATE;
        let [code, data, accum, mix, out] = [&inp.code, &inp.data, &inp.accum, &inp.mix, &inp.out]
            .map(|b| hal.copy_from_elem("in", b));
        let groups = [&accum, &code, &data];
        let globals = [&mix, &out];
        let check = hal.alloc_elem("check", ExtVal::EXT_SIZE * domain);
        c.eval_check_evalfast(&check, &groups, &globals, inp.poly_mix, po2, steps)
            .unwrap();
        let reference = check.to_vec();
        if std::env::var("R15E_CPU").as_deref() == Ok("1") {
            let cpu = cpu_check(&inp, po2);
            assert_eq!(mismatches(&cpu, &reference), 0, "embedded evalfast vs CPU");
            println!(
                "R15E po2 {po2}: embedded evalfast == CPU ({} elements)",
                cpu.len()
            );
        }
        let load0 = load_average();
        let mut times = vec![(Vec::new(), Vec::new()); vars.len()];
        let mut bad = vec![0usize; vars.len()];
        for _ in 0..pairs {
            for (i, (_, lib, threads, split)) in vars.iter().enumerate() {
                check.view_mut(|v| v.fill(Val::new_raw(0x7fff_ffff)));
                let k = EfKernels {
                    split: lib.pipeline(split),
                    combine: lib.pipeline("ef_combine"),
                    threads: *threads,
                };
                let mut t = [std::time::Duration::ZERO; 2];
                c.eval_check_ef_with(
                    &k,
                    &check,
                    &groups,
                    &globals,
                    inp.poly_mix,
                    po2,
                    steps,
                    Some(&mut t),
                )
                .unwrap();
                times[i].0.push(t[0].as_secs_f64() * 1e3);
                times[i].1.push(t[1].as_secs_f64() * 1e3);
                let mut n = 0;
                check.view(|v| n = mismatches(&reference, v));
                if n != 0 {
                    bad[i] += 1;
                }
            }
        }
        println!(
            "R15E po2 {po2}, {pairs} rounds, load average before {load0} / after {}",
            load_average()
        );
        for (i, (name, _, _, _)) in vars.iter().enumerate() {
            let (s, smin, smax) = median_min_max(&times[i].0);
            let (cm, cmin, cmax) = median_min_max(&times[i].1);
            let tot: Vec<f64> = times[i]
                .0
                .iter()
                .zip(&times[i].1)
                .map(|(a, b)| a + b)
                .collect();
            let (tm, tmin, tmax) = median_min_max(&tot);
            println!(
                "R15E {name:>12}: split {s:8.1} ms [{smin:.1}, {smax:.1}]  combine {cm:6.2} ms \
                 [{cmin:.2}, {cmax:.2}]  total {tm:8.1} ms [{tmin:.1}, {tmax:.1}]  mismatching \
                 runs {}/{pairs}",
                bad[i]
            );
        }
        // R15E_ALLOW_MISMATCH=1: measurement-only builds (e.g. one phase skipped).
        if std::env::var("R15E_ALLOW_MISMATCH").as_deref() != Ok("1") {
            assert!(bad.iter().all(|&b| b == 0), "a variant differs: {bad:?}");
        }
    }

    /// R15e: the non-default eval_check PSOs are not built when the library loads, and
    /// [lazy_pipeline] builds each once.
    #[test]
    fn non_default_eval_psos_are_lazy() {
        let hal = Arc::new(MetalHalSha256::new());
        let c = MetalCircuitHal::new(hal).unwrap();
        let k = &c.ctx.kernels.kernels;
        for name in ["eval_check", "ef_eval_split_r12"] {
            assert!(k.get(name).is_none(), "{name} built at load");
            assert!(
                c.ctx.library().function_names().iter().any(|f| f == name),
                "{name} missing from the metallib"
            );
        }
        assert!(k.get("ef_eval_split").is_some() && k.get("ef_combine").is_some());
        let lib = c.ctx.library();
        let a = lazy_pipeline(lib, "ef_combine").unwrap() as *const _;
        let b = lazy_pipeline(lib, "ef_combine").unwrap() as *const _;
        assert_eq!(a, b, "built twice");
        assert!(lazy_pipeline(lib, "r15e_not_a_kernel").is_err());
    }

    #[test]
    fn eval_kernel_parsing() {
        assert_eq!(EvalKernel::parse("").unwrap(), EvalKernel::DEFAULT);
        assert_eq!(EvalKernel::parse("evalfast").unwrap(), EvalKernel::Evalfast);
        assert_eq!(
            EvalKernel::parse("evalfast_r12").unwrap(),
            EvalKernel::EvalfastR12
        );
        assert_eq!(
            EvalKernel::parse(" Baseline ").unwrap(),
            EvalKernel::Baseline
        );
        assert!(EvalKernel::parse("fastest").is_err());
    }
}
