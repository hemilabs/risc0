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

//! The keccak circuit on Apple GPUs, on the zkp crate's Metal HAL (docs/metal-port/PLAN.md 4.4).
//!
//! Skeleton (item K1): the prover runs on [MetalHalPoseidon2] (NTTs, hashing, FRI and the rest of
//! the zkp prover on the GPU), and the circuit methods run the CPU implementations over the
//! HAL's shared (unified-memory) buffers: the witness is made on the host (`HOST_WITNESS`, as on
//! Intel), and eval_check calls the CPU `poly_fp` through host views. The keccak metallib
//! (`risc0_circuit_keccak_sys::METAL_LIB`) is loaded with a PSO per kernel; so far it holds
//! `scatter_preflight` ([MetalCircuitHal::scatter_preflight_gpu]), a port of the CUDA kernel,
//! tested against the host path but not yet used by `scatter_preflight` (item K2 switches it on).
//! K3/K4 add the staged eval_check kernels, K6 the GPU witness.
//!
//! No `Drop` is needed: CUDA's (`cuda_reset`) resets a CUDA stack-size limit its witgen raised,
//! and Metal has no such process state; the PSOs and the HAL reference drop on their own, and
//! buffers return to the HAL's pool after their last batch completes.

use std::rc::Rc;

use anyhow::{bail, ensure, Result};
use metal::ComputePipelineState;
use rayon::prelude::*;
use risc0_circuit_keccak_sys::{
    risc0_circuit_keccak_cpu_poly_fp, ScatterInfo, METAL_KERNEL_NAMES, METAL_LIB,
};
use risc0_core::{field::RootsOfUnity as _, scope};
use risc0_zkp::{
    core::log2_ceil,
    field::{map_pow, Elem as _, ExtElem as _},
    hal::{
        metal::{
            as_bytes, BufferImpl as MetalBuffer, Grid, KernelArg, MetalHal, MetalHalPoseidon2,
            MetalHash, MetalHashPoseidon2, MetalKernels,
        },
        AccumPreflight, Buffer as _, CircuitHal,
    },
    INV_RATE,
};

use crate::{
    prove::{KeccakProver, KeccakProverImpl, GLOBAL_MIX, GLOBAL_OUT},
    zirgen::{
        circuit::{ExtVal, Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_DATA},
        info::POLY_MIX_POWERS,
    },
};

use super::{
    cpu::{generate_witness_host, scatter_preflight_host, HostMetaBuffer},
    CircuitWitnessGenerator, ForwardPreflightOrder, MetaBuffer, PreflightCycleOrder,
    PreflightTrace, StepMode,
};

/// `scatter_preflight` (kernels/metal/scatter.metal) bindings.
const SCATTER_INTO: usize = 0;
const SCATTER_INFOS: usize = 1;
const SCATTER_FROM: usize = 2;
const SCATTER_FAULT: usize = 3;
const SCATTER_PARAMS: usize = 4;
const SCATTER_NARGS: usize = 5;

/// `ScatterParams` of scatter.metal: `rows`, `cols`, `count`, `from_len` (16 bytes, set_bytes).
type ScatterParams = [u32; 4];
const _: () = assert!(std::mem::size_of::<ScatterParams>() == 16);

pub struct MetalCircuitHal<MH: MetalHash> {
    hal: Rc<MetalHal<MH>>,
    /// The keccak metallib, with a PSO for each of `METAL_KERNEL_NAMES`.
    kernels: MetalKernels,
}

impl<MH: MetalHash> MetalCircuitHal<MH> {
    /// Loads the keccak metallib and builds every PSO; panics if a kernel is missing.
    pub fn new(hal: Rc<MetalHal<MH>>) -> Self {
        let kernels = scope!(
            "load_keccak_metallib",
            hal.load_library(METAL_LIB, METAL_KERNEL_NAMES)
        );
        Self { hal, kernels }
    }

    /// The PSO of a keccak kernel (one of `METAL_KERNEL_NAMES`).
    #[allow(dead_code)]
    pub(crate) fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.kernels.pipeline(name)
    }

    /// Scatter the preflight's packed words `from` into `into` on the GPU: the CUDA
    /// `scatter_preflight` kernel, a thread per info. Syncs, and fails if the kernel rejected an
    /// info (out of bounds, or bits not in 1..32), where the host path would panic.
    // Used by the tests until K2 makes `scatter_preflight` call it.
    #[allow(dead_code)]
    pub(crate) fn scatter_preflight_gpu(
        &self,
        into: &MetaBuffer<MetalHal<MH>>,
        infos: &[ScatterInfo],
        from: &[u32],
    ) -> Result<()> {
        scope!("scatter");
        ensure!(into.buf.size() == into.rows * into.cols);
        ensure!(
            u32::try_from(into.buf.size()).is_ok(),
            "scatter: witness of {} elements does not fit 32-bit indexing",
            into.buf.size()
        );
        if infos.is_empty() {
            return Ok(());
        }
        let infos_buf: MetalBuffer<ScatterInfo> = self.hal.copy_from_slice("scatter_infos", infos);
        // A zero-length Metal buffer cannot be made; a lone 0 is never read (count == 0 infos).
        let from_buf = if from.is_empty() {
            self.hal.copy_from_slice("scatter_from", &[0u32])
        } else {
            self.hal.copy_from_slice("scatter_from", from)
        };
        let fault = self.hal.copy_from_slice("scatter_fault", &[0u32]);
        let params: ScatterParams = [
            into.rows as u32,
            into.cols as u32,
            u32::try_from(infos.len())?,
            u32::try_from(from.len())?,
        ];

        let mut args: [KernelArg; SCATTER_NARGS] = std::array::from_fn(|_| KernelArg::Null);
        args[SCATTER_INTO] = into.buf.as_arg();
        args[SCATTER_INFOS] = infos_buf.as_arg();
        args[SCATTER_FROM] = from_buf.as_arg();
        args[SCATTER_FAULT] = fault.as_arg();
        args[SCATTER_PARAMS] = KernelArg::Bytes(as_bytes(&params));
        self.hal.dispatch(
            self.pipeline("scatter_preflight"),
            &args,
            Grid::Threads(infos.len() as u64),
            &[],
        );
        self.hal.try_sync()?;
        if fault.get_at(0) != 0 {
            bail!(
                "keccak scatter_preflight (Metal): the preflight trace has an invalid ScatterInfo"
            );
        }
        Ok(())
    }
}

impl<MH: MetalHash> CircuitWitnessGenerator<MetalHal<MH>> for MetalCircuitHal<MH> {
    type PreferredPreflightOrder = ForwardPreflightOrder;
    // The witness is the CPU's (`generate_witness` below), so a `host_witness` can be proved.
    const HOST_WITNESS: bool = true;

    fn scatter_preflight(
        &self,
        into: &MetaBuffer<MetalHal<MH>>,
        infos: &[ScatterInfo],
        from: &[u32],
    ) -> Result<()> {
        scope!("scatter");
        // K1: the CPU path over the shared buffer; K2 switches to `scatter_preflight_gpu`.
        into.buf
            .view_mut(|into_host| scatter_preflight_host(into_host, into.rows, infos, from));
        Ok(())
    }

    fn generate_witness<O: PreflightCycleOrder>(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace<O>,
        global: &MetaBuffer<MetalHal<MH>>,
        data: &MetaBuffer<MetalHal<MH>>,
    ) -> Result<()> {
        scope!("witgen");
        // The CPU witness generator, writing straight into the shared buffers (unified memory).
        let mut result = Ok(());
        global.buf.view_mut(|global_host| {
            data.buf.view_mut(|data_host| {
                result = generate_witness_host(
                    mode,
                    preflight,
                    HostMetaBuffer::of(global, global_host),
                    HostMetaBuffer::of(data, data_host),
                );
            })
        });
        result
    }
}

impl<MH: MetalHash> CircuitHal<MetalHal<MH>> for MetalCircuitHal<MH> {
    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        _ctrl: &MetalBuffer<Val>,
        _io: &MetalBuffer<Val>,
        _data: &MetalBuffer<Val>,
        _mix: &MetalBuffer<Val>,
        _accum: &MetalBuffer<Val>,
        _steps: usize,
    ) {
        // Nothing to do, as in cuda.rs/cpu.rs: the keccak circuit has no accumulation step.
    }

    /// K1: the CPU `poly_fp` (as `CpuCircuitHal::eval_check`) over host views of the shared
    /// buffers; K4 replaces it with the staged Metal kernels.
    fn eval_check(
        &self,
        check: &MetalBuffer<Val>,
        groups: &[&MetalBuffer<Val>],
        globals: &[&MetalBuffer<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) {
        scope!("eval_check");
        #[cfg(test)]
        tests::METAL_EVAL_CHECKS.with(|n| n.set(n.get() + 1));

        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);
        let poly_mix_pows = poly_mix_pows.as_slice();
        let accum = groups[REGISTER_GROUP_ACCUM];
        let data = groups[REGISTER_GROUP_DATA];
        let mix = globals[GLOBAL_MIX];
        let out = globals[GLOBAL_OUT];
        tracing::debug!(
            "check: {}, data: {}, accum: {}, mix: {} out: {}",
            check.size(),
            data.size(),
            accum.size(),
            mix.size(),
            out.size()
        );
        assert_eq!(check.size(), ExtVal::EXT_SIZE * domain);

        data.view(|data| {
            out.view(|out| {
                check.view_mut(|check| {
                    struct Check(*mut Val);
                    // SAFETY: each cycle writes its own `EXT_SIZE` cells only.
                    unsafe impl Sync for Check {}
                    let check = Check(check.as_mut_ptr());
                    (0..domain).into_par_iter().for_each(|cycle| {
                        let check = &check;
                        let args = &[data.as_ptr(), out.as_ptr()];
                        let mut tot = ExtVal::ZERO;
                        // SAFETY: `data` and `out` are the circuit's data and output groups, as
                        // the CPU HAL passes them; `tot` is a single FpExt.
                        unsafe {
                            risc0_circuit_keccak_cpu_poly_fp(
                                cycle,
                                domain,
                                poly_mix_pows.as_ptr(),
                                args.as_ptr(),
                                &mut tot,
                            )
                        };
                        let x = Val::ROU_FWD[po2 + EXP_PO2].pow(cycle);
                        // As CpuCircuitHal: 3 is the coset shift of the evaluation domain.
                        let y = (Val::new(3) * x).pow(1 << po2);
                        let ret = tot * (y - Val::new(1)).inv();
                        for i in 0..ExtVal::EXT_SIZE {
                            // SAFETY: i * domain + cycle < EXT_SIZE * domain == check.len(), and
                            // only this cycle writes it.
                            unsafe { *check.0.add(i * domain + cycle) = ret.elems()[i] };
                        }
                    });
                })
            })
        });
    }

    /// Round 1 uses one command queue (plan 2.0): nothing to wait for.
    fn eval_check_dep(&self) {}
}

pub type MetalCircuitHalPoseidon2 = MetalCircuitHal<MetalHashPoseidon2>;

pub fn keccak_prover() -> Result<Box<dyn KeccakProver>> {
    let hal = Rc::new(MetalHalPoseidon2::new());
    let circuit_hal = Rc::new(MetalCircuitHalPoseidon2::new(hal.clone()));
    tracing::debug!("keccak: Metal prover on {}", hal.device.name());
    Ok(Box::new(KeccakProverImpl { hal, circuit_hal }))
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use risc0_circuit_keccak_sys::ScatterInfo;
    use risc0_core::field::Elem as _;
    use risc0_zkp::hal::{
        metal::{MetalHalPoseidon2, MetalHalSha256},
        Buffer as _,
    };
    use test_log::test;

    use super::{MetalCircuitHal, MetalCircuitHalPoseidon2};
    use crate::{
        prove::{
            hal::{cpu::scatter_preflight_host, MetaBuffer},
            preflight::{ForwardPreflightOrder, PreflightTrace},
            testutil::test_inputs,
        },
        zirgen::circuit::{Val, REGCOUNT_DATA},
    };

    thread_local! {
        /// Calls of the Metal circuit HAL's eval_check on this thread.
        pub(super) static METAL_EVAL_CHECKS: Cell<usize> = const { Cell::new(0) };
    }

    fn scatter_matches_host(po2: usize) {
        let hal = Rc::new(MetalHalSha256::new());
        let circuit_hal = MetalCircuitHal::new(hal.clone());
        let cycles = 1 << po2;
        let preflight = PreflightTrace::<ForwardPreflightOrder>::new(&test_inputs(po2), cycles);
        assert!(!preflight.scatter.is_empty());

        let mut expected = vec![Val::INVALID; cycles * REGCOUNT_DATA];
        scatter_preflight_host(&mut expected, cycles, &preflight.scatter, &preflight.data);

        let data = MetaBuffer::new("data", hal.as_ref(), cycles, REGCOUNT_DATA, true);
        circuit_hal
            .scatter_preflight_gpu(&data, &preflight.scatter, &preflight.data)
            .unwrap();
        let actual = data.buf.to_vec();
        assert_eq!(actual.len(), expected.len());
        let diffs = actual.iter().zip(&expected).filter(|(a, e)| a != e).count();
        assert_eq!(
            diffs, 0,
            "po2 {po2}: {diffs} cells differ from the host scatter"
        );
        // The scatter wrote something (not only INVALID cells survive).
        assert!(actual.iter().any(|v| *v != Val::INVALID));
    }

    #[test]
    fn metal_scatter_matches_host() {
        for po2 in [8, 12] {
            scatter_matches_host(po2);
        }
    }

    #[test]
    fn metal_scatter_rejects_bad_info() {
        let hal = Rc::new(MetalHalSha256::new());
        let circuit_hal = MetalCircuitHal::new(hal.clone());
        let rows = 16;
        let cols = 8;
        let from = [0x0403_0201u32, 0x0807_0605];
        // An info at column 1: `count` values of `bits` bits from `from[offset..]` into `row`.
        let info = |offset, row, count, bits| ScatterInfo {
            offset,
            row,
            col: 1,
            count,
            bits,
        };

        // A valid info: the first 7 bytes of `from`, into row 3, columns 1..8.
        let data = MetaBuffer::new("data", hal.as_ref(), rows, cols, true);
        circuit_hal
            .scatter_preflight_gpu(&data, &[info(0, 3, 7, 8)], &from)
            .unwrap();
        let got = data.buf.to_vec();
        for i in 0..7usize {
            assert_eq!(got[(1 + i) * rows + 3], Val::new(i as u32 + 1));
        }

        let bad = [
            info(0, 3, 8, 8),  // column past `cols`
            info(0, 16, 1, 8), // row past `rows`
            info(2, 3, 1, 8),  // word past `from`
            info(1, 3, 5, 8),  // last word past `from`
            info(0, 3, 1, 0),  // 32 / 0
            info(0, 3, 1, 32), // 1 << 32
        ];
        for info in bad {
            let data = MetaBuffer::new("data", hal.as_ref(), rows, cols, true);
            let desc = format!("{info:?}");
            let err = circuit_hal
                .scatter_preflight_gpu(&data, &[info], &from)
                .expect_err(&desc);
            assert!(
                err.to_string().contains("invalid ScatterInfo"),
                "{desc}: {err}"
            );
            // A rejected info writes nothing.
            assert!(
                data.buf.to_vec().iter().all(|v| *v == Val::INVALID),
                "{desc}"
            );
        }
    }

    /// The factory `prove::keccak_prover()` picks with `--features metal` is this Metal HAL
    /// prover (unless `RISC0_METAL=0` / `RISC0_METAL_SKIP=keccak`), and it proves and verifies.
    #[test]
    fn metal_prover_runs_on_metal_hal() {
        if !risc0_zkp::hal::metal::enabled("keccak") {
            eprintln!("skipped: the Metal backend is disabled for keccak");
            return;
        }
        let po2 = 8;
        let inputs = test_inputs(po2);
        let before = METAL_EVAL_CHECKS.with(Cell::get);
        let prover = crate::prove::keccak_prover().unwrap();
        let seal = prover.prove(&inputs, po2).unwrap();
        assert_eq!(
            METAL_EVAL_CHECKS.with(Cell::get),
            before + 1,
            "keccak_prover() did not prove on the Metal circuit HAL"
        );
        prover.verify(&seal).expect("Verification failed");

        // The host-witness path goes through the same HAL.
        let witness = crate::prove::host_witness(&inputs, po2).unwrap();
        let seal = prover
            .prove_host_witness(witness)
            .expect("the Metal prover takes a host witness")
            .unwrap();
        assert_eq!(METAL_EVAL_CHECKS.with(Cell::get), before + 2);
        prover.verify(&seal).expect("Verification failed");

        // Constructing the Poseidon2 circuit HAL directly also loads every PSO.
        let hal = Rc::new(MetalHalPoseidon2::new());
        let circuit_hal = MetalCircuitHalPoseidon2::new(hal);
        for name in risc0_circuit_keccak_sys::METAL_KERNEL_NAMES {
            let _ = circuit_hal.pipeline(name);
        }
    }
}
