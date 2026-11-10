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
//! Milestone 1 (items K1, K2): the prover runs on [MetalHalPoseidon2] (NTTs, hashing, FRI and the
//! rest of the zkp prover on the GPU). Of the circuit methods:
//! - `scatter_preflight` runs the `scatter_preflight` Metal kernel and `generate_witness` the CPU
//!   witness generator over the HAL's shared (unified-memory) buffers (`HOST_WITNESS`, as on
//!   Intel): see [witgen];
//! - `accumulate` does nothing, exactly as cuda.rs (the keccak circuit has no accumulation step);
//! - `eval_check` runs the staged Metal kernels `eval_stage_0..S-1` (K3's translation of the CUDA
//!   constraint code, K4's chunked launcher): see [eval_check].
//!
//! K6 adds the GPU witness.
//!
//! No `Drop` is needed: CUDA's (`cuda_reset`) resets a CUDA stack-size limit its witgen raised,
//! and Metal has no such process state; the PSOs and the HAL reference drop on their own, and
//! buffers return to the HAL's pool after their last batch completes.

mod eval_check;
mod witgen;

use std::rc::Rc;

use anyhow::Result;
use metal::ComputePipelineState;
use risc0_circuit_keccak_sys::{METAL_KERNEL_NAMES, METAL_LIB};
use risc0_core::scope;
use risc0_zkp::hal::{
    metal::{
        BufferImpl as MetalBuffer, MetalHal, MetalHalPoseidon2, MetalHash, MetalHashPoseidon2,
        MetalKernels,
    },
    AccumPreflight, CircuitHal,
};

use crate::{
    prove::{KeccakProver, KeccakProverImpl},
    zirgen::circuit::{ExtVal, Val},
};

pub struct MetalCircuitHal<MH: MetalHash> {
    hal: Rc<MetalHal<MH>>,
    /// The keccak metallib, with a PSO for each of `METAL_KERNEL_NAMES`.
    kernels: MetalKernels,
    /// eval_check's chunk size and cached scratch/carry ([eval_check]).
    eval: eval_check::EvalCheckState,
}

impl<MH: MetalHash> MetalCircuitHal<MH> {
    /// Loads the keccak metallib and builds every PSO; panics if a kernel is missing.
    pub fn new(hal: Rc<MetalHal<MH>>) -> Self {
        let kernels = scope!(
            "load_keccak_metallib",
            hal.load_library(METAL_LIB, METAL_KERNEL_NAMES)
        );
        Self {
            hal,
            kernels,
            eval: eval_check::EvalCheckState::new(),
        }
    }

    /// The PSO of a keccak kernel (one of `METAL_KERNEL_NAMES`).
    pub(crate) fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.kernels.pipeline(name)
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
        // Exactly as cuda.rs (and cpu.rs): the keccak circuit has no accumulation step.
    }

    /// The staged Metal kernels, a chunk of cycles at a time ([eval_check]).
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
        #[cfg(test)]
        tests::mark_phase(&self.hal.device, tests::Phase::EvalCheck);

        self.eval_check_metal(check, groups, globals, poly_mix, po2, steps, None);

        #[cfg(test)]
        tests::mark_phase(&self.hal.device, tests::Phase::Finalize);
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
    use std::{
        cell::Cell,
        rc::Rc,
        sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        time::{Duration, Instant},
    };

    use risc0_circuit_keccak_sys::ScatterInfo;
    use risc0_core::field::Elem as _;
    use risc0_zkp::{
        core::hash::sha::Sha256HashSuite,
        field::baby_bear::BabyBear,
        hal::{
            cpu::CpuHal,
            metal::{MetalHalPoseidon2, MetalHalSha256},
            Buffer as _,
        },
    };
    use test_log::test;

    use super::{MetalCircuitHal, MetalCircuitHalPoseidon2};
    use crate::{
        prove::{
            hal::{
                cpu::{scatter_preflight_host, CpuCircuitHal},
                MetaBuffer,
            },
            preflight::{ForwardPreflightOrder, PreflightTrace},
            testutil::test_inputs,
            KeccakProver as _, KeccakProverImpl,
        },
        zirgen::circuit::{Val, REGCOUNT_DATA},
    };

    thread_local! {
        /// Calls of the Metal circuit HAL's eval_check on this thread.
        pub(super) static METAL_EVAL_CHECKS: Cell<usize> = const { Cell::new(0) };
        /// Phases the Metal circuit HAL entered on this thread ([mark_phase]), by [Phase].
        static PHASE_ENTRIES: [Cell<usize>; PHASES] = const { [const { Cell::new(0) }; PHASES] };
    }

    /// The phases of a Metal keccak proof, as the circuit HAL sees them.
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Phase {
        /// Preflight, the `data`/`global` allocation and INVALID fill (before `scatter_preflight`).
        Setup = 0,
        /// `scatter_preflight` (GPU kernel).
        Scatter,
        /// `generate_witness` (CPU witgen into the shared buffers).
        Witgen,
        /// From the end of witgen to eval_check: zeroize, commit of code/data/accum (NTTs,
        /// Merkle trees), mix, and the check buffer `finalize` allocates before eval_check.
        Commit,
        /// `eval_check` (the staged Metal kernels).
        EvalCheck,
        /// After eval_check: check commit, DEEP-ALI, FRI, until the seal.
        Finalize,
    }
    const PHASES: usize = 6;
    const PHASE_NAMES: [&str; PHASES] = [
        "setup",
        "scatter",
        "witgen",
        "commit",
        "eval_check",
        "finalize",
    ];

    /// Peak `current_allocated_size` per phase while [SAMPLING] is on.
    static PEAKS: [AtomicU64; PHASES] = [const { AtomicU64::new(0) }; PHASES];
    static PHASE: AtomicUsize = AtomicUsize::new(0);
    static SAMPLING: AtomicBool = AtomicBool::new(false);

    fn record(phase: usize, bytes: u64) {
        PEAKS[phase].fetch_max(bytes, Ordering::Relaxed);
    }

    /// Called by the Metal circuit HAL as it enters a phase: counts the entry, and while a
    /// memory measurement runs, closes the previous phase and opens `phase` with a sample.
    pub(crate) fn mark_phase(device: &metal::DeviceRef, phase: Phase) {
        PHASE_ENTRIES.with(|e| e[phase as usize].set(e[phase as usize].get() + 1));
        if SAMPLING.load(Ordering::Relaxed) {
            let bytes = device.current_allocated_size();
            record(PHASE.load(Ordering::Relaxed), bytes);
            PHASE.store(phase as usize, Ordering::Relaxed);
            record(phase as usize, bytes);
        }
    }

    fn phase_entries() -> [usize; PHASES] {
        PHASE_ENTRIES.with(|e| std::array::from_fn(|i| e[i].get()))
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
        let entries = phase_entries();
        let prover = crate::prove::keccak_prover().unwrap();
        let seal = prover.prove(&inputs, po2).unwrap();
        assert_eq!(
            METAL_EVAL_CHECKS.with(Cell::get),
            before + 1,
            "keccak_prover() did not prove on the Metal circuit HAL"
        );
        // The prover went through the Metal HAL's (GPU) scatter, witgen and eval_check once each.
        let after = phase_entries();
        for phase in [Phase::Scatter, Phase::Witgen, Phase::EvalCheck] {
            assert_eq!(
                after[phase as usize],
                entries[phase as usize] + 1,
                "{phase:?}"
            );
        }
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

    /// The staged Metal eval_check matches the CPU HAL's at PO2 4 (wider shapes: eval_check.rs),
    /// on random inputs, bit-exactly.
    #[test]
    fn eval_check() {
        const PO2: usize = 4;
        let cpu_hal: CpuHal<BabyBear> = CpuHal::new(Sha256HashSuite::new_suite());
        let metal_hal = Rc::new(MetalHalSha256::new());
        let metal_eval = MetalCircuitHal::new(metal_hal.clone());
        crate::prove::testutil::eval_check(
            &cpu_hal,
            CpuCircuitHal,
            metal_hal.as_ref(),
            metal_eval,
            PO2,
        );
    }

    /// Peak GPU memory (`MTLDevice.currentAllocatedSize`) per phase of a Metal keccak proof, at
    /// the po2 values in `K2_MEM_PO2S` (default 8,12,14). Run it alone (other tests in the same
    /// process allocate on the same device):
    /// `cargo test -p risc0-circuit-keccak --release --features metal --lib -- metal_peak_memory --ignored --nocapture`
    #[test]
    #[ignore]
    fn metal_peak_memory() {
        let po2s: Vec<usize> = std::env::var("K2_MEM_PO2S")
            .unwrap_or_else(|_| "8,12,14".into())
            .split(',')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        for po2 in po2s {
            let hal = Rc::new(MetalHalPoseidon2::new());
            let device = hal.device.clone();
            let circuit_hal = Rc::new(MetalCircuitHalPoseidon2::new(hal.clone()));
            let prover = KeccakProverImpl { hal, circuit_hal };
            let inputs = test_inputs(po2);

            let baseline = device.current_allocated_size();
            for p in &PEAKS {
                p.store(0, Ordering::Relaxed);
            }
            PHASE.store(Phase::Setup as usize, Ordering::Relaxed);
            record(Phase::Setup as usize, baseline);
            SAMPLING.store(true, Ordering::Relaxed);
            let stop = std::sync::Arc::new(AtomicBool::new(false));
            let sampler = {
                let stop = stop.clone();
                let device = device.clone();
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        record(
                            PHASE.load(Ordering::Relaxed),
                            device.current_allocated_size(),
                        );
                        std::thread::sleep(Duration::from_micros(100));
                    }
                })
            };
            let t0 = Instant::now();
            let seal = prover.prove(&inputs, po2).unwrap();
            let elapsed = t0.elapsed();
            record(
                PHASE.load(Ordering::Relaxed),
                device.current_allocated_size(),
            );
            SAMPLING.store(false, Ordering::Relaxed);
            stop.store(true, Ordering::Relaxed);
            sampler.join().unwrap();
            prover.verify(&seal).expect("Verification failed");

            let peaks: Vec<u64> = PEAKS.iter().map(|p| p.load(Ordering::Relaxed)).collect();
            let overall = *peaks.iter().max().unwrap();
            println!(
                "po2 {po2}: prove {:.2} s, baseline {:.1} MiB, peak {:.1} MiB (working set limit {:.1} MiB)",
                elapsed.as_secs_f64(),
                mib(baseline),
                mib(overall),
                mib(device.recommended_max_working_set_size()),
            );
            for (name, peak) in PHASE_NAMES.iter().zip(&peaks) {
                println!("  {name:>10}: {:>9.1} MiB", mib(*peak));
            }
            assert!(overall > 0);
        }
    }
}
