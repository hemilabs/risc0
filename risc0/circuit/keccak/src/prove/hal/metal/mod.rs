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
//! - `scatter_preflight` runs the `scatter_preflight` Metal kernel and `generate_witness` the
//!   per-arm GPU witness kernels (K6; the CPU witness generator over the shared buffers below
//!   po2 13 and in the sequential modes): see [witgen];
//! - `accumulate` does nothing, exactly as cuda.rs (the keccak circuit has no accumulation step);
//! - `eval_check` runs the staged Metal kernels `eval_stage_0..S-1` (K3's translation of the CUDA
//!   constraint code, K4's chunked launcher): see [eval_check].
//!
//! K6 adds the GPU witness.
//!
//! All keccak PSOs (`scatter_preflight`, the eval_check stages and the witgen arms) are built once
//! per process from one load of the metallib, through a binary archive on disk (K5, [archive]):
//! cold creation costs minutes and the OS shader cache was seen to miss. [warmup] builds them
//! without a HAL (and without the GPU lock), for a worker's startup.
//!
//! No `Drop` is needed: CUDA's (`cuda_reset`) resets a CUDA stack-size limit its witgen raised,
//! and Metal has no such process state; the PSOs are process-wide ([keccak_kernels]) and stay,
//! the HAL reference drops on its own, and buffers return to the HAL's pool after their last
//! batch completes.

pub(crate) mod archive;
mod eval_check;
mod witgen;

use std::{rc::Rc, sync::OnceLock};

use anyhow::Result;
use metal::{ComputePipelineState, Device, DeviceRef};
use risc0_circuit_keccak_sys::{metal_witgen, METAL_KERNEL_NAMES, METAL_LIB};
use risc0_core::scope;
use risc0_zkp::hal::{
    metal::{
        BufferImpl as MetalBuffer, MetalHal, MetalHalPoseidon2, MetalHash, MetalHashPoseidon2,
    },
    AccumPreflight, CircuitHal,
};

use crate::{
    prove::{KeccakProver, KeccakProverImpl},
    zirgen::circuit::{ExtVal, Val},
};

pub struct MetalCircuitHal<MH: MetalHash> {
    hal: Rc<MetalHal<MH>>,
    /// The keccak metallib, with a PSO for each of [kernel_names]: built once per process
    /// ([keccak_kernels]) and shared by every circuit HAL.
    kernels: &'static KeccakKernels,
    /// eval_check's chunk size and cached scratch/carry ([eval_check]).
    eval: eval_check::EvalCheckState,
}

/// The keccak PSOs of one device.
pub(crate) struct KeccakKernels {
    registry_id: u64,
    kernels: archive::ArchivedKernels,
}

impl KeccakKernels {
    pub(crate) fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.kernels.pipeline(name)
    }

    /// The binary archive's account of how the PSOs were built (K5).
    pub(crate) fn archive_report(&self) -> &archive::ArchiveReport {
        &self.kernels.report
    }
}

/// Every keccak kernel: `METAL_KERNEL_NAMES` (scatter, eval_check stages) and the witgen arms.
pub(crate) fn kernel_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = METAL_KERNEL_NAMES.to_vec();
    for name in metal_witgen::WITGEN_KERNEL_NAMES {
        if !names.contains(name) {
            names.push(name);
        }
    }
    names
}

/// The keccak PSOs (the hand-written kernels, the eval_check stages and the witgen arms), built
/// once per process, as the zkp crate's own (`zkp_kernels`): the zkvm makes a new keccak prover,
/// so a new circuit HAL, for every proof, and building the stage PSOs takes 40-190 s whenever
/// neither the binary archive ([archive]) nor the OS shader cache has them. Panics if the
/// metallib does not load or lacks a kernel. Every MetalHal uses the system default device,
/// which these PSOs are built on (asserted).
pub(crate) fn keccak_kernels(device: &DeviceRef) -> &'static KeccakKernels {
    static ONCE: OnceLock<KeccakKernels> = OnceLock::new();
    let k = ONCE.get_or_init(|| {
        scope!("load_keccak_metallib");
        let kernels = archive::ArchivedKernels::new(device, "keccak", METAL_LIB, &kernel_names())
            .unwrap_or_else(|e| panic!("keccak metallib: {e:#}"));
        KeccakKernels {
            registry_id: device.registry_id(),
            kernels,
        }
    });
    assert_eq!(
        k.registry_id,
        device.registry_id(),
        "keccak PSOs were built for another Metal device"
    );
    k
}

/// Build every keccak PSO now (through the binary archive), e.g. on a background thread at
/// worker startup, so no proof pays for it. Uses the system default device directly: it creates
/// no HAL and takes no GPU lock, so it can run beside a proof (which then waits for the same
/// `OnceLock` instead of building twice). Returns the archive report.
pub(crate) fn warmup() -> Result<archive::ArchiveReport> {
    let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
    // keccak_kernels panics on a broken metallib (a build error); report it instead.
    std::panic::catch_unwind(|| keccak_kernels(&device).archive_report().clone()).map_err(|e| {
        let msg = e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        anyhow::anyhow!("building the keccak PSOs: {msg}")
    })
}

impl<MH: MetalHash> MetalCircuitHal<MH> {
    /// A circuit HAL on `hal`; the first one in the process loads the keccak metallib and builds
    /// every PSO ([keccak_kernels]), later ones reuse them.
    pub fn new(hal: Rc<MetalHal<MH>>) -> Self {
        let kernels = keccak_kernels(&hal.device);
        Self {
            hal,
            kernels,
            eval: eval_check::EvalCheckState::new(),
        }
    }

    /// The PSO of a keccak kernel (one of [kernel_names]).
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

        // The Metal prover makes its own witness (K6: `HOST_WITNESS = false`), so it takes none
        // from `host_witness`.
        let witness = crate::prove::host_witness(&inputs, po2).unwrap();
        assert!(prover.prove_host_witness(witness).is_none());

        // Constructing the Poseidon2 circuit HAL directly also loads every PSO.
        let hal = Rc::new(MetalHalPoseidon2::new());
        let circuit_hal = MetalCircuitHalPoseidon2::new(hal);
        for name in super::kernel_names() {
            let _ = circuit_hal.pipeline(name);
        }
    }

    /// The keccak PSOs are built once per process: a second circuit HAL (as the zkvm makes for
    /// every keccak proof) shares the first one's kernels and builds no PSO.
    #[test]
    fn kernels_are_process_wide() {
        let first = MetalCircuitHalPoseidon2::new(Rc::new(MetalHalPoseidon2::new()));
        let t0 = Instant::now();
        let second = MetalCircuitHalPoseidon2::new(Rc::new(MetalHalPoseidon2::new()));
        let elapsed = t0.elapsed();
        assert!(std::ptr::eq(first.kernels, second.kernels));
        for name in super::kernel_names() {
            assert!(
                std::ptr::eq(first.pipeline(name), second.pipeline(name)),
                "{name}"
            );
        }
        assert!(
            elapsed < Duration::from_millis(500),
            "second circuit HAL took {elapsed:?}"
        );
    }

    /// The process-wide keccak PSOs cover every kernel, including K6's witgen arms (one load of
    /// the metallib, K5), and were built through the binary archive unless it is disabled.
    #[test]
    fn warmup_builds_every_pso() {
        let names = super::kernel_names();
        assert_eq!(
            names.len(),
            risc0_circuit_keccak_sys::METAL_KERNEL_NAMES.len()
                + risc0_circuit_keccak_sys::metal_witgen::WITGEN_KERNEL_NAMES.len()
        );
        let report = super::warmup().unwrap();
        assert_eq!(report.hits + report.misses, names.len(), "{report:?}");
        if super::archive::archive_dir().is_some() {
            assert!(report.path.is_some(), "{report:?}");
            // No archive error on this machine (errors fall back silently in production).
            assert!(report.errors.is_empty(), "{report:?}");
        }
        let kernels = super::keccak_kernels(&metal::Device::system_default().unwrap());
        for name in &names {
            let _ = kernels.pipeline(name);
        }
        // The witgen arms come from the same (process-wide) set.
        let hal = MetalCircuitHalPoseidon2::new(Rc::new(MetalHalPoseidon2::new()));
        for name in risc0_circuit_keccak_sys::metal_witgen::WITGEN_ARM_KERNELS {
            assert!(std::ptr::eq(hal.pipeline(name), kernels.pipeline(name)));
        }
        println!("keccak PSO report: {report:?}");
    }

    /// One prover, proofs at the po2 values in `K4_SEQ_PO2S` (default 14,17) in turn, under the
    /// working-set budget `K4_SEQ_BUDGET` (bytes, default 4 GiB, the minimum target): an earlier
    /// proof's eval_check must not leave memory behind that a later, larger proof needs (K4
    /// review: the cached eval_check scratch made po2 17 after po2 14 fail under 4 GB). Run it
    /// alone (the budget counts all of the process's Metal memory):
    /// `cargo test -p risc0-circuit-keccak --release --features metal --lib -- metal_prover_reuse_under_budget --ignored --nocapture`
    #[test]
    #[ignore]
    fn metal_prover_reuse_under_budget() {
        let po2s: Vec<usize> = std::env::var("K4_SEQ_PO2S")
            .unwrap_or_else(|_| "14,17".into())
            .split(',')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        let budget: u64 = std::env::var("K4_SEQ_BUDGET")
            .map(|s| s.trim().parse().unwrap())
            .unwrap_or(4 << 30);
        let hal = Rc::new(MetalHalPoseidon2::new());
        hal.set_working_set_limit(Some(budget));
        let circuit_hal = Rc::new(MetalCircuitHalPoseidon2::new(hal.clone()));
        // K4_SEQ_KEEP=1 restores the pre-review cache (to show this test catches it).
        let keep = std::env::var("K4_SEQ_KEEP").is_ok_and(|v| v == "1");
        circuit_hal.set_keep_eval_scratch(keep);
        let prover = KeccakProverImpl {
            hal: hal.clone(),
            circuit_hal: circuit_hal.clone(),
        };
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        for po2 in po2s {
            let seal = prover.prove(&test_inputs(po2), po2).unwrap();
            prover.verify(&seal).expect("Verification failed");
            if !keep {
                assert_eq!(circuit_hal.eval_scratch_bytes(), 0);
            }
            println!(
                "po2 {po2}: ok under {:.0} MiB; allocated after the proof {:.1} MiB",
                mib(budget),
                mib(hal.allocated_size()),
            );
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
