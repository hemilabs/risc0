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

//! Item Z12 (docs/metal-port/notes/Z12.md): the segment prover's memory during recursion.
//!
//! `z12_stark_phases` runs the worker's staged STARK (segments as a composite receipt, then
//! `compress` to succinct: lift/join) on a BusyLoop guest, `Z12_REPS` times in one process, and
//! prints per phase the wall time and the peak of the device's `currentAllocatedSize` (sampled
//! every 2 ms), plus what is still allocated when recursion starts. Between repetitions it calls
//! `release_hal_buffers` (what the Groth16 wrap does after every worker proof). With `Z12_AB=1`
//! the repetitions alternate `RISC0_METAL_RELEASE_SEG_BEFORE_REC` between unset (B, the default)
//! (or `Z12_AB_B`'s value) and `0` (A), starting with A, for an interleaved A/B in one process.
//!
//! `z12_release_reason` pins the decision and `z12_release_then_reprove` proves through a forced
//! release (a succinct receipt that verifies, then a second session on the rebuilt prover).
//!
//! `z12_stark_phases` reads the working-set budget from the environment (`RISC0_METAL_MAX_WORKING_SET_BYTES`), so
//! it is `#[ignore]`d and run on its own:
//! `cargo test -p risc0-zkvm --features metal --release --lib -- tests_z12::z12_stark_phases
//! --ignored --nocapture`. Knobs: `Z12_PO2` (default 18), `Z12_CYCLES` (default 2^21),
//! `Z12_REPS` (default 1), `Z12_AB`.

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use risc0_zkp::hal::metal::MetalHalPoseidon2;
use risc0_zkvm_methods::{multi_test::MultiTestSpec, MULTI_TEST_ELF, MULTI_TEST_ID};

use super::prover_impl::{
    seg_release_reason, RELEASE_SEG_BEFORE_REC_ENV, RELEASE_SEG_MODE, SEG_RELEASES,
};
use crate::{get_prover_server, sha::Digestible, ExecutorEnv, InnerReceipt, ProverOpts};

/// Peak of the device's `currentAllocatedSize` since the last [Sampler::phase].
struct Sampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    now: Arc<AtomicU64>,
    handle: Option<JoinHandle<()>>,
}

impl Sampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let now = Arc::new(AtomicU64::new(0));
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = {
            let (stop, peak, now) = (stop.clone(), peak.clone(), now.clone());
            std::thread::spawn(move || {
                let hal = MetalHalPoseidon2::new();
                tx.send(()).unwrap();
                while !stop.load(Ordering::Relaxed) {
                    let a = hal.allocated_size();
                    now.store(a, Ordering::Relaxed);
                    peak.fetch_max(a, Ordering::Relaxed);
                    std::thread::sleep(Duration::from_millis(2));
                }
            })
        };
        rx.recv().unwrap();
        std::thread::sleep(Duration::from_millis(10));
        Self {
            stop,
            peak,
            now,
            handle: Some(handle),
        }
    }

    fn current(&self) -> u64 {
        std::thread::sleep(Duration::from_millis(10));
        self.now.load(Ordering::Relaxed)
    }

    /// The peak since the last call, then restart it at the current value.
    fn phase(&self) -> u64 {
        std::thread::sleep(Duration::from_millis(10));
        let peak = self.peak.load(Ordering::Relaxed);
        self.peak
            .store(self.now.load(Ordering::Relaxed), Ordering::Relaxed);
        peak
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn mib(b: u64) -> f64 {
    b as f64 / (1u64 << 20) as f64
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "reads the process-wide working-set budget and times whole proofs; run alone"]
fn z12_stark_phases() {
    let po2 = env_u64("Z12_PO2", 18) as u32;
    let cycles = env_u64("Z12_CYCLES", 1 << 21);
    let reps = env_u64("Z12_REPS", 1);
    let ab = std::env::var("Z12_AB").is_ok_and(|v| v == "1");
    let budget = std::env::var("RISC0_METAL_MAX_WORKING_SET_BYTES").unwrap_or("none".into());
    let sampler = Sampler::start();
    let base = sampler.current();
    for rep in 0..reps {
        let arm = if ab {
            if rep % 2 == 0 {
                std::env::set_var(RELEASE_SEG_BEFORE_REC_ENV, "0");
                "A"
            } else {
                match std::env::var("Z12_AB_B") {
                    Ok(v) => std::env::set_var(RELEASE_SEG_BEFORE_REC_ENV, v),
                    Err(_) => std::env::remove_var(RELEASE_SEG_BEFORE_REC_ENV),
                }
                "B"
            }
        } else {
            "-"
        };
        sampler.phase();
        let env = ExecutorEnv::builder()
            .write(&MultiTestSpec::BusyLoop { cycles })
            .unwrap()
            .segment_limit_po2(po2)
            .build()
            .unwrap();
        let opts = ProverOpts::composite();
        let t = Instant::now();
        let composite = get_prover_server(&opts)
            .unwrap()
            .prove(env, MULTI_TEST_ELF)
            .unwrap_or_else(|e| panic!("segments failed (budget {budget}, po2 {po2}): {e:#}"))
            .receipt;
        let seg_s = t.elapsed().as_secs_f64();
        let seg_peak = sampler.phase();
        let held = sampler.current();
        let segments = composite.inner.composite().unwrap().segments.len();

        let releases = SEG_RELEASES.with(|n| n.get());
        let sopts = ProverOpts::succinct();
        let t = Instant::now();
        let succinct = get_prover_server(&sopts)
            .unwrap()
            .compress(&sopts, &composite)
            .unwrap_or_else(|e| panic!("recursion failed (budget {budget}, po2 {po2}): {e:#}"));
        let rec_s = t.elapsed().as_secs_f64();
        let rec_peak = sampler.phase();
        let released = SEG_RELEASES.with(|n| n.get()) - releases;
        assert!(matches!(succinct.inner, InnerReceipt::Succinct(_)));
        succinct.verify(MULTI_TEST_ID).unwrap();
        let after = sampler.current();
        // What the Groth16 wrap does after every worker proof.
        crate::release_hal_buffers();
        let cleared = sampler.current();
        eprintln!(
            "[Z12] rep={rep} arm={arm} budget={budget} po2={po2} cycles={cycles} \
             segments={segments} base={:.0} MiB | segments {seg_s:.3}s peak {:.0} MiB, held \
             {:.0} MiB | seg-release={released} | recursion {rec_s:.3}s peak {:.0} MiB, after \
             {:.0} MiB | stark {:.3}s | after release_hal_buffers {:.0} MiB",
            mib(base),
            mib(seg_peak),
            mib(held),
            mib(rec_peak),
            mib(after),
            seg_s + rec_s,
            mib(cleared),
        );
    }
}

const GIB: u64 = 1 << 30;

#[test]
fn z12_release_reason() {
    // The knob wins over the budget.
    assert_eq!(seg_release_reason("0", Some((4 * GIB, 4 * GIB))), None);
    assert!(seg_release_reason("1", Some((28 * GIB, 0))).is_some());
    assert!(seg_release_reason(" 1 ", None).is_some());
    // Auto: release only when held + recursion's ~2 GiB would not fit.
    // Default budget (36 GB Mac, 28 GiB), po2 20: 9.4 GB held -> keep.
    assert_eq!(seg_release_reason("", Some((28 * GIB, 9352 << 20))), None);
    // 4 GiB, po2 18: 1.5 GB held -> fits, keep.
    assert_eq!(seg_release_reason("", Some((4 * GIB, 1516 << 20))), None);
    // 4 GiB with 2.5 GB held -> release.
    assert!(seg_release_reason("auto", Some((4 * GIB, 2560 << 20))).is_some());
    // 8 GiB, po2 19: 3.4 GB held -> keep; with 6.5 GB held -> release.
    assert_eq!(seg_release_reason("", Some((8 * GIB, 3425 << 20))), None);
    assert!(seg_release_reason("", Some((8 * GIB, 6656 << 20))).is_some());
    // No Metal device: nothing to release.
    assert_eq!(seg_release_reason("", None), None);
}

/// A forced release before lift/join: the succinct receipt verifies, the release happened, and
/// the next session on this thread rebuilds the segment prover and proves the same journal.
/// Uses the thread-local mode override, so it does not change the tests running beside it.
#[test]
fn z12_release_then_reprove() {
    let prove = |mode: &'static str| {
        RELEASE_SEG_MODE.with(|m| *m.borrow_mut() = Some(mode));
        let before = SEG_RELEASES.with(|n| n.get());
        let env = ExecutorEnv::builder()
            .write(&MultiTestSpec::BusyLoop { cycles: 1 << 17 })
            .unwrap()
            .segment_limit_po2(16)
            .build()
            .unwrap();
        let opts = ProverOpts::succinct();
        let receipt = get_prover_server(&opts)
            .unwrap()
            .prove(env, MULTI_TEST_ELF)
            .unwrap()
            .receipt;
        RELEASE_SEG_MODE.with(|m| *m.borrow_mut() = None);
        assert!(matches!(receipt.inner, InnerReceipt::Succinct(_)));
        receipt.verify(MULTI_TEST_ID).unwrap();
        (receipt, SEG_RELEASES.with(|n| n.get()) - before)
    };
    let (released, n1) = prove("1");
    assert_eq!(n1, 1, "forced mode did not release before recursion");
    let (kept, n0) = prove("0");
    assert_eq!(n0, 0, "mode 0 released");
    assert_eq!(released.journal.bytes, kept.journal.bytes);
    assert_eq!(
        released.claim().unwrap().digest(),
        kept.claim().unwrap().digest()
    );
}
