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

//! The Groth16 wrap after a STARK proof on Metal under a working-set budget (item G8,
//! docs/metal-port/notes/G8.md; zk-miner M1's P0 for 16 GB Macs), and the public
//! `release_hal_buffers`.
//!
//! `metal_wrap_after_stark` proves a succinct receipt (segments + lift/join: the cached segment
//! and recursion provers keep their HAL pools on this thread afterwards, as in a worker), then
//! compresses it to Groth16 (`succinct_to_groth16`: identity_p254 on the CPU, then
//! `shrink_wrap_with(.., release_hal_buffers)`), sampling the device's `currentAllocatedSize`
//! every 2 ms. Before G8 the key load started before the release: under an 8 GiB budget the wrap
//! failed ("... with 4602757120 bytes already allocated would exceed the limit ..."), without one
//! it peaked at STARK + key.
//!
//! It sets `RISC0_METAL_MAX_WORKING_SET_BYTES` for the whole process, so it is `#[ignore]`d and run
//! on its own:
//! `cargo test -p risc0-zkvm --features metal --lib -- tests_metal_wrap --ignored --nocapture`.
//! Knobs: `RISC0_G8_BUDGET` (bytes, default 8589934592; `none` for no budget), `RISC0_G8_PO2`
//! (default 18), `RISC0_G8_CYCLES` (default 2^19), `RISC0_G8_JOBS` (default 1: STARK + wrap
//! jobs in one process, as a worker runs them; from the second job on, identity_p254 rebuilds
//! its ctrl group from the host copy C13 keeps instead of committing it).

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use risc0_zkp::hal::{metal::MetalHalPoseidon2, Hal as _};
use risc0_zkvm_methods::{multi_test::MultiTestSpec, MULTI_TEST_ELF, MULTI_TEST_ID};

use crate::{get_prover_server, ExecutorEnv, InnerReceipt, ProverOpts};

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
                // The device's currentAllocatedSize through a HAL made here, before any session
                // holds the GPU lock (`MetalHal::new` takes it); reading it takes no lock.
                let hal = MetalHalPoseidon2::new();
                tx.send(()).unwrap();
                while !stop.load(Ordering::Relaxed) {
                    let a = (hal.gpu_total_memory() - hal.gpu_free_memory()) as u64;
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
#[ignore = "sets the process-wide working-set budget; run alone (see the module docs)"]
fn metal_wrap_after_stark() {
    let budget = match std::env::var("RISC0_G8_BUDGET").as_deref() {
        Ok("none") => None,
        Ok(v) => Some(v.parse::<u64>().expect("RISC0_G8_BUDGET")),
        Err(_) => Some(8u64 << 30),
    };
    let po2 = env_u64("RISC0_G8_PO2", 18) as u32;
    let cycles = env_u64("RISC0_G8_CYCLES", 1 << 19);
    let jobs = env_u64("RISC0_G8_JOBS", 1);
    match budget {
        Some(b) => std::env::set_var("RISC0_METAL_MAX_WORKING_SET_BYTES", b.to_string()),
        None => std::env::remove_var("RISC0_METAL_MAX_WORKING_SET_BYTES"),
    }
    // The key must not be cached from before (that is the caller's choice, not the wrap's).
    risc0_groth16::prove::release_srs().unwrap();

    let sampler = Sampler::start();
    let base = sampler.current();
    sampler.phase();
    for job in 0..jobs {
        wrap_job(&sampler, base, budget, po2, cycles, job);
    }
}

fn wrap_job(sampler: &Sampler, base: u64, budget: Option<u64>, po2: u32, cycles: u64, job: u64) {
    // STARK: segments + lift/join, pools stay cached on this thread.
    let env = ExecutorEnv::builder()
        .write(&MultiTestSpec::BusyLoop { cycles })
        .unwrap()
        .segment_limit_po2(po2)
        .build()
        .unwrap();
    let t = Instant::now();
    let succinct = get_prover_server(&ProverOpts::succinct())
        .unwrap()
        .prove(env, MULTI_TEST_ELF)
        .unwrap()
        .receipt;
    let stark_s = t.elapsed().as_secs_f64();
    let stark_peak = sampler.phase();
    let held = sampler.current();
    succinct.verify(MULTI_TEST_ID).unwrap();

    // Wrap: identity_p254 (CPU) + shrink_wrap (release, key load, prove).
    let opts = ProverOpts::groth16();
    let t = Instant::now();
    let groth16 = get_prover_server(&opts)
        .unwrap()
        .compress(&opts, &succinct)
        .unwrap_or_else(|e| panic!("the wrap failed (budget {budget:?}): {e:#}"));
    let wrap_s = t.elapsed().as_secs_f64();
    let wrap_peak = sampler.phase();
    assert!(matches!(groth16.inner, InnerReceipt::Groth16(_)));
    groth16.verify(MULTI_TEST_ID).unwrap();
    let after_wrap = sampler.current();

    // The public release (a worker's call after a failed proof) on what is left.
    crate::release_hal_buffers();
    risc0_groth16::prove::release_srs().unwrap();
    let released = sampler.current();

    eprintln!(
        "[G8] job={job} budget={} po2={po2} cycles={cycles} base={:.1} MiB | STARK: {stark_s:.2}s peak \
         {:.1} MiB, held before the wrap {:.1} MiB | wrap (identity_p254 + shrink_wrap): \
         {wrap_s:.2}s peak {:.1} MiB, after {:.1} MiB | after release_hal_buffers + release_srs \
         {:.1} MiB",
        budget.map_or("none".to_string(), |b| b.to_string()),
        mib(base),
        mib(stark_peak),
        mib(held),
        mib(wrap_peak),
        mib(after_wrap),
        mib(released),
    );
    if let Some(b) = budget {
        assert!(wrap_peak <= b, "wrap peak {wrap_peak} above the budget {b}");
    }
    // The key (5.7 GB) never sits beside the STARK pools: the wrap's peak stays well below
    // pools + key (it is about max(pools, key + identity_p254's CPU-side nothing)).
    assert!(
        wrap_peak < held + (5u64 << 30),
        "the wrap held the key beside the STARK pools: held {:.1} MiB, wrap peak {:.1} MiB",
        mib(held),
        mib(wrap_peak)
    );
}
