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

//! End-to-end tests of the Metal Groth16 prover (item G7; PLAN.md 4.5 acceptance and errata 8.1's
//! `metal_proofs_verify`): wraps of a fixed `identity_p254` seal, each verified with
//! [crate::Verifier] both from the zkvm's view (control root, claim digest, BN254 control id) and
//! from the prover's `proof.json` + `public.json`, whose public signals must equal the ones the
//! reference witness generator (circom-witnesscalc over [to_json](super::super::to_json)) computes.
//!
//! The fixture (`tests/data/identity_p254_seal.{bin,json}`) is the seal of
//! `identity_p254(prove_nothing_succinct())` from risc0-zkvm's tests (multi_test `DoNothing`),
//! with the claim digest and the default Groth16 verifier parameters it must verify against.
//! Needs the rzup `risc0-groth16` component (`RISC0_GROTH16_DIR` or rzup's directory); skips
//! without it. Under a `RISC0_METAL_MAX_WORKING_SET_BYTES` budget too small for the prover, the
//! wrap must fail with the clean budget error (and the test passes, saying so).

use std::{
    ffi::OsString,
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, MutexGuard, OnceLock,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use hex::FromHex as _;
use risc0_zkp::core::digest::Digest;

use super::*;
use crate::{verifying_key, Fr, PublicInputsJson, Verifier};

const SEAL_BYTES: &[u8] = include_bytes!("../../tests/data/identity_p254_seal.bin");
const META: &str = include_str!("../../tests/data/identity_p254_seal.json");

/// The fixture seal, 4-byte aligned: `fast::witness` and `to_json` read it with
/// `bytemuck::cast_slice::<u8, u32>`, which panics on the unaligned `include_bytes!` data.
fn seal_words() -> &'static [u8] {
    static ONCE: OnceLock<Vec<u32>> = OnceLock::new();
    let words = ONCE.get_or_init(|| {
        SEAL_BYTES
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    });
    bytemuck::cast_slice(words)
}

/// Serializes this crate's tests that use the GPU prover, its process-wide cache, the device-wide
/// allocation figure or the working-set budget variable (PLAN 8.7), so that they hold under the
/// default parallel harness. Poison-tolerant.
pub(super) fn gpu_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Sets (or removes) an environment variable for the life of the guard, then restores it. Only
/// under [gpu_test_lock].
struct EnvGuard {
    name: &'static str,
    old: Option<OsString>,
}

impl EnvGuard {
    fn set(name: &'static str, value: Option<String>) -> Self {
        let old = std::env::var_os(name);
        match value {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
        Self { name, old }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match self.old.take() {
            Some(v) => std::env::set_var(self.name, v),
            None => std::env::remove_var(self.name),
        }
    }
}

/// Peak of the device's `currentAllocatedSize`, sampled every millisecond on its own thread.
struct PeakSampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    handle: Option<JoinHandle<()>>,
}

impl PeakSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(risc0_groth16_sys::metal::allocated_bytes()));
        let handle = {
            let (stop, peak) = (stop.clone(), peak.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    peak.fetch_max(
                        risc0_groth16_sys::metal::allocated_bytes(),
                        Ordering::Relaxed,
                    );
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
        };
        Self {
            stop,
            peak,
            handle: Some(handle),
        }
    }

    fn finish(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().unwrap();
        }
        self.peak.load(Ordering::Relaxed)
    }
}

fn mib(b: u64) -> f64 {
    b as f64 / (1u64 << 20) as f64
}

/// (control_root, claim_digest, bn254_control_id) of the fixture.
fn fixture_meta() -> (Digest, Digest, Digest) {
    let v: serde_json::Value = serde_json::from_str(META).unwrap();
    let d = |k: &str| Digest::from_hex(v[k].as_str().unwrap()).unwrap();
    (d("control_root"), d("claim_digest"), d("bn254_control_id"))
}

fn available() -> bool {
    if !enabled() {
        eprintln!("skipped: the Metal Groth16 prover is not enabled here");
        return false;
    }
    if let Err(e) = component_dir() {
        eprintln!("skipped: {e:#}");
        return false;
    }
    true
}

/// Whether `err` is the clean working-set budget error and a budget is set (budget runs).
fn refused_by_budget(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}");
    let refused = risc0_groth16_sys::metal::working_set_budget().is_some()
        && msg.contains("working-set budget exceeded");
    if refused {
        eprintln!("refused under the working-set budget (clean error): {msg}");
    }
    refused
}

/// The public signals of the fixture's witness from the reference pipeline: seal_to_json ->
/// circom-witnesscalc's own evaluator -> witness[1..=nPublic], as decimal strings.
fn reference_public() -> &'static Vec<String> {
    static ONCE: OnceLock<Vec<String>> = OnceLock::new();
    ONCE.get_or_init(|| {
        let t = Instant::now();
        let input = super::super::to_json(seal_words()).unwrap();
        let graph = std::fs::read(super::super::fast::graph_path().unwrap()).unwrap();
        let wtns = circom_witnesscalc::calc_witness(&input, &graph).unwrap();
        let file = wtns_file::WtnsFile::<32>::read(&wtns[..]).unwrap();
        let n_public = 5;
        let out: Vec<String> = file.witness.0[1..=n_public]
            .iter()
            .map(|e| {
                use ark_ff::PrimeField as _;
                ark_bn254::Fr::from_le_bytes_mod_order(e.as_bytes()).to_string()
            })
            .collect();
        eprintln!("reference witness (circom-witnesscalc): {:?}", t.elapsed());
        out
    })
}

/// Verify `seal` against the fixture's claim (as `Groth16Receipt::verify_integrity` does) and
/// the prover's `proof.json`/`public.json` in `dir` (as `tests/verify.rs` does), and check
/// `public.json` against the reference witness generator.
fn check_proof(seal: &Seal, dir: &Path) {
    let (control_root, claim_digest, bn254_id) = fixture_meta();
    Verifier::new(
        &seal.to_vec(),
        control_root,
        claim_digest,
        bn254_id,
        &verifying_key(),
    )
    .unwrap()
    .verify()
    .expect("the Metal proof does not verify against the fixture's claim");

    let proof: crate::ProofJson =
        serde_json::from_str(&std::fs::read_to_string(dir.join("proof.json")).unwrap()).unwrap();
    let public = PublicInputsJson {
        values: serde_json::from_str(&std::fs::read_to_string(dir.join("public.json")).unwrap())
            .unwrap(),
    };
    assert_eq!(&public.values, reference_public(), "public.json");
    let scalars: Vec<Fr> = public.to_scalar().unwrap();
    Verifier::new_inner(&proof.try_into().unwrap(), &scalars, &verifying_key())
        .unwrap()
        .verify()
        .expect("proof.json does not verify with public.json");
}

/// Three wraps of the fixture through the GPU path, each verified (claim and JSON files);
/// blinding makes every proof different.
#[test]
fn metal_shrink_wrap() {
    let _lock = gpu_test_lock();
    if !available() {
        return;
    }
    let mut seals = Vec::new();
    for i in 0..3 {
        let dir = tempfile::tempdir().unwrap();
        let t = Instant::now();
        let r = gpu_prove(seal_words(), &mut Some(|| {}), dir.path());
        let seal = match r {
            Err(e) if refused_by_budget(&e) => return,
            r => r.unwrap(),
        };
        eprintln!("metal_shrink_wrap {i}: {:?}", t.elapsed());
        check_proof(&seal, dir.path());
        seals.push(seal.to_vec());
    }
    assert!(seals[0] != seals[1] && seals[1] != seals[2] && seals[0] != seals[2]);
}

/// Release + reprove through the crate's public entry points (`prove::release_srs`,
/// `prove::shrink_wrap`, `prove::shrink_wrap_with`), every proof verified against the claim.
#[test]
fn metal_proofs_verify() {
    let _lock = gpu_test_lock();
    if !available() {
        return;
    }
    let (control_root, claim_digest, bn254_id) = fixture_meta();
    let verify = |seal: &Seal| {
        Verifier::new(
            &seal.to_vec(),
            control_root,
            claim_digest,
            bn254_id,
            &verifying_key(),
        )
        .unwrap()
        .verify()
        .unwrap()
    };
    for round in 0..2 {
        crate::prove::release_srs().unwrap();
        let t = Instant::now();
        let seal = match crate::prove::shrink_wrap(seal_words()) {
            Err(e) if refused_by_budget(&e) => return,
            r => r.unwrap(),
        };
        eprintln!(
            "round {round}: release + shrink_wrap (cold key) {:?}",
            t.elapsed()
        );
        verify(&seal);
        let t = Instant::now();
        let mut ran = 0;
        let seal = crate::prove::shrink_wrap_with(seal_words(), || ran += 1).unwrap();
        eprintln!(
            "round {round}: shrink_wrap_with (key cached) {:?}",
            t.elapsed()
        );
        assert_eq!(ran, 1);
        verify(&seal);
    }
    crate::prove::release_srs().unwrap();
}

/// What the STARK prover's HAL pools hold when the wrap starts, in the M1 failure: 4602757120
/// bytes ("already allocated") at po2 18 under an 8 GiB budget.
const STARK_POOL_BYTES: u64 = 4602757120;

/// The zkvm's wrap after a STARK proof (`succinct_to_groth16`), with the STARK HAL's pools
/// modelled by `STARK_POOL_BYTES` of device buffers that `pre_gpu` frees (as
/// `release_hal_buffers` drops the cached provers, after ~100 ms): `prestart_prover` first (the
/// zkvm calls it before identity_p254), then `shrink_wrap_with`. Under `budget` (None: no budget,
/// whatever the environment says). Returns (seal, peak allocation, allocation before).
fn wrap_beside_stark_pools(budget: Option<u64>) -> (Result<Seal>, u64, u64) {
    let _env = EnvGuard::set(
        risc0_groth16_sys::metal::MAX_WORKING_SET_ENV,
        budget.map(|b| b.to_string()),
    );
    crate::prove::release_srs().unwrap();
    let device = &risc0_groth16_sys::metal::prover::kernels().unwrap().device;
    let before = risc0_groth16_sys::metal::allocated_bytes();
    let sampler = PeakSampler::start();
    let pools: Vec<_> = (0..4)
        .map(|_| {
            risc0_groth16_sys::metal::new_buffer_checked(
                device,
                STARK_POOL_BYTES / 4,
                true,
                "a modelled STARK pool",
            )
            .unwrap()
        })
        .collect();
    crate::prove::prestart_prover().unwrap();
    // identity_p254 runs on the CPU meanwhile (~13 s in the worker).
    std::thread::sleep(Duration::from_millis(1500));
    let seal = crate::prove::shrink_wrap_with(seal_words(), move || {
        std::thread::sleep(Duration::from_millis(100));
        drop(pools);
    });
    let peak = sampler.finish();
    (seal, peak, before)
}

/// M1's P0 for 16 GB Macs (G8): under an 8 GiB budget, a wrap after a STARK proof must load the
/// key only after `pre_gpu` has released the STARK pools. Before, the key loader started before
/// `pre_gpu` (and `prestart_prover` loaded during identity_p254), so the load was checked against
/// memory about to be freed and failed: "Metal working-set budget exceeded: allocating 5930236936
/// bytes for the Groth16 prover ... with 4602757120 bytes already allocated ...".
#[test]
fn wrap_after_stark_under_8gib_budget() {
    let _lock = gpu_test_lock();
    if !available() {
        return;
    }
    let budget = 8u64 << 30;
    let (seal, peak, before) = wrap_beside_stark_pools(Some(budget));
    let seal = seal.unwrap_or_else(|e| panic!("the wrap failed under the 8 GiB budget: {e:#}"));
    let (control_root, claim_digest, bn254_id) = fixture_meta();
    Verifier::new(
        &seal.to_vec(),
        control_root,
        claim_digest,
        bn254_id,
        &verifying_key(),
    )
    .unwrap()
    .verify()
    .unwrap();
    eprintln!(
        "wrap after modelled STARK pools under 8 GiB: before {:.1} MiB, peak {:.1} MiB",
        mib(before),
        mib(peak)
    );
    assert!(peak <= budget, "peak {peak} above the budget {budget}");
    crate::prove::release_srs().unwrap();
}

/// Without a budget the same wrap must not hold the key beside the STARK pools: the peak is about
/// max(pools, key), not pools + key (M1 measured 16.6-17.6 GB instead of ~7.4 GB at po2 20).
#[test]
fn wrap_after_stark_peak_is_not_stark_plus_key() {
    let _lock = gpu_test_lock();
    if !available() {
        return;
    }
    let (seal, peak, before) = wrap_beside_stark_pools(None);
    seal.unwrap();
    let key = risc0_groth16_sys::metal::allocated_bytes().saturating_sub(before);
    eprintln!(
        "wrap after modelled STARK pools, no budget: before {:.1} MiB, peak {:.1} MiB, prover \
         {:.1} MiB",
        mib(before),
        mib(peak),
        mib(key)
    );
    assert!(
        key > STARK_POOL_BYTES,
        "the prover is not cached ({key} bytes)"
    );
    assert!(
        peak < before + key + STARK_POOL_BYTES / 2,
        "the key was loaded beside the STARK pools: peak {:.1} MiB, before {:.1}, prover {:.1}, \
         pools {:.1}",
        mib(peak),
        mib(before),
        mib(key),
        mib(STARK_POOL_BYTES)
    );
    crate::prove::release_srs().unwrap();
}

/// A seal that is not 4-byte aligned (any `&[u8]` a caller may pass) wraps like an aligned one.
/// Before (G8), `fast::witness` and `to_json` read it with `bytemuck::cast_slice::<u8, u32>`,
/// which panics on unaligned data ("groth16 witness thread panicked").
#[test]
fn unaligned_seal_wraps() {
    let _lock = gpu_test_lock();
    if !available() {
        return;
    }
    let mut buf = vec![0u32; SEAL_BYTES.len() / 4 + 1];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut buf);
    bytes[1..=SEAL_BYTES.len()].copy_from_slice(SEAL_BYTES);
    let unaligned = &bytes[1..=SEAL_BYTES.len()];
    assert_ne!(unaligned.as_ptr() as usize % 4, 0);
    let dir = tempfile::tempdir().unwrap();
    let seal = match gpu_prove(unaligned, &mut Some(|| {}), dir.path()) {
        Err(e) if refused_by_budget(&e) => return,
        r => r.unwrap_or_else(|e| panic!("{e:#}")),
    };
    check_proof(&seal, dir.path());
    // And through the public entry point.
    match crate::prove::shrink_wrap(unaligned) {
        Err(e) if refused_by_budget(&e) => {}
        r => {
            r.unwrap_or_else(|e| panic!("{e:#}"));
        }
    }
}
