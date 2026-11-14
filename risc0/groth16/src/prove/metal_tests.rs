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
//! Item P31-12 regenerated it with the 3.1 CPU prover (no GPU feature; control root e6feef13...,
//! BN254 identity control id de81d091..., Poseidon254 x^7), for the v0.2.0 key; see
//! docs/metal-port/notes/P31-12.md for how.
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
    /// Peak of the process' physical footprint (G10).
    footprint: Arc<AtomicU64>,
    handle: Option<JoinHandle<()>>,
}

impl PeakSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(risc0_groth16_sys::metal::allocated_bytes()));
        let footprint = Arc::new(AtomicU64::new(0));
        let handle = {
            let (stop, peak, footprint) = (stop.clone(), peak.clone(), footprint.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    peak.fetch_max(
                        risc0_groth16_sys::metal::allocated_bytes(),
                        Ordering::Relaxed,
                    );
                    footprint.fetch_max(
                        risc0_groth16_sys::metal::filemap::process_memory().phys_footprint,
                        Ordering::Relaxed,
                    );
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
        };
        Self {
            stop,
            peak,
            footprint,
            handle: Some(handle),
        }
    }

    fn finish(self) -> u64 {
        self.finish_both().0
    }

    /// (peak device allocation, peak process footprint).
    fn finish_both(mut self) -> (u64, u64) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().unwrap();
        }
        (
            self.peak.load(Ordering::Relaxed),
            self.footprint.load(Ordering::Relaxed),
        )
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
    // G10: the prover (4.1 GB with the default memory switches and the v0.2.0 key, P31-12) is now
    // smaller than the pools.
    assert!(key > 1 << 30, "the prover is not cached ({key} bytes)");
    assert!(
        peak < before + key.max(STARK_POOL_BYTES) + key.min(STARK_POOL_BYTES) / 2,
        "the key was loaded beside the STARK pools: peak {:.1} MiB, before {:.1}, prover {:.1}, \
         pools {:.1}",
        mib(peak),
        mib(before),
        mib(key),
        mib(STARK_POOL_BYTES)
    );
    crate::prove::release_srs().unwrap();
}

/// Item G13 (c): with nothing else allocated, `prestart_prover` loads the key in the background
/// (identity_p254 would run meanwhile) when the working-set limit has room for it and
/// identity_p254's reserve, and the wrap then finds it cached; under a 4 GiB budget it does not
/// load early (the wrap loads it after `pre_gpu`, and still fits). Each proof verifies.
#[test]
fn early_key_load_follows_the_budget() {
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
    for (budget, early) in [
        (None, true),
        (Some(8u64 << 30), true),
        (Some(4u64 << 30), false),
    ] {
        let _env = EnvGuard::set(
            risc0_groth16_sys::metal::MAX_WORKING_SET_ENV,
            budget.map(|b| b.to_string()),
        );
        let _knob = EnvGuard::set("RISC0_G16_METAL_EARLY_KEY", None);
        crate::prove::release_srs().unwrap();
        let before = risc0_groth16_sys::metal::allocated_bytes();
        crate::prove::prestart_prover().unwrap();
        // identity_p254 takes ~1.1 s; the early load ~0.2-0.5 s.
        let t = Instant::now();
        while !risc0_groth16_sys::metal::metal_is_loaded() && t.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(20));
        }
        let loaded = risc0_groth16_sys::metal::metal_is_loaded();
        eprintln!(
            "budget {budget:?}: before {:.1} MiB; loaded early: {loaded}; allocated {:.1} MiB",
            mib(before),
            mib(risc0_groth16_sys::metal::allocated_bytes())
        );
        if before > 1 << 30 {
            eprintln!("skipped: {:.1} MiB already allocated", mib(before));
            return;
        }
        assert_eq!(loaded, early, "budget {budget:?}");
        let seal = crate::prove::shrink_wrap(seal_words()).unwrap_or_else(|e| panic!("{e:#}"));
        verify(&seal);
    }
    crate::prove::release_srs().unwrap();
}

/// Item G10: with the default memory switches the whole wrap (key load, witness, proof) fits a
/// 4 GiB working-set budget (the minimum tier, PLAN 9.1); with the G9 layout
/// (`RISC0_G16_METAL_G10=0`) the same budget refuses it cleanly before loading. Reports the peak
/// device allocation and the process' peak physical footprint of a cold wrap and a cached one.
#[test]
fn wrap_under_4gib_budget() {
    let _lock = gpu_test_lock();
    if !available() {
        return;
    }
    let budget = 4u64 << 30;
    let _env = EnvGuard::set(
        risc0_groth16_sys::metal::MAX_WORKING_SET_ENV,
        Some(budget.to_string()),
    );
    {
        let _g9 = EnvGuard::set("RISC0_G16_METAL_G10", Some("0".into()));
        crate::prove::release_srs().unwrap();
        let err = crate::prove::shrink_wrap(seal_words()).unwrap_err();
        assert!(
            format!("{err:#}").contains("working-set budget exceeded"),
            "G9 layout under 4 GiB: {err:#}"
        );
    }
    if risc0_groth16_sys::metal::prover::MemOptions::from_env()
        != risc0_groth16_sys::metal::prover::MemOptions::all()
    {
        eprintln!("skipped: a G10 memory switch is off in the environment (revert run)");
        return;
    }
    let (control_root, claim_digest, bn254_id) = fixture_meta();
    crate::prove::release_srs().unwrap();
    for what in ["cold key", "key cached"] {
        let before = risc0_groth16_sys::metal::allocated_bytes();
        let sampler = PeakSampler::start();
        let t = Instant::now();
        let seal = crate::prove::shrink_wrap(seal_words())
            .unwrap_or_else(|e| panic!("the wrap failed under 4 GiB ({what}): {e:#}"));
        let elapsed = t.elapsed();
        let (peak, footprint) = sampler.finish_both();
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
            "G10 wrap under 4 GiB ({what}): {elapsed:?}; device allocation before {:.1} MiB, peak \
             {:.1} MiB; process peak footprint {:.1} MiB",
            mib(before),
            mib(peak),
            mib(footprint)
        );
        assert!(peak <= budget, "peak {peak} above the budget {budget}");
    }
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

/// The full witness of the reference pipeline (seal_to_json -> circom-witnesscalc's own
/// evaluator), as canonical little-endian bytes per element.
fn reference_witness(seal: &[u8]) -> Vec<[u8; 32]> {
    let input = super::super::to_json(seal).unwrap();
    let graph = std::fs::read(super::super::fast::graph_path().unwrap()).unwrap();
    let wtns = circom_witnesscalc::calc_witness(&input, &graph).unwrap();
    let file = wtns_file::WtnsFile::<32>::read(&wtns[..]).unwrap();
    file.witness
        .0
        .iter()
        .map(|e| e.as_bytes().try_into().unwrap())
        .collect()
}

/// A second seal for the witness comparisons: the fixture with some field elements changed
/// (the graph evaluates any input; the result need not be a valid proof's witness).
fn perturbed_seal() -> Vec<u32> {
    use super::super::seal_format::{IopType, K_SEAL_ELEMS, K_SEAL_TYPES};
    let words: &[u32] = bytemuck::cast_slice(seal_words());
    let mut w = words.to_vec();
    let mut pos = 0;
    for (k, t) in K_SEAL_TYPES.iter().take(K_SEAL_ELEMS).enumerate() {
        let x = 0x5a5a_1234u32.wrapping_mul(k as u32 + 1);
        match t {
            // A BabyBear element: stay below the modulus.
            IopType::Fp => {
                if k % 7 == 0 {
                    w[pos] = (w[pos] ^ x) % 0x7800_0001;
                }
                pos += 1;
            }
            // A digest (< the BN254 modulus): change its lowest word only.
            _ => {
                if k % 5 == 0 {
                    w[pos] ^= x;
                }
                pos += 8;
            }
        }
    }
    w
}

/// Item G13: the in-process witness (the memoized graph build, `fast::witness`) is bit-equal to
/// the reference generator's, element by element, for the fixture and a perturbed seal.
#[test]
fn witness_bit_equal_to_reference() {
    if component_dir().is_err() {
        eprintln!("skipped: no risc0-groth16 component");
        return;
    }
    let perturbed = perturbed_seal();
    for seal in [seal_words(), bytemuck::cast_slice(&perturbed)] {
        let reference = reference_witness(seal);
        let ours = super::super::fast::witness(seal).unwrap();
        assert_eq!(ours.len(), reference.len());
        for (i, (a, b)) in ours.iter().zip(reference.iter()).enumerate() {
            assert_eq!(a.as_bytes(), &b[..], "witness element {i}");
        }
    }
}

/// Item W7-G1: the cached graph is released and rebuilt on demand; a rebuilt graph gives the same
/// witness, a holder keeps a released graph alive, and the idle hook frees nothing that is in use.
#[test]
fn graph_release_and_rebuild() {
    use super::super::metal_witness as mw;
    let _lock = gpu_test_lock();
    if component_dir().is_err() || mw::kind() == mw::Kind::Ref {
        eprintln!("skipped: no risc0-groth16 component, or the ref evaluator");
        return;
    }
    let seal = seal_words();
    mw::clear_reservations();
    mw::preload_for_wrap().unwrap();
    assert!(mw::is_cached());
    let first = mw::witness(seal).unwrap();
    let builds = mw::builds();

    // Held by a wrap: the idle hook leaves it (it may still free an idle `fast.rs` program of
    // another test); a release drops the cache, not the holder's copy.
    let held = mw::wrap_graph().unwrap();
    let d0 = mw::drops();
    mw::release_idle_now();
    assert!(mw::is_cached());
    assert_eq!(mw::drops(), d0);
    assert!(mw::release());
    assert!(!mw::is_cached());
    assert!(held.witness(seal, true).unwrap() == first);

    // Rebuilt on the next use, bit-equal.
    let t = Instant::now();
    let again = mw::witness(seal).unwrap();
    eprintln!(
        "W7-G1 rebuild + witness: {:.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    assert!(mw::builds() > builds);
    assert!(mw::is_cached());
    assert!(again == first, "the rebuilt graph's witness differs");

    // Fix round 1: a wrap still holding the old graph does not empty the cache of the new one
    // (another wrap's prestart built it while this wrap proved).
    let d0 = mw::drops();
    assert!(mw::release_slot(Some(&held), 0).is_none());
    assert!(mw::is_cached());
    drop(held);
    assert_eq!(
        mw::drops(),
        d0 + 1,
        "the old graph is dropped by its last holder"
    );

    // Idle: the hook's mechanism drops it (and says how much it freed).
    let d0 = mw::drops();
    let freed = mw::release_idle_now();
    assert_eq!(mw::drops(), d0 + 1);
    eprintln!("W7-G1 release_idle freed {:.1} MiB", mib(freed));
    assert!(!mw::is_cached());
    let d1 = mw::drops();
    mw::release_idle_now();
    assert_eq!(mw::drops(), d1);
    // The public hook (Metal enabled or not) never fails.
    let _ = crate::prove::release_graph();
}

/// Item W7-G1 (fix round 1, PLAN 8.7): the drop counter the release tests assert on counts only
/// graphs that were the process's cached graph, so tests that compile throw-away evaluators
/// (`witness_with`, in [witness_evaluators_bit_equal], which runs in parallel without the lock)
/// cannot move it.
#[test]
fn uncached_graphs_do_not_count_as_cached_drops() {
    use super::super::metal_witness as mw;
    let _lock = gpu_test_lock();
    if component_dir().is_err() {
        eprintln!("skipped: no risc0-groth16 component");
        return;
    }
    let d0 = mw::drops();
    let w = mw::witness_with(mw::Kind::Fast, 1, seal_words()).unwrap();
    assert!(!w.is_empty());
    assert_eq!(mw::drops(), d0, "a throw-away graph's drop was counted");
}

/// Item W7-G1 (fix round 1): the pressure-drain hook does not drop the graph when the drop cannot
/// lower the footprint (libmalloc's large cache on and no explicit `RISC0_G16_GRAPH_RELEASE=1`):
/// there a drop frees nothing and each rebuild leaves ~580 MiB more behind.
#[test]
fn release_hook_keeps_the_graph_with_the_large_cache_on() {
    use super::super::metal_witness as mw;
    let _lock = gpu_test_lock();
    if component_dir().is_err() || mw::kind() == mw::Kind::Ref {
        eprintln!("skipped: no risc0-groth16 component, or the ref evaluator");
        return;
    }
    mw::clear_reservations();
    mw::preload_for_wrap().unwrap();
    assert!(mw::is_cached());
    let d0 = mw::drops();
    let freed = crate::prove::release_graph();
    if mw::idle_release_allowed() {
        eprintln!("large cache off or release forced: the hook drops the idle graph");
        assert_eq!(mw::drops(), d0 + 1);
        assert!(!mw::is_cached());
    } else {
        assert_eq!(freed, 0);
        assert_eq!(
            mw::drops(),
            d0,
            "the hook dropped the graph with the large cache on"
        );
        assert!(mw::is_cached());
    }
}

/// Item W7-G1 fix round 2 (the round-2 review's finding): a pressure drain that fires while a wrap
/// runs (here from `pre_gpu`, where the zkvm releases the STARK pools and where W7-A1's drain
/// would fire) must not drop the wrap's graph: before the fix the hook freed ~950 MiB and the wrap
/// rebuilt the graph synchronously (0.7-0.9 s, under the GPU lock on the in-place path). Uses the
/// hook's mechanism ([release_idle_now]: the public hook is a no-op with the large cache on),
/// three cycles, every proof verified.
#[test]
fn drain_during_a_wrap_keeps_its_graph() {
    use super::super::metal_witness as mw;
    let _lock = gpu_test_lock();
    if !available() || mw::kind() == mw::Kind::Ref {
        return;
    }
    for i in 0..3 {
        mw::preload_for_wrap().unwrap();
        let b0 = mw::builds();
        let freed = std::cell::Cell::new(None);
        let dir = tempfile::tempdir().unwrap();
        let r = gpu_prove(
            seal_words(),
            &mut Some(|| freed.set(Some(mw::release_idle_now()))),
            dir.path(),
        );
        let seal = match r {
            Err(e) if refused_by_budget(&e) => return,
            r => r.unwrap(),
        };
        check_proof(&seal, dir.path());
        eprintln!(
            "W7G1DRAIN cycle {i}: the drain in pre_gpu freed {:.1} MiB; builds {} -> {}",
            mib(freed.get().unwrap_or(0)),
            b0,
            mw::builds()
        );
        assert_eq!(mw::wraps_in_progress(), 0);
        assert_eq!(freed.get(), Some(0), "the drain in pre_gpu freed memory");
        assert_eq!(
            mw::builds(),
            b0,
            "the drain dropped the wrap's graph: the wrap rebuilt it synchronously"
        );
    }
}

/// Item W7-G1 fix round 2: what keeps the graph from a drain or a wrap's after-witness release
/// besides a handle: a wrap in progress before its witness, another wrap still in progress (the
/// last one out releases), and a prestart reservation until a wrap takes it.
#[test]
fn wraps_in_progress_and_reservations_keep_the_graph() {
    use super::super::metal_witness as mw;
    let _lock = gpu_test_lock();
    if component_dir().is_err() || mw::kind() == mw::Kind::Ref {
        eprintln!("skipped: no risc0-groth16 component, or the ref evaluator");
        return;
    }
    assert_eq!(mw::wraps_in_progress(), 0);
    mw::clear_reservations();
    mw::preload_for_wrap().unwrap();
    let d0 = mw::drops();
    {
        // A wrap in its pre_gpu (no handle yet): the drain leaves the graph.
        let w1 = mw::WrapGuard::begin();
        assert_eq!(mw::release_idle_now(), 0);
        assert!(mw::is_cached());
        // A second wrap in progress: the first one's after-witness release (with or without its
        // handle) leaves the cache to the last wrap out.
        let w2 = mw::WrapGuard::begin();
        let held = mw::wrap_graph().unwrap();
        assert!(mw::release_slot(Some(&held), 1).is_none());
        assert!(mw::release_slot(None, 1).is_none());
        assert!(mw::is_cached());
        drop((held, w2, w1));
    }
    assert_eq!(mw::wraps_in_progress(), 0);
    // The next job's prestart reserved the graph: neither a drain nor a failed wrap's release
    // (no handle) drops it, until a wrap takes the reservation and finishes.
    mw::reserve_for_wrap();
    assert_eq!(mw::release_idle_now(), 0);
    assert!(mw::release_slot(None, 0).is_none());
    assert!(mw::is_cached());
    {
        let _w = mw::WrapGuard::begin();
        assert_eq!(mw::release_idle_now(), 0);
        assert!(mw::is_cached());
    }
    assert_eq!(
        mw::drops(),
        d0,
        "the graph was dropped while in use or reserved"
    );
    // Taken and finished: idle, the drain's mechanism drops it.
    mw::release_idle_now();
    assert!(!mw::is_cached());
    assert_eq!(mw::drops(), d0 + 1);
}

/// Item W7-G1 fix round 1 (the review's hook finding, kept as a check): five cycles of a wrap's
/// prestart build, a verified wrap and the pressure-drain hook. The footprint before each hook
/// call must not climb cycle over cycle (before the fix, with the large cache on, it rose ~580
/// MiB per cycle: 4044 -> 6365 MiB). Heavy: `cargo test ... w7g1_hook_cycle -- --ignored
/// --nocapture --test-threads 1` under gpu-run.
#[test]
#[ignore]
fn w7g1_hook_cycle() {
    use risc0_groth16_sys::metal::filemap::process_memory;
    let _lock = gpu_test_lock();
    if !available() {
        return;
    }
    let mut before = Vec::new();
    for i in 0..5 {
        super::super::metal_witness::preload_for_wrap().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let seal = match gpu_prove(seal_words(), &mut Some(|| {}), dir.path()) {
            Err(e) if refused_by_budget(&e) => return,
            r => r.unwrap(),
        };
        check_proof(&seal, dir.path());
        let f = process_memory().phys_footprint;
        let freed = crate::prove::release_graph();
        eprintln!(
            "W7G1HOOK cycle {i}: footprint {:.1} MiB before the hook, freed {:.1} MiB",
            mib(f),
            mib(freed)
        );
        before.push(f);
    }
    // Cycles 1..4 against cycle 1 (cycle 0 is before the key's pools settle): within 256 MiB.
    let growth = before[4].saturating_sub(before[1]);
    assert!(
        growth < 256 << 20,
        "footprint grew {:.1} MiB over 3 hook cycles",
        mib(growth)
    );
}

/// Item W7-G1 measurement helper (not a check): the graph's share of the process footprint and
/// its build times (first build, rebuild with the digest known), then the footprint after the
/// release. `cargo test ... w7g1_graph_share -- --ignored --nocapture --test-threads 1`.
#[test]
#[ignore]
fn w7g1_graph_share() {
    use super::super::metal_witness as mw;
    use risc0_groth16_sys::metal::filemap::process_memory;
    let _lock = gpu_test_lock();
    if component_dir().is_err() {
        return;
    }
    let fp = || process_memory().phys_footprint;
    let rounds: usize = std::env::var("W7G1_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let pause: u64 = std::env::var("W7G1_PAUSE_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let start = fp();
    for round in 0..rounds {
        std::thread::sleep(Duration::from_millis(pause));
        let f0 = fp();
        eprintln!(
            "W7G1SHARE round {round}: footprint {:.1} MiB at the start (+{:.1} over the first)",
            mib(f0),
            mib(f0.saturating_sub(start))
        );
        let t = Instant::now();
        mw::preload_for_wrap().unwrap();
        let build_ms = t.elapsed().as_secs_f64() * 1e3;
        let f1 = fp();
        let t = Instant::now();
        let w = mw::witness(seal_words()).unwrap();
        let wit_ms = t.elapsed().as_secs_f64() * 1e3;
        drop(w);
        let f2 = fp();
        let t = Instant::now();
        let d0 = mw::drops();
        mw::release();
        let rel_ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "W7G1SHARE round {round}: graphs dropped by the release: {}",
            mw::drops() - d0
        );
        let f3 = fp();
        eprintln!(
            "W7G1SHARE round {round} kind {:?}: build {build_ms:.1} ms, footprint +{:.1} MiB; \
             witness {wit_ms:.1} ms, after it +{:.1} MiB; release {rel_ms:.1} ms, after it \
             {:+.1} MiB vs before the build",
            mw::kind(),
            mib(f1.saturating_sub(f0)),
            mib(f2.saturating_sub(f0)),
            (f3 as f64 - f0 as f64) / (1u64 << 20) as f64
        );
    }
}

/// Item G13: the compiled evaluators (`par` with several thread counts, `fast`) give the same
/// witness as the reference generator, element by element, for the fixture and a perturbed seal.
#[test]
fn witness_evaluators_bit_equal() {
    use super::super::metal_witness::{witness_with, Kind};
    if component_dir().is_err() {
        eprintln!("skipped: no risc0-groth16 component");
        return;
    }
    let perturbed = perturbed_seal();
    for seal in [seal_words(), bytemuck::cast_slice(&perturbed)] {
        let reference = reference_witness(seal);
        for (kind, threads) in [
            (Kind::Fast, 1),
            (Kind::Par, 1),
            (Kind::Par, 4),
            (Kind::Par, 18),
            (Kind::ParTransient, 4),
        ] {
            let t = Instant::now();
            let ours = witness_with(kind, threads, seal).unwrap();
            eprintln!("{kind:?} x{threads}: compile + eval {:?}", t.elapsed());
            assert_eq!(ours.len(), reference.len());
            let bytes: &[u8] = bytemuck::cast_slice(&ours);
            for (i, b) in reference.iter().enumerate() {
                assert_eq!(
                    &bytes[32 * i..32 * i + 32],
                    &b[..],
                    "{kind:?} x{threads}: element {i}"
                );
            }
        }
    }
}

/// Item G15 measurement helper (not a check): footprint and evaluation times of one evaluator
/// (`G15_WITNESS_KIND=par|fast`, default `par_transient`). `cargo test ... g15_par_footprint -- --ignored --nocapture
/// --test-threads 1`.
#[test]
#[ignore]
fn g15_par_footprint() {
    if component_dir().is_err() {
        return;
    }
    use super::super::metal_witness::{footprint_probe, Kind};
    let kind = match std::env::var("G15_WITNESS_KIND").as_deref() {
        Ok("par") => Kind::Par,
        Ok("fast") => Kind::Fast,
        _ => Kind::ParTransient,
    };
    footprint_probe(kind, seal_words(), 5).unwrap();
}

/// G13 measurement helper (not a check): the time to build the witness graph in a fresh
/// process. `cargo test ... graph_ready_time -- --ignored --nocapture`.
#[test]
#[ignore]
fn graph_ready_time() {
    let t = Instant::now();
    super::super::fast::preload().unwrap();
    eprintln!(
        "G13 graph ready: {:.1} ms",
        t.elapsed().as_secs_f64() * 1000.0
    );
}
