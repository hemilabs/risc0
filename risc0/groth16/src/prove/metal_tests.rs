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

use std::{path::Path, sync::OnceLock, time::Instant};

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
