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

//! Tests of the prover orchestration (`prover.rs`). The end-to-end proofs (witness from an
//! identity_p254 seal, verified with `risc0_groth16::Verifier`) are in risc0-groth16
//! (`src/prove/metal_tests.rs`), which this crate cannot depend on.

use std::path::PathBuf;

use ark_bn254::{Fq, Fq2, G1Affine, G2Affine};
use ark_ec::AffineRepr;
use ark_ff::Zero;

use super::*;

/// The rzup component directory (`RISC0_GROTH16_DIR`, else the default rzup layout), if its
/// files are present.
fn component_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("RISC0_GROTH16_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join(".risc0/extensions/v0.1.0-risc0-groth16"))
        })?;
    if dir.join("stark_verify_final.zkey").is_file() {
        Some(dir)
    } else {
        eprintln!(
            "skipped: no risc0-groth16 component at {} (rzup install risc0-groth16)",
            dir.display()
        );
        None
    }
}

fn mib(b: u64) -> f64 {
    b as f64 / (1u64 << 20) as f64
}

/// `MSM(P, f) + fuzzed == O` for A, B1, B2 and C on the real proving key: the GPU fuzz (G4), the
/// MSM (G6) and the key/fuzzed-results loaders (G3) agree with the files the CUDA setup wrote.
/// Also checks a wrong-length witness is rejected and the build/release cycle.
#[test]
fn fuzz_msm_cancels_fuzzed_results() {
    let Some(dir) = component_dir() else {
        return;
    };
    let setup = SetupParams::new(&dir).unwrap();
    let mut prover = match MetalProver::new(&setup) {
        Ok(p) => p,
        Err(e) if super::super::working_set_budget().is_some() => {
            // Budget runs: the error must be the clean budget error, before loading.
            let msg = format!("{e:#}");
            assert!(msg.contains("working-set budget exceeded"), "{msg}");
            eprintln!("prover build refused under the budget: {msg}");
            return;
        }
        Err(e) => panic!("{e:#}"),
    };
    let st = &prover.build_stats;
    eprintln!(
        "build: kernels {:?} srs {:?} coeffs {:?} validate {:?} gpu setup {:?} total {:?}; \
         planned {:.1} MiB, allocated {:.1} -> {:.1} MiB",
        st.kernels,
        st.srs_load,
        st.coeffs_load,
        st.coeffs_validate,
        st.gpu_setup,
        st.total,
        mib(st.planned_bytes),
        mib(st.allocated_before),
        mib(st.allocated_after)
    );
    // The plan is an upper bound of what was allocated (other tests may allocate concurrently,
    // so only check it when nothing else shrank the figure).
    assert!(st.allocated_after >= st.allocated_before);

    let t = Instant::now();
    let (a, b1, c, b2) = prover.fuzz_check().unwrap();
    eprintln!("fuzz check: {:?}", t.elapsed());
    assert!(a.is_zero(), "MSM(A, f) + fuzzed.a != O");
    assert!(b1.is_zero(), "MSM(B1, f) + fuzzed.b_g1 != O");
    assert!(c.is_zero(), "MSM(C, f) + fuzzed.c != O");
    assert!(b2.is_zero(), "MSM(B2, f) + fuzzed.b_g2 != O");

    let err = prover
        .prove(&[0u8; 64], &ProveOptions::default())
        .unwrap_err();
    assert!(format!("{err}").contains("witness of 64 bytes"), "{err:#}");
}

/// The JSON files have the CUDA/Intel shape: decimal coordinates, pi_b pairs (c0, c1), "1"
/// projective coordinates, public signals as a decimal array.
#[test]
fn proof_json_format() {
    let g1 = G1Affine::generator();
    let g2 = G2Affine::generator();
    let proof = Proof {
        a: g1,
        b: g2,
        c: (g1 + g1).into(),
        public: vec![Fr::from(0u64), Fr::from(33u64), -Fr::from(1u64)],
        stats: ProveStats::default(),
    };
    let j = proof.proof_json();
    assert!(j.starts_with("{ \"pi_a\":[\"1\",\"2\",\"1\"]"), "{j}");
    let b0 = format!("[\"{}\",\"{}\"]", g2.x.c0, g2.x.c1);
    assert!(j.contains(&b0), "{j}");
    assert!(j.contains("[\"1\",\"0\"]]"), "{j}");
    assert!(j.contains("\"protocol\":\"groth16\""), "{j}");
    let p = proof.public_json();
    assert_eq!(
        p,
        "[\"0\",\"33\",\"21888242871839275222246405745257275088548364400416034343698204186575808495616\"]"
    );
    // Infinity prints as (0, 0), as the Intel prover's to_affine of the identity.
    let inf = Proof {
        a: G1Affine::zero(),
        b: G2Affine::new_unchecked(Fq2::zero(), Fq2::zero()),
        c: G1Affine::new_unchecked(Fq::zero(), Fq::zero()),
        public: vec![],
        stats: ProveStats::default(),
    };
    assert!(inf
        .proof_json()
        .starts_with("{ \"pi_a\":[\"0\",\"0\",\"1\"]"));
    assert_eq!(inf.public_json(), "[]");
}

#[test]
fn canonical_and_random_scalars() {
    let mut b = [0u8; 32];
    b[0] = 5;
    assert_eq!(fr_from_canonical(&b), Some(Fr::from(5u64)));
    // r itself is not canonical.
    let r = <Fr as ark_ff::PrimeField>::MODULUS;
    let mut rb = [0u8; 32];
    for (i, l) in r.0.iter().enumerate() {
        rb[8 * i..8 * i + 8].copy_from_slice(&l.to_le_bytes());
    }
    assert_eq!(fr_from_canonical(&rb), None);
    let (x, y) = (random_fr().unwrap(), random_fr().unwrap());
    assert!(!x.is_zero() && x != y);
}
