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

/// The tests that build a [MetalProver] (5.6 GB each with the rzup key) run alone (PLAN 8.7):
/// two builds never coexist, and no kernel test is refused under a budget because of one.
fn prover_test_lock() -> std::sync::RwLockWriteGuard<'static, ()> {
    super::super::test_lock::exclusive()
}

/// [MetalProver::new] for `dir`, or `None` (with the clean budget error checked) when a
/// `RISC0_METAL_MAX_WORKING_SET_BYTES` budget refuses it.
fn build_or_refused(dir: &std::path::Path) -> Option<MetalProver> {
    let setup = SetupParams::new(dir).unwrap();
    match MetalProver::new(&setup) {
        Ok(p) => Some(p),
        Err(e) if super::super::working_set_budget().is_some() => {
            let msg = format!("{e:#}");
            assert!(msg.contains("working-set budget exceeded"), "{msg}");
            eprintln!("prover build refused under the budget: {msg}");
            None
        }
        Err(e) => panic!("{e:#}"),
    }
}

fn no_pre_blinding() -> PreBlinding {
    PreBlinding {
        a: G1Affine::zero(),
        b1: G1Affine::zero(),
        b2: G2Affine::zero(),
        c: G1Affine::zero(),
        h: G1Affine::zero(),
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
    let _lock = prover_test_lock();
    let Some(dir) = component_dir() else {
        return;
    };
    // Budget runs: the error must be the clean budget error, before loading.
    let Some(mut prover) = build_or_refused(&dir) else {
        return;
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
        pre_blinding: no_pre_blinding(),
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
        pre_blinding: no_pre_blinding(),
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

/// The pre-blinding MSM results (G8, dumped under `RISC0_VERBOSE`) depend only on the witness:
/// two proofs of one witness with different blinding scalars give the same values and different
/// proofs, and with r = s = 0 the proof is the pre-blinding values plus the verifying key's
/// alpha/beta terms. The witness is arbitrary canonical data (the MSMs and H do not need a
/// satisfying assignment).
#[test]
fn pre_blinding_is_independent_of_blinding() {
    use ark_ec::CurveGroup as _;
    use rand::{Rng as _, SeedableRng as _};

    let _lock = prover_test_lock();
    let Some(dir) = component_dir() else {
        return;
    };
    let Some(mut prover) = build_or_refused(&dir) else {
        return;
    };
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(8);
    let mut witness = vec![0u8; prover.witness_len() * 32];
    for e in witness.chunks_exact_mut(32) {
        rng.fill(&mut e[..31]);
    }
    let opts = ProveOptions::default();
    let p1 = prover
        .prove_with_blinding(&witness, &opts, Fr::from(3u64), Fr::from(5u64))
        .unwrap();
    let p2 = prover
        .prove_with_blinding(&witness, &opts, Fr::from(7u64), Fr::from(11u64))
        .unwrap();
    assert_eq!(p1.pre_blinding, p2.pre_blinding);
    assert!(p1.a != p2.a && p1.b != p2.b && p1.c != p2.c);
    let p0 = prover
        .prove_with_blinding(&witness, &opts, Fr::zero(), Fr::zero())
        .unwrap();
    assert_eq!(p0.pre_blinding, p1.pre_blinding);
    let vk = prover.verifying_key();
    let pre = &p0.pre_blinding;
    assert_eq!(p0.a, (pre.a + vk.alpha_g1).into_affine());
    assert_eq!(p0.b, (pre.b2 + vk.beta_g2).into_affine());
    assert_eq!(p0.c, (pre.c + pre.h).into_affine());
    let dump = pre.dump();
    assert_eq!(dump.lines().count(), 5, "{dump}");
    assert!(dump.starts_with("A'=(0x"), "{dump}");
    assert!(dump.contains("\nB2'=((0x"), "{dump}");
}

/// A stage that panics (a programming-error `assert!` in an encoder) unwinds cleanly: the open
/// encoder is ended and nothing is committed (G8; before, dropping the unended compute encoder
/// made Metal abort the process), and the queue keeps working.
#[test]
fn run_survives_a_panicking_stage() {
    let Ok(k) = kernels() else {
        return;
    };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut run = Run::new(&k.queue, false);
        run.stage("first", 0, |_| {});
        run.stage("second", 1, |_| panic!("an encoder assert"));
    }));
    assert!(r.is_err());
    run_one(&k.queue, |_| {}).unwrap();
}

/// G9: every optimization switch is bit-exact. One witness proved with every switch off (the
/// G7/G8 path), each single switch off, and all on: the pre-blinding MSM results (A', B1', B2',
/// C', H) are equal. Also the switch parser.
#[test]
fn g9_tuning_bit_exact() {
    use rand::{Rng as _, SeedableRng as _};

    assert_eq!(Tuning::parse("base"), Some(Tuning::none()));
    assert_eq!(Tuning::parse("all"), Some(Tuning::all()));
    assert_eq!(Tuning::parse(""), Some(Tuning::all()));
    assert_eq!(Tuning::parse("bogus"), None);
    for name in Tuning::NAMES {
        let mut t = Tuning::all();
        assert!(t.set(name, false));
        assert_ne!(t, Tuning::all(), "{name}");
        assert_eq!(Tuning::parse(&format!("-{name}")), Some(t));
        assert_eq!(
            Tuning::parse(&format!("base,+{name},-{name}")),
            Some(Tuning::none())
        );
    }

    let _lock = prover_test_lock();
    let Some(dir) = component_dir() else {
        return;
    };
    let Some(mut prover) = build_or_refused(&dir) else {
        return;
    };
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x69);
    let mut witness = vec![0u8; prover.witness_len() * 32];
    for e in witness.chunks_exact_mut(32) {
        rng.fill(&mut e[..31]);
    }
    let prove = |prover: &mut MetalProver, tune: Tuning| {
        let opts = ProveOptions {
            tune,
            ..Default::default()
        };
        prover
            .prove_with_blinding(&witness, &opts, Fr::from(3u64), Fr::from(5u64))
            .unwrap()
    };
    let base = prove(&mut prover, Tuning::none());
    let mut variants = vec![("all".to_string(), Tuning::all())];
    for name in Tuning::NAMES {
        variants.push((
            format!("-{name}"),
            Tuning::parse(&format!("-{name}")).unwrap(),
        ));
    }
    for (name, tune) in variants {
        let p = prove(&mut prover, tune);
        assert_eq!(p.pre_blinding, base.pre_blinding, "{name}");
        assert_eq!((p.a, p.b, p.c), (base.a, base.b, base.c), "{name}");
        eprintln!("{name}: gpu {:.1} ms", p.stats.gpu_total_ms());
    }
}

/// G9: interleaved A/B of [Tuning] variants on the real key (a benchmark, ignored).
/// `G9_VARIANTS` = variants separated by `;` ([Tuning::parse]; default "base;all"), `G9_REPS`
/// rounds (default 7) of A B ... in turn, `G9_PROFILE=1` one command buffer per stage. Every
/// proof's pre-blinding values must equal the first variant's (bit-exact MSMs and H).
#[test]
#[ignore]
fn g9_bench() {
    use rand::{Rng as _, SeedableRng as _};

    let _lock = prover_test_lock();
    let Some(dir) = component_dir() else {
        return;
    };
    let Some(mut prover) = build_or_refused(&dir) else {
        return;
    };
    let specs = std::env::var("G9_VARIANTS").unwrap_or_else(|_| "base;all".into());
    let variants: Vec<(String, Tuning)> = specs
        .split(';')
        .map(|s| {
            (
                s.to_string(),
                Tuning::parse(s).expect("bad G9_VARIANTS item"),
            )
        })
        .collect();
    let reps: usize = std::env::var("G9_REPS").map_or(7, |v| v.parse().unwrap());
    let profile = std::env::var("G9_PROFILE").as_deref() == Ok("1");
    let c = &prover.coeffs;
    eprintln!(
        "G9 coefficients: A {} in {} rows, B {} in {} rows; witness {}, domain {}",
        c.count_a,
        c.ind_a - 1,
        c.count_b,
        c.ind_b - 1,
        prover.witness_len(),
        prover.domain_size()
    );
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(9);
    let mut witness = vec![0u8; prover.witness_len() * 32];
    for e in witness.chunks_exact_mut(32) {
        rng.fill(&mut e[..31]);
    }
    let opts = |tune| ProveOptions {
        profile,
        verbose: false,
        tune,
    };
    // Warm-up (and the reference pre-blinding values) with the first variant.
    let reference = prover
        .prove_with_blinding(
            &witness,
            &opts(variants[0].1),
            Fr::from(3u64),
            Fr::from(5u64),
        )
        .unwrap()
        .pre_blinding;
    for (_, t) in &variants[1..] {
        let p = prover
            .prove_with_blinding(&witness, &opts(*t), Fr::from(3u64), Fr::from(5u64))
            .unwrap();
        if std::env::var_os("G9_NOCHECK").is_none() {
            assert_eq!(p.pre_blinding, reference);
        }
    }
    let mut gpu: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
    let mut total: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
    let mut stages: Vec<std::collections::BTreeMap<String, Vec<f64>>> =
        vec![Default::default(); variants.len()];
    for rep in 0..reps {
        for (vi, (name, t)) in variants.iter().enumerate() {
            let p = prover
                .prove_with_blinding(&witness, &opts(*t), Fr::from(3u64), Fr::from(5u64))
                .unwrap();
            if std::env::var_os("G9_NOCHECK").is_none() {
                assert_eq!(p.pre_blinding, reference, "variant {name} rep {rep}");
            }
            let st = &p.stats;
            gpu[vi].push(st.gpu_total_ms());
            total[vi].push(st.total_ms);
            for (k, v) in &st.gpu_ms {
                stages[vi].entry(k.clone()).or_default().push(*v);
            }
            eprintln!(
                "G9BENCH rep {rep} {name}: gpu {:.1} total {:.1} ms [{}]",
                st.gpu_total_ms(),
                st.total_ms,
                st.gpu_ms
                    .iter()
                    .map(|(k, v)| format!("{k}={v:.1}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    fn stat(v: &[f64]) -> String {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        format!("{:.1} [{:.1}..{:.1}]", s[s.len() / 2], s[0], s[s.len() - 1])
    }
    for (vi, (name, _)) in variants.iter().enumerate() {
        eprintln!(
            "G9SUMMARY {name}: gpu {} total {} ms",
            stat(&gpu[vi]),
            stat(&total[vi])
        );
        for (k, v) in &stages[vi] {
            eprintln!("G9SUMMARY {name}   {k}: {}", stat(v));
        }
    }
}
