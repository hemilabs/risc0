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

/// The rzup component directory (v0.2.0, see [srs::tests::component_dir]), if present.
fn component_dir() -> Option<PathBuf> {
    super::super::srs::tests::component_dir()
}

/// The tests that build a [MetalProver] (3.9-5.9 GiB each with the v0.2.0 key) run alone (PLAN 8.7):
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
/// rounds (default 7) of A B ... in turn, `G9_PROFILE=1` one command buffer per stage,
/// `G9_WITNESS=<file>` a real witness instead of random values. Every
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
    let (ca, cb, ua, ub) = prover.coeffs().counts();
    eprintln!(
        "G9 coefficients: A {} in {} rows, B {} in {} rows; witness {}, domain {}",
        ca,
        ua - 1,
        cb,
        ub - 1,
        prover.witness_len(),
        prover.domain_size()
    );
    // `G9_WITNESS=<file>`: a real witness (witness_len canonical 32-byte elements, e.g. dumped
    // from a wrap) instead of random 248-bit values. Item G14: the real witness is mostly 0 and
    // 1, which is what `nofuzz` exploits.
    let witness = match std::env::var("G9_WITNESS") {
        Ok(path) => {
            let w = std::fs::read(&path).expect("G9_WITNESS");
            assert_eq!(w.len(), prover.witness_len() * 32, "G9_WITNESS size");
            w
        }
        Err(_) => {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(9);
            let mut witness = vec![0u8; prover.witness_len() * 32];
            for e in witness.chunks_exact_mut(32) {
                rng.fill(&mut e[..31]);
            }
            witness
        }
    };
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

fn random_witness(prover: &MetalProver, seed: u64) -> Vec<u8> {
    use rand::{Rng as _, SeedableRng as _};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut witness = vec![0u8; prover.witness_len() * 32];
    for e in witness.chunks_exact_mut(32) {
        rng.fill(&mut e[..31]);
    }
    witness
}

fn report_build(name: &str, p: &MetalProver) {
    let st = &p.build_stats;
    eprintln!(
        "G10BUILD {name}: srs {:?} coeffs {:?} validate {:?} gpu setup {:?} total {:?}; split_h \
         {}; tw_lent {}; planned {:.1} MiB, allocated {:.1} -> {:.1} MiB (+{:.1}), footprint {:.1} -> {:.1} MiB (+{:.1}), \
         coefficients {:.1} MiB",
        st.srs_load,
        st.coeffs_load,
        st.coeffs_validate,
        st.gpu_setup,
        st.total,
        st.split_h,
        st.tw_lent,
        mib(st.planned_bytes),
        mib(st.allocated_before),
        mib(st.allocated_after),
        mib(st.allocated_after.saturating_sub(st.allocated_before)),
        mib(st.footprint_before),
        mib(st.footprint_after),
        mib(st.footprint_after.saturating_sub(st.footprint_before)),
        mib(p.coeffs().resident_bytes()),
    );
}

/// Item G10: every memory switch (built one prover at a time) gives the same pre-blinding values
/// and, with fixed r and s, the same proof as the G9 layout; the alias layout's fuzz check holds.
#[test]
fn g10_mem_options_bit_exact() {
    assert_eq!(MemOptions::parse("base"), Some(MemOptions::none()));
    assert_eq!(MemOptions::parse("all"), Some(MemOptions::all()));
    assert_eq!(MemOptions::parse("-bogus"), None);
    for name in MemOptions::NAMES {
        let mut t = MemOptions::all();
        assert!(t.set(name, false));
        assert_eq!(MemOptions::parse(&format!("-{name}")), Some(t));
    }

    let _lock = prover_test_lock();
    let Some(dir) = component_dir() else {
        return;
    };
    let setup = SetupParams::new(&dir).unwrap();
    let mut variants = vec![("base".to_string(), MemOptions::none())];
    variants.push(("all".into(), MemOptions::all()));
    for name in MemOptions::NAMES {
        variants.push((
            format!("-{name}"),
            MemOptions::parse(&format!("-{name}")).unwrap(),
        ));
        variants.push((
            format!("base,+{name}"),
            MemOptions::parse(&format!("base,+{name}")).unwrap(),
        ));
    }
    let mut reference: Option<Proof> = None;
    let mut witness = None;
    for (name, mem) in variants {
        let mut prover = match MetalProver::new_with(&setup, mem) {
            Ok(p) => p,
            Err(e) if super::super::working_set_budget().is_some() => {
                let msg = format!("{e:#}");
                assert!(msg.contains("working-set budget exceeded"), "{msg}");
                eprintln!("G10 {name}: refused under the budget: {msg}");
                continue;
            }
            Err(e) => panic!("{name}: {e:#}"),
        };
        report_build(&name, &prover);
        assert_eq!(prover.mem_options(), mem);
        let w = witness.get_or_insert_with(|| random_witness(&prover, 0x10));
        let p = prover
            .prove_with_blinding(w, &ProveOptions::default(), Fr::from(3u64), Fr::from(5u64))
            .unwrap();
        if name == "all" {
            let (a, b1, c, b2) = prover.fuzz_check().unwrap();
            assert!(
                a.is_zero() && b1.is_zero() && c.is_zero() && b2.is_zero(),
                "fuzz check"
            );
            // And a proof after the fuzz check (which regenerated the fuzz into a, b) is unchanged.
            let p2 = prover
                .prove_with_blinding(w, &ProveOptions::default(), Fr::from(3u64), Fr::from(5u64))
                .unwrap();
            assert_eq!(p2.pre_blinding, p.pre_blinding);
        }
        eprintln!(
            "G10 {name}: gpu {:.1} ms, allocated after proof {:.1} MiB",
            p.stats.gpu_total_ms(),
            mib(p.stats.allocated)
        );
        match &reference {
            None => reference = Some(p),
            Some(r) => {
                assert_eq!(p.pre_blinding, r.pre_blinding, "{name}");
                assert_eq!((p.a, p.b, p.c), (r.a, r.b, r.c), "{name}");
            }
        }
    }
}

/// A witness with the shape of the real shrink_wrap witness (item G14): 44 % zeros, 29 % ones,
/// some r - 1 (Booth: the same digit in every window), small values and a few repeated
/// full-width values (heavy buckets in every window), the rest random. Canonical Fr.
fn witness_like(prover: &MetalProver, seed: u64) -> Vec<u8> {
    use ark_ff::{BigInteger as _, UniformRand as _};
    use rand::{Rng as _, SeedableRng as _};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let heavy: Vec<Fr> = (0..8).map(|_| Fr::rand(&mut rng)).collect();
    let mut witness = vec![0u8; prover.witness_len() * 32];
    for e in witness.chunks_exact_mut(32) {
        let v = match rng.gen_range(0..100) {
            0..=43 => Fr::zero(),
            44..=72 => Fr::from(1u64),
            73..=75 => -Fr::from(1u64),
            76..=79 => Fr::from(rng.gen_range(2..1u64 << 40)),
            80..=81 => heavy[rng.gen_range(0..heavy.len())],
            _ => Fr::rand(&mut rng),
        };
        e.copy_from_slice(&v.into_bigint().to_bytes_le());
    }
    witness
}

/// G14: `nofuzz` (the witness MSMs without the ChaCha fuzz and its correction) gives the same
/// pre-blinding values and proof as the fuzzed path, for a witness-like and a random witness,
/// in the G10 layout (fuzz regenerated per proof into a, b) and, when the budget allows, the G9
/// layout (persistent fuzz buffer). `fuzz_check` still holds after a `nofuzz` proof, and a
/// fuzzed proof after it is unchanged.
#[test]
fn g14_nofuzz_bit_exact() {
    let fuzzed = Tuning {
        nofuzz: false,
        ..Tuning::all()
    };
    assert_eq!(Tuning::parse("-nofuzz"), Some(fuzzed));
    assert!(Tuning::all().nofuzz && !Tuning::none().nofuzz);

    let _lock = prover_test_lock();
    let Some(dir) = component_dir() else {
        return;
    };
    let setup = SetupParams::new(&dir).unwrap();
    for mem in [MemOptions::all(), MemOptions::none()] {
        let mut prover = match MetalProver::new_with(&setup, mem) {
            Ok(p) => p,
            Err(e) if super::super::working_set_budget().is_some() => {
                let msg = format!("{e:#}");
                assert!(msg.contains("working-set budget exceeded"), "{msg}");
                eprintln!("G14 {mem:?}: refused under the budget: {msg}");
                continue;
            }
            Err(e) => panic!("{mem:?}: {e:#}"),
        };
        for (what, w) in [
            ("witness-like", witness_like(&prover, 0x14)),
            ("random", random_witness(&prover, 0x15)),
        ] {
            let prove = |prover: &mut MetalProver, tune: Tuning| {
                let opts = ProveOptions {
                    tune,
                    ..Default::default()
                };
                prover
                    .prove_with_blinding(&w, &opts, Fr::from(3u64), Fr::from(5u64))
                    .unwrap()
            };
            let f = prove(&mut prover, fuzzed);
            let nf = prove(&mut prover, Tuning::all());
            assert_eq!(nf.pre_blinding, f.pre_blinding, "{what} {mem:?}");
            assert_eq!((nf.a, nf.b, nf.c), (f.a, f.b, f.c), "{what} {mem:?}");
            assert!(!nf.stats.gpu_ms.iter().any(|(k, _)| k == "fuzz"));
            let (a, b1, c, b2) = prover.fuzz_check().unwrap();
            assert!(
                a.is_zero() && b1.is_zero() && c.is_zero() && b2.is_zero(),
                "fuzz check {mem:?}"
            );
            let f2 = prove(&mut prover, fuzzed);
            assert_eq!(
                f2.pre_blinding, f.pre_blinding,
                "{what} {mem:?} after fuzz_check"
            );
            eprintln!(
                "G14 {what} {mem:?}: gpu fuzzed {:.1} ms, nofuzz {:.1} ms",
                f.stats.gpu_total_ms(),
                nf.stats.gpu_total_ms()
            );
        }
    }
}

/// Item P31-12: `split_h` (MSM H over the domain's two halves, a smaller MSM scratch) gives the
/// same pre-blinding values and, with fixed r and s, the same proof as one MSM over the domain,
/// for a witness-like and a random witness; the fuzz check holds on the split prover; and
/// [SplitH::Auto] splits exactly when the unsplit footprint exceeds the room.
#[test]
fn p31_12_split_h_bit_exact() {
    assert_eq!(MemOptions::all().split_h, SplitH::Auto);
    assert_eq!(MemOptions::none().split_h, SplitH::Off);
    assert_eq!(MemOptions::parse("split_h=on").unwrap().split_h, SplitH::On);
    assert_eq!(
        MemOptions::parse("base,split_h=auto").unwrap().split_h,
        SplitH::Auto
    );
    assert_eq!(MemOptions::parse("-split_h").unwrap().split_h, SplitH::Off);
    assert_eq!(MemOptions::parse("split_h=maybe"), None);

    let _lock = prover_test_lock();
    let Some(dir) = component_dir() else {
        return;
    };
    let setup = SetupParams::new(&dir).unwrap();
    // P31-12's rules, without item B9-4G's lent twiddles (`b9_4g_tw_lent_bit_exact`).
    let lay = |split_h: SplitH, room: u64| {
        let mem = MemOptions {
            split_h,
            tw_lent: TwLent::Off,
            ..MemOptions::all()
        };
        layout(
            setup.srs_path.as_path(),
            setup.pcoeffs_path.as_path(),
            mem,
            room,
        )
        .unwrap()
    };
    let whole = lay(SplitH::Off, 0);
    let split = lay(SplitH::On, u64::MAX);
    assert!(!whole.split_h && split.split_h);
    assert!(split.planned_bytes < whole.planned_bytes);
    assert_eq!(split.polys, whole.polys);
    assert!(!lay(SplitH::Auto, whole.planned_bytes).split_h);
    assert!(lay(SplitH::Auto, whole.planned_bytes - 1).split_h);
    eprintln!(
        "P31-12 layout: wn {} n {}; whole cap {} planned {:.1} MiB (c in scratch {}, pieces lent \
         {}); split cap {} planned {:.1} MiB (c in scratch {}, pieces lent {})",
        whole.wn,
        whole.n,
        whole.cap,
        mib(whole.planned_bytes),
        whole.c_in_scratch,
        whole.pieces_lent,
        split.cap,
        mib(split.planned_bytes),
        split.c_in_scratch,
        split.pieces_lent
    );

    let mut reference: Vec<Proof> = vec![];
    for split_h in [SplitH::Off, SplitH::On] {
        let mem = MemOptions {
            split_h,
            tw_lent: TwLent::Off,
            ..MemOptions::all()
        };
        let mut prover = match MetalProver::new_with(&setup, mem) {
            Ok(p) => p,
            Err(e) if super::super::working_set_budget().is_some() => {
                let msg = format!("{e:#}");
                assert!(msg.contains("working-set budget exceeded"), "{msg}");
                eprintln!("P31-12 {split_h:?}: refused under the budget: {msg}");
                continue;
            }
            Err(e) => panic!("{split_h:?}: {e:#}"),
        };
        report_build(&format!("split_h={split_h:?}"), &prover);
        assert_eq!(prover.split_h(), split_h == SplitH::On);
        assert_eq!(prover.build_stats.split_h, prover.split_h());
        let mut proofs = vec![];
        for (what, w) in [
            ("witness-like", witness_like(&prover, 0x3112)),
            ("random", random_witness(&prover, 0x3113)),
        ] {
            let p = prover
                .prove_with_blinding(&w, &ProveOptions::default(), Fr::from(3u64), Fr::from(5u64))
                .unwrap();
            assert_eq!(
                p.stats.gpu_ms.iter().any(|(k, _)| k.contains("msm_h2")),
                prover.split_h(),
                "{what}"
            );
            eprintln!(
                "P31-12 {split_h:?} {what}: gpu {:.1} ms",
                p.stats.gpu_total_ms()
            );
            proofs.push(p);
        }
        if prover.split_h() {
            let (a, b1, c, b2) = prover.fuzz_check().unwrap();
            assert!(
                a.is_zero() && b1.is_zero() && c.is_zero() && b2.is_zero(),
                "fuzz check (split)"
            );
        }
        if reference.is_empty() {
            reference = proofs;
        } else {
            for (p, r) in proofs.iter().zip(&reference) {
                assert_eq!(p.pre_blinding, r.pre_blinding);
                assert_eq!((p.a, p.b, p.c), (r.a, r.b, r.c));
            }
        }
    }
}

/// Item B9-4G: the lent NTT twiddles ([MemOptions::tw_lent]).
/// - Layout choice: `Auto` keeps P31-12's layouts where they leave [AUTO_HEADROOM_BYTES]
///   (whole at the 8 GiB and default budgets, the split layout with room to spare) and takes
///   split + lent twiddles at a 4 GiB budget beside 62 MiB of pipeline objects; forced switches
///   are taken as given; the lent layouts save the table less the 2.1 MiB `sorted` grows by.
/// - The planned footprint is the real growth plus at most 4 MiB (page rounding; the check).
/// - Proofs: split + lent and unsplit + lent give the same pre-blinding values and proof as the
///   reference layout (fixed r, s), for a witness-like and a random witness, three proofs in a
///   row (the table is rebuilt before each), and after the fuzz check (whose MSMs overwrite
///   `sorted`).
#[test]
fn b9_4g_tw_lent_bit_exact() {
    assert_eq!(MemOptions::all().tw_lent, TwLent::Auto);
    assert_eq!(MemOptions::none().tw_lent, TwLent::Off);
    assert_eq!(MemOptions::parse("tw_lent=on").unwrap().tw_lent, TwLent::On);
    assert_eq!(MemOptions::parse("-tw_lent").unwrap().tw_lent, TwLent::Off);
    assert_eq!(
        MemOptions::parse("base,+tw_lent").unwrap().tw_lent,
        TwLent::Auto
    );
    assert_eq!(MemOptions::parse("tw_lent=maybe"), None);

    let _lock = prover_test_lock();
    let Some(dir) = component_dir() else {
        return;
    };
    let setup = SetupParams::new(&dir).unwrap();
    let lay = |split_h: SplitH, tw_lent: TwLent, room: u64| {
        let mem = MemOptions {
            split_h,
            tw_lent,
            ..MemOptions::all()
        };
        layout(
            setup.srs_path.as_path(),
            setup.pcoeffs_path.as_path(),
            mem,
            room,
        )
        .unwrap()
    };
    let gib = 1u64 << 30;
    let pso = 62 << 20;
    let whole = lay(SplitH::Off, TwLent::Off, 0);
    let split = lay(SplitH::On, TwLent::Off, 0);
    let whole_tw = lay(SplitH::Off, TwLent::On, 0);
    let split_tw = lay(SplitH::On, TwLent::On, 0);
    assert!(!whole.tw_lent && whole_tw.tw_lent && split_tw.tw_lent && split_tw.split_h);
    let tw = super::super::ntt::NttPlan::twiddle_bytes_for(whole.lg);
    for (a, b) in [(&whole, &whole_tw), (&split, &split_tw)] {
        let saved = a.planned_bytes - b.planned_bytes;
        assert!(saved <= tw && saved + (4 << 20) > tw, "saved {saved}");
    }
    let auto = |room: u64| {
        let l = lay(SplitH::Auto, TwLent::Auto, room);
        (l.split_h, l.tw_lent)
    };
    assert_eq!(auto(u64::MAX), (false, false));
    assert_eq!(auto(8 * gib - pso), (false, false));
    // Fix round 1: the unsplit layout, too, only with the headroom (P31-12 kept it whenever it
    // fitted: `p31_12_split_h_bit_exact` tests that rule with `TwLent::Off`).
    assert_eq!(
        auto(whole.planned_bytes + AUTO_HEADROOM_BYTES),
        (false, false)
    );
    assert_eq!(
        auto(whole.planned_bytes + AUTO_HEADROOM_BYTES - 1),
        (true, false)
    );
    assert_eq!(auto(whole.planned_bytes), (true, false));
    assert_eq!(
        auto(split.planned_bytes + AUTO_HEADROOM_BYTES),
        (true, false)
    );
    assert_eq!(
        auto(split.planned_bytes + AUTO_HEADROOM_BYTES - 1),
        (true, true)
    );
    assert_eq!(auto(4 * gib - pso), (true, true));
    assert_eq!(auto(4 * gib - (46 << 20)), (true, true));
    // Smaller than every layout: the smallest (then refused by the budget check).
    assert_eq!(auto(0), (true, true));
    // Forced switches.
    assert_eq!(
        {
            let l = lay(SplitH::Off, TwLent::Auto, 4 * gib - pso);
            (l.split_h, l.tw_lent)
        },
        (false, true)
    );
    assert_eq!(
        {
            let l = lay(SplitH::Auto, TwLent::Off, 4 * gib - pso);
            (l.split_h, l.tw_lent)
        },
        (true, false)
    );
    assert_eq!(
        {
            let l = lay(SplitH::Off, TwLent::On, u64::MAX);
            (l.split_h, l.tw_lent)
        },
        (false, true)
    );
    eprintln!(
        "B9-4G layout: planned whole {:.1}, split {:.1}, whole+tw {:.1}, split+tw {:.1} MiB; \
         sorted min {:.1} MiB (scratch sorted {:.1} MiB)",
        mib(whole.planned_bytes),
        mib(split.planned_bytes),
        mib(whole_tw.planned_bytes),
        mib(split_tw.planned_bytes),
        mib(split_tw.sorted_min),
        mib(MsmScratch::sizes_for(split_tw.cap, split_tw.cfg)[4]),
    );

    let mut reference: Vec<Proof> = vec![];
    for (split_h, tw_lent) in [
        (SplitH::Off, TwLent::Off),
        (SplitH::On, TwLent::On),
        (SplitH::Off, TwLent::On),
    ] {
        let mem = MemOptions {
            split_h,
            tw_lent,
            ..MemOptions::all()
        };
        let mut prover = match MetalProver::new_with(&setup, mem) {
            Ok(p) => p,
            Err(e) if super::super::working_set_budget().is_some() => {
                let msg = format!("{e:#}");
                assert!(msg.contains("working-set budget exceeded"), "{msg}");
                eprintln!("B9-4G {split_h:?} {tw_lent:?}: refused under the budget: {msg}");
                continue;
            }
            Err(e) => panic!("{split_h:?} {tw_lent:?}: {e:#}"),
        };
        let name = format!("split_h={split_h:?},tw_lent={tw_lent:?}");
        report_build(&name, &prover);
        assert_eq!(prover.tw_lent(), tw_lent == TwLent::On);
        assert_eq!(prover.build_stats.tw_lent, prover.tw_lent());
        // Fix round 1: a lent region is touched at the build (the table is rebuilt per proof
        // anyway; the proofs below check that the result is unchanged).
        assert_eq!(prover.build_stats.tw_prefault.is_some(), prover.tw_lent());
        let st = &prover.build_stats;
        let growth = st.allocated_after - st.allocated_before;
        // The check's slack (4 MiB; 32 with `RISC0_G16_METAL_ARCHIVE=0`, see [plan_slack]).
        assert!(
            st.planned_bytes >= growth && st.planned_bytes - growth <= plan_slack(),
            "{name}: planned {} vs growth {growth}",
            st.planned_bytes
        );
        let mut proofs = vec![];
        let witnesses = [
            ("witness-like", witness_like(&prover, 0x3112)),
            ("random", random_witness(&prover, 0x3113)),
            ("witness-like again", witness_like(&prover, 0x3112)),
        ];
        for (what, w) in &witnesses {
            let p = prover
                .prove_with_blinding(w, &ProveOptions::default(), Fr::from(3u64), Fr::from(5u64))
                .unwrap();
            eprintln!(
                "B9-4G {name} {what}: gpu {:.1} ms, allocated after {:.1} MiB",
                p.stats.gpu_total_ms(),
                mib(p.stats.allocated)
            );
            assert_eq!(
                p.stats.allocated, prover.build_stats.allocated_after,
                "{what}"
            );
            proofs.push(p);
        }
        if prover.tw_lent() {
            let (a, b1, c, b2) = prover.fuzz_check().unwrap();
            assert!(
                a.is_zero() && b1.is_zero() && c.is_zero() && b2.is_zero(),
                "fuzz check ({name})"
            );
            let p = prover
                .prove_with_blinding(
                    &witnesses[1].1,
                    &ProveOptions::default(),
                    Fr::from(3u64),
                    Fr::from(5u64),
                )
                .unwrap();
            assert_eq!(
                p.pre_blinding, proofs[1].pre_blinding,
                "after the fuzz check"
            );
            // Profiled (one command buffer per stage): the same proof.
            let p = prover
                .prove_with_blinding(
                    &witnesses[0].1,
                    &ProveOptions {
                        profile: true,
                        ..Default::default()
                    },
                    Fr::from(3u64),
                    Fr::from(5u64),
                )
                .unwrap();
            assert_eq!(p.pre_blinding, proofs[0].pre_blinding, "profiled");
        }
        assert_eq!(proofs[0].pre_blinding, proofs[2].pre_blinding);
        if reference.is_empty() {
            reference = proofs;
        } else {
            for (p, r) in proofs.iter().zip(&reference) {
                assert_eq!(p.pre_blinding, r.pre_blinding, "{name}");
                assert_eq!((p.a, p.b, p.c), (r.a, r.b, r.c), "{name}");
            }
        }
    }
    assert!(!reference.is_empty(), "no layout was built");
}

/// G10: interleaved A/B of memory layouts on the real key (benchmark, ignored). `G10_VARIANTS`
/// = [MemOptions::parse] specs separated by `;` (default "base;all"); every variant's prover is
/// built once and kept (they coexist: ~10 GB for two), then `G10_REPS` rounds (default 7) prove
/// with each in turn. `G10_PROFILE=1`: one command buffer per stage. `G10_WITNESS=<file>`: a
/// real witness (as `G9_WITNESS`) instead of random values. Pre-blinding values must equal the
/// first variant's.
#[test]
#[ignore]
fn g10_bench() {
    let _lock = prover_test_lock();
    let Some(dir) = component_dir() else {
        return;
    };
    let setup = SetupParams::new(&dir).unwrap();
    let specs = std::env::var("G10_VARIANTS").unwrap_or_else(|_| "base;all".into());
    let reps: usize = std::env::var("G10_REPS").map_or(7, |v| v.parse().unwrap());
    let profile = std::env::var("G10_PROFILE").as_deref() == Ok("1");
    let mut provers: Vec<(String, MetalProver)> = specs
        .split(';')
        .map(|s| {
            let p = MetalProver::new_with(&setup, MemOptions::parse(s).expect("bad G10_VARIANTS"))
                .unwrap();
            report_build(s, &p);
            (s.to_string(), p)
        })
        .collect();
    let witness = match std::env::var("G10_WITNESS") {
        Ok(path) => {
            let w = std::fs::read(&path).expect("G10_WITNESS");
            assert_eq!(w.len(), provers[0].1.witness_len() * 32, "G10_WITNESS size");
            w
        }
        Err(_) => random_witness(&provers[0].1, 9),
    };
    let opts = ProveOptions {
        profile,
        ..Default::default()
    };
    let mut reference = None;
    for (_, p) in provers.iter_mut() {
        let pre = p
            .prove_with_blinding(&witness, &opts, Fr::from(3u64), Fr::from(5u64))
            .unwrap()
            .pre_blinding;
        assert_eq!(reference.get_or_insert(pre.clone()), &pre);
    }
    let mut gpu = vec![Vec::new(); provers.len()];
    let mut total = vec![Vec::new(); provers.len()];
    let mut stages: Vec<std::collections::BTreeMap<String, Vec<f64>>> =
        vec![Default::default(); provers.len()];
    for rep in 0..reps {
        for (vi, (name, p)) in provers.iter_mut().enumerate() {
            let pr = p
                .prove_with_blinding(&witness, &opts, Fr::from(3u64), Fr::from(5u64))
                .unwrap();
            assert_eq!(
                Some(&pr.pre_blinding),
                reference.as_ref(),
                "{name} rep {rep}"
            );
            let st = &pr.stats;
            gpu[vi].push(st.gpu_total_ms());
            total[vi].push(st.total_ms);
            for (k, v) in &st.gpu_ms {
                stages[vi].entry(k.clone()).or_default().push(*v);
            }
            eprintln!(
                "G10BENCH rep {rep} {name}: gpu {:.1} total {:.1} ms [{}]",
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
    for (vi, (name, _)) in provers.iter().enumerate() {
        eprintln!(
            "G10SUMMARY {name}: gpu {} total {} ms",
            stat(&gpu[vi]),
            stat(&total[vi])
        );
        for (k, v) in &stages[vi] {
            eprintln!("G10SUMMARY {name}   {k}: {}", stat(v));
        }
    }
}

/// Item G13: a witness written straight into the shared witness buffer proves exactly as the same
/// witness passed as a slice (same blinding: equal proofs), and [planned_bytes] is the footprint
/// the build checked against the budget. (Asked before the build: the `Auto` layouts depend on
/// the room left, and once the prover is built an 8 GiB budget has room only for a reduced one;
/// item B9-4G, the assertion failed at 8 GiB before.)
#[test]
fn g13_in_place_witness_bit_exact() {
    let _lock = prover_test_lock();
    let Some(dir) = component_dir() else {
        return;
    };
    let setup = SetupParams::new(&dir).unwrap();
    let planned = planned_bytes(&setup, MemOptions::from_env()).unwrap();
    let Some(mut prover) = build_or_refused(&dir) else {
        return;
    };
    assert_eq!(planned, prover.build_stats.planned_bytes);
    let opts = ProveOptions::default();
    for seed in [13u64, 14] {
        let witness = random_witness(&prover, seed);
        let copied = prover
            .prove_with_blinding(&witness, &opts, Fr::from(3u64), Fr::from(5u64))
            .unwrap();
        // Overwrite the buffer with something else first, then the witness, in place.
        let buf = prover.witness_buffer();
        assert_eq!(buf.len() * 32, witness.len());
        buf.iter_mut().for_each(|e| *e = [7, 0, 0, 0]);
        for (e, w) in prover
            .witness_buffer()
            .iter_mut()
            .zip(witness.chunks_exact(32))
        {
            for (j, limb) in e.iter_mut().enumerate() {
                *limb = u64::from_le_bytes(w[8 * j..8 * j + 8].try_into().unwrap());
            }
        }
        let in_place = prover
            .prove_from(None, &opts, Fr::from(3u64), Fr::from(5u64))
            .unwrap();
        assert_eq!(in_place.a, copied.a);
        assert_eq!(in_place.b, copied.b);
        assert_eq!(in_place.c, copied.c);
        assert_eq!(in_place.public, copied.public);
    }
}
