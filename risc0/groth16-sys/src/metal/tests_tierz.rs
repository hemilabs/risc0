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

//! P31-T1: bypass-forcing stress of the `msm_window_reduce_{g1,g2,g1w,g2w}` kernels, which
//! `tools/metal_audit/ir_scan.py` tier Z flags (W7-T1's unvalidated heuristic for Z13's shape: a
//! runtime guard that skips a loop updating a private aggregate, read after the merge). Here the
//! guard is the entry of `for (j = tid; j < nsegs; j += nthr) acc = acc.add(..)`: threads with
//! `tid >= nsegs` skip it and store `acc = inf` to threadgroup memory. The shapes below make
//! `nsegs` smaller than the threadgroup width (<= 64) or zero, and every MSM is compared with
//! arkworks.
//!
//! Both tests are opt-in by environment (they return at once otherwise), so that they can run
//! inside the default parallel harness of the whole suite:
//!
//! `R17_TIERZ_REPS=20 R17_TIERZ_G14_REPS=50 cargo test -p risc0-groth16-sys --features metal \
//!  --lib` (the whole suite, in parallel) or `... --lib -- tierz_ --nocapture` (alone).
//! `tools/metal_audit/run_stress.sh` runs them in its `groth16_tierz` step.
//!
//! Lines: `R17AUDIT groth16 tierz <shape> <curve> <tune>: m/n reps differ from ark`.

use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup, VariableBaseMSM};
use ark_ff::UniformRand;
use ark_std::{One, Zero};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::msm::{scalar_words, MsmBuf, MsmConfig, MsmCurve, MsmGpu, MsmScratch, MsmTune, G1, G2};

/// An [MsmGpu] holding the shared GPU test lock for the test (as `msm_tests.rs`).
struct LockedGpu {
    gpu: MsmGpu,
    _lock: std::sync::RwLockReadGuard<'static, ()>,
}

impl std::ops::Deref for LockedGpu {
    type Target = MsmGpu;

    fn deref(&self) -> &MsmGpu {
        &self.gpu
    }
}

fn gpu() -> Option<LockedGpu> {
    let lock = super::test_lock::shared();
    let g = MsmGpu::new().unwrap();
    if g.is_none() {
        eprintln!("no Metal device; skipping");
    }
    g.map(|gpu| LockedGpu { gpu, _lock: lock })
}

/// `key` as a rep count, or None (the test is skipped) when unset.
fn reps_env(key: &str, min: usize) -> Option<usize> {
    let v = std::env::var(key).ok()?;
    let n: usize = v
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("{key}={v}: not a count"));
    Some(n.max(min))
}

/// The ark reference MSM, split over threads.
fn ark_msm<G>(points: &[G::MulBase], scalars: &[Fr]) -> G
where
    G: VariableBaseMSM<ScalarField = Fr> + Send,
    G::MulBase: Sync,
{
    if points.is_empty() {
        return G::zero();
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = points.len().div_ceil(threads).max(1024);
    std::thread::scope(|s| {
        let hs: Vec<_> = points
            .chunks(chunk)
            .zip(scalars.chunks(chunk))
            .map(|(p, k)| s.spawn(move || G::msm(p, k).unwrap()))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

/// `n` points P0 + i Q, ~7% infinity, every 997th equal to its predecessor (as `msm_tests.rs`).
fn points<G>(n: usize, rng: &mut ChaCha8Rng) -> Vec<G::Affine>
where
    G: CurveGroup<ScalarField = Fr> + PrimeGroup,
{
    let p0 = G::generator() * Fr::rand(rng);
    let q = G::generator() * Fr::rand(rng);
    let mut acc = p0;
    let proj: Vec<G> = (0..n)
        .map(|_| {
            let p = acc;
            acc += q;
            p
        })
        .collect();
    let mut aff = G::normalize_batch(&proj);
    for i in 0..n {
        if rng.gen_ratio(7, 100) {
            aff[i] = <G::Affine as AffineRepr>::zero();
        } else if i > 0 && i % 997 == 0 {
            aff[i] = aff[i - 1];
        }
    }
    aff
}

fn words<C: MsmCurve>(points: &[C::Affine]) -> Vec<u32> {
    let mut w = Vec::new();
    for p in points {
        C::write_affine(p, &mut w);
    }
    w
}

/// Item G14's witness-like scalars (`msm_tests.rs`).
fn witness_like_scalars(n: usize, rng: &mut ChaCha8Rng) -> Vec<Fr> {
    let heavy: Vec<Fr> = (0..8).map(|_| Fr::rand(rng)).collect();
    (0..n)
        .map(|_| match rng.gen_range(0..100) {
            0..=43 => Fr::zero(),
            44..=72 => Fr::one(),
            73..=75 => -Fr::one(),
            76..=79 => Fr::from(rng.gen_range(2..1u64 << 40)),
            80..=81 => heavy[rng.gen_range(0..heavy.len())],
            _ => Fr::rand(rng),
        })
        .collect()
}

/// One input set: G1 and G2 points, scalars, and the ark results.
struct Case {
    b1: metal::Buffer,
    b2: metal::Buffer,
    bs: metal::Buffer,
    n: usize,
    want1: G1Affine,
    want2: G2Affine,
}

impl Case {
    fn new(g: &MsmGpu, p1: &[G1Affine], p2: &[G2Affine], s: &[Fr]) -> Self {
        let n = s.len();
        let want1 = ark_msm::<G1Projective>(&p1[..n], s).into_affine();
        let want2 = ark_msm::<G2Projective>(&p2[..n], s).into_affine();
        // Metal buffers cannot be empty; n >= 1 here.
        let sw: Vec<u32> = s.iter().flat_map(scalar_words).collect();
        Self {
            b1: g.buffer(&words::<G1>(&p1[..n])),
            b2: g.buffer(&words::<G2>(&p2[..n])),
            bs: g.buffer(&sw),
            n,
            want1,
            want2,
        }
    }

    /// `reps` x (sort, G1 MSM, G2 MSM) on a scratch of `cfg` with `tune` and `bits`
    /// ([MsmScratch::set_scalar_bits]); returns the reps whose G1 and G2 results differ from ark.
    fn run(
        &self,
        g: &MsmGpu,
        cfg: MsmConfig,
        tune: MsmTune,
        bits: u32,
        reps: usize,
    ) -> (usize, usize, Vec<String>) {
        let mut sc = MsmScratch::new(&g.device, self.n as u32, cfg).unwrap();
        sc.set_scalar_bits(bits);
        sc.set_tune(tune);
        let (mut bad1, mut bad2, mut detail) = (0, 0, vec![]);
        for rep in 0..reps {
            g.sort(&mut sc, MsmBuf::new(&self.bs), self.n as u32)
                .unwrap();
            let (got1, _) = g.msm_sorted::<G1>(&sc, MsmBuf::new(&self.b1), 0).unwrap();
            let (got2, _) = g.msm_sorted::<G2>(&sc, MsmBuf::new(&self.b2), 0).unwrap();
            if got1.into_affine() != self.want1 {
                bad1 += 1;
                detail.push(format!("rep {rep}: G1 differs"));
            }
            if got2.into_affine() != self.want2 {
                bad2 += 1;
                detail.push(format!("rep {rep}: G2 differs"));
            }
        }
        (bad1, bad2, detail)
    }
}

fn tune_name(t: &MsmTune) -> &'static str {
    match (t.wide_mul, t.booth) {
        (true, true) => "all (g1w/g2w, booth)",
        (true, false) => "wide (g1w/g2w, unsigned)",
        (false, _) => "none (g1/g2, unsigned)",
    }
}

/// Runs `case` over `tunes` and scalar bits 254 and 256 (as g14's `check_scalars`); prints one
/// line per curve, tune and bits; returns the failing lines.
fn stress_case(
    g: &MsmGpu,
    what: &str,
    case: &Case,
    cfg: MsmConfig,
    tunes: &[MsmTune],
    reps: usize,
) -> Vec<String> {
    let mut bad = vec![];
    for tune in tunes {
        for bits in [254, 256] {
            let (b1, b2, detail) = case.run(g, cfg, *tune, bits, reps);
            for (curve, k) in [("G1", b1), ("G2", b2)] {
                let line = format!(
                    "R17AUDIT groth16 tierz {what} n {} wbits {} seg {} bits {bits} {curve} {}: \
                     {k}/{reps} reps differ from ark",
                    case.n,
                    cfg.wbits,
                    cfg.seg,
                    tune_name(tune)
                );
                println!("{line}");
                if k > 0 {
                    println!("   {detail:?}");
                    bad.push(line);
                }
            }
        }
    }
    bad
}

/// The tier-Z bypass shapes of `msm_window_reduce` (threadgroup width <= 64): every window has
/// fewer segments than threads (n 1, 3, 33, 1000; n 4096 with seg 512; c = 4 with seg 1), no pieces at all
/// (all-zero scalars: every thread skips the loop), or only window 0 non-empty (scalars below
/// 2^16), plus a no-bypass control (n 2^14, ~256 segments per window). G1 and G2 with the
/// default kernels (`_g1w`/`_g2w`, Booth) and the G6 kernels (`_g1`/`_g2`); `R17_TIERZ_REPS`
/// (>= 20) reps per shape.
#[test]
fn tierz_msm_window_bypass() {
    let Some(reps) = reps_env("R17_TIERZ_REPS", 20) else {
        eprintln!("tierz_msm_window_bypass: set R17_TIERZ_REPS to run (skipped)");
        return;
    };
    let Some(g) = gpu() else { return };
    let mut rng = ChaCha8Rng::seed_from_u64(0x31_71);
    let nmax = 1 << 14;
    let p1 = points::<G1Projective>(nmax, &mut rng);
    let p2 = points::<G2Projective>(nmax, &mut rng);
    let rand_s = |n: usize, rng: &mut ChaCha8Rng| (0..n).map(|_| Fr::rand(rng)).collect();
    let tunes = [
        MsmTune::all(),
        MsmTune {
            booth: false,
            ..MsmTune::all()
        },
        MsmTune::none(),
    ];
    let dflt = MsmConfig::default();
    let shapes: Vec<(&str, Vec<Fr>, MsmConfig)> = vec![
        ("n1", rand_s(1, &mut rng), dflt),
        ("n3 for_size", rand_s(3, &mut rng), MsmConfig::for_size(3)),
        ("n33", rand_s(33, &mut rng), dflt),
        ("all-zero (np 0 everywhere)", vec![Fr::zero(); 1000], dflt),
        (
            "below 2^16 (windows >= 1 empty)",
            (0..1000)
                .map(|_| Fr::from(rng.gen_range(0..1u64 << 16)))
                .collect(),
            dflt,
        ),
        ("n1000 (~16 segs/window)", rand_s(1000, &mut rng), dflt),
        (
            "seg 512 (~9 segs/window)",
            rand_s(4096, &mut rng),
            MsmConfig { seg: 512, ..dflt },
        ),
        (
            "c 4 seg 1 (~23 segs/window)",
            rand_s(1000, &mut rng),
            MsmConfig {
                wbits: 4,
                seg: 1,
                ..dflt
            },
        ),
        ("control n 2^14", rand_s(nmax, &mut rng), dflt),
    ];
    let mut bad = vec![];
    for (what, s, cfg) in &shapes {
        let t0 = std::time::Instant::now();
        let case = Case::new(&g, &p1, &p2, s);
        bad.extend(stress_case(&g, what, &case, *cfg, &tunes, reps));
        println!("TIERZ {what}: {:.1} s", t0.elapsed().as_secs_f64());
    }
    assert!(bad.is_empty(), "tier-Z MSM shapes wrong: {bad:#?}");
}

/// W7-G1's open issue: `g14_witness_like_scalars` failed once in 3 full-suite runs (a G2 MSM
/// mismatch, all r - 1, n 2^17) and passed 8/8 alone. This repeats g14's inputs (witness-like
/// n 2^18 at c = 16 and c = 11, all ones at 2^18 (c = 11) and 2^17, all r - 1 at 2^17) with every
/// g14 tune and scalar bits 254 and 256, `R17_TIERZ_G14_REPS` (>= 50) times each, against ark.
/// `R17_TIERZ_G14_CASES=4` runs only the all r - 1 case (indices in the order above).
#[test]
fn tierz_g14_repeat() {
    let Some(reps) = reps_env("R17_TIERZ_G14_REPS", 50) else {
        eprintln!("tierz_g14_repeat: set R17_TIERZ_G14_REPS to run (skipped)");
        return;
    };
    let Some(g) = gpu() else { return };
    let mut rng = ChaCha8Rng::seed_from_u64(0x14);
    let n = 1 << 18;
    let p1 = points::<G1Projective>(n, &mut rng);
    let p2 = points::<G2Projective>(n, &mut rng);
    let tunes = [
        MsmTune::all(),
        MsmTune {
            lenperm: false,
            lent_groups: false,
            ..MsmTune::all()
        },
        MsmTune::none(),
    ];
    let s = witness_like_scalars(n, &mut rng);
    let small = MsmConfig {
        wbits: 11,
        seg: 5,
        ..Default::default()
    };
    let dflt = MsmConfig::default();
    let n2 = n / 2;
    let cases: Vec<(&str, Vec<Fr>, MsmConfig, &[MsmTune])> = vec![
        ("g14 witness-like", s.clone(), dflt, &tunes),
        ("g14 witness-like", s, small, &tunes[..1]),
        ("g14 all ones", vec![Fr::one(); n], small, &tunes[..1]),
        ("g14 all ones", vec![Fr::one(); n2], dflt, &tunes),
        ("g14 all r - 1", vec![-Fr::one(); n2], dflt, &tunes),
    ];
    let mut bad = vec![];
    // R17_TIERZ_G14_CASES: comma-separated indices into `cases` (default all), to split the
    // repeat over several locked runs.
    let only: Option<Vec<usize>> = std::env::var("R17_TIERZ_G14_CASES")
        .ok()
        .map(|v| v.split(',').map(|x| x.trim().parse().unwrap()).collect());
    for (i, (what, s, cfg, tunes)) in cases.iter().enumerate() {
        if only.as_ref().is_some_and(|o| !o.contains(&i)) {
            continue;
        }
        let t0 = std::time::Instant::now();
        let case = Case::new(&g, &p1, &p2, s);
        bad.extend(stress_case(&g, what, &case, *cfg, tunes, reps));
        println!("TIERZ {what}: {:.1} s", t0.elapsed().as_secs_f64());
    }
    assert!(bad.is_empty(), "g14 inputs wrong: {bad:#?}");
}
