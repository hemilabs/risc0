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

//! MSM equivalence tests against arkworks' `VariableBaseMSM` (item G6).
//!
//! Inputs: random points (consecutive multiples of a random point, so generating 2^20 of them is
//! cheap) with ~7% replaced by infinity and a few duplicated (bucket doubling path), and random
//! scalars with 0, 1 and r - 1 forced at the start and sprinkled at ~1% each. Buffers are bound at
//! non-zero byte offsets; `first` (the C offset) is tested separately.

use std::time::Instant;

use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup, VariableBaseMSM};
use ark_ff::{BigInteger, PrimeField, UniformRand};
use ark_std::{One, Zero};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::*;

/// An [MsmGpu] holding the shared GPU test lock (`super::super::test_lock`) for the test.
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
    let lock = super::super::test_lock::shared();
    let g = MsmGpu::new().unwrap();
    if g.is_none() {
        eprintln!("no Metal device; skipping");
    }
    g.map(|gpu| LockedGpu { gpu, _lock: lock })
}

/// The ark reference MSM, split over threads (ark-ec is built without `parallel`).
fn ark_msm<G>(points: &[G::MulBase], scalars: &[Fr]) -> G
where
    G: VariableBaseMSM<ScalarField = Fr> + Send,
    G::MulBase: Sync,
{
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

/// `n` points: P0 + i Q for random P0, Q, normalized; ~7% infinity; every 997th equal to its
/// predecessor.
fn points<G>(n: usize, rng: &mut ChaCha8Rng) -> Vec<G::Affine>
where
    G: CurveGroup<ScalarField = Fr> + PrimeGroup + Send + Sync,
{
    let p0 = G::generator() * Fr::rand(rng);
    let q = G::generator() * Fr::rand(rng);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = n.div_ceil(threads).max(256);
    let proj: Vec<G> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|start| {
                s.spawn(move || {
                    let end = (start + chunk).min(n);
                    let mut acc = p0 + q * Fr::from(start as u64);
                    let mut v = Vec::with_capacity(end - start);
                    for _ in start..end {
                        v.push(acc);
                        acc += q;
                    }
                    v
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
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

fn scalars(n: usize, rng: &mut ChaCha8Rng) -> Vec<Fr> {
    let specials = [Fr::zero(), Fr::one(), -Fr::one()];
    (0..n)
        .map(|i| {
            if i < specials.len() {
                specials[i]
            } else {
                match rng.gen_range(0..100) {
                    0 => Fr::zero(),
                    1 => Fr::one(),
                    2 => -Fr::one(),
                    _ => Fr::rand(rng),
                }
            }
        })
        .collect()
}

fn words<C: MsmCurve>(points: &[C::Affine]) -> Vec<u32> {
    let mut w = Vec::new();
    for p in points {
        C::write_affine(p, &mut w);
    }
    w
}

/// Bytes the GPU device reports allocated by this process.
fn allocated(g: &MsmGpu) -> u64 {
    g.device.current_allocated_size()
}

/// G1 and G2 MSMs of `n` random terms sharing one sort; points and scalars bound at a non-zero
/// byte offset. Returns (sort, G1, G2) GPU seconds (best of `reps`) and the peak allocation.
fn check_size(g: &MsmGpu, n: usize, seed: u64, cfg: MsmConfig, reps: usize) -> (MsmTiming, f64) {
    check_size_cap(g, n, n, seed, cfg, reps)
}

/// [check_size] on a scratch sized for `cap >= n` scalars.
fn check_size_cap(
    g: &MsmGpu,
    n: usize,
    cap: usize,
    seed: u64,
    cfg: MsmConfig,
    reps: usize,
) -> (MsmTiming, f64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let t0 = Instant::now();
    let p1 = points::<G1Projective>(n, &mut rng);
    let p2 = points::<G2Projective>(n, &mut rng);
    let s = scalars(n, &mut rng);
    let gen_s = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let want1: G1Projective = ark_msm(&p1, &s);
    let want2: G2Projective = ark_msm(&p2, &s);
    let ark_s = t0.elapsed().as_secs_f64();

    // Offsets: 3 points / 5 scalars of junk in front.
    let off_p = 3usize;
    let off_s = 5usize;
    let mut w1 = vec![0xdeadbeefu32; off_p * 16];
    w1.extend(words::<G1>(&p1));
    let mut w2 = vec![0xdeadbeefu32; off_p * 32];
    w2.extend(words::<G2>(&p2));
    let mut ws = vec![0xffffffffu32; off_s * 8];
    ws.extend(s.iter().flat_map(scalar_words));
    let b1 = g.buffer(&w1);
    let b2 = g.buffer(&w2);
    let bs = g.buffer(&ws);
    let mut sc = MsmScratch::new(&g.device, cap as u32, cfg).unwrap();
    let peak = allocated(g);

    let mut best = MsmTiming {
        sort: f64::MAX,
        msm: f64::MAX,
    };
    let mut best2 = f64::MAX;
    for _ in 0..reps {
        let ts = g
            .sort(&mut sc, MsmBuf::at(&bs, off_s as u64 * 32), n as u32)
            .unwrap();
        let (got1, t1) = g
            .msm_sorted::<G1>(&sc, MsmBuf::at(&b1, off_p as u64 * G1::AFFINE_BYTES), 0)
            .unwrap();
        let (got2, t2) = g
            .msm_sorted::<G2>(&sc, MsmBuf::at(&b2, off_p as u64 * G2::AFFINE_BYTES), 0)
            .unwrap();
        assert_eq!(
            got1.into_affine(),
            want1.into_affine(),
            "G1 MSM n = {n} cap = {cap} {cfg:?}"
        );
        assert_eq!(
            got2.into_affine(),
            want2.into_affine(),
            "G2 MSM n = {n} cap = {cap} {cfg:?}"
        );
        best.sort = best.sort.min(ts);
        best.msm = best.msm.min(t1);
        best2 = best2.min(t2);
    }
    if reps > 1 {
        // One more run with a command buffer per stage, for the per-stage times.
        let (got1, _) = g
            .msm_sorted_profiled::<G1>(&sc, MsmBuf::at(&b1, off_p as u64 * G1::AFFINE_BYTES), 0)
            .unwrap();
        let (got2, _) = g
            .msm_sorted_profiled::<G2>(&sc, MsmBuf::at(&b2, off_p as u64 * G2::AFFINE_BYTES), 0)
            .unwrap();
        assert_eq!(
            got1.into_affine(),
            want1.into_affine(),
            "G1 MSM (profiled) n = {n}"
        );
        assert_eq!(
            got2.into_affine(),
            want2.into_affine(),
            "G2 MSM (profiled) n = {n}"
        );
    }
    eprintln!(
        "n = {n:>8} {cfg:?}: sort {:.2} ms, G1 {:.2} ms, G2 {:.2} ms (GPU, best of {reps}); \
         scratch {:.1} MiB, process GPU allocation {:.1} MiB (inputs+scratch {:.1} MiB); \
         inputs generated in {gen_s:.1} s, ark reference {ark_s:.1} s",
        best.sort * 1e3,
        best.msm * 1e3,
        best2 * 1e3,
        sc.bytes() as f64 / 1048576.0,
        peak as f64 / 1048576.0,
        (b1.length() + b2.length() + bs.length() + sc.bytes()) as f64 / 1048576.0,
    );
    (best, best2)
}

#[test]
fn msm_sizes() {
    let Some(g) = gpu() else { return };
    for (i, n) in [1usize, 2, 3, 31, 32, 33, 1000, 1 << 16]
        .into_iter()
        .enumerate()
    {
        check_size(&g, n, 100 + i as u64, MsmConfig::default(), 1);
        if MsmConfig::for_size(n as u32) != MsmConfig::default() {
            check_size(&g, n, 150 + i as u64, MsmConfig::for_size(n as u32), 1);
        }
    }
}

/// 2^20 terms: correctness and the timing/memory figures recorded in the G6 note.
#[test]
fn msm_2_20() {
    let Some(g) = gpu() else { return };
    check_size(&g, 1 << 20, 7, MsmConfig::default(), 3);
}

/// `max_pieces` / `max_segs` bound the pieces of every n up to their argument and never
/// decrease, so a scratch sized for `cap_n` fits every sort of n <= cap_n. ceil(n / piece(n))
/// alone drops where piece(n) steps up (e.g. c = 16: 2^19 -> 2^19 + 1), which a scratch sized
/// from it at cap_n would miss. Exhaustive over n for each c.
#[test]
fn msm_piece_bound() {
    for wbits in 1..=16u32 {
        let cfg = MsmConfig {
            wbits,
            seg: 7,
            ..Default::default()
        };
        let nb = cfg.nbuckets();
        let top = (64 * nb).max(1 << 14);
        let (mut prev_p, mut prev_s) = (0, 0);
        for n in 0..=top {
            let exact = n.div_ceil(cfg.piece(n)) + n.min(nb - 1);
            let (mp, ms) = (cfg.max_pieces(n), cfg.max_segs(n));
            assert!(
                exact <= mp,
                "c = {wbits} n = {n}: {exact} pieces > bound {mp}"
            );
            assert!(
                mp >= prev_p && ms >= prev_s,
                "c = {wbits} n = {n}: bound decreased"
            );
            (prev_p, prev_s) = (mp, ms);
        }
        // The bound is tight at the knee (no memory regression vs the old per-n formula there).
        let knee = 8 * nb;
        assert_eq!(
            cfg.max_pieces(knee),
            knee.div_ceil(cfg.piece(knee)) + knee.min(nb - 1)
        );
    }
}

/// G9: the piece bound with the Booth bucket count nb = 2^c + 1 ([MsmConfig::layout]) holds and
/// is non-decreasing for every n up to 64 nb, and the scratch's bound (sized for nb, either digit
/// mode) covers the prover's sizes (witness 5,635,930, domain 2^23).
#[test]
fn msm_piece_bound_booth() {
    for wbits in 1..=16u32 {
        let cfg = MsmConfig {
            wbits,
            seg: 7,
            ..Default::default()
        };
        let nb = cfg.nbuckets() + 1;
        let top = (64 * nb).max(1 << 14);
        let (mut prev_p, mut prev_s) = (0, 0);
        for n in 0..=top {
            let exact = n.div_ceil(MsmConfig::piece_nb(n, nb)) + n.min(nb - 1);
            let (mp, ms) = (MsmConfig::max_pieces_nb(n, nb), cfg.max_segs_nb(n, nb));
            assert!(
                exact <= mp,
                "c = {wbits} n = {n}: {exact} pieces > bound {mp}"
            );
            assert!(
                mp >= prev_p && ms >= prev_s,
                "c = {wbits} n = {n}: bound decreased"
            );
            (prev_p, prev_s) = (mp, ms);
        }
        // The unsigned layout's bound is the old one, and below the Booth sizing.
        for n in [0, 1, 1000, top] {
            assert_eq!(MsmConfig::max_pieces_nb(n, nb - 1), cfg.max_pieces(n));
            assert!(cfg.max_pieces(n) <= MsmConfig::max_pieces_nb(n, nb));
        }
    }
    let cfg = MsmConfig::default();
    for n in [5_635_930u32, 1 << 23] {
        let nb = cfg.nbuckets() + 1;
        let exact = n.div_ceil(MsmConfig::piece_nb(n, nb)) + n.min(nb - 1);
        assert!(exact <= MsmConfig::max_pieces_nb(n, nb), "n = {n}");
    }
}

/// A scratch sized for more scalars than the sort: the strides are the scratch's, and the piece
/// bound of n must fit it, including just below a step of piece(n) (where the old per-n bound was
/// larger than the bound at cap). Both curves (check_size_cap checks G1 and G2).
#[test]
fn msm_scratch_cap_above_n() {
    let Some(g) = gpu() else { return };
    let cases: [(u32, u32, usize, usize); 6] = [
        // (wbits, seg, n, cap)
        (8, 1, 1937, 2049),
        (8, 1, 2048, 2049),
        (8, 7, 33, 2049),
        (6, 16, 497, 513),
        (11, 1, 16085, 16385),
        (16, 64, 1 << 19, (1 << 19) + 1),
    ];
    for (i, (wbits, seg, n, cap)) in cases.into_iter().enumerate() {
        let cfg = MsmConfig {
            wbits,
            seg,
            ..Default::default()
        };
        check_size_cap(&g, n, cap, 500 + i as u64, cfg, 1);
    }
    // Window groups with a cap above n (pieces buffer of a few windows at cap's stride).
    let cfg = MsmConfig {
        wbits: 8,
        seg: 1,
        max_piece_bytes: 3 * 1600 * 256,
    };
    check_size_cap(&g, 1937, 2049, 520, cfg, 1);
}

/// Window widths that do not divide 32 (digits straddle limbs) and window grouping (a pieces
/// buffer that holds one or a few windows).
#[test]
fn msm_configs() {
    let Some(g) = gpu() else { return };
    for (i, cfg) in [
        MsmConfig {
            wbits: 13,
            ..Default::default()
        },
        MsmConfig {
            wbits: 8,
            seg: 7,
            ..Default::default()
        },
        MsmConfig {
            wbits: 1,
            seg: 3,
            ..Default::default()
        },
        MsmConfig {
            max_piece_bytes: 1, // one window per group
            ..Default::default()
        },
        MsmConfig {
            wbits: 11,
            seg: 1,
            max_piece_bytes: 3 * 1500 * 256, // a few windows per group
        },
    ]
    .into_iter()
    .enumerate()
    {
        check_size(&g, 1000, 200 + i as u64, cfg, 1);
        check_size(&g, 33, 300 + i as u64, cfg, 1);
    }
}

/// n = 1 with each special scalar, and all-zero / all-infinity / single-bucket inputs.
#[test]
fn msm_edges() {
    let Some(g) = gpu() else { return };
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let p1 = points::<G1Projective>(5000, &mut rng);
    let p2 = points::<G2Projective>(5000, &mut rng);
    let p1: Vec<G1Affine> = p1.into_iter().filter(|p| !p.infinity).collect();
    let p2: Vec<G2Affine> = p2.into_iter().filter(|p| !p.infinity).collect();

    let check = |what: &str, p1: &[G1Affine], p2: &[G2Affine], s: &[Fr]| {
        let n = s.len();
        let got1 = g.msm::<G1>(&p1[..n], s).unwrap();
        let got2 = g.msm::<G2>(&p2[..n], s).unwrap();
        let want1: G1Projective = ark_msm(&p1[..n], s);
        let want2: G2Projective = ark_msm(&p2[..n], s);
        assert_eq!(got1.into_affine(), want1.into_affine(), "G1 {what}");
        assert_eq!(got2.into_affine(), want2.into_affine(), "G2 {what}");
    };

    for s in [
        Fr::zero(),
        Fr::one(),
        -Fr::one(),
        Fr::from(2u64),
        Fr::rand(&mut rng),
    ] {
        check(&format!("n = 1, s = {s}"), &p1, &p2, &[s]);
    }
    check("no terms", &p1, &p2, &[]);
    check("all scalars zero", &p1, &p2, &vec![Fr::zero(); 1000]);
    // One bucket holds every entry of window 0 (4000 points, many pieces).
    check("all scalars one", &p1, &p2, &vec![Fr::one(); 4000]);
    check("all scalars r - 1", &p1, &p2, &vec![-Fr::one(); 4000]);
    // Every digit the same: each window has one bucket.
    let rep = Fr::from_bigint(ark_ff::BigInt::new([0x0001_0001_0001_0001; 4])).unwrap();
    check("repeated digit", &p1, &p2, &vec![rep; 3000]);
    // Half the scalars 1, half random (witness-like skew).
    let skew: Vec<Fr> = (0..4000)
        .map(|i| {
            if i % 2 == 0 {
                Fr::one()
            } else {
                Fr::rand(&mut rng)
            }
        })
        .collect();
    check("skewed", &p1, &p2, &skew);
    // The same point with the same scalar many times (doubling inside a piece).
    let same1 = vec![p1[0]; 300];
    let same2 = vec![p2[0]; 300];
    let s: Vec<Fr> = (0..300).map(|i| Fr::from(1 + (i % 3) as u64)).collect();
    check("repeated point", &same1, &same2, &s);
    // A point and its negation in the same bucket (P + (-P)).
    let pm1 = vec![p1[1], -p1[1], p1[2]];
    let pm2 = vec![p2[1], -p2[1], p2[2]];
    check("P + (-P)", &pm1, &pm2, &[Fr::from(5u64); 3]);

    // All points at infinity.
    let inf1 = vec![<G1Affine as AffineRepr>::zero(); 100];
    let inf2 = vec![<G2Affine as AffineRepr>::zero(); 100];
    let s: Vec<Fr> = (0..100).map(|_| Fr::rand(&mut rng)).collect();
    assert!(g.msm::<G1>(&inf1, &s).unwrap().is_zero());
    assert!(g.msm::<G2>(&inf2, &s).unwrap().is_zero());
}

/// Raw 256-bit scalars >= r (the kernels take any integer): the result is (s mod r) * P.
#[test]
fn msm_raw_scalars_above_r() {
    let Some(g) = gpu() else { return };
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let n = 500;
    let p1 = points::<G1Projective>(n, &mut rng);
    let p2 = points::<G2Projective>(n, &mut rng);
    let r = Fr::MODULUS;
    let mut raw = Vec::new();
    let mut reduced = Vec::new();
    for i in 0..n {
        let w: [u32; 8] = match i % 4 {
            0 => [u32::MAX; 8],
            1 => {
                let mut v = r;
                v.add_with_carry(&ark_ff::BigInt::from(i as u64));
                let mut w = [0u32; 8];
                for (k, l) in v.0.iter().enumerate() {
                    w[2 * k] = *l as u32;
                    w[2 * k + 1] = (*l >> 32) as u32;
                }
                w
            }
            _ => std::array::from_fn(|_| rng.gen()),
        };
        let mut bytes = Vec::new();
        for x in w {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        raw.push(w);
        reduced.push(Fr::from_le_bytes_mod_order(&bytes));
    }
    let (got1, _) = g.msm_raw::<G1>(&p1, &raw, MsmConfig::default()).unwrap();
    let (got2, _) = g.msm_raw::<G2>(&p2, &raw, MsmConfig::default()).unwrap();
    assert_eq!(
        got1.into_affine(),
        ark_msm::<G1Projective>(&p1, &reduced).into_affine()
    );
    assert_eq!(
        got2.into_affine(),
        ark_msm::<G2Projective>(&p2, &reduced).into_affine()
    );
}

/// The C offset: one sort of the witness, MSMs from `first` on (points[i - first] for i >= first),
/// for several `first`, both curves, with a re-sort in between (scratch reuse).
#[test]
fn msm_first_offset() {
    let Some(g) = gpu() else { return };
    let mut rng = ChaCha8Rng::seed_from_u64(13);
    let n = 3000usize;
    let mut sc = MsmScratch::new(&g.device, n as u32, MsmConfig::default()).unwrap();
    for round in 0..2 {
        let s = scalars(n, &mut rng);
        let p1 = points::<G1Projective>(n, &mut rng);
        let p2 = points::<G2Projective>(n, &mut rng);
        let sw: Vec<u32> = s.iter().flat_map(scalar_words).collect();
        let bs = g.buffer(&sw);
        g.sort(&mut sc, MsmBuf::new(&bs), n as u32).unwrap();
        for first in [0usize, 1, 6, 1000, n - 1, n] {
            let m = n - first;
            let b1 = g.buffer(&words::<G1>(&p1[..m]));
            let b2 = g.buffer(&words::<G2>(&p2[..m]));
            let (got1, _) = g
                .msm_sorted::<G1>(&sc, MsmBuf::new(&b1), first as u32)
                .unwrap();
            let (got2, _) = g
                .msm_sorted::<G2>(&sc, MsmBuf::new(&b2), first as u32)
                .unwrap();
            let want1: G1Projective = ark_msm(&p1[..m], &s[first..]);
            let want2: G2Projective = ark_msm(&p2[..m], &s[first..]);
            assert_eq!(
                got1.into_affine(),
                want1.into_affine(),
                "G1 round {round} first {first}"
            );
            assert_eq!(
                got2.into_affine(),
                want2.into_affine(),
                "G2 round {round} first {first}"
            );
        }
    }
}

/// Every MSM kernel name is in the generated `METAL_KERNEL_NAMES`; prints the pipeline creation
/// time (minutes when the OS shader cache is cold for this metallib, then well under a second).
#[test]
fn msm_kernel_names_listed() {
    for name in MSM_KERNEL_NAMES {
        assert!(crate::METAL_KERNEL_NAMES.contains(name), "{name}");
    }
    let t0 = Instant::now();
    if gpu().is_some() {
        eprintln!("MSM pipelines built in {:.2} s", t0.elapsed().as_secs_f64());
    }
}

/// R17 audit: `R17_AUDIT_REPS` (default 10) G1 + G2 MSMs of 2^`R17_AUDIT_LG` (default 20) terms
/// on one input, each checked against ark (check_size asserts every rep). msm_reduce.metal has
/// the scanner's tier-B shape (Xyzz accumulators updated by calls in loops).
#[test]
#[ignore]
fn r17_msm_repeat() {
    let Some(g) = gpu() else { return };
    let reps: usize = std::env::var("R17_AUDIT_REPS").map_or(10, |v| v.parse().unwrap());
    let lg: u32 = std::env::var("R17_AUDIT_LG").map_or(20, |v| v.parse().unwrap());
    check_size(&g, 1 << lg, 17, MsmConfig::default(), reps);
    println!("R17AUDIT groth16 msm G1+G2 2^{lg}: {reps}/{reps} reps equal ark (0 differ)");
    // G10's open issue (a G2 wbits = 1 result seen wrong once under load): that config, too.
    let tiny = MsmConfig {
        wbits: 1,
        seg: 3,
        ..Default::default()
    };
    check_size(&g, 1000, 200 + 2, tiny, reps);
    println!(
        "R17AUDIT groth16 msm G1+G2 n 1000 wbits 1 seg 3: {reps}/{reps} reps equal ark (0 differ)"
    );
}

/// G12 investigation (ignored): the G10 note's one-off wrong G2 result (n = 1000, wbits = 1,
/// seg = 3) as a repeat test per [MsmTune] variant. `G12_TINY_REPS` (default 50) sorts + MSMs
/// per variant, `G12_TINY_VARIANTS` (default "base,wide,booth,lenperm"; all = G12 default).
/// Prints the number of wrong G1 / G2 results per variant (no assertion).
#[test]
#[ignore]
fn g12_tiny_g2_repeat() {
    let Some(g) = gpu() else { return };
    let reps: usize = std::env::var("G12_TINY_REPS").map_or(50, |v| v.parse().unwrap());
    let variants = std::env::var("G12_TINY_VARIANTS").unwrap_or_else(|_| "g6,g9,all".into());
    let n = std::env::var("G12_TINY_N").map_or(1000usize, |v| v.parse().unwrap());
    let seed = std::env::var("G12_TINY_SEED").map_or(202u64, |v| v.parse().unwrap());
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let p1 = points::<G1Projective>(n, &mut rng);
    let p2 = points::<G2Projective>(n, &mut rng);
    let s = scalars(n, &mut rng);
    let want1: G1Projective = ark_msm(&p1, &s);
    let want2: G2Projective = ark_msm(&p2, &s);
    let b1 = g.buffer(&words::<G1>(&p1));
    let b2 = g.buffer(&words::<G2>(&p2));
    let sw: Vec<u32> = s.iter().flat_map(scalar_words).collect();
    let bs = g.buffer(&sw);
    let cfg = MsmConfig {
        wbits: std::env::var("G12_TINY_WBITS").map_or(1, |v| v.parse().unwrap()),
        seg: std::env::var("G12_TINY_SEG").map_or(3, |v| v.parse().unwrap()),
        ..Default::default()
    };
    let mut sc = MsmScratch::new(&g.device, n as u32, cfg).unwrap();
    for v in variants.split(',') {
        let tune = match v {
            "g6" => MsmTune::none(),
            "g9" => MsmTune {
                lenperm: false,
                lent_groups: false,
                ..MsmTune::all()
            },
            "g9nobooth" => MsmTune {
                booth: false,
                lenperm: false,
                lent_groups: false,
                ..MsmTune::all()
            },
            "nobooth" => MsmTune {
                booth: false,
                ..MsmTune::all()
            },
            "all" => MsmTune::all(),
            other => panic!("unknown variant {other}"),
        };
        sc.set_tune(tune);
        let (mut bad1, mut bad2) = (0, 0);
        // Per rep, the G2 window sums as affine points (the Xyzz words depend on the
        // nondeterministic order inside a bucket); at the end the reps whose result is wrong are
        // listed with the windows that differ from the majority value.
        let mut all_w: Vec<Vec<G2Affine>> = Vec::new();
        #[allow(clippy::type_complexity)]
        let mut stage_rows: Vec<(Vec<(G2Affine, u32)>, Vec<G2Affine>)> = Vec::new();
        let mut wrong_reps = Vec::new();
        for rep in 0..reps {
            g.sort(&mut sc, MsmBuf::new(&bs), n as u32).unwrap();
            let got1 = if std::env::var("G12_TINY_MODE").as_deref() == Ok("skipg1") {
                want1
            } else {
                g.msm_sorted::<G1>(&sc, MsmBuf::new(&b1), 0).unwrap().0
            };
            let nw = sc.windows() as usize;
            let out = g.device.new_buffer(
                nw as u64 * G2::XYZZ_BYTES,
                MTLResourceOptions::StorageModeShared,
            );
            // G12_TINY_MODE: "one" (default) one command buffer per MSM; "split" one per
            // dispatch; "skipg1" no G1 MSM before the G2 one.
            let mode = std::env::var("G12_TINY_MODE").unwrap_or_else(|_| "one".into());
            if mode == "split" {
                for st in g
                    .kernels
                    .msm_stages::<G2>(&sc, MsmBuf::new(&b2), 0, MsmBuf::new(&out))
                {
                    g.run(|enc| {
                        g.kernels.encode_msm_stage::<G2>(
                            enc,
                            &sc,
                            MsmBuf::new(&b2),
                            MsmBuf::new(&out),
                            st,
                        )
                    })
                    .unwrap();
                }
            } else {
                g.run(|enc| {
                    g.kernels
                        .encode_msm::<G2>(enc, &sc, MsmBuf::new(&b2), 0, MsmBuf::new(&out))
                })
                .unwrap();
            }
            let (snw, wbits) = sc.sorted_layout().unwrap();
            let per = 4 * G2::COORD_WORDS;
            // Safety: completed; shared buffer of at least snw * per words.
            let w: Vec<u32> = unsafe {
                std::slice::from_raw_parts(out.contents() as *const u32, snw as usize * per)
            }
            .to_vec();
            let got2 = G2::combine_windows(&w, wbits);
            bad1 += (got1.into_affine() != want1.into_affine()) as usize;
            let wrong2 = got2.into_affine() != want2.into_affine();
            bad2 += wrong2 as usize;
            if wrong2 {
                wrong_reps.push(rep);
            }
            all_w.push(
                (0..snw as usize)
                    .map(|k| G2::xyzz_to_projective(&w[k * per..(k + 1) * per]).into_affine())
                    .collect(),
            );
            // With RISC0_G16_METAL_SCRATCH_SHARED=1 (and one window group): the pieces and the
            // segment partials of every window, as affine points.
            if std::env::var("RISC0_G16_METAL_SCRATCH_SHARED").as_deref() == Ok("1") {
                let (pieces, pb, partial, mp, ms) = sc.debug_buffers();
                let read = |b: &BufferRef, words: usize| -> Vec<u32> {
                    // Safety: shared scratch, the command buffer completed.
                    unsafe { std::slice::from_raw_parts(b.contents() as *const u32, words) }
                        .to_vec()
                };
                let pw = read(pieces, snw as usize * mp as usize * per);
                let pbw = read(pb, snw as usize * mp as usize);
                let qw = read(partial, snw as usize * ms as usize * per);
                let at = |v: &[u32], i: usize| {
                    G2::xyzz_to_projective(&v[i * per..(i + 1) * per]).into_affine()
                };
                let mut row = Vec::new();
                for k in 0..snw as usize {
                    for j in 0..2usize.min(mp as usize) {
                        row.push((at(&pw, k * mp as usize + j), pbw[k * mp as usize + j]));
                    }
                }
                let parts: Vec<G2Affine> = (0..snw as usize)
                    .map(|k| at(&qw, k * ms as usize))
                    .collect();
                stage_rows.push((row, parts));
            }
        }
        let nwin = all_w[0].len();
        let majority: Vec<G2Affine> = (0..nwin)
            .map(|k| {
                let mut best = (all_w[0][k], 0);
                for cand in all_w.iter().map(|r| r[k]) {
                    let c = all_w.iter().filter(|r| r[k] == cand).count();
                    if c > best.1 {
                        best = (cand, c);
                    }
                }
                best.0
            })
            .collect();
        let mut bad_windows = 0;
        for (rep, wins) in all_w.iter().enumerate() {
            let dw: Vec<usize> = (0..nwin).filter(|&k| wins[k] != majority[k]).collect();
            bad_windows += dw.len();
            if !dw.is_empty() && bad_windows <= 40 {
                println!(
                    "G12TINY {v} rep {rep}: windows {dw:?} differ from the majority (result wrong: {})",
                    wrong_reps.contains(&rep)
                );
            }
        }
        if !stage_rows.is_empty() {
            // Pieces (first two of each window) and first segment partials in the minority.
            let np = stage_rows[0].0.len();
            let nq = stage_rows[0].1.len();
            for rep in 0..reps {
                let dp: Vec<usize> = (0..np)
                    .filter(|&i| {
                        let c = stage_rows
                            .iter()
                            .filter(|r| r.0[i] == stage_rows[rep].0[i])
                            .count();
                        c * 2 < reps
                    })
                    .collect();
                let dq: Vec<usize> = (0..nq)
                    .filter(|&i| {
                        let c = stage_rows
                            .iter()
                            .filter(|r| r.1[i] == stage_rows[rep].1[i])
                            .count();
                        c * 2 < reps
                    })
                    .collect();
                if !dp.is_empty() || !dq.is_empty() {
                    println!(
                        "G12TINY {v} rep {rep}: minority pieces {:?} (window, slot, bucket), \
                         minority partials (windows) {dq:?}",
                        dp.iter()
                            .map(|&i| (i / 2, i % 2, stage_rows[rep].0[i].1))
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
        let diff_reps = (0..reps)
            .filter(|&r| (0..nwin).any(|k| all_w[r][k] != majority[k]))
            .count();
        let bad_host = wrong_reps
            .iter()
            .filter(|&&r| (0..nwin).all(|k| all_w[r][k] == majority[k]))
            .count();
        println!(
            "G12TINY {v} {cfg:?} n {n}: G1 {bad1}/{reps} wrong, G2 {bad2}/{reps} wrong; \
             G2 window sums differ from the majority in {diff_reps} reps ({bad_windows} windows), \
             majority sums but wrong in {bad_host}"
        );
    }
}

/// R17 audit / item G12: msm_window_reduce (G2) alone, re-run `R17_AUDIT_REPS` (default 300)
/// times on one MSM's segment partials (n 1000, wbits 1, seg 3: 129 windows), each result
/// normalized and compared with the majority; fails on any minority window. With the point
/// additions out of line (before G12) this found 58 of 18000 re-runs wrong while a prover ran
/// concurrently (G12 note); run it under such a load to exercise it.
#[test]
#[ignore]
fn r17_msm_window_reduce_repeat() {
    let Some(g) = gpu() else { return };
    let reps: usize = std::env::var("R17_AUDIT_REPS")
        .ok()
        .or_else(|| std::env::var("G12_WR_REPS").ok())
        .map_or(300, |v| v.parse().unwrap());
    let n = std::env::var("G12_TINY_N").map_or(1000usize, |v| v.parse().unwrap());
    let mut rng = ChaCha8Rng::seed_from_u64(202);
    let _p1 = points::<G1Projective>(n, &mut rng);
    let p2 = points::<G2Projective>(n, &mut rng);
    let s = scalars(n, &mut rng);
    let b2 = g.buffer(&words::<G2>(&p2));
    let sw: Vec<u32> = s.iter().flat_map(scalar_words).collect();
    let bs = g.buffer(&sw);
    let cfg = MsmConfig {
        wbits: std::env::var("G12_TINY_WBITS").map_or(1, |v| v.parse().unwrap()),
        seg: std::env::var("G12_TINY_SEG").map_or(3, |v| v.parse().unwrap()),
        ..Default::default()
    };
    let mut sc = MsmScratch::new(&g.device, n as u32, cfg).unwrap();
    g.sort(&mut sc, MsmBuf::new(&bs), n as u32).unwrap();
    let nw = sc.windows() as usize;
    let out = g.device.new_buffer(
        nw as u64 * G2::XYZZ_BYTES,
        MTLResourceOptions::StorageModeShared,
    );
    let ob = MsmBuf::new(&out);
    let stages = g.kernels.msm_stages::<G2>(&sc, MsmBuf::new(&b2), 0, ob);
    g.run(|enc| {
        for st in &stages {
            g.kernels
                .encode_msm_stage::<G2>(enc, &sc, MsmBuf::new(&b2), ob, *st)
        }
    })
    .unwrap();
    let wr = *stages.last().unwrap();
    let (snw, _) = sc.sorted_layout().unwrap();
    let per = 4 * G2::COORD_WORDS;
    let mut runs: Vec<Vec<G2Affine>> = Vec::new();
    for _ in 0..reps {
        g.run(|enc| {
            g.kernels
                .encode_msm_stage::<G2>(enc, &sc, MsmBuf::new(&b2), ob, wr)
        })
        .unwrap();
        // Safety: completed; shared.
        let w =
            unsafe { std::slice::from_raw_parts(out.contents() as *const u32, snw as usize * per) };
        runs.push(
            (0..snw as usize)
                .map(|k| G2::xyzz_to_projective(&w[k * per..(k + 1) * per]).into_affine())
                .collect(),
        );
    }
    let mut bad = 0;
    let mut bad_runs = 0;
    for r in 0..reps {
        let d: Vec<usize> = (0..snw as usize)
            .filter(|&k| runs.iter().filter(|x| x[k] == runs[r][k]).count() * 2 < reps)
            .collect();
        bad += d.len();
        bad_runs += !d.is_empty() as usize;
        if !d.is_empty() && bad_runs <= 10 {
            println!("G12WR run {r}: windows {d:?} in the minority");
        }
    }
    println!(
        "G12WR {cfg:?} n {n}: {bad_runs}/{reps} window_reduce runs with a minority window \
         ({bad} windows of {snw} x {reps})"
    );
    println!(
        "R17AUDIT groth16 msm_window_reduce G2 129 windows: {bad_runs}/{reps} reps differ from the \
         majority"
    );
    assert_eq!(bad_runs, 0, "msm_window_reduce is not deterministic");
}

/// G12: `lent_groups` groups the windows by a lent piece buffer's whole length. With a 1-byte
/// cap (one window per group) and a lent buffer that holds every window, the MSM runs as one
/// group with the switch and as one group per window without; both equal ark, for G1 and G2,
/// with and without the length order.
#[test]
fn msm_lent_groups() {
    let Some(g) = gpu() else { return };
    let mut rng = ChaCha8Rng::seed_from_u64(0x612);
    let n = 2000usize;
    let p1 = points::<G1Projective>(n, &mut rng);
    let p2 = points::<G2Projective>(n, &mut rng);
    let s = scalars(n, &mut rng);
    let want1: G1Projective = ark_msm(&p1, &s);
    let want2: G2Projective = ark_msm(&p2, &s);
    let b1 = g.buffer(&words::<G1>(&p1));
    let b2 = g.buffer(&words::<G2>(&p2));
    let sw: Vec<u32> = s.iter().flat_map(scalar_words).collect();
    let bs = g.buffer(&sw);
    let cfg = MsmConfig {
        wbits: 8,
        max_piece_bytes: 1,
        ..Default::default()
    };
    let one_group = MsmScratch::sizes_for(n as u32, cfg)[5] * cfg.nwins() as u64;
    let lent = g
        .device
        .new_buffer(one_group, MTLResourceOptions::StorageModePrivate);
    let mut sc = MsmScratch::new_with(&g.device, n as u32, cfg, Some(&lent)).unwrap();
    assert!(sc.pieces_lent());
    for (lent_groups, lenperm) in [(false, false), (true, false), (false, true), (true, true)] {
        sc.set_tune(MsmTune {
            lent_groups,
            lenperm,
            ..MsmTune::all()
        });
        g.sort(&mut sc, MsmBuf::new(&bs), n as u32).unwrap();
        let out = g
            .device
            .new_buffer(1 << 16, MTLResourceOptions::StorageModeShared);
        let groups = g
            .kernels
            .msm_stages::<G2>(&sc, MsmBuf::new(&b2), 0, MsmBuf::new(&out))
            .iter()
            .filter(|st| matches!(st, MsmStage::Buckets(_)))
            .count();
        let nw = sc.sorted_layout().unwrap().0 as usize;
        assert_eq!(
            groups,
            if lent_groups { 1 } else { nw },
            "lent_groups {lent_groups}"
        );
        let what = format!("lent_groups {lent_groups} lenperm {lenperm}");
        let (got1, _) = g.msm_sorted::<G1>(&sc, MsmBuf::new(&b1), 0).unwrap();
        assert_eq!(got1.into_affine(), want1.into_affine(), "G1 {what}");
        let (got2, _) = g.msm_sorted::<G2>(&sc, MsmBuf::new(&b2), 0).unwrap();
        assert_eq!(got2.into_affine(), want2.into_affine(), "G2 {what}");
    }
}

/// G9 (G12: with `lenperm`): every [MsmTune] combination, with scalars declared below 2^256 and below 2^254 (the
/// prover's canonical Fr, fewer Booth windows), against ark, on one scratch re-sorted for each.
/// Scalars include Booth edge patterns: windows of all ones, digits of exactly +-2^16 at c = 17,
/// every window's top bit set, and 2^253 - 1.
#[test]
fn msm_tune_variants() {
    let Some(g) = gpu() else { return };
    let mut rng = ChaCha8Rng::seed_from_u64(0x99);
    let n = 3000usize;
    let p1 = points::<G1Projective>(n, &mut rng);
    let p2 = points::<G2Projective>(n, &mut rng);
    let mut s = scalars(n, &mut rng);
    let two = Fr::from(2u64);
    let pow2 = |k: u64| two.pow([k]);
    let mut edges = vec![
        pow2(253) - Fr::one(),
        pow2(253),
        pow2(252) + pow2(16),
        (0..15).map(|k| pow2(17 * k + 16)).sum::<Fr>(),
        (0..15)
            .map(|k| pow2(17 * k + 16) * Fr::from(3u64))
            .sum::<Fr>(),
        pow2(17) - Fr::one(),
        pow2(16),
        pow2(16) - Fr::one(),
        pow2(15),
        pow2(34) - Fr::one(),
        -Fr::one(),
        -two,
    ];
    for k in 0..15 {
        edges.push(pow2(17 * k) * (pow2(17) - Fr::one()));
        edges.push(pow2(17 * k + 16));
    }
    for (i, e) in edges.into_iter().enumerate() {
        s[7 * i + 1] = e;
    }
    let want1: G1Projective = ark_msm(&p1, &s);
    let want2: G2Projective = ark_msm(&p2, &s);
    let b1 = g.buffer(&words::<G1>(&p1));
    let b2 = g.buffer(&words::<G2>(&p2));
    let sw: Vec<u32> = s.iter().flat_map(scalar_words).collect();
    let bs = g.buffer(&sw);
    for cfg in [MsmConfig::default(), MsmConfig::for_size(n as u32)] {
        let mut sc = MsmScratch::new(&g.device, n as u32, cfg).unwrap();
        for bits in [256, 254] {
            sc.set_scalar_bits(bits);
            for (wide_mul, booth, lenperm) in [
                (false, false, false),
                (true, false, false),
                (false, true, false),
                (true, true, false),
                (true, false, true),
                (true, true, true),
                (false, true, true),
            ] {
                sc.set_tune(MsmTune {
                    wide_mul,
                    booth,
                    lenperm,
                    lent_groups: lenperm,
                });
                g.sort(&mut sc, MsmBuf::new(&bs), n as u32).unwrap();
                let what =
                    format!("{cfg:?} bits {bits} wide {wide_mul} booth {booth} lenperm {lenperm}");
                let (nw, _) = sc.sorted_layout().unwrap();
                let expect_nw = if booth {
                    (bits + 1).div_ceil(cfg.wbits + 1)
                } else {
                    cfg.nwins()
                };
                assert_eq!(nw, expect_nw, "{what}");
                let (got1, _) = g.msm_sorted::<G1>(&sc, MsmBuf::new(&b1), 0).unwrap();
                assert_eq!(got1.into_affine(), want1.into_affine(), "G1 {what}");
                let (got2, _) = g.msm_sorted::<G2>(&sc, MsmBuf::new(&b2), 0).unwrap();
                assert_eq!(got2.into_affine(), want2.into_affine(), "G2 {what}");
                // C-style offset (the first 1000 scalars skipped).
                let (got, _) = g.msm_sorted::<G1>(&sc, MsmBuf::new(&b1), 1000).unwrap();
                let want: G1Projective = ark_msm(&p1[..n - 1000], &s[1000..]);
                assert_eq!(got.into_affine(), want.into_affine(), "G1 first {what}");
            }
        }
    }
}

/// G11: PSOs loaded from the binary archive run as fast as plainly built ones and give the same
/// results. One process, both kernel sets on one device, interleaved A B / B A pairs (A = plain
/// `MsmKernels::new`, B = `MsmKernels::from_source` on archive hits), sort + G1 + G2 MSM of
/// 2^`G11_MSM_LOG` (default 20) terms per arm, `G11_PAIRS` (default 9) pairs; every MSM is checked
/// against arkworks. Prints the medians and the per-pair B - A deltas (GPU ms).
#[test]
#[ignore]
fn g11_archived_msm_same_speed() {
    use crate::metal::pipelines::{archived_psos, build_archived, PsoSource};
    let _lock = super::super::test_lock::exclusive();
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let env = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let n = 1usize << env("G11_MSM_LOG", 20);
    let pairs = env("G11_PAIRS", 9);

    let lib = device.new_library_with_data(crate::METAL_LIB).unwrap();
    let plain = MsmGpu {
        device: device.clone(),
        queue: device.new_command_queue(),
        kernels: MsmKernels::new(&device, &lib).unwrap(),
    };
    let dir = std::env::temp_dir().join(format!("g11-msm-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    build_archived(&device, Some(dir.clone()), crate::METAL_LIB).unwrap();
    let ak = build_archived(&device, Some(dir.clone()), crate::METAL_LIB).unwrap();
    assert_eq!(
        ak.report.hits,
        archived_psos().len(),
        "second load not all hits"
    );
    let src = PsoSource::archived(&device, &ak);
    let archived = MsmGpu {
        device: device.clone(),
        queue: device.new_command_queue(),
        kernels: MsmKernels::from_source(&device, ak.library(), &src).unwrap(),
    };
    assert_eq!(src.outside(), 0);
    let _ = std::fs::remove_dir_all(&dir);

    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let p1 = points::<G1Projective>(n, &mut rng);
    let p2 = points::<G2Projective>(n, &mut rng);
    let s = scalars(n, &mut rng);
    let want1: G1Projective = ark_msm(&p1, &s);
    let want2: G2Projective = ark_msm(&p2, &s);
    let b1 = plain.buffer(&words::<G1>(&p1));
    let b2 = plain.buffer(&words::<G2>(&p2));
    let bs = plain.buffer(&s.iter().flat_map(scalar_words).collect::<Vec<_>>());
    let mut scratch = [
        MsmScratch::new(&device, n as u32, MsmConfig::default()).unwrap(),
        MsmScratch::new(&device, n as u32, MsmConfig::default()).unwrap(),
    ];
    let gpus = [&plain, &archived];
    // [arm][pair] = (sort + G1 + G2, G2) GPU ms
    let mut t = [vec![], vec![]];
    for pair in 0..=pairs {
        let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
        for arm in order {
            let g = gpus[arm];
            let sc = &mut scratch[arm];
            let ts = g.sort(sc, MsmBuf::at(&bs, 0), n as u32).unwrap();
            let (got1, t1) = g.msm_sorted::<G1>(sc, MsmBuf::at(&b1, 0), 0).unwrap();
            let (got2, t2) = g.msm_sorted::<G2>(sc, MsmBuf::at(&b2, 0), 0).unwrap();
            assert_eq!(got1.into_affine(), want1.into_affine(), "G1 arm {arm}");
            assert_eq!(got2.into_affine(), want2.into_affine(), "G2 arm {arm}");
            if pair > 0 {
                // pair 0 is a warm-up
                t[arm].push(((ts + t1 + t2) * 1e3, t2 * 1e3));
            }
        }
    }
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        (v[v.len() / 2], v[0], v[v.len() - 1])
    };
    for (what, f) in [
        ("total", (|x: &(f64, f64)| x.0) as fn(&(f64, f64)) -> f64),
        ("G2", |x: &(f64, f64)| x.1),
    ] {
        let a: Vec<f64> = t[0].iter().map(f).collect();
        let b: Vec<f64> = t[1].iter().map(f).collect();
        let d: Vec<f64> = a.iter().zip(&b).map(|(a, b)| b - a).collect();
        let slower = d.iter().filter(|&&x| x > 0.0).count();
        println!(
            "G11 msm n=2^{} {what}: plain {:.2?} ms, archived {:.2?} ms, archived - plain per pair \
             {:.2?} ms (median, min, max), archived slower {slower}/{}",
            n.trailing_zeros(),
            med(a),
            med(b),
            med(d.clone()),
            d.len()
        );
    }
}
