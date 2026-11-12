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

//! Z9 (optimization round): interleaved A/B measurements (`--ignored`, run alone) and the
//! bit-exactness tests of the kept optimizations. Every measurement alternates the variants
//! (A B .. / .. B A per pair), times encode + sync of each sample on the wall clock, and reports
//! the median, min and max per variant plus the median of the per-pair ratios to variant 0 (the
//! ratio of two adjacent samples cancels most of the slow drift other agents' GPU work causes).
//! Knobs: `Z9_PAIRS` (default 9), `Z9_PO2` (default 20), `Z9_COLS` (default 211).

use std::time::Instant;

use rand::Rng;
use risc0_core::field::baby_bear::{BabyBearElem, P};

use super::{MetalHalPoseidon2, MetalHash};
use crate::hal::{Buffer as _, Hal};

type Fp = BabyBearElem;

pub(crate) fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

pub(crate) fn random_vec(n: usize) -> Vec<Fp> {
    let mut rng = rand::rng();
    (0..n).map(|_| Fp::new(rng.random_range(0..P))).collect()
}

fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    (s[s.len() / 2], s[0], s[s.len() - 1])
}

fn load_avg() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// The GPU's "Device Utilization %" (IORegistry, averaged by the driver over a short window), or
/// None if unavailable.
pub(crate) fn gpu_utilization() -> Option<u32> {
    let out = std::process::Command::new("ioreg")
        .args(["-r", "-d", "1", "-c", "IOAccelerator"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let key = "\"Device Utilization %\"=";
    let i = text.find(key)? + key.len();
    text[i..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

/// Wait (up to `Z9_IDLE_WAIT_S`, default 120 s) until the GPU utilization is below
/// `Z9_IDLE_PCT` (default 60): other agents' GPU work otherwise inflates samples several-fold.
/// Returns the seconds waited.
pub(crate) fn wait_gpu_quiet() -> f64 {
    let max: f64 = env_or("Z9_IDLE_WAIT_S", 120.0);
    let pct: u32 = env_or("Z9_IDLE_PCT", 60);
    let t0 = Instant::now();
    let mut quiet = 0;
    while t0.elapsed().as_secs_f64() < max {
        match gpu_utilization() {
            Some(u) if u >= pct => quiet = 0,
            _ => {
                quiet += 1;
                if quiet >= 2 {
                    break;
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    t0.elapsed().as_secs_f64()
}

/// A named A/B variant: one sample per call.
pub(crate) type Variant<'a> = (String, Box<dyn FnMut() + 'a>);

/// Interleaved A/B: each closure runs one sample (encode, then this function syncs `hal`; a
/// closure that uses another HAL syncs it itself). Returns the
/// median ms per variant and prints the table under `title`.
pub(crate) fn ab<MH: MetalHash>(
    hal: &super::MetalHal<MH>,
    title: &str,
    pairs: usize,
    variants: &mut [Variant<'_>],
) -> Vec<f64> {
    // Warm-up (PSOs, tables, first touch of the buffers).
    for (_, f) in variants.iter_mut() {
        f();
        hal.sync();
    }
    let k = variants.len();
    let mut t = vec![Vec::with_capacity(pairs); k];
    let mut waited = 0.0;
    for p in 0..pairs {
        waited += wait_gpu_quiet();
        let order: Vec<usize> = if p % 2 == 0 {
            (0..k).collect()
        } else {
            (0..k).rev().collect()
        };
        // Discarded warm-up: the GPU clocks down during the idle wait.
        (variants[order[0]].1)();
        hal.sync();
        for i in order {
            let t0 = Instant::now();
            (variants[i].1)();
            hal.sync();
            t[i].push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    println!(
        "Z9AB {title} ({pairs} pairs, loadavg {}, waited {waited:.0} s for a quiet GPU)",
        load_avg()
    );
    let mut meds = Vec::new();
    for i in 0..k {
        let (m, lo, hi) = stats(&t[i]);
        let r: Vec<f64> = (0..pairs).map(|p| t[0][p] / t[i][p]).collect();
        let (rm, rlo, rhi) = stats(&r);
        println!(
            "Z9AB   {:34} median {m:9.3} ms [min {lo:9.3}, max {hi:9.3}]  speedup vs {}: x{rm:.3} [{rlo:.3}..{rhi:.3}]",
            variants[i].0, variants[0].0
        );
        meds.push(m);
    }
    meds
}

/// NTT commit chunking (`nttx_auto_chunk`: CUDA sizes a chunk to ~96 MB of L2).
#[test]
#[ignore = "measurement"]
fn z9_ntt_commit_chunk_sweep() {
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let lg: usize = env_or("Z9_PO2", 20);
    let cols: usize = env_or("Z9_COLS", 211);
    let pairs = env_or("Z9_PAIRS", 9);
    let n = cols << lg;
    let w = hal.copy_from_elem("witness", &random_vec(n));
    let c = hal.alloc_elem("coeffs", n);
    let e = hal.alloc_elem("evaluated", n * crate::INV_RATE);
    let chunks: Vec<usize> = std::env::var("Z9_CHUNKS")
        .unwrap_or_else(|_| "0,1,2,8,16,32,64".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let mut variants: Vec<(String, Box<dyn FnMut() + '_>)> = chunks
        .iter()
        .map(|&chunk| {
            let (h, c, e, w) = (&hal, &c, &e, &w);
            (
                format!("chunk {chunk}"),
                Box::new(move || {
                    super::ntt::test_hooks::commit_chunked(h, c, e, w, lg, cols, chunk)
                }) as Box<dyn FnMut()>,
            )
        })
        .collect();
    ab(
        &hal,
        &format!("commit_coeffs_lde lg {lg} x {cols} by chunk (0 = auto)"),
        pairs,
        &mut variants,
    );
}

fn time_ms<MH: MetalHash>(hal: &super::MetalHal<MH>, reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    hal.sync();
    let mut v = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        f();
        hal.sync();
        v.push(t.elapsed().as_secs_f64() * 1e3);
    }
    stats(&v).0
}

/// Baseline numbers (plan 4.1 row Z9): empty-dispatch cost, copy / fill GB/s, NTT pass GB/s,
/// Poseidon2 permutations/s. Medians of `Z9_REPS` (default 7).
#[test]
#[ignore = "measurement"]
fn z9_microbench() {
    use super::{KernelArg, Poseidon2Config, Poseidon2Kernels};
    let _lock = super::gpu_test_lock();
    let t_new = Instant::now();
    let hal = MetalHalPoseidon2::new();
    let t_new = t_new.elapsed().as_secs_f64() * 1e3;
    let reps = env_or("Z9_REPS", 7);
    println!(
        "Z9MB loadavg {}; first MetalHal::new in this process (builds {} zkp PSOs, OS shader cache warm) {t_new:.1} ms",
        load_avg(),
        super::KERNEL_NAMES.len()
    );
    wait_gpu_quiet();
    // Empty dispatch: a 1-thread fill_u32.
    let one = hal.alloc_u32("one", 1);
    let fill1 = || {
        hal.launch(
            "fill_u32",
            1,
            &[one.as_arg(), KernelArg::Integer(7), KernelArg::Integer(1)],
        )
    };
    let t1 = time_ms(&hal, reps * 4, fill1);
    let t1000 = time_ms(&hal, reps, || {
        for _ in 0..1000 {
            fill1();
        }
    });
    println!(
        "Z9MB empty dispatch: 1 dispatch + sync {:.1} us; 1000 dispatches + 1 sync {:.2} us per dispatch",
        t1 * 1e3,
        t1000
    );
    // Copy and fill bandwidth.
    for lg in [24usize, 26, 28] {
        let n = 1usize << lg;
        let a = hal.copy_from_elem("a", &random_vec(n));
        let b = hal.alloc_elem("b", n);
        let t = time_ms(&hal, reps, || hal.eltwise_copy_elem(&b, &a));
        let tf = time_ms(&hal, reps, || {
            hal.blit_fill(b.as_buf(), 0, 4 * n as u64, 0xff)
        });
        let tz = time_ms(&hal, reps, || hal.eltwise_zeroize_elem(&b));
        let gb = (4 * n) as f64 / 1e9;
        println!(
            "Z9MB 2^{lg} words ({:.0} MB): eltwise_copy {t:.3} ms = {:.0} GB/s (r+w); blit_fill {tf:.3} ms = {:.0} GB/s; eltwise_zeroize {tz:.3} ms",
            gb * 1e3,
            2.0 * gb / (t / 1e3),
            gb / (tf / 1e3)
        );
    }
    // Heap placement go/no-go: a fresh 1 GiB allocation (pool drained first) + fill vs a pooled
    // one (the steady state of a prover that keeps its HAL).
    {
        let n = 1usize << 28;
        let fresh = time_ms(&hal, reps, || {
            hal.trim_device_memory();
            let b = hal.alloc_elem("fresh", n);
            hal.blit_fill(b.as_buf(), 0, 4 * n as u64, 0);
            hal.sync();
        });
        let pooled = time_ms(&hal, reps, || {
            let b = hal.alloc_elem("pooled", n);
            hal.blit_fill(b.as_buf(), 0, 4 * n as u64, 0);
            hal.sync();
        });
        println!(
            "Z9MB 1 GiB buffer + fill: fresh allocation {fresh:.2} ms, from the pool {pooled:.2} ms"
        );
    }
    // NTT passes (in place, r+w of the whole matrix per pass).
    for (lg, cols) in [(20usize, 211usize), (22, 32)] {
        let n = cols << lg;
        let io = hal.copy_from_elem("io", &random_vec(n));
        let gb = 2.0 * (4 * n) as f64 / 1e9;
        let mut line = format!("Z9MB NTT lg {lg} x {cols}:");
        for ct in [true, false] {
            let steps = if ct {
                super::ntt::test_hooks::ct_steps(lg)
            } else {
                super::ntt::test_hooks::gs_steps(lg)
            };
            let mut stage = 0;
            for &it in &steps {
                let t = time_ms(&hal, reps, || {
                    super::ntt::test_hooks::step(&hal, ct, &io, lg, cols, stage, it)
                });
                line += &format!(
                    " {}({stage},{it}) {t:.2} ms {:.0} GB/s;",
                    if ct { "CT" } else { "GS" },
                    gb / (t / 1e3)
                );
                stage += it;
            }
        }
        let t = time_ms(&hal, reps, || {
            super::ntt::test_hooks::bitrev(&hal, &io, lg, cols)
        });
        line += &format!(" bitrev {t:.2} ms {:.0} GB/s", gb / (t / 1e3));
        println!("{line}");
    }
    // Poseidon2 throughput.
    let rows = 1usize << 20;
    let cols = 211usize;
    let m = hal.copy_from_elem("m", &random_vec(rows * cols));
    let out = hal.alloc_digest("out", rows);
    let fold = hal.alloc_digest("fold", 4 << 22);
    for kernels in [
        Poseidon2Kernels::PvZ9,
        Poseidon2Kernels::Pv,
        Poseidon2Kernels::PvM32,
    ] {
        hal.hash().set_config(Poseidon2Config {
            kernels,
            ..Poseidon2Config::default()
        });
        let t = time_ms(&hal, reps, || hal.hash_rows(&out, &m));
        let tf = time_ms(&hal, reps, || hal.hash_fold(&fold, 2 << 22, 1 << 22));
        println!(
            "Z9MB Poseidon2 {kernels:?}: hash_rows 2^20 x 211 {t:.2} ms = {:.0} Mperm/s; hash_fold 2^22 {tf:.2} ms = {:.0} Mperm/s",
            (rows * cols.div_ceil(16)) as f64 / t / 1e3,
            (1u64 << 22) as f64 / tf / 1e3
        );
    }
    hal.hash().set_config(Poseidon2Config::default());
}

/// Compositions of `lg` into `parts` passes of `lo..=10` iterations.
pub(crate) fn compositions(lg: usize, parts: usize, lo: usize) -> Vec<Vec<usize>> {
    if parts == 1 {
        return if (lo..=10).contains(&lg) {
            vec![vec![lg]]
        } else {
            vec![]
        };
    }
    let mut out = Vec::new();
    for first in lo..=10.min(lg) {
        for mut rest in compositions(lg - first, parts - 1, lo) {
            rest.insert(0, first);
            out.push(rest);
        }
    }
    out
}

/// NTT pass splits: every composition of the forward (LDE, `Z9_LG` + 2 with expand-on-load)
/// and inverse (`Z9_LG`) transforms into 2 or 3 passes of `Z9_MIN_IT..=10` iterations, as
/// `batch_expand_into_evaluate_ntt` / `batch_interpolate_ntt` run them. Variant 0 is sppark's.
#[test]
#[ignore = "measurement"]
fn z9_ntt_split_sweep() {
    use super::ntt::test_hooks;
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let lg: usize = env_or("Z9_LG", 20);
    let cols: usize = env_or("Z9_COLS", 32);
    let pairs = env_or("Z9_PAIRS", 5);
    let lo = env_or("Z9_MIN_IT", 3);
    let n = cols << lg;
    let small = hal.copy_from_elem("small", &random_vec(n));
    let big = hal.alloc_elem("big", 4 * n);
    for ct in [false, true] {
        let tlg = if ct { lg + 2 } else { lg };
        let base = if ct {
            test_hooks::ct_steps(tlg)
        } else {
            test_hooks::gs_steps(tlg)
        };
        let mut splits = vec![base.clone()];
        for parts in 1..=3 {
            for c in compositions(tlg, parts, lo) {
                if c != base && !(ct && c[0] <= 2) {
                    splits.push(c);
                }
            }
        }
        let mut variants: Vec<(String, Box<dyn FnMut() + '_>)> = splits
            .into_iter()
            .map(|split| {
                let (h, small, big) = (&hal, &small, &big);
                let name = format!("{split:?}");
                (
                    name,
                    Box::new(move || {
                        test_hooks::set_split(h, ct, tlg, Some(split.clone()));
                        if ct {
                            h.batch_expand_into_evaluate_ntt(big, small, cols, 2);
                        } else {
                            h.batch_interpolate_ntt(small, cols);
                        }
                    }) as Box<dyn FnMut()>,
                )
            })
            .collect();
        let meds = ab(
            &hal,
            &format!(
                "{} lg {tlg} x {cols} by pass split",
                if ct {
                    "expand+CT (LDE x4)"
                } else {
                    "GS interpolate"
                }
            ),
            pairs,
            &mut variants,
        );
        let best = meds
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        println!("Z9AB   best: {} ({:.3} ms)", variants[best.0].0, best.1);
        test_hooks::set_split(&hal, ct, tlg, None);
    }
}

fn parse_split(s: &str) -> Option<Vec<usize>> {
    if s == "sppark" {
        None
    } else {
        Some(s.split('+').map(|x| x.trim().parse().unwrap()).collect())
    }
}

/// Full `commit_coeffs_lde` (po2 `Z9_PO2` x `Z9_COLS`) for every combination of the chunks in
/// `Z9_CHUNKS` and the (GS, CT) splits in `Z9_SPLITS` (`gs/ct` pairs separated by `;`, each split
/// `a+b+c` or `sppark`).
#[test]
#[ignore = "measurement"]
fn z9_ntt_commit_combo() {
    use super::ntt::test_hooks;
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let lg: usize = env_or("Z9_PO2", 20);
    let cols: usize = env_or("Z9_COLS", 211);
    let pairs = env_or("Z9_PAIRS", 7);
    let n = cols << lg;
    let w = hal.copy_from_elem("witness", &random_vec(n));
    let c = hal.alloc_elem("coeffs", n);
    let e = hal.alloc_elem("evaluated", n * crate::INV_RATE);
    let chunks: Vec<usize> = std::env::var("Z9_CHUNKS")
        .unwrap_or_else(|_| "0,8,16,32".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let splits: Vec<(String, String)> = std::env::var("Z9_SPLITS")
        .unwrap_or_else(|_| "sppark/sppark;10+10/10+6+6".into())
        .split(';')
        .map(|p| {
            let (a, b) = p.split_once('/').unwrap();
            (a.to_string(), b.to_string())
        })
        .collect();
    let mut variants: Vec<(String, Box<dyn FnMut() + '_>)> = Vec::new();
    for (gs, ct) in &splits {
        for &chunk in &chunks {
            let (h, c, e, w) = (&hal, &c, &e, &w);
            let (gsv, ctv) = (parse_split(gs), parse_split(ct));
            variants.push((
                format!("chunk {chunk} gs {gs} ct {ct}"),
                Box::new(move || {
                    test_hooks::set_split(h, false, lg, gsv.clone());
                    test_hooks::set_split(h, true, lg + 2, ctv.clone());
                    test_hooks::commit_chunked(h, c, e, w, lg, cols, chunk)
                }),
            ));
        }
    }
    ab(
        &hal,
        &format!("commit_coeffs_lde lg {lg} x {cols} by chunk and split"),
        pairs,
        &mut variants,
    );
}

// ---- Poseidon2 `_z9` form (default): bit-exact against the CPU HAL ----

mod poseidon2_z9 {
    use rand::{rngs::SmallRng, RngCore, SeedableRng};
    use risc0_core::field::baby_bear::{BabyBear, BabyBearElem};

    use super::super::{
        poseidon2::PvKernel, MetalHalPoseidon2, MetalHashPoseidon2, Poseidon2Config,
        Poseidon2Kernels,
    };
    use crate::{
        core::{digest::Digest, hash::poseidon2::Poseidon2HashSuite},
        hal::{cpu::CpuHal, Buffer, Hal},
    };

    const P: u32 = 0x7800_0001;
    const Z9: Poseidon2Kernels = Poseidon2Kernels::PvZ9;

    fn cpu() -> CpuHal<BabyBear> {
        CpuHal::new(Poseidon2HashSuite::new_suite())
    }

    fn elems(rng: &mut SmallRng, n: usize) -> Vec<BabyBearElem> {
        (0..n)
            .map(|_| BabyBearElem::new_raw(rng.next_u32() % P))
            .collect()
    }

    fn digests(rng: &mut SmallRng, n: usize) -> Vec<Digest> {
        (0..n)
            .map(|_| Digest::from(std::array::from_fn(|_| rng.next_u32() % P)))
            .collect()
    }

    #[test]
    fn default_is_z9() {
        assert_eq!(Poseidon2Config::default().kernels, Z9);
    }

    /// The `_z9` thread-per-hash rows kernel against the CPU, at block / crossover boundaries,
    /// plus extreme inputs (0 and p - 1 everywhere) where the lazy-reduction bounds are tight.
    #[test]
    fn rows_z9_matches_cpu() {
        let hal = MetalHalPoseidon2::new();
        let cpu = cpu();
        let mut rng = SmallRng::seed_from_u64(91);
        for &r in &[1usize, 2, 31, 32, 33, 127, 128, 129, 513, 2049, 65537] {
            for &c in &[1usize, 7, 8, 15, 16, 17, 24, 33, 103, 211, 300] {
                if r * c > 1 << 22 {
                    continue;
                }
                for fill in [None, Some(0u32), Some(P - 1)] {
                    let m = match fill {
                        None => elems(&mut rng, r * c),
                        Some(v) => vec![BabyBearElem::new_raw(v); r * c],
                    };
                    let cm = cpu.copy_from_elem("m", &m);
                    let cout = cpu.alloc_digest("out", r);
                    cpu.hash_rows(&cout, &cm);
                    let mm = hal.copy_from_elem("m", &m);
                    let out = hal.alloc_digest("out", r);
                    MetalHashPoseidon2::encode_rows_form(
                        &hal,
                        PvKernel::Pv,
                        Z9,
                        &out,
                        &mm,
                        r as u32,
                        c as u32,
                    );
                    assert!(
                        out.to_vec() == cout.to_vec(),
                        "rows={r} cols={c} fill={fill:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn fold_and_tail_z9_match_cpu() {
        let hal = MetalHalPoseidon2::new();
        let cpu = cpu();
        let mut rng = SmallRng::seed_from_u64(92);
        for n in [1usize, 2, 3, 31, 32, 33, 127, 128, 129, 1000, 4097, 1 << 16] {
            let io = digests(&mut rng, 4 * n);
            let b = cpu.copy_from_digest("io", &io);
            cpu.hash_fold(&b, 2 * n, n);
            let want = b.to_vec();
            let m = hal.copy_from_digest("io", &io);
            MetalHashPoseidon2::encode_fold_form(&hal, PvKernel::Pv, Z9, &m, n, 2 * n, n as u32);
            let got = m.to_vec();
            assert!(got[n..2 * n] == want[n..2 * n], "n={n}");
            assert!(got[2 * n..] == io[2 * n..], "n={n}: inputs changed");
        }
        for layers in 1..=7usize {
            let io = digests(&mut rng, 2 << layers);
            let b = cpu.copy_from_digest("io", &io);
            cpu.hash_fold_tree(&b, layers);
            let want = b.to_vec();
            let m = hal.copy_from_digest("io", &io);
            MetalHashPoseidon2::encode_fold_tail_form(&hal, Z9, &m, 1 << (layers - 1));
            assert!(m.to_vec()[1..] == want[1..], "tail layers={layers}");
        }
    }

    /// Hal entry points under the default (z9) and the revert knob (`pv`), large shapes.
    #[test]
    fn hal_z9_and_revert_match_cpu() {
        let cpu = cpu();
        let mut rng = SmallRng::seed_from_u64(93);
        let rows = 1usize << 18;
        let cols = 24;
        let m = elems(&mut rng, rows * cols);
        let cm = cpu.copy_from_elem("m", &m);
        let cout = cpu.alloc_digest("out", rows);
        cpu.hash_rows(&cout, &cm);
        let io = digests(&mut rng, 2 << 18);
        let ct = cpu.copy_from_digest("io", &io);
        cpu.hash_fold_tree(&ct, 18);
        for kernels in [Z9, Poseidon2Kernels::Pv] {
            let hal = MetalHalPoseidon2::new();
            hal.hash().set_config(Poseidon2Config {
                kernels,
                ..Poseidon2Config::default()
            });
            let mm = hal.copy_from_elem("m", &m);
            let out = hal.alloc_digest("out", rows);
            hal.hash_rows(&out, &mm);
            assert!(out.to_vec() == cout.to_vec(), "{kernels:?} rows");
            let t = hal.copy_from_digest("io", &io);
            hal.hash_fold_tree(&t, 18);
            assert!(t.to_vec()[1..] == ct.to_vec()[1..], "{kernels:?} tree");
        }
    }
}

/// Interleaved A/B of the Poseidon2 forms at the rv32im data-commit shape (po2 20 LDE rows x 211
/// columns = 2^22 x 211 is the real one; 2^20 x 211 keeps the buffer at 885 MB) and a 2^22 fold.
#[test]
#[ignore = "measurement"]
fn z9_ab_poseidon2() {
    use super::{Poseidon2Config, Poseidon2Kernels};
    let _lock = super::gpu_test_lock();
    let pairs = env_or("Z9_PAIRS", 9);
    let rows = 1usize << env_or("Z9_ROWS_LG", 20usize);
    let cols: usize = env_or("Z9_COLS", 211);
    let forms = [
        Poseidon2Kernels::Pv,
        Poseidon2Kernels::PvZ9,
        Poseidon2Kernels::PvM32,
    ];
    let hals: Vec<MetalHalPoseidon2> = forms
        .iter()
        .map(|&kernels| {
            let h = MetalHalPoseidon2::new();
            h.hash().set_config(Poseidon2Config {
                kernels,
                ..Poseidon2Config::default()
            });
            h
        })
        .collect();
    let m: Vec<_> = hals
        .iter()
        .map(|h| h.copy_from_elem("m", &random_vec(rows * cols)))
        .collect();
    let out: Vec<_> = hals.iter().map(|h| h.alloc_digest("out", rows)).collect();
    let fold: Vec<_> = hals.iter().map(|h| h.alloc_digest("f", 4 << 22)).collect();
    let mut v: Vec<(String, Box<dyn FnMut() + '_>)> = Vec::new();
    for i in 0..forms.len() {
        let (h, m, o) = (&hals[i], &m[i], &out[i]);
        v.push((
            format!("hash_rows {:?}", forms[i]),
            Box::new(move || {
                h.hash_rows(o, m);
                h.sync()
            }),
        ));
    }
    ab(
        &hals[0],
        &format!("hash_rows 2^{} x {cols}", rows.trailing_zeros()),
        pairs,
        &mut v,
    );
    let mut v: Vec<(String, Box<dyn FnMut() + '_>)> = Vec::new();
    for i in 0..forms.len() {
        let (h, f) = (&hals[i], &fold[i]);
        v.push((
            format!("hash_fold_tree 22 {:?}", forms[i]),
            Box::new(move || {
                h.hash_fold_tree(f, 22);
                h.sync()
            }),
        ));
    }
    ab(&hals[0], "hash_fold_tree 22 layers", pairs, &mut v);
}

/// Poseidon2 latency-kernel crossovers (pvs_w / pvs_q vs the thread-per-hash `_z9` kernel), as
/// interleaved A/B per size: `Z9_REPS_PER_SAMPLE` (default 20) dependent dispatches per sample.
#[test]
#[ignore = "measurement"]
fn z9_poseidon2_crossovers() {
    use super::{poseidon2::PvKernel, MetalHashPoseidon2, Poseidon2Kernels};
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let pairs = env_or("Z9_PAIRS", 7);
    let reps: usize = env_or("Z9_REPS_PER_SAMPLE", 20);
    let kernels = [
        (PvKernel::Pv, Poseidon2Kernels::PvZ9, "pv_z9"),
        (PvKernel::PvsW, Poseidon2Kernels::Pv, "pvs_w"),
        (PvKernel::PvsQ, Poseidon2Kernels::Pv, "pvs_q"),
    ];
    let mut summary = Vec::new();
    for lg in 5..=13usize {
        let n = 1usize << lg;
        let io = hal.copy_from_digest(
            "io",
            &(0..4 * n)
                .map(|i| crate::core::digest::Digest::from([i as u32 % P; 8]))
                .collect::<Vec<_>>(),
        );
        let mut v: Vec<(String, Box<dyn FnMut() + '_>)> = kernels
            .iter()
            .map(|&(k, f, name)| {
                let (h, io) = (&hal, &io);
                (
                    format!("fold n=2^{lg} {name}"),
                    Box::new(move || {
                        for _ in 0..reps {
                            MetalHashPoseidon2::encode_fold_form(h, k, f, io, n, 2 * n, n as u32)
                        }
                    }) as Box<dyn FnMut()>,
                )
            })
            .collect();
        let m = ab(&hal, &format!("fold 2^{lg} x{reps}"), pairs, &mut v);
        summary.push(format!(
            "fold 2^{lg}: pv_z9 {:.1} us, pvs_w {:.1} us, pvs_q {:.1} us",
            m[0] * 1e3 / reps as f64,
            m[1] * 1e3 / reps as f64,
            m[2] * 1e3 / reps as f64
        ));
    }
    for cols in [16usize, 211] {
        for lg in 5..=13usize {
            let rows = 1usize << lg;
            let mm = hal.copy_from_elem("m", &random_vec(rows * cols));
            let out = hal.alloc_digest("out", rows);
            let mut v: Vec<(String, Box<dyn FnMut() + '_>)> = kernels
                .iter()
                .map(|&(k, f, name)| {
                    let (h, mm, out) = (&hal, &mm, &out);
                    (
                        format!("rows 2^{lg} x {cols} {name}"),
                        Box::new(move || {
                            for _ in 0..reps {
                                MetalHashPoseidon2::encode_rows_form(
                                    h,
                                    k,
                                    f,
                                    out,
                                    mm,
                                    rows as u32,
                                    cols as u32,
                                )
                            }
                        }) as Box<dyn FnMut()>,
                    )
                })
                .collect();
            let m = ab(
                &hal,
                &format!("rows 2^{lg} x {cols} x{reps}"),
                pairs,
                &mut v,
            );
            summary.push(format!(
                "rows 2^{lg} x {cols}: pv_z9 {:.1} us, pvs_w {:.1} us, pvs_q {:.1} us",
                m[0] * 1e3 / reps as f64,
                m[1] * 1e3 / reps as f64,
                m[2] * 1e3 / reps as f64
            ));
        }
    }
    for line in summary {
        println!("Z9XO {line}");
    }
}

/// eu2 S sweep (plan: "eu2 S sweep"): the ffi.cu rule picks the smallest S with
/// `S * nEvals >= 1024` (CUDA's block target); here every power-of-two S at the rv32im shapes
/// (deg `2^Z9_PO2`, `Z9_POLYS` polynomials, nEvals in `Z9_EVALS`).
#[test]
#[ignore = "measurement"]
fn z9_eu2_s_sweep() {
    use risc0_core::field::baby_bear::BabyBearExtElem;

    use super::eval::{batch_evaluate_any, eval_plan, EvalPath};
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let pairs = env_or("Z9_PAIRS", 7);
    let po2: usize = env_or("Z9_PO2", 20);
    let polys: usize = env_or("Z9_POLYS", 211);
    let deg = 1usize << po2;
    let coeffs = hal.copy_from_elem("coeffs", &random_vec(polys * deg));
    let evals: Vec<usize> = std::env::var("Z9_EVALS")
        .unwrap_or_else(|_| "64,256,500,1000".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let mut rng = rand::rng();
    for n in evals {
        let which: Vec<u32> = (0..n).map(|_| rng.random_range(0..polys as u32)).collect();
        let xs: Vec<BabyBearExtElem> = (0..n)
            .map(|_| {
                let r = random_vec(4);
                BabyBearExtElem::new(r[0], r[1], r[2], r[3])
            })
            .collect();
        let which = hal.copy_from_u32("which", &which);
        let xs = hal.copy_from_extelem("xs", &xs);
        let out = hal.alloc_extelem("out", n);
        let default = eval_plan(deg as u32, n as u32, coeffs.as_device_ptr());
        let mut paths = vec![default];
        let mut s = 1u32;
        while s as usize <= deg / 4096 && s <= 256 {
            let p = EvalPath::Eu2 {
                s,
                chunk: deg as u32 / s,
            };
            if p != default {
                paths.push(p);
            }
            s <<= 1;
        }
        paths.push(EvalPath::Horner);
        let mut v: Vec<(String, Box<dyn FnMut() + '_>)> = paths
            .into_iter()
            .map(|p| {
                let (h, c, w, x, o) = (&hal, &coeffs, &which, &xs, &out);
                (
                    format!("{p:?}"),
                    Box::new(move || {
                        batch_evaluate_any(h, c, polys, w, x, o, Some(p));
                    }) as Box<dyn FnMut()>,
                )
            })
            .collect();
        ab(
            &hal,
            &format!("eval deg 2^{po2} x {polys} polys, {n} evals (variant 0 = ffi.cu rule)"),
            pairs,
            &mut v,
        );
    }
}

/// combos_divide GPU (mdiv) vs the host loop on the same unified-memory buffer, po2 `Z9_LGS`
/// (default 11..=20), 7 slices and 6 chunks of 2,2,2,2,2,1 roots (Z5's timing case). Also E = 8
/// vs E = 16 at po2 18 and 19 (Z5 left those open).
#[test]
#[ignore = "measurement"]
fn z9_mdiv_crossover() {
    use risc0_core::field::baby_bear::BabyBearExtElem;

    use super::mdiv::{mdiv_default_e, mdiv_shape};
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let pairs = env_or("Z9_PAIRS", 7);
    let lgs: Vec<usize> = std::env::var("Z9_LGS")
        .unwrap_or_else(|_| "11,12,13,14,15,16,17,18,19,20".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let ext = |v: &[Fp]| BabyBearExtElem::new(v[0], v[1], v[2], v[3]);
    for lg in lgs {
        let n = 1usize << lg;
        let data: Vec<BabyBearExtElem> = random_vec(4 * 7 * n).chunks(4).map(ext).collect();
        let combos = hal.copy_from_extelem("combos", &data);
        let roots = [2usize, 2, 2, 2, 2, 1];
        let chunks: Vec<(usize, Vec<BabyBearExtElem>)> = roots
            .iter()
            .enumerate()
            .map(|(i, &k)| (i, random_vec(4 * k).chunks(4).map(ext).collect()))
            .collect();
        let e_def = mdiv_default_e(n);
        let mut shapes = vec![(format!("gpu E{e_def}"), mdiv_shape(n, e_def))];
        if (16..=19).contains(&lg) {
            let other = if e_def == 8 { 16 } else { 8 };
            shapes.push((format!("gpu E{other}"), mdiv_shape(n, other)));
        }
        shapes.push(("host loop".into(), None));
        if shapes[0].1.is_none() {
            println!("Z9AB po2 {lg}: no GPU shape (host loop only)");
            continue;
        }
        let mut v: Vec<(String, Box<dyn FnMut() + '_>)> = shapes
            .into_iter()
            .map(|(name, shape)| {
                let (h, c, ch) = (&hal, &combos, &chunks);
                (
                    name,
                    Box::new(move || {
                        h.mdiv_combos_divide_with(c, ch, n, shape);
                    }) as Box<dyn FnMut()>,
                )
            })
            .collect();
        ab(&hal, &format!("combos_divide po2 {lg}"), pairs, &mut v);
    }
}

/// The Z9 mdiv defaults: host loop below po2 16 (the Hal entry point), and the GPU shapes the
/// crossover no longer selects (po2 11..=15, E = 8 and 16) still equal the host loop.
#[test]
fn z9_mdiv_small_po2_gpu_equals_host() {
    use risc0_core::field::baby_bear::BabyBearExtElem;

    use super::mdiv::{mdiv_gpu_size, mdiv_shape};
    let hal = MetalHalPoseidon2::new();
    if std::env::var("RISC0_METAL_MDIV_GPU_MIN_PO2").is_err() {
        assert!(!mdiv_gpu_size(1 << 15) && mdiv_gpu_size(1 << 16));
    }
    let ext = |v: &[Fp]| BabyBearExtElem::new(v[0], v[1], v[2], v[3]);
    for lg in 11..=16usize {
        let n = 1usize << lg;
        let data: Vec<BabyBearExtElem> = random_vec(4 * 5 * n).chunks(4).map(ext).collect();
        let chunks: Vec<(usize, Vec<BabyBearExtElem>)> = [(3usize, 2usize), (0, 1), (4, 3)]
            .iter()
            .map(|&(c, k)| (c, random_vec(4 * k).chunks(4).map(ext).collect()))
            .collect();
        let host = hal.copy_from_extelem("host", &data);
        let want_r = hal.mdiv_combos_divide_with(&host, &chunks, n, None);
        let want = host.to_vec();
        for e in [8, 16] {
            let Some(shape) = mdiv_shape(n, e) else {
                continue;
            };
            let gpu = hal.copy_from_extelem("gpu", &data);
            let got_r = hal.mdiv_combos_divide_with(&gpu, &chunks, n, Some(shape));
            assert!(got_r == want_r, "remainders po2 {lg} E{e}");
            assert!(gpu.to_vec() == want, "data po2 {lg} E{e}");
        }
        // The Hal path (host below po2 16 by default) gives the same.
        let def = hal.copy_from_extelem("def", &data);
        assert!(hal.mdiv_combos_divide(&def, &chunks, n) == want_r);
        assert!(def.to_vec() == want, "default path po2 {lg}");
    }
}

/// Threadgroup width of the 1-D elementwise dispatches (`Grid::Threads`, default 256): copy, add,
/// zeroize over 2^28 words. Changes a process-wide setting: run alone.
#[test]
#[ignore = "measurement"]
fn z9_threads_1d_sweep() {
    use super::runtime::set_threads_1d;
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let pairs = env_or("Z9_PAIRS", 7);
    let n = 1usize << env_or("Z9_LG", 28usize);
    let a = hal.copy_from_elem("a", &random_vec(n));
    let b = hal.copy_from_elem("b", &random_vec(n));
    let c = hal.alloc_elem("c", n);
    for op in ["copy", "add", "zeroize"] {
        let mut v: Vec<(String, Box<dyn FnMut() + '_>)> = [256u64, 64, 128, 512, 1024]
            .iter()
            .map(|&w| {
                let (h, a, b, c) = (&hal, &a, &b, &c);
                (
                    format!("{op} tg {w}"),
                    Box::new(move || {
                        set_threads_1d(w);
                        match op {
                            "copy" => h.eltwise_copy_elem(c, a),
                            "add" => h.eltwise_add_elem(c, a, b),
                            _ => h.eltwise_zeroize_elem(c),
                        }
                    }) as Box<dyn FnMut()>,
                )
            })
            .collect();
        ab(
            &hal,
            &format!("{op} 2^{} words", n.trailing_zeros()),
            pairs,
            &mut v,
        );
    }
    set_threads_1d(256);
}

/// Second queue go/no-go: does the GPU overlap an ALU-bound Poseidon2 `hash_rows` (queue A) with
/// a bandwidth-bound `commit_coeffs_lde` (queue B, a second HAL)? Variants: each alone, both on one
/// queue back to back, and both committed to two queues before either waits.
#[test]
#[ignore = "measurement"]
fn z9_second_queue_overlap() {
    let _lock = super::gpu_test_lock();
    let pairs = env_or("Z9_PAIRS", 7);
    let ha = MetalHalPoseidon2::new();
    let hb = MetalHalPoseidon2::new();
    let lg = 20usize;
    let cols: usize = env_or("Z9_COLS", 64);
    let rows = 1usize << 22;
    // Queue A: hash 2^22 rows x cols (an LDE-sized hash); queue B: commit of 2^20 x cols.
    let m_a = ha.copy_from_elem("m", &random_vec(rows * cols));
    let out_a = ha.alloc_digest("out", rows);
    let w_b = hb.copy_from_elem("w", &random_vec(cols << lg));
    let c_b = hb.alloc_elem("c", cols << lg);
    let e_b = hb.alloc_elem("e", (cols << lg) * crate::INV_RATE);
    // Same work on one queue (HAL A's own buffers).
    let w_a = ha.copy_from_elem("w", &random_vec(cols << lg));
    let c_a = ha.alloc_elem("c", cols << lg);
    let e_a = ha.alloc_elem("e", (cols << lg) * crate::INV_RATE);
    let mut v: Vec<(String, Box<dyn FnMut() + '_>)> = vec![
        (
            "hash alone".into(),
            Box::new(|| {
                ha.hash_rows(&out_a, &m_a);
                ha.sync()
            }),
        ),
        (
            "commit alone".into(),
            Box::new(|| {
                hb.commit_coeffs_lde(&c_b, &e_b, &w_b, cols);
                hb.sync()
            }),
        ),
        (
            "one queue: hash then commit".into(),
            Box::new(|| {
                ha.hash_rows(&out_a, &m_a);
                ha.commit_coeffs_lde(&c_a, &e_a, &w_a, cols);
                ha.sync()
            }),
        ),
        (
            "two queues concurrently".into(),
            Box::new(|| {
                ha.hash_rows(&out_a, &m_a);
                ha.flush();
                hb.commit_coeffs_lde(&c_b, &e_b, &w_b, cols);
                hb.flush();
                ha.sync();
                hb.sync()
            }),
        ),
    ];
    ab(
        &ha,
        &format!("hash 2^22 x {cols} || commit 2^20 x {cols}"),
        pairs,
        &mut v,
    );
}
