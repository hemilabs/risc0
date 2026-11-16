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

//! Z13 (commit NTT, LEADS #3): every sub-track was a NO-GO (docs/metal-port/notes/Z13.md), so
//! nothing here changes a default. What stays is the measurement tooling the note's numbers come
//! from, and a bit-exactness check of the non-sppark pass splits it measured:
//!
//! * [ab_gpu]: interleaved A/B on summed per-dispatch GPU time (`RISC0_METAL_PROFILE=1`), which
//!   is far less disturbed by other processes' GPU work than wall time; wall time otherwise.
//! * `z13_split_ab`: forward / inverse / expand pass-split sweeps (`--ignored`).
//! * `z13_commit_chunk_ab`: commit_coeffs_lde by (forward split, column chunk) (`--ignored`).
//!
//! Run the measurements alone with `--release --ignored --nocapture`. Knobs: `Z9_PAIRS`,
//! `Z9_COLS`, `Z9_PO2`, `Z9_IDLE_WAIT_S`, `Z13_SPLITS`, `Z13_COMBOS`, `Z13_AA`.

use risc0_core::field::Elem as _;

use super::{
    ntt::test_hooks as th,
    tests_z9::{ab, env_or, random_vec, wait_gpu_quiet, Variant},
    MetalHalPoseidon2, MetalHash,
};
use crate::{
    core::ntt as cpu_ntt,
    hal::{Buffer as _, Hal},
};

type Fp = risc0_core::field::baby_bear::BabyBearElem;

/// Interleaved A/B (A B .. / .. B A per pair, a discarded warm-up sample after each idle wait).
/// With `RISC0_METAL_PROFILE=1` every dispatch is committed in a command buffer of its own and a
/// sample is the sum of its dispatches' GPU time (no host gaps; each dispatch pays a few us of
/// command-buffer overhead, so prefer wall time for workloads of hundreds of small dispatches);
/// otherwise [ab] (wall time). Prints median / min / max per variant and the median of the
/// per-pair time ratios to variant 0.
pub(crate) fn ab_gpu<MH: MetalHash>(
    hal: &super::MetalHal<MH>,
    title: &str,
    pairs: usize,
    variants: &mut [Variant<'_>],
) {
    use super::profile_take;
    if std::env::var(super::PROFILE_ENV).as_deref() != Ok("1") {
        ab(hal, title, pairs, variants);
        return;
    }
    let gpu_ms = |f: &mut Box<dyn FnMut() + '_>| {
        profile_take();
        f();
        hal.sync();
        profile_take().iter().map(|(_, e)| e.gpu_secs).sum::<f64>() * 1e3
    };
    for (_, f) in variants.iter_mut() {
        gpu_ms(f);
    }
    let k = variants.len();
    let mut t = vec![Vec::new(); k];
    for p in 0..pairs {
        wait_gpu_quiet();
        let order: Vec<usize> = if p % 2 == 0 {
            (0..k).collect()
        } else {
            (0..k).rev().collect()
        };
        // Discarded warm-up: the GPU clocks down during the idle wait.
        gpu_ms(&mut variants[order[0]].1);
        for i in order {
            t[i].push(gpu_ms(&mut variants[i].1));
        }
    }
    let la = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("Z13GPU {title} ({pairs} pairs, loadavg {la})");
    let st = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        (s[s.len() / 2], s[0], s[s.len() - 1])
    };
    for i in 0..k {
        let (m, lo, hi) = st(&t[i]);
        let r: Vec<f64> = (0..pairs).map(|p| t[i][p] / t[0][p]).collect();
        let (rm, rlo, rhi) = st(&r);
        println!(
            "Z13GPU   {:34} median {m:9.3} ms [min {lo:9.3}, max {hi:9.3}]  time vs {}: x{rm:.3} [{rlo:.3}..{rhi:.3}]",
            variants[i].0, variants[0].0
        );
    }
}

fn parse_split(s: &str) -> Vec<usize> {
    s.split('+').map(|x| x.trim().parse().unwrap()).collect()
}

/// The forward splits the note measured fastest per transform (lg 17..23, kernel level) are
/// valid: expand x4 + CT under each equals sppark's split for 1 and 3 columns, and the CPU up to
/// lg 20. (Not every split is: a CT pass of 11 iterations needs a plus-one twiddle table beyond
/// 2^10 entries, and some splits whose first pass is shorter than 6 iterations are wrong; see
/// the note.)
#[test]
fn z13_measured_ct_splits_are_exact() {
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    for (lg, split) in [
        (17usize, "10+7"),
        (18, "10+8"),
        (19, "9+6+4"),
        (20, "9+7+4"),
        (21, "10+7+4"),
        (22, "10+8+4"),
    ] {
        let split = parse_split(split);
        assert_eq!(split.iter().sum::<usize>(), lg);
        for cols in [1usize, 3] {
            let n = 1usize << (lg - 2);
            let input = random_vec(n * cols);
            let gin = hal.copy_from_elem("in", &input);
            let mut outs = vec![];
            for s in [None, Some(split.clone())] {
                th::set_split(&hal, true, lg, s);
                let a = hal.alloc_elem("a", 4 * n * cols);
                th::ct_expand4(&hal, cols, &gin, &a, lg);
                let b = hal.alloc_elem("b", 4 * n * cols);
                hal.batch_expand_into_evaluate_ntt(&b, &gin, cols, 2);
                outs.push((a.to_vec(), b.to_vec()));
            }
            th::set_split(&hal, true, lg, None);
            assert!(
                outs[0] == outs[1],
                "lg {lg} {split:?} cols {cols}: != sppark"
            );
            assert!(outs[1].0 == outs[1].1, "lg {lg} {split:?}: ct_x != expand");
            if lg <= 20 {
                let mut want = vec![Fp::ZERO; 4 * n * cols];
                for c in 0..cols {
                    let w = &mut want[c * 4 * n..(c + 1) * 4 * n];
                    cpu_ntt::expand(w, &input[c * n..(c + 1) * n], 2);
                    cpu_ntt::evaluate_ntt::<Fp, Fp>(w, 2);
                }
                assert!(outs[1].0 == want, "lg {lg} {split:?} cols {cols}: != cpu");
            }
        }
    }
}

/// Pass-split A/B (ntt_x kernels): `Z13_SPLITS` = `ct|expand|gs:lg:a+b+c/d+e+f/..` items
/// separated by `,`. Variant 0 (and the A-A arm with `Z13_AA`) is sppark's split of that lg. `ct`
/// is the commit's expand x4 + CT and `expand` is batch_expand_into_evaluate_ntt, both of the LDE
/// lg (input lg - 2); `gs` is the inverse of lg. `Z9_COLS` columns (default 211) in one call.
#[test]
#[ignore = "measurement"]
fn z13_split_ab() {
    crate::hal::metal::require_gpu_run();
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let cols: usize = env_or("Z9_COLS", 211);
    let pairs = env_or("Z9_PAIRS", 9);
    let spec = std::env::var("Z13_SPLITS")
        .unwrap_or_else(|_| "ct:22:10+8+4/10+7+5/9+9+4,gs:20:10+10/4+7+9".into());
    for item in spec.split(',') {
        let f: Vec<&str> = item.split(':').collect();
        let kind = f[0];
        let tlg: usize = f[1].parse().unwrap();
        let splits: Vec<Vec<usize>> = f[2].split('/').map(parse_split).collect();
        let ct = kind != "gs";
        let lg = if ct { tlg - 2 } else { tlg };
        let n = 1usize << lg;
        let small = hal.copy_from_elem("small", &random_vec(n * cols));
        let big = hal.alloc_elem("big", if ct { 4 * n * cols } else { n * cols });
        let (h, sr, br) = (&hal, &small, &big);
        let run = move |s: Option<Vec<usize>>| {
            th::set_split(h, ct, tlg, s);
            match kind {
                "ct" => th::ct_expand4(h, cols, sr, br, tlg),
                "expand" => h.batch_expand_into_evaluate_ntt(br, sr, cols, 2),
                _ => th::gs(h, cols, sr, br, lg),
            }
        };
        let base = if ct {
            th::ct_steps(tlg)
        } else {
            th::gs_steps(tlg)
        };
        let mut vs: Vec<Variant> = vec![(
            format!("{kind} lg {tlg} sppark {base:?}"),
            Box::new(move || run(None)),
        )];
        if std::env::var("Z13_AA").is_ok() {
            vs.push((
                format!("{kind} lg {tlg} sppark (A-A)"),
                Box::new(move || run(None)),
            ));
        }
        for s2 in splits {
            vs.push((
                format!("{kind} lg {tlg} {s2:?}"),
                Box::new(move || run(Some(s2.clone()))),
            ));
        }
        ab_gpu(&hal, &format!("{kind} lg {tlg} x {cols}"), pairs, &mut vs);
        th::set_split(&hal, ct, tlg, None);
    }
}

/// commit (`nttx_commit`) at po2 `Z9_PO2` (20) x `Z9_COLS` (211) by (forward split, column
/// chunk[, inverse split]): `Z13_COMBOS` = `split:chunk[:gs_split]` items separated by `,`, splits
/// `sppark` or `a+b+c` (forward: the transform of 2^(po2 + 2); inverse: of 2^po2), chunk 0 =
/// auto. Variant 0 is the first item.
#[test]
#[ignore = "measurement"]
fn z13_commit_chunk_ab() {
    crate::hal::metal::require_gpu_run();
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let po2: usize = env_or("Z9_PO2", 20);
    let cols: usize = env_or("Z9_COLS", 211);
    let pairs = env_or("Z9_PAIRS", 9);
    let n = 1usize << po2;
    let w = hal.copy_from_elem("w", &random_vec(n * cols));
    let c = hal.alloc_elem("c", n * cols);
    let e = hal.alloc_elem("e", 4 * n * cols);
    let spec = std::env::var("Z13_COMBOS")
        .unwrap_or_else(|_| "sppark:0,sppark:0,10+8+4:0,10+8+4:8,10+8+4:211,sppark:211".into());
    let (h, wr, cr, er) = (&hal, &w, &c, &e);
    let mut vs: Vec<Variant> = spec
        .split(',')
        .map(|item| {
            let f: Vec<&str> = item.split(':').collect();
            let split = (f[0] != "sppark").then(|| parse_split(f[0]));
            let chunk: usize = f[1].parse().unwrap();
            let gs = f.get(2).filter(|g| **g != "sppark").map(|g| parse_split(g));
            (
                item.to_string(),
                Box::new(move || {
                    th::set_split(h, true, po2 + 2, split.clone());
                    th::set_split(h, false, po2, gs.clone());
                    th::commit_chunked(h, cr, er, wr, po2, cols, chunk);
                }) as Box<dyn FnMut()>,
            )
        })
        .collect();
    ab_gpu(
        &hal,
        &format!("commit po2 {po2} x {cols} by forward split, chunk"),
        pairs,
        &mut vs,
    );
    th::set_split(&hal, true, po2 + 2, None);
    th::set_split(&hal, false, po2, None);
}
