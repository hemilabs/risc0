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

//! Z2 (NTT) conformance: every NTT entry point of the ported `ntt_x` path against the CPU HAL,
//! bit for bit. (Until Z7 the legacy Metal kernels were a third oracle; they were retired once
//! this two-way equality and the DualHal suites held.)

use std::time::Instant;

use rand::Rng;
use risc0_core::field::{
    baby_bear::{BabyBearElem, P},
    Elem as _, RootsOfUnity,
};
use test_log::test;

use super::{
    ntt::{test_hooks, NttTables, NTT_X_KERNELS},
    MetalHalPoseidon2, MetalHalSha256, KERNEL_NAMES,
};
use crate::{
    core::log2_ceil,
    hal::{cpu::CpuHal, testutil, Buffer, Hal},
    INV_RATE,
};

type Fp = BabyBearElem;

const COLS: [usize; 3] = [1, 3, 211];

fn random_vec(n: usize) -> Vec<Fp> {
    let mut rng = rand::rng();
    (0..n).map(|_| Fp::new(rng.random_range(0..P))).collect()
}

fn cpu() -> CpuHal<risc0_core::field::baby_bear::BabyBear> {
    CpuHal::new(crate::core::hash::sha::Sha256HashSuite::new_suite())
}

fn new_hal() -> MetalHalSha256 {
    MetalHalSha256::new()
}

#[track_caller]
fn assert_same(what: &str, lg: usize, cols: usize, got: &[Fp], want: &[Fp]) {
    assert_eq!(got.len(), want.len(), "{what} lg={lg} cols={cols}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!(
            "{what} lg={lg} cols={cols}: first difference at {i} (col {}, row {}): {:?} != {:?}",
            i >> lg.min(63),
            i & ((1 << lg) - 1),
            got[i],
            want[i]
        );
    }
}

/// The input/output lgs of `batch_expand_into_evaluate_ntt` for an output domain of `2^lg`.
fn expand_shape(lg: usize) -> (usize, usize) {
    let e = lg.min(log2_ceil(INV_RATE));
    (lg - e, e)
}

/// Run `f` on the CPU and on Metal with the same data and compare bit for bit.
fn two_way<F>(what: &str, lg: usize, cols: usize, inputs: &[Vec<Fp>], f: F)
where
    F: Fn(&dyn NttRunner, &[Vec<Fp>]) -> Vec<Vec<Fp>>,
{
    let want = f(&Runner(cpu()), inputs);
    let got = f(&Runner(new_hal()), inputs);
    for (k, w) in want.iter().enumerate() {
        assert_same(&format!("{what}[{k}] metal"), lg, cols, &got[k], w);
    }
}

/// Object-safe view of the Hal NTT entry points over host vectors.
trait NttRunner {
    fn expand(&self, input: &[Fp], cols: usize, out_len: usize) -> Vec<Fp>;
    fn interpolate(&self, io: &[Fp], cols: usize) -> Vec<Fp>;
    fn interpolate_zk(&self, io: &[Fp], cols: usize) -> Vec<Fp>;
    /// (dst, src after the call)
    fn interpolate_zk_from(&self, src: &[Fp], cols: usize) -> (Vec<Fp>, Vec<Fp>);
    fn zk_shift(&self, io: &[Fp], cols: usize) -> Vec<Fp>;
    fn bit_reverse(&self, io: &[Fp], cols: usize) -> Vec<Fp>;
    /// (coeffs, evaluated, witness after the call); unfused composition when the HAL has no
    /// fused commit for this size.
    fn commit(&self, witness: &[Fp], cols: usize) -> (Vec<Fp>, Vec<Fp>, Vec<Fp>);
}

struct Runner<H: Hal>(H);

impl<H: Hal<Elem = Fp>> NttRunner for Runner<H> {
    fn expand(&self, input: &[Fp], cols: usize, out_len: usize) -> Vec<Fp> {
        run_expand(&self.0, input, cols, out_len)
    }
    fn interpolate(&self, io: &[Fp], cols: usize) -> Vec<Fp> {
        let b = self.0.copy_from_elem("io", io);
        self.0.batch_interpolate_ntt(&b, cols);
        b.to_vec()
    }
    fn interpolate_zk(&self, io: &[Fp], cols: usize) -> Vec<Fp> {
        let b = self.0.copy_from_elem("io", io);
        self.0.batch_interpolate_ntt_zk_shift(&b, cols);
        b.to_vec()
    }
    fn interpolate_zk_from(&self, src: &[Fp], cols: usize) -> (Vec<Fp>, Vec<Fp>) {
        run_zk_from(&self.0, src, cols)
    }
    fn zk_shift(&self, io: &[Fp], cols: usize) -> Vec<Fp> {
        let b = self.0.copy_from_elem("io", io);
        self.0.zk_shift(&b, cols);
        b.to_vec()
    }
    fn bit_reverse(&self, io: &[Fp], cols: usize) -> Vec<Fp> {
        let b = self.0.copy_from_elem("io", io);
        self.0.batch_bit_reverse(&b, cols);
        b.to_vec()
    }
    fn commit(&self, witness: &[Fp], cols: usize) -> (Vec<Fp>, Vec<Fp>, Vec<Fp>) {
        run_commit(&self.0, witness, cols)
    }
}

fn run_expand<H: Hal<Elem = Fp>>(hal: &H, input: &[Fp], cols: usize, out_len: usize) -> Vec<Fp> {
    let i = hal.copy_from_elem("input", input);
    let o = hal.alloc_elem("output", out_len);
    let e = log2_ceil(out_len / input.len());
    hal.batch_expand_into_evaluate_ntt(&o, &i, cols, e);
    assert!(i.to_vec() == input, "expand changed its input");
    o.to_vec()
}

fn run_zk_from<H: Hal<Elem = Fp>>(hal: &H, src: &[Fp], cols: usize) -> (Vec<Fp>, Vec<Fp>) {
    let s = hal.copy_from_elem("src", src);
    let d = hal.alloc_elem("dst", src.len());
    hal.batch_interpolate_ntt_zk_shift_from(&d, &s, cols);
    (d.to_vec(), s.to_vec())
}

fn run_commit<H: Hal<Elem = Fp>>(
    hal: &H,
    witness: &[Fp],
    cols: usize,
) -> (Vec<Fp>, Vec<Fp>, Vec<Fp>) {
    let n = witness.len() / cols;
    let po2 = log2_ceil(n);
    let w = hal.copy_from_elem("witness", witness);
    let c = hal.alloc_elem("coeffs", witness.len());
    let e = hal.alloc_elem("evaluated", witness.len() * INV_RATE);
    if hal.has_commit_coeffs_lde(po2) {
        hal.commit_coeffs_lde(&c, &e, &w, cols);
    } else {
        hal.batch_interpolate_ntt_zk_shift_from(&c, &w, cols);
        hal.batch_expand_into_evaluate_ntt(&e, &c, cols, log2_ceil(INV_RATE));
        hal.batch_bit_reverse(&c, cols);
    }
    (c.to_vec(), e.to_vec(), w.to_vec())
}

// ---- tables and host logic ----

#[test]
fn ntt_tables_match_sppark_layout() {
    let t = NttTables::get();
    for (dir, roots) in [(0, Fp::ROU_FWD), (1, Fp::ROU_REV)] {
        let tab = &t.dirs[dir];
        let r27 = roots[27];
        // partial_twiddles[off][tid] = r27^(tid << 6 off)
        for off in 0..5 {
            for tid in [0usize, 1, 2, 31, 63] {
                assert_eq!(
                    tab[off * 64 + tid],
                    r27.pow(tid << (6 * off)),
                    "partial {off} {tid}"
                );
            }
        }
        // blob: radix11 at 0 (1024), ..., radix6 at 1984 (32): radix k entry t = root_k^t
        let blob = 320;
        for (k, off) in [
            (11usize, 0usize),
            (10, 1024),
            (9, 1536),
            (8, 1792),
            (7, 1920),
            (6, 1984),
        ] {
            for t in [0usize, 1, 3, (1 << (k - 1)) - 1] {
                assert_eq!(tab[blob + off + t], roots[k].pow(t), "radix{k} [{t}]");
            }
        }
        // plus_one[off][t] = r27^(bitrev10(t) << off)
        let p1 = 320 + 2016;
        for off in [0usize, 1, 5, 11] {
            for t in [0usize, 1, 2, 512, 1023] {
                let rev = (t as u32).reverse_bits() as usize >> 22;
                assert_eq!(
                    tab[p1 + off * 1024 + t],
                    r27.pow(rev << off),
                    "plus_one {off} {t}"
                );
            }
        }
        assert_eq!(tab.len(), p1 + 12 * 1024);
    }
    // zk tables: lo[t] * hi[u] = 3^(t + 1024 u)
    for lg in [1usize, 5, 10, 11, 16, 23] {
        let z = test_hooks::zk_table(lg);
        let nhi = if lg > 10 { 1 << (lg - 10) } else { 1 };
        assert_eq!(z.len(), 1024 + nhi);
        let g = Fp::new(3);
        for k in [0usize, 1, 7, 1023, 1024, 4097, (1 << lg) - 1] {
            if k >= 1 << lg {
                continue;
            }
            assert_eq!(
                z[k & 1023] * z[1024 + (k >> 10)],
                g.pow(k),
                "zk lg={lg} k={k}"
            );
        }
    }
}

#[test]
fn ntt_host_logic() {
    for lg in 1..=27 {
        let ct = test_hooks::ct_steps(lg);
        let gs = test_hooks::gs_steps(lg);
        assert_eq!(ct.iter().sum::<usize>(), lg);
        assert_eq!(gs.iter().sum::<usize>(), lg);
        assert!(
            ct.iter().chain(&gs).all(|&s| s <= 10),
            "lg {lg}: {ct:?} {gs:?}"
        );
    }
    assert_eq!(test_hooks::ct_steps(20), vec![6, 6, 8]);
    assert_eq!(test_hooks::gs_steps(20), vec![8, 6, 6]);
    assert_eq!(test_hooks::ct_steps(17), vec![9, 8]);
    assert_eq!(test_hooks::gs_steps(17), vec![8, 9]);
    assert_eq!(test_hooks::nttx_auto_chunk(20, 2), 4);
    assert_eq!(test_hooks::nttx_auto_chunk(10, 2), 64);
    for name in NTT_X_KERNELS {
        assert!(
            KERNEL_NAMES.contains(name),
            "{name} missing from KERNEL_NAMES"
        );
    }
    let hal = new_hal();
    for name in [
        "ct_x_1",
        "ct_x_8",
        "ct_x_8_coalesced",
        "gs_x_1",
        "gs_x_8",
        "gs_x_8_coalesced",
    ] {
        assert_eq!(
            hal.pipeline(name).max_total_threads_per_threadgroup(),
            512,
            "{name}"
        );
    }
    assert!(
        hal.pipeline("bit_rev_z_batched_64")
            .max_total_threads_per_threadgroup()
            >= 64
    );
    assert!(hal.has_commit_coeffs_lde(10) && hal.has_commit_coeffs_lde(23));
    assert!(!hal.has_commit_coeffs_lde(9) && !hal.has_commit_coeffs_lde(24));
}

// ---- CPU / Metal equality for every entry point, lg 1..20 ----

fn for_shapes(mut f: impl FnMut(usize, usize)) {
    for cols in COLS {
        for lg in 1..=20 {
            f(lg, cols);
        }
    }
}

#[test]
fn ntt_expand_into_evaluate_lg_1_20() {
    for_shapes(|lg, cols| {
        let (lg_in, _) = expand_shape(lg);
        let input = vec![random_vec(cols << lg_in)];
        two_way("expand", lg, cols, &input, |r, i| {
            vec![r.expand(&i[0], cols, cols << lg)]
        });
    });
}

#[test]
fn ntt_interpolate_lg_1_20() {
    for_shapes(|lg, cols| {
        let io = vec![random_vec(cols << lg)];
        two_way("interpolate", lg, cols, &io, |r, i| {
            vec![r.interpolate(&i[0], cols)]
        });
    });
}

#[test]
fn ntt_interpolate_zk_shift_lg_1_20() {
    for_shapes(|lg, cols| {
        let io = vec![random_vec(cols << lg)];
        two_way("interpolate_zk", lg, cols, &io, |r, i| {
            vec![r.interpolate_zk(&i[0], cols)]
        });
    });
}

#[test]
fn ntt_interpolate_zk_shift_from_lg_1_20() {
    for_shapes(|lg, cols| {
        let io = vec![random_vec(cols << lg)];
        two_way("interpolate_zk_from", lg, cols, &io, |r, i| {
            let (dst, src) = r.interpolate_zk_from(&i[0], cols);
            assert!(src == i[0], "lg={lg} cols={cols}: src changed");
            vec![dst]
        });
    });
}

#[test]
fn ntt_zk_shift_lg_1_20() {
    for_shapes(|lg, cols| {
        let io = vec![random_vec(cols << lg)];
        two_way("zk_shift", lg, cols, &io, |r, i| {
            vec![r.zk_shift(&i[0], cols)]
        });
    });
}

#[test]
fn ntt_bit_reverse_lg_1_20() {
    for_shapes(|lg, cols| {
        let io = vec![random_vec(cols << lg)];
        two_way("bit_reverse", lg, cols, &io, |r, i| {
            vec![r.bit_reverse(&i[0], cols)]
        });
    });
}

/// batch_expand_into_evaluate_ntt at every expand ratio other than log2(INV_RATE): 0, 1 (the
/// skip_expand case supra ntt_x.cuh:123 gets wrong), 3, 5 (largest skip), 6 and 7 (synthesized
/// zero padding, no skip).
#[test]
fn ntt_expand_ratios_lg_1_20() {
    for e in [0usize, 1, 3, 5, 6, 7] {
        for cols in COLS {
            for lg in e.max(1)..=20 {
                if cols == 211 && lg > 12 && lg != 16 && lg != 20 {
                    continue;
                }
                let input = vec![random_vec(cols << (lg - e))];
                two_way(&format!("expand by {e}"), lg, cols, &input, |r, i| {
                    vec![r.expand(&i[0], cols, cols << lg)]
                });
            }
        }
    }
}

/// commit_coeffs_lde over its whole lg range up to 20 (fused on Metal; unfused CPU).
#[test]
fn ntt_commit_coeffs_lde_lg_10_20() {
    for cols in COLS {
        for lg in 10..=20 {
            let w = vec![random_vec(cols << lg)];
            two_way("commit", lg, cols, &w, |r, i| {
                let (c, e, w_after) = r.commit(&i[0], cols);
                assert!(w_after == i[0], "lg={lg} cols={cols}: witness changed");
                vec![c, e]
            });
        }
    }
}

/// nttx_commit with explicit chunk sizes (column chunks bound by buffer offset) at po2 20.
#[test]
fn ntt_commit_chunks_po2_20() {
    let (lg, cols) = (20, 9);
    let witness = random_vec(cols << lg);
    let (want_c, want_e, _) = Runner(cpu()).commit(&witness, cols);
    let hal = new_hal();
    for chunk in [0usize, 1, 2, 4, 9] {
        let w = hal.copy_from_elem("witness", &witness);
        let c = hal.alloc_elem("coeffs", witness.len());
        let e = hal.alloc_elem("evaluated", witness.len() * INV_RATE);
        test_hooks::commit_chunked(&hal, &c, &e, &w, lg, cols, chunk);
        assert_same(
            &format!("commit chunk {chunk} coeffs"),
            lg,
            cols,
            &c.to_vec(),
            &want_c,
        );
        assert_same(
            &format!("commit chunk {chunk} evaluated"),
            lg + 2,
            cols,
            &e.to_vec(),
            &want_e,
        );
    }
}

// ---- CUDA-only testutil shapes, enabled on Metal ----

#[test]
fn batch_expand_into_evaluate_ntt_large() {
    testutil::batch_expand_into_evaluate_ntt_large(new_hal());
}

/// The 13 CUDA shapes, including po2 22.
#[test]
fn commit_coeffs_lde() {
    testutil::commit_coeffs_lde(new_hal());
}

#[test]
fn zk_shift_testutil() {
    testutil::zk_shift(MetalHalPoseidon2::new());
}

/// The `_large` shapes at po2 22 (expand to 2^24, interpolate, fused zk-shift, bit reversal).
#[test]
fn ntt_large_po2_22() {
    let (lg, cols) = (22, 1);
    let input = vec![random_vec(cols << lg)];
    two_way("expand", lg + 2, cols, &input, |r, i| {
        vec![r.expand(&i[0], cols, cols << (lg + 2))]
    });
    two_way("interpolate_zk_from", lg, cols, &input, |r, i| {
        vec![r.interpolate_zk_from(&i[0], cols).0]
    });
    two_way("bit_reverse", lg, cols, &input, |r, i| {
        vec![r.bit_reverse(&i[0], cols)]
    });
    let input = vec![random_vec(2 << lg)];
    two_way("interpolate", lg, 2, &input, |r, i| {
        vec![r.interpolate(&i[0], 2)]
    });
}

// ---- indicative timings (not asserted; run with --ignored --nocapture) ----

fn time_ms(hal: &MetalHalSha256, reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    hal.sync();
    let t = Instant::now();
    for _ in 0..reps {
        f();
        hal.sync();
    }
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

#[test]
#[ignore]
fn ntt_timings() {
    let hal = new_hal();
    for (lg, cols) in [(16usize, 211usize), (20, 211), (20, 9), (21, 3)] {
        let n = cols << lg;
        let a = hal.copy_from_elem("a", &random_vec(n));
        let b = hal.alloc_elem("b", n);
        let e = hal.alloc_elem("e", n * INV_RATE);
        let small = hal.copy_from_elem("small", &random_vec(n));
        let r = 3;
        let t_exp = time_ms(&hal, r, || {
            hal.batch_expand_into_evaluate_ntt(&e, &small, cols, 2)
        });
        let t_int = time_ms(&hal, r, || hal.batch_interpolate_ntt(&a, cols));
        let t_izk = time_ms(&hal, r, || hal.batch_interpolate_ntt_zk_shift(&a, cols));
        let t_from = time_ms(&hal, r, || {
            hal.batch_interpolate_ntt_zk_shift_from(&b, &a, cols)
        });
        let t_zk = time_ms(&hal, r, || hal.zk_shift(&a, cols));
        let t_br = time_ms(&hal, r, || hal.batch_bit_reverse(&a, cols));
        let t_commit = if hal.has_commit_coeffs_lde(lg) {
            time_ms(&hal, r, || hal.commit_coeffs_lde(&b, &e, &a, cols))
        } else {
            f64::NAN
        };
        println!(
            "TIMING ntt_x lg={lg} cols={cols}: expand(lg{}) {t_exp:.2} ms, interpolate {t_int:.2}, \
             interpolate_zk {t_izk:.2}, interpolate_zk_from {t_from:.2}, zk_shift {t_zk:.2}, \
             bit_reverse {t_br:.2}, commit_coeffs_lde {t_commit:.2}",
            lg + 2
        );
        // per kernel (one CT/GS pass or one bit reversal, synced on its own)
        for (ct, steps) in [
            (true, test_hooks::ct_steps(lg)),
            (false, test_hooks::gs_steps(lg)),
        ] {
            let mut stage = if ct { 0 } else { lg };
            for it in steps {
                let t = time_ms(&hal, r, || {
                    test_hooks::step(&hal, ct, &a, lg, cols, stage, it)
                });
                println!(
                    "TIMING ntt_x pass {} lg={lg} cols={cols} stage={stage} iterations={it}: {t:.2} ms",
                    if ct { "ct" } else { "gs" }
                );
                if ct {
                    stage += it;
                } else {
                    stage -= it;
                }
            }
        }
        let t = time_ms(&hal, r, || test_hooks::bitrev(&hal, &a, lg, cols));
        println!("TIMING ntt_x bit_rev_z_batched_64 lg={lg} cols={cols}: {t:.2} ms");
    }
}
