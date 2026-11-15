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

//! Z14 (commit fusion, LEADS #8): the coefficient bit reversal of a commit runs inside the
//! Merkle row hash (`pv_rows_side_k_z9`). Bit-exactness against the separate passes and against
//! the CPU prover's commit (coefficients, evaluated domain, Merkle root), with the fusion on and
//! off (`RISC0_METAL_COMMIT_FUSE=0` / [Poseidon2Config::commit_fuse]).

use risc0_core::field::{baby_bear::BabyBearElem as Fp, Elem as _};

use super::{poseidon2::Poseidon2Config, MetalHalPoseidon2, MetalHash as _, MetalHashPoseidon2};
use crate::{
    core::hash::poseidon2::Poseidon2HashSuite,
    hal::{cpu::CpuHal, Buffer as _, Hal},
    prove::poly_group::PolyGroup,
    INV_RATE,
};

fn random_vec(seed: u64, n: usize) -> Vec<Fp> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            Fp::new((state % (risc0_core::field::baby_bear::P as u64)) as u32)
        })
        .collect()
}

fn bit_rev(i: usize, lg: usize) -> usize {
    if lg == 0 {
        0
    } else {
        i.reverse_bits() >> (usize::BITS as usize - lg)
    }
}

fn set_fuse(hal: &MetalHalPoseidon2, on: bool) {
    let hash: &MetalHashPoseidon2 = hal.hash();
    hash.set_config(Poseidon2Config {
        commit_fuse: on,
        ..hash.config()
    });
}

/// `pv_rows_side_k_z9` against `pv_rows` + a host bit reversal: odd column counts, more and fewer
/// tile pairs than threadgroups (the strided unit loop), 2^10 (a single self-paired tile).
#[test]
fn z14_rows_bit_reverse_matches_separate() {
    let hal = MetalHalPoseidon2::new();
    // The fused path under test, whatever RISC0_METAL_COMMIT_FUSE says.
    set_fuse(&hal, true);
    // (lg of the reversed columns, their count, hash rows, hash columns)
    let cases = [
        (10, 1, 4096, 1),
        (11, 2, 4096, 3),
        (12, 1, 4096, 1),
        (12, 3, 16384, 16),
        (13, 17, 4096, 23),
        (14, 5, 1 << 16, 211),
        (16, 5, 4096, 16),
        (16, 211, 1 << 18, 3),
        (18, 23, 1 << 20, 17),
        (20, 4, 1 << 22, 16),
    ];
    for (k, &(lg, count, rows, cols)) in cases.iter().enumerate() {
        assert!(
            hal.fuses_bit_reverse_into_hash(rows, cols, lg),
            "{lg} {rows} {cols}"
        );
        let n = 1usize << lg;
        let coeffs_in = random_vec(17 + k as u64, n * count);
        let matrix_in = random_vec(1001 + k as u64, rows * cols);
        let coeffs = hal.copy_from_elem("coeffs", &coeffs_in);
        let matrix = hal.copy_from_elem("matrix", &matrix_in);
        let fused = hal.alloc_digest("fused", rows);
        let plain = hal.alloc_digest("plain", rows);
        hal.hash_rows_bit_reverse(&fused, &matrix, &coeffs, count);
        hal.hash_rows(&plain, &matrix);
        assert!(
            fused.to_vec() == plain.to_vec(),
            "digests lg {lg} rows {rows} cols {cols}"
        );
        let got = coeffs.to_vec();
        for c in 0..count {
            for i in 0..n {
                assert_eq!(
                    got[c * n + i],
                    coeffs_in[c * n + bit_rev(i, lg)],
                    "lg {lg} count {count}: column {c} index {i}"
                );
            }
        }
    }
}

/// The shapes the fusion declines (pvs row kernels, columns below 2^12, the knob, other hash
/// forms) run the separate passes through the same entry point, with the same results.
#[test]
fn z14_rows_bit_reverse_fallbacks() {
    let hal = MetalHalPoseidon2::new();
    set_fuse(&hal, true);
    assert!(!hal.fuses_bit_reverse_into_hash(2048, 16, 12), "pvs rows");
    assert!(
        !hal.fuses_bit_reverse_into_hash(1 << 14, 16, 9),
        "below one tile pair"
    );
    set_fuse(&hal, false);
    assert!(!hal.fuses_bit_reverse_into_hash(1 << 14, 16, 12), "knob");
    set_fuse(&hal, true);
    for (lg, count, rows, cols) in [(9, 3, 8192, 16), (12, 2, 2048, 5), (4, 1, 4096, 1)] {
        let n = 1usize << lg;
        let coeffs_in = random_vec(5 + lg as u64, n * count);
        let matrix_in = random_vec(77, rows * cols);
        let coeffs = hal.copy_from_elem("coeffs", &coeffs_in);
        let matrix = hal.copy_from_elem("matrix", &matrix_in);
        let fused = hal.alloc_digest("fused", rows);
        let plain = hal.alloc_digest("plain", rows);
        hal.hash_rows_bit_reverse(&fused, &matrix, &coeffs, count);
        hal.hash_rows(&plain, &matrix);
        assert!(fused.to_vec() == plain.to_vec());
        let got = coeffs.to_vec();
        for c in 0..count {
            for i in 0..n {
                assert_eq!(got[c * n + i], coeffs_in[c * n + bit_rev(i, lg)]);
            }
        }
    }
}

/// Coefficients, evaluated domain and every Merkle node of a commit against the CPU prover's
/// (`make_coeffs` + `PolyGroup::new`), through both Metal commit paths (Prover::commit_group's
/// fused commit and PolyGroup::new's, which the check group takes), fusion on and off.
#[test]
fn z14_commit_matches_cpu() {
    let cpu = CpuHal::new(Poseidon2HashSuite::new_suite());
    let hal = MetalHalPoseidon2::new();
    for (po2, cols) in [(10, 1), (12, 16), (13, 17), (16, 3), (18, 16)] {
        let size = 1usize << po2;
        let witness_in = random_vec(po2 as u64 * 31 + cols as u64, size * cols);
        // CPU reference.
        let want = {
            let w = cpu.copy_from_elem("witness", &witness_in);
            let coeffs = cpu.alloc_elem("coeffs", w.size());
            cpu.batch_interpolate_ntt_zk_shift_from(&coeffs, &w, cols);
            let g = PolyGroup::new(&cpu, coeffs, cols, size, "ref");
            (
                g.coeffs.as_ref().unwrap().to_vec(),
                g.evaluated.as_ref().unwrap().to_vec(),
                *g.merkle.root(),
            )
        };
        let check = |what: &str, g: &PolyGroup<MetalHalPoseidon2>| {
            assert!(
                g.coeffs.as_ref().unwrap().to_vec() == want.0,
                "{what}: coeffs po2 {po2} cols {cols}"
            );
            assert!(
                g.evaluated.as_ref().unwrap().to_vec() == want.1,
                "{what}: evaluated po2 {po2} cols {cols}"
            );
            assert_eq!(
                *g.merkle.root(),
                want.2,
                "{what}: root po2 {po2} cols {cols}"
            );
        };
        for fuse in [true, false] {
            set_fuse(&hal, fuse);
            let fuses = hal.fuses_bit_reverse_into_hash(size * INV_RATE, cols, po2);
            assert_eq!(fuses, fuse && size * INV_RATE > 2048, "po2 {po2}");
            let witness = hal.copy_from_elem("witness", &witness_in);
            // Prover::commit_group's path.
            let coeffs = hal.alloc_elem("coeffs", witness.size());
            let evaluated = hal.alloc_elem("evaluated", witness.size() * INV_RATE);
            let g = if fuses {
                hal.commit_coeffs_lde_bit_reversed(&coeffs, &evaluated, &witness, cols);
                PolyGroup::new_with_evaluated_bit_reversed(
                    &hal, coeffs, evaluated, cols, size, "z14",
                )
            } else {
                hal.commit_coeffs_lde(&coeffs, &evaluated, &witness, cols);
                PolyGroup::new_with_evaluated(&hal, coeffs, evaluated, cols, size, "z14")
            };
            check(if fuse { "commit fused" } else { "commit plain" }, &g);
            // PolyGroup::new's path (the check group's).
            let coeffs = hal.alloc_elem("coeffs", witness.size());
            hal.batch_interpolate_ntt_zk_shift_from(&coeffs, &witness, cols);
            let g = PolyGroup::new(&hal, coeffs, cols, size, "z14b");
            check(if fuse { "new fused" } else { "new plain" }, &g);
        }
        set_fuse(&hal, true);
    }
}

/// Fill `buf` in place with pseudo-random field elements (no host copy of the whole buffer: the
/// measurement tests' buffers are GiB-sized, and a host `Vec` plus its device copy doubled their
/// footprint).
fn fill_random(buf: &super::BufferImpl<Fp>, seed: u64) {
    let mut state = seed | 1;
    buf.view_mut(|s| {
        for x in s.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *x = Fp::new((state % (risc0_core::field::baby_bear::P as u64)) as u32);
        }
    });
}

/// Interleaved A/B (`--ignored`): the commit (`commit_coeffs_lde` + row hash) and the check
/// group's (expand NTT + bit reversal + row hash) with the coefficient bit reversal as its own
/// pass (A, twice: the A-A spread) and folded into the hash (B), plus the kernel-level split
/// (hash alone, reversal alone, both separate, fused). GPU time with `RISC0_METAL_PROFILE=1`,
/// else wall time; `fill`: the check group's row hash with the next segment's data / accum fills
/// as blits or folded in. Knobs: `Z9_PAIRS` (7), `Z9_PO2` (20), `Z9_COLS` (211), `Z14_SHAPES`
/// (`commit,check,fill,kernel`), `Z9_IDLE_WAIT_S`.
///
/// Memory: every shape shares one set of buffers (witness, coeffs, evaluated: 5.3 GB at po2 20 x
/// 211, plus 1.3 GB for `fill`), filled in place, and the pool is drained after each shape. Run
/// it under the exclusive GPU lock (`gpu-run`), never beside another prover.
#[test]
#[ignore = "measurement: run alone with --release --ignored --nocapture"]
fn z14_commit_ab() {
    crate::hal::metal::require_gpu_run();
    use super::{tests_z13::ab_gpu, tests_z9::env_or};
    let _lock = super::gpu_test_lock();
    let pairs: usize = env_or("Z9_PAIRS", 7);
    let po2: usize = env_or("Z9_PO2", 20);
    let cols: usize = env_or("Z9_COLS", 211);
    let shapes = std::env::var("Z14_SHAPES").unwrap_or_else(|_| "commit,check,fill,kernel".into());
    let hal = MetalHalPoseidon2::new();
    let size = 1usize << po2;
    let rows = size * INV_RATE;
    let ccols = 16.min(cols);
    let witness = hal.alloc_elem("witness", size * cols);
    fill_random(&witness, 3);
    let coeffs = hal.alloc_elem("coeffs", size * cols);
    let evaluated = hal.alloc_elem("evaluated", rows * cols);
    let nodes = hal.alloc_digest("nodes", rows);
    // Real commit data for the hash-only shapes.
    hal.commit_coeffs_lde(&coeffs, &evaluated, &witness, cols);
    hal.sync();
    if shapes.contains("commit") {
        let plain = || {
            hal.commit_coeffs_lde(&coeffs, &evaluated, &witness, cols);
            hal.hash_rows(&nodes, &evaluated);
        };
        let fused = || {
            hal.commit_coeffs_lde_bit_reversed(&coeffs, &evaluated, &witness, cols);
            hal.hash_rows_bit_reverse(&nodes, &evaluated, &coeffs, cols);
        };
        assert!(hal.fuses_bit_reverse_into_hash(rows, cols, po2));
        ab_gpu(
            &hal,
            &format!("commit + row hash, po2 {po2} x {cols}"),
            pairs,
            &mut [
                ("A separate bitrev".into(), Box::new(plain)),
                ("A2 separate bitrev".into(), Box::new(plain)),
                ("B fused (pv_rows_side)".into(), Box::new(fused)),
            ],
        );
        hal.trim_device_memory();
    }
    // The check group's shapes: the first `ccols` columns of the buffers above.
    let ccoeffs = coeffs.slice(0, size * ccols);
    let cevaluated = evaluated.slice(0, rows * ccols);
    if shapes.contains("check") {
        let plain = || {
            hal.batch_expand_into_evaluate_ntt(&cevaluated, &ccoeffs, ccols, 2);
            hal.batch_bit_reverse(&ccoeffs, ccols);
            hal.hash_rows(&nodes, &cevaluated);
        };
        let fused = || {
            hal.batch_expand_into_evaluate_ntt(&cevaluated, &ccoeffs, ccols, 2);
            hal.hash_rows_bit_reverse(&nodes, &cevaluated, &ccoeffs, ccols);
        };
        ab_gpu(
            &hal,
            &format!("check group: expand + bitrev + row hash, po2 {po2} x {ccols}"),
            pairs,
            &mut [
                ("A separate bitrev".into(), Box::new(plain)),
                ("A2 separate bitrev".into(), Box::new(plain)),
                ("B fused (pv_rows_side)".into(), Box::new(fused)),
            ],
        );
        hal.trim_device_memory();
    }
    if shapes.contains("fill") {
        // The check group's row hash (its bit reversal folded in), with the next segment's data
        // and accum fills (211 and 103 columns) as blits or folded in.
        let data = hal.alloc_elem("data", size * 211);
        let accum = hal.alloc_elem("accum", size * 103);
        let fills = [
            (data.as_buf().to_owned(), (size * 211 * 4) as u64, 0u32),
            (accum.as_buf().to_owned(), (size * 103 * 4) as u64, u32::MAX),
        ];
        let blits = || {
            hal.fill_u32(&data, 0);
            hal.fill_u32(&accum, u32::MAX);
        };
        let side = |f: &[(metal::Buffer, u64, u32)]| {
            MetalHashPoseidon2::encode_rows_side(
                &hal,
                &nodes,
                &cevaluated,
                Some((&ccoeffs, po2, ccols)),
                f,
                super::poseidon2::PV_SIDE_TG_BYTES,
            )
        };
        // `Z14_INNER` back-to-back repetitions per sample keep the GPU clocked up.
        let inner: usize = env_or("Z14_INNER", 1);
        let rep = |f: &dyn Fn()| {
            for _ in 0..inner {
                f();
            }
        };
        ab_gpu(
            &hal,
            &format!(
                "check row hash + fills {} MB, po2 {po2}, x{inner} per sample",
                (size * 314 * 4) >> 20
            ),
            pairs,
            &mut [
                (
                    "A hash, fills as blits".into(),
                    Box::new(|| {
                        rep(&|| {
                            blits();
                            side(&[]);
                        })
                    }),
                ),
                (
                    "A2 hash, fills as blits".into(),
                    Box::new(|| {
                        rep(&|| {
                            blits();
                            side(&[]);
                        })
                    }),
                ),
                (
                    "B fills in the hash".into(),
                    Box::new(|| rep(&|| side(&fills))),
                ),
                ("hash alone".into(), Box::new(|| rep(&|| side(&[])))),
                ("blits alone".into(), Box::new(|| rep(&blits))),
            ],
        );
        drop((data, accum));
        hal.trim_device_memory();
    }
    if shapes.contains("kernel") {
        ab_gpu(
            &hal,
            &format!("kernels: rows {rows} x {cols}, bitrev {cols} x 2^{po2}"),
            pairs,
            &mut [
                (
                    "hash + bitrev, separate".into(),
                    Box::new(|| {
                        hal.hash_rows(&nodes, &evaluated);
                        hal.batch_bit_reverse(&coeffs, cols);
                    }),
                ),
                (
                    "hash only".into(),
                    Box::new(|| hal.hash_rows(&nodes, &evaluated)),
                ),
                (
                    "bitrev only".into(),
                    Box::new(|| hal.batch_bit_reverse(&coeffs, cols)),
                ),
                (
                    "fused".into(),
                    Box::new(|| hal.hash_rows_bit_reverse(&nodes, &evaluated, &coeffs, cols)),
                ),
                (
                    "fused kernel, no reversal".into(),
                    Box::new(|| {
                        MetalHashPoseidon2::encode_rows_side(
                            &hal,
                            &nodes,
                            &evaluated,
                            None,
                            &[],
                            super::poseidon2::PV_SIDE_TG_BYTES,
                        )
                    }),
                ),
            ],
        );
    }
}

/// Occupancy probe (`--ignored`): the row hash alone against `pv_rows_side_k_z9` with no reversal
/// work and `Z14_TG` (comma list) bytes of threadgroup memory. GPU time minima need
/// `RISC0_METAL_PROFILE=1`. 3.5 GB at po2 20 x 211 (filled in place): run under the exclusive
/// GPU lock.
#[test]
#[ignore = "measurement: run alone with --release --ignored --nocapture"]
fn z14_tg_probe() {
    crate::hal::metal::require_gpu_run();
    use super::{tests_z13::ab_gpu, tests_z9::env_or};
    let _lock = super::gpu_test_lock();
    let pairs: usize = env_or("Z9_PAIRS", 7);
    let po2: usize = env_or("Z9_PO2", 20);
    let cols: usize = env_or("Z9_COLS", 211);
    let tgs: Vec<u64> = std::env::var("Z14_TG")
        .unwrap_or_else(|_| "16,8448,16640".into())
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect();
    let hal = MetalHalPoseidon2::new();
    let rows = (1usize << po2) * INV_RATE;
    let evaluated = hal.alloc_elem("evaluated", rows * cols);
    fill_random(&evaluated, 6);
    let nodes = hal.alloc_digest("nodes", rows);
    let mut variants: Vec<super::tests_z9::Variant<'_>> = vec![(
        "hash only".into(),
        Box::new(|| hal.hash_rows(&nodes, &evaluated)),
    )];
    for &tg in &tgs {
        let (hal, nodes, evaluated) = (&hal, &nodes, &evaluated);
        variants.push((
            format!("br kernel, no units, tg {tg}"),
            Box::new(move || {
                MetalHashPoseidon2::encode_rows_side(hal, nodes, evaluated, None, &[], tg)
            }),
        ));
    }
    ab_gpu(
        &hal,
        &format!("tg probe rows {rows} x {cols}"),
        pairs,
        &mut variants,
    );
}

/// `prefill_elem` + `alloc_elem_init` (Z14): the set-aside buffer comes back filled whether its
/// fill rode on a row hash, was never encoded (filled at allocation), or its last GPU user had not
/// completed when it was set aside; a pool drain and the budget cap release / refuse it; values
/// with equal bytes (blit) and unequal bytes (fill kernel).
#[test]
fn z14_prefill_round_trip() {
    // Sets this HAL's working-set budget and trims its pool: take the lock like other tests that
    // touch budgets.
    let _lock = super::gpu_test_lock();
    if !super::prefill_enabled() {
        eprintln!("z14_prefill_round_trip: RISC0_METAL_PREFILL=0, skipped");
        return;
    }
    let hal = MetalHalPoseidon2::new();
    // Fills ride on the fused hash, whatever RISC0_METAL_COMMIT_FUSE says.
    set_fuse(&hal, true);
    let n = 3 << 18;
    let (rows, cols) = (1usize << 14, 16usize);
    let matrix = hal.copy_from_elem("m", &random_vec(9, rows * cols));
    let nodes = hal.alloc_digest("nodes", rows);
    let want_nodes = {
        hal.hash_rows(&nodes, &matrix);
        nodes.to_vec()
    };
    for value in [Fp::INVALID, Fp::ZERO, Fp::new(12345)] {
        for case in ["hash", "hash_br", "unfilled", "pending", "drain", "budget"] {
            // A buffer of n elements back in the pool, dirty.
            let first = hal.copy_from_elem("dirty", &random_vec(3, n));
            if case == "pending" {
                // Its last user still in flight when it is set aside.
                let fold = hal.alloc_digest("fold", n / 16);
                hal.hash().hash_rows(&hal, &fold, &first);
            } else {
                hal.sync();
            }
            drop(first);
            if case == "budget" {
                hal.set_working_set_limit(Some((n as u64 * 4) * 8 - 16));
            }
            hal.prefill_elem(n, value);
            if case == "budget" {
                hal.set_working_set_limit(None);
            }
            let held = if case == "budget" { 0 } else { 1 };
            assert_eq!(hal.rt.prefill_stats(), (held, 0), "{case}: set aside");
            if case == "drain" {
                hal.trim_device_memory();
                assert_eq!(hal.rt.prefill_stats(), (0, 0), "{case}: released");
            }
            if matches!(case, "hash" | "pending") {
                hal.hash_rows(&nodes, &matrix);
                assert_eq!(hal.rt.prefill_stats(), (1, 1), "{case}: fill encoded");
                assert!(nodes.to_vec() == want_nodes, "{case}: digests");
            }
            if case == "hash_br" {
                // With a bit reversal in the same dispatch.
                let lg = 12;
                let coeffs_in = random_vec(21, 3 << lg);
                let coeffs = hal.copy_from_elem("coeffs", &coeffs_in);
                hal.hash_rows_bit_reverse(&nodes, &matrix, &coeffs, 3);
                assert_eq!(hal.rt.prefill_stats(), (1, 1), "{case}: fill encoded");
                assert!(nodes.to_vec() == want_nodes, "{case}: digests");
                let got = coeffs.to_vec();
                for (i, &x) in got.iter().enumerate() {
                    let (c, k) = (i >> lg, i & ((1 << lg) - 1));
                    assert_eq!(x, coeffs_in[(c << lg) + bit_rev(k, lg)], "{case}: coeffs");
                }
            }
            let got = hal.alloc_elem_init("data", n, value);
            assert_eq!(hal.rt.prefill_stats(), (0, 0), "{case}: taken");
            // Bit patterns, not `==`: Fp's PartialEq debug-asserts validity, and INVALID is not.
            assert!(
                got.to_vec()
                    .iter()
                    .all(|x| x.as_u32_montgomery() == value.as_u32_montgomery()),
                "{case} {value:?}: not filled"
            );
            drop(got);
            hal.sync();
        }
    }
}
