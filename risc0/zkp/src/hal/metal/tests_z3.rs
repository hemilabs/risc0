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

//! Z3: the `pv.metal` Poseidon2 kernels against the CPU HAL, bit for bit. Every kernel is checked
//! on its own (forced through `encode_rows`/`encode_fold`/`encode_fold_tail`), and through the
//! `Hal` entry points with the default crossovers, the classic oracle, and the `m32` variants.
//! `bench_*` (ignored) are the crossover measurements recorded in docs/metal-port/notes/Z3.md.

use std::time::Instant;

use rand::{rngs::SmallRng, RngCore, SeedableRng};
use risc0_core::field::baby_bear::BabyBearElem;
use test_log::test;

use super::{
    pv_poseidon2::PvKernel, BufferImpl, MetalHalPoseidon2, MetalHalPoseidon2Legacy, MetalHash,
    MetalHashPoseidon2, Poseidon2Config, Poseidon2Kernels,
};
use crate::{
    core::{digest::Digest, hash::poseidon2::Poseidon2HashSuite},
    hal::{cpu::CpuHal, Buffer, Hal},
};

const P: u32 = 0x7800_0001;
/// Every kernel, in both arithmetic forms (`true`: the `_m32` instantiation).
const ALL: [(PvKernel, bool); 6] = [
    (PvKernel::Pv, false),
    (PvKernel::Pv, true),
    (PvKernel::PvsW, false),
    (PvKernel::PvsW, true),
    (PvKernel::PvsQ, false),
    (PvKernel::PvsQ, true),
];

type Cpu = CpuHal<risc0_core::field::baby_bear::BabyBear>;

fn cpu() -> Cpu {
    CpuHal::new(Poseidon2HashSuite::new_suite())
}

fn rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

/// Random field elements (raw Montgomery words in [0, p)).
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

fn cpu_rows(cpu: &Cpu, matrix: &[BabyBearElem], rows: usize) -> Vec<Digest> {
    let m = cpu.copy_from_elem("m", matrix);
    let out = cpu.alloc_digest("out", rows);
    cpu.hash_rows(&out, &m);
    out.to_vec()
}

/// The fold of `n` outputs of `io` (size >= 4n) on the CPU: io[n..2n] = H(io[2n..4n]).
fn cpu_fold(cpu: &Cpu, io: &[Digest], n: usize) -> Vec<Digest> {
    let b = cpu.copy_from_digest("io", io);
    cpu.hash_fold(&b, 2 * n, n);
    b.to_vec()
}

fn cpu_tree(cpu: &Cpu, io: &[Digest], layers: usize) -> Vec<Digest> {
    let b = cpu.copy_from_digest("io", io);
    cpu.hash_fold_tree(&b, layers);
    b.to_vec()
}

fn metal_rows_with(
    hal: &MetalHalPoseidon2,
    (kernel, m32): (PvKernel, bool),
    matrix: &[BabyBearElem],
    rows: usize,
) -> Vec<Digest> {
    let m = hal.copy_from_elem("m", matrix);
    let out = hal.alloc_digest("out", rows);
    let cols = matrix.len() / rows;
    MetalHashPoseidon2::encode_rows(hal, kernel, m32, &out, &m, rows as u32, cols as u32);
    out.to_vec()
}

fn metal_fold_with(
    hal: &MetalHalPoseidon2,
    (kernel, m32): (PvKernel, bool),
    io: &[Digest],
    n: usize,
) -> Vec<Digest> {
    let b = hal.copy_from_digest("io", io);
    MetalHashPoseidon2::encode_fold(hal, kernel, m32, &b, n, 2 * n, n as u32);
    b.to_vec()
}

fn with_config(cfg: Poseidon2Config) -> MetalHalPoseidon2 {
    let hal = MetalHalPoseidon2::new();
    hal.hash().set_config(cfg);
    hal
}

/// Every kernel on its own, for row counts around every block / SIMD-group / crossover boundary
/// and column counts that are and are not multiples of 8 and 16 (incl. 1 and > 256).
#[test]
fn rows_each_kernel() {
    let hal = MetalHalPoseidon2::new();
    let cpu = cpu();
    let mut rng = rng(1);
    let rows = [
        1usize, 2, 3, 7, 31, 32, 33, 127, 128, 129, 255, 513, 1000, 2049,
    ];
    let cols = [
        1usize, 2, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 64, 103, 211, 255, 256, 300,
    ];
    for &r in &rows {
        for &c in &cols {
            let m = elems(&mut rng, r * c);
            let want = cpu_rows(&cpu, &m, r);
            for k in ALL {
                assert!(
                    metal_rows_with(&hal, k, &m, r) == want,
                    "{k:?} rows={r} cols={c}"
                );
            }
        }
    }
}

/// `hash_rows` through the Hal with the default crossovers, rows 1..2^20 and cols 1..300.
#[test]
fn rows_dispatcher_large() {
    let hal = MetalHalPoseidon2::new();
    let cpu = cpu();
    let mut rng = rng(2);
    let shapes: &[(usize, &[usize])] = &[
        (1, &[1, 16, 17, 300]),
        (511, &[1, 9, 300]),
        (512, &[1, 9, 300]),
        (513, &[8, 211]),
        (4095, &[16, 23, 300]),
        (4096, &[16, 23, 300]),
        (4097, &[1, 23, 211]),
        (8191, &[7, 64]),
        (8193, &[7, 64, 300]),
        (65537, &[1, 17, 103]),
        (1 << 18, &[16, 33]),
        (1 << 20, &[1, 9, 24, 33]),
    ];
    for &(r, cs) in shapes {
        for &c in cs {
            let m = elems(&mut rng, r * c);
            let want = cpu_rows(&cpu, &m, r);
            let mb = hal.copy_from_elem("m", &m);
            let out = hal.alloc_digest("out", r);
            hal.hash_rows(&out, &mb);
            assert!(out.to_vec() == want, "rows={r} cols={c}");
        }
    }
}

/// Every fold kernel on its own, including counts that are not powers of two.
#[test]
fn fold_each_kernel() {
    let hal = MetalHalPoseidon2::new();
    let cpu = cpu();
    let mut rng = rng(3);
    for n in [
        1usize, 2, 3, 5, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1000, 2048, 4097, 8193,
    ] {
        let io = digests(&mut rng, 4 * n);
        let want = cpu_fold(&cpu, &io, n);
        for k in ALL {
            let got = metal_fold_with(&hal, k, &io, n);
            assert!(got[n..2 * n] == want[n..2 * n], "{k:?} n={n}");
            assert!(got[2 * n..] == io[2 * n..], "{k:?} n={n}: inputs changed");
        }
    }
}

/// `hash_fold` through the Hal (default crossovers), up to 2^16 outputs.
#[test]
fn fold_dispatcher() {
    let hal = MetalHalPoseidon2::new();
    let cpu = cpu();
    let mut rng = rng(4);
    for n in [1usize, 3, 512, 513, 4096, 4097, 1 << 13, (1 << 16) + 3] {
        let io = digests(&mut rng, 4 * n);
        let want = cpu_fold(&cpu, &io, n);
        let b = hal.copy_from_digest("io", &io);
        hal.hash_fold(&b, 2 * n, n);
        assert!(b.to_vec()[n..2 * n] == want[n..2 * n], "n={n}");
    }
}

fn check_tree(hal: &MetalHalPoseidon2, what: &str, seed: u64) {
    let cpu = cpu();
    let mut rng = rng(seed);
    for layers in 1..=14 {
        let io = digests(&mut rng, 2 << layers);
        let want = cpu_tree(&cpu, &io, layers);
        let b = hal.copy_from_digest("io", &io);
        hal.hash_fold_tree(&b, layers);
        assert!(b.to_vec()[1..] == want[1..], "{what}: layers={layers}");
    }
}

/// `hash_fold_tree` layers 1..14 for every kernel selection.
#[test]
fn fold_tree_configs() {
    let d = Poseidon2Config::default();
    let configs = [
        ("default", d),
        ("tail128", Poseidon2Config { tail_max: 128, ..d }),
        ("tail1", Poseidon2Config { tail_max: 1, ..d }),
        ("no-small", Poseidon2Config { small: false, ..d }),
        (
            "no-small-tail",
            Poseidon2Config {
                small: false,
                tail_max: 128,
                ..d
            },
        ),
        (
            "m32-tail",
            Poseidon2Config {
                kernels: Poseidon2Kernels::PvM32,
                small: false,
                tail_max: 64,
                ..d
            },
        ),
        (
            "all-w",
            Poseidon2Config {
                w_max_fold: u32::MAX,
                ..d
            },
        ),
        (
            "all-q",
            Poseidon2Config {
                w_max_fold: 0,
                q_max_fold: u32::MAX,
                ..d
            },
        ),
        (
            "classic",
            Poseidon2Config {
                kernels: Poseidon2Kernels::Classic,
                ..d
            },
        ),
    ];
    for (i, (name, cfg)) in configs.into_iter().enumerate() {
        check_tree(&with_config(cfg), name, 100 + i as u64);
    }
}

/// The classic kernels (`MetalHalPoseidon2Legacy`, F0's code path) and pv agree.
#[test]
fn legacy_oracle_matches_pv() {
    let pv = MetalHalPoseidon2::new();
    let legacy = MetalHalPoseidon2Legacy::new();
    let mut rng = rng(5);
    for (r, c) in [
        (1usize, 1usize),
        (33, 17),
        (1000, 211),
        (5000, 24),
        (20000, 16),
    ] {
        let m = elems(&mut rng, r * c);
        let a = {
            let mb = pv.copy_from_elem("m", &m);
            let out = pv.alloc_digest("out", r);
            pv.hash_rows(&out, &mb);
            out.to_vec()
        };
        let b = {
            let mb = legacy.copy_from_elem("m", &m);
            let out = legacy.alloc_digest("out", r);
            legacy.hash_rows(&out, &mb);
            out.to_vec()
        };
        assert!(a == b, "rows={r} cols={c}");
    }
    for layers in [1usize, 9, 14] {
        let io = digests(&mut rng, 2 << layers);
        let a = pv.copy_from_digest("io", &io);
        let b = legacy.copy_from_digest("io", &io);
        pv.hash_fold_tree(&a, layers);
        legacy.hash_fold_tree(&b, layers);
        assert!(a.to_vec()[1..] == b.to_vec()[1..], "layers={layers}");
    }
}

/// The tail kernel alone: layers n0..1 of a tree in one dispatch.
#[test]
fn fold_tail_kernel() {
    let hal = MetalHalPoseidon2::new();
    let cpu = cpu();
    let mut rng = rng(6);
    for layers in 1..=7usize {
        let n0 = 1u32 << (layers - 1);
        let io = digests(&mut rng, 2 << layers);
        let want = cpu_tree(&cpu, &io, layers);
        for m32 in [false, true] {
            let b = hal.copy_from_digest("io", &io);
            MetalHashPoseidon2::encode_fold_tail(&hal, m32, &b, n0);
            assert!(b.to_vec()[1..] == want[1..], "m32={m32} layers={layers}");
        }
    }
}

// ---- measurements (indicative; run with --ignored --nocapture --test-threads=1) ----

/// Wall time per dispatch of `reps` dispatches encoded into one batch, best of 3.
fn time_per<F: Fn()>(hal: &MetalHalPoseidon2, reps: usize, f: F) -> f64 {
    f();
    hal.sync();
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        hal.sync();
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    best * 1e6
}

fn reps_for(perms: usize) -> usize {
    (4_000_000 / perms.max(1)).clamp(3, 200)
}

#[test]
#[ignore]
fn bench_fold_kernels() {
    let hal = MetalHalPoseidon2::new();
    let mut rng = rng(7);
    println!("fold n | pv | pv_m32 | pvs_w | pvs_w_m32 | pvs_q | pvs_q_m32 (us per dispatch)");
    for lg in 0..=22 {
        let n = 1usize << lg;
        let io: BufferImpl<Digest> = hal.copy_from_digest("io", &digests(&mut rng, 4 * n));
        let mut row = format!("2^{lg}");
        for k in ALL {
            if matches!(k.0, PvKernel::PvsW | PvKernel::PvsQ) && lg > 16 {
                row += " | -";
                continue;
            }
            let us = time_per(&hal, reps_for(n), || {
                MetalHashPoseidon2::encode_fold(&hal, k.0, k.1, &io, n, 2 * n, n as u32)
            });
            row += &format!(" | {us:.1}");
        }
        println!("{row}");
    }
}

#[test]
#[ignore]
fn bench_rows_kernels() {
    let hal = MetalHalPoseidon2::new();
    let mut rng = rng(8);
    for cols in [16usize, 211] {
        println!("rows (cols={cols}) | pv | pv_m32 | pvs_w | pvs_w_m32 | pvs_q | pvs_q_m32 (us per dispatch)");
        for lg in 0..=20 {
            let rows = 1usize << lg;
            if rows * cols > 1 << 28 {
                break;
            }
            let m: BufferImpl<BabyBearElem> =
                hal.copy_from_elem("m", &elems(&mut rng, rows * cols));
            let out = hal.alloc_digest("out", rows);
            let perms = rows * cols.div_ceil(16);
            let mut row = format!("2^{lg}");
            for k in ALL {
                if matches!(k.0, PvKernel::PvsW | PvKernel::PvsQ) && lg > 16 {
                    row += " | -";
                    continue;
                }
                let us = time_per(&hal, reps_for(perms), || {
                    MetalHashPoseidon2::encode_rows(
                        &hal,
                        k.0,
                        k.1,
                        &out,
                        &m,
                        rows as u32,
                        cols as u32,
                    )
                });
                row += &format!(" | {us:.1}");
            }
            println!("{row}");
        }
    }
}

/// Fold trees: per-layer dispatches with the default crossovers vs the tail kernel, and classic.
#[test]
#[ignore]
fn bench_fold_tree() {
    let d = Poseidon2Config::default();
    let mut rng = rng(9);
    let cfgs = [
        ("default", d),
        ("tail128", Poseidon2Config { tail_max: 128, ..d }),
        ("tail16", Poseidon2Config { tail_max: 16, ..d }),
        ("no-small", Poseidon2Config { small: false, ..d }),
        (
            "classic",
            Poseidon2Config {
                kernels: Poseidon2Kernels::Classic,
                ..d
            },
        ),
    ];
    const LAYERS: [usize; 5] = [7, 10, 14, 18, 21];
    let hals: Vec<_> = cfgs.iter().map(|(_, cfg)| with_config(*cfg)).collect();
    let ios: Vec<Vec<_>> = hals
        .iter()
        .map(|hal| {
            LAYERS
                .iter()
                .map(|&l| hal.copy_from_digest("io", &digests(&mut rng, 2 << l)))
                .collect()
        })
        .collect();
    // Interleaved rounds, best per cell: the GPU clock and the other agents' load drift.
    let mut best = vec![[f64::MAX; LAYERS.len()]; cfgs.len()];
    for _round in 0..4 {
        for (c, hal) in hals.iter().enumerate() {
            for (j, &layers) in LAYERS.iter().enumerate() {
                let io = &ios[c][j];
                let us = time_per(hal, reps_for(1 << layers).min(20), || {
                    hal.hash().hash_fold_tree(hal, io, layers)
                });
                best[c][j] = best[c][j].min(us);
            }
        }
    }
    for (c, (name, _)) in cfgs.iter().enumerate() {
        let mut row = format!("{name}:");
        for (j, layers) in LAYERS.iter().enumerate() {
            row += &format!(" L{layers}={:.1}us", best[c][j]);
        }
        println!("{row}");
    }
}

/// Per-call GPU+encode time of the Hal entry points at the sizes the tests use.
#[test]
#[ignore]
fn bench_hal_entry_points() {
    let hal = MetalHalPoseidon2::new();
    let legacy = MetalHalPoseidon2Legacy::new();
    let mut rng = rng(10);
    for (rows, cols) in [
        (1usize << 20, 24usize),
        (1 << 20, 211),
        (1 << 21, 211),
        (1 << 22, 16),
    ] {
        let m = elems(&mut rng, rows * cols);
        let a = hal.copy_from_elem("m", &m);
        let ao = hal.alloc_digest("out", rows);
        let pv_us = time_per(&hal, 3, || hal.hash().hash_rows(&hal, &ao, &a));
        drop(a);
        let b = legacy.copy_from_elem("m", &m);
        let bo = legacy.alloc_digest("out", rows);
        let t = Instant::now();
        legacy.hash_rows(&bo, &b);
        legacy.sync();
        let classic_us = t.elapsed().as_secs_f64() * 1e6;
        println!(
            "hash_rows {rows}x{cols}: pv {:.2} ms, classic {:.2} ms",
            pv_us / 1e3,
            classic_us / 1e3
        );
    }
}
