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

//! Z10: the `poseidon254.metal` kernels against the CPU Poseidon254 hash suite, bit for bit.
//! Every arithmetic variant (`w64`, `w64c`, `ref`) are checked on their own (forced through
//! `encode_rows` / `encode_fold`) and through the `Hal` entry points, at sizes around every
//! threadgroup / partial-block boundary and at the exact shapes of a po2-18 recursion proof
//! (identity_p254). `bench_*` (ignored) are the measurements recorded in
//! docs/metal-port/notes/Z10.md.

use std::time::Instant;

use rand::{rngs::SmallRng, RngCore, SeedableRng};
use risc0_core::field::baby_bear::BabyBearElem;
use test_log::test;

use super::{
    BufferImpl, MetalHalPoseidon254, MetalHashPoseidon254, Poseidon254Config, Poseidon254Kernels,
};
use crate::{
    core::{digest::Digest, hash::poseidon_254::Poseidon254HashSuite},
    hal::{cpu::CpuHal, Buffer, Hal},
};

const P: u32 = 0x7800_0001;
/// BN254 Fr modulus r, little-endian words.
const R: [u32; 8] = [
    0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72,
];
const ALL: [Poseidon254Kernels; 3] = [
    Poseidon254Kernels::W64,
    Poseidon254Kernels::W64Cios,
    Poseidon254Kernels::Ref,
];

type Cpu = CpuHal<risc0_core::field::baby_bear::BabyBear>;

fn cpu() -> Cpu {
    CpuHal::new(Poseidon254HashSuite::new_suite())
}

fn rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

/// Random field elements (raw Montgomery words in [0, p)), with the edge values 0, 1, p - 1 at
/// the front.
fn elems(rng: &mut SmallRng, n: usize) -> Vec<BabyBearElem> {
    (0..n)
        .map(|i| match i {
            0 => BabyBearElem::new_raw(0),
            1 => BabyBearElem::new(P - 1),
            2 => BabyBearElem::new(1),
            _ => BabyBearElem::new_raw(rng.next_u32() % P),
        })
        .collect()
}

/// Random canonical Fr digests (< r), with 0, 1 and r - 1 among them.
fn digests(rng: &mut SmallRng, n: usize) -> Vec<Digest> {
    (0..n)
        .map(|i| {
            let mut w: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            w[7] %= R[7]; // < r
            match i % 97 {
                5 => w = [0; 8],
                6 => w = [1, 0, 0, 0, 0, 0, 0, 0],
                7 => {
                    w = R;
                    w[0] -= 1;
                }
                _ => {}
            }
            Digest::from(w)
        })
        .collect()
}

fn cpu_rows(cpu: &Cpu, matrix: &[BabyBearElem], rows: usize) -> Vec<Digest> {
    let m = cpu.copy_from_elem("m", matrix);
    let out = cpu.alloc_digest("out", rows);
    cpu.hash_rows(&out, &m);
    out.to_vec()
}

/// The CPU hash of row `row` alone (for sampled checks of large shapes).
fn cpu_row(matrix: &[BabyBearElem], rows: usize, row: usize) -> Digest {
    let cols = matrix.len() / rows;
    let v: Vec<BabyBearElem> = (0..cols).map(|c| matrix[c * rows + row]).collect();
    *Poseidon254HashSuite::new_suite().hashfn.hash_elem_slice(&v)
}

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

fn cfg(kernels: Poseidon254Kernels) -> Poseidon254Config {
    Poseidon254Config {
        kernels,
        ..Default::default()
    }
}

fn metal_rows_with(
    hal: &MetalHalPoseidon254,
    c: &Poseidon254Config,
    matrix: &[BabyBearElem],
    rows: usize,
) -> Vec<Digest> {
    let m = hal.copy_from_elem("m", matrix);
    let out = hal.alloc_digest("out", rows);
    let cols = matrix.len() / rows;
    MetalHashPoseidon254::encode_rows(hal, c, &out, &m, rows, cols);
    out.to_vec()
}

fn metal_fold_with(
    hal: &MetalHalPoseidon254,
    c: &Poseidon254Config,
    io: &[Digest],
    n: usize,
) -> Vec<Digest> {
    let b: BufferImpl<Digest> = hal.copy_from_digest("io", io);
    MetalHashPoseidon254::encode_fold(hal, c, &b, n, 2 * n, n);
    b.to_vec()
}

fn first_diff<T: PartialEq>(a: &[T], b: &[T]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// The zkp TU's `bn254.h` is a byte copy of groth16's (G2) header.
#[test]
fn bn254_header_is_groth16_copy() {
    let zkp = include_str!("../../../../sys/kernels/zkp/metal/bn254.h");
    let g16 = include_str!("../../../../groth16-sys/kernels/metal/bn254.h");
    assert!(
        zkp == g16,
        "sys/kernels/zkp/metal/bn254.h differs from groth16-sys/kernels/metal/bn254.h: copy it again"
    );
}

/// hash_fold, both variants, output counts around threadgroup boundaries (and a custom
/// threadgroup size that leaves a partial group).
#[test]
fn fold_matches_cpu() {
    let cpu = cpu();
    let hal = MetalHalPoseidon254::new();
    for (i, &n) in [1usize, 2, 3, 7, 31, 32, 33, 255, 256, 257, 1000, 4099]
        .iter()
        .enumerate()
    {
        let io = digests(&mut rng(100 + i as u64), 4 * n);
        let want = cpu_fold(&cpu, &io, n);
        for k in ALL {
            for tg in [0, 96] {
                let c = Poseidon254Config {
                    kernels: k,
                    tg,
                    ..Default::default()
                };
                let got = metal_fold_with(&hal, &c, &io, n);
                assert!(
                    got == want,
                    "fold {k:?} tg {tg} n {n}: first diff at {:?}",
                    first_diff(&got, &want)
                );
            }
        }
    }
}

/// hash_rows, both variants, every column count around the 8-element cell and 16-element
/// permutation boundaries plus the recursion group widths (12, 16, 23, 64, 128), row counts
/// around threadgroup boundaries.
#[test]
fn rows_matches_cpu() {
    let cpu = cpu();
    let hal = MetalHalPoseidon254::new();
    let mut seed = 0;
    for &rows in &[1usize, 5, 33, 256, 1001] {
        for &cols in &[
            1usize, 2, 7, 8, 9, 12, 15, 16, 17, 23, 24, 31, 32, 33, 64, 128, 129,
        ] {
            seed += 1;
            let m = elems(&mut rng(seed), rows * cols);
            let want = cpu_rows(&cpu, &m, rows);
            for k in ALL {
                let got = metal_rows_with(&hal, &cfg(k), &m, rows);
                assert!(
                    got == want,
                    "rows {k:?} {rows} x {cols}: first diff at {:?}",
                    first_diff(&got, &want)
                );
            }
        }
    }
}

/// The `Hal` entry points (default config): hash_rows, hash_fold, hash_fold_tree vs the CPU HAL.
#[test]
fn hal_entry_points_match_cpu() {
    let cpu = cpu();
    let hal = MetalHalPoseidon254::new();
    assert_eq!(hal.get_hash_suite().name, "poseidon254");
    for (rows, cols) in [(1024usize, 23usize), (4096, 64), (2048, 12)] {
        let m = elems(&mut rng(rows as u64 * 7 + cols as u64), rows * cols);
        let want = cpu_rows(&cpu, &m, rows);
        let mb = hal.copy_from_elem("m", &m);
        let out = hal.alloc_digest("out", rows);
        hal.hash_rows(&out, &mb);
        assert!(out.to_vec() == want, "hash_rows {rows} x {cols}");
    }
    for layers in [0usize, 1, 2, 5, 10, 12] {
        let io = digests(&mut rng(900 + layers as u64), 2 << layers);
        let want = cpu_tree(&cpu, &io, layers);
        let b = hal.copy_from_digest("io", &io);
        hal.hash_fold_tree(&b, layers);
        let got = b.to_vec();
        // io[0] is never written by either HAL.
        assert!(got[1..] == want[1..], "fold tree {layers} layers");
    }
    let n = 777;
    let io = digests(&mut rng(5), 4 * n);
    let want = cpu_fold(&cpu, &io, n);
    let b = hal.copy_from_digest("io", &io);
    hal.hash_fold(&b, 2 * n, n);
    assert!(b.to_vec() == want, "hash_fold {n}");
}

/// The CPU crossovers (`cpu_tail`, `cpu_rows`) give the same digests: trees whose top runs on
/// the CPU, single folds and row hashes below / at / above the thresholds.
#[test]
fn cpu_crossovers_match_cpu() {
    let cpu = cpu();
    let hal = MetalHalPoseidon254::new();
    for (tail, rows_max) in [(1u32, 1u32), (64, 100), (4096, 4096)] {
        hal.hash().set_config(Poseidon254Config {
            cpu_tail: tail,
            cpu_rows: rows_max,
            ..Default::default()
        });
        for layers in [0usize, 1, 6, 7, 13] {
            let io = digests(&mut rng(77 + layers as u64), 2 << layers);
            let want = cpu_tree(&cpu, &io, layers);
            let b = hal.copy_from_digest("io", &io);
            hal.hash_fold_tree(&b, layers);
            assert!(b.to_vec()[1..] == want[1..], "tail {tail}: tree {layers}");
        }
        for n in [1usize, 63, 64, 65, 300] {
            let io = digests(&mut rng(n as u64), 4 * n);
            let want = cpu_fold(&cpu, &io, n);
            let b = hal.copy_from_digest("io", &io);
            hal.hash_fold(&b, 2 * n, n);
            assert!(b.to_vec() == want, "tail {tail}: fold {n}");
        }
        for rows in [1usize, 99, 100, 101, 4096] {
            let m = elems(&mut rng(rows as u64 + 3), rows * 23);
            let want = cpu_rows(&cpu, &m, rows);
            let mb = hal.copy_from_elem("m", &m);
            let out = hal.alloc_digest("out", rows);
            hal.hash_rows(&out, &mb);
            assert!(out.to_vec() == want, "cpu_rows {rows_max}: rows {rows}");
        }
    }
}

/// The exact Merkle shapes of a po2-18 recursion proof (identity_p254: code 23, data 128, accum
/// 12, check 16 columns over 2^20 rows; FRI 64 columns over 2^16 / 2^12 / 2^8 rows), with the
/// default kernels through the `Hal`: rows checked against the CPU on a sample (every 251st row,
/// the first and last 64), and the full fold tree of the 2^20 code group against the CPU.
#[test]
fn recursion_shapes_match_cpu() {
    let _lock = super::gpu_test_lock();
    let cpu = cpu();
    let hal = MetalHalPoseidon254::new();
    let shapes = [
        (1usize << 20, 23usize),
        (1 << 20, 128),
        (1 << 20, 12),
        (1 << 20, 16),
        (1 << 16, 64),
        (1 << 12, 64),
        (1 << 8, 64),
    ];
    for (i, &(rows, cols)) in shapes.iter().enumerate() {
        let m = elems(&mut rng(4242 + i as u64), rows * cols);
        let mb = hal.copy_from_elem("m", &m);
        let nodes = hal.alloc_digest("nodes", 2 * rows);
        hal.hash_rows(&nodes.slice(rows, rows), &mb);
        let got = nodes.to_vec();
        let sample: Vec<usize> = (0..rows)
            .filter(|r| r % 251 == 0 || *r < 64 || *r >= rows - 64)
            .collect();
        for &r in &sample {
            assert!(
                got[rows + r] == cpu_row(&m, rows, r),
                "rows {rows} x {cols}: row {r} differs"
            );
        }
        if i == 0 || rows <= 1 << 16 {
            // The whole tree, against the CPU HAL's fold tree over the same leaves.
            let layers = rows.trailing_zeros() as usize;
            hal.hash_fold_tree(&nodes, layers);
            let want = cpu_tree(&cpu, &got, layers);
            let got = nodes.to_vec();
            assert!(
                got[1..] == want[1..],
                "fold tree of {rows}: first diff at {:?}",
                first_diff(&got[1..], &want[1..])
            );
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Time one encode + sync of `f` (ms).
fn time_ms(hal: &MetalHalPoseidon254, f: impl FnOnce()) -> f64 {
    hal.sync();
    let t = Instant::now();
    f();
    hal.sync();
    t.elapsed().as_secs_f64() * 1e3
}

/// Kernel timing at the recursion shapes, the variants interleaved (`Z10_BENCH_REPS`, default 7;
/// `Z10_BENCH_VARIANTS`, `kernels[:tg[:cpu_tail[:cpu_rows]]]` comma list, e.g. `w64,w64c,ref` or
/// `w64:0:0:0,w64:0:2048:1024` for threadgroup sizes). Prints
/// `Z10BENCH` lines: median / min / max ms per shape and variant, and the per-proof total.
#[test]
#[ignore]
fn bench_kernels() {
    let _lock = super::gpu_test_lock();
    let reps: usize = std::env::var("Z10_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let variants: Vec<Poseidon254Config> = std::env::var("Z10_BENCH_VARIANTS")
        .unwrap_or_else(|_| "w64,w64c,ref".into())
        .split(',')
        .map(|v| {
            // kernels[:tg[:cpu_tail[:cpu_rows]]]
            let f: Vec<&str> = v.split(':').collect();
            let num = |i: usize| f.get(i).map_or(0, |x| x.parse().unwrap());
            let k = f[0];
            Poseidon254Config {
                tg: num(1),
                cpu_tail: num(2),
                cpu_rows: num(3),
                kernels: match k {
                    "ref" => Poseidon254Kernels::Ref,
                    "w64c" => Poseidon254Kernels::W64Cios,
                    _ => Poseidon254Kernels::W64,
                },
            }
        })
        .collect();
    let hal = MetalHalPoseidon254::new();
    let shapes = [
        ("code", 1usize << 20, 23usize),
        ("data", 1 << 20, 128),
        ("accum", 1 << 20, 12),
        ("check", 1 << 20, 16),
        ("fri0", 1 << 16, 64),
        ("fri1", 1 << 12, 64),
        ("fri2", 1 << 8, 64),
    ];
    let mut totals = vec![vec![0.0; reps]; variants.len()];
    for (name, rows, cols) in shapes {
        let m = hal.copy_from_elem(
            "m",
            &elems(&mut rng(rows as u64 + cols as u64), rows * cols),
        );
        let nodes = hal.alloc_digest("nodes", 2 * rows);
        let layers = rows.trailing_zeros() as usize;
        let mut tr = vec![vec![]; variants.len()];
        let mut tf = vec![vec![]; variants.len()];
        // warm-up
        for c in &variants {
            hal.hash().set_config(*c);
            hal.hash_rows(&nodes.slice(rows, rows), &m);
            hal.hash_fold_tree(&nodes, layers);
            hal.sync();
        }
        // Interleaved: rep-major, variant-minor.
        #[allow(clippy::needless_range_loop)]
        for rep in 0..reps {
            for (vi, c) in variants.iter().enumerate() {
                hal.hash().set_config(*c);
                let a = time_ms(&hal, || hal.hash_rows(&nodes.slice(rows, rows), &m));
                let b = time_ms(&hal, || hal.hash_fold_tree(&nodes, layers));
                tr[vi].push(a);
                tf[vi].push(b);
                totals[vi][rep] += a + b;
            }
        }
        for (vi, c) in variants.iter().enumerate() {
            let (rmin, rmax) = minmax(&tr[vi]);
            let (fmin, fmax) = minmax(&tf[vi]);
            println!(
                "Z10BENCH {name} {rows}x{cols} {:?}/tg{}/t{}/r{}: rows med {:.2} ms [{rmin:.2}, {rmax:.2}] \
                 tree med {:.2} ms [{fmin:.2}, {fmax:.2}]",
                c.kernels,
                c.tg,
                c.cpu_tail,
                c.cpu_rows,
                median(&mut tr[vi].clone()),
                median(&mut tf[vi].clone()),
            );
        }
    }
    for (vi, c) in variants.iter().enumerate() {
        let (mn, mx) = minmax(&totals[vi]);
        println!(
            "Z10BENCH total {:?}/tg{}/t{}/r{}: med {:.1} ms [{mn:.1}, {mx:.1}]",
            c.kernels,
            c.tg,
            c.cpu_tail,
            c.cpu_rows,
            median(&mut totals[vi].clone())
        );
    }
}

fn minmax(v: &[f64]) -> (f64, f64) {
    v.iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)))
}

/// PSO properties of the Z10 kernels (register pressure shows as a lower thread limit).
#[test]
#[ignore]
fn pso_report() {
    let hal = MetalHalPoseidon254::new();
    for name in super::KERNEL_NAMES
        .iter()
        .filter(|n| n.starts_with("p254_"))
    {
        let p = hal.pipeline(name);
        println!(
            "Z10PSO {name}: max threads/tg {}, execution width {}, static tg mem {}",
            p.max_total_threads_per_threadgroup(),
            p.thread_execution_width(),
            p.static_threadgroup_memory_length()
        );
    }
}

/// Microbenchmarks from a scratch metallib (`Z10_LIB`, kernels `Z10_KERNELS` comma list): each
/// kernel gets `io` (16 x `Z10_N` words, random) and `n`, one thread per `n`; the kernels are
/// interleaved for `Z10_BENCH_REPS` (7) rounds. Prints median / min / max ms per kernel and the
/// rate `Z10_OPS` (ops per thread, default 512) implies. For the arithmetic choices in the note.
#[test]
#[ignore]
fn bench_scratch_lib() {
    let _lock = super::gpu_test_lock();
    let path = std::env::var("Z10_LIB").expect("Z10_LIB");
    let names: Vec<String> = std::env::var("Z10_KERNELS")
        .expect("Z10_KERNELS")
        .split(',')
        .map(String::from)
        .collect();
    let env = |k: &str, d: usize| -> usize {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let n = env("Z10_N", 1 << 20);
    let reps = env("Z10_BENCH_REPS", 7);
    let ops = env("Z10_OPS", 512) as f64;
    let hal = MetalHalPoseidon254::new();
    let bytes = std::fs::read(&path).unwrap();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let lib = hal.load_library(&bytes, &name_refs);
    let mut r = rng(1);
    let init: Vec<u32> = (0..16 * n)
        .map(|i| {
            if i % 8 == 7 {
                r.next_u32() % R[7]
            } else {
                r.next_u32()
            }
        })
        .collect();
    let io = hal.copy_from_slice("io", &init);
    let mut t = vec![vec![]; names.len()];
    for rep in 0..=reps {
        for (k, name) in names.iter().enumerate() {
            let ms = time_ms(&hal, || {
                hal.dispatch(
                    lib.pipeline(name),
                    &[io.as_arg(), super::KernelArg::Integer(n as u32)],
                    super::Grid::Threads(n as u64),
                    &[],
                )
            });
            if rep > 0 {
                t[k].push(ms);
            }
        }
    }
    for (k, name) in names.iter().enumerate() {
        let (mn, mx) = minmax(&t[k]);
        let med = median(&mut t[k].clone());
        println!(
            "Z10MB {name}: med {med:.2} ms [{mn:.2}, {mx:.2}] = {:.2} Gop/s",
            n as f64 * ops / med / 1e6
        );
    }
}

/// R17 audit (docs/metal-port/notes/R17.md) for the Poseidon254 kernels: `R17_AUDIT_REPS` (10)
/// repeated hash_rows (23 and 128 columns) and hash_fold over 2^`R17_AUDIT_PO2` (18) rows, every
/// arithmetic variant, each rep against rep 0 and against the CPU suite. Prints `R17AUDIT` k/n
/// lines like `tests_r17`; fails on any difference.
#[test]
#[ignore]
fn r17_p254_stress() {
    let _lock = super::gpu_test_lock();
    let env = |k: &str, d: usize| -> usize {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (reps, po2) = (env("R17_AUDIT_REPS", 10), env("R17_AUDIT_PO2", 18));
    let rows = 1usize << po2;
    let cpu = cpu();
    let hal = MetalHalPoseidon254::new();
    let mut bad = vec![];
    for cols in [23usize, 128] {
        let m = elems(&mut rng(cols as u64), rows * cols);
        let want = cpu_rows(&cpu, &m, rows);
        let mut io = vec![Digest::ZERO; rows];
        io.extend_from_slice(&want);
        let want_fold = cpu_fold(&cpu, &io, rows / 2);
        let mb = hal.copy_from_elem("m", &m);
        for k in ALL {
            let c = cfg(k);
            let (mut vs_first, mut vs_cpu, mut fvs_first, mut fvs_cpu) = (0, 0, 0, 0);
            let (mut first, mut ffirst): (Option<Vec<Digest>>, Option<Vec<Digest>>) = (None, None);
            for _ in 0..reps {
                let out = hal.alloc_digest("out", rows);
                MetalHashPoseidon254::encode_rows(&hal, &c, &out, &mb, rows, cols);
                let got = out.to_vec();
                vs_cpu += usize::from(got != want);
                match &first {
                    None => first = Some(got),
                    Some(f) => vs_first += usize::from(*f != got),
                }
                let fold = metal_fold_with(&hal, &c, &io, rows / 2);
                fvs_cpu += usize::from(fold != want_fold);
                match &ffirst {
                    None => ffirst = Some(fold),
                    Some(f) => fvs_first += usize::from(*f != fold),
                }
            }
            println!(
                "R17AUDIT zkp hash_rows/poseidon254 {k:?} {cols}x2^{po2}: {vs_first}/{reps} reps \
                 differ from rep 0, {vs_cpu}/{reps} differ from CPU"
            );
            println!(
                "R17AUDIT zkp hash_fold/poseidon254 {k:?} 2^{po2}: {fvs_first}/{reps} reps differ \
                 from rep 0, {fvs_cpu}/{reps} differ from CPU"
            );
            if vs_first + vs_cpu + fvs_first + fvs_cpu > 0 {
                bad.push((k, cols));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "R17AUDIT poseidon254 nondeterministic or wrong: {bad:?}"
    );
}
