// Copyright 2025 RISC Zero, Inc.
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

//! CPU-vs-Metal bit-exact tests of the Z1 small kernels (`small.rs`, `misc.metal`,
//! `eltwise.metal`, and the sha/poseidon2 scalar bindings). Methods `DualHal` forwards are
//! checked through it; `scatter_bits`, `combos_prepare`, `alloc_elem_init` and
//! `alloc_extelem_zeroed` (which `DualHal` does not forward to the inner HALs) are run on a
//! `CpuHal` and a `MetalHal` from the same inputs and compared.

use std::{rc::Rc, time::Instant};

use rand::{rngs::StdRng, Rng, SeedableRng};
use risc0_core::field::{
    baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
    Elem as _, ExtElem as _,
};
use test_log::test;

use super::{MetalHalPoseidon2, MetalHalSha256};
use crate::{
    core::digest::Digest,
    hal::{cpu::CpuHal, dual::DualHal, Buffer, Hal},
    INV_RATE,
};

const COUNTS: [usize; 8] = [1, 9, 12, 255, 1001, 1024, 1025, 1 << 20];

fn rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn cpu_for(hal: &MetalHalSha256) -> CpuHal<BabyBear> {
    CpuHal::new(hal.get_hash_suite().clone())
}

fn dual() -> impl Hal<Elem = BabyBearElem, ExtElem = BabyBearExtElem> {
    let gpu = MetalHalSha256::new();
    let cpu = cpu_for(&gpu);
    DualHal::new(Rc::new(cpu), Rc::new(gpu))
}

/// Random elements, about one in four `INVALID`.
fn with_invalid(rng: &mut StdRng, n: usize) -> Vec<BabyBearElem> {
    (0..n)
        .map(|_| {
            if rng.random_range(0..4) == 0 {
                BabyBearElem::INVALID
            } else {
                BabyBearElem::random(&mut *rng)
            }
        })
        .collect()
}

#[test]
fn eltwise_zeroize_elem() {
    let hal = dual();
    let mut rng = rng(1);
    for count in COUNTS {
        let buf = hal.copy_from_elem("zeroize", &with_invalid(&mut rng, count));
        hal.eltwise_zeroize_elem(&buf);
        // A slice in the middle of a larger buffer: only the slice is touched.
        if count > 2 {
            // The first and last elements stay outside the slice, so keep them valid (Elem
            // comparison asserts validity).
            let mut init = with_invalid(&mut rng, count);
            init[0] = BabyBearElem::ONE;
            init[count - 1] = BabyBearElem::new(2);
            let buf = hal.copy_from_elem("zeroize", &init);
            hal.eltwise_zeroize_elem(&buf.slice(1, count - 2));
            // DualHal compared the slice; copying the whole buffer compares all of it.
            let whole = hal.alloc_elem("whole", count);
            hal.eltwise_copy_elem(&whole, &buf);
        }
    }
}

#[test]
fn eltwise_sum_extelem_shapes() {
    let hal = dual();
    let mut rng = rng(2);
    for (count, to_add) in [(1, 1), (9, 3), (1025, 4), (4096, 7), (1 << 16, 2)] {
        let input: Vec<_> = (0..count * to_add)
            .map(|_| BabyBearExtElem::random(&mut rng))
            .collect();
        let input = hal.copy_from_extelem("in", &input);
        let output = hal.alloc_elem("out", count * BabyBearExtElem::EXT_SIZE);
        hal.eltwise_sum_extelem(&output, &input);
    }
}

/// Unique offsets (a permutation prefix), variable-length (often empty) cycle ranges.
fn scatter_inputs(
    rng: &mut StdRng,
    cycles: usize,
    size: usize,
) -> (Vec<u32>, Vec<u32>, Vec<BabyBearElem>) {
    let mut index = vec![0u32];
    for _ in 0..cycles {
        let n = match rng.random_range(0..4) {
            0 => 0,
            1 => 1,
            _ => rng.random_range(0..9),
        };
        index.push(index.last().unwrap() + n);
    }
    let total = *index.last().unwrap() as usize;
    assert!(total <= size);
    let mut perm: Vec<u32> = (0..size as u32).collect();
    for i in 0..total {
        let j = rng.random_range(i..size);
        perm.swap(i, j);
    }
    perm.truncate(total);
    let values = (0..total)
        .map(|_| BabyBearElem::random(&mut *rng))
        .collect();
    (index, perm, values)
}

#[test]
fn scatter() {
    let hal = dual();
    let mut rng = rng(3);
    for cycles in [1, 9, 255, 1025, 1 << 16] {
        let size = cycles * 8 + 64;
        let init: Vec<_> = (0..size).map(|_| BabyBearElem::random(&mut rng)).collect();
        let into = hal.copy_from_elem("into", &init);
        let (index, offsets, values) = scatter_inputs(&mut rng, cycles, size);
        hal.scatter(&into, &index, &offsets, &values);
    }
    // Empty and single-entry index: no-ops on both HALs (the CPU HAL needs index.len() >= 1).
    let into = hal.copy_from_elem("into", &[BabyBearElem::ONE; 4]);
    hal.scatter(&into, &[0], &[], &[]);
    hal.scatter(&into, &[0, 0, 0], &[], &[]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn scatter_offset_out_of_bounds_panics() {
    let hal = MetalHalSha256::new();
    let into = hal.alloc_elem("into", 4);
    hal.scatter(&into, &[0, 1], &[4], &[BabyBearElem::ONE]);
}

#[test]
#[should_panic(expected = "beyond offsets/values")]
fn scatter_index_beyond_values_panics() {
    let hal = MetalHalSha256::new();
    let into = hal.alloc_elem("into", 4);
    hal.scatter(&into, &[0, 2], &[0, 1], &[BabyBearElem::ONE]);
}

#[test]
fn scatter_bits() {
    let gpu = MetalHalSha256::new();
    let cpu = cpu_for(&gpu);
    let mut rng = rng(4);
    for (cycles, blocks, n) in [
        (1u32, 1u32, 1usize),
        (16, 2, 20),
        (1024, 3, 2000),
        (1 << 16, 4, 1 << 17),
    ] {
        // Distinct (row, block) pairs so no two triples write the same cell.
        let mut cells: Vec<(u32, u32)> = (0..blocks)
            .flat_map(|b| (0..cycles).map(move |r| (r, b)))
            .collect();
        let n = n.min(cells.len());
        for i in 0..n {
            let j = rng.random_range(i..cells.len());
            cells.swap(i, j);
        }
        let mut data = Vec::with_capacity(n * 3 + 2);
        for &(row, block) in &cells[..n] {
            data.extend([row, block * 32, rng.random()]);
        }
        // Trailing partial triple is ignored by both.
        data.extend([0, 0]);
        let size = (blocks as usize * 32) * cycles as usize + 7;
        let init: Vec<_> = (0..size).map(|_| BabyBearElem::random(&mut rng)).collect();
        let c = cpu.copy_from_elem("into", &init);
        let g = gpu.copy_from_elem("into", &init);
        cpu.scatter_bits(&c, &data, cycles);
        gpu.scatter_bits(&g, &data, cycles);
        assert_eq!(c.to_vec(), g.to_vec(), "cycles {cycles}, n {n}");
    }
    // No triples: no-op.
    let g = gpu.copy_from_elem("into", &[BabyBearElem::ONE; 4]);
    gpu.scatter_bits(&g, &[1, 2], 1);
    assert_eq!(g.to_vec(), vec![BabyBearElem::ONE; 4]);
}

#[test]
#[should_panic(expected = "scatter_bits")]
fn scatter_bits_out_of_bounds_panics() {
    let hal = MetalHalSha256::new();
    let into = hal.alloc_elem("into", 32 * 4);
    // base 1 needs (1 + 31) * 4 + row < 128.
    hal.scatter_bits(&into, &[0, 1, 5], 4);
}

#[test]
fn eltwise_copy_elem_slice() {
    let hal = dual();
    let mut rng = rng(5);
    // (rows, cols, from_offset, from_stride, into_offset, into_stride)
    let shapes = [
        (1, 1, 0, 1, 0, 1),
        (7, 3, 2, 5, 1, 4),
        (1000, 17, 13, 17, 0, 20),
        (1 << 14, 33, 5, 40, 3, 33),
        (0, 5, 0, 5, 0, 5),
        (5, 0, 0, 5, 0, 5),
    ];
    for (rows, cols, fo, fs, io, is) in shapes {
        let from_len = fo + rows * fs + 3;
        let into_len = io + rows * is + 3;
        let from: Vec<_> = (0..from_len)
            .map(|_| BabyBearElem::random(&mut rng))
            .collect();
        let init: Vec<_> = (0..into_len)
            .map(|_| BabyBearElem::random(&mut rng))
            .collect();
        let into = hal.copy_from_elem("into", &init);
        hal.eltwise_copy_elem_slice(&into, &from, rows, cols, fo, fs, io, is);
    }
}

#[test]
#[should_panic(expected = "target region")]
fn eltwise_copy_elem_slice_out_of_bounds_panics() {
    let hal = MetalHalSha256::new();
    let into = hal.alloc_elem("into", 10);
    hal.eltwise_copy_elem_slice(&into, &[BabyBearElem::ONE; 100], 3, 4, 0, 4, 0, 4);
}

fn combos_case(
    rng: &mut StdRng,
    cycles: usize,
    combo_count: usize,
    regs: usize,
) -> (
    Vec<BabyBearExtElem>,
    Vec<BabyBearExtElem>,
    Vec<u32>,
    Vec<u32>,
    BabyBearExtElem,
) {
    let reg_sizes: Vec<u32> = (0..regs)
        .map(|_| rng.random_range(1..=cycles.min(8)) as u32)
        .collect();
    // Repeated combo ids: several registers hit the same offsets (exercises delta merging).
    let reg_combo_ids: Vec<u32> = (0..regs)
        .map(|_| rng.random_range(0..combo_count) as u32)
        .collect();
    let check = INV_RATE * BabyBearExtElem::EXT_SIZE;
    let n_u = reg_sizes.iter().map(|&s| s as usize).sum::<usize>() + check;
    let coeff_u = (0..n_u)
        .map(|_| BabyBearExtElem::random(&mut *rng))
        .collect();
    let combos = (0..cycles * (combo_count + 1))
        .map(|_| BabyBearExtElem::random(&mut *rng))
        .collect();
    let mix = BabyBearExtElem::random(&mut *rng);
    (combos, coeff_u, reg_sizes, reg_combo_ids, mix)
}

#[test]
fn combos_prepare() {
    let gpu = MetalHalSha256::new();
    let cpu = cpu_for(&gpu);
    let mut rng = rng(6);
    for (cycles, combo_count, regs) in [
        (16, 1, 1),
        (16, 3, 20),
        (1024, 10, 300),
        (1 << 16, 40, 2000),
    ] {
        let (combos, coeff_u, reg_sizes, reg_combo_ids, mix) =
            combos_case(&mut rng, cycles, combo_count, regs);
        let c = cpu.copy_from_extelem("combos", &combos);
        let g = gpu.copy_from_extelem("combos", &combos);
        cpu.combos_prepare(
            &c,
            &coeff_u,
            combo_count,
            cycles,
            &reg_sizes,
            &reg_combo_ids,
            &mix,
        );
        gpu.combos_prepare(
            &g,
            &coeff_u,
            combo_count,
            cycles,
            &reg_sizes,
            &reg_combo_ids,
            &mix,
        );
        assert_eq!(
            c.to_vec(),
            g.to_vec(),
            "cycles {cycles}, combos {combo_count}"
        );
    }
}

#[test]
fn alloc_elem_init_and_extelem_zeroed() {
    let gpu = MetalHalSha256::new();
    let mut rng = rng(7);
    let values = [
        BabyBearElem::ZERO,
        BabyBearElem::INVALID,
        BabyBearElem::ONE,
        BabyBearElem::random(&mut rng),
    ];
    for size in [0usize, 1, 9, 1025, 1 << 20] {
        for value in values {
            // Dirty the pool at this size first so a reused buffer must really be filled.
            drop(gpu.copy_from_elem("dirty", &vec![BabyBearElem::new(5); size]));
            gpu.sync();
            let buf = gpu.alloc_elem_init("init", size, value);
            // Compare raw words: Elem equality asserts validity, and INVALID is a test value.
            let raw: Vec<u32> = buf.to_vec().iter().map(|e| e.as_u32_montgomery()).collect();
            assert_eq!(raw, vec![value.as_u32_montgomery(); size], "size {size}");
        }
        drop(gpu.copy_from_extelem("dirty", &vec![BabyBearExtElem::ONE; size]));
        gpu.sync();
        let buf = gpu.alloc_extelem_zeroed("zeroed", size);
        assert_eq!(
            buf.to_vec(),
            vec![BabyBearExtElem::ZERO; size],
            "size {size}"
        );
    }
    // A fill followed by compute in the same batch is ordered.
    let a = gpu.alloc_elem_init("a", 1000, BabyBearElem::new(3));
    let b = gpu.alloc_elem_init("b", 1000, BabyBearElem::new(4));
    let o = gpu.alloc_elem("o", 1000);
    gpu.eltwise_add_elem(&o, &a, &b);
    assert_eq!(o.to_vec(), vec![BabyBearElem::new(7); 1000]);
}

/// sha_rows/sha_fold/poseidon2_fold/poseidon2_rows with their scalars as `constant` set_bytes
/// arguments, at sizes not a multiple of the threadgroup width.
#[test]
fn hash_scalar_bindings() {
    fn run<H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>>(
        gpu: H,
        cpu: CpuHal<BabyBear>,
    ) {
        let hal = DualHal::new(Rc::new(cpu), Rc::new(gpu));
        let mut rng = rng(8);
        for (rows, cols) in [(1, 1), (3, 17), (257, 9), (1000, 33)] {
            let m: Vec<_> = (0..rows * cols)
                .map(|_| BabyBearElem::random(&mut rng))
                .collect();
            let m = hal.copy_from_elem("m", &m);
            let out = hal.alloc_digest("out", rows);
            hal.hash_rows(&out, &m);
        }
        for out_size in [1usize, 3, 257, 1000] {
            // io = [unused | outputs | inputs]: 4 * out_size digests.
            let d: Vec<_> = (0..out_size * 4)
                .map(|_| Digest::from([rng.random::<u32>() % 0x7800_0001; 8]))
                .collect();
            let io = hal.copy_from_digest("io", &d);
            hal.hash_fold(&io, out_size * 2, out_size);
        }
    }
    let gpu = MetalHalSha256::new();
    let cpu = CpuHal::new(gpu.get_hash_suite().clone());
    run(gpu, cpu);
    let gpu = MetalHalPoseidon2::new();
    let cpu = CpuHal::new(gpu.get_hash_suite().clone());
    run(gpu, cpu);
}

/// Indicative per-kernel timings (wall clock of encode + commit + wait, median of 9 after a
/// warm-up; includes the host upload for scatter/scatter_bits/copy_elem_slice/combos_prepare).
/// `cargo test -p risc0-zkp --features metal --lib -- hal::metal::tests_z1::timings --ignored
/// --nocapture`
#[test]
#[ignore]
fn timings() {
    let hal = MetalHalSha256::new();
    let mut rng = rng(9);
    fn time(hal: &MetalHalSha256, name: &str, mut f: impl FnMut()) {
        f();
        hal.sync();
        let mut ts: Vec<f64> = (0..9)
            .map(|_| {
                let t = Instant::now();
                f();
                hal.sync();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("Z1 timing {name}: {:.3} ms", ts[4]);
    }
    for po2 in [20usize, 22] {
        let n = 1 << po2;
        let buf = hal.copy_from_elem("buf", &with_invalid(&mut rng, n));
        time(&hal, &format!("eltwise_zeroize_fp n=2^{po2}"), || {
            hal.eltwise_zeroize_elem(&buf)
        });
        time(&hal, &format!("alloc_elem_init(blit) n=2^{po2}"), || {
            drop(hal.alloc_elem_init("i", n, BabyBearElem::ZERO))
        });
        time(
            &hal,
            &format!("alloc_elem_init(fill_u32) n=2^{po2}"),
            || drop(hal.alloc_elem_init("i", n, BabyBearElem::new(3))),
        );
        let ext = hal.copy_from_extelem(
            "ext",
            &(0..n)
                .map(|_| BabyBearExtElem::random(&mut rng))
                .collect::<Vec<_>>(),
        );
        let out = hal.alloc_elem("out", n);
        time(
            &hal,
            &format!("eltwise_sum_fpext count=2^{} to_add=4", po2 - 2),
            || hal.eltwise_sum_extelem(&out, &ext),
        );
        let cycles = n / 8;
        let into = hal.alloc_elem("into", cycles * 8 + 64);
        let (index, offsets, values) = scatter_inputs(&mut rng, cycles, cycles * 8 + 64);
        time(
            &hal,
            &format!("scatter cycles=2^{} entries={}", po2 - 3, values.len()),
            || hal.scatter(&into, &index, &offsets, &values),
        );
        let cyc = (n / 128) as u32;
        let bits_into = hal.alloc_elem("bits", 4 * 32 * cyc as usize);
        let data: Vec<u32> = (0..cyc)
            .flat_map(|r| (0..4u32).flat_map(move |b| [r, b * 32, r.wrapping_mul(2654435761)]))
            .collect();
        time(
            &hal,
            &format!("scatter_bits triples={} (32 writes each)", data.len() / 3),
            || hal.scatter_bits(&bits_into, &data, cyc),
        );
        let rows = n / 64;
        let from: Vec<_> = (0..rows * 64)
            .map(|_| BabyBearElem::random(&mut rng))
            .collect();
        let into = hal.alloc_elem("into", rows * 64);
        time(
            &hal,
            &format!("eltwise_copy_fp_region rows={rows} cols=64"),
            || hal.eltwise_copy_elem_slice(&into, &from, rows, 64, 0, 64, 0, 64),
        );
        let (combos, coeff_u, reg_sizes, reg_combo_ids, mix) =
            combos_case(&mut rng, n / 64, 60, 4000);
        let c = hal.copy_from_extelem("combos", &combos);
        time(
            &hal,
            &format!("combos_prepare cycles=2^{} regs=4000", po2 - 6),
            || hal.combos_prepare(&c, &coeff_u, 60, n / 64, &reg_sizes, &reg_combo_ids, &mix),
        );
    }
}
