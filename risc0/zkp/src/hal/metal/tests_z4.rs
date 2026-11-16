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

//! Z4 conformance: eval_u (eu2 trio + Horner fallback, every dispatcher branch) and
//! mix_poly_coeffs (v2 / v2_dcids / v1-with-table / table-less around both 32 KB gates),
//! bit-exact against the CPU HAL.

use std::{rc::Rc, time::Instant};

use rand::{rngs::SmallRng, Rng, SeedableRng};
use risc0_core::field::{
    baby_bear::{BabyBearElem, BabyBearExtElem},
    Elem as _,
};
use test_log::test;

use super::{
    eval::{
        self, eval_plan, mix_plan, EvalPath, MixKernel, MIX_MID_KERNEL, MIX_TABLE_MAX_INPUTS,
        MIX_V2_MAX_INPUTS,
    },
    MetalHalSha256,
};
use crate::hal::{cpu::CpuHal, dual::DualHal, Buffer, Hal};

fn rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

fn rand_elems(rng: &mut SmallRng, n: usize) -> Vec<BabyBearElem> {
    (0..n).map(|_| BabyBearElem::random(rng)).collect()
}

fn rand_ext(rng: &mut SmallRng, n: usize) -> Vec<BabyBearExtElem> {
    (0..n).map(|_| BabyBearExtElem::random(rng)).collect()
}

// ---- eval_u ----

#[test]
fn eval_plan_branches() {
    // nEvals == 0, deg < 4096, deg % 4096 != 0, misaligned coeffs: Horner.
    assert_eq!(eval_plan(1 << 16, 0, 0), EvalPath::Horner);
    assert_eq!(eval_plan(2048, 10, 0), EvalPath::Horner);
    assert_eq!(eval_plan(4096 + 2048, 10, 0), EvalPath::Horner);
    assert_eq!(eval_plan(1 << 16, 10, 4), EvalPath::Horner);
    assert_eq!(eval_plan(1 << 16, 10, 8), EvalPath::Horner);
    // S = 1 at the minimum chunk.
    assert_eq!(eval_plan(4096, 1, 0), EvalPath::Eu2 { s: 1, chunk: 4096 });
    // S capped by deg / 4096.
    assert_eq!(
        eval_plan(1 << 16, 1, 16),
        EvalPath::Eu2 { s: 16, chunk: 4096 }
    );
    // S stops once S * nEvals >= 1024.
    assert_eq!(
        eval_plan(1 << 16, 865, 0),
        EvalPath::Eu2 {
            s: 2,
            chunk: 1 << 15
        }
    );
    assert_eq!(
        eval_plan(1 << 16, 1024, 0),
        EvalPath::Eu2 {
            s: 1,
            chunk: 1 << 16
        }
    );
    assert_eq!(
        eval_plan(1 << 22, 3, 0),
        EvalPath::Eu2 {
            s: 512,
            chunk: 8192
        }
    );
    assert_eq!(
        eval_plan(1 << 20, 4, 0),
        EvalPath::Eu2 {
            s: 256,
            chunk: 4096
        }
    );
}

/// One batch_evaluate_any case: `poly_count` random polys of `deg` coefficients placed at element
/// offset `offset` of a larger buffer, `n_evals` random (which, x) pairs. Checks the path taken
/// and Metal == CPU bit for bit.
fn eval_case(
    seed: u64,
    deg: usize,
    poly_count: usize,
    n_evals: usize,
    offset: usize,
    force: Option<EvalPath>,
    expect: EvalPath,
) {
    let mut rng = rng(seed);
    let metal = MetalHalSha256::new();
    let cpu = CpuHal::new(metal.get_hash_suite().clone());

    let coeffs = rand_elems(&mut rng, deg * poly_count);
    let which: Vec<u32> = (0..n_evals)
        .map(|_| rng.random_range(0..poly_count as u32))
        .collect();
    let xs = rand_ext(&mut rng, n_evals);

    let cpu_coeffs = cpu.copy_from_elem("coeffs", &coeffs);
    let cpu_which = cpu.copy_from_u32("which", &which);
    let cpu_xs = cpu.copy_from_extelem("xs", &xs);
    let cpu_out = cpu.alloc_extelem("out", n_evals);
    cpu.batch_evaluate_any(&cpu_coeffs, poly_count, &cpu_which, &cpu_xs, &cpu_out);

    let mut padded = vec![BabyBearElem::new(7); offset];
    padded.extend_from_slice(&coeffs);
    let m_all = metal.copy_from_elem("coeffs", &padded);
    let m_coeffs = m_all.slice(offset, deg * poly_count);
    let m_which = metal.copy_from_u32("which", &which);
    let m_xs = metal.copy_from_extelem("xs", &xs);
    let m_out = metal.alloc_extelem("out", n_evals);
    let path = if force.is_none() {
        // Through the Hal entry point, then confirm the plan it used.
        metal.batch_evaluate_any(&m_coeffs, poly_count, &m_which, &m_xs, &m_out);
        eval_plan(deg as u32, n_evals as u32, m_coeffs.as_device_ptr())
    } else {
        eval::batch_evaluate_any(
            &metal, &m_coeffs, poly_count, &m_which, &m_xs, &m_out, force,
        )
    };
    assert_eq!(path, expect, "deg {deg} n_evals {n_evals} offset {offset}");
    assert_eq!(
        cpu_out.to_vec(),
        m_out.to_vec(),
        "deg {deg} poly_count {poly_count} n_evals {n_evals} offset {offset} path {path:?}"
    );
}

#[test]
fn eval_horner_small_degrees() {
    // deg < 256 (most threads idle), deg = 256, deg not a multiple of 4096 below the gate.
    for (i, deg) in [1usize, 2, 16, 255 + 1, 1024, 2048].into_iter().enumerate() {
        eval_case(10 + i as u64, deg, 3, 37, 0, None, EvalPath::Horner);
    }
}

#[test]
fn eval_horner_misaligned() {
    // deg passes the size gate but coeffs are at a 4/8/12-byte offset: Horner.
    for offset in [1, 2, 3] {
        eval_case(
            20 + offset as u64,
            1 << 13,
            5,
            41,
            offset,
            None,
            EvalPath::Horner,
        );
    }
    // 16-byte aligned slice offset: eu2.
    eval_case(
        24,
        1 << 13,
        5,
        41,
        4,
        None,
        EvalPath::Eu2 {
            s: 2,
            chunk: 1 << 12,
        },
    );
}

#[test]
fn eval_no_evals() {
    let metal = MetalHalSha256::new();
    let coeffs = metal.copy_from_elem("coeffs", &vec![BabyBearElem::new(1); 1 << 12]);
    let which = metal.alloc_u32("which", 0);
    let xs = metal.alloc_extelem("xs", 0);
    let out = metal.alloc_extelem("out", 0);
    let path = eval::batch_evaluate_any(&metal, &coeffs, 1, &which, &xs, &out, None);
    assert_eq!(path, EvalPath::Horner);
    metal.sync();
}

#[test]
fn eval_eu2_shapes() {
    // S = 1, one iteration (deg = 4096).
    eval_case(30, 4096, 4, 7, 0, None, EvalPath::Eu2 { s: 1, chunk: 4096 });
    // S capped at deg / 4096 (nEvals 1).
    eval_case(
        31,
        1 << 16,
        2,
        1,
        0,
        None,
        EvalPath::Eu2 { s: 16, chunk: 4096 },
    );
    // The testutil shape: deg 2^16, 865 evals → S = 2, chunk 2^15 (8 iterations).
    eval_case(
        32,
        1 << 16,
        223,
        865,
        0,
        None,
        EvalPath::Eu2 {
            s: 2,
            chunk: 1 << 15,
        },
    );
    // nEvals ≥ 1024 → S = 1, chunk = deg (16 iterations).
    eval_case(
        33,
        1 << 16,
        3,
        1100,
        0,
        None,
        EvalPath::Eu2 {
            s: 1,
            chunk: 1 << 16,
        },
    );
    // nEvals not a power of two, S between the bounds.
    eval_case(
        34,
        1 << 14,
        9,
        300,
        0,
        None,
        EvalPath::Eu2 { s: 4, chunk: 4096 },
    );
    // Large deg, many blocks per eval.
    eval_case(
        35,
        1 << 20,
        2,
        4,
        0,
        None,
        EvalPath::Eu2 {
            s: 256,
            chunk: 4096,
        },
    );
}

#[test]
fn eval_forced_paths_agree() {
    // Every kernel path on the same input: Horner at an eu2-eligible shape, and eu2 at every
    // legal S for deg 2^15.
    eval_case(
        40,
        1 << 15,
        6,
        50,
        0,
        Some(EvalPath::Horner),
        EvalPath::Horner,
    );
    for s in [1u32, 2, 4, 8] {
        let p = EvalPath::Eu2 {
            s,
            chunk: (1 << 15) / s,
        };
        eval_case(40, 1 << 15, 6, 50, 0, Some(p), p);
    }
}

#[test]
fn eval_dual_hal() {
    // The DualHal path (as the prover uses it) at a few shapes.
    for (deg, polys, evals) in [
        (1usize << 12, 3usize, 5usize),
        (1 << 10, 4, 9),
        (1 << 16, 2, 70),
    ] {
        let mut rng = rng(deg as u64 + evals as u64);
        let metal = MetalHalSha256::new();
        let cpu = CpuHal::new(metal.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(cpu), Rc::new(metal));
        let coeffs = hal.copy_from_elem("coeffs", &rand_elems(&mut rng, deg * polys));
        let which: Vec<u32> = (0..evals).map(|i| (i % polys) as u32).collect();
        let which = hal.copy_from_u32("which", &which);
        let xs = hal.copy_from_extelem("xs", &rand_ext(&mut rng, evals));
        let out = hal.alloc_extelem("out", evals);
        hal.batch_evaluate_any(&coeffs, polys, &which, &xs, &out);
    }
}

// ---- mix_poly_coeffs ----

#[test]
fn mix_plan_gates() {
    assert_eq!(mix_plan(1), MixKernel::V2);
    assert_eq!(mix_plan(1638), MixKernel::V2);
    assert_eq!(mix_plan(1639), MIX_MID_KERNEL);
    assert_eq!(mix_plan(2048), MIX_MID_KERNEL);
    assert_eq!(mix_plan(2049), MixKernel::NoTable);
    assert_eq!(MixKernel::V2.tg_bytes(MIX_V2_MAX_INPUTS), 32760);
    assert!(MixKernel::V2.tg_bytes(MIX_V2_MAX_INPUTS + 1) > 32768);
    assert_eq!(MixKernel::V1Table.tg_bytes(MIX_TABLE_MAX_INPUTS), 32768);
    assert_eq!(
        MixKernel::V2DeviceCids.tg_bytes(MIX_TABLE_MAX_INPUTS),
        32768
    );
}

/// Combo ids with runs of 1, 2, 3, ... equal ids (exercises v2's pairing and run-end RMW),
/// some ids repeated non-contiguously, over `n_combos` outputs.
fn combo_ids(rng: &mut SmallRng, input_size: usize, n_combos: usize) -> Vec<u32> {
    let mut ids = Vec::with_capacity(input_size);
    let mut run = 1;
    while ids.len() < input_size {
        let id = rng.random_range(0..n_combos as u32);
        for _ in 0..run {
            if ids.len() < input_size {
                ids.push(id);
            }
        }
        run = run % 5 + 1;
    }
    ids
}

/// One mix case: Metal (with `force`, or the Hal entry point) == CPU bit for bit, including the
/// read-modify-write of a random initial output.
fn mix_case(
    seed: u64,
    input_size: usize,
    count: usize,
    force: Option<MixKernel>,
    expect: MixKernel,
) {
    let mut rng = rng(seed);
    let n_combos = 5;
    let metal = MetalHalSha256::new();
    let cpu = CpuHal::new(metal.get_hash_suite().clone());

    let input = rand_elems(&mut rng, input_size * count);
    let combos = combo_ids(&mut rng, input_size, n_combos);
    let out0 = rand_ext(&mut rng, n_combos * count);
    let mix_start = BabyBearExtElem::random(&mut rng);
    let mix = BabyBearExtElem::random(&mut rng);

    let c_out = cpu.copy_from_extelem("out", &out0);
    let c_in = cpu.copy_from_elem("in", &input);
    let c_combos = cpu.copy_from_u32("combos", &combos);
    cpu.mix_poly_coeffs(
        &c_out, &mix_start, &mix, &c_in, &c_combos, input_size, count,
    );

    let m_out = metal.copy_from_extelem("out", &out0);
    let m_in = metal.copy_from_elem("in", &input);
    let m_combos = metal.copy_from_u32("combos", &combos);
    let used = if force.is_none() {
        metal.mix_poly_coeffs(
            &m_out, &mix_start, &mix, &m_in, &m_combos, input_size, count,
        );
        mix_plan(input_size)
    } else {
        eval::mix_poly_coeffs(
            &metal, &m_out, &mix_start, &mix, &m_in, &m_combos, input_size, count, force,
        )
    };
    assert_eq!(used, expect);
    assert!(
        c_out.to_vec() == m_out.to_vec(),
        "mix mismatch: input_size {input_size} count {count} kernel {used:?}"
    );
}

#[test]
fn mix_gates_default_path() {
    // Around both gates, with counts that are not multiples of 256.
    for (i, &input_size) in [1usize, 2, 7, 100, 1637, 1638, 1639, 2047, 2048, 2049, 2500]
        .iter()
        .enumerate()
    {
        for (j, &count) in [1usize, 255, 257, 1000].iter().enumerate() {
            mix_case(
                (100 + i * 10 + j) as u64,
                input_size,
                count,
                None,
                mix_plan(input_size),
            );
        }
    }
}

#[test]
fn mix_every_kernel() {
    // Each kernel at the largest size it is legal for, and at a small one.
    for (input_size, kernels) in [
        (
            3usize,
            &[
                MixKernel::V2,
                MixKernel::V2DeviceCids,
                MixKernel::V1Table,
                MixKernel::NoTable,
            ][..],
        ),
        (
            MIX_V2_MAX_INPUTS,
            &[
                MixKernel::V2,
                MixKernel::V2DeviceCids,
                MixKernel::V1Table,
                MixKernel::NoTable,
            ][..],
        ),
        (
            MIX_TABLE_MAX_INPUTS,
            &[
                MixKernel::V2DeviceCids,
                MixKernel::V1Table,
                MixKernel::NoTable,
            ][..],
        ),
    ] {
        for &k in kernels {
            mix_case(200 + input_size as u64, input_size, 777, Some(k), k);
        }
    }
}

#[test]
fn mix_large_count() {
    mix_case(300, 64, (1 << 16) + 3, None, MixKernel::V2);
}

#[test]
#[should_panic(expected = "threadgroup memory")]
fn mix_v2_over_32k_panics() {
    mix_case(
        400,
        MIX_V2_MAX_INPUTS + 1,
        16,
        Some(MixKernel::V2),
        MixKernel::V2,
    );
}

#[test]
fn mix_dual_hal() {
    let mut rng = rng(500);
    let metal = MetalHalSha256::new();
    let cpu = CpuHal::new(metal.get_hash_suite().clone());
    let hal = DualHal::new(Rc::new(cpu), Rc::new(metal));
    let (input_size, count, n_combos) = (300, 1023, 4);
    let out = hal.copy_from_extelem("out", &rand_ext(&mut rng, n_combos * count));
    let input = hal.copy_from_elem("in", &rand_elems(&mut rng, input_size * count));
    let combos = hal.copy_from_u32("combos", &combo_ids(&mut rng, input_size, n_combos));
    let (ms, m) = (
        BabyBearExtElem::random(&mut rng),
        BabyBearExtElem::random(&mut rng),
    );
    hal.mix_poly_coeffs(&out, &ms, &m, &input, &combos, input_size, count);
    hal.mix_poly_coeffs(&out, &m, &ms, &input, &combos, input_size, count);
}

// ---- indicative timings (not a conformance test) ----

/// Wall time per call over `reps` calls encoded into one batch, after a warm-up call.
fn time_per_call(metal: &MetalHalSha256, reps: u32, mut f: impl FnMut()) -> f64 {
    f();
    metal.sync();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    metal.sync();
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

#[test]
#[ignore = "timing run: cargo test ... tests_z4::z4_timings -- --ignored --nocapture"]
fn z4_timings() {
    crate::hal::metal::require_gpu_run();
    let mut rng = rng(900);
    let metal = MetalHalSha256::new();
    let reps = 20;
    for (deg, polys, n_evals) in [
        (1usize << 16, 223usize, 865usize),
        (1 << 20, 4, 8),
        (1 << 22, 2, 3),
    ] {
        let coeffs = metal.copy_from_elem("coeffs", &rand_elems(&mut rng, deg * polys));
        let which: Vec<u32> = (0..n_evals).map(|i| (i % polys) as u32).collect();
        let which = metal.copy_from_u32("which", &which);
        let xs = metal.copy_from_extelem("xs", &rand_ext(&mut rng, n_evals));
        let out = metal.alloc_extelem("out", n_evals);
        let plan = eval_plan(deg as u32, n_evals as u32, coeffs.as_device_ptr());
        for p in [plan, EvalPath::Horner] {
            let ms = time_per_call(&metal, reps, || {
                eval::batch_evaluate_any(&metal, &coeffs, polys, &which, &xs, &out, Some(p));
            });
            println!(
                "eval deg 2^{} nEvals {n_evals} {p:?}: {ms:.3} ms",
                deg.trailing_zeros()
            );
        }
    }
    let count = 1 << 18;
    let n_combos = 8;
    for input_size in [100usize, 1000, 1638, 2000, 2048, 2500] {
        let out = metal.alloc_extelem("out", n_combos * count);
        let input = metal.copy_from_elem("in", &rand_elems(&mut rng, input_size * count));
        let combos = metal.copy_from_u32("combos", &combo_ids(&mut rng, input_size, n_combos));
        let (ms0, m) = (BabyBearExtElem::random(&mut rng), BabyBearExtElem::ONE);
        let mut kernels = vec![MixKernel::NoTable];
        if input_size <= MIX_TABLE_MAX_INPUTS {
            kernels.extend([MixKernel::V1Table, MixKernel::V2DeviceCids]);
        }
        if input_size <= MIX_V2_MAX_INPUTS {
            kernels.push(MixKernel::V2);
        }
        for k in kernels {
            let ms = time_per_call(&metal, reps, || {
                eval::mix_poly_coeffs(
                    &metal,
                    &out,
                    &ms0,
                    &m,
                    &input,
                    &combos,
                    input_size,
                    count,
                    Some(k),
                );
            });
            println!("mix count 2^18 inputSize {input_size} {k:?}: {ms:.3} ms");
        }
    }
}
