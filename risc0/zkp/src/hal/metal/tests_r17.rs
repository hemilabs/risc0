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

//! R17 audit: repeated-run determinism stress of the zkp Metal kernels (the Apple GPU back-end
//! defect R17 found in rv32im witgen loses a private-array store; `tools/metal_audit/ir_scan.py`
//! flags the kernels below as having the same IR shape). Every kernel runs `R17_AUDIT_REPS`
//! times (default 10) on one large input and each output is compared bit for bit with the first
//! run and with the CPU HAL. Run after a toolchain or macOS update:
//!
//! `R17_AUDIT_REPS=30 R17_AUDIT_PO2=20 cargo test -p risc0-zkp --features metal --lib -- \
//!  hal::metal::tests_r17 --ignored --nocapture --test-threads 1`
//!
//! Each case prints `R17AUDIT zkp <case>: <k>/<n> reps differ from rep 0, <m>/<n> differ from
//! CPU`; the test fails if any k or m is nonzero.

use rand::{rngs::SmallRng, Rng, SeedableRng};
use risc0_core::field::{
    baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
    Elem as _,
};
use test_log::test;

use super::{
    eval::{self, EvalPath, MixKernel},
    mdiv::{host_combos_divide, mdiv_shape},
    MetalHalPoseidon2, MetalHalSha256,
};
use crate::{
    hal::{cpu::CpuHal, Buffer, Hal},
    FRI_FOLD,
};

fn env_or(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn reps() -> usize {
    env_or("R17_AUDIT_REPS", 10)
}

fn po2() -> usize {
    env_or("R17_AUDIT_PO2", 20)
}

fn rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

fn elems(rng: &mut SmallRng, n: usize) -> Vec<BabyBearElem> {
    (0..n).map(|_| BabyBearElem::random(rng)).collect()
}

fn exts(rng: &mut SmallRng, n: usize) -> Vec<BabyBearExtElem> {
    (0..n).map(|_| BabyBearExtElem::random(rng)).collect()
}

/// Runs `run` `reps` times; returns (reps differing from rep 0, reps differing from `cpu`).
fn stress<T: PartialEq>(
    case: &str,
    cpu: &[T],
    reps: usize,
    mut run: impl FnMut() -> Vec<T>,
) -> (usize, usize) {
    let mut first: Option<Vec<T>> = None;
    let (mut vs_first, mut vs_cpu) = (0, 0);
    let mut detail = Vec::new();
    for rep in 0..reps {
        let got = run();
        assert_eq!(got.len(), cpu.len(), "{case}: output length");
        let n_cpu = got.iter().zip(cpu).filter(|(a, b)| a != b).count();
        if n_cpu > 0 {
            vs_cpu += 1;
            detail.push(format!("rep {rep}: {n_cpu} differ from CPU"));
        }
        match &first {
            None => first = Some(got),
            Some(f) => {
                let n = got.iter().zip(f).filter(|(a, b)| a != b).count();
                if n > 0 {
                    vs_first += 1;
                    detail.push(format!("rep {rep}: {n} differ from rep 0"));
                }
            }
        }
    }
    println!(
        "R17AUDIT zkp {case}: {vs_first}/{reps} reps differ from rep 0, {vs_cpu}/{reps} differ \
         from CPU {detail:?}"
    );
    (vs_first, vs_cpu)
}

struct Tally(Vec<(String, usize, usize)>);

impl Tally {
    fn add(&mut self, case: &str, r: (usize, usize)) {
        self.0.push((case.to_string(), r.0, r.1));
    }

    fn check(self) {
        let bad: Vec<_> = self.0.iter().filter(|(_, a, b)| a + b > 0).collect();
        assert!(
            bad.is_empty(),
            "R17AUDIT nondeterministic or wrong: {bad:?}"
        );
    }
}

/// batch_evaluate_any: Horner (forced) and the eu2 path (`eu2_pows`, `eu2_main`, `eu2_reduce`).
#[test]
#[ignore]
fn r17_eval() {
    let (reps, po2) = (reps(), po2());
    let mut t = Tally(vec![]);
    let metal = MetalHalSha256::new();
    let cpu = CpuHal::<BabyBear>::new(metal.get_hash_suite().clone());
    let mut rng = rng(17);
    // rv32im-like: many polys of 2^po2 coefficients, ~1 eval per poly plus extras.
    let deg = 1usize << po2.min(20);
    let poly_count = 24;
    let n_evals = 64;
    let coeffs = elems(&mut rng, deg * poly_count);
    let which: Vec<u32> = (0..n_evals)
        .map(|_| rng.random_range(0..poly_count as u32))
        .collect();
    let xs = exts(&mut rng, n_evals);
    let c_out = cpu.alloc_extelem("out", n_evals);
    cpu.batch_evaluate_any(
        &cpu.copy_from_elem("c", &coeffs),
        poly_count,
        &cpu.copy_from_u32("w", &which),
        &cpu.copy_from_extelem("x", &xs),
        &c_out,
    );
    let want = c_out.to_vec();
    let m_coeffs = metal.copy_from_elem("c", &coeffs);
    let m_which = metal.copy_from_u32("w", &which);
    let m_xs = metal.copy_from_extelem("x", &xs);
    let plan = eval::eval_plan(deg as u32, n_evals as u32, m_coeffs.as_device_ptr());
    for (name, force) in [("eval/plan", None), ("eval/horner", Some(EvalPath::Horner))] {
        let case = format!("{name} (plan {plan:?}) deg 2^{} x{poly_count}", po2.min(20));
        let r = stress(&case, &want, reps, || {
            let out = metal.alloc_extelem("out", n_evals);
            eval::batch_evaluate_any(&metal, &m_coeffs, poly_count, &m_which, &m_xs, &out, force);
            out.to_vec()
        });
        t.add(&case, r);
    }
    t.check();
}

/// mix_poly_coeffs: every kernel (V2 / V2DeviceCids / V1Table / NoTable).
#[test]
#[ignore]
fn r17_mix() {
    let (reps, po2) = (reps(), po2());
    let mut t = Tally(vec![]);
    let metal = MetalHalSha256::new();
    let cpu = CpuHal::<BabyBear>::new(metal.get_hash_suite().clone());
    let mut rng = rng(18);
    let count = 1usize << po2;
    for (input_size, kernels) in [
        (
            64usize,
            vec![
                MixKernel::V2,
                MixKernel::V2DeviceCids,
                MixKernel::V1Table,
                MixKernel::NoTable,
            ],
        ),
        (
            400,
            vec![MixKernel::V2, MixKernel::V2DeviceCids, MixKernel::NoTable],
        ),
    ] {
        let n_combos = 5;
        let input = elems(&mut rng, input_size * count);
        let combos: Vec<u32> = (0..input_size)
            .map(|_| rng.random_range(0..n_combos as u32))
            .collect();
        let out0 = exts(&mut rng, n_combos * count);
        let mix_start = BabyBearExtElem::random(&mut rng);
        let mix = BabyBearExtElem::random(&mut rng);
        let c_out = cpu.copy_from_extelem("out", &out0);
        cpu.mix_poly_coeffs(
            &c_out,
            &mix_start,
            &mix,
            &cpu.copy_from_elem("in", &input),
            &cpu.copy_from_u32("combos", &combos),
            input_size,
            count,
        );
        let want = c_out.to_vec();
        let m_in = metal.copy_from_elem("in", &input);
        let m_combos = metal.copy_from_u32("combos", &combos);
        for k in kernels {
            let case = format!("mix/{k:?} input_size {input_size} count 2^{po2}");
            let r = stress(&case, &want, reps, || {
                let out = metal.copy_from_extelem("out", &out0);
                eval::mix_poly_coeffs(
                    &metal,
                    &out,
                    &mix_start,
                    &mix,
                    &m_in,
                    &m_combos,
                    input_size,
                    count,
                    Some(k),
                );
                out.to_vec()
            });
            t.add(&case, r);
        }
        // Z15: the fused write-once mix (one group, garbage initial contents) against the CPU
        // mix into zeroes.
        let c_zero = cpu.alloc_extelem_zeroed("out", n_combos * count);
        cpu.mix_poly_coeffs(
            &c_zero,
            &BabyBearExtElem::ONE,
            &mix,
            &cpu.copy_from_elem("in", &input),
            &cpu.copy_from_u32("combos", &combos),
            input_size,
            count,
        );
        let want = c_zero.to_vec();
        let case = format!("mix/Z15 fused input_size {input_size} count 2^{po2}");
        let r = stress(&case, &want, reps, || {
            let out = metal.copy_from_extelem("out", &out0);
            let fused = super::mix::mix_poly_coeffs_fused(
                &metal,
                &out,
                &mix,
                &[(&m_in, combos.as_slice())],
                count,
                false,
            );
            assert!(fused);
            out.to_vec()
        });
        t.add(&case, r);
    }
    t.check();
}

/// fri_fold and prefix_products.
#[test]
#[ignore]
fn r17_fri_prefix() {
    let (reps, po2) = (reps(), po2());
    let mut t = Tally(vec![]);
    let metal = MetalHalSha256::new();
    let cpu = CpuHal::<BabyBear>::new(metal.get_hash_suite().clone());
    let mut rng = rng(19);
    let count = 1usize << (po2 - 2);
    let input = elems(&mut rng, count * 4 * FRI_FOLD);
    let mix = BabyBearExtElem::random(&mut rng);
    let c_out = cpu.alloc_elem("out", count * 4);
    cpu.fri_fold(&c_out, &cpu.copy_from_elem("in", &input), &mix);
    let want = c_out.to_vec();
    let m_in = metal.copy_from_elem("in", &input);
    let case = format!("fri_fold count 2^{}", po2 - 2);
    let r = stress(&case, &want, reps, || {
        let out = metal.alloc_elem("out", count * 4);
        metal.fri_fold(&out, &m_in, &mix);
        out.to_vec()
    });
    t.add(&case, r);

    let io = exts(&mut rng, 1 << po2);
    let c_io = cpu.copy_from_extelem("io", &io);
    cpu.prefix_products(&c_io);
    let want = c_io.to_vec();
    let case = format!("prefix_products 2^{po2}");
    let r = stress(&case, &want, reps, || {
        let m = metal.copy_from_extelem("io", &io);
        metal.prefix_products(&m);
        m.to_vec()
    });
    t.add(&case, r);
    t.check();
}

/// combos_divide (mdiv): every scan width (`mdiv_scan_32` .. `_1024`) with e = 8 and e = 16.
#[test]
#[ignore]
fn r17_mdiv() {
    let reps = reps();
    let mut t = Tally(vec![]);
    let metal = MetalHalSha256::new();
    let mut rng = rng(20);
    // rv32im-like: 6 slices, 1..3 roots each.
    let roots = [1usize, 2, 3, 1, 2, 1];
    let max_po2 = env_or("R17_AUDIT_MDIV_MAX_PO2", 22);
    for po2 in 15..=max_po2 {
        let n = 1usize << po2;
        let slices = roots.len();
        let data = exts(&mut rng, slices * n);
        let chunks: Vec<(usize, Vec<BabyBearExtElem>)> =
            (0..slices).map(|c| (c, exts(&mut rng, roots[c]))).collect();
        // Random slices: nonzero remainders, so the reference is the CUDA-semantics host loop
        // (the CPU HAL panics on a nonzero remainder).
        let mut want = data.clone();
        let _ = host_combos_divide(&mut want, &chunks, n);
        for e in [8usize, 16] {
            let Some(shape) = mdiv_shape(n, e) else {
                continue;
            };
            let case = format!("mdiv e{e} po2 {po2} {shape:?}");
            let r = stress(&case, &want, reps, || {
                let m = metal.copy_from_extelem("combos", &data);
                let _ = metal.mdiv_combos_divide_with(&m, &chunks, n, Some(shape));
                m.to_vec()
            });
            t.add(&case, r);
        }
    }
    t.check();
}

/// hash_rows / hash_fold for Poseidon2 (pv.metal: the scanner's tier-B permutation-state loops)
/// and SHA-256.
#[test]
#[ignore]
fn r17_hash() {
    let (reps, po2) = (reps(), po2());
    let mut t = Tally(vec![]);
    let mut rng = rng(21);
    let rows = 1usize << po2;
    for (name, metal) in [
        (
            "poseidon2",
            Box::new(MetalHalPoseidon2::new()) as Box<dyn HashCase>,
        ),
        (
            "sha256",
            Box::new(MetalHalSha256::new()) as Box<dyn HashCase>,
        ),
    ] {
        for cols in [16usize, 200] {
            let matrix = elems(&mut rng, rows * cols);
            let (want_rows, want_fold) = metal.cpu_ref(&matrix, rows);
            let case = format!("hash_rows/{name} {cols}x2^{po2}");
            let r = stress(&case, &want_rows, reps, || metal.rows(&matrix, rows));
            t.add(&case, r);
            let case = format!("hash_fold/{name} 2^{po2}");
            let r = stress(&case, &want_fold, reps, || metal.fold(&want_rows));
            t.add(&case, r);
        }
    }
    // Z14: the row hash with a commit's coefficient bit reversal (threadgroup-memory tile pairs)
    // folded in (`pv_rows_side_k_z9`): digests and reversed coefficients, as u32 words.
    {
        let hal = MetalHalPoseidon2::new();
        let cols = 200;
        let (lg, count) = (po2 - 2, 23);
        let matrix = elems(&mut rng, rows * cols);
        let coeffs_in = elems(&mut rng, count << lg);
        assert!(hal.fuses_bit_reverse_into_hash(rows, cols, lg));
        let words = |d: &[crate::core::digest::Digest], c: &[BabyBearElem]| -> Vec<u32> {
            let mut w: Vec<u32> = d.iter().flat_map(|d| d.as_words().to_vec()).collect();
            w.extend(c.iter().map(|x| x.as_u32()));
            w
        };
        let want = {
            let (want_rows, _) = (&hal as &dyn HashCase).cpu_ref(&matrix, rows);
            let cpu = CpuHal::<BabyBear>::new(hal.get_hash_suite().clone());
            let c = cpu.copy_from_elem("c", &coeffs_in);
            cpu.batch_bit_reverse(&c, count);
            words(&want_rows, &c.to_vec())
        };
        let m = hal.copy_from_elem("m", &matrix);
        let case = format!("hash_rows_bit_reverse/poseidon2 {cols}x2^{po2}, {count}x2^{lg}");
        let r = stress(&case, &want, reps, || {
            let out = hal.alloc_digest("rows", rows);
            let c = hal.copy_from_elem("c", &coeffs_in);
            hal.hash_rows_bit_reverse(&out, &m, &c, count);
            words(&out.to_vec(), &c.to_vec())
        });
        t.add(&case, r);
    }
    t.check();
}

/// Hash entry points of a Metal HAL, type-erased so one loop covers both suites.
trait HashCase {
    fn cpu_ref(
        &self,
        matrix: &[BabyBearElem],
        rows: usize,
    ) -> (
        Vec<crate::core::digest::Digest>,
        Vec<crate::core::digest::Digest>,
    );
    fn rows(&self, matrix: &[BabyBearElem], rows: usize) -> Vec<crate::core::digest::Digest>;
    fn fold(&self, leaves: &[crate::core::digest::Digest]) -> Vec<crate::core::digest::Digest>;
}

impl<H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>> HashCase for H {
    fn cpu_ref(
        &self,
        matrix: &[BabyBearElem],
        rows: usize,
    ) -> (
        Vec<crate::core::digest::Digest>,
        Vec<crate::core::digest::Digest>,
    ) {
        let cpu = CpuHal::<BabyBear>::new(self.get_hash_suite().clone());
        let out = cpu.alloc_digest("rows", rows);
        cpu.hash_rows(&out, &cpu.copy_from_elem("m", matrix));
        let leaves = out.to_vec();
        let mut init = vec![crate::core::digest::Digest::ZERO; rows];
        init.extend_from_slice(&leaves);
        let io = cpu.copy_from_digest("io", &init);
        cpu.hash_fold(&io, rows, rows / 2);
        (leaves, io.to_vec()[rows / 2..rows].to_vec())
    }

    fn rows(&self, matrix: &[BabyBearElem], rows: usize) -> Vec<crate::core::digest::Digest> {
        let out = self.alloc_digest("rows", rows);
        self.hash_rows(&out, &self.copy_from_elem("m", matrix));
        out.to_vec()
    }

    fn fold(&self, leaves: &[crate::core::digest::Digest]) -> Vec<crate::core::digest::Digest> {
        let rows = leaves.len();
        let mut init = vec![crate::core::digest::Digest::ZERO; rows];
        init.extend_from_slice(leaves);
        let io = self.copy_from_digest("io", &init);
        self.hash_fold(&io, rows, rows / 2);
        io.to_vec()[rows / 2..rows].to_vec()
    }
}

/// The R17 reproducer (rv32im witgen's defect reduced to 35 lines), compiled at run time: a
/// constructor-initialised private array, a non-unrolled loop calling a `noinline` function and
/// storing into the array; the last element reads back its initial value (77). `CONTROL`
/// initialises the array with an index loop instead (correct on the affected GPU).
const R17_REPRO: &str = r#"
#include <metal_stdlib>
using namespace metal;
__attribute__((noinline)) static void put(device uint* row, uint col, uint v) { row[col] = v; }
struct Elem { uint v; Elem() : v(77u) {} };
struct ElemI { uint v; };
#ifdef CONTROL
struct A3 { ElemI e[3]; A3() { for (uint i = 0; i < 3; i++) e[i].v = 77u; } };
#else
struct A3 { Elem e[3]; };
#endif
kernel void test_kernel(device uint* scratch [[buffer(0)]], device uint* out [[buffer(1)]],
                        constant uint4& prm [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= prm.x) return;
  device uint* row = scratch + gid * prm.z;
  A3 x;
#pragma nounroll
  for (uint i = 0; i < 3; i++) {
    uint v = (gid + i + 1) * 2654435761u;
    put(row, i, v);
    x.e[i].v = v;
  }
  uint code = 0;
  for (uint i = 0; i < 3; i++)
    if (x.e[i].v != (gid + i + 1) * 2654435761u) code |= (1u << i);
  out[gid] = code;
}
"#;

/// Positive control for the R17 stress tests: is the Apple GPU defect present on this
/// machine/toolchain now? Prints `R17AUDIT control repro: k/n dispatches wrong` (expected k = n
/// while the defect exists; 0 once Apple fixes it) and the index-init control (expected 0). Does
/// not fail on the defect: it reports it. Fails only if the control variant is wrong.
#[test]
#[ignore]
fn r17_defect_control() {
    let reps = reps();
    let n: u32 = env_or("R17_AUDIT_CONTROL_THREADS", 1 << 20) as u32;
    let device = metal::Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let mut control_bad = 0;
    for (name, control) in [("repro", false), ("index-init control", true)] {
        let src = format!(
            "{}{R17_REPRO}",
            if control { "#define CONTROL 1\n" } else { "" }
        );
        let lib = device
            .new_library_with_source(&src, &metal::CompileOptions::new())
            .unwrap();
        let f = lib.get_function("test_kernel", None).unwrap();
        let pso = device.new_compute_pipeline_state_with_function(&f).unwrap();
        let stride = 4u32;
        let scratch = device.new_buffer(
            n as u64 * stride as u64 * 4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let out = device.new_buffer(n as u64 * 4, metal::MTLResourceOptions::StorageModeShared);
        let prm = [n, 0, stride, 0];
        let (mut bad_dispatches, mut bad_threads) = (0, 0u64);
        for _ in 0..reps {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&scratch), 0);
            enc.set_buffer(1, Some(&out), 0);
            enc.set_bytes(2, 16, prm.as_ptr() as *const _);
            enc.dispatch_threads(
                metal::MTLSize::new(n as u64, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let codes =
                unsafe { std::slice::from_raw_parts(out.contents() as *const u32, n as usize) };
            let wrong = codes.iter().filter(|c| **c != 0).count() as u64;
            if wrong > 0 {
                bad_dispatches += 1;
                bad_threads += wrong;
            }
        }
        println!(
            "R17AUDIT control {name}: {bad_dispatches}/{reps} dispatches of {n} threads wrong \
             ({bad_threads} wrong threads in all)"
        );
        if control {
            control_bad = bad_dispatches;
        }
    }
    assert_eq!(control_bad, 0, "the index-init control must be correct");
}
