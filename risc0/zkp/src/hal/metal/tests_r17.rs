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

/// W7-T1: the shape Z13 reported as a second R17-family defect (in an NTT prototype that no longer
/// exists): a private `uint v[16]` in fully inlined code (every helper `always_inline`, loops
/// `#pragma unroll`, no call anywhere) gets a conditional element-wise update, `if (bz != 0) { for
/// m: v[m] = mul(v[m], t); t = mul(t, st); }`; then butterflies, a threadgroup exchange and the
/// store. `CONTROL` applies the twist unconditionally, as an identity for bz == 0 (t = st =
/// Montgomery one). Both compute the same values (the CPU reference `z13_ref`).
///
/// Fix round 1: the first version's CONTROL and reference set only t = ONE for bz == 0 and kept
/// st, i.e. multiplied those blocks by st^m, which the guarded kernel never does. Its "k = n,
/// exactly the bz == 0 blocks" was that reference bug, not a GPU defect. With the semantics fixed
/// the shape is correct on this GPU (macOS 27.2, M5 Max): it is NOT a reproduction of Z13's
/// nondeterministic failure (see docs/metal-port/r17/z13/README.md).
const R17_Z13: &str = r#"
#include <metal_stdlib>
using namespace metal;
constant uint P = 0x78000001u;
constant uint MI = 0x77ffffffu;
constant uint ONE = 0x0ffffffeu;
static inline __attribute__((always_inline)) uint mul(uint a, uint b) {
  ulong o = ulong(a) * ulong(b);
  uint red = uint(o) * MI;
  o += ulong(red) * ulong(P);
  uint r = uint(o >> 32);
  return r >= P ? r - P : r;
}
static inline __attribute__((always_inline)) uint add(uint a, uint b) { uint r = a + b; return r >= P ? r - P : r; }
static inline __attribute__((always_inline)) uint sub(uint a, uint b) { return a >= b ? a - b : a + P - b; }
static inline __attribute__((always_inline)) void bfly(thread uint (&v)[16], constant uint* w) {
#pragma unroll
  for (uint s = 0; s < 4; s++) {
    uint h = 8u >> s;
#pragma unroll
    for (uint i = 0; i < 16; i++) {
      if ((i & h) == 0) {
        uint a = v[i], b = mul(v[i + h], w[s * 16 + i]);
        v[i] = add(a, b); v[i + h] = sub(a, b);
      }
    }
  }
}
kernel void z13_shape(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                      constant uint4& prm [[buffer(2)]], constant uint* w [[buffer(3)]],
                      threadgroup uint* sh [[threadgroup(0)]],
                      uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                      uint3 tpg [[threads_per_threadgroup]]) {
  uint n = prm.x, bz = tg.y, T = tpg.x;
  uint col = tg.x * T + lid;
  uint v[16];
#pragma unroll
  for (uint m = 0; m < 16; m++) v[m] = in[(bz * 16 + m) * n + col];
  uint t = mul(prm.y, bz + col + 1u), st = prm.z;
#ifdef CONTROL
  if (bz == 0) { t = ONE; st = ONE; }
#pragma unroll
  for (uint m = 0; m < 16; m++) { v[m] = mul(v[m], t); t = mul(t, st); }
#else
  if (bz != 0) {
#pragma unroll
    for (uint m = 0; m < 16; m++) { v[m] = mul(v[m], t); t = mul(t, st); }
  }
#endif
  bfly(v, w);
#pragma unroll
  for (uint m = 0; m < 16; m++) sh[m * T + lid] = v[(m + lid) & 15u];
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (uint m = 0; m < 16; m++) v[m] = sh[m * T + ((lid + m) % T)];
  bfly(v, w);
#pragma unroll
  for (uint m = 0; m < 16; m++) out[(bz * 16 + m) * n + col] = v[m];
}
"#;

mod z13_ref {
    //! CPU reference of `R17_Z13` (both variants compute this): the bz == 0 blocks get no twist.
    pub const P: u32 = 0x7800_0001;
    const MI: u32 = 0x77ff_ffff;

    pub fn mul(a: u32, b: u32) -> u32 {
        let o = a as u64 * b as u64;
        let red = (o as u32).wrapping_mul(MI);
        let r = ((o + red as u64 * P as u64) >> 32) as u32;
        if r >= P {
            r - P
        } else {
            r
        }
    }

    fn add(a: u32, b: u32) -> u32 {
        let r = a + b;
        if r >= P {
            r - P
        } else {
            r
        }
    }

    fn sub(a: u32, b: u32) -> u32 {
        if a >= b {
            a - b
        } else {
            a + P - b
        }
    }

    fn bfly(v: &mut [u32; 16], w: &[u32]) {
        for s in 0..4 {
            let h = 8 >> s;
            for i in 0..16 {
                if i & h == 0 {
                    let (a, b) = (v[i], mul(v[i + h], w[s * 16 + i]));
                    v[i] = add(a, b);
                    v[i + h] = sub(a, b);
                }
            }
        }
    }

    /// `input` is `bz_count` blocks of 16 rows of `n` columns; threadgroups of `t` threads.
    pub fn run(
        input: &[u32],
        w: &[u32],
        n: usize,
        bz_count: usize,
        t: usize,
        prm: [u32; 4],
    ) -> Vec<u32> {
        let mut out = vec![0u32; input.len()];
        for bz in 0..bz_count {
            for g in 0..n / t {
                let mut vs = vec![[0u32; 16]; t];
                for (l, v) in vs.iter_mut().enumerate() {
                    let col = g * t + l;
                    for m in 0..16 {
                        v[m] = input[(bz * 16 + m) * n + col];
                    }
                    if bz != 0 {
                        let mut tw = mul(prm[1], (bz + col + 1) as u32);
                        for x in v.iter_mut() {
                            *x = mul(*x, tw);
                            tw = mul(tw, prm[2]);
                        }
                    }
                    bfly(v, w);
                }
                let mut sh = vec![0u32; 16 * t];
                for (l, v) in vs.iter().enumerate() {
                    for m in 0..16 {
                        sh[m * t + l] = v[(m + l) & 15];
                    }
                }
                for l in 0..t {
                    let col = g * t + l;
                    let mut v = [0u32; 16];
                    for (m, x) in v.iter_mut().enumerate() {
                        *x = sh[m * t + (l + m) % t];
                    }
                    bfly(&mut v, w);
                    for m in 0..16 {
                        out[(bz * 16 + m) * n + col] = v[m];
                    }
                }
            }
        }
        out
    }
}

/// Probe of Z13's reported shape (`R17_Z13`); NOT a positive control (it does not reproduce on
/// macOS 27.2 / M5 Max: 0/n). Prints `R17AUDIT z13 shape <variant>: k/n dispatches wrong` for the
/// guarded loop and the unconditional-twist control and fails if either is wrong: a failure means
/// the shape now reproduces (a P0 for the R17 owners). Part of the `zkp` stress step of
/// run_stress.sh. `R17_AUDIT_Z13_COLS` (default 65536) columns, 4 column blocks, 256 threads.
#[test]
#[ignore]
fn r17_shape_probe_z13() {
    let reps = reps();
    let n = env_or("R17_AUDIT_Z13_COLS", 1 << 16);
    let (bz_count, t) = (4usize, 256usize);
    assert_eq!(n % t, 0);
    let mut r = rng(23);
    let p = z13_ref::P;
    let input: Vec<u32> = (0..bz_count * 16 * n)
        .map(|_| r.random_range(0..p))
        .collect();
    let w: Vec<u32> = (0..64).map(|_| r.random_range(0..p)).collect();
    let prm = [n as u32, 0x0123_4567 % p, 0x0246_8ace % p, 0];
    let want = z13_ref::run(&input, &w, n, bz_count, t, prm);
    let device = metal::Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let opts = metal::MTLResourceOptions::StorageModeShared;
    let bin =
        device.new_buffer_with_data(input.as_ptr() as *const _, (input.len() * 4) as u64, opts);
    let bout = device.new_buffer((input.len() * 4) as u64, opts);
    let mut bad = Vec::new();
    for (name, control) in [
        ("guarded-loop", false),
        ("unconditional-twist-control", true),
    ] {
        let src = format!(
            "{}{R17_Z13}",
            if control { "#define CONTROL 1\n" } else { "" }
        );
        let lib = device
            .new_library_with_source(&src, &metal::CompileOptions::new())
            .unwrap();
        let f = lib.get_function("z13_shape", None).unwrap();
        let pso = device.new_compute_pipeline_state_with_function(&f).unwrap();
        let (mut bad_dispatches, mut bad_out, mut bad_bz0) = (0, 0usize, 0usize);
        for _ in 0..reps {
            unsafe { std::ptr::write_bytes(bout.contents() as *mut u8, 0, input.len() * 4) };
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&bin), 0);
            enc.set_buffer(1, Some(&bout), 0);
            enc.set_bytes(2, 16, prm.as_ptr() as *const _);
            enc.set_bytes(3, 256, w.as_ptr() as *const _);
            enc.set_threadgroup_memory_length(0, (16 * t * 4) as u64);
            enc.dispatch_thread_groups(
                metal::MTLSize::new((n / t) as u64, bz_count as u64, 1),
                metal::MTLSize::new(t as u64, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let got =
                unsafe { std::slice::from_raw_parts(bout.contents() as *const u32, input.len()) };
            let wrong: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
            if !wrong.is_empty() {
                bad_dispatches += 1;
                bad_out += wrong.len();
                bad_bz0 += wrong.iter().filter(|&&i| i < 16 * n).count();
            }
        }
        println!(
            "R17AUDIT z13 shape {name}: {bad_dispatches}/{reps} dispatches wrong ({bad_out} wrong \
             outputs in all, {bad_bz0} of them in the bz == 0 threadgroups that skip the branch)"
        );
        if bad_dispatches != 0 {
            bad.push(name);
        }
    }
    assert!(
        bad.is_empty(),
        "Z13 shape variant(s) {bad:?} gave wrong results: the shape reproduces on this GPU (P0 for \
         the R17 owners, see docs/metal-port/r17/z13/README.md)"
    );
}

/// P31-T1: bypass-forcing stress of the zkp functions `ir_scan.py` tier Z flags (a runtime guard
/// that skips a loop updating a private `FpExt`, read after the merge; W7-T1's heuristic for
/// Z13's shape, never reproduced). Each shape makes the guard skip the loop on some or all
/// threads, and runs `R17_TIERZ_REPS` times (default `max(R17_AUDIT_REPS, 20)`) against the CPU
/// HAL with the output pre-filled with garbage:
/// - `eltwise_sum_fpext`: guard `to_add != 0` (the `for i < to_add` entry); `to_add == 0` skips
///   it on every thread (output must be zeros), and counts below / not a multiple of the
///   threadgroup width leave threads past `count` idle;
/// - `eu2_reduce`: guard `threadIdx < S` (32 threads per point); `S < 32` skips the loop on the
///   threads at or above `S` (S = 1..16), with S = 32 and 64 as no-bypass controls;
/// - `batch_evaluate_any` (Horner): guard `K - 1 >= 0`, `K = ceil(deg / 256)`, which skips the
///   loop only at deg 0 (not reachable through the HAL, so dispatched directly; the CPU value of
///   an empty polynomial is 0); tiny degrees (1..128, below the 256 threads) also skip the
///   in-loop `i < deg` add on most threads.
///
/// Prints `R17AUDIT zkp tierz <kernel> <shape>: k/n ...` per shape and fails on any nonzero k
/// or m (a P0 for the kernel's owner).
#[test]
#[ignore]
fn r17_tierz_bypass() {
    use metal::MTLSize;

    use super::{as_bytes, eval::HORNER_THREADS, Grid, KernelArg};

    let reps = env_or("R17_TIERZ_REPS", reps().max(20));
    let mut t = Tally(vec![]);
    let metal = MetalHalSha256::new();
    let cpu = CpuHal::<BabyBear>::new(metal.get_hash_suite().clone());
    let mut r = rng(31);

    // eltwise_sum_fpext.
    for (count, to_add) in [
        (1usize, 0usize),
        (3, 0),
        (31, 0),
        (33, 0),
        (257, 0),
        (1 << 16, 0),
        (1, 1),
        (3, 2),
        (31, 1),
        (33, 3),
        (257, 2),
        (4097, 5),
        (1 << 16, 1),
        (1 << 18, 4),
    ] {
        let input = exts(&mut r, count * to_add);
        let garbage = elems(&mut r, 4 * count);
        let c_out = cpu.copy_from_elem("out", &garbage);
        cpu.eltwise_sum_extelem(&c_out, &cpu.copy_from_extelem("in", &input));
        let want = c_out.to_vec();
        if to_add == 0 {
            assert!(want.iter().all(|x| *x == BabyBearElem::ZERO));
        }
        let m_in = metal.copy_from_extelem("in", &input);
        let case = format!("tierz eltwise_sum_fpext count {count} to_add {to_add}");
        let res = stress(&case, &want, reps, || {
            let out = metal.copy_from_elem("out", &garbage);
            metal.eltwise_sum_extelem(&out, &m_in);
            out.to_vec()
        });
        t.add(&case, res);
    }

    // eu2_reduce (through the forced eu2 path: eu2_pows, eu2_main(_lz), eu2_reduce).
    let chunk = super::eval::EU_MIN_CHUNK;
    for (s, poly_count, n_evals) in [
        (1u32, 2usize, 1usize),
        (1, 3, 37),
        (2, 2, 5),
        (4, 2, 3),
        (8, 1, 7),
        (16, 2, 33),
        (32, 1, 4),
        (64, 1, 2),
    ] {
        let deg = (s * chunk) as usize;
        let coeffs = elems(&mut r, deg * poly_count);
        let which: Vec<u32> = (0..n_evals)
            .map(|_| r.random_range(0..poly_count as u32))
            .collect();
        let xs = exts(&mut r, n_evals);
        let garbage = exts(&mut r, n_evals);
        let c_out = cpu.copy_from_extelem("out", &garbage);
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
        let force = EvalPath::Eu2 { s, chunk };
        let case = format!(
            "tierz eu2_reduce S {s} (bypass on {} of 32 threads) deg {deg} x{poly_count} evals \
             {n_evals}",
            32u32.saturating_sub(s)
        );
        let res = stress(&case, &want, reps, || {
            let out = metal.copy_from_extelem("out", &garbage);
            let path = eval::batch_evaluate_any(
                &metal,
                &m_coeffs,
                poly_count,
                &m_which,
                &m_xs,
                &out,
                Some(force),
            );
            assert_eq!(path, force);
            out.to_vec()
        });
        t.add(&case, res);
    }

    // batch_evaluate_any (Horner), tiny degrees through the HAL path.
    for (deg, poly_count, n_evals) in [
        (1usize, 1usize, 1usize),
        (1, 3, 5),
        (2, 2, 3),
        (4, 1, 9),
        (16, 2, 4),
        (128, 3, 17),
        (256, 2, 6),
        (512, 1, 2),
    ] {
        let coeffs = elems(&mut r, deg * poly_count);
        let which: Vec<u32> = (0..n_evals)
            .map(|_| r.random_range(0..poly_count as u32))
            .collect();
        let xs = exts(&mut r, n_evals);
        let garbage = exts(&mut r, n_evals);
        let c_out = cpu.copy_from_extelem("out", &garbage);
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
        let case = format!("tierz batch_evaluate_any deg {deg} x{poly_count} evals {n_evals}");
        let res = stress(&case, &want, reps, || {
            let out = metal.copy_from_extelem("out", &garbage);
            eval::batch_evaluate_any(
                &metal,
                &m_coeffs,
                poly_count,
                &m_which,
                &m_xs,
                &out,
                Some(EvalPath::Horner),
            );
            out.to_vec()
        });
        t.add(&case, res);
    }

    // batch_evaluate_any at deg 0: K = 0, every thread skips the Horner loop. Direct dispatch
    // with the HAL's binding and grid (eval.rs); the value of an empty polynomial is 0.
    for n_evals in [1usize, 7, 64] {
        let xs = exts(&mut r, n_evals);
        let garbage = exts(&mut r, n_evals);
        let want = vec![BabyBearExtElem::ZERO; n_evals];
        let m_coeffs = metal.copy_from_elem("c", &elems(&mut r, 4));
        let m_which = metal.copy_from_u32("w", &vec![0u32; n_evals]);
        let m_xs = metal.copy_from_extelem("x", &xs);
        let params = [0u32; 4];
        let case = format!("tierz batch_evaluate_any deg 0 (all threads bypass) evals {n_evals}");
        let res = stress(&case, &want, reps, || {
            let out = metal.copy_from_extelem("out", &garbage);
            metal.dispatch(
                metal.pipeline("batch_evaluate_any"),
                &[
                    out.as_arg(),
                    m_coeffs.as_arg(),
                    m_which.as_arg(),
                    m_xs.as_arg(),
                    KernelArg::Bytes(as_bytes(&params)),
                ],
                Grid::Groups {
                    groups: MTLSize::new(n_evals as u64, 1, 1),
                    threads: MTLSize::new(HORNER_THREADS as u64, 1, 1),
                },
                &[HORNER_THREADS as u64 * 16],
            );
            out.to_vec()
        });
        t.add(&case, res);
    }
    t.check();
}
