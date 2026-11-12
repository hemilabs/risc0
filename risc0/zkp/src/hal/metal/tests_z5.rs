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

//! Z5: `combos_divide` (mdiv) CPU vs Metal, bit-exact.
//!
//! `DualHal` does not forward `combos_divide` to its second HAL (the trait default runs on the
//! first through `view_mut` and copies), so these tests run the CPU HAL and the Metal HAL
//! separately on identical inputs and compare the buffers.
//!
//! * Zero remainders: the CPU HAL's `combos_divide` is the oracle (it asserts zero remainders).
//! * Nonzero remainders: the CPU HAL panics on those, so the oracle is the host loop with the
//!   CUDA semantics (`poly_divide`, then the remainder at `a[n - 1]`, as sppark
//!   `div_by_x_minus_z<true>`); the remainders and the `report_nonzero_remainders` log lines must
//!   match exactly.

use std::{
    io::Write,
    sync::{Arc, Mutex},
    time::Instant,
};

use metal::{
    foreign_types::ForeignTypeRef,
    objc::{msg_send, runtime::Object, sel, sel_impl},
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use risc0_core::field::{
    baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
    Elem as _,
};
use test_log::test;

use super::{
    mdiv::{host_combos_divide, mdiv_default_e, mdiv_plan, mdiv_shape, MdShape, MD_MAXJ},
    MetalHalSha256,
};
use crate::hal::{cpu::CpuHal, Buffer, Hal};

type Chunks = Vec<(usize, Vec<BabyBearExtElem>)>;

fn rand_ext(rng: &mut SmallRng) -> BabyBearExtElem {
    BabyBearExtElem::new(
        BabyBearElem::new(rng.random::<u32>() % 0x7800_0001),
        BabyBearElem::new(rng.random::<u32>() % 0x7800_0001),
        BabyBearElem::new(rng.random::<u32>() % 0x7800_0001),
        BabyBearElem::new(rng.random::<u32>() % 0x7800_0001),
    )
}

/// A slice of `n` coefficients: random, or (if `divisible`) a random polynomial times
/// prod (X - pow) over `pows`, so every division leaves a zero remainder.
fn make_slice(
    rng: &mut SmallRng,
    n: usize,
    pows: &[BabyBearExtElem],
    divisible: bool,
) -> Vec<BabyBearExtElem> {
    if !divisible {
        return (0..n).map(|_| rand_ext(rng)).collect();
    }
    let m = pows.len();
    let mut p: Vec<BabyBearExtElem> = (0..n - m).map(|_| rand_ext(rng)).collect();
    p.resize(n, BabyBearExtElem::ZERO);
    // `deg`: current length.
    for (deg, &z) in (n - m..).zip(pows) {
        // p * (X - z): q[i] = p[i - 1] - z * p[i]
        for i in (0..=deg).rev() {
            let lo = if i > 0 {
                p[i - 1]
            } else {
                BabyBearExtElem::ZERO
            };
            let hi = if i < deg { p[i] } else { BabyBearExtElem::ZERO };
            p[i] = lo - z * hi;
        }
    }
    p
}

/// `slices` slices of `n` elements; chunk c names slice c (`permute` false: the CPU HAL pairs
/// chunk k with slice k, as the prover builds them) or slice (3c + 1) % slices (`permute`: a
/// subset out of order, as the GPU HALs allow), with root counts cycling through `roots`. `bad`
/// lists chunk positions with random (nonzero-remainder) slices; the others are divisible by all
/// their roots. Slices not named by a chunk must stay untouched.
fn make_case(
    seed: u64,
    n: usize,
    slices: usize,
    nchunks: usize,
    roots: &[usize],
    bad: &[usize],
    permute: bool,
) -> (Vec<BabyBearExtElem>, Chunks) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut data: Vec<BabyBearExtElem> = (0..slices * n).map(|_| rand_ext(&mut rng)).collect();
    // Chunk c uses slice (c * 3 + 1) % slices: skips some, and is not in index order.
    assert!(nchunks <= slices && (!permute || !slices.is_multiple_of(3)));
    let mut chunks = vec![];
    for c in 0..nchunks {
        let idx = if permute { (c * 3 + 1) % slices } else { c };
        let pows: Vec<BabyBearExtElem> = (0..roots[c % roots.len()])
            .map(|_| rand_ext(&mut rng))
            .collect();
        let slice = make_slice(&mut rng, n, &pows, !bad.contains(&c));
        data[idx * n..(idx + 1) * n].copy_from_slice(&slice);
        chunks.push((idx, pows));
    }
    (data, chunks)
}

/// Collects `tracing` output (the `report_nonzero_remainders` lines) of `f` on this thread.
fn capture_logs(f: impl FnOnce()) -> Vec<String> {
    #[derive(Clone)]
    struct W(Arc<Mutex<Vec<u8>>>);
    impl Write for W {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let buf = Arc::new(Mutex::new(vec![]));
    let w = W(buf.clone());
    let sub = tracing_subscriber::fmt()
        .with_writer(move || w.clone())
        .with_ansi(false)
        .without_time()
        .with_target(false)
        .with_max_level(tracing::Level::ERROR)
        .finish();
    {
        let _guard = tracing_subscriber::util::SubscriberInitExt::set_default(sub);
        f();
    }
    let s = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
    s.lines()
        .filter(|l| l.contains("combos_divide"))
        .map(str::to_string)
        .collect()
}

/// Zero remainders: CPU HAL vs Metal HAL (via `Hal::combos_divide`), bit-exact, plus the
/// remainders reported by the Metal path are all zero.
fn check_zero(po2: usize, slices: usize, nchunks: usize, roots: &[usize], shape: Option<MdShape>) {
    let n = 1 << po2;
    let (data, chunks) = make_case(po2 as u64, n, slices, nchunks, roots, &[], false);

    let cpu = CpuHal::<BabyBear>::new(crate::core::hash::sha::Sha256HashSuite::new_suite());
    let cbuf = cpu.copy_from_extelem("combos", &data);
    cpu.combos_divide(&cbuf, chunks.clone(), n);
    let expect = cbuf.to_vec();
    // The host reference used by the nonzero tests agrees with the CPU HAL here.
    let mut host = data.clone();
    let host_rems = host_combos_divide(&mut host, &chunks, n);
    assert!(host_rems.iter().all(|r| *r == BabyBearExtElem::ZERO));
    assert!(host == expect, "po2 {po2}: host reference vs CPU HAL");

    let hal = MetalHalSha256::new();
    let mbuf = hal.copy_from_extelem("combos", &data);
    let t0 = Instant::now();
    let rems = match shape {
        Some(s) => hal.mdiv_combos_divide_with(&mbuf, &chunks, n, Some(s)),
        None => {
            hal.combos_divide(&mbuf, chunks.clone(), n);
            vec![]
        }
    };
    let dt = t0.elapsed();
    let got = mbuf.to_vec();
    assert!(rems.iter().all(|r| *r == BabyBearExtElem::ZERO));
    if let Some(i) = (0..expect.len()).find(|&i| expect[i] != got[i]) {
        panic!(
            "po2 {po2}: first mismatch at element {i} (slice {}, offset {}): cpu {:?} metal {:?}",
            i / n,
            i % n,
            expect[i],
            got[i]
        );
    }
    tracing::info!("po2 {po2} {slices}x{nchunks} roots {roots:?} shape {shape:?}: metal {dt:?}");
}

/// Nonzero remainders (chunk positions in `bad`): Metal vs the CUDA-semantics host loop, data
/// and remainders bit-exact, and identical `report_nonzero_remainders` lines.
fn check_nonzero(
    po2: usize,
    slices: usize,
    nchunks: usize,
    roots: &[usize],
    bad: &[usize],
    shape: Option<Option<MdShape>>,
) {
    let n = 1 << po2;
    let (data, chunks) = make_case(1000 + po2 as u64, n, slices, nchunks, roots, bad, true);

    let mut expect = data.clone();
    let expect_rems = host_combos_divide(&mut expect, &chunks, n);
    let nonzero = expect_rems
        .iter()
        .filter(|r| **r != BabyBearExtElem::ZERO)
        .count();
    let expected_bad: usize = bad.iter().map(|&c| roots[c % roots.len()]).sum();
    assert_eq!(
        nonzero, expected_bad,
        "test setup: random slices have nonzero remainders"
    );
    let expect_log = capture_logs(|| crate::hal::report_nonzero_remainders(&chunks, &expect_rems));
    assert_eq!(expect_log.len(), nonzero);

    let hal = MetalHalSha256::new();
    let mbuf = hal.copy_from_extelem("combos", &data);
    let (rems, log) = match shape {
        Some(s) => {
            let rems = hal.mdiv_combos_divide_with(&mbuf, &chunks, n, s);
            let log = capture_logs(|| crate::hal::report_nonzero_remainders(&chunks, &rems));
            (Some(rems), log)
        }
        None => {
            let log = capture_logs(|| hal.combos_divide(&mbuf, chunks.clone(), n));
            (None, log)
        }
    };
    if let Some(rems) = rems {
        assert_eq!(rems, expect_rems, "po2 {po2}: remainders");
    }
    assert_eq!(log, expect_log, "po2 {po2}: reported remainders");
    let got = mbuf.to_vec();
    if let Some(i) = (0..expect.len()).find(|&i| expect[i] != got[i]) {
        panic!(
            "po2 {po2}: first mismatch at element {i} (slice {}, offset {}): host {:?} metal {:?}",
            i / n,
            i % n,
            expect[i],
            got[i]
        );
    }
}

// rv32im-like: 5 combo slices + the check slice, 1..3 roots each.
const ROOTS: &[usize] = &[1, 2, 3, 1, 2, 1];

#[test]
fn mdiv_shapes() {
    assert_eq!(mdiv_shape(1 << 10, 8), None);
    let s = mdiv_shape(1 << 13, 8).unwrap();
    assert_eq!((s.ntiles, s.scan_nt, s.k), (4, 32, 1));
    let s = mdiv_shape(1 << 16, 8).unwrap();
    assert_eq!((s.ntiles, s.scan_nt, s.k), (32, 32, 1));
    let s = mdiv_shape(1 << 21, 8).unwrap();
    assert_eq!((s.ntiles, s.scan_nt, s.k), (1024, 1024, 1));
    let s = mdiv_shape(1 << 22, 8).unwrap();
    assert_eq!((s.ntiles, s.scan_nt, s.k), (2048, 1024, 2));
    let s = mdiv_shape(1 << 22, 16).unwrap();
    assert_eq!((s.ntiles, s.scan_nt, s.k), (1024, 1024, 1));
    assert_eq!(mdiv_shape(3 << 13, 8), None);
    assert_eq!(mdiv_shape(1 << 24, 8), None);
    assert_eq!(mdiv_shape(1 << 16, 4), None);
    if std::env::var("RISC0_METAL_MDIV_E").is_err()
        && std::env::var("RISC0_METAL_MDIV_E16_MIN_PO2").is_err()
    {
        // Z9 moved the E = 16 switch from po2 20 to 17.
        assert_eq!(mdiv_default_e(1 << 16), 8);
        assert_eq!(mdiv_default_e(1 << 17), 16);
        assert_eq!(mdiv_default_e(1 << 20), 16);
        assert_eq!(mdiv_default_e(1 << 22), 16);
    }
    // Grouping: 10 combos -> 2 groups, each agg0 + 2 * maxm + final.
    let chunks: Chunks = (0..10)
        .map(|i| (i, vec![BabyBearExtElem::ONE; 1 + i % 2]))
        .collect();
    let plan = mdiv_plan(mdiv_shape(1 << 16, 8).unwrap(), 10 << 16, 1 << 16, &chunks);
    assert_eq!(plan.steps.len(), 2 * (1 + 2 * 2 + 1));
    assert_eq!(plan.rems_off, 2 * MD_MAXJ * 32);
    assert_eq!(plan.total_pows, 15);
}

macro_rules! zero_rem_tests {
    ($($name:ident: $po2:expr,)*) => {$(
        #[test]
        fn $name() {
            check_zero($po2, 7, 6, ROOTS, None);
        }
    )*};
}

zero_rem_tests! {
    combos_divide_po2_13: 13,
    combos_divide_po2_14: 14,
    combos_divide_po2_15: 15,
    combos_divide_po2_16: 16,
    combos_divide_po2_17: 17,
    combos_divide_po2_18: 18,
    combos_divide_po2_19: 19,
    combos_divide_po2_20: 20,
    combos_divide_po2_21: 21,
    combos_divide_po2_22: 22,
}

macro_rules! nonzero_rem_tests {
    ($($name:ident: $po2:expr,)*) => {$(
        #[test]
        fn $name() {
            // Chunks 1 (2 roots) and 4 (2 roots) are random: 4 nonzero remainders.
            check_nonzero($po2, 7, 6, ROOTS, &[1, 4], None);
        }
    )*};
}

nonzero_rem_tests! {
    combos_divide_nonzero_po2_13: 13,
    combos_divide_nonzero_po2_14: 14,
    combos_divide_nonzero_po2_15: 15,
    combos_divide_nonzero_po2_16: 16,
    combos_divide_nonzero_po2_17: 17,
    combos_divide_nonzero_po2_18: 18,
    combos_divide_nonzero_po2_19: 19,
    combos_divide_nonzero_po2_20: 20,
    combos_divide_nonzero_po2_21: 21,
    combos_divide_nonzero_po2_22: 22,
}

/// Both E variants regardless of the default (`mdiv_default_e`): E = 16 at po2 16 (scan padding)
/// and 18, E = 8 at po2 20..=22 (po2 22 at E = 8 is the K = 2 two-level scan).
#[test]
fn combos_divide_explicit_e() {
    for (po2, e) in [(16, 16), (18, 16), (20, 8), (21, 8), (22, 8), (22, 16)] {
        let shape = mdiv_shape(1 << po2, e);
        assert!(shape.is_some());
        check_zero(po2, 7, 6, ROOTS, shape);
        check_nonzero(po2, 7, 6, ROOTS, &[0, 5], Some(shape));
    }
}

/// More combos than MD_MAXJ (two groups in one batch), and a chunk with no roots.
#[test]
fn combos_divide_many_combos() {
    let roots = &[1, 3, 0, 2, 1, 1, 2, 4, 1, 2, 1];
    check_zero(15, 11, 11, roots, None);
    check_nonzero(18, 11, 11, roots, &[1, 7, 10], None);
}

/// Shapes the kernels do not take (po2 < 11) and `RISC0_METAL_MDIV=0`-style forced host loop:
/// the same semantics and reports.
#[test]
fn combos_divide_host_fallback() {
    for po2 in [4, 10] {
        check_zero(po2, 7, 6, ROOTS, None);
        check_nonzero(po2, 7, 6, ROOTS, &[2], None);
    }
    check_nonzero(16, 7, 6, ROOTS, &[2, 3], Some(None));
}

/// Two divides back to back, the first only encoded (its batch still open) when the second
/// starts: the second must see the first's writes, including the remainders it installs.
#[test]
fn combos_divide_after_unsynced_divide() {
    let po2 = 17;
    let n = 1 << po2;
    // Each slice divisible by both root sets.
    let (data, chunks) = make_case(77, n, 4, 3, &[3, 2, 4], &[], false);
    let first: Chunks = chunks.iter().map(|(i, p)| (*i, p[..1].to_vec())).collect();
    let second: Chunks = chunks.iter().map(|(i, p)| (*i, p[1..].to_vec())).collect();
    let cpu = CpuHal::<BabyBear>::new(crate::core::hash::sha::Sha256HashSuite::new_suite());
    let cbuf = cpu.copy_from_extelem("combos", &data);
    cpu.combos_divide(&cbuf, first.clone(), n);
    cpu.combos_divide(&cbuf, second.clone(), n);

    let hal = MetalHalSha256::new();
    let mbuf = hal.copy_from_extelem("combos", &data);
    let shape = mdiv_shape(n, 8).unwrap();
    let plan = mdiv_plan(shape, data.len(), n, &first);
    let scratch = hal.alloc_buffer::<BabyBearExtElem>("s", plan.scratch_len);
    hal.mdiv_encode(&mbuf, &scratch, &plan);
    assert!(hal.rt.unsynced());
    hal.combos_divide(&mbuf, second, n);
    assert_eq!(mbuf.to_vec(), cbuf.to_vec());
}

fn gpu_ms(cb: &metal::CommandBufferRef) -> f64 {
    #[allow(unexpected_cfgs)]
    let (s, e): (f64, f64) = unsafe {
        let p = cb.as_ptr() as *mut Object;
        (msg_send![p, GPUStartTime], msg_send![p, GPUEndTime])
    };
    (e - s) * 1e3
}

/// Indicative timings (not asserted): per-kernel GPU time (one command buffer per dispatch,
/// `GPUStartTime/GPUEndTime`), Metal wall time of `combos_divide`, and the CPU HAL. Run with
/// `cargo test -p risc0-zkp --features metal --lib --release -- hal::metal::tests_z5::mdiv_timing --ignored --nocapture`.
#[test]
#[ignore]
fn mdiv_timing() {
    let hal = MetalHalSha256::new();
    let cpu = CpuHal::<BabyBear>::new(crate::core::hash::sha::Sha256HashSuite::new_suite());
    let roots = &[2, 2, 2, 2, 2, 1];
    for po2 in [13, 14, 15, 16, 18, 20, 21, 22] {
        let n = 1 << po2;
        let (data, chunks) = make_case(5, n, 7, 6, roots, &[], false);
        for e in [8, 16] {
            let Some(shape) = mdiv_shape(n, e) else {
                continue;
            };
            let plan = mdiv_plan(shape, data.len(), n, &chunks);
            let combos = hal.copy_from_extelem("combos", &data);
            let scratch = hal.alloc_buffer::<BabyBearExtElem>("s", plan.scratch_len);
            // Per-kernel GPU time: best of 3 over the whole step list, summed by kernel.
            let mut best: Vec<(String, f64)> = vec![];
            for rep in 0..3 {
                let mut per: Vec<(String, f64)> = vec![];
                for step in &plan.steps {
                    let cb = hal.cmd_queue.new_command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(hal.pipeline(&step.kernel));
                    enc.set_buffer(0, Some(combos.as_buf()), (combos.offset() * 16) as u64);
                    enc.set_buffer(1, Some(scratch.as_buf()), (scratch.offset() * 16) as u64);
                    let jobs = [step.jobs];
                    let bytes = super::as_bytes(&jobs);
                    enc.set_bytes(2, bytes.len() as u64, bytes.as_ptr() as *const _);
                    if let Some(nj) = step.njobs {
                        enc.set_bytes(3, 4, &nj as *const u32 as *const _);
                    }
                    enc.dispatch_thread_groups(step.groups, step.threads);
                    enc.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    let ms = gpu_ms(cb);
                    let name = step.kernel.clone();
                    match per.iter_mut().find(|(k, _)| *k == name) {
                        Some((_, t)) => *t += ms,
                        None => per.push((name, ms)),
                    }
                }
                if rep == 0 {
                    best = per;
                } else {
                    for (k, t) in per {
                        let b = best.iter_mut().find(|(bk, _)| *bk == k).unwrap();
                        b.1 = b.1.min(t);
                    }
                }
            }
            // End to end (one batch, wall time incl. the sync), best of 3.
            let mut wall = f64::MAX;
            for _ in 0..3 {
                let b = hal.copy_from_extelem("combos", &data);
                let t0 = Instant::now();
                hal.mdiv_combos_divide_with(&b, &chunks, n, Some(shape));
                wall = wall.min(t0.elapsed().as_secs_f64() * 1e3);
            }
            let total: f64 = best.iter().map(|(_, t)| t).sum();
            println!(
                "po2 {po2} E{e} ntiles {} scan {}x{}: wall {wall:.3} ms, gpu sum {total:.3} ms: {}",
                shape.ntiles,
                shape.scan_nt,
                shape.k,
                best.iter()
                    .map(|(k, t)| format!("{k} {t:.3}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let mut cpu_ms = f64::MAX;
        for _ in 0..3 {
            let b = cpu.copy_from_extelem("combos", &data);
            let t0 = Instant::now();
            cpu.combos_divide(&b, chunks.clone(), n);
            cpu_ms = cpu_ms.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        println!("po2 {po2} cpu hal {cpu_ms:.3} ms");
    }
}
