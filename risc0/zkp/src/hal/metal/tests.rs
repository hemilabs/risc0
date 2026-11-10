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

use std::rc::Rc;

use risc0_core::field::{
    baby_bear::{BabyBearElem, BabyBearExtElem},
    Elem as _,
};
use test_log::test;

use super::{Grid, KernelArg, MetalHalPoseidon2, MetalHalSha256, KERNEL_NAMES, METAL_LIB};
use crate::hal::{cpu::CpuHal, dual::DualHal, testutil, Buffer, Hal};

// ---- the original 18 conformance tests (DualHal / testutil against the CPU HAL) ----

#[test]
fn batch_bit_reverse() {
    testutil::batch_bit_reverse(MetalHalSha256::new());
}

#[test]
fn batch_evaluate_any() {
    testutil::batch_evaluate_any(MetalHalSha256::new());
}

#[test]
fn batch_expand_into_evaluate_ntt() {
    testutil::batch_expand_into_evaluate_ntt(MetalHalSha256::new());
}

#[test]
fn batch_interpolate_ntt() {
    testutil::batch_interpolate_ntt(MetalHalSha256::new());
}

#[test]
#[should_panic]
fn check_req() {
    testutil::check_req(MetalHalSha256::new());
}

#[test]
fn eltwise_add_fp() {
    testutil::eltwise_add_elem(MetalHalSha256::new());
}

#[test]
fn eltwise_copy_fp() {
    testutil::eltwise_copy_elem(MetalHalSha256::new());
}

#[test]
fn eltwise_sum_extelem() {
    testutil::eltwise_sum_extelem(MetalHalSha256::new());
}

#[test]
fn fri_fold() {
    testutil::fri_fold(MetalHalSha256::new());
}

#[test]
fn mix_poly_coeffs() {
    testutil::mix_poly_coeffs(MetalHalSha256::new());
}

#[test]
fn hash_fold_sha256() {
    testutil::hash_fold(MetalHalSha256::new());
}

#[test]
fn hash_rows_sha256() {
    testutil::hash_rows(MetalHalSha256::new());
}

#[test]
fn hash_fold_poseidon2() {
    testutil::hash_fold(MetalHalPoseidon2::new());
}

#[test]
fn hash_rows_poseidon2() {
    testutil::hash_rows(MetalHalPoseidon2::new());
}

#[test]
fn slice() {
    testutil::slice(MetalHalSha256::new());
}

#[test]
fn zk_shift() {
    testutil::zk_shift(MetalHalSha256::new());
}

#[test]
fn gather_sample() {
    testutil::gather_sample(MetalHalSha256::new());
}

#[test]
fn prefix_products() {
    fn test(size: usize) {
        let hal_gpu = MetalHalSha256::new();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));
        let mut rng = rand::rng();
        let io: Vec<_> = (0..size)
            .map(|_| BabyBearExtElem::random(&mut rng))
            .collect();
        let io = hal.copy_from_extelem("io", io.as_slice());
        hal.prefix_products(&io);
    }

    for i in 15..=20 {
        test(1 << i);
    }
}

// ---- F0 runtime tests ----

fn elems(values: impl IntoIterator<Item = u32>) -> Vec<BabyBearElem> {
    values.into_iter().map(BabyBearElem::new).collect()
}

/// `hash_fold_tree` (one batch per tree) against the CPU HAL, both hash suites.
#[test]
fn hash_fold_tree_sha256() {
    testutil::hash_fold_tree(MetalHalSha256::new());
}

#[test]
fn hash_fold_tree_poseidon2() {
    testutil::hash_fold_tree(MetalHalPoseidon2::new());
}

/// 50 dependent dispatches, then a host read with no explicit flush/sync: the read must flush
/// and wait. Run with the default batch limit (one batch) and with a limit of 7 (eight command
/// buffers, ordered by the queue and hazard tracking).
#[test]
fn dispatches_then_read_without_flush() {
    for limit in [None, Some(7)] {
        let hal = MetalHalSha256::new();
        if let Some(limit) = limit {
            hal.set_batch_limit(limit);
        }
        let n = 4097;
        let one = hal.copy_from_elem("one", &elems(std::iter::repeat_n(1, n)));
        let acc = hal.copy_from_elem("acc", &elems(0..n as u32));
        for _ in 0..50 {
            hal.eltwise_add_elem(&acc, &acc, &one);
        }
        assert!(hal.rt.unsynced(), "dispatches must only encode");
        let got = acc.to_vec();
        assert!(!hal.rt.unsynced());
        assert_eq!(got, elems((0..n as u32).map(|i| i + 50)), "limit {limit:?}");
        // get_at / view also sync.
        hal.eltwise_add_elem(&acc, &acc, &one);
        assert_eq!(acc.get_at(3), BabyBearElem::new(54));
        hal.eltwise_add_elem(&acc, &acc, &one);
        acc.view(|v| assert_eq!(v[0], BabyBearElem::new(52)));
    }
}

/// A buffer dropped while its batch is open is not reused until that batch completes; after a
/// sync, the same allocation comes back from the pool and computes correctly.
#[test]
fn pool_reuse_across_batches() {
    let hal = MetalHalSha256::new();
    let n = 1 << 14;
    let ones = hal.copy_from_elem("ones", &elems(std::iter::repeat_n(1, n)));

    let a = hal.alloc_elem("a", n);
    let a_addr = a.as_device_ptr();
    hal.eltwise_add_elem(&a, &ones, &ones);
    drop(a);
    assert_eq!(
        hal.rt.pool_stats().1,
        1,
        "dropped while the batch is open: pending"
    );

    // Same size while the batch is still open: must not be the pending allocation.
    let b = hal.alloc_elem("b", n);
    assert_ne!(b.as_device_ptr(), a_addr);
    hal.eltwise_add_elem(&b, &ones, &ones);
    assert_eq!(b.to_vec(), elems(std::iter::repeat_n(2, n)));

    // Batch completed: the allocation is reused, in a new batch, with correct results.
    let c = hal.alloc_elem("c", n);
    assert_eq!(c.as_device_ptr(), a_addr, "pool did not reuse the buffer");
    hal.eltwise_add_elem(&c, &ones, &ones);
    hal.eltwise_add_elem(&c, &c, &ones);
    assert_eq!(c.to_vec(), elems(std::iter::repeat_n(3, n)));

    // A buffer dropped with no batch open goes straight to the ready list.
    drop(c);
    assert_eq!(hal.rt.pool_stats().1, 0);
    let d = hal.alloc_elem("d", n);
    assert_eq!(d.as_device_ptr(), a_addr);

    // trim_device_memory drains the pool.
    drop(d);
    assert!(hal.rt.pool_stats().0 > 0);
    hal.trim_device_memory();
    assert_eq!(hal.rt.pool_stats(), (0, 0));
}

/// Two threads each construct HALs and run GPU work concurrently; no deadlock, correct results.
#[test]
fn concurrent_hals() {
    let threads: Vec<_> = (0..2)
        .map(|t| {
            std::thread::spawn(move || {
                for round in 0..4 {
                    let hal_gpu = MetalHalPoseidon2::new();
                    hal_gpu.set_batch_limit(3 + t);
                    let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
                    let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));
                    let mut rng = rand::rng();
                    let rows = 1000 + 17 * round;
                    let values: Vec<_> = (0..rows * 16)
                        .map(|_| BabyBearElem::random(&mut rng))
                        .collect();
                    let matrix = hal.copy_from_elem("matrix", &values);
                    let sum = hal.alloc_elem("sum", rows * 16);
                    for _ in 0..10 {
                        hal.eltwise_add_elem(&sum, &matrix, &matrix);
                    }
                    let out = hal.alloc_digest("out", rows);
                    hal.hash_rows(&out, &sum);
                    // DualHal compares GPU and CPU results on drop / read.
                    let _ = out.to_vec();
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
}

/// `to_vec` on a slice returns the slice, not the whole allocation.
#[test]
fn to_vec_slice() {
    let hal = MetalHalSha256::new();
    let data = elems(0..100);
    let buf = hal.copy_from_elem("data", &data);
    assert_eq!(buf.slice(10, 20).to_vec(), data[10..30]);
    assert_eq!(buf.slice(10, 20).slice(5, 3).to_vec(), data[15..18]);
    assert_eq!(buf.slice(10, 20).get_at(2), data[12]);
    buf.slice(90, 10).view(|v| assert_eq!(v, &data[90..]));
}

/// A kernel name missing from a metallib fails `load_library`.
#[test]
fn load_library_missing_name_errors() {
    let hal = MetalHalSha256::new();
    let err = hal
        .try_load_library(METAL_LIB, &["eltwise_add_fp", "no_such_kernel"])
        .unwrap_err();
    assert!(format!("{err:#}").contains("no_such_kernel"), "{err:#}");
    let kernels = hal.load_library(METAL_LIB, KERNEL_NAMES);
    assert_eq!(kernels.names().count(), KERNEL_NAMES.len());
}

#[test]
#[should_panic(expected = "no_such_kernel")]
fn load_library_missing_name_panics() {
    let hal = MetalHalSha256::new();
    hal.load_library(METAL_LIB, &["no_such_kernel"]);
}

/// `dispatch` with `Grid::Groups`, `KernelArg::Bytes` and an explicit PSO from `load_library`.
#[test]
fn dispatch_groups_bytes_and_loaded_library() {
    let hal = MetalHalSha256::new();
    let kernels = hal.load_library(METAL_LIB, &["eltwise_mul_factor_fp"]);
    let n = 1000;
    let io = hal.copy_from_elem("io", &elems(0..n as u32));
    let factor = [BabyBearElem::new(3)];
    hal.dispatch(
        kernels.pipeline("eltwise_mul_factor_fp"),
        &[io.as_arg(), KernelArg::Bytes(super::as_bytes(&factor))],
        Grid::Threads(n as u64),
        &[],
    );
    assert_eq!(io.to_vec(), elems((0..n as u32).map(|i| 3 * i)));
}

/// blit_fill and alloc_private.
#[test]
fn blit_fill_and_private() {
    let hal = MetalHalSha256::new();
    let buf = hal.copy_from_u32("buf", &[7u32; 64]);
    hal.blit_fill(buf.as_buf(), 16, 32, 0xff);
    // Mixing blit and compute in one batch.
    let ones = hal.copy_from_elem("ones", &elems(std::iter::repeat_n(1, 8)));
    let acc = hal.alloc_elem("acc", 8);
    hal.blit_fill(acc.as_buf(), 0, 32, 0);
    hal.eltwise_add_elem(&acc, &acc, &ones);
    let got = buf.to_vec();
    for (i, v) in got.iter().enumerate() {
        let want = if (4..12).contains(&i) { 0xffff_ffff } else { 7 };
        assert_eq!(*v, want, "index {i}");
    }
    assert_eq!(acc.to_vec(), elems(std::iter::repeat_n(1, 8)));
    let private = hal.alloc_private(1 << 20);
    assert_eq!(private.length(), 1 << 20);
}

/// Memory queries and the deprecated sync-on-return dispatch path.
#[test]
#[allow(deprecated)]
fn memory_queries_and_compat_dispatch() {
    let hal = MetalHalSha256::new();
    let total = hal.gpu_total_memory();
    let free = hal.gpu_free_memory();
    assert!(total > 0 && total != usize::MAX);
    assert!(free <= total);

    let n = 256;
    let a = hal.copy_from_elem("a", &elems(0..n as u32));
    let out = hal.alloc_elem("out", n);
    let pso = hal.pipeline("eltwise_add_fp").clone();
    hal.dispatch_with_resources(
        &*pso,
        &[out.as_arg(), a.as_arg(), a.as_arg()],
        n as u64,
        None,
        |_| {},
    );
    assert!(
        !hal.rt.unsynced(),
        "dispatch_with_resources syncs on return"
    );
    assert_eq!(out.to_vec(), elems((0..n as u32).map(|i| 2 * i)));
    hal.dispatch_by_name("eltwise_add_fp", &[out.as_arg(), a.as_arg(), a.as_arg()], 0);
    assert!(!hal.rt.unsynced(), "empty grids encode nothing");

    // Old-style calls passing the owned metal-rs types (no `&*`) must still compile.
    hal.dispatch_with_resources(
        &pso,
        &[out.as_arg(), out.as_arg(), a.as_arg()],
        n as u64,
        None,
        |_| {},
    );
    assert_eq!(out.to_vec(), elems((0..n as u32).map(|i| 3 * i)));
    let func = hal
        .kernels
        .library
        .get_function("eltwise_add_fp", None)
        .unwrap();
    let desc = metal::ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&func));
    hal.dispatch_with_resources(
        &desc,
        &[out.as_arg(), out.as_arg(), a.as_arg()],
        n as u64,
        None,
        |_| {},
    );
    assert_eq!(out.to_vec(), elems((0..n as u32).map(|i| 4 * i)));
}

/// `as_bytes` is a safe byte view for padding-free (`NoUninit`) types, including ExtElem.
#[test]
fn as_bytes_no_uninit() {
    let e = [BabyBearElem::new(7), BabyBearElem::new(9)];
    assert_eq!(super::as_bytes(&e).len(), 8);
    let x = [BabyBearExtElem::from(BabyBearElem::new(5))];
    assert_eq!(super::as_bytes(&x).len(), 16);
    assert_eq!(super::as_bytes(&[1u32, 2]), &[1, 0, 0, 0, 2, 0, 0, 0]);
}

/// RISC0_METAL_DEBUG: a raw host pointer to a buffer an unsynced batch uses panics.
#[test]
fn debug_detects_unsynced_host_access() {
    let hal = MetalHalSha256::new();
    hal.set_debug(true);
    let ones = hal.copy_from_elem("ones", &elems(std::iter::repeat_n(1, 64)));
    let acc = hal.alloc_elem("acc", 64);
    let _ = acc.as_ptr(); // untouched by the GPU: fine
    hal.eltwise_add_elem(&acc, &ones, &ones);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| acc.as_ptr()));
    let msg = res.expect_err("as_ptr on an unsynced buffer must panic");
    let msg = msg.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("RISC0_METAL_DEBUG"), "{msg}");
    hal.sync();
    let _ = acc.as_ptr();
    assert_eq!(acc.to_vec(), elems(std::iter::repeat_n(2, 64)));
}

/// Plan 2.9: a failed allocation drains the pool and retries once; a second failure returns an
/// error naming the buffer and size.
#[test]
fn alloc_failure_drains_pool_and_retries() {
    let hal = MetalHalSha256::new();
    let small = hal.alloc_elem("small", 1 << 10);
    let other = hal.alloc_elem("other", 1 << 12);
    drop(small);
    drop(other);
    hal.sync();
    assert_eq!(hal.rt.pool_stats(), (2, 0), "dropped buffers are pooled");

    // One injected failure: pool drained, retry succeeds, buffer is usable.
    let bytes = 4 * (1 << 16);
    hal.rt.inject_alloc_failures(1);
    let buf = hal.rt.alloc_shared("retry", bytes).expect("retry succeeds");
    assert_eq!(hal.rt.pool_stats(), (0, 0), "failure drained the pool");
    assert_eq!(buf.length(), bytes);
    drop(buf);

    // Two injected failures: the named error.
    hal.rt.inject_alloc_failures(2);
    let err = hal.rt.alloc_shared("doomed", bytes).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("doomed") && msg.contains(&bytes.to_string()),
        "{msg}"
    );

    // Private allocations follow the same path.
    hal.rt.inject_alloc_failures(1);
    assert!(hal.rt.alloc_private(1024).is_ok());
    hal.rt.inject_alloc_failures(2);
    assert!(hal.rt.alloc_private(1024).is_err());

    // Hooks consumed: normal allocation and compute still work.
    let ones = hal.copy_from_elem("ones", &elems(std::iter::repeat_n(1, 64)));
    let out = hal.alloc_elem("out", 64);
    hal.eltwise_add_elem(&out, &ones, &ones);
    assert_eq!(out.to_vec(), elems(std::iter::repeat_n(2, 64)));
}

/// Plan 2.9: a command buffer that ends in `Error` makes `try_sync` return the error once; the
/// HAL keeps working afterwards.
#[test]
fn command_buffer_error_try_sync() {
    let hal = MetalHalSha256::new();
    let ones = hal.copy_from_elem("ones", &elems(std::iter::repeat_n(1, 64)));
    let out = hal.alloc_elem("out", 64);
    hal.rt.inject_batch_error();
    hal.eltwise_add_elem(&out, &ones, &ones);
    let err = hal.try_sync().unwrap_err();
    assert!(
        format!("{err:#}").contains("injected test error"),
        "{err:#}"
    );
    hal.try_sync().expect("error reported once");
    hal.eltwise_add_elem(&out, &out, &ones);
    assert_eq!(out.to_vec(), elems(std::iter::repeat_n(3, 64)));
}

/// `sync` (and host reads through it) panic on a command-buffer error.
#[test]
#[should_panic(expected = "Metal command buffer failed: injected test error")]
fn command_buffer_error_sync_panics() {
    let hal = MetalHalSha256::new();
    let ones = hal.copy_from_elem("ones", &elems(std::iter::repeat_n(1, 64)));
    let out = hal.alloc_elem("out", 64);
    hal.rt.inject_batch_error();
    hal.eltwise_add_elem(&out, &ones, &ones);
    let _ = out.to_vec();
}

// ---- Z6: generic testutil coverage of every Metal Hal method ----

#[test]
fn z6_scatter() {
    testutil::scatter(MetalHalSha256::new());
}

#[test]
fn z6_scatter_bits() {
    testutil::scatter_bits(MetalHalSha256::new());
}

#[test]
fn z6_eltwise_zeroize_elem() {
    testutil::eltwise_zeroize_elem(MetalHalSha256::new());
}

#[test]
fn z6_eltwise_copy_elem_slice() {
    testutil::eltwise_copy_elem_slice(MetalHalSha256::new());
}

#[test]
fn z6_alloc_init() {
    testutil::alloc_init(MetalHalSha256::new());
}

#[test]
fn z6_combos_prepare() {
    testutil::combos_prepare(MetalHalSha256::new());
}

macro_rules! combos_divide_po2 {
    ($($name:ident: $po2:expr,)*) => {$(
        #[test]
        fn $name() {
            testutil::combos_divide(MetalHalSha256::new(), $po2);
        }
    )*};
}

combos_divide_po2! {
    z6_combos_divide_po2_13: 13,
    z6_combos_divide_po2_14: 14,
    z6_combos_divide_po2_15: 15,
    z6_combos_divide_po2_16: 16,
    z6_combos_divide_po2_17: 17,
    z6_combos_divide_po2_18: 18,
    z6_combos_divide_po2_19: 19,
    z6_combos_divide_po2_20: 20,
    z6_combos_divide_po2_21: 21,
    z6_combos_divide_po2_22: 22,
}

#[test]
fn z6_zk_shift_variants() {
    testutil::zk_shift_variants(MetalHalSha256::new());
}

#[test]
fn z6_eval_any_shapes() {
    testutil::eval_any_shapes(MetalHalSha256::new());
}

#[test]
fn z6_mix_poly_coeffs_shapes() {
    testutil::mix_poly_coeffs_shapes(MetalHalSha256::new());
}

/// Full prove / verify round trip of the toy circuit on CPU, Metal and DualHal(CPU, Metal):
/// identical seals that the CPU verifier accepts.
///
/// The HALs zero pooled buffers on reuse (test hook): `DualHal` compares whole buffers after each
/// call, including cells the method leaves unwritten (the head of a Merkle `nodes` buffer, which
/// the CPU HAL allocates zeroed and the pool hands back with old contents).
#[test]
fn z6_prove_verify_round_trip_sha256() {
    testutil::prove_verify_round_trip(
        || {
            let hal = MetalHalSha256::new();
            hal.rt.set_zero_pooled(true);
            hal
        },
        &[10, 13, 16],
    );
}

#[test]
fn z6_prove_verify_round_trip_poseidon2() {
    testutil::prove_verify_round_trip(
        || {
            let hal = MetalHalPoseidon2::new();
            hal.rt.set_zero_pooled(true);
            hal
        },
        &[10, 12, 15, 17],
    );
}

// ---- Z6: sync / pool stress ----

/// Random interleaving of allocations (pooled sizes), dependent GPU adds, drops while a batch is
/// open, host reads and host writes (both sync), explicit flush/sync, and batch limits 1..=9 or
/// the default; every live buffer is mirrored on the host and compared on read and at the end.
fn stress_one(seed: u64, iters: usize, debug: bool) {
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    let hal = MetalHalSha256::new();
    hal.set_debug(debug);
    let mut rng = SmallRng::seed_from_u64(seed);
    let sizes = [1usize, 255, 4096, 1 << 14, (1 << 16) + 3];
    let mut live: Vec<(super::BufferImpl<BabyBearElem>, Vec<BabyBearElem>)> = vec![];
    for it in 0..iters {
        if it % 61 == 0 {
            let limit = match rng.random_range(0..4) {
                0 => super::DEFAULT_BATCH_DISPATCHES,
                _ => rng.random_range(1..=9),
            };
            hal.set_batch_limit(limit);
        }
        match rng.random_range(0..12) {
            0..=2 => {
                let n = sizes[rng.random_range(0..sizes.len())];
                let v: Vec<_> = (0..n).map(|_| BabyBearElem::random(&mut rng)).collect();
                live.push((hal.copy_from_elem("stress", &v), v));
            }
            3..=6 if !live.is_empty() => {
                let i = rng.random_range(0..live.len());
                let n = live[i].1.len();
                let same: Vec<usize> = (0..live.len()).filter(|&j| live[j].1.len() == n).collect();
                let j = same[rng.random_range(0..same.len())];
                hal.eltwise_add_elem(&live[i].0, &live[i].0, &live[j].0);
                let src = live[j].1.clone();
                for (d, s) in live[i].1.iter_mut().zip(src) {
                    *d += s;
                }
            }
            7 if !live.is_empty() => {
                let i = rng.random_range(0..live.len());
                live.swap_remove(i);
            }
            8 if !live.is_empty() => {
                let i = rng.random_range(0..live.len());
                assert!(
                    live[i].0.to_vec() == live[i].1,
                    "seed {seed} iter {it}: read"
                );
            }
            9 if !live.is_empty() => {
                let i = rng.random_range(0..live.len());
                let k = rng.random_range(0..live[i].1.len());
                let v = BabyBearElem::random(&mut rng);
                live[i].0.view_mut(|s| s[k] = v);
                live[i].1[k] = v;
            }
            10 => hal.flush(),
            11 => hal.sync(),
            _ => {}
        }
        if live.len() > 24 {
            live.remove(0);
        }
    }
    for (i, (buf, want)) in live.iter().enumerate() {
        assert!(buf.to_vec() == *want, "seed {seed}: final buffer {i}");
    }
}

#[test]
fn z6_stress_sync_pool() {
    stress_one(1, 4000, false);
}

#[test]
fn z6_stress_sync_pool_debug() {
    stress_one(2, 1500, true);
}

/// Four threads, each with its own HAL, stressing concurrently (shared `exclusive()` lock,
/// per-HAL batches and pools).
#[test]
fn z6_stress_sync_pool_threads() {
    let threads: Vec<_> = (0..4)
        .map(|t| std::thread::spawn(move || stress_one(100 + t, 1200, t == 3)))
        .collect();
    for t in threads {
        t.join().unwrap();
    }
}

// ---- Z6: working-set budget (RISC0_METAL_MAX_WORKING_SET_BYTES), peak, runtime fixes ----

#[test]
fn z6_parse_bytes() {
    use super::runtime::parse_bytes;
    assert_eq!(parse_bytes("4294967296"), Some(4 << 30));
    assert_eq!(parse_bytes("4G"), Some(4 << 30));
    assert_eq!(parse_bytes("4GiB"), Some(4 << 30));
    assert_eq!(parse_bytes(" 8g "), Some(8 << 30));
    assert_eq!(parse_bytes("512M"), Some(512 << 20));
    assert_eq!(parse_bytes("512MB"), Some(512 << 20));
    assert_eq!(parse_bytes("1k"), Some(1024));
    assert_eq!(parse_bytes("1T"), Some(1 << 40));
    for bad in ["", "abc", "0", "4X", "-1", "G", "1.5G"] {
        assert_eq!(parse_bytes(bad), None, "{bad:?}");
    }
}

/// `gpu_total_memory` / `gpu_free_memory` honour the budget; `None` restores
/// `recommendedMaxWorkingSetSize`; a budget above it is capped.
#[test]
fn z6_working_set_limit_queries() {
    let hal = MetalHalSha256::new();
    let rec = hal.device.recommended_max_working_set_size();
    hal.set_working_set_limit(None);
    assert_eq!(hal.gpu_total_memory() as u64, rec);
    assert_eq!(hal.working_set_limit(), rec);
    let budget = (3u64 << 30).min(rec);
    hal.set_working_set_limit(Some(budget));
    assert_eq!(hal.gpu_total_memory() as u64, budget);
    assert!(hal.gpu_free_memory() as u64 <= budget);
    hal.set_working_set_limit(Some(u64::MAX));
    assert_eq!(hal.gpu_total_memory() as u64, rec);
    hal.set_working_set_limit(None);
}

/// Over budget, a new allocation drains the pool and then fails with a clear error (the Hal
/// entry points panic with it); reusing a pooled buffer of the same size is not a new allocation.
#[test]
fn z6_working_set_budget_errors_early() {
    let hal = MetalHalSha256::new();
    let pooled = hal.alloc_elem("pooled", 1 << 18);
    drop(pooled);
    hal.sync();
    assert!(hal.rt.pool_stats().0 >= 1);
    hal.set_working_set_limit(Some(1));
    assert_eq!(hal.gpu_total_memory(), 1);
    assert_eq!(hal.gpu_free_memory(), 0);
    // Same size as the pooled buffer: served from the pool, no new allocation.
    drop(hal.alloc_elem("reused", 1 << 18));
    let err = hal.rt.alloc_shared("over", 3 << 20).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("working-set budget exceeded")
            && msg.contains("over")
            && msg.contains(super::MAX_WORKING_SET_ENV),
        "{msg}"
    );
    assert_eq!(hal.rt.pool_stats(), (0, 0), "the pool is drained first");
    assert!(hal.rt.alloc_private(1 << 20).is_err());
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hal.alloc_elem("hal_alloc", 1 << 20);
    }));
    let panic = res.expect_err("an over-budget Hal allocation must panic");
    let panic = panic.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(panic.contains("working-set budget exceeded"), "{panic}");
    hal.set_working_set_limit(None);
    let ones = hal.copy_from_elem("ones", &elems(std::iter::repeat_n(1, 64)));
    let out = hal.alloc_elem("out", 64);
    hal.eltwise_add_elem(&out, &ones, &ones);
    assert_eq!(out.to_vec(), elems(std::iter::repeat_n(2, 64)));
}

/// The process-wide peak follows allocations.
#[test]
fn z6_peak_allocated_size() {
    let hal = MetalHalSha256::new();
    hal.reset_peak_allocated_size();
    let big = hal.alloc_elem("big", 16 << 20);
    assert!(hal.peak_allocated_size() >= hal.allocated_size().min(64 << 20));
    assert!(hal.peak_allocated_size() >= 64 << 20);
    drop(big);
}

/// F0 open issue (Z2 note): an unchanged threadgroup memory length is not re-issued within an
/// encoder (the validation layer reported "redundant setting of (null)").
#[test]
fn z6_threadgroup_length_dedupe() {
    let mut bound = super::runtime::Bound::default();
    assert!(bound.set_tg_length(0, 4096));
    assert!(!bound.set_tg_length(0, 4096));
    assert!(bound.set_tg_length(0, 8192));
    assert!(bound.set_tg_length(2, 16));
    assert!(!bound.set_tg_length(2, 16));
    assert!(super::runtime::Bound::default().set_tg_length(0, 8192));
    // Real dispatches with dynamic threadgroup memory, repeated in one encoder.
    testutil::batch_evaluate_any(MetalHalSha256::new());
}

/// Plan 9.2: on a unified-memory GPU HAL, `PolyGroup::new` keeps large coefficient buffers on
/// the device (no host spill at >= 2^22 rows and >= 512 MiB).
#[test]
fn z6_poly_group_no_coeffs_spill() {
    use crate::prove::poly_group::PolyGroup;
    let hal = MetalHalPoseidon2::new();
    let count = 32;
    let size = 1 << 22;
    let coeffs = hal.alloc_elem_init("coeffs", count * size, BabyBearElem::new(5));
    let pg = PolyGroup::new(&hal, coeffs, count, size, "data");
    assert!(
        pg.coeffs.is_some() && pg.coeffs_host.is_none(),
        "coeffs were spilled"
    );
}
