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
