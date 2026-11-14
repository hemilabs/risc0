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

//! Z8b review round (Poseidon2 / SHA / mix / FRI kernels and the HAL runtime): one regression
//! test per fixed finding (each fails on the pre-fix code; see docs/metal-port/notes/Z8b.md), plus
//! the Poseidon2 crossover-boundary coverage the Z3 dispatcher tests missed (they used CUDA's
//! 512 / 4096 boundaries, not Metal's 256 / 2048).
//!
//! Tests that read or change process-wide state (other HALs' pools, budgets) take
//! [super::gpu_test_lock] (plan 8.7).

use rand::{rngs::SmallRng, RngCore, SeedableRng};
use risc0_core::field::baby_bear::{BabyBear, BabyBearElem};
use test_log::test;

use super::{
    gpu_test_lock, Grid, KernelArg, MetalHalPoseidon2, MetalHalSha256, Poseidon2Config,
    Poseidon2Kernels,
};
use crate::{
    core::{
        digest::Digest,
        hash::{poseidon2::Poseidon2HashSuite, sha::Sha256HashSuite, HashSuite},
    },
    hal::{cpu::CpuHal, Buffer, Hal},
};
use metal::MTLSize;

const P: u32 = 0x7800_0001;

fn rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

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

// ---- F1 (P1): grids beyond the 32-bit thread index ----

/// Every zkp kernel indexes with a 32-bit `thread_position_in_grid`; a grid of 2^32 or more
/// threads in one dimension used to be dispatched and wrapped silently (e.g. an elementwise op
/// over a 2^24-row LDE with >= 256 columns). `fill_u32` with `count = 0` writes nothing even if
/// the dispatch ran, so the pre-fix behaviour (no panic) is harmless to observe.
#[test]
#[should_panic(expected = "32-bit thread index")]
fn grid_threads_beyond_u32_panics() {
    let hal = MetalHalSha256::new();
    let out = hal.alloc_u32("out", 16);
    hal.dispatch(
        hal.pipeline("fill_u32"),
        &[out.as_arg(), KernelArg::Integer(7), KernelArg::Integer(0)],
        Grid::Threads(1u64 << 32),
        &[],
    );
    hal.sync();
}

/// The same for `dispatch_thread_groups` shapes (groups x threads >= 2^32 in x).
#[test]
#[should_panic(expected = "32-bit thread index")]
fn grid_groups_beyond_u32_panics() {
    let hal = MetalHalSha256::new();
    let out = hal.alloc_u32("out", 16);
    hal.dispatch(
        hal.pipeline("fill_u32"),
        &[out.as_arg(), KernelArg::Integer(7), KernelArg::Integer(0)],
        Grid::Groups {
            groups: MTLSize::new(1 << 25, 1, 1),
            threads: MTLSize::new(128, 1, 1),
        },
        &[],
    );
    hal.sync();
}

/// The largest legal 1-D grid still runs (2^32 - 1 threads would be too slow; check the boundary
/// logic with a grid just below 2^32 / 2^8 groups of 256 instead: plain correctness of a normal
/// dispatch after the check).
#[test]
fn grid_below_u32_runs() {
    let hal = MetalHalSha256::new();
    let out = hal.alloc_u32("out", 1000);
    hal.dispatch(
        hal.pipeline("fill_u32"),
        &[
            out.as_arg(),
            KernelArg::Integer(7),
            KernelArg::Integer(1000),
        ],
        Grid::Threads(1000),
        &[],
    );
    assert_eq!(out.to_vec(), vec![7u32; 1000]);
}

// ---- F2 (P1): the budget is device-wide, the drain was per HAL ----

/// `currentAllocatedSize` (what the budget is checked against) counts every HAL's buffers,
/// including buffers idling in other HALs' pools, but an over-budget allocation only drained its
/// own HAL's pool. In a prover process with several HALs (rv32im, recursion, keccak), one HAL's
/// pool (up to half the budget) made another HAL's allocations fail. Now every pool is drained.
#[test]
fn budget_failure_drains_every_pool() {
    let _lock = gpu_test_lock();
    let a = MetalHalSha256::new();
    // W7-A1: a pool group of its own, so other tests' pools never evict this one's.
    a.rt.set_pool_group(super::runtime::Runtime::new_test_pool_group());
    let b = MetalHalSha256::new();
    drop(a.alloc_elem("a_pooled", 1 << 18));
    a.sync();
    assert!(a.rt.pool_stats().0 >= 1, "a's buffer is pooled");
    // The device-wide scope (the default; RISC0_METAL_BUDGET_SCOPE may say otherwise).
    b.set_budget_scope(super::BudgetScope::Device);
    b.set_working_set_limit(Some(1));
    assert!(b.rt.alloc_shared("over", 3 << 20).is_err());
    b.set_working_set_limit(None);
    assert_eq!(
        a.rt.pool_stats(),
        (0, 0),
        "an over-budget allocation on b must drain a's pool too"
    );
}

/// The same for a real allocation failure (plan 2.9 drain-and-retry).
#[test]
fn alloc_failure_drains_every_pool() {
    let _lock = gpu_test_lock();
    let a = MetalHalSha256::new();
    // W7-A1: a pool group of its own, so other tests' pools never evict this one's.
    a.rt.set_pool_group(super::runtime::Runtime::new_test_pool_group());
    let b = MetalHalSha256::new();
    drop(a.alloc_elem("a_pooled", 1 << 18));
    a.sync();
    assert!(a.rt.pool_stats().0 >= 1, "a's buffer is pooled");
    b.rt.inject_alloc_failures(1);
    let buf = b.rt.alloc_shared("retry", 1 << 20).expect("retry succeeds");
    assert_eq!(buf.length(), 1 << 20);
    assert_eq!(
        a.rt.pool_stats(),
        (0, 0),
        "an allocation failure on b must drain a's pool too"
    );
}

// ---- F3 (P2): host-side bounds the CPU HAL enforces ----

/// Binding past the end of a view gave the kernel an out-of-bounds pointer.
#[test]
#[should_panic(expected = "as_arg_with_offset")]
fn arg_offset_past_view_panics() {
    let hal = MetalHalSha256::new();
    let b = hal.alloc_elem("b", 8);
    let _ = b.as_arg_with_offset(9);
}

/// The CPU HAL asserts `io.size() >= 2 * input_size`; Metal read past the buffer.
#[test]
#[should_panic(expected = "hash_fold")]
fn hash_fold_small_io_panics_sha() {
    let hal = MetalHalSha256::new();
    let io = hal.alloc_digest("io", 12);
    hal.hash_fold(&io, 8, 4);
}

#[test]
#[should_panic(expected = "hash_fold")]
fn hash_fold_small_io_panics_poseidon2() {
    let hal = MetalHalPoseidon2::new();
    let io = hal.alloc_digest("io", 12);
    hal.hash_fold(&io, 8, 4);
}

/// `hash_fold_tree` with too few digests (the Poseidon2 hash already asserted; SHA did not).
#[test]
#[should_panic(expected = "hash_fold_tree")]
fn hash_fold_tree_small_io_panics_sha() {
    let hal = MetalHalSha256::new();
    let io = hal.alloc_digest("io", 15);
    hal.hash_fold_tree(&io, 3);
}

/// `gather_sample` read `src[gid * stride + idx]` without any check (CPU: panics).
#[test]
#[should_panic(expected = "gather_sample")]
fn gather_sample_src_out_of_range_panics() {
    let hal = MetalHalSha256::new();
    let src = hal.alloc_elem("src", 64);
    let dst = hal.alloc_elem("dst", 8);
    // last read: 7 * 9 + 1 = 64 >= 64
    hal.gather_sample(&dst, &src, 1, 8, 9);
}

#[test]
#[should_panic(expected = "gather_sample")]
fn gather_sample_dst_too_small_panics() {
    let hal = MetalHalSha256::new();
    let src = hal.alloc_elem("src", 64);
    let dst = hal.alloc_elem("dst", 4);
    hal.gather_sample(&dst, &src, 0, 8, 8);
}

/// In-range gathers still match the CPU HAL.
#[test]
fn gather_sample_matches_cpu() {
    let hal = MetalHalSha256::new();
    let cpu = CpuHal::<BabyBear>::new(Sha256HashSuite::new_suite());
    let mut rng = rng(81);
    let (rows, cols) = (37usize, 11usize);
    let m = elems(&mut rng, rows * cols);
    let src = hal.copy_from_elem("src", &m);
    let csrc = cpu.copy_from_elem("src", &m);
    for idx in [0, 1, rows - 1] {
        let dst = hal.alloc_elem("dst", cols);
        let cdst = cpu.alloc_elem("dst", cols);
        hal.gather_sample(&dst, &src, idx, cols, rows);
        cpu.gather_sample(&cdst, &csrc, idx, cols, rows);
        assert_eq!(dst.to_vec(), cdst.to_vec(), "idx={idx}");
    }
}

// ---- Poseidon2 crossover boundaries (pvs_w <= 256 < pvs_q <= 2048 < pv) ----

const BOUNDARY_N: [usize; 8] = [1, 255, 256, 257, 1024, 2047, 2048, 2049];

fn check_boundaries(hal: &MetalHalPoseidon2, suite: HashSuite<BabyBear>, what: &str) {
    let cpu = CpuHal::<BabyBear>::new(suite);
    let mut rng = rng(82);
    for &rows in &BOUNDARY_N {
        for cols in [1usize, 8, 16, 17, 211, 256, 257] {
            let m = elems(&mut rng, rows * cols);
            let cm = cpu.copy_from_elem("m", &m);
            let cout = cpu.alloc_digest("out", rows);
            cpu.hash_rows(&cout, &cm);
            let gm = hal.copy_from_elem("m", &m);
            let gout = hal.alloc_digest("out", rows);
            hal.hash_rows(&gout, &gm);
            assert!(
                gout.to_vec() == cout.to_vec(),
                "{what}: hash_rows rows={rows} cols={cols}"
            );
        }
    }
    for &n in &BOUNDARY_N {
        let io = digests(&mut rng, 4 * n);
        let cio = cpu.copy_from_digest("io", &io);
        cpu.hash_fold(&cio, 2 * n, n);
        let gio = hal.copy_from_digest("io", &io);
        hal.hash_fold(&gio, 2 * n, n);
        assert!(
            gio.to_vec() == cio.to_vec(),
            "{what}: hash_fold n={n} (inputs must also be unchanged)"
        );
    }
}

#[test]
fn poseidon2_crossover_boundaries() {
    let hal = MetalHalPoseidon2::new();
    let d = Poseidon2Config::default();
    assert_eq!((d.w_max_fold, d.q_max_fold), (256, 2048));
    assert_eq!((d.w_max_rows, d.q_max_rows), (256, 2048));
    check_boundaries(&hal, Poseidon2HashSuite::new_suite(), "pv");
    hal.hash().set_config(Poseidon2Config {
        kernels: Poseidon2Kernels::PvM32,
        ..d
    });
    check_boundaries(&hal, Poseidon2HashSuite::new_suite(), "m32");
}

/// The launch shapes the host uses are within what the PSOs report: 128-thread groups for every
/// `pv.metal` kernel (pv / pvs / tail), 256 for the table `mix` kernels, and a SIMD width of 32
/// for the pvs shuffles (else the HAL turns pvs off). Measured on the M5 Max: every pv kernel
/// reports exactly 128 (its `max_total_threads_per_threadgroup` attribute).
#[test]
fn launch_shapes_fit_psos() {
    let hal = MetalHalPoseidon2::new();
    for name in super::KERNEL_NAMES {
        let p = hal.pipeline(name);
        let need = if name.starts_with("pv") {
            128
        } else if name.starts_with("mix_poly_coeffs_v") {
            256
        } else {
            continue;
        };
        assert!(
            p.max_total_threads_per_threadgroup() >= need,
            "{name}: max {} < {need}",
            p.max_total_threads_per_threadgroup()
        );
    }
    let small = hal.hash().config().small;
    let width32 = super::KERNEL_NAMES
        .iter()
        .filter(|n| n.starts_with("pvs_"))
        .all(|n| hal.pipeline(n).thread_execution_width() == 32);
    assert!(width32 || !small, "pvs enabled without SIMD width 32");
}

// ---- F4 (P1, tests): a per-HAL budget scope for parallel budget runs ----

/// With `BudgetScope::Hal` the budget counts only this HAL's allocations (live, pooled, private),
/// so other HALs' memory (other tests running in parallel) cannot make it fail, and pooled
/// memory is reclaimed by draining this HAL's pool. Deterministic under the parallel harness.
#[test]
fn budget_scope_hal_counts_own_allocations() {
    // Other tests drain every HAL's pool (F2), which would lower owned bytes mid-test (plan 8.7).
    let _lock = gpu_test_lock();
    use super::BudgetScope;
    let other = MetalHalSha256::new();
    let hal = MetalHalSha256::new();
    hal.rt
        .set_pool_group(super::runtime::Runtime::new_test_pool_group());
    hal.set_budget_scope(BudgetScope::Hal);
    assert_eq!(hal.budget_scope(), BudgetScope::Hal);
    // Device-wide usage far above the budget below.
    let big = other.alloc_elem("other_live", 16 << 20);
    let mib = 1u64 << 20;
    let base = hal.owned_allocated_size();
    hal.set_working_set_limit(Some(base + 32 * mib));
    assert!(hal.allocated_size() > base + 32 * mib);

    let a = hal.alloc_elem("a", 4 << 20); // 16 MiB: fits
    assert_eq!(hal.owned_allocated_size(), base + 16 * mib);
    assert!(hal.gpu_free_memory() as u64 <= 16 * mib);
    let err = hal.rt.alloc_shared("b", 20 * mib).unwrap_err();
    assert!(
        format!("{err:#}").contains("already allocated by this HAL"),
        "{err:#}"
    );
    // Pooled bytes stay owned; an over-budget request drains this HAL's pool and then fits.
    drop(a);
    hal.sync();
    assert_eq!(
        hal.owned_allocated_size(),
        base + 16 * mib,
        "pooled bytes are still owned"
    );
    let b = hal
        .rt
        .alloc_shared("b", 20 * mib)
        .expect("fits after draining the own pool");
    assert_eq!(hal.owned_allocated_size(), base + 20 * mib);
    drop(b);
    // A buffer that bypasses the pool is no longer owned once dropped.
    let c = hal.alloc_elem("c", 1 << 20);
    c.set_bypass_pool();
    let before = hal.owned_allocated_size();
    drop(c);
    assert_eq!(hal.owned_allocated_size(), before - 4 * mib);
    // trim_device_memory releases the pool.
    hal.trim_device_memory();
    hal.set_working_set_limit(None);
    drop(big);
}
