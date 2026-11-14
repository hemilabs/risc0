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

//! W7-A1: process-wide pool cap with LRU eviction across pools, eviction on a miss and of stale
//! sizes, the memory-pressure drain (idle buffers only), the system-wide GPU memory check, purge
//! on free, and the Z14 single-thread check.
//!
//! Every test puts its HALs into a pool accounting group of its own
//! ([Runtime::new_test_pool_group]) so parallel tests neither evict each other's buffers nor
//! count toward each other's cap, and takes [gpu_test_lock] because the pressure drain and the
//! system check drain every pool in the process.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use metal::{MTLPurgeableState, MTLResourceOptions};
use test_log::test;

use super::{gpu_test_lock, runtime::Runtime, MetalHalSha256};
use crate::hal::Hal;

const MIB: u64 = 1 << 20;

/// A HAL in a pool group of its own, without an explicit budget (gpu-run exports one, and the
/// system check only applies without one).
fn hal_in(group: usize) -> MetalHalSha256 {
    let hal = MetalHalSha256::new();
    hal.set_working_set_limit(None);
    hal.rt.set_pool_group(group);
    hal
}

/// Allocate `bytes` on `hal` and give the buffer back with no batch open (straight to `ready`).
fn pool_one(hal: &MetalHalSha256, bytes: u64) {
    hal.sync();
    let buf = hal.rt.alloc_shared("a1", bytes).unwrap();
    hal.rt.release(bytes, buf);
}

/// The pooled sizes of at least 64 KiB (the evictable ones) and their counts.
fn ready_sizes(hal: &MetalHalSha256) -> Vec<(u64, usize)> {
    hal.rt
        .pool_sizes()
        .into_iter()
        .filter(|s| s.0 >= 64 << 10)
        .collect()
}

/// Raise `hal`'s live high-water mark to `bytes` without pooling anything, so a test of the
/// cross-pool cap is not also a test of the eviction on a miss.
fn raise_peak(hal: &MetalHalSha256, bytes: u64) {
    let buf = hal.rt.alloc_shared("peak", bytes).unwrap();
    hal.rt.freed(bytes, buf);
}

/// True when `RISC0_METAL_POOL_GLOBAL=0` restores the per-runtime pool: the process-wide cap,
/// the eviction on a miss and of stale sizes do not exist then, so the tests of them skip.
fn global_pool_off() -> bool {
    let off = std::env::var(super::POOL_GLOBAL_ENV).is_ok_and(|v| v.trim() == "0");
    if off {
        eprintln!("skipped: {}=0", super::POOL_GLOBAL_ENV);
    }
    off
}

/// Clears a group's test cap when the test ends (also on panic).
struct GroupCap(usize);

impl GroupCap {
    fn set(group: usize, cap: u64) -> Self {
        Runtime::set_group_cap(group, Some(cap));
        Self(group)
    }
}

impl Drop for GroupCap {
    fn drop(&mut self) {
        Runtime::set_group_cap(self.0, None);
    }
}

/// Three HALs share one process-wide cap: the pooled bytes of all three stay under it, and the
/// least recently returned buffer (the first HAL's) is the one evicted.
#[test]
fn global_cap_three_hals() {
    let _lock = gpu_test_lock();
    if global_pool_off() {
        return;
    }
    let g = Runtime::new_test_pool_group();
    let _cap = GroupCap::set(g, 40 * MIB);
    let hals = [hal_in(g), hal_in(g), hal_in(g)];
    for hal in &hals {
        pool_one(hal, 16 * MIB);
    }
    assert_eq!(
        Runtime::group_cached(g),
        32 * MIB,
        "two 16 MiB buffers fit in 40 MiB"
    );
    assert_eq!(
        ready_sizes(&hals[0]),
        vec![],
        "the oldest (first HAL's) buffer was evicted"
    );
    assert_eq!(ready_sizes(&hals[1]), vec![(16 * MIB, 1)]);
    assert_eq!(ready_sizes(&hals[2]), vec![(16 * MIB, 1)]);
    // Each pool keeps its own accounting straight.
    assert_eq!(hals[0].rt.pool_bytes().0, 0);
    assert_eq!(hals[1].rt.pool_bytes().0, 16 * MIB);
    drop(hals);
    assert_eq!(
        Runtime::group_cached(g),
        0,
        "dropped runtimes leave nothing counted"
    );
}

/// The same cap with the per-runtime limit only (what `RISC0_METAL_POOL_GLOBAL=0` restores) would
/// keep 48 MiB here; the process-wide cap is what bounds the sum. Also: LRU is by the time a
/// buffer was returned, across pools, so a buffer used again survives an older one in another
/// pool.
#[test]
fn lru_eviction_across_pools() {
    let _lock = gpu_test_lock();
    if global_pool_off() {
        return;
    }
    let g = Runtime::new_test_pool_group();
    let _cap = GroupCap::set(g, 24 * MIB);
    let a = hal_in(g);
    let b = hal_in(g);
    raise_peak(&a, 64 * MIB);
    raise_peak(&b, 64 * MIB);
    pool_one(&a, 8 * MIB); // a: X (oldest)
    pool_one(&b, 4 * MIB); // b: Y
    pool_one(&a, 2 * MIB); // a: Z
    assert_eq!(Runtime::group_cached(g), 14 * MIB);
    // Reuse X: it becomes the most recently returned.
    pool_one(&a, 8 * MIB);
    assert_eq!(ready_sizes(&a), vec![(2 * MIB, 1), (8 * MIB, 1)]);
    // 12 MiB more on b: 26 MiB > 24 MiB, so the least recently returned (Y, in b) goes, then
    // the cap holds.
    pool_one(&b, 12 * MIB);
    assert_eq!(ready_sizes(&b), vec![(12 * MIB, 1)], "Y was the LRU buffer");
    assert_eq!(ready_sizes(&a), vec![(2 * MIB, 1), (8 * MIB, 1)]);
    assert_eq!(Runtime::group_cached(g), 22 * MIB);
    // One more 6 MiB on a: 28 MiB, the oldest now is Z (a), then 26 > 24 still, then X (a).
    pool_one(&a, 6 * MIB);
    assert_eq!(ready_sizes(&a), vec![(6 * MIB, 1)]);
    assert_eq!(ready_sizes(&b), vec![(12 * MIB, 1)]);
    assert!(Runtime::group_cached(g) <= 24 * MIB);
}

/// After a po2 change (a commit over a new number of rows), a miss (a size the pool does not
/// hold) evicts LRU idle buffers of the previous po2's sizes so that live + pooled never exceeds
/// the runtime's high-water mark of live + pooled bytes: the new po2 reuses the old sizes'
/// memory. Without a po2 change nothing is evicted on a miss.
#[test]
fn miss_after_po2_change_evicts_to_high_water_mark() {
    let _lock = gpu_test_lock();
    if global_pool_off() {
        return;
    }
    let hal = hal_in(Runtime::new_test_pool_group());
    // Private: not counted in the pool's live bytes.
    let scratch = hal.alloc_private(4096);
    let next_batch = || {
        hal.blit_fill(&scratch, 0, 4096, 0);
        hal.sync();
    };
    hal.rt.note_domain(1 << 18);
    next_batch();
    // "po2 A": four live 16 MiB buffers, then all returned.
    let bufs: Vec<_> = (0..4)
        .map(|_| hal.rt.alloc_shared("a1", 16 * MIB).unwrap())
        .collect();
    let (_, live, peak) = hal.rt.pool_bytes();
    assert_eq!((live, peak), (64 * MIB, 64 * MIB));
    for buf in bufs {
        hal.rt.release(16 * MIB, buf);
    }
    assert_eq!(hal.rt.pool_bytes(), (64 * MIB, 0, 64 * MIB));
    // Same po2, a new size: nothing is evicted.
    next_batch();
    hal.rt.note_domain(1 << 18);
    let x = hal.rt.alloc_shared("a1", 20 * MIB).unwrap();
    assert_eq!(
        hal.rt.pool_bytes().0,
        64 * MIB,
        "no po2 change: no eviction"
    );
    hal.rt.release(20 * MIB, x);
    // "po2 B": 24 MiB buffers. Each miss evicts as much of po2 A's idle memory as it adds.
    next_batch();
    hal.rt.note_domain(1 << 19);
    let b1 = hal.rt.alloc_shared("a1", 24 * MIB).unwrap();
    assert_eq!(
        hal.rt.pool_bytes().0,
        52 * MIB,
        "the two oldest 16 MiB buffers evicted"
    );
    let b2 = hal.rt.alloc_shared("a1", 24 * MIB).unwrap();
    assert_eq!(
        hal.rt.pool_bytes().0,
        36 * MIB,
        "the third 16 MiB buffer evicted"
    );
    // Hits never evict.
    hal.rt.release(24 * MIB, b2);
    let b2 = hal.rt.alloc_shared("a1", 24 * MIB).unwrap();
    assert_eq!(hal.rt.pool_bytes().0, 36 * MIB);
    // Above the old mark: po2 A's last buffer goes; po2 B's own sizes are never evicted.
    let b3 = hal.rt.alloc_shared("a1", 24 * MIB).unwrap();
    let b4 = hal.rt.alloc_shared("a1", 24 * MIB).unwrap();
    assert_eq!(hal.rt.pool_bytes().0, 0);
    hal.rt.release(24 * MIB, b4);
    let b5 = hal.rt.alloc_shared("a1", 30 * MIB).unwrap();
    assert_eq!(
        hal.rt
            .pool_sizes()
            .iter()
            .filter(|s| s.0 == 24 * MIB)
            .count(),
        1
    );
    drop((b1, b2, b3, b5));
}

/// Same-size steady state: once the pool holds a job's buffers, later jobs hit for every buffer
/// and nothing is evicted.
#[test]
fn steady_same_size_never_misses() {
    let _lock = gpu_test_lock();
    let hal = hal_in(Runtime::new_test_pool_group());
    let sizes = [32 * MIB, 8 * MIB, 8 * MIB, 2 * MIB, 128 << 10];
    let mut first_ptrs = Vec::new();
    for job in 0..4 {
        hal.sync();
        let bufs: Vec<_> = sizes
            .iter()
            .map(|&s| hal.rt.alloc_shared("a1", s).unwrap())
            .collect();
        let mut ptrs: Vec<u64> = bufs.iter().map(|b| b.gpu_address()).collect();
        ptrs.sort();
        for (s, b) in sizes.iter().zip(bufs) {
            hal.rt.release(*s, b);
        }
        if job == 0 {
            first_ptrs = ptrs;
        } else {
            assert_eq!(ptrs, first_ptrs, "job {job} reused every buffer");
        }
    }
}

/// Sizes not requested within the staleness horizon (batches of the pool's own runtime) go.
#[test]
fn stale_sizes_are_evicted() {
    let _lock = gpu_test_lock();
    if global_pool_off() {
        return;
    }
    let hal = hal_in(Runtime::new_test_pool_group());
    hal.rt.set_stale_batches(8);
    pool_one(&hal, 4 * MIB);
    let scratch = hal.alloc_u32("scratch", 1024);
    for _ in 0..12 {
        hal.blit_fill(scratch.as_buf(), 0, 4096, 0);
        hal.flush();
    }
    hal.sync();
    assert_eq!(
        ready_sizes(&hal),
        vec![(4 * MIB, 1)],
        "no allocation yet: no scan"
    );
    // The next allocation scans: the 4 MiB size was last requested 12 batches ago.
    pool_one(&hal, MIB);
    assert_eq!(ready_sizes(&hal), vec![(MIB, 1)]);
}

/// A memory-pressure event (WARN or CRITICAL) releases idle pooled buffers and runs registered
/// releasers; it never touches a pending buffer (a batch may still use it), a buffer set aside
/// for prefill, or a live buffer. NORMAL does nothing.
#[test]
fn pressure_drains_only_idle_entries() {
    let _lock = gpu_test_lock();
    let hal = hal_in(Runtime::new_test_pool_group());
    raise_peak(&hal, 64 * MIB);
    pool_one(&hal, 2 * MIB);
    pool_one(&hal, 4 * MIB);
    // Set one 4 MiB buffer aside for prefill (Z14).
    assert!(hal.rt.prefill_reserve(4 * MIB, 7, u64::MAX));
    pool_one(&hal, 3 * MIB);
    let live = hal.rt.alloc_shared("live", MIB).unwrap();
    // A pending buffer: returned while a batch is open.
    let scratch = hal.alloc_u32("scratch", 1024);
    let pending = hal.rt.alloc_shared("pending", 5 * MIB).unwrap();
    hal.blit_fill(&pending, 0, 5 * MIB, 1);
    hal.rt.release(5 * MIB, pending);
    let before = hal.rt.pool_sizes();
    assert!(before.contains(&(5 * MIB, 1)) && before.contains(&(3 * MIB, 1)));

    let calls = Arc::new(AtomicU64::new(0));
    let c = calls.clone();
    let handle = super::register_idle_releaser(move || {
        c.fetch_add(1, Ordering::Relaxed);
        1234
    });
    assert_eq!(
        super::runtime::on_memory_pressure(super::PRESSURE_NORMAL),
        0
    );
    assert_eq!(hal.rt.pool_sizes(), before, "NORMAL does nothing");
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    let released = super::runtime::on_memory_pressure(super::PRESSURE_WARN);
    assert!(released >= 5 * MIB + 1234, "released {released}");
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        ready_sizes(&hal),
        vec![(5 * MIB, 1)],
        "only the pending buffer stays pooled"
    );
    assert_eq!(
        hal.rt.prefill_stats(),
        (1, 0),
        "the prefill buffer stays set aside"
    );
    // The pending buffer is intact and still pooled after its batch completes.
    hal.sync();
    drop(scratch);
    super::runtime::on_memory_pressure(super::PRESSURE_CRITICAL);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(
        ready_sizes(&hal),
        vec![],
        "completed pending buffers are idle"
    );
    drop(handle);
    super::runtime::on_memory_pressure(super::PRESSURE_WARN);
    assert_eq!(
        calls.load(Ordering::Relaxed),
        2,
        "an unregistered hook is not called"
    );
    // The live buffer was never touched.
    assert_eq!(live.length(), MIB);
    hal.rt.release(MIB, live);
}

/// Without an explicit budget, an allocation of 256 MiB or more that would take the system-wide
/// GPU memory past `recommendedMaxWorkingSetSize` drains this process's pools, then fails with
/// the working-set budget error (a fake IOAccelerator reading here). Smaller allocations and
/// runs with an explicit budget are not checked.
#[test]
fn system_check_fails_cleanly() {
    let _lock = gpu_test_lock();
    let hal = hal_in(Runtime::new_test_pool_group());
    let rec = hal.device.recommended_max_working_set_size();
    pool_one(&hal, 2 * MIB);
    hal.rt.set_fake_system_in_use(Some(rec - 100 * MIB));
    let big = super::SYSTEM_CHECK_MIN_BYTES;
    let err = hal.rt.alloc_shared("big", big).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("working-set budget exceeded") && msg.contains("system-wide"),
        "{msg}"
    );
    assert!(
        msg.contains("big") && msg.contains(&big.to_string()),
        "{msg}"
    );
    assert_eq!(
        ready_sizes(&hal),
        vec![],
        "the pool was drained before giving up"
    );
    assert!(hal.rt.alloc_private(big).is_err(), "private buffers too");
    // Below the threshold: not checked.
    let small = hal.rt.alloc_shared("small", 2 * MIB).unwrap();
    hal.rt.release(2 * MIB, small);
    // An explicit budget replaces the system check.
    hal.set_working_set_limit(Some(rec));
    let b = hal.rt.alloc_shared("big", big).unwrap();
    hal.rt.release(big, b);
    hal.set_working_set_limit(None);
    // Room again: allocates.
    hal.rt.set_fake_system_in_use(Some(MIB));
    let b = hal.rt.alloc_shared("big", big).unwrap();
    hal.rt.release(big, b);
    hal.rt.set_fake_system_in_use(None);
    hal.trim_device_memory();
}

/// The real IOAccelerator counter is readable without privileges and sees this process's
/// allocations.
#[test]
fn system_gpu_in_use_is_readable() {
    let v = super::system_gpu_in_use().expect("IOAccelerator 'In use system memory'");
    assert!(v > 0);
}

/// A real OS pressure event in a test build waits for [gpu_test_lock] before it drains, so it
/// cannot empty the pools of a test that holds the lock and asserts pool state (W7-A1 review:
/// the libdispatch source is installed in test processes too).
#[test]
fn os_pressure_event_waits_for_test_lock() {
    let lock = gpu_test_lock();
    let hal = hal_in(Runtime::new_test_pool_group());
    raise_peak(&hal, 64 * MIB);
    pool_one(&hal, 2 * MIB);
    let os = std::thread::spawn(|| super::runtime::os_pressure_event(super::PRESSURE_WARN));
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert!(
        !os.is_finished(),
        "the OS event must wait for the test lock"
    );
    assert_eq!(
        ready_sizes(&hal),
        vec![(2 * MIB, 1)],
        "pool drained under the lock"
    );
    drop(lock);
    // Once the lock is free the event drains (another test that takes the lock first may drain
    // this pool before it, so only the end state is checked).
    os.join().unwrap();
    let _lock = gpu_test_lock();
    assert_eq!(ready_sizes(&hal), vec![], "drained once the lock was free");
}

/// Idle buffers released because memory is short (a pressure drain, the system check's drain)
/// are purged (`setPurgeableState(Empty)`) first; routine frees (cap eviction, a dropped
/// runtime) are not, unless `RISC0_METAL_PURGE_ON_FREE=1`; a buffer a batch may still use never
/// is.
#[test]
fn relief_frees_are_purged() {
    let _lock = gpu_test_lock();
    if global_pool_off() {
        return;
    }
    if std::env::var_os(super::PURGE_ON_FREE_ENV).is_some() {
        return; // the default mode is what this test checks
    }
    let g = Runtime::new_test_pool_group();
    let hal = hal_in(g);
    let state = |b: &metal::Buffer| b.set_purgeable_state(MTLPurgeableState::KeepCurrent);
    // Drained by a pressure event: purged.
    hal.sync();
    let a = hal.rt.alloc_shared("a", 2 * MIB).unwrap();
    let keep_a = a.clone();
    hal.rt.release(2 * MIB, a);
    assert_eq!(state(&keep_a), MTLPurgeableState::NonVolatile);
    super::drain_idle_pools();
    assert_eq!(state(&keep_a), MTLPurgeableState::Empty);
    // Drained by the system check: purged.
    let b = hal.rt.alloc_shared("b", 2 * MIB).unwrap();
    let keep_b = b.clone();
    hal.rt.release(2 * MIB, b);
    let rec = hal.device.recommended_max_working_set_size();
    hal.rt.set_fake_system_in_use(Some(rec));
    assert!(hal
        .rt
        .alloc_shared("big", super::SYSTEM_CHECK_MIN_BYTES)
        .is_err());
    hal.rt.set_fake_system_in_use(None);
    assert_eq!(state(&keep_b), MTLPurgeableState::Empty);
    // Evicted by the cap: a routine free, not purged.
    let _cap = GroupCap::set(g, 3 * MIB);
    let c = hal.rt.alloc_shared("c", 2 * MIB).unwrap();
    let keep_c = c.clone();
    hal.rt.release(2 * MIB, c);
    raise_peak(&hal, 64 * MIB);
    pool_one(&hal, 2 * MIB + 4096);
    assert!(
        ready_sizes(&hal).iter().all(|s| s.0 != 2 * MIB),
        "c was evicted"
    );
    assert_eq!(state(&keep_c), MTLPurgeableState::NonVolatile);
    // A pending buffer drained by a pressure event is neither dropped nor purged.
    let scratch = hal.alloc_private(4096);
    let d = hal.rt.alloc_shared("d", MIB).unwrap();
    let keep_d = d.clone();
    hal.blit_fill(&d, 0, MIB, 3);
    hal.blit_fill(&scratch, 0, 4096, 3);
    hal.rt.release(MIB, d);
    super::drain_idle_pools();
    hal.sync();
    assert_eq!(
        state(&keep_d),
        MTLPurgeableState::NonVolatile,
        "was in flight"
    );
    drop(hal);
}

/// Z14's single-thread assumption is checked: a prefill buffer marked filled on one thread and
/// taken on another trips a debug assertion.
#[cfg(debug_assertions)]
#[test]
fn prefill_cross_thread_take_is_caught() {
    let _lock = gpu_test_lock();
    let hal = hal_in(Runtime::new_test_pool_group());
    pool_one(&hal, MIB);
    assert!(hal.rt.prefill_reserve(MIB, 5, u64::MAX));
    assert_eq!(hal.rt.prefill_unfilled(4).len(), 1);
    let rt = hal.rt.clone();
    let r =
        std::thread::spawn(move || rt.prefill_take(MIB, 5).map(|(b, f)| (b.length(), f))).join();
    assert!(
        r.is_err(),
        "the cross-thread take must panic in debug builds"
    );
    // Same thread: fine.
    pool_one(&hal, MIB);
    assert!(hal.rt.prefill_reserve(MIB, 6, u64::MAX));
    assert_eq!(hal.rt.prefill_unfilled(4).len(), 1);
    let (buf, filled) = hal.rt.prefill_take(MIB, 6).unwrap();
    assert!(filled);
    hal.rt.release(MIB, buf);
}

/// Probe (ignored): does `memory_pressure -S -l warn` (no sudo) deliver a WARN event to the
/// libdispatch source? Prints the outcome; asserts nothing about the OS.
#[test]
#[ignore]
fn probe_simulated_pressure_event() {
    super::require_gpu_run();
    let _lock = gpu_test_lock();
    let hal = hal_in(Runtime::new_test_pool_group());
    pool_one(&hal, 8 * MIB);
    let before = super::pool_counters().pressure_events;
    let out = std::process::Command::new("memory_pressure")
        .args(["-S", "-l", "warn"])
        .output();
    eprintln!("[a1] memory_pressure -S -l warn: {out:?}");
    for _ in 0..50 {
        if super::pool_counters().pressure_events > before {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let after = super::pool_counters().pressure_events;
    eprintln!(
        "[a1] pressure events {before} -> {after}; pooled now {:?}",
        hal.rt.pool_sizes()
    );
    let _ = std::process::Command::new("memory_pressure")
        .args(["-S", "-l", "normal"])
        .output();
}

/// Probe (ignored): IOAccelerator 'In use system memory' / 'Alloc system memory' against this
/// process's `currentAllocatedSize`, for a 1 GiB buffer through its life (new, CPU-touched,
/// used by a GPU blit, purged, dropped), and the read cost.
#[test]
#[ignore]
fn probe_system_counter() {
    super::require_gpu_run();
    let _lock = gpu_test_lock();
    let hal = hal_in(Runtime::new_test_pool_group());
    let t = std::time::Instant::now();
    for _ in 0..100 {
        super::system_gpu_in_use();
    }
    let read_us = t.elapsed().as_secs_f64() * 1e4;
    let m = |b: u64| b as f64 / MIB as f64;
    let read = |what: &str| {
        let u = super::system_gpu_in_use().unwrap();
        let a = super::runtime::accelerator_stat(c"Alloc system memory").unwrap_or(0);
        eprintln!(
            "[a1] {what:>12}: in-use {:.0} MiB, alloc {:.0} MiB, currentAllocatedSize {:.0} MiB",
            m(u),
            m(a),
            m(hal.allocated_size())
        );
    };
    eprintln!("[a1] read {read_us:.1} us");
    read("before");
    for opts in [
        MTLResourceOptions::StorageModeShared,
        MTLResourceOptions::StorageModePrivate,
    ] {
        eprintln!("[a1] {opts:?}");
        let buf = hal.device.new_buffer(1 << 30, opts);
        read("new 1 GiB");
        if opts == MTLResourceOptions::StorageModeShared {
            unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 1, 1 << 30) };
            read("CPU-touched");
        }
        hal.blit_fill(&buf, 0, 1 << 30, 2);
        hal.sync();
        read("GPU blit");
        std::thread::sleep(std::time::Duration::from_millis(200));
        read("+200 ms");
        buf.set_purgeable_state(MTLPurgeableState::Empty);
        read("purged");
        drop(buf);
        read("dropped");
        std::thread::sleep(std::time::Duration::from_millis(200));
        read("+200 ms");
    }
}
