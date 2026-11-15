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

//! C14: the recursion witgen inputs staged in a Metal shared buffer on the prepare thread (the
//! Metal counterpart of CUDA's `pinned.rs`, PLAN 4.3 / LEADS #12).
//!
//! `prepare()` copies the preflight trace (wom, cycles, iops) and the host chains (heads, cycles,
//! rows, wom_idx, wom_rows) into one `StorageModeShared` buffer, off the GPU thread. The Metal
//! witgen finds that buffer from the trace pointers ([lookup]) and binds it directly: wom and iops
//! are read in place, the cycles are copied to the HAL scratch by a GPU kernel (the kernels
//! advance `iopIdx`, and the checks re-run the witgen on the same trace), and the chains are
//! applied by C1's `markHostChains` / `scatterHostChains`. The witgen then needs no host copies
//! and no sync before parStepExec.
//!
//! The buffer belongs to the [crate::prove::Preflight] (`pinned`) and goes back to a small pool
//! when the preflight is dropped (a fresh buffer costs its page faults again, ~1 ms per 4 MB).
//! Every witgen that reads it validates before encoding and waits for its command buffers before
//! returning, so the GPU has finished reading a buffer before it can be reused (a witgen that ends
//! early retires its buffer instead, W7-C1); Metal also retains a bound buffer until its command
//! buffer completes. The buffers are device allocations, so the device's `currentAllocatedSize`,
//! which the HAL's default budget is checked against, includes them: about 24 MB per lift and
//! 36 MB per join at po2 18 (with headroom), at most [POOL_CAP] pooled plus one per prepared, not
//! yet proven preflight.
//!
//! On by default since W7-C1; `RISC0_METAL_REC_STAGE=0` turns it off (the witgen copies on the GPU
//! thread, as before C14). A new buffer is allocated only if it leaves [STAGE_MARGIN_BYTES]
//! (512 MiB) under the working-set limit (W7-C1; otherwise the witgen copies on the GPU thread),
//! and the pool is drained before the Groth16 wrap (`release_cached_provers`). The pool is
//! process-wide and is not drained by the HAL's budget check (`drain_all_pools` cannot see it):
//! a process that proves succinct receipts without a wrap keeps up to [POOL_CAP] buffers
//! (<= ~135 MB) into its next session, like the cached recursion provers' ctrl groups; call
//! `release_cached_provers` to return them. With `RISC0_METAL_DEBUG=1` the witgen asserts, before
//! it returns, that no
//! unsynced batch still uses a staged buffer (the HAL's in-flight tracking records raw
//! `KernelArg::Buffer` bindings too), i.e. that the buffer is safe to pool and rewrite.

use std::sync::{LazyLock, Mutex};

use metal::{Buffer, Device, MTLResourceOptions};
use risc0_circuit_recursion_sys::RawPreflightCycle;
use risc0_zkp::field::baby_bear::BabyBearExtElem;

use crate::prove::witgen::HostChains;

pub(crate) const STAGE_ENV: &str = "RISC0_METAL_REC_STAGE";

thread_local! {
    /// Test override of [STAGE_ENV].
    static STAGE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether `prepare()` stages the witgen inputs and the witgen binds them (read once). **On by
/// default since W7-C1** (`RISC0_METAL_REC_STAGE=0` reverts to the pre-C14 host copies): a quiet
/// interleaved A/B with its own A-vs-A arm measured the worker's memory-merkle lift/join time
/// -36 ms (default budget), -121 ms (4 GiB) and -68 ms (8 GiB), 7/7 rounds each, against
/// A-vs-A medians of 1-7 ms. Re-measured base binary against branch binary (fix round 1):
/// -29 / -138 / -63 ms, 8/8 rounds each (notes/W7-C1.md section 3b).
pub(crate) fn stage_enabled() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| parse_stage(std::env::var(STAGE_ENV).ok().as_deref()));
    STAGE_OVERRIDE.with(|c| c.get()).unwrap_or(*ON)
}

/// [STAGE_ENV]'s value: on unless `0` / `off` / `false`.
pub(crate) fn parse_stage(v: Option<&str>) -> bool {
    !matches!(
        v.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "off" | "false")
    )
}

/// Run `f` with staging forced on or off on this thread (tests).
#[allow(dead_code)]
pub(crate) fn with_stage<R>(on: bool, f: impl FnOnce() -> R) -> R {
    let old = STAGE_OVERRIDE.with(|c| c.replace(Some(on)));
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            STAGE_OVERRIDE.with(|c| c.set(self.0));
        }
    }
    let _r = Restore(old);
    f()
}

/// Live staged buffers: (base address, length, buffer), for [lookup].
static LIVE: Mutex<Vec<(usize, usize, Buffer)>> = Mutex::new(Vec::new());

static DEVICE: LazyLock<Option<Device>> = LazyLock::new(Device::system_default);

/// Released staging buffers, reused by the next [MetalStaged::new] that fits.
static POOL: Mutex<Vec<Buffer>> = Mutex::new(Vec::new());
/// Pooled buffers at most (a lift and a join in flight, plus one).
const POOL_CAP: usize = 3;

/// A buffer of at least `bytes` from the pool, else a new one, or `None` when a new one would take
/// the device above its working-set limit (W7-C1): staging buffers are device allocations, so they
/// count against the HAL's default (`device`-scope) budget like every HAL buffer, and staging
/// gives way (the witgen copies on the GPU thread instead) rather than push a small budget over.
/// Under `RISC0_METAL_BUDGET_SCOPE=hal` they are in no HAL's own count.
fn take_buffer(device: &Device, bytes: usize) -> Option<Buffer> {
    {
        let mut pool = POOL.lock().unwrap();
        if let Some(i) = pool
            .iter()
            .enumerate()
            .filter(|(_, b)| b.length() as usize >= bytes)
            .min_by_key(|(_, b)| b.length())
            .map(|(i, _)| i)
        {
            return Some(pool.swap_remove(i));
        }
    }
    // Headroom so the next proof of a similar size reuses it.
    let cap = (bytes + bytes / 4 + (1 << 20)) & !((1 << 20) - 1);
    let fits = || admit(device_memory(), cap as u64);
    if !fits() {
        // The pooled buffers are too small for this proof: give them back first.
        drain_pool();
        if !fits() {
            tracing::debug!("recursion Metal staging: {cap} B would leave under {STAGE_MARGIN_BYTES} B of the working-set limit");
            return None;
        }
    }
    Some(device.new_buffer(cap as u64, MTLResourceOptions::StorageModeShared))
}

/// Free bytes a new staging buffer must leave under the device's working-set limit (W7-C1 fix
/// round 1). `prepare()` allocates on its own thread, without `exclusive()`, while a Prove may be
/// allocating on the GPU thread: the check and the allocation are not atomic with the HAL's, so
/// staging keeps out of the last 512 MiB (the p254 prewarm's margin) rather than take the bytes
/// the Prove's next allocation needs. A staging buffer is 24-45 MB, and at most [POOL_CAP] plus
/// one per prepared proof exist. At a 4 GiB budget the recursion phase peaks at 1910 MiB, far
/// below the point where the margin turns staging off (W7-C1 note, section 3).
pub(crate) const STAGE_MARGIN_BYTES: u64 = 512 << 20;

/// Whether a new `cap`-byte staging buffer is admitted, given the device's `(limit, allocated)`.
pub(crate) fn admit(mem: Option<(u64, u64)>, cap: u64) -> bool {
    mem.is_none_or(|(limit, allocated)| {
        allocated
            .saturating_add(cap)
            .saturating_add(STAGE_MARGIN_BYTES)
            <= limit
    })
}

#[cfg(test)]
thread_local! {
    /// Test override of the device's `(limit, allocated)` for [take_buffer] on this thread.
    pub(crate) static FAKE_DEVICE_MEMORY: std::cell::Cell<Option<(u64, u64)>> =
        const { std::cell::Cell::new(None) };
}

fn device_memory() -> Option<(u64, u64)> {
    #[cfg(test)]
    if let Some(m) = FAKE_DEVICE_MEMORY.with(|f| f.get()) {
        return Some(m);
    }
    risc0_zkp::hal::metal::device_memory()
}

fn return_buffer(buf: Buffer) {
    let mut pool = POOL.lock().unwrap();
    if pool.len() < POOL_CAP {
        pool.push(buf);
    } else if let Some(small) = pool
        .iter_mut()
        .filter(|b| b.length() < buf.length())
        .min_by_key(|b| b.length())
    {
        // Keep the larger buffer: later proofs of this size then find one.
        *small = buf;
    }
}

/// Drop the pooled buffers (also before the Groth16 wrap, `release_cached_provers`; the pool
/// refills on demand). Returns how many were dropped.
pub(crate) fn drain_pool() -> usize {
    std::mem::take(&mut *POOL.lock().unwrap()).len()
}

/// The staged trace: wom, cycles, iops, then chain heads, cycles, rows, wom_idx, wom_rows (the
/// order of CUDA's `PinnedTrace`), each 16-byte aligned in one shared buffer.
pub(crate) struct MetalStaged {
    buf: Option<Buffer>,
    off: [usize; 8],
}

// SAFETY: an MTLBuffer may be used from any thread; the contents are written only in `new`
// (before the value is shared) and read by the GPU afterwards.
unsafe impl Send for MetalStaged {}
unsafe impl Sync for MetalStaged {}

impl MetalStaged {
    pub(crate) fn new(
        wom: &[BabyBearExtElem],
        cycles: &[RawPreflightCycle],
        iops: &[BabyBearExtElem],
        chains: Option<&HostChains>,
    ) -> Option<Self> {
        let e: &[u32] = &[];
        let (h, c, r, wi, wr) = match chains {
            Some(hc) => (
                &hc.heads[..],
                &hc.cycles[..],
                &hc.rows[..],
                &hc.wom_idx[..],
                &hc.wom_rows[..],
            ),
            None => (e, e, e, e, e),
        };
        // SAFETY: plain-old-data views (BabyBearExtElem and RawPreflightCycle are u32 words).
        let parts: [&[u8]; 8] = unsafe {
            [
                std::slice::from_raw_parts(wom.as_ptr() as *const u8, std::mem::size_of_val(wom)),
                std::slice::from_raw_parts(
                    cycles.as_ptr() as *const u8,
                    std::mem::size_of_val(cycles),
                ),
                std::slice::from_raw_parts(iops.as_ptr() as *const u8, std::mem::size_of_val(iops)),
                bytemuck::cast_slice(h),
                bytemuck::cast_slice(c),
                bytemuck::cast_slice(r),
                bytemuck::cast_slice(wi),
                bytemuck::cast_slice(wr),
            ]
        };
        let mut off = [0usize; 8];
        let mut len = 0usize;
        for (i, p) in parts.iter().enumerate() {
            off[i] = len;
            len += p.len().next_multiple_of(16);
        }
        let device = DEVICE.as_ref()?;
        let buf = take_buffer(device, len.max(16))?;
        let base = buf.contents() as *mut u8;
        if base.is_null() {
            return None;
        }
        for (i, p) in parts.iter().enumerate() {
            copy_par(unsafe { base.add(off[i]) }, p);
        }
        LIVE.lock()
            .unwrap()
            .push((base as usize, len.max(16), buf.clone()));
        Some(Self {
            buf: Some(buf),
            off,
        })
    }

    /// The address of array `i` (the order of [MetalStaged]).
    pub(crate) fn ptr<T>(&self, i: usize) -> *const T {
        let buf = self.buf.as_ref().expect("live staging buffer");
        unsafe { (buf.contents() as *const u8).add(self.off[i]) as *const T }
    }
}

impl Drop for MetalStaged {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            let base = buf.contents() as usize;
            LIVE.lock().unwrap().retain(|(b, _, _)| *b != base);
            let retired = {
                let mut r = RETIRED.lock().unwrap();
                let n = r.len();
                r.retain(|b| *b != base);
                r.len() != n
            };
            if retired {
                count_retired();
            } else {
                return_buffer(buf);
            }
        }
    }
}

/// Bases of staged buffers a witgen left without finishing (W7-C1): dropped, never pooled.
static RETIRED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

thread_local! {
    /// Staged buffers this thread dropped instead of pooled (tests; per thread, so a test's delta
    /// does not see other tests' retirements).
    static RETIRED_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn count_retired() {
    RETIRED_COUNT.with(|c| c.set(c.get() + 1));
}

/// Staged buffers this thread dropped instead of pooled so far (tests).
#[allow(dead_code)]
pub(crate) fn retired_count() -> usize {
    RETIRED_COUNT.with(|c| c.get())
}

/// Mark the staged buffer `buf` (any binding of one) as not reusable: when its preflight drops it,
/// it is released instead of returned to the pool (a witgen that bound it ended early, W7-C1).
pub(crate) fn retire(buf: &metal::BufferRef) {
    let base = buf.contents() as usize;
    let mut r = RETIRED.lock().unwrap();
    if !r.contains(&base) {
        r.push(base);
    }
}

/// The staged buffer holding `bytes` bytes at `ptr`, and the byte offset of `ptr` in it.
pub(crate) fn lookup(ptr: *const u8, bytes: usize) -> Option<(Buffer, u64)> {
    let p = ptr as usize;
    LIVE.lock()
        .unwrap()
        .iter()
        .find(|(b, len, _)| p >= *b && p + bytes <= b + len)
        .map(|(b, _, buf)| (buf.clone(), (p - b) as u64))
}

fn copy_par(dst: *mut u8, src: &[u8]) {
    use rayon::prelude::*;
    const CH: usize = 1 << 20;
    let dst = dst as usize;
    src.par_chunks(CH).enumerate().for_each(|(i, c)| unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), (dst + i * CH) as *mut u8, c.len());
    });
}
