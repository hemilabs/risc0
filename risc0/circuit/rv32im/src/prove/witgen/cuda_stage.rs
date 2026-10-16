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

//! Pinned staging of the preflight trace for the CUDA witgen upload.
//!
//! The witgen FFI used to copy the trace (cycles, txns, bigint bytes: ~124 MB at po2=20) into
//! a pinned buffer on the proving thread, with the GPU idle (~9.5 ms/segment). The preflight
//! thread now does that copy right after the preflight, into a slot from a small pool of pinned
//! buffers, and the trace owns the slot ([`PreflightTrace::staged`]) until it is dropped (after
//! the eval_check launch, long after the stream has passed the H2D copies, which precede the
//! globals readback in `WitnessGenerator::header`).

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, LazyLock, Mutex, Weak,
};

use rayon::prelude::*;
use risc0_circuit_rv32im_sys::{
    risc0_circuit_rv32im_cuda_event_wait, risc0_circuit_rv32im_cuda_host_alloc,
    risc0_circuit_rv32im_cuda_host_free, risc0_circuit_rv32im_cuda_prefetch,
    RawMemoryTransaction, RawPreflightCycle,
};
use risc0_sys::ffi_wrap;

use super::preflight::{Back, PreflightTrace};
use crate::zirgen::circuit::ExtVal;

/// Staging on unless RISC0_C2_NOSTAGE is set (A/B switch).
pub(crate) static STAGE: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("RISC0_C2_NOSTAGE").is_none());

/// Free pinned slots: (address, capacity).
static POOL: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

pub(crate) struct StagedTrace {
    ptr: *mut u8,
    cap: usize,
    /// Byte lengths of cycles, txns and bigint bytes, back to back from `ptr`.
    lens: [usize; 3],
    /// Unique per staging (the device copy is matched by it, not by address).
    pub(crate) id: u64,
    /// Event recorded after this slot's prefetch copy (0: none). The DMA reads the pinned
    /// slot asynchronously, so Drop waits for it before the slot goes back to the pool.
    copy_ev: Mutex<usize>,
}

/// Prefetch on unless RISC0_C2_NOPREFETCH is set (A/B switch).
static PREFETCH: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("RISC0_C2_NOPREFETCH").is_none());

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Staged traces not yet prefetched (pushed by the preflight thread).
static READY: Mutex<Vec<Weak<StagedTrace>>> = Mutex::new(Vec::new());

/// Start the device copy of the staged traces that are ready, on a side stream overlapping the
/// GPU work in flight. Call on the proving thread (GPU lock held), between segments' GPU work:
/// the CUDA HAL calls it once eval_check is done, while the finalize kernels run.
pub(crate) fn prefetch_ready() {
    if !*PREFETCH {
        return;
    }
    let ready: Vec<Weak<StagedTrace>> = std::mem::take(&mut *READY.lock().unwrap());
    for weak in ready {
        // Dropped (already proved): nothing to do.
        if let Some(staged) = weak.upgrade() {
            let total: usize = staged.lens.iter().sum();
            let mut ev: *mut std::ffi::c_void = std::ptr::null_mut();
            if let Err(err) = ffi_wrap(|| unsafe {
                risc0_circuit_rv32im_cuda_prefetch(staged.ptr, total, staged.id, &mut ev)
            }) {
                tracing::debug!("cuda_stage: prefetch failed: {err}");
            }
            if !ev.is_null() {
                let old = std::mem::replace(&mut *staged.copy_ev.lock().unwrap(), ev as usize);
                if old != 0 {
                    let _ = ffi_wrap(|| unsafe { risc0_circuit_rv32im_cuda_event_wait(old as *mut _) });
                }
            }
            // `staged` (the Arc) may be the last reference: its Drop waits for the copy.
        }
    }
}

// SAFETY: the slot is plain pinned host memory owned by this value.
unsafe impl Send for StagedTrace {}
unsafe impl Sync for StagedTrace {}

impl StagedTrace {
    /// The staged bytes, if they are those of `trace` (same lengths).
    pub(crate) fn ptr_for(&self, trace: &PreflightTrace) -> Option<*const u8> {
        (self.lens == byte_lens(trace)).then_some(self.ptr as *const u8)
    }
}

impl Drop for StagedTrace {
    fn drop(&mut self) {
        // The prefetch DMA may still be reading the slot (e.g. a trace dropped unproven on an
        // error path): wait for it before another trace can be staged into the slot.
        let ev = *self.copy_ev.get_mut().unwrap_or_else(|e| e.into_inner());
        if ev != 0 {
            if let Err(err) = ffi_wrap(|| unsafe { risc0_circuit_rv32im_cuda_event_wait(ev as *mut _) }) {
                // Don't recycle a slot that may still be read by the device.
                tracing::warn!("cuda_stage: prefetch event wait failed: {err}; leaking slot");
                return;
            }
        }
        POOL.lock().unwrap().push((self.ptr as usize, self.cap));
    }
}

fn byte_lens(trace: &PreflightTrace) -> [usize; 3] {
    [
        std::mem::size_of_val(trace.cycles.as_slice()),
        std::mem::size_of_val(trace.txns.as_slice()),
        trace.bigint_bytes.len(),
    ]
}

fn take_slot(len: usize) -> Option<(*mut u8, usize)> {
    let mut stale = Vec::new();
    let found = {
        let mut pool = POOL.lock().unwrap();
        match pool.iter().position(|&(_, cap)| cap >= len) {
            Some(i) => Some(pool.swap_remove(i)),
            None => {
                // Too small for this trace: free one (the trace grows over the first segments).
                if let Some(slot) = pool.pop() {
                    stale.push(slot);
                }
                None
            }
        }
    };
    for (ptr, _) in stale {
        // cudaFreeHost synchronizes the device; this runs on the preflight thread.
        let _ = ffi_wrap(|| unsafe { risc0_circuit_rv32im_cuda_host_free(ptr as *mut _) });
    }
    if let Some((ptr, cap)) = found {
        return Some((ptr as *mut u8, cap));
    }
    let cap = len + len / 4;
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    match ffi_wrap(|| unsafe { risc0_circuit_rv32im_cuda_host_alloc(cap, &mut ptr) }) {
        Ok(()) if !ptr.is_null() => Some((ptr as *mut u8, cap)),
        Ok(()) => None,
        Err(err) => {
            tracing::debug!("cuda_stage: pinned alloc of {cap} bytes failed: {err}");
            None
        }
    }
}

fn par_copy(dst: *mut u8, src: &[u8]) {
    const CHUNK: usize = 4 << 20;
    let dst = dst as usize;
    src.par_chunks(CHUNK).enumerate().for_each(|(i, chunk)| unsafe {
        std::ptr::copy_nonoverlapping(
            chunk.as_ptr(),
            (dst as *mut u8).add(i * CHUNK),
            chunk.len(),
        );
    });
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data repr(C) structs / bytes, read only.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Copy `trace`'s device inputs into a pinned slot (None: staging off or no pinned memory, the
/// witgen FFI then stages on the proving thread as before).
pub(crate) fn stage(trace: &PreflightTrace) -> Option<Arc<StagedTrace>> {
    if !*STAGE {
        return None;
    }
    let lens = byte_lens(trace);
    let total: usize = lens.iter().sum();
    let (ptr, cap) = take_slot(total.max(1))?;
    let cycles: &[RawPreflightCycle] = &trace.cycles;
    let txns: &[RawMemoryTransaction] = &trace.txns;
    par_copy(ptr, as_bytes(cycles));
    par_copy(unsafe { ptr.add(lens[0]) }, as_bytes(txns));
    par_copy(unsafe { ptr.add(lens[0] + lens[1]) }, &trace.bigint_bytes);
    let staged = Arc::new(StagedTrace {
        ptr,
        cap,
        lens,
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        copy_ev: Mutex::new(0),
    });
    if *PREFETCH {
        READY.lock().unwrap().push(Arc::downgrade(&staged));
    }
    Some(staged)
}

/// Reuse the trace vectors of finished segments (RISC0_C2_RECYCLE=0 disables): a fresh ~300 MB
/// of vectors per po2=20 segment costs the preflight thread page faults and zeroing, and the
/// txns vector its doubling reallocations.
static RECYCLE: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("RISC0_C2_RECYCLE").map_or(true, |v| v != "0")
});

type TraceVecs = (Vec<RawPreflightCycle>, Vec<RawMemoryTransaction>, Vec<u8>, Vec<Back>);
static TRACE_POOL: Mutex<Vec<TraceVecs>> = Mutex::new(Vec::new());

pub(crate) fn recycled_trace(total_cycles: usize, rand_z: ExtVal) -> PreflightTrace {
    let vecs = if *RECYCLE {
        TRACE_POOL.lock().unwrap().pop()
    } else {
        None
    };
    let (mut cycles, txns, bigint_bytes, mut backs) = vecs.unwrap_or_default();
    cycles.reserve(total_cycles);
    backs.reserve(total_cycles);
    // (No struct update syntax: PreflightTrace implements Drop.)
    let mut trace = PreflightTrace::default();
    trace.cycles = cycles;
    trace.txns = txns;
    trace.bigint_bytes = bigint_bytes;
    trace.backs = backs;
    trace.rand_z = rand_z;
    trace
}

pub(crate) fn recycle_trace(trace: &mut PreflightTrace) {
    if !*RECYCLE || trace.cycles.capacity() == 0 {
        return;
    }
    let mut vecs = (
        std::mem::take(&mut trace.cycles),
        std::mem::take(&mut trace.txns),
        std::mem::take(&mut trace.bigint_bytes),
        std::mem::take(&mut trace.backs),
    );
    vecs.0.clear();
    vecs.1.clear();
    vecs.2.clear();
    vecs.3.clear();
    let mut pool = TRACE_POOL.lock().unwrap();
    if pool.len() < 2 {
        pool.push(vecs);
    }
}
