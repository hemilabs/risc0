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

//! The command-buffer batch, the buffer pool and the encoding helpers behind `MetalHal`
//! (plan sections 2.6 and 2.9).
//!
//! Sync model: `dispatch`/`launch`/`blit_fill` only encode into the open batch (one command
//! buffer, one serial compute encoder; a blit switches encoders). `flush` ends encoding and
//! commits without waiting; `sync` flushes and waits for every committed batch. Host access to a
//! `BufferImpl` (`view`, `view_mut`, `get_at`, `to_vec`) syncs first. A batch is also committed
//! after `batch_limit` dispatches (default [DEFAULT_BATCH_DISPATCHES], `RISC0_METAL_BATCH_DISPATCHES`
//! overrides it).
//!
//! Lock order: [super::exclusive] (process-wide, reentrant) is taken before the per-HAL state
//! mutex on every path that takes the state mutex, except the debug host-access check, which only
//! `try_lock`s it. The pool mutex is never held while taking the state mutex.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    ffi::{c_char, c_void, CStr},
    mem,
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

use anyhow::{anyhow, Result};
use metal::{
    foreign_types::{ForeignType, ForeignTypeRef},
    objc::{msg_send, rc::autoreleasepool, runtime::Object, sel, sel_impl},
    BlitCommandEncoder, Buffer as MtlBuffer, BufferRef, CommandBuffer, CommandQueue,
    ComputeCommandEncoder, ComputeCommandEncoderRef, ComputePipelineStateRef, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSRange,
};
use parking_lot::{Mutex, MutexGuard};

use super::exclusive;

/// Dispatches after which an open batch is committed. D0 measured no encoding limit up to 262144
/// dispatches in one command buffer/encoder (`probes/README.md`, "Maximum dispatches per command
/// buffer"); 2048 bounds the latency of the first committed work, as plan 2.6 chose.
pub const DEFAULT_BATCH_DISPATCHES: usize = 2048;

/// Largest payload `set_bytes` accepts (Metal API limit).
pub const MAX_SET_BYTES: usize = 4096;

/// Buffers below this size are always pooled; larger ones only within the pool cap.
const POOL_SMALL_BYTES: u64 = 64 << 10;

/// One argument bound at index `i` for the `i`-th entry of the slice given to a dispatch.
pub enum KernelArg<'a> {
    /// No buffer bound (`set_buffer(i, None, 0)`).
    Null,
    /// A buffer at a byte offset.
    Buffer { buffer: &'a BufferRef, offset: u64 },
    /// A `u32` scalar, bound with `set_bytes` (kernels take `const device uint&` / `constant uint&`).
    Integer(u32),
    /// Up to 4096 bytes bound with `set_bytes` (POD param structs, small field constants).
    Bytes(&'a [u8]),
}

/// The shape of a dispatch.
#[derive(Clone, Copy, Debug)]
pub enum Grid {
    /// `dispatch_threads` over `n` threads, threadgroups of `min(256, pso max)` threads.
    Threads(u64),
    /// `dispatch_threads` over `x * y` threads, threadgroups of `simd width x (min(256, max)/simd width)`.
    Threads2D(u64, u64),
    /// `dispatch_thread_groups(groups, threads)` (CUDA `<<<grid, block>>>`).
    Groups { groups: MTLSize, threads: MTLSize },
}

/// The bytes of a slice of padding-free values, for [KernelArg::Bytes].
///
/// `T: bytemuck::NoUninit` guarantees every byte of `T` is initialised (no padding), so this is
/// sound for any caller. Kernel `Params` structs passed through `set_bytes` must therefore be
/// `#[repr(C)]` with explicit padding fields and `#[derive(bytemuck::NoUninit)]`.
pub fn as_bytes<T: bytemuck::NoUninit>(values: &[T]) -> &[u8] {
    bytemuck::cast_slice(values)
}

enum Encoder {
    None,
    Compute(ComputeCommandEncoder),
    Blit(BlitCommandEncoder),
}

impl Encoder {
    fn end(&mut self) {
        match mem::replace(self, Encoder::None) {
            Encoder::None => {}
            Encoder::Compute(enc) => enc.end_encoding(),
            Encoder::Blit(enc) => enc.end_encoding(),
        }
    }
}

/// What the current compute encoder has bound, to skip redundant state changes (the validation
/// layer warns about them, and they cost encode time).
#[derive(Default)]
pub(crate) struct Bound {
    pso: usize,
    buffers: Vec<Option<(usize, u64)>>,
}

impl Bound {
    fn set_buffer(&mut self, index: usize, value: Option<(usize, u64)>) -> bool {
        if self.buffers.len() <= index {
            self.buffers.resize(index + 1, None);
        }
        let changed = self.buffers[index] != value || value.is_none();
        self.buffers[index] = value;
        changed
    }
}

struct OpenBatch {
    seq: u64,
    cmd: CommandBuffer,
    encoder: Encoder,
    bound: Bound,
    dispatches: usize,
    /// Buffers bound in this batch (object addresses); filled only in debug mode.
    used: HashSet<usize>,
}

struct InFlight {
    seq: u64,
    cmd: CommandBuffer,
    used: HashSet<usize>,
}

#[derive(Default)]
struct State {
    open: Option<OpenBatch>,
    in_flight: VecDeque<InFlight>,
    /// The first command-buffer error not yet reported by `try_sync`.
    error: Option<String>,
    next_seq: u64,
}

/// Size-keyed pool of shared buffers. A buffer dropped while a batch that may use it is still
/// open or running stays in `pending` until that batch has completed (plan 2.6).
#[derive(Default)]
struct Pool {
    ready: HashMap<u64, Vec<MtlBuffer>>,
    pending: Vec<(u64, u64, MtlBuffer)>,
    cached_bytes: u64,
}

impl Pool {
    fn reclaim(&mut self, completed: u64) {
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].0 <= completed {
                let (_, len, buf) = self.pending.swap_remove(i);
                self.ready.entry(len).or_default().push(buf);
            } else {
                i += 1;
            }
        }
    }

    fn pop(&mut self, len: u64) -> Option<MtlBuffer> {
        let buf = self.ready.get_mut(&len)?.pop()?;
        self.cached_bytes -= len;
        Some(buf)
    }

    fn drain(&mut self) -> u64 {
        let bytes = self.cached_bytes;
        // Dropping a buffer still referenced by a committed command buffer is safe: the command
        // buffer retains every resource it binds until it completes.
        self.ready.clear();
        self.pending.clear();
        self.cached_bytes = 0;
        bytes
    }
}

/// State shared by a `MetalHal` and every buffer it allocated.
pub(crate) struct Runtime {
    pub(crate) device: Device,
    pub(crate) queue: CommandQueue,
    state: Mutex<State>,
    pool: Mutex<Pool>,
    /// Sequence number of the most recently opened batch (0: none yet).
    open_seq: AtomicU64,
    /// Every batch with a sequence number up to this one has completed.
    completed_seq: AtomicU64,
    debug: AtomicBool,
    batch_limit: AtomicUsize,
    pub(crate) max_tg_mem: u64,
    pub(crate) max_buffer_len: u64,
    pool_cap: u64,
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v.trim() == "1")
}

/// The `localizedDescription` of a command buffer's `error` (metal-rs 0.29 has no accessor).
// objc's `msg_send!` expands a `cfg(feature = "cargo-clippy")` this crate does not declare.
#[allow(unexpected_cfgs)]
fn command_buffer_error(cmd: &CommandBuffer) -> String {
    // SAFETY: plain Objective-C messages on live objects; nil results are checked.
    unsafe {
        let err: *mut Object = msg_send![cmd.as_ptr() as *mut Object, error];
        if err.is_null() {
            return "unknown error".into();
        }
        let desc: *mut Object = msg_send![err, localizedDescription];
        if desc.is_null() {
            return "unknown error".into();
        }
        let cstr: *const c_char = msg_send![desc, UTF8String];
        if cstr.is_null() {
            return "unknown error".into();
        }
        CStr::from_ptr(cstr).to_string_lossy().into_owned()
    }
}

impl Runtime {
    pub(crate) fn new(device: Device) -> Self {
        let queue = device.new_command_queue();
        let batch_limit = std::env::var("RISC0_METAL_BATCH_DISPATCHES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_BATCH_DISPATCHES);
        let debug = env_flag("RISC0_METAL_DEBUG");
        if debug && !env_flag("MTL_DEBUG_LAYER") {
            tracing::warn!(
                "RISC0_METAL_DEBUG=1 without MTL_DEBUG_LAYER=1: the Metal validation layer is off"
            );
        }
        let max_tg_mem = device.max_threadgroup_memory_length();
        let max_buffer_len = device.max_buffer_length();
        let pool_cap = device.recommended_max_working_set_size() / 2;
        Self {
            device,
            queue,
            state: Mutex::new(State {
                next_seq: 1,
                ..Default::default()
            }),
            pool: Mutex::new(Pool::default()),
            open_seq: AtomicU64::new(0),
            completed_seq: AtomicU64::new(0),
            debug: AtomicBool::new(debug),
            batch_limit: AtomicUsize::new(batch_limit),
            max_tg_mem,
            max_buffer_len,
            pool_cap,
        }
    }

    pub(crate) fn set_debug(&self, on: bool) {
        self.debug.store(on, Ordering::Relaxed);
    }

    pub(crate) fn debug(&self) -> bool {
        self.debug.load(Ordering::Relaxed)
    }

    pub(crate) fn set_batch_limit(&self, n: usize) {
        self.batch_limit.store(n.max(1), Ordering::Relaxed);
    }

    /// Whether some encoded or committed work has not been observed complete.
    pub(crate) fn unsynced(&self) -> bool {
        self.open_seq.load(Ordering::Acquire) != self.completed_seq.load(Ordering::Acquire)
    }

    fn open_batch<'a>(&self, st: &'a mut State) -> &'a mut OpenBatch {
        if st.open.is_none() {
            let seq = st.next_seq;
            st.next_seq += 1;
            let cmd = autoreleasepool(|| self.queue.new_command_buffer().to_owned());
            self.open_seq.store(seq, Ordering::Release);
            st.open = Some(OpenBatch {
                seq,
                cmd,
                encoder: Encoder::None,
                bound: Bound::default(),
                dispatches: 0,
                used: HashSet::new(),
            });
        }
        st.open.as_mut().unwrap()
    }

    /// Commit the open batch (no wait).
    fn close(&self, st: &mut State) {
        if let Some(mut open) = st.open.take() {
            open.encoder.end();
            open.cmd.commit();
            st.in_flight.push_back(InFlight {
                seq: open.seq,
                cmd: open.cmd,
                used: open.used,
            });
        }
    }

    /// Retire completed batches in commit order and advance `completed_seq`.
    fn poll(&self, st: &mut State) {
        while let Some(front) = st.in_flight.front() {
            match front.cmd.status() {
                MTLCommandBufferStatus::Completed => {}
                MTLCommandBufferStatus::Error => {
                    let msg = command_buffer_error(&front.cmd);
                    tracing::error!("Metal command buffer {} failed: {msg}", front.seq);
                    st.error
                        .get_or_insert_with(|| format!("Metal command buffer failed: {msg}"));
                }
                _ => break,
            }
            let done = st.in_flight.pop_front().unwrap();
            self.completed_seq.store(done.seq, Ordering::Release);
        }
        if st.open.is_none() && st.in_flight.is_empty() {
            self.completed_seq
                .store(self.open_seq.load(Ordering::Acquire), Ordering::Release);
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, State> {
        self.state.lock()
    }

    /// Run `f` with the open batch's compute encoder; counts as one dispatch.
    pub(crate) fn with_compute<R>(
        &self,
        f: impl FnOnce(&ComputeCommandEncoderRef, &mut Bound, Option<&mut HashSet<usize>>) -> R,
    ) -> R {
        let _lock = exclusive();
        let mut st = self.lock_state();
        let debug = self.debug();
        let open = self.open_batch(&mut st);
        if !matches!(open.encoder, Encoder::Compute(_)) {
            open.encoder.end();
            let enc = autoreleasepool(|| open.cmd.new_compute_command_encoder().to_owned());
            open.encoder = Encoder::Compute(enc);
            open.bound = Bound::default();
        }
        let Encoder::Compute(enc) = &open.encoder else {
            unreachable!()
        };
        let r = f(enc, &mut open.bound, debug.then_some(&mut open.used));
        open.dispatches += 1;
        if open.dispatches >= self.batch_limit.load(Ordering::Relaxed) {
            self.close(&mut st);
            self.poll(&mut st);
        }
        r
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch(
        &self,
        pso: &ComputePipelineStateRef,
        args: &[KernelArg],
        grid: Grid,
        tg_bytes: &[u64],
        pre: Option<&dyn Fn(&ComputeCommandEncoderRef)>,
    ) {
        let (threads_per_grid, threads_per_group, groups) = match grid {
            Grid::Threads(n) => {
                let w = pso.max_total_threads_per_threadgroup().min(256);
                (MTLSize::new(n, 1, 1), MTLSize::new(w, 1, 1), false)
            }
            Grid::Threads2D(x, y) => {
                let max = pso.max_total_threads_per_threadgroup().min(256);
                let w = pso.thread_execution_width().min(max).max(1);
                (
                    MTLSize::new(x, y, 1),
                    MTLSize::new(w, (max / w).max(1), 1),
                    false,
                )
            }
            Grid::Groups { groups, threads } => (groups, threads, true),
        };
        let total = threads_per_grid.width * threads_per_grid.height * threads_per_grid.depth;
        let group_threads =
            threads_per_group.width * threads_per_group.height * threads_per_group.depth;
        if total == 0 || group_threads == 0 {
            return;
        }
        assert!(
            group_threads <= pso.max_total_threads_per_threadgroup(),
            "threadgroup of {group_threads} threads exceeds the pipeline maximum {}",
            pso.max_total_threads_per_threadgroup()
        );
        let mut tg_total = pso.static_threadgroup_memory_length();
        for &bytes in tg_bytes {
            tg_total += bytes.next_multiple_of(16);
        }
        assert!(
            tg_total <= self.max_tg_mem,
            "threadgroup memory {tg_total} bytes exceeds the device limit {}",
            self.max_tg_mem
        );
        for arg in args {
            if let KernelArg::Bytes(bytes) = arg {
                assert!(
                    !bytes.is_empty() && bytes.len() <= MAX_SET_BYTES,
                    "KernelArg::Bytes of {} bytes (1..={MAX_SET_BYTES} allowed)",
                    bytes.len()
                );
            }
        }
        self.with_compute(|enc, bound, used| {
            if let Some(pre) = pre {
                pre(enc);
                // The callback may have changed any state.
                *bound = Bound::default();
            }
            let pso_key = pso.as_ptr() as usize;
            if bound.pso != pso_key {
                enc.set_compute_pipeline_state(pso);
                bound.pso = pso_key;
            }
            let mut used = used;
            for (index, arg) in args.iter().enumerate() {
                let slot = index as u64;
                match arg {
                    KernelArg::Null => {
                        if bound.set_buffer(index, Some((0, 0))) {
                            enc.set_buffer(slot, None, 0);
                        }
                    }
                    KernelArg::Buffer { buffer, offset } => {
                        let key = buffer.as_ptr() as usize;
                        if bound.set_buffer(index, Some((key, *offset))) {
                            enc.set_buffer(slot, Some(buffer), *offset);
                        }
                        if let Some(used) = used.as_deref_mut() {
                            used.insert(key);
                        }
                    }
                    KernelArg::Integer(value) => {
                        bound.set_buffer(index, None);
                        enc.set_bytes(
                            slot,
                            mem::size_of::<u32>() as u64,
                            value as *const u32 as *const c_void,
                        )
                    }
                    KernelArg::Bytes(bytes) => {
                        bound.set_buffer(index, None);
                        enc.set_bytes(slot, bytes.len() as u64, bytes.as_ptr() as *const c_void)
                    }
                }
            }
            for (index, &bytes) in tg_bytes.iter().enumerate() {
                enc.set_threadgroup_memory_length(index as u64, bytes.next_multiple_of(16));
            }
            if groups {
                enc.dispatch_thread_groups(threads_per_grid, threads_per_group);
            } else {
                enc.dispatch_threads(threads_per_grid, threads_per_group);
            }
        });
    }

    pub(crate) fn blit_fill(&self, buf: &BufferRef, offset: u64, len: u64, byte: u8) {
        if len == 0 {
            return;
        }
        assert!(
            offset
                .checked_add(len)
                .is_some_and(|end| end <= buf.length()),
            "blit_fill range {offset}+{len} outside a buffer of {} bytes",
            buf.length()
        );
        let _lock = exclusive();
        let mut st = self.lock_state();
        let debug = self.debug();
        let open = self.open_batch(&mut st);
        if !matches!(open.encoder, Encoder::Blit(_)) {
            open.encoder.end();
            let enc = autoreleasepool(|| open.cmd.new_blit_command_encoder().to_owned());
            open.encoder = Encoder::Blit(enc);
        }
        let Encoder::Blit(enc) = &open.encoder else {
            unreachable!()
        };
        enc.fill_buffer(buf, NSRange::new(offset, len), byte);
        if debug {
            open.used.insert(buf.as_ptr() as usize);
        }
        open.dispatches += 1;
        if open.dispatches >= self.batch_limit.load(Ordering::Relaxed) {
            self.close(&mut st);
            self.poll(&mut st);
        }
    }

    /// End encoding and commit the open batch without waiting.
    pub(crate) fn flush(&self) {
        let _lock = exclusive();
        let mut st = self.lock_state();
        self.close(&mut st);
        self.poll(&mut st);
    }

    /// Flush and wait for every committed batch; the first command-buffer error since the last
    /// call is returned (plan 2.9).
    pub(crate) fn try_sync(&self) -> Result<()> {
        let _lock = exclusive();
        let mut st = self.lock_state();
        self.close(&mut st);
        for batch in st.in_flight.iter() {
            batch.cmd.wait_until_completed();
        }
        self.poll(&mut st);
        debug_assert!(st.in_flight.is_empty());
        match st.error.take() {
            Some(err) => Err(anyhow!(err)),
            None => Ok(()),
        }
    }

    pub(crate) fn sync(&self) {
        if let Err(err) = self.try_sync() {
            panic!("{err:#}");
        }
    }

    /// Called by every host access to a HAL buffer: waits for outstanding GPU work, then (in
    /// debug mode) checks that no unsynced batch uses the buffer.
    pub(crate) fn sync_for_host(&self, buf: &BufferRef, name: &str, what: &str) {
        if self.unsynced() {
            self.sync();
        }
        self.check_host_access(buf, name, what);
    }

    /// `RISC0_METAL_DEBUG=1`: panic if `buf` is bound in a batch that has not been observed
    /// complete, i.e. the host is about to touch memory the GPU may still read or write.
    pub(crate) fn check_host_access(&self, buf: &BufferRef, name: &str, what: &str) {
        if !self.debug() || !self.unsynced() {
            return;
        }
        // try_lock: the caller may be inside `with_encoder` on this thread.
        let Some(mut st) = self.state.try_lock() else {
            return;
        };
        self.poll(&mut st);
        let key = buf.as_ptr() as usize;
        let in_open = st.open.as_ref().is_some_and(|b| b.used.contains(&key));
        let in_flight = st.in_flight.iter().any(|b| b.used.contains(&key));
        drop(st);
        assert!(
            !in_open && !in_flight,
            "RISC0_METAL_DEBUG: host {what} of buffer '{name}' while a GPU batch that uses it is \
             not synced ({}); call sync() first",
            if in_open {
                "batch still open"
            } else {
                "batch committed, not complete"
            }
        );
    }

    fn new_buffer_raw(&self, bytes: u64, options: MTLResourceOptions) -> Option<MtlBuffer> {
        let buf = self.device.new_buffer(bytes, options);
        if buf.as_ptr().is_null() {
            mem::forget(buf);
            None
        } else {
            Some(buf)
        }
    }

    /// A shared buffer of `bytes` (> 0) bytes: from the pool when one of that size has been
    /// returned and its last batch completed, else a new one. On allocation failure the pool is
    /// drained and the allocation retried once (plan 2.9).
    pub(crate) fn alloc_shared(&self, name: &str, bytes: u64) -> Result<MtlBuffer> {
        assert!(
            bytes <= self.max_buffer_len,
            "allocation of {bytes} bytes for {name} exceeds max_buffer_length {}",
            self.max_buffer_len
        );
        if self.unsynced() {
            let _lock = exclusive();
            let mut st = self.lock_state();
            self.poll(&mut st);
        }
        let completed = self.completed_seq.load(Ordering::Acquire);
        {
            let mut pool = self.pool.lock();
            pool.reclaim(completed);
            if let Some(buf) = pool.pop(bytes) {
                return Ok(buf);
            }
        }
        let options = MTLResourceOptions::StorageModeShared;
        if let Some(buf) = self.new_buffer_raw(bytes, options) {
            return Ok(buf);
        }
        tracing::warn!("Metal allocation of {bytes} bytes for {name} failed; draining the pool");
        self.drain_pool();
        self.new_buffer_raw(bytes, options)
            .ok_or_else(|| anyhow!("Metal allocation failed on {name}: {bytes} bytes"))
    }

    pub(crate) fn alloc_private(&self, bytes: u64) -> Result<MtlBuffer> {
        assert!(
            bytes <= self.max_buffer_len,
            "private allocation of {bytes} bytes exceeds max_buffer_length {}",
            self.max_buffer_len
        );
        let options = MTLResourceOptions::StorageModePrivate;
        let bytes = bytes.max(16);
        if let Some(buf) = self.new_buffer_raw(bytes, options) {
            return Ok(buf);
        }
        self.drain_pool();
        self.new_buffer_raw(bytes, options)
            .ok_or_else(|| anyhow!("Metal private allocation failed: {bytes} bytes"))
    }

    /// Return a dropped buffer. Never blocks on GPU work and never takes the state mutex.
    pub(crate) fn release(&self, bytes: u64, buf: MtlBuffer) {
        let needed = self.open_seq.load(Ordering::Acquire);
        let completed = self.completed_seq.load(Ordering::Acquire);
        let mut pool = self.pool.lock();
        if bytes >= POOL_SMALL_BYTES && pool.cached_bytes + bytes > self.pool_cap {
            return;
        }
        pool.cached_bytes += bytes;
        if needed <= completed {
            pool.ready.entry(bytes).or_default().push(buf);
        } else {
            pool.pending.push((needed, bytes, buf));
        }
    }

    /// Release every pooled buffer; returns the bytes released.
    pub(crate) fn drain_pool(&self) -> u64 {
        self.pool.lock().drain()
    }

    #[cfg(test)]
    pub(crate) fn pool_stats(&self) -> (usize, usize) {
        let pool = self.pool.lock();
        (pool.ready.values().map(Vec::len).sum(), pool.pending.len())
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // Commit what is still encoded so no encoder is released without endEncoding.
        let st = self.state.get_mut();
        if let Some(mut open) = st.open.take() {
            open.encoder.end();
            open.cmd.commit();
            open.cmd.wait_until_completed();
        }
        for batch in st.in_flight.drain(..) {
            batch.cmd.wait_until_completed();
        }
    }
}
