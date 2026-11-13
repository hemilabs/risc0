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
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Weak,
    },
};

use anyhow::{anyhow, Result};
use metal::{
    foreign_types::{ForeignType, ForeignTypeRef},
    objc::{msg_send, rc::autoreleasepool, runtime::Object, sel, sel_impl},
    BlitCommandEncoder, Buffer as MtlBuffer, BufferRef, CommandBuffer, CommandQueue,
    ComputeCommandEncoder, ComputeCommandEncoderRef, ComputePipelineStateRef, Device, MTLBuffer,
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

/// Threads per threadgroup of `Grid::Threads` dispatches (capped by the PSO): 256, CUDA's
/// elementwise block size (plan 2.5). `RISC0_METAL_THREADS_1D` overrides it (Z9 sweep); tests use
/// [set_threads_1d].
static THREADS_1D: AtomicU64 = AtomicU64::new(0);

pub(crate) fn threads_1d() -> u64 {
    let v = THREADS_1D.load(Ordering::Relaxed);
    if v != 0 {
        return v;
    }
    let v = std::env::var("RISC0_METAL_THREADS_1D")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| (1..=1024).contains(&n))
        .unwrap_or(256);
    THREADS_1D.store(v, Ordering::Relaxed);
    v
}

#[cfg(test)]
pub(crate) fn set_threads_1d(n: u64) {
    THREADS_1D.store(n, Ordering::Relaxed);
}

/// Z9 per-kernel GPU-time profiler (measurement tool, off by default). With
/// `RISC0_METAL_PROFILE=1` every dispatch and blit is committed in a command buffer of its own and
/// its `GPUEndTime - GPUStartTime` is added to a process-wide table keyed by kernel name (names
/// come from [register_pso_name], called for every PSO [super::MetalKernels] builds). The table is
/// printed to stderr when a runtime is dropped, and [profile_take] returns and clears it. One
/// command buffer per dispatch serializes the GPU work, so wall time grows; the per-kernel GPU
/// times stay comparable (they include a few microseconds of per-command-buffer overhead).
pub const PROFILE_ENV: &str = "RISC0_METAL_PROFILE";

fn profiling() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_flag(PROFILE_ENV))
}

static PSO_NAMES: Mutex<Option<HashMap<usize, &'static str>>> = Mutex::new(None);

/// Record the kernel name of a PSO for the profiler (cheap; names are leaked once per PSO).
/// Public (C15) so circuit HALs that build their own PSOs (recursion `Pipelines`) are named in
/// the profile too; a no-op unless [PROFILE_ENV] is set.
pub fn register_pso_name(pso: &ComputePipelineStateRef, name: &str) {
    if !profiling() {
        return;
    }
    let mut names = PSO_NAMES.lock();
    names.get_or_insert_with(HashMap::new).insert(
        pso.as_ptr() as usize,
        Box::leak(name.to_string().into_boxed_str()),
    );
}

fn pso_name(key: usize) -> &'static str {
    PSO_NAMES
        .lock()
        .as_ref()
        .and_then(|m| m.get(&key).copied())
        .unwrap_or("(unnamed pso)")
}

/// Per-kernel profile entry: dispatch count, summed GPU seconds and the fastest dispatch (the
/// least disturbed by other processes' GPU work; 0 until the first sample).
#[derive(Clone, Copy, Debug, Default)]
pub struct ProfileEntry {
    pub count: u64,
    pub gpu_secs: f64,
    pub min_secs: f64,
}

static PROFILE: Mutex<Option<HashMap<&'static str, ProfileEntry>>> = Mutex::new(None);

#[allow(unexpected_cfgs)]
fn record_profile(label: &'static str, cmd: &CommandBuffer) {
    // SAFETY: plain Objective-C property reads on a live, completed command buffer.
    let (start, end): (f64, f64) = unsafe {
        let p = cmd.as_ptr() as *mut Object;
        (msg_send![p, GPUStartTime], msg_send![p, GPUEndTime])
    };
    let mut prof = PROFILE.lock();
    let e = prof
        .get_or_insert_with(HashMap::new)
        .entry(label)
        .or_default();
    let dt = (end - start).max(0.0);
    e.min_secs = if e.count == 0 { dt } else { e.min_secs.min(dt) };
    e.count += 1;
    e.gpu_secs += dt;
}

/// Take (and clear) the per-kernel profile, sorted by GPU time, descending.
pub fn profile_take() -> Vec<(&'static str, ProfileEntry)> {
    let mut v: Vec<_> = PROFILE
        .lock()
        .take()
        .unwrap_or_default()
        .into_iter()
        .collect();
    v.sort_by(|a, b| b.1.gpu_secs.total_cmp(&a.1.gpu_secs));
    v
}

/// Format [profile_take]'s result as a table.
pub fn profile_report() -> String {
    let v = profile_take();
    let total: f64 = v.iter().map(|(_, e)| e.gpu_secs).sum();
    let mut out = format!("[metal-profile] total GPU {:.3} ms\n", total * 1e3);
    for (name, e) in v {
        out += &format!(
            "[metal-profile] {name:32} {:8} calls {:10.3} ms {:6.2}% {:9.1} us/call {:9.1} us min\n",
            e.count,
            e.gpu_secs * 1e3,
            100.0 * e.gpu_secs / total.max(1e-12),
            1e6 * e.gpu_secs / e.count.max(1) as f64,
            1e6 * e.min_secs
        );
    }
    out
}

/// Environment variable naming an artificial GPU working-set budget in bytes (plan 9.1), e.g.
/// `4294967296`, `4G`, `4GiB`, `512M` (suffixes are binary: K = 1024). When set, it replaces
/// `recommendedMaxWorkingSetSize` as the working-set limit (if smaller): `gpu_total_memory` and
/// `gpu_free_memory` report against it, the pool cap is half of it, and a new allocation that
/// would bring the device's `currentAllocatedSize` (or, with [BUDGET_SCOPE_ENV]`=hal`, this HAL's
/// own allocations) above it fails (after draining every HAL's pool once; with `hal` scope, this
/// HAL's pool) with an error naming the budget, instead of running into the OS memory pressure
/// path.
pub const MAX_WORKING_SET_ENV: &str = "RISC0_METAL_MAX_WORKING_SET_BYTES";

/// Environment variable choosing what an explicit working-set budget is checked against:
/// `device` (default): the device's `currentAllocatedSize`, i.e. every Metal allocation of this
/// process (all HALs, pipelines, libraries) -- the faithful model of a smaller GPU for a whole
/// prover process; `hal`: only the bytes this HAL allocated (live and pooled shared buffers, plus
/// its private buffers, which are counted until the HAL is dropped). `hal` makes budget runs of a
/// test suite independent of the other tests running in parallel (plan 8.7, Z8b).
pub const BUDGET_SCOPE_ENV: &str = "RISC0_METAL_BUDGET_SCOPE";

/// What an explicit working-set budget is checked against (see [BUDGET_SCOPE_ENV]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetScope {
    /// The device's `currentAllocatedSize` (default).
    Device,
    /// The bytes this HAL allocated.
    Hal,
}

fn env_budget_scope() -> BudgetScope {
    match std::env::var(BUDGET_SCOPE_ENV) {
        Ok(v) if v.trim().eq_ignore_ascii_case("hal") => BudgetScope::Hal,
        Ok(v) if v.trim().eq_ignore_ascii_case("device") || v.trim().is_empty() => {
            BudgetScope::Device
        }
        Ok(v) => {
            tracing::error!("{BUDGET_SCOPE_ENV}={v:?} is not `device` or `hal`; using `device`");
            BudgetScope::Device
        }
        Err(_) => BudgetScope::Device,
    }
}

/// Highest `currentAllocatedSize` observed after any Metal allocation in this process.
static PEAK_ALLOCATED: AtomicU64 = AtomicU64::new(0);

/// Every live runtime's pool. The working-set budget and the allocation-failure retry look at
/// the device's `currentAllocatedSize`, which counts the buffers idling in *every* HAL's pool, so
/// they drain all pools, not just the allocating HAL's (Z8b; CUDA's OOM hook drains its single
/// process-wide pool the same way).
static POOLS: parking_lot::Mutex<Vec<Weak<Mutex<Pool>>>> = parking_lot::Mutex::new(Vec::new());

/// Drain every live runtime's pool; returns the bytes released. Takes each pool mutex in turn
/// (never two at once, never the registry lock while holding a pool), so it cannot deadlock with
/// [Runtime::release] or another drain.
fn drain_all_pools() -> u64 {
    let pools: Vec<Arc<Mutex<Pool>>> = {
        let mut reg = POOLS.lock();
        reg.retain(|w| w.strong_count() > 0);
        reg.iter().filter_map(Weak::upgrade).collect()
    };
    pools.iter().map(|p| p.lock().drain()).sum()
}

/// Largest grid extent per dimension: every zkp and circuit kernel indexes with a 32-bit
/// `thread_position_in_grid` (`uint`), which silently wraps beyond this (Metal accepts the
/// dispatch; Z8b measured it running a 2^32-thread grid without an error).
const MAX_GRID_DIM: u64 = u32::MAX as u64;

/// Parse a byte count with an optional binary suffix (`K`, `M`, `G`, `T`, optionally followed by
/// `iB` or `B`); `None` if malformed or zero.
pub(crate) fn parse_bytes(text: &str) -> Option<u64> {
    let t = text.trim();
    let t = t
        .strip_suffix("iB")
        .or_else(|| t.strip_suffix("ib"))
        .or_else(|| t.strip_suffix('B'))
        .or_else(|| t.strip_suffix('b'))
        .unwrap_or(t);
    let (digits, shift) = match t.chars().last()? {
        'k' | 'K' => (&t[..t.len() - 1], 10),
        'm' | 'M' => (&t[..t.len() - 1], 20),
        'g' | 'G' => (&t[..t.len() - 1], 30),
        't' | 'T' => (&t[..t.len() - 1], 40),
        _ => (t, 0),
    };
    let n: u64 = digits.trim().parse().ok()?;
    n.checked_mul(1u64 << shift).filter(|&b| b > 0)
}

/// The budget named by [MAX_WORKING_SET_ENV], if set and valid (a malformed value is reported and
/// ignored).
pub(crate) fn env_working_set_limit() -> Option<u64> {
    let text = std::env::var(MAX_WORKING_SET_ENV).ok()?;
    let parsed = parse_bytes(&text);
    if parsed.is_none() {
        tracing::error!("{MAX_WORKING_SET_ENV}={text:?} is not a byte count; ignored");
    }
    parsed
}

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
///
/// Keys are Objective-C object addresses. That is sound only because command buffers come from
/// `new_command_buffer` and so retain every bound resource: a bound object cannot be freed and
/// its address reused within the encoder. Switching to `commandBufferWithUnretainedReferences`
/// would break this (a reused address could silently skip a `set_buffer`).
#[derive(Default)]
pub(crate) struct Bound {
    pso: usize,
    buffers: Vec<Option<(usize, u64)>>,
    /// Threadgroup memory lengths set in this encoder, per `[[threadgroup(i)]]` index.
    tg_lengths: Vec<Option<u64>>,
}

impl Bound {
    /// Record threadgroup memory length `bytes` at `index`; true if it differs from what this
    /// encoder already has (so `setThreadgroupMemoryLength` must be issued).
    pub(crate) fn set_tg_length(&mut self, index: usize, bytes: u64) -> bool {
        if self.tg_lengths.len() <= index {
            self.tg_lengths.resize(index + 1, None);
        }
        let changed = self.tg_lengths[index] != Some(bytes);
        self.tg_lengths[index] = Some(bytes);
        changed
    }

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
    /// Profiler label of the batch's only dispatch ([PROFILE_ENV]).
    label: Option<&'static str>,
}

struct InFlight {
    seq: u64,
    cmd: CommandBuffer,
    used: HashSet<usize>,
    label: Option<&'static str>,
    /// Test hook: report this batch as `Error` once it completes.
    #[cfg(test)]
    injected_error: bool,
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
    /// Bytes this runtime allocated and has not freed: live and pooled shared buffers, plus
    /// private buffers (never subtracted; their drop is not observable). The [BudgetScope::Hal]
    /// budget is checked against it.
    owned_bytes: u64,
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
        self.owned_bytes = self.owned_bytes.saturating_sub(bytes);
        bytes
    }
}

/// State shared by a `MetalHal` and every buffer it allocated.
pub(crate) struct Runtime {
    pub(crate) device: Device,
    pub(crate) queue: CommandQueue,
    state: Mutex<State>,
    pool: Arc<Mutex<Pool>>,
    /// Sequence number of the most recently opened batch (0: none yet).
    open_seq: AtomicU64,
    /// Every batch with a sequence number up to this one has completed.
    completed_seq: AtomicU64,
    debug: AtomicBool,
    batch_limit: AtomicUsize,
    pub(crate) max_tg_mem: u64,
    pub(crate) max_buffer_len: u64,
    /// `recommendedMaxWorkingSetSize` of the device.
    recommended_ws: u64,
    /// Explicit working-set budget ([MAX_WORKING_SET_ENV] or [Runtime::set_working_set_limit]);
    /// 0 = none (the limit is then `recommended_ws` and is not enforced on allocation).
    budget: AtomicU64,
    /// [BudgetScope::Hal] if true.
    hal_scope: AtomicBool,
    pool_cap: AtomicU64,
    /// Test hook: number of upcoming raw allocations to fail.
    #[cfg(test)]
    fail_allocs: AtomicUsize,
    /// Test hook: report the next committed batch as failed.
    #[cfg(test)]
    fail_next_batch: AtomicBool,
    /// Test hook: zero buffers handed out from the pool, as new allocations are.
    #[cfg(test)]
    zero_pooled: AtomicBool,
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
        let recommended_ws = device.recommended_max_working_set_size();
        let budget = env_working_set_limit().unwrap_or(0);
        let limit = if budget == 0 {
            recommended_ws
        } else {
            budget.min(recommended_ws)
        };
        let pool_cap = limit / 2;
        let pool = Arc::new(Mutex::new(Pool::default()));
        {
            let mut reg = POOLS.lock();
            reg.retain(|w| w.strong_count() > 0);
            reg.push(Arc::downgrade(&pool));
        }
        Self {
            device,
            queue,
            state: Mutex::new(State {
                next_seq: 1,
                ..Default::default()
            }),
            pool,
            open_seq: AtomicU64::new(0),
            completed_seq: AtomicU64::new(0),
            debug: AtomicBool::new(debug),
            batch_limit: AtomicUsize::new(batch_limit),
            max_tg_mem,
            max_buffer_len,
            recommended_ws,
            budget: AtomicU64::new(budget),
            hal_scope: AtomicBool::new(env_budget_scope() == BudgetScope::Hal),
            pool_cap: AtomicU64::new(pool_cap),
            #[cfg(test)]
            fail_allocs: AtomicUsize::new(0),
            #[cfg(test)]
            fail_next_batch: AtomicBool::new(false),
            #[cfg(test)]
            zero_pooled: AtomicBool::new(false),
        }
    }

    pub(crate) fn set_debug(&self, on: bool) {
        self.debug.store(on, Ordering::Relaxed);
    }

    pub(crate) fn debug(&self) -> bool {
        self.debug.load(Ordering::Relaxed)
    }

    /// The explicit working-set budget, if one is set.
    pub(crate) fn budget(&self) -> Option<u64> {
        match self.budget.load(Ordering::Relaxed) {
            0 => None,
            b => Some(b),
        }
    }

    /// The working-set limit: the explicit budget if set (capped at
    /// `recommendedMaxWorkingSetSize`), else `recommendedMaxWorkingSetSize`.
    pub(crate) fn working_set_limit(&self) -> u64 {
        match self.budget() {
            Some(b) => b.min(self.recommended_ws),
            None => self.recommended_ws,
        }
    }

    /// Set (`Some`) or clear (`None`) the explicit working-set budget of this runtime, overriding
    /// [MAX_WORKING_SET_ENV]. The pool cap follows (half the limit).
    pub(crate) fn set_working_set_limit(&self, bytes: Option<u64>) {
        self.budget
            .store(bytes.map_or(0, |b| b.max(1)), Ordering::Relaxed);
        self.pool_cap
            .store(self.working_set_limit() / 2, Ordering::Relaxed);
    }

    /// What the explicit budget is checked against.
    pub(crate) fn budget_scope(&self) -> BudgetScope {
        if self.hal_scope.load(Ordering::Relaxed) {
            BudgetScope::Hal
        } else {
            BudgetScope::Device
        }
    }

    /// Override [BUDGET_SCOPE_ENV] for this runtime.
    pub(crate) fn set_budget_scope(&self, scope: BudgetScope) {
        self.hal_scope
            .store(scope == BudgetScope::Hal, Ordering::Relaxed);
    }

    /// Bytes this runtime allocated and has not freed (see [Pool::owned_bytes]).
    pub(crate) fn owned_bytes(&self) -> u64 {
        self.pool.lock().owned_bytes
    }

    /// The allocation level the budget is checked against.
    fn budget_level(&self) -> u64 {
        match self.budget_scope() {
            BudgetScope::Device => self.allocated_size(),
            BudgetScope::Hal => self.owned_bytes(),
        }
    }

    /// `currentAllocatedSize` of the device (every Metal allocation of this process on it,
    /// including pooled buffers and pipeline/library memory).
    pub(crate) fn allocated_size(&self) -> u64 {
        self.device.current_allocated_size()
    }

    /// Bytes left under the working-set limit (against the budget scope's allocation level when
    /// a budget is set, else against `currentAllocatedSize`).
    pub(crate) fn free_size(&self) -> u64 {
        let level = match self.budget() {
            Some(_) => self.budget_level(),
            None => self.allocated_size(),
        };
        self.working_set_limit().saturating_sub(level)
    }

    /// Highest `currentAllocatedSize` seen after any allocation in this process since the last
    /// [reset_peak_allocated].
    pub(crate) fn peak_allocated() -> u64 {
        PEAK_ALLOCATED.load(Ordering::Relaxed)
    }

    /// Restart [Self::peak_allocated] from the current allocation level.
    pub(crate) fn reset_peak_allocated(&self) {
        PEAK_ALLOCATED.store(self.allocated_size(), Ordering::Relaxed);
    }

    fn record_peak(&self) {
        PEAK_ALLOCATED.fetch_max(self.allocated_size(), Ordering::Relaxed);
    }

    /// With an explicit budget: fail early if a new allocation of `bytes` would exceed it. Every
    /// HAL's pool (all of them count toward the device-wide `currentAllocatedSize`) is drained
    /// once before giving up.
    fn check_budget(&self, what: &str, bytes: u64) -> Result<()> {
        let Some(budget) = self.budget() else {
            return Ok(());
        };
        let limit = budget.min(self.recommended_ws);
        if self.budget_level().saturating_add(bytes) <= limit {
            return Ok(());
        }
        let scope = self.budget_scope();
        let drained = match scope {
            BudgetScope::Device => drain_all_pools(),
            BudgetScope::Hal => self.drain_pool(),
        };
        let current = self.budget_level();
        if current.saturating_add(bytes) <= limit {
            tracing::debug!("working-set budget: drained {drained} pooled bytes for {what}");
            return Ok(());
        }
        let level = match scope {
            BudgetScope::Device => "already allocated on the device",
            BudgetScope::Hal => "already allocated by this HAL",
        };
        Err(anyhow!(
            "Metal working-set budget exceeded: allocating {bytes} bytes for {what} with \
             {current} bytes {level} would exceed the limit of {limit} bytes \
             ({MAX_WORKING_SET_ENV}={budget}, {BUDGET_SCOPE_ENV}={}, \
             recommendedMaxWorkingSetSize={})",
            match scope {
                BudgetScope::Device => "device",
                BudgetScope::Hal => "hal",
            },
            self.recommended_ws
        ))
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
                label: None,
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
                label: open.label,
                #[cfg(test)]
                injected_error: self.fail_next_batch.swap(false, Ordering::AcqRel),
            });
        }
    }

    /// Retire completed batches in commit order and advance `completed_seq`.
    fn poll(&self, st: &mut State) {
        while let Some(front) = st.in_flight.front() {
            #[allow(unused_mut)]
            let mut status = front.cmd.status();
            #[cfg(test)]
            if front.injected_error && status == MTLCommandBufferStatus::Completed {
                status = MTLCommandBufferStatus::Error;
            }
            match status {
                MTLCommandBufferStatus::Completed => {
                    if let Some(label) = front.label {
                        record_profile(label, &front.cmd);
                    }
                }
                MTLCommandBufferStatus::Error => {
                    #[allow(unused_mut)]
                    let mut msg = command_buffer_error(&front.cmd);
                    #[cfg(test)]
                    if front.injected_error {
                        msg = "injected test error".into();
                    }
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
        self.with_compute_labeled(None, f)
    }

    fn with_compute_labeled<R>(
        &self,
        label: Option<&'static str>,
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
        let prof = profiling();
        if prof {
            open.label = Some(label.unwrap_or("(with_encoder)"));
        }
        if prof || open.dispatches >= self.batch_limit.load(Ordering::Relaxed) {
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
                let w = pso.max_total_threads_per_threadgroup().min(threads_1d());
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
        // Thread positions per dimension (groups x threads for dispatch_thread_groups).
        let extent = |grid: u64, group: u64| {
            if groups {
                grid.checked_mul(group)
            } else {
                Some(grid)
            }
        };
        for (dim, (g, t)) in [
            (threads_per_grid.width, threads_per_group.width),
            (threads_per_grid.height, threads_per_group.height),
            (threads_per_grid.depth, threads_per_group.depth),
        ]
        .into_iter()
        .enumerate()
        {
            let n = extent(g, t);
            assert!(
                n.is_some_and(|n| n <= MAX_GRID_DIM),
                "grid dimension {dim} of {g} x {t} threads exceeds the 32-bit thread index \
                 (thread_position_in_grid is uint): split the dispatch"
            );
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
        let label = profiling().then(|| pso_name(pso.as_ptr() as usize));
        self.with_compute_labeled(label, |enc, bound, used| {
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
            // The length persists in the encoder across dispatches; re-issuing an unchanged value
            // is the validation layer's "redundant setting of (null)" (Z2 note).
            for (index, &bytes) in tg_bytes.iter().enumerate() {
                let bytes = bytes.next_multiple_of(16);
                if bound.set_tg_length(index, bytes) {
                    enc.set_threadgroup_memory_length(index as u64, bytes);
                }
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
        let prof = profiling();
        if prof {
            open.label = Some("(blit_fill)");
        }
        if prof || open.dispatches >= self.batch_limit.load(Ordering::Relaxed) {
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

    /// `newBufferWithLength:options:`, returning `None` on nil. metal-rs's `Device::new_buffer`
    /// returns a `NonNull`-backed `Buffer`, so a nil result there is already undefined behaviour
    /// and a later null check is folded away by the optimiser; the raw pointer is checked here
    /// before it is wrapped.
    // objc's `msg_send!` expands a `cfg(feature = "cargo-clippy")` this crate does not declare.
    #[allow(unexpected_cfgs)]
    fn new_buffer_raw(&self, bytes: u64, options: MTLResourceOptions) -> Option<MtlBuffer> {
        #[cfg(test)]
        if self
            .fail_allocs
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
            .is_ok()
        {
            return None;
        }
        // SAFETY: `newBufferWithLength:options:` returns a +1 (owned) reference or nil; only a
        // non-nil result is wrapped, and `from_ptr` takes over that reference.
        let ptr: *mut MTLBuffer = unsafe {
            msg_send![self.device.as_ptr() as *mut Object, newBufferWithLength: bytes options: options]
        };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { MtlBuffer::from_ptr(ptr) })
        }
    }

    /// Test hook: the next `n` raw allocations fail as if `newBufferWithLength:` returned nil.
    #[cfg(test)]
    pub(crate) fn inject_alloc_failures(&self, n: usize) {
        self.fail_allocs.store(n, Ordering::Release);
    }

    /// Test hook: buffers reused from the pool come back zeroed, like new Metal buffers (and like
    /// the CPU HAL's). `DualHal` compares whole buffers, including cells a method never writes
    /// (e.g. the unused head of a Merkle `nodes` buffer), so it needs this for full-prover runs.
    #[cfg(test)]
    pub(crate) fn set_zero_pooled(&self, on: bool) {
        self.zero_pooled.store(on, Ordering::Relaxed);
    }

    /// Test hook: the next committed batch is reported as `MTLCommandBufferStatus::Error`.
    #[cfg(test)]
    pub(crate) fn inject_batch_error(&self) {
        self.fail_next_batch.store(true, Ordering::Release);
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
                #[cfg(test)]
                if self.zero_pooled.load(Ordering::Relaxed) {
                    // SAFETY: every batch that used this buffer has completed; `bytes` is its
                    // length.
                    unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes as usize) };
                }
                return Ok(buf);
            }
        }
        self.check_budget(name, bytes)?;
        let options = MTLResourceOptions::StorageModeShared;
        if let Some(buf) = self.new_buffer_raw(bytes, options) {
            self.allocated(bytes);
            return Ok(buf);
        }
        tracing::warn!("Metal allocation of {bytes} bytes for {name} failed; draining the pools");
        drain_all_pools();
        let buf = self
            .new_buffer_raw(bytes, options)
            .ok_or_else(|| anyhow!("Metal allocation failed on {name}: {bytes} bytes"))?;
        self.allocated(bytes);
        Ok(buf)
    }

    pub(crate) fn alloc_private(&self, bytes: u64) -> Result<MtlBuffer> {
        assert!(
            bytes <= self.max_buffer_len,
            "private allocation of {bytes} bytes exceeds max_buffer_length {}",
            self.max_buffer_len
        );
        let options = MTLResourceOptions::StorageModePrivate;
        let bytes = bytes.max(16);
        self.check_budget("a private buffer", bytes)?;
        if let Some(buf) = self.new_buffer_raw(bytes, options) {
            self.allocated(bytes);
            return Ok(buf);
        }
        drain_all_pools();
        let buf = self
            .new_buffer_raw(bytes, options)
            .ok_or_else(|| anyhow!("Metal private allocation failed: {bytes} bytes"))?;
        self.allocated(bytes);
        Ok(buf)
    }

    /// Book-keeping after a new allocation of `bytes`.
    fn allocated(&self, bytes: u64) {
        self.pool.lock().owned_bytes += bytes;
        self.record_peak();
    }

    /// A shared buffer of `bytes` this runtime allocated was freed without going to the pool
    /// (`set_bypass_pool`).
    pub(crate) fn freed(&self, bytes: u64) {
        let mut pool = self.pool.lock();
        pool.owned_bytes = pool.owned_bytes.saturating_sub(bytes);
    }

    /// Return a dropped buffer. Never blocks on GPU work and never takes the state mutex.
    pub(crate) fn release(&self, bytes: u64, buf: MtlBuffer) {
        let needed = self.open_seq.load(Ordering::Acquire);
        let completed = self.completed_seq.load(Ordering::Acquire);
        let mut pool = self.pool.lock();
        if bytes >= POOL_SMALL_BYTES
            && pool.cached_bytes + bytes > self.pool_cap.load(Ordering::Relaxed)
        {
            // Freed (the command buffers that use it keep their own reference).
            pool.owned_bytes = pool.owned_bytes.saturating_sub(bytes);
            return;
        }
        pool.cached_bytes += bytes;
        if needed <= completed {
            pool.ready.entry(bytes).or_default().push(buf);
        } else {
            pool.pending.push((needed, bytes, buf));
        }
    }

    /// Release every buffer pooled by this runtime (`trim_device_memory`); returns the bytes
    /// released. Budget and allocation failures use [drain_all_pools] instead.
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
            if let Some(label) = batch.label {
                record_profile(label, &batch.cmd);
            }
        }
        if profiling() {
            let report = profile_report();
            if report.lines().count() > 1 {
                eprint!("{report}");
            }
        }
    }
}
