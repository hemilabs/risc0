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

/// Z15 sync tracer (measurement tool, off by default). With `RISC0_METAL_SYNC_TRACE=1` every
/// [Runtime::try_sync] that had GPU work to wait for adds one record to a process-wide table: the
/// host wait, the wake latency (host time when the wait returned minus the last command buffer's
/// `GPUEndTime`) and the call site (the first `risc0` frame of a backtrace outside this module).
/// Each commit that finds the GPU idle after a sync also records the host gap since that sync
/// returned (GPU idle time spent on the host). [sync_trace_take] returns and clears the table;
/// `RISC0_METAL_SYNC_TRACE=2` also prints every record to stderr. Capturing backtraces costs
/// tens of microseconds per sync, so wall times under the tracer are not comparable; counts and
/// wake latencies are.
pub const SYNC_TRACE_ENV: &str = "RISC0_METAL_SYNC_TRACE";

fn sync_trace_level() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var(SYNC_TRACE_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .unwrap_or(0)
    })
}

/// One [SYNC_TRACE_ENV] record.
#[derive(Clone, Debug)]
pub struct SyncRecord {
    /// The first `risc0` frame outside the Metal runtime/buffer modules.
    pub site: String,
    /// Host time blocked in `waitUntilCompleted`, seconds.
    pub wait_secs: f64,
    /// Host time the wait returned minus the last batch's `GPUEndTime`, seconds.
    pub wake_secs: f64,
    /// Host gap between the previous sync's return and the next commit that found the GPU idle
    /// (filled in when that commit happens; 0 if none followed yet).
    pub idle_after_secs: f64,
}

static SYNC_TRACE: Mutex<Vec<SyncRecord>> = Mutex::new(Vec::new());
static LAST_SYNC_END: Mutex<Option<f64>> = Mutex::new(None);

/// Host time in seconds on the same clock as `GPUStartTime` / `GPUEndTime` (mach absolute time).
fn host_secs() -> f64 {
    #[repr(C)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }
    extern "C" {
        fn mach_absolute_time() -> u64;
        fn mach_timebase_info(info: *mut Timebase) -> i32;
    }
    static TB: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();
    let (n, d) = *TB.get_or_init(|| {
        let mut tb = Timebase { numer: 0, denom: 0 };
        // SAFETY: plain libSystem call filling a local struct.
        unsafe { mach_timebase_info(&mut tb) };
        (tb.numer.max(1), tb.denom.max(1))
    });
    // SAFETY: plain libSystem call.
    let t = unsafe { mach_absolute_time() };
    t as f64 * n as f64 / d as f64 * 1e-9
}

fn sync_site() -> String {
    let bt = std::backtrace::Backtrace::force_capture().to_string();
    let lines: Vec<&str> = bt.lines().map(str::trim).collect();
    let mut out = Vec::new();
    for (k, l) in lines.iter().enumerate() {
        if out.len() == 3 {
            break;
        }
        if l.starts_with("at ") {
            continue;
        }
        let l = l.trim_start_matches(|c: char| c.is_ascii_digit() || c == ':' || c == ' ');
        if !l.contains("risc0")
            || l.contains("hal::metal::runtime")
            || l.contains("hal::metal::buffer")
            || l.contains("sync_site")
        {
            continue;
        }
        let l = match l.rsplit_once("::h") {
            Some((a, h)) if h.len() == 16 && h.chars().all(|c| c.is_ascii_hexdigit()) => a,
            _ => l,
        };
        // Keep the method path, drop generic arguments.
        let mut depth = 0;
        let short: String = l
            .chars()
            .filter(|&c| {
                match c {
                    '<' => depth += 1,
                    '>' => depth -= 1,
                    _ => return depth <= 1,
                }
                false
            })
            .collect();
        let at = lines
            .get(k + 1)
            .filter(|n| n.starts_with("at "))
            .and_then(|n| n.rsplit('/').next())
            .unwrap_or("");
        out.push(format!("{short} [{at}]"));
    }
    out.join(" <- ")
}

/// Take (and clear) the [SYNC_TRACE_ENV] records.
pub fn sync_trace_take() -> Vec<SyncRecord> {
    std::mem::take(&mut *SYNC_TRACE.lock())
}

#[allow(unexpected_cfgs)]
fn gpu_end_time(cmd: &CommandBuffer) -> f64 {
    // SAFETY: plain Objective-C property read on a live, completed command buffer.
    unsafe { msg_send![cmd.as_ptr() as *mut Object, GPUEndTime] }
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

// ---- W7-A1: process-wide pool policy, memory-pressure drain, system-wide check, purge ----

/// `RISC0_METAL_POOL_GLOBAL=0` restores the pre-W7-A1 pool: a per-runtime cap of half the
/// working-set limit (a buffer released past it is freed), no cross-pool eviction, no eviction on
/// a pool miss and no eviction of stale sizes. Default on: the pooled (idle) bytes of every
/// runtime in the process together stay under half of the process working-set limit (LRU
/// eviction across pools); after a po2 change (a commit over a new number of rows), a miss evicts
/// this pool's LRU idle buffers of the previous po2's sizes so that pooled buffers never lift the
/// runtime above its own high-water mark of live + pooled bytes (the new po2 reuses that memory
/// instead of adding to it); and sizes not requested within
/// [POOL_STALE_BATCHES_ENV] batches of their runtime are evicted.
pub const POOL_GLOBAL_ENV: &str = "RISC0_METAL_POOL_GLOBAL";

/// Batches of a runtime after which a pooled size that was not requested is evicted (default
/// [DEFAULT_POOL_STALE_BATCHES]; `0` disables the staleness eviction). Only counts the batches of
/// the pool's own runtime, so an idle HAL's pool does not age.
pub const POOL_STALE_BATCHES_ENV: &str = "RISC0_METAL_POOL_STALE_BATCHES";

/// Default of [POOL_STALE_BATCHES_ENV]: a po2 20 rv32im segment opens ~28 batches (W7-A1
/// note), so a size has to sit unused for ~36 segments' worth of the HAL's own work before it
/// goes.
pub const DEFAULT_POOL_STALE_BATCHES: u64 = 1024;

/// `RISC0_METAL_PRESSURE_DRAIN=0` disables the memory-pressure drain: a libdispatch
/// `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` source that, on a WARN or CRITICAL event, runs
/// [drain_idle_pools] (idle pooled buffers of every runtime plus every registered
/// [register_idle_releaser]).
pub const PRESSURE_DRAIN_ENV: &str = "RISC0_METAL_PRESSURE_DRAIN";

/// `RISC0_METAL_SYSTEM_CHECK=0` disables the system-wide GPU memory check: without an explicit
/// budget, an allocation of at least [SYSTEM_CHECK_MIN_BYTES] that would take the IOAccelerator
/// "In use system memory" (every process's GPU memory) past `recommendedMaxWorkingSetSize` first
/// drains this process's pools and then fails with the working-set budget error instead of
/// wiring the machine (the reading is [system_gpu_committed]: the larger of "In use" and "Alloc
/// system memory").
pub const SYSTEM_CHECK_ENV: &str = "RISC0_METAL_SYSTEM_CHECK";

/// Allocations below this size skip the system-wide check.
pub const SYSTEM_CHECK_MIN_BYTES: u64 = 256 << 20;

/// The system-wide reading is refreshed at most once per this many seconds.
const SYSTEM_CHECK_CACHE_SECS: f64 = 0.050;

/// When idle buffers the pool frees get `setPurgeableState(Empty)` before release, which lets the
/// OS take their pages back at once instead of seconds later (M9 open issue 2):
/// unset (default): only buffers released by [drain_idle_pools] (memory-pressure events) and by
/// the system-wide check's drain; `1`: every idle buffer the pool frees (eviction, drains, the
/// pool of a dropped runtime, bypass frees with no batch in flight); `0`: never. Purging every
/// free is not the default because it costs time: Metal reuses the pages of recently freed
/// buffers for new allocations, and purged pages have to be faulted in again (W7-A1 worker A/B,
/// memory-merkle: recursion +0.10 s, identity_p254 +28 ms, shrink_wrap +21 ms with `1`).
pub const PURGE_ON_FREE_ENV: &str = "RISC0_METAL_PURGE_ON_FREE";

/// Measurement tool: `RISC0_METAL_POOL_TRACE=<ms>` prints [pool_report] to stderr every `<ms>`
/// milliseconds (from a background thread) and when a runtime drops.
pub const POOL_TRACE_ENV: &str = "RISC0_METAL_POOL_TRACE";

fn env_on(name: &str) -> bool {
    std::env::var(name).map_or(true, |v| v.trim() != "0")
}

fn pool_global() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_on(POOL_GLOBAL_ENV))
}

/// [PURGE_ON_FREE_ENV]: 0 = never, 1 = pressure/system drains only (default), 2 = every free.
fn purge_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var(PURGE_ON_FREE_ENV).as_deref().map(str::trim) {
            Ok("0") => 0,
            Ok("1") => 2,
            _ => 1,
        },
    )
}

thread_local! {
    /// Set while this thread releases memory because the machine is short of it.
    static RELIEF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` as a memory-relief release (pressure drain, system-check drain): idle buffers it
/// frees are purged unless [PURGE_ON_FREE_ENV]`=0`.
fn as_relief<R>(f: impl FnOnce() -> R) -> R {
    let prev = RELIEF.with(|r| r.replace(true));
    let out = f();
    RELIEF.with(|r| r.set(prev));
    out
}

fn purge_on_free() -> bool {
    match purge_mode() {
        0 => false,
        2 => true,
        _ => RELIEF.with(|r| r.get()),
    }
}

fn system_check_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_on(SYSTEM_CHECK_ENV))
}

fn stale_batches() -> u64 {
    static K: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var(POOL_STALE_BATCHES_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_POOL_STALE_BATCHES)
    })
}

/// Accounting groups for the process-wide cap. Group 0 is every production runtime; tests move
/// their runtimes into a group of their own ([Runtime::set_pool_group]) so that parallel tests
/// neither evict each other's buffers nor count toward each other's cap.
const POOL_GROUPS: usize = 16;

/// Pooled bytes (ready, pending and set aside) of every live runtime, per group.
static GROUP_CACHED: [AtomicU64; POOL_GROUPS] = [const { AtomicU64::new(0) }; POOL_GROUPS];

/// Test override of a group's cap (0 = the process cap).
static GROUP_CAP: [AtomicU64; POOL_GROUPS] = [const { AtomicU64::new(0) }; POOL_GROUPS];

/// The process working-set limit ([MAX_WORKING_SET_ENV] capped at the device's
/// `recommendedMaxWorkingSetSize`, else the latter); set by the first runtime.
static PROCESS_LIMIT: AtomicU64 = AtomicU64::new(0);

/// LRU clock: every buffer put back into a pool takes the next tick.
static TICK: AtomicU64 = AtomicU64::new(1);

/// Process-wide pool counters (see [PoolCounters]).
static COUNTERS: PoolCountersAtomic = PoolCountersAtomic::new();

/// Counters of the pool policy since process start ([pool_counters]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolCounters {
    /// Shared allocations served from a pool / newly allocated.
    pub hits: u64,
    pub misses: u64,
    /// Bytes evicted by the process-wide (or per-runtime) cap.
    pub evicted_cap: u64,
    /// Bytes evicted on a miss to keep live + pooled under the runtime's high-water mark.
    pub evicted_miss: u64,
    /// Bytes evicted as stale ([POOL_STALE_BATCHES_ENV]).
    pub evicted_stale: u64,
    /// Bytes released by [drain_idle_pools] (pressure events and explicit calls).
    pub drained_idle: u64,
    /// Buffers handed to `setPurgeableState(Empty)` before release.
    pub purged: u64,
    /// Memory-pressure events seen (WARN or CRITICAL).
    pub pressure_events: u64,
    /// Allocations refused by the system-wide check.
    pub system_refusals: u64,
}

struct PoolCountersAtomic {
    hits: AtomicU64,
    misses: AtomicU64,
    evicted_cap: AtomicU64,
    evicted_miss: AtomicU64,
    evicted_stale: AtomicU64,
    drained_idle: AtomicU64,
    purged: AtomicU64,
    pressure_events: AtomicU64,
    system_refusals: AtomicU64,
}

impl PoolCountersAtomic {
    const fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evicted_cap: AtomicU64::new(0),
            evicted_miss: AtomicU64::new(0),
            evicted_stale: AtomicU64::new(0),
            drained_idle: AtomicU64::new(0),
            purged: AtomicU64::new(0),
            pressure_events: AtomicU64::new(0),
            system_refusals: AtomicU64::new(0),
        }
    }
}

/// The process-wide pool counters.
pub fn pool_counters() -> PoolCounters {
    let c = &COUNTERS;
    let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
    PoolCounters {
        hits: l(&c.hits),
        misses: l(&c.misses),
        evicted_cap: l(&c.evicted_cap),
        evicted_miss: l(&c.evicted_miss),
        evicted_stale: l(&c.evicted_stale),
        drained_idle: l(&c.drained_idle),
        purged: l(&c.purged),
        pressure_events: l(&c.pressure_events),
        system_refusals: l(&c.system_refusals),
    }
}

/// Release an idle buffer (no batch that may use it is open or running), with
/// `setPurgeableState(Empty)` first when [PURGE_ON_FREE_ENV] says so, so its pages go back to the
/// OS now.
fn discard_idle(buf: MtlBuffer) {
    if purge_on_free() {
        buf.set_purgeable_state(metal::MTLPurgeableState::Empty);
        COUNTERS.purged.fetch_add(1, Ordering::Relaxed);
    }
    drop(buf);
}

/// The cap of the pooled bytes of every runtime in `group` together: half the process limit
/// (as the per-runtime cap has always been), or a test override.
fn group_cap(group: usize) -> u64 {
    match GROUP_CAP[group].load(Ordering::Relaxed) {
        0 => PROCESS_LIMIT.load(Ordering::Relaxed) / 2,
        cap => cap,
    }
}

/// Opaque id of the calling thread (the address of a thread-local), for the Z14 prefill
/// single-thread check.
fn thread_token() -> usize {
    thread_local!(static TOKEN: u8 = const { 0 });
    TOKEN.with(|t| t as *const u8 as usize)
}

/// One idle pooled buffer and its LRU tick.
struct Entry {
    buf: MtlBuffer,
    tick: u64,
}

/// Size-keyed pool of shared buffers. A buffer dropped while a batch that may use it is still
/// open or running stays in `pending` until that batch has completed (plan 2.6).
struct Pool {
    ready: HashMap<u64, Vec<Entry>>,
    /// (batch sequence number that must complete first, bytes, buffer)
    pending: Vec<(u64, u64, Entry)>,
    cached_bytes: u64,
    /// Bytes this runtime allocated and has not freed: live and pooled shared buffers, plus
    /// private buffers (never subtracted; their drop is not observable). The [BudgetScope::Hal]
    /// budget is checked against it.
    owned_bytes: u64,
    /// Z14: pooled buffers set aside for `Hal::prefill_elem` (counted in `cached_bytes`; a drain
    /// releases them like the rest of the pool).
    prefilled: Vec<Prefilled>,
    /// The runtime's `completed_seq` (shared), so drains can tell idle pending buffers apart.
    completed: Arc<AtomicU64>,
    /// Accounting group ([POOL_GROUPS]).
    group: usize,
    /// Shared bytes handed out by this pool's runtime and not returned yet, and its high-water
    /// mark.
    live: u64,
    live_peak: u64,
    /// High-water mark of live + pooled bytes (the pool's footprint).
    total_peak: u64,
    /// Batch sequence number of the last request of each size (hit or miss).
    last_req: HashMap<u64, u64>,
    /// Sequence number of the last staleness scan.
    last_stale_scan: u64,
    /// Rows of the last commit (`MetalHal` commit / zk-shift interpolation) and the batch
    /// sequence number at which that domain last changed: the po2 epoch. A miss evicts only sizes
    /// last requested before the epoch began (the previous po2's), never ones of the current
    /// po2, so a steady workload never churns.
    domain: u64,
    epoch: u64,
    /// Test override of [stale_batches] for this pool (0 = the process value).
    #[cfg(test)]
    stale_k: u64,
}

/// Z14: a pooled buffer of `len` bytes to be handed out by `alloc_elem_init(.., word)`; `filled`
/// once a fill of `word` has been encoded (in a hash kernel).
struct Prefilled {
    len: u64,
    word: u32,
    buf: MtlBuffer,
    filled: bool,
    /// [thread_token] of the thread that marked it filled (0 until then).
    marked_by: usize,
}

/// Why idle buffers are evicted (counter selection).
#[derive(Clone, Copy)]
enum Evict {
    Cap,
    Miss,
    Stale,
}

impl Evict {
    fn counter(self) -> &'static AtomicU64 {
        match self {
            Evict::Cap => &COUNTERS.evicted_cap,
            Evict::Miss => &COUNTERS.evicted_miss,
            Evict::Stale => &COUNTERS.evicted_stale,
        }
    }
}

impl Pool {
    fn new(completed: Arc<AtomicU64>) -> Self {
        Self {
            ready: HashMap::new(),
            pending: Vec::new(),
            cached_bytes: 0,
            owned_bytes: 0,
            prefilled: Vec::new(),
            completed,
            group: 0,
            live: 0,
            live_peak: 0,
            total_peak: 0,
            last_req: HashMap::new(),
            last_stale_scan: 0,
            domain: 0,
            epoch: 0,
            #[cfg(test)]
            stale_k: 0,
        }
    }

    fn stale_k(&self) -> u64 {
        #[cfg(test)]
        if self.stale_k != 0 {
            return self.stale_k;
        }
        stale_batches()
    }

    fn add_cached(&mut self, bytes: u64) {
        self.cached_bytes += bytes;
        GROUP_CACHED[self.group].fetch_add(bytes, Ordering::Relaxed);
    }

    fn sub_cached(&mut self, bytes: u64) {
        self.cached_bytes -= bytes;
        GROUP_CACHED[self.group].fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Bytes of this pool's buffers freed (not reused): no longer cached, no longer owned.
    fn forget(&mut self, bytes: u64) {
        self.sub_cached(bytes);
        self.owned_bytes = self.owned_bytes.saturating_sub(bytes);
    }

    fn reclaim(&mut self, completed: u64) {
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].0 <= completed {
                let (_, len, entry) = self.pending.swap_remove(i);
                self.ready.entry(len).or_default().push(entry);
            } else {
                i += 1;
            }
        }
    }

    fn pop(&mut self, len: u64) -> Option<MtlBuffer> {
        let entries = self.ready.get_mut(&len)?;
        let entry = entries.pop()?;
        if entries.is_empty() {
            self.ready.remove(&len);
        }
        self.sub_cached(len);
        Some(entry.buf)
    }

    /// Hand-out bookkeeping of `bytes` live bytes.
    fn handed_out(&mut self, bytes: u64) {
        self.live += bytes;
        self.live_peak = self.live_peak.max(self.live);
        self.total_peak = self.total_peak.max(self.live + self.cached_bytes);
    }

    /// Release every pooled buffer (budget failure, allocation failure, `trim_device_memory`).
    /// Idle ones are purged first; ones a batch may still use (pending, set aside) are only
    /// dropped (their command buffers keep them alive until they complete).
    fn drain(&mut self) -> u64 {
        let bytes = self.cached_bytes;
        self.reclaim(self.completed.load(Ordering::Acquire));
        // Dropping a buffer still referenced by a committed command buffer is safe: the command
        // buffer retains every resource it binds until it completes.
        for (_, entries) in self.ready.drain() {
            for e in entries {
                discard_idle(e.buf);
            }
        }
        self.pending.clear();
        self.prefilled.clear();
        self.forget(bytes);
        bytes
    }

    /// Release the idle buffers only: ready ones and pending ones whose batch completed; never
    /// buffers a batch may still use (in-flight pending ones, set-aside prefill buffers).
    fn drain_idle(&mut self) -> u64 {
        self.reclaim(self.completed.load(Ordering::Acquire));
        let mut bytes = 0;
        for (len, entries) in self.ready.drain().collect::<Vec<_>>() {
            for e in entries {
                bytes += len;
                discard_idle(e.buf);
            }
        }
        self.forget(bytes);
        bytes
    }

    /// The (size, index, tick) of the least recently returned idle buffer of at least
    /// [POOL_SMALL_BYTES] (small buffers are never evicted).
    fn lru(&self) -> Option<(u64, usize, u64)> {
        self.lru_where(|_| true)
    }

    /// [Self::lru] among the sizes `keep` accepts.
    fn lru_where(&self, keep: impl Fn(u64) -> bool) -> Option<(u64, usize, u64)> {
        let mut best: Option<(u64, usize, u64)> = None;
        for (&len, entries) in &self.ready {
            if len < POOL_SMALL_BYTES || !keep(len) {
                continue;
            }
            for (i, e) in entries.iter().enumerate() {
                if best.is_none_or(|b| e.tick < b.2) {
                    best = Some((len, i, e.tick));
                }
            }
        }
        best
    }

    fn evict_at(&mut self, len: u64, i: usize, why: Evict) {
        let entries = self.ready.get_mut(&len).unwrap();
        let e = entries.swap_remove(i);
        if entries.is_empty() {
            self.ready.remove(&len);
        }
        self.forget(len);
        why.counter().fetch_add(len, Ordering::Relaxed);
        discard_idle(e.buf);
    }

    /// Evict least recently returned idle buffers until at least `bytes` are freed or none is
    /// left; returns the bytes freed. With `pending`, buffers still waiting for their batch go
    /// too once no idle one is left (oldest first; dropped without a purge, their command buffer
    /// keeps them alive until it completes).
    fn evict_lru(&mut self, bytes: u64, why: Evict, pending: bool) -> u64 {
        let mut freed = 0;
        while freed < bytes {
            if let Some((len, i, _)) = self.lru() {
                self.evict_at(len, i, why);
                freed += len;
                continue;
            }
            if !pending {
                break;
            }
            let Some(i) = (0..self.pending.len())
                .filter(|&i| self.pending[i].1 >= POOL_SMALL_BYTES)
                .min_by_key(|&i| self.pending[i].2.tick)
            else {
                break;
            };
            let (_, len, _) = self.pending.swap_remove(i);
            self.forget(len);
            why.counter().fetch_add(len, Ordering::Relaxed);
            freed += len;
        }
        freed
    }

    /// Evict least recently returned idle buffers of sizes last requested before the current po2
    /// epoch until at least `bytes` are freed or none is left.
    fn evict_previous_epoch(&mut self, bytes: u64) {
        let epoch = self.epoch;
        let mut freed = 0;
        while freed < bytes {
            let last_req = &self.last_req;
            let Some((len, i, _)) =
                self.lru_where(|len| last_req.get(&len).is_none_or(|&r| r < epoch))
            else {
                break;
            };
            self.evict_at(len, i, Evict::Miss);
            freed += len;
        }
    }

    /// Evict every idle buffer of a size not requested within `k` batches of `seq`.
    fn evict_stale(&mut self, seq: u64, k: u64) {
        let stale: Vec<u64> = self
            .ready
            .keys()
            .copied()
            .filter(|&len| {
                len >= POOL_SMALL_BYTES
                    && self.last_req.get(&len).is_none_or(|&last| last + k < seq)
            })
            .collect();
        for len in stale {
            while self.ready.get(&len).is_some_and(|v| !v.is_empty()) {
                self.evict_at(len, 0, Evict::Stale);
            }
            self.ready.remove(&len);
            self.last_req.remove(&len);
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // The runtime waited for every batch before its pool goes: all of it is idle.
        let bytes = self.cached_bytes;
        self.reclaim(u64::MAX);
        for (_, entries) in self.ready.drain() {
            for e in entries {
                discard_idle(e.buf);
            }
        }
        for p in self.prefilled.drain(..) {
            discard_idle(p.buf);
        }
        GROUP_CACHED[self.group].fetch_sub(bytes, Ordering::Relaxed);
    }
}

/// Evict least recently returned idle buffers across every pool of `group` until the group's
/// pooled bytes are under its cap (or nothing idle is left). Takes one pool lock at a time and
/// never the registry lock while holding a pool lock.
fn evict_group_to_cap(group: usize) {
    let cap = group_cap(group);
    if cap == 0 {
        return;
    }
    let pools: Vec<Arc<Mutex<Pool>>> = {
        let reg = POOLS.lock();
        reg.iter().filter_map(Weak::upgrade).collect()
    };
    while GROUP_CACHED[group].load(Ordering::Relaxed) > cap {
        // The pool with the globally oldest idle buffer.
        let mut best: Option<(usize, u64)> = None;
        for (k, p) in pools.iter().enumerate() {
            let p = p.lock();
            if p.group != group {
                continue;
            }
            if let Some((_, _, tick)) = p.lru() {
                if best.is_none_or(|b| tick < b.1) {
                    best = Some((k, tick));
                }
            }
        }
        let Some((k, tick)) = best else {
            return;
        };
        let mut p = pools[k].lock();
        // Re-check under the lock (another thread may have taken it).
        if let Some((len, i, t)) = p.lru() {
            if t == tick {
                p.evict_at(len, i, Evict::Cap);
            }
        }
    }
}

/// Release the idle pooled buffers of every runtime in the process (never buffers a batch may
/// still use, never live ones) and run every [register_idle_releaser] hook; returns the bytes
/// released. Called by the memory-pressure source; callable at any time, from any thread. Takes
/// one pool lock at a time, never the GPU lock.
pub fn drain_idle_pools() -> u64 {
    let pools: Vec<Arc<Mutex<Pool>>> = {
        let mut reg = POOLS.lock();
        reg.retain(|w| w.strong_count() > 0);
        reg.iter().filter_map(Weak::upgrade).collect()
    };
    let mut bytes: u64 = as_relief(|| pools.iter().map(|p| p.lock().drain_idle()).sum());
    COUNTERS.drained_idle.fetch_add(bytes, Ordering::Relaxed);
    let hooks: Vec<(u64, Arc<IdleReleaser>)> = RELEASERS.lock().clone();
    for (_, f) in hooks {
        bytes += f();
    }
    bytes
}

type IdleReleaser = dyn Fn() -> u64 + Send + Sync;

static RELEASERS: Mutex<Vec<(u64, Arc<IdleReleaser>)>> = Mutex::new(Vec::new());
static NEXT_RELEASER: AtomicU64 = AtomicU64::new(1);

/// Handle of a hook registered with [register_idle_releaser]; dropping it unregisters the hook.
#[must_use = "dropping the handle unregisters the releaser"]
pub struct IdleReleaserHandle(u64);

impl Drop for IdleReleaserHandle {
    fn drop(&mut self) {
        RELEASERS.lock().retain(|(id, _)| *id != self.0);
    }
}

impl IdleReleaserHandle {
    /// Keep the hook registered for the rest of the process.
    pub fn leak(self) {
        mem::forget(self);
    }
}

/// Register a hook that frees memory a cache holds idle (e.g. recursion's kept host copies, the
/// Groth16 witness graph) and returns the bytes freed. [drain_idle_pools] (and so every
/// memory-pressure event) calls it. It may run on a libdispatch thread at any time: it must not
/// take [super::exclusive] or wait for GPU work, and must only release what no caller is using.
pub fn register_idle_releaser(f: impl Fn() -> u64 + Send + Sync + 'static) -> IdleReleaserHandle {
    let id = NEXT_RELEASER.fetch_add(1, Ordering::Relaxed);
    RELEASERS.lock().push((id, Arc::new(f)));
    IdleReleaserHandle(id)
}

/// A memory-pressure level from libdispatch (`DISPATCH_MEMORYPRESSURE_*`).
pub const PRESSURE_NORMAL: usize = 0x1;
pub const PRESSURE_WARN: usize = 0x2;
pub const PRESSURE_CRITICAL: usize = 0x4;

/// What the memory-pressure handler does with an event of `level` (also the test seam).
pub(crate) fn on_memory_pressure(level: usize) -> u64 {
    if level & (PRESSURE_WARN | PRESSURE_CRITICAL) == 0 {
        return 0;
    }
    COUNTERS.pressure_events.fetch_add(1, Ordering::Relaxed);
    let bytes = drain_idle_pools();
    tracing::info!(
        "Metal memory pressure ({}): released {:.1} MiB of idle pooled memory",
        if level & PRESSURE_CRITICAL != 0 {
            "critical"
        } else {
            "warn"
        },
        bytes as f64 / (1u64 << 20) as f64
    );
    bytes
}

#[allow(non_upper_case_globals)]
mod dispatch_ffi {
    use std::ffi::c_void;

    #[repr(C)]
    pub struct SourceType {
        _private: [u8; 0],
    }

    extern "C" {
        pub static _dispatch_source_type_memorypressure: SourceType;
        pub fn dispatch_source_create(
            kind: *const SourceType,
            handle: usize,
            mask: usize,
            queue: *mut c_void,
        ) -> *mut c_void;
        pub fn dispatch_source_set_event_handler_f(
            source: *mut c_void,
            handler: extern "C" fn(*mut c_void),
        );
        pub fn dispatch_set_context(object: *mut c_void, context: *mut c_void);
        pub fn dispatch_source_get_data(source: *mut c_void) -> usize;
        pub fn dispatch_resume(object: *mut c_void);
        pub fn dispatch_get_global_queue(identifier: isize, flags: usize) -> *mut c_void;
    }
}

extern "C" fn pressure_handler(source: *mut c_void) {
    // SAFETY: `source` is the live dispatch source this handler is installed on (its context).
    let level = unsafe { dispatch_ffi::dispatch_source_get_data(source) };
    os_pressure_event(level);
}

/// What the OS memory-pressure handler runs. In test builds a real event first takes
/// [gpu_test_lock](super::gpu_test_lock), as the budget-failure drains do (Z8b): tests that
/// assert pool state hold that lock, so an OS WARN during the suite cannot drain their pools
/// between two assertions. Production builds drain at once (the drain never takes the GPU lock).
pub(crate) fn os_pressure_event(level: usize) -> u64 {
    #[cfg(test)]
    let _lock = super::gpu_test_lock();
    on_memory_pressure(level)
}

/// Install the process's memory-pressure source once (unless [PRESSURE_DRAIN_ENV]`=0`).
fn install_pressure_source() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    if !env_on(PRESSURE_DRAIN_ENV) {
        return;
    }
    ONCE.call_once(|| {
        // SAFETY: plain libdispatch calls; the source lives for the rest of the process (never
        // cancelled or released), and its context is the source itself.
        unsafe {
            use dispatch_ffi::*;
            // QOS_CLASS_UTILITY
            let queue = dispatch_get_global_queue(0x11, 0);
            let source = dispatch_source_create(
                &_dispatch_source_type_memorypressure,
                0,
                PRESSURE_WARN | PRESSURE_CRITICAL,
                queue,
            );
            if source.is_null() {
                tracing::warn!("could not create a memory-pressure dispatch source");
                return;
            }
            dispatch_set_context(source, source);
            dispatch_source_set_event_handler_f(source, pressure_handler);
            dispatch_resume(source);
        }
    });
}

#[allow(non_upper_case_globals, non_camel_case_types)]
mod iokit_ffi {
    use std::ffi::{c_char, c_void};

    pub type CFTypeRef = *const c_void;
    pub const kCFStringEncodingUTF8: u32 = 0x0800_0100;
    pub const kCFNumberSInt64Type: isize = 4;

    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        pub fn IOServiceMatching(name: *const c_char) -> *mut c_void;
        pub fn IOServiceGetMatchingServices(
            main_port: u32,
            matching: *mut c_void,
            existing: *mut u32,
        ) -> i32;
        pub fn IOIteratorNext(iterator: u32) -> u32;
        pub fn IOObjectRelease(object: u32) -> i32;
        pub fn IORegistryEntryCreateCFProperty(
            entry: u32,
            key: CFTypeRef,
            allocator: CFTypeRef,
            options: u32,
        ) -> CFTypeRef;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        pub fn CFStringCreateWithCString(
            alloc: CFTypeRef,
            cstr: *const c_char,
            encoding: u32,
        ) -> CFTypeRef;
        pub fn CFDictionaryGetValue(dict: CFTypeRef, key: CFTypeRef) -> CFTypeRef;
        pub fn CFNumberGetValue(number: CFTypeRef, kind: isize, value: *mut c_void) -> u8;
        pub fn CFGetTypeID(cf: CFTypeRef) -> usize;
        pub fn CFDictionaryGetTypeID() -> usize;
        pub fn CFNumberGetTypeID() -> usize;
        pub fn CFRelease(cf: CFTypeRef);
    }
}

/// The IOAccelerator "In use system memory" in bytes (GPU memory of every process on the
/// machine), summed over every accelerator; `None` if it cannot be read. Readable without
/// privileges (`ioreg -r -c IOAccelerator`). Uncached; about 0.1 ms.
pub fn system_gpu_in_use() -> Option<u64> {
    accelerator_stat(c"In use system memory")
}

/// What the system-wide check compares: the larger of the IOAccelerator "In use system memory"
/// and "Alloc system memory". W7-A1 probe (`tests_a1::probe_system_counter`): a new 1 GiB
/// buffer adds 1 GiB to "Alloc" at once but to "In use" only after its first GPU use (CPU
/// writes do not count), and both drop when the buffer is released (purging alone changes
/// neither). "In use" alone would let a process allocate its whole working set before the
/// counter moves.
pub fn system_gpu_committed() -> Option<u64> {
    match (
        accelerator_stat(c"In use system memory"),
        accelerator_stat(c"Alloc system memory"),
    ) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

/// One integer of the IOAccelerator `PerformanceStatistics` dictionary, summed over every
/// accelerator.
pub(crate) fn accelerator_stat(key: &std::ffi::CStr) -> Option<u64> {
    use iokit_ffi::*;
    // SAFETY: IOKit / CoreFoundation calls with checked results; every created object is
    // released, the matching dictionary is consumed by IOServiceGetMatchingServices.
    unsafe {
        let matching = IOServiceMatching(c"IOAccelerator".as_ptr());
        if matching.is_null() {
            return None;
        }
        let mut iter = 0u32;
        if IOServiceGetMatchingServices(0, matching, &mut iter) != 0 {
            return None;
        }
        let stats_key = CFStringCreateWithCString(
            std::ptr::null(),
            c"PerformanceStatistics".as_ptr(),
            kCFStringEncodingUTF8,
        );
        let use_key =
            CFStringCreateWithCString(std::ptr::null(), key.as_ptr(), kCFStringEncodingUTF8);
        let mut total: Option<u64> = None;
        loop {
            let entry = IOIteratorNext(iter);
            if entry == 0 {
                break;
            }
            let stats = IORegistryEntryCreateCFProperty(entry, stats_key, std::ptr::null(), 0);
            if !stats.is_null() {
                if CFGetTypeID(stats) == CFDictionaryGetTypeID() {
                    let n = CFDictionaryGetValue(stats, use_key);
                    let mut v: i64 = 0;
                    if !n.is_null()
                        && CFGetTypeID(n) == CFNumberGetTypeID()
                        && CFNumberGetValue(n, kCFNumberSInt64Type, &mut v as *mut i64 as _) != 0
                    {
                        *total.get_or_insert(0) += v.max(0) as u64;
                    }
                }
                CFRelease(stats);
            }
            IOObjectRelease(entry);
        }
        CFRelease(stats_key);
        CFRelease(use_key);
        IOObjectRelease(iter);
        total
    }
}

/// [system_gpu_committed], refreshed at most once per [SYSTEM_CHECK_CACHE_SECS] unless `fresh`.
fn system_gpu_in_use_cached(fresh: bool) -> Option<u64> {
    static CACHE: Mutex<Option<(f64, Option<u64>)>> = Mutex::new(None);
    let now = host_secs();
    let mut c = CACHE.lock();
    if !fresh {
        if let Some((t, v)) = *c {
            if now - t < SYSTEM_CHECK_CACHE_SECS {
                return v;
            }
        }
    }
    let v = system_gpu_committed();
    *c = Some((now, v));
    v
}

/// A text table of every live pool: group, live and pooled bytes, and the pooled sizes (largest
/// first), plus the process counters and the system-wide GPU memory reading.
pub fn pool_report() -> String {
    let pools: Vec<Arc<Mutex<Pool>>> = {
        let reg = POOLS.lock();
        reg.iter().filter_map(Weak::upgrade).collect()
    };
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    let mut out = String::new();
    let c = pool_counters();
    out += &format!(
        "[pool] system in-use/alloc {:.0} MiB | group0 pooled {:.0} MiB cap {:.0} MiB | hits {} misses {} \
         | evicted cap {:.0} miss {:.0} stale {:.0} idle {:.0} MiB | purged {} | pressure {} | \
         refused {}\n",
        mib(system_gpu_committed().unwrap_or(0)),
        mib(GROUP_CACHED[0].load(Ordering::Relaxed)),
        mib(group_cap(0)),
        c.hits,
        c.misses,
        mib(c.evicted_cap),
        mib(c.evicted_miss),
        mib(c.evicted_stale),
        mib(c.drained_idle),
        c.purged,
        c.pressure_events,
        c.system_refusals
    );
    for (k, p) in pools.iter().enumerate() {
        let p = p.lock();
        let mut sizes: Vec<(u64, usize)> = p.ready.iter().map(|(&l, v)| (l, v.len())).collect();
        for (_, l, _) in &p.pending {
            sizes.push((*l, 1));
        }
        let mut merged: HashMap<u64, usize> = HashMap::new();
        for (l, n) in sizes {
            *merged.entry(l).or_default() += n;
        }
        let mut sizes: Vec<(u64, usize)> = merged.into_iter().collect();
        sizes.sort_by_key(|s| std::cmp::Reverse(s.0));
        let small: u64 = sizes
            .iter()
            .filter(|s| s.0 < 1 << 20)
            .map(|s| s.0 * s.1 as u64)
            .sum();
        let list: Vec<String> = sizes
            .iter()
            .filter(|s| s.0 >= 1 << 20)
            .take(16)
            .map(|(l, n)| format!("{:.1}x{n}", mib(*l)))
            .collect();
        out += &format!(
            "[pool] #{k} g{} seq {} domain {} epoch {} live {:.0} peak {:.0} total-peak {:.0} pooled \
             {:.0} (pending {}, prefilled {}) owned {:.0} MiB | sizes MiB: {} | <1MiB {:.1} MiB\n",
            p.group,
            p.last_req.values().max().copied().unwrap_or(0),
            p.domain,
            p.epoch,
            mib(p.live),
            mib(p.live_peak),
            mib(p.total_peak),
            mib(p.cached_bytes),
            p.pending.len(),
            p.prefilled.len(),
            mib(p.owned_bytes),
            list.join(" "),
            mib(small)
        );
    }
    out
}

fn start_pool_trace() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let Some(ms) = std::env::var(POOL_TRACE_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&ms| ms > 0)
    else {
        return;
    };
    ONCE.call_once(|| {
        std::thread::Builder::new()
            .name("metal-pool-trace".into())
            .spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_millis(ms));
                eprint!("{}", pool_report());
            })
            .ok();
    });
}

/// State shared by a `MetalHal` and every buffer it allocated.
pub(crate) struct Runtime {
    pub(crate) device: Device,
    pub(crate) queue: CommandQueue,
    state: Mutex<State>,
    pool: Arc<Mutex<Pool>>,
    /// Sequence number of the most recently opened batch (0: none yet).
    open_seq: AtomicU64,
    /// Every batch with a sequence number up to this one has completed (shared with the pool).
    completed_seq: Arc<AtomicU64>,
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
    /// Test hook: a fake system-wide GPU in-use reading (0 = read IOAccelerator).
    #[cfg(test)]
    fake_system_in_use: AtomicU64,
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
        // The process limit follows the environment, not per-HAL overrides.
        let process_limit = match env_working_set_limit() {
            Some(b) => b.min(recommended_ws),
            None => recommended_ws,
        };
        PROCESS_LIMIT.store(process_limit, Ordering::Relaxed);
        install_pressure_source();
        start_pool_trace();
        if budget == 0 && system_check_on() {
            if let Some(in_use) = system_gpu_in_use_cached(true) {
                let own = device.current_allocated_size();
                let others = in_use.saturating_sub(own);
                if others > recommended_ws / 2 {
                    tracing::warn!(
                        "other processes hold {:.1} GiB of GPU memory (IOAccelerator 'In use \
                         system memory' {:.1} GiB, this process {:.1} GiB, \
                         recommendedMaxWorkingSetSize {:.1} GiB): large allocations may be \
                         refused",
                        others as f64 / (1u64 << 30) as f64,
                        in_use as f64 / (1u64 << 30) as f64,
                        own as f64 / (1u64 << 30) as f64,
                        recommended_ws as f64 / (1u64 << 30) as f64
                    );
                }
            }
        }
        let completed_seq = Arc::new(AtomicU64::new(0));
        let pool = Arc::new(Mutex::new(Pool::new(completed_seq.clone())));
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
            completed_seq,
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
            #[cfg(test)]
            fake_system_in_use: AtomicU64::new(0),
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

    /// The system-wide GPU memory in use (IOAccelerator, cached), or the test override.
    fn system_in_use(&self, fresh: bool) -> Option<u64> {
        #[cfg(test)]
        match self.fake_system_in_use.load(Ordering::Relaxed) {
            0 => {}
            v => return Some(v),
        }
        system_gpu_in_use_cached(fresh)
    }

    /// Without an explicit budget: refuse an allocation of `bytes` (at least
    /// [SYSTEM_CHECK_MIN_BYTES]) that would take the system-wide GPU memory in use past
    /// `recommendedMaxWorkingSetSize`, after draining every pool of this process once. macOS
    /// rarely fails `newBuffer` before the machine runs out of wired memory, so without this a
    /// second large prover on the machine hangs it instead of failing (LEADS2 S3).
    fn check_system(&self, what: &str, bytes: u64) -> Result<()> {
        if bytes < SYSTEM_CHECK_MIN_BYTES || self.budget().is_some() || !system_check_on() {
            return Ok(());
        }
        let limit = self.recommended_ws;
        let Some(in_use) = self.system_in_use(false) else {
            return Ok(());
        };
        if in_use.saturating_add(bytes) <= limit {
            return Ok(());
        }
        let drained = as_relief(drain_all_pools);
        let in_use = self.system_in_use(true).unwrap_or(in_use);
        if in_use.saturating_add(bytes) <= limit {
            tracing::debug!("system-wide GPU memory: drained {drained} pooled bytes for {what}");
            return Ok(());
        }
        COUNTERS.system_refusals.fetch_add(1, Ordering::Relaxed);
        Err(anyhow!(
            "Metal working-set budget exceeded: allocating {bytes} bytes for {what} with {in_use} \
             bytes of GPU memory in use system-wide (IOAccelerator 'In use' / 'Alloc system \
             memory'; {} of them this process's) would exceed recommendedMaxWorkingSetSize={limit} bytes \
             (another GPU process is using the memory; {SYSTEM_CHECK_ENV}=0 disables this check)",
            self.allocated_size()
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
            if sync_trace_level() > 0 && st.in_flight.is_empty() {
                if let Some(t) = LAST_SYNC_END.lock().take() {
                    if let Some(r) = SYNC_TRACE.lock().last_mut() {
                        r.idle_after_secs = host_secs() - t;
                        if sync_trace_level() > 1 {
                            eprintln!("[sync-trace] idle {:.3} ms", r.idle_after_secs * 1e3);
                        }
                    }
                }
            }
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
        let trace = sync_trace_level() > 0 && (st.open.is_some() || !st.in_flight.is_empty());
        let t0 = if trace { host_secs() } else { 0.0 };
        self.close(&mut st);
        for batch in st.in_flight.iter() {
            batch.cmd.wait_until_completed();
        }
        if trace {
            let t1 = host_secs();
            let end = st.in_flight.back().map_or(t1, |b| gpu_end_time(&b.cmd));
            let rec = SyncRecord {
                site: sync_site(),
                wait_secs: t1 - t0,
                wake_secs: t1 - end,
                idle_after_secs: 0.0,
            };
            if sync_trace_level() > 1 {
                eprintln!(
                    "[sync-trace] wait {:.3} ms wake {:.1} us at {}",
                    rec.wait_secs * 1e3,
                    rec.wake_secs * 1e6,
                    rec.site
                );
            }
            SYNC_TRACE.lock().push(rec);
            *LAST_SYNC_END.lock() = Some(host_secs());
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
            let global = pool_global();
            let seq = self.open_seq.load(Ordering::Acquire);
            pool.last_req.insert(bytes, seq);
            if global {
                let k = pool.stale_k();
                if k > 0 && seq >= pool.last_stale_scan + k.div_ceil(4) {
                    pool.last_stale_scan = seq;
                    pool.evict_stale(seq, k);
                }
            }
            if let Some(buf) = pool.pop(bytes) {
                pool.handed_out(bytes);
                COUNTERS.hits.fetch_add(1, Ordering::Relaxed);
                #[cfg(test)]
                if self.zero_pooled.load(Ordering::Relaxed) {
                    // SAFETY: every batch that used this buffer has completed; `bytes` is its
                    // length.
                    unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes as usize) };
                }
                return Ok(buf);
            }
            COUNTERS.misses.fetch_add(1, Ordering::Relaxed);
            if global && bytes >= POOL_SMALL_BYTES {
                // Pooled buffers must not lift this runtime above its own high-water mark of
                // live + pooled bytes: on a po2 or phase change the new sizes reuse the memory
                // of the old ones. In a steady state (same sizes every job) the first job sets
                // the mark and later jobs hit, so nothing is evicted.
                let bound = pool.total_peak.max(pool.live + bytes);
                let excess = (pool.live + pool.cached_bytes + bytes).saturating_sub(bound);
                if excess > 0 {
                    pool.evict_previous_epoch(excess);
                }
            }
        }
        self.check_budget(name, bytes)?;
        self.check_system(name, bytes)?;
        let options = MTLResourceOptions::StorageModeShared;
        if let Some(buf) = self.new_buffer_raw(bytes, options) {
            self.allocated(bytes);
            self.pool.lock().handed_out(bytes);
            return Ok(buf);
        }
        tracing::warn!("Metal allocation of {bytes} bytes for {name} failed; draining the pools");
        drain_all_pools();
        let buf = self
            .new_buffer_raw(bytes, options)
            .ok_or_else(|| anyhow!("Metal allocation failed on {name}: {bytes} bytes"))?;
        self.allocated(bytes);
        self.pool.lock().handed_out(bytes);
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
        self.check_system("a private buffer", bytes)?;
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
    /// (`set_bypass_pool`). It is purged first when no batch of this runtime is open or running.
    pub(crate) fn freed(&self, bytes: u64, buf: MtlBuffer) {
        let idle = !self.unsynced();
        {
            let mut pool = self.pool.lock();
            pool.owned_bytes = pool.owned_bytes.saturating_sub(bytes);
            pool.live = pool.live.saturating_sub(bytes);
        }
        if idle {
            discard_idle(buf);
        }
    }

    /// W7-A1: a commit over `rows` rows (the segment's po2 domain). A new value starts a new po2
    /// epoch for the pool's eviction on a miss.
    pub(crate) fn note_domain(&self, rows: usize) {
        let rows = rows as u64;
        let mut pool = self.pool.lock();
        if pool.domain != rows {
            pool.domain = rows;
            pool.epoch = self.open_seq.load(Ordering::Acquire);
        }
    }

    /// Return a dropped buffer. Never blocks on GPU work and never takes the state mutex.
    pub(crate) fn release(&self, bytes: u64, buf: MtlBuffer) {
        let needed = self.open_seq.load(Ordering::Acquire);
        let completed = self.completed_seq.load(Ordering::Acquire);
        let idle = needed <= completed;
        let cap = self.pool_cap.load(Ordering::Relaxed);
        let group = {
            let mut pool = self.pool.lock();
            pool.live = pool.live.saturating_sub(bytes);
            if !pool_global() {
                if bytes >= POOL_SMALL_BYTES && pool.cached_bytes + bytes > cap {
                    // Freed (the command buffers that use it keep their own reference).
                    pool.owned_bytes = pool.owned_bytes.saturating_sub(bytes);
                    drop(pool);
                    if idle {
                        discard_idle(buf);
                    }
                    return;
                }
                pool.add_cached(bytes);
                let entry = Entry { buf, tick: 0 };
                if idle {
                    pool.ready.entry(bytes).or_default().push(entry);
                } else {
                    pool.pending.push((needed, bytes, entry));
                }
                return;
            }
            if bytes >= POOL_SMALL_BYTES && bytes > cap.min(group_cap(pool.group)) {
                // Larger than any cap: never pooled.
                pool.owned_bytes = pool.owned_bytes.saturating_sub(bytes);
                drop(pool);
                if idle {
                    discard_idle(buf);
                }
                return;
            }
            pool.add_cached(bytes);
            let entry = Entry {
                buf,
                tick: TICK.fetch_add(1, Ordering::Relaxed),
            };
            if idle {
                pool.ready.entry(bytes).or_default().push(entry);
            } else {
                pool.pending.push((needed, bytes, entry));
            }
            if bytes >= POOL_SMALL_BYTES && pool.cached_bytes > cap {
                // This runtime's own cap (an explicit per-HAL budget): its LRU idle buffers go.
                let over = pool.cached_bytes - cap;
                pool.evict_lru(over, Evict::Cap, true);
            }
            pool.group
        };
        if bytes >= POOL_SMALL_BYTES
            && GROUP_CACHED[group].load(Ordering::Relaxed) > group_cap(group)
        {
            evict_group_to_cap(group);
        }
    }

    /// Release every buffer pooled by this runtime (`trim_device_memory`); returns the bytes
    /// released. Budget and allocation failures use [drain_all_pools] instead.
    pub(crate) fn drain_pool(&self) -> u64 {
        self.pool.lock().drain()
    }

    /// Z14 (`Hal::prefill_elem`): set aside a pooled buffer of `len` bytes for a later
    /// `alloc_elem_init` with `word`: one that is ready, else one still pending (its GPU users
    /// were encoded earlier, and the fill and every later use are ordered after them by the
    /// queue's hazard tracking). Never a new allocation. False if one is already set aside for
    /// (`len`, `word`), none is pooled, or the set-aside bytes would exceed `max_bytes`.
    pub(crate) fn prefill_reserve(&self, len: u64, word: u32, max_bytes: u64) -> bool {
        let completed = self.completed_seq.load(Ordering::Acquire);
        let mut pool = self.pool.lock();
        pool.reclaim(completed);
        let held: u64 = pool.prefilled.iter().map(|p| p.len).sum();
        if held + len > max_bytes
            || pool
                .prefilled
                .iter()
                .any(|p| p.len == len && p.word == word)
        {
            return false;
        }
        let buf = match pool.ready.get_mut(&len).and_then(Vec::pop) {
            Some(entry) => entry.buf,
            None => match pool.pending.iter().position(|p| p.1 == len) {
                Some(i) => pool.pending.swap_remove(i).2.buf,
                None => return false,
            },
        };
        if pool.ready.get(&len).is_some_and(Vec::is_empty) {
            pool.ready.remove(&len);
        }
        // Still pool memory: `cached_bytes` keeps counting it.
        pool.prefilled.push(Prefilled {
            len,
            word,
            buf,
            filled: false,
            marked_by: 0,
        });
        true
    }

    /// Z14: the set-aside buffers whose fill is not encoded yet, at most `max`, now marked filled:
    /// the caller encodes their fills (len / 16 uint4s of `word`) in its next dispatch.
    ///
    /// Assumes one thread drives the HAL (true for every caller today: rv32im witgen/finalize and
    /// the recursion HALs): a `prefill_take` on another thread between this call and the caller's
    /// encode would see `filled` for a fill that is not in the command stream yet. W7-A1: each
    /// buffer records the thread that marked it, and [Self::prefill_take] checks it.
    pub(crate) fn prefill_unfilled(&self, max: usize) -> Vec<(MtlBuffer, u64, u32)> {
        let me = thread_token();
        let mut pool = self.pool.lock();
        pool.prefilled
            .iter_mut()
            .filter(|p| !p.filled)
            .take(max)
            .map(|p| {
                p.filled = true;
                p.marked_by = me;
                (p.buf.clone(), p.len, p.word)
            })
            .collect()
    }

    /// Z14: take the buffer set aside for (`len`, `word`), with whether its fill is encoded.
    pub(crate) fn prefill_take(&self, len: u64, word: u32) -> Option<(MtlBuffer, bool)> {
        let mut pool = self.pool.lock();
        let i = pool
            .prefilled
            .iter()
            .position(|p| p.len == len && p.word == word)?;
        let p = pool.prefilled.swap_remove(i);
        pool.sub_cached(len);
        pool.handed_out(len);
        COUNTERS.hits.fetch_add(1, Ordering::Relaxed);
        drop(pool);
        // Z14 single-thread assumption (W7-A1): a buffer marked filled on another thread may be
        // taken before that thread encoded the fill. One thread drives a HAL today; catch a
        // change of that in debug builds, and say so once in release builds.
        if p.filled && p.marked_by != thread_token() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::error!(
                    "Metal prefill: a buffer marked filled on another thread was taken; \
                     Hal::prefill_elem assumes one thread drives a HAL"
                )
            });
            debug_assert!(
                false,
                "Metal prefill: a buffer marked filled on another thread was taken \
                 (Hal::prefill_elem assumes one thread drives a HAL)"
            );
        }
        Some((p.buf, p.filled))
    }

    /// Z14 test hook: (set-aside buffers, of which fill-encoded).
    #[cfg(test)]
    pub(crate) fn prefill_stats(&self) -> (usize, usize) {
        let pool = self.pool.lock();
        let filled = pool.prefilled.iter().filter(|p| p.filled).count();
        (pool.prefilled.len(), filled)
    }

    /// Move this runtime's pool into accounting group `group` (tests; see [POOL_GROUPS]).
    #[cfg(test)]
    pub(crate) fn set_pool_group(&self, group: usize) {
        let mut pool = self.pool.lock();
        let bytes = pool.cached_bytes;
        GROUP_CACHED[pool.group].fetch_sub(bytes, Ordering::Relaxed);
        pool.group = group % POOL_GROUPS;
        GROUP_CACHED[pool.group].fetch_add(bytes, Ordering::Relaxed);
    }

    /// Test hook: this pool's staleness horizon in batches.
    #[cfg(test)]
    pub(crate) fn set_stale_batches(&self, k: u64) {
        self.pool.lock().stale_k = k;
    }

    /// Test hook: set (`Some`) or clear the cap of accounting group `group`.
    #[cfg(test)]
    pub(crate) fn set_group_cap(group: usize, cap: Option<u64>) {
        GROUP_CAP[group % POOL_GROUPS].store(cap.unwrap_or(0), Ordering::Relaxed);
    }

    /// Test hook: pooled bytes of accounting group `group`.
    #[cfg(test)]
    pub(crate) fn group_cached(group: usize) -> u64 {
        GROUP_CACHED[group % POOL_GROUPS].load(Ordering::Relaxed)
    }

    /// Test hook: a pool accounting group of its own (1..[POOL_GROUPS], round robin).
    #[cfg(test)]
    pub(crate) fn new_test_pool_group() -> usize {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        1 + NEXT.fetch_add(1, Ordering::Relaxed) % (POOL_GROUPS - 1)
    }

    /// Test hook: pretend the system-wide GPU memory in use is `bytes` (`None`: read it).
    #[cfg(test)]
    pub(crate) fn set_fake_system_in_use(&self, bytes: Option<u64>) {
        self.fake_system_in_use
            .store(bytes.unwrap_or(0), Ordering::Relaxed);
    }

    /// Test hook: (pooled bytes, live bytes, live high-water mark) of this runtime.
    #[cfg(test)]
    pub(crate) fn pool_bytes(&self) -> (u64, u64, u64) {
        let pool = self.pool.lock();
        (pool.cached_bytes, pool.live, pool.live_peak)
    }

    /// Test hook: the pooled sizes and their buffer counts (ready and pending).
    #[cfg(test)]
    pub(crate) fn pool_sizes(&self) -> Vec<(u64, usize)> {
        let pool = self.pool.lock();
        let mut v: Vec<(u64, usize)> = pool.ready.iter().map(|(&l, e)| (l, e.len())).collect();
        v.extend(pool.pending.iter().map(|p| (p.1, 1)));
        v.sort();
        v
    }

    #[cfg(test)]
    pub(crate) fn pool_stats(&self) -> (usize, usize) {
        let pool = self.pool.lock();
        (pool.ready.values().map(Vec::len).sum(), pool.pending.len())
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        if std::env::var_os(POOL_TRACE_ENV).is_some() {
            eprint!("[pool] runtime drop\n{}", pool_report());
        }
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
