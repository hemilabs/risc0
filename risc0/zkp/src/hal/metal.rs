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

//! Metal HAL (plan `docs/metal-port/PLAN.md` sections 2.6, 2.7, 2.9).
//!
//! * PSOs for every zkp kernel are built once per process ([MetalHal::new]); circuit metallibs are
//!   loaded with [MetalHal::load_library].
//! * GPU work is batched: [MetalHal::dispatch] / [MetalHal::launch] / [MetalHal::blit_fill] only
//!   encode; [MetalHal::flush] commits, [MetalHal::sync] commits and waits. Host access to a
//!   [BufferImpl] (`view`, `view_mut`, `get_at`, `to_vec`) syncs first. Raw pointers
//!   ([BufferImpl::as_ptr]) are not synchronised; `RISC0_METAL_DEBUG=1` asserts they are not
//!   taken on a buffer an unsynced batch uses.
//! * Buffers come from a size-keyed pool; a dropped buffer is reused only after every batch that
//!   may use it completed.
//! * [exclusive] / [singleton] are taken per call (around every encode, commit and wait), never
//!   stored in a HAL.
//! * A [BufferImpl] may only be used (bound to a dispatch, read, dropped) through the HAL that
//!   allocated it: host reads sync, and pool returns wait for, only that HAL's batches. Binding a
//!   buffer of HAL A into a dispatch on HAL B can give stale reads and early pool reuse.

pub mod archive;
mod buffer;
mod eval;
mod kernels;
mod mdiv;
mod mix;
mod ntt;
mod poseidon2;
mod poseidon254;
mod query;
mod runtime;
mod small;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_z1;
#[cfg(test)]
mod tests_z2;
#[cfg(test)]
mod tests_z3;
#[cfg(test)]
mod tests_z4;
#[cfg(test)]
mod tests_z5;
#[cfg(test)]
mod tests_z7;
#[cfg(test)]
mod tests_z8b;
#[cfg(test)]
mod tests_r17;
#[cfg(test)]
mod tests_z9;
#[cfg(test)]
mod tests_z15;
#[cfg(test)]
mod tests_z10;
#[cfg(test)]
mod tests_z13;
#[cfg(test)]
mod tests_z14;
#[cfg(test)]
mod tests_z16;

use std::{
    fmt::Debug,
    sync::{Arc, OnceLock},
};

use anyhow::Result;
use metal::{
    Buffer as MtlBuffer, BufferRef, CommandQueue, ComputeCommandEncoderRef,
    ComputePipelineDescriptor, ComputePipelineDescriptorRef, ComputePipelineState,
    ComputePipelineStateRef, Device, MTLSize,
};
use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
use risc0_core::{
    field::{
        baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
        ExtElem,
    },
    scope,
};

use self::runtime::Runtime;
pub use self::{
    buffer::BufferImpl,
    kernels::{MetalKernels, ZKP_ARCHIVE_ENV, ZKP_ARCHIVE_UNIT},
    poseidon2::{MetalHashPoseidon2, Poseidon2Config, Poseidon2Kernels},
    poseidon254::{MetalHashPoseidon254, Poseidon254Config, Poseidon254Kernels},
    runtime::{
        as_bytes, BudgetScope, Grid, KernelArg, BUDGET_SCOPE_ENV, DEFAULT_BATCH_DISPATCHES,
        MAX_SET_BYTES, MAX_WORKING_SET_ENV,
    },
};
pub use self::runtime::{
    profile_report, profile_take, sync_trace_take, ProfileEntry, SyncRecord, PROFILE_ENV,
    SYNC_TRACE_ENV,
};
use super::{Buffer as _, Hal};
use crate::{
    core::{
        digest::Digest,
        hash::{sha::Sha256HashSuite, HashSuite},
    },
    FRI_FOLD,
};

const METAL_LIB: &[u8] = include_bytes!(env!("ZKP_METAL_PATH"));

/// Every kernel of the zkp metallib (one group per `sys/kernels/zkp/metal/*.metal` file, in the
/// order of `sys/build.rs`); a PSO is built for each when the first HAL is created. The legacy
/// kernels (`multi_ntt_*`, `multi_poly_eval`, `batch_expand`, `zk_shift`, `multi_bit_reverse`,
/// `poseidon2_*`) and `eltwise_zeroize_fpext` were retired in Z7.
pub const KERNEL_NAMES: &[&str] = &[
    // eltwise.metal
    "eltwise_add_fp",
    "eltwise_copy_fp",
    "eltwise_mul_factor_fp",
    "eltwise_sum_fpext",
    "eltwise_zeroize_fp",
    // fri.metal
    "fri_fold",
    // mix.metal (Z4)
    "mix_poly_coeffs",
    "mix_poly_coeffs_v1",
    "mix_poly_coeffs_v2",
    "mix_poly_coeffs_v2_dcids",
    "mix_poly_coeffs_fused",
    // sha.metal
    "sha_fold",
    "sha_rows",
    // misc.metal (Z1)
    "combos_sub",
    "eltwise_copy_fp_region",
    "fill_u32",
    "gather_sample",
    "scatter",
    "scatter_bits",
    // ntt_x.metal (Z2; also `ntt::NTT_X_KERNELS`)
    "bit_rev_permutation_batched",
    "bit_rev_z_batched_64",
    "ct_x_1",
    "ct_x_8",
    "ct_x_8_coalesced",
    "gs_x_1",
    "gs_x_8",
    "gs_x_8_coalesced",
    "nttx_zk_shift_batched",
    // pv.metal (Z3)
    "pv_fold_k",
    "pv_fold_k_m32",
    "pv_fold_tail_k",
    "pv_fold_tail_k_m32",
    "pv_rows_k",
    "pv_rows_k_m32",
    "pvs_q_fold_k",
    "pvs_q_fold_k_m32",
    "pvs_q_rows_k",
    "pvs_q_rows_k_m32",
    "pvs_w_fold_k",
    "pvs_w_fold_k_m32",
    "pvs_w_rows_k",
    "pvs_w_rows_k_m32",
    // eval.metal (Z4)
    "batch_evaluate_any",
    "eu2_pows",
    "eu2_main",
    "eu2_main_lz",
    "eu2_reduce",
    // mdiv.metal (Z5)
    "mdiv_agg0_e8",
    "mdiv_agg0_e16",
    "mdiv_apply_e8",
    "mdiv_apply_e16",
    "mdiv_scan_32",
    "mdiv_scan_64",
    "mdiv_scan_128",
    "mdiv_scan_256",
    "mdiv_scan_512",
    "mdiv_scan_1024",
    "mdiv_final",
    // pv.metal `_z9` instantiations (Z9)
    "pv_fold_k_z9",
    "pv_fold_tail_k_z9",
    "pv_rows_k_z9",
    // pv.metal: the row hash with a commit's side work (bit reversal, fills) folded in (Z14)
    "pv_rows_side_k_z9",
    // poseidon254.metal (Z10)
    "p254_fold_ref",
    "p254_fold_w64",
    "p254_fold_w64c",
    "p254_rows_ref",
    "p254_rows_w64",
    "p254_rows_w64c",
];

/// `RISC0_METAL_PREFILL=0` turns [MetalHal]'s `Hal::prefill_elem` off (Z14).
fn prefill_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RISC0_METAL_PREFILL").map_or(true, |v| v.trim() != "0"))
}

/// Whether the Metal backend may be used for `unit` (one of `zkp`, `rv32im`, `recursion`,
/// `keccak`, `groth16`): false if `RISC0_METAL=0`, or if the comma-separated list in
/// `RISC0_METAL_SKIP` names the unit. Callers fall back to the CPU prover (groth16: Docker).
pub fn enabled(unit: &str) -> bool {
    if std::env::var("RISC0_METAL").is_ok_and(|v| v.trim() == "0") {
        return false;
    }
    match std::env::var("RISC0_METAL_SKIP") {
        Ok(skip) => !skip
            .split(',')
            .any(|name| name.trim().eq_ignore_ascii_case(unit)),
        Err(_) => true,
    }
}

/// The name of the system default Metal device, or `None` if there is none.
pub fn device_name() -> Option<String> {
    Device::system_default().map(|device| device.name().to_string())
}

/// The working-set limit and the device's `currentAllocatedSize` in bytes, without making a HAL
/// (item Z12): `RISC0_METAL_MAX_WORKING_SET_BYTES` when set and smaller, else
/// `recommendedMaxWorkingSetSize` (what [MetalHal::working_set_limit] reports for a HAL whose
/// budget was not overridden), and every Metal allocation of this process on the device. `None`
/// without a Metal device.
pub fn device_memory() -> Option<(u64, u64)> {
    let device = Device::system_default()?;
    let recommended = device.recommended_max_working_set_size();
    let limit = match runtime::env_working_set_limit() {
        Some(budget) => budget.min(recommended),
        None => recommended,
    };
    Some((limit, device.current_allocated_size()))
}

/// The process-wide GPU lock (the GPU becomes unstable as the number of concurrent provers grows).
///
/// Taken per call: around every encode, commit and wait of every `MetalHal`, and by code that
/// wants a whole proving call to run without interleaving with other threads (`exclusive`). It is
/// never stored in a HAL, so HALs cached per thread do not block other threads.
pub fn singleton() -> &'static ReentrantMutex<()> {
    static ONCE: OnceLock<ReentrantMutex<()>> = OnceLock::new();
    ONCE.get_or_init(|| ReentrantMutex::new(()))
}

/// Exclusive use of the GPU against other threads for as long as the guard lives. Reentrant.
///
/// Reentrant only on the owning thread: while a thread holds this guard, a HAL call made from
/// another thread (e.g. a rayon worker spawned by `rayon::join` / `par_iter` under the guard)
/// blocks until the guard drops, which deadlocks if the holder waits for that worker. Do not
/// call the HAL from rayon workers while holding `exclusive()`.
pub fn exclusive() -> ReentrantMutexGuard<'static, ()> {
    singleton().lock()
}

/// Test-only lock for tests that read or change process-wide state (other HALs' pools, the
/// device-wide allocation total, working-set budgets, the peak counter): plan 8.7. Poison-tolerant,
/// so one failing test does not fail every later one.
#[cfg(test)]
pub(crate) fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The zkp PSOs, built once per process.
fn zkp_kernels(device: &Device) -> &'static MetalKernels {
    static ONCE: OnceLock<MetalKernels> = OnceLock::new();
    ONCE.get_or_init(|| {
        MetalKernels::new_zkp(device, METAL_LIB, KERNEL_NAMES)
            .unwrap_or_else(|e| panic!("zkp metallib: {e:#}"))
    })
}

/// Anything `dispatch_with_resources` accepts as a kernel: a PSO, or (deprecated) a pipeline
/// descriptor, for which a PSO is built on every call.
pub trait PipelineSource {
    fn with_pso(&self, device: &Device, f: &mut dyn FnMut(&ComputePipelineStateRef));
}

impl PipelineSource for ComputePipelineStateRef {
    fn with_pso(&self, _device: &Device, f: &mut dyn FnMut(&ComputePipelineStateRef)) {
        f(self)
    }
}

impl PipelineSource for ComputePipelineState {
    fn with_pso(&self, device: &Device, f: &mut dyn FnMut(&ComputePipelineStateRef)) {
        (**self).with_pso(device, f)
    }
}

impl PipelineSource for ComputePipelineDescriptor {
    fn with_pso(&self, device: &Device, f: &mut dyn FnMut(&ComputePipelineStateRef)) {
        (**self).with_pso(device, f)
    }
}

impl PipelineSource for ComputePipelineDescriptorRef {
    fn with_pso(&self, device: &Device, f: &mut dyn FnMut(&ComputePipelineStateRef)) {
        let pso = device
            .new_compute_pipeline_state(self)
            .unwrap_or_else(|e| panic!("compute pipeline: {e}"));
        f(&pso)
    }
}

pub trait MetalHash: Sized {
    /// Create a hash implementation
    fn new(hal: &MetalHal<Self>) -> Self;

    /// Run the hash_fold function (encode only)
    fn hash_fold(&self, hal: &MetalHal<Self>, io: &BufferImpl<Digest>, output_size: usize);

    /// Run the hash_rows function (encode only)
    fn hash_rows(
        &self,
        hal: &MetalHal<Self>,
        output: &BufferImpl<Digest>,
        matrix: &BufferImpl<BabyBearElem>,
    );

    /// Fold a whole tree of `2^layers` leaves stored at `io[2^layers..2^(layers+1)]`, every level
    /// encoded into the same batch (encode only; dependent levels are ordered by the serial
    /// encoder and hazard tracking).
    fn hash_fold_tree(&self, hal: &MetalHal<Self>, io: &BufferImpl<Digest>, layers: usize) {
        for i in (0..layers).rev() {
            self.hash_fold(hal, io, 1 << i);
        }
    }

    /// Return the HashSuite
    fn get_hash_suite(&self) -> &HashSuite<BabyBear>;

    /// Z14: whether this hash's row kernel encodes the fills of `Hal::prefill_elem` buffers
    /// (else `prefill_elem` sets nothing aside).
    fn carries_fills(&self) -> bool {
        false
    }

    /// Z14: whether [Self::hash_rows_bit_reverse] fuses for a `rows`-row matrix and `cols`
    /// columns of `2^lg` coefficients.
    fn fuses_bit_reverse(&self, _rows: usize, _cols: usize, _lg: usize) -> bool {
        false
    }

    /// Z14: encode `hash_rows` with the in-place bit reversal of the `count` columns of `coeffs`
    /// folded into the hash kernel; false (nothing encoded) when it does not fuse that shape.
    fn hash_rows_bit_reverse(
        &self,
        _hal: &MetalHal<Self>,
        _output: &BufferImpl<Digest>,
        _matrix: &BufferImpl<BabyBearElem>,
        _coeffs: &BufferImpl<BabyBearElem>,
        _count: usize,
    ) -> bool {
        false
    }
}

pub struct MetalHashSha256 {
    suite: HashSuite<BabyBear>,
}

impl MetalHash for MetalHashSha256 {
    fn new(_hal: &MetalHal<Self>) -> Self {
        MetalHashSha256 {
            suite: Sha256HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, hal: &MetalHal<Self>, io: &BufferImpl<Digest>, output_size: usize) {
        let n = u32::try_from(output_size).expect("hash_fold: too many outputs");
        let args = &[
            io.as_arg_with_offset(output_size),
            io.as_arg_with_offset(output_size * 2),
            KernelArg::Integer(n),
        ];
        hal.launch("sha_fold", output_size as u64, args);
    }

    fn hash_rows(
        &self,
        hal: &MetalHal<Self>,
        output: &BufferImpl<Digest>,
        matrix: &BufferImpl<BabyBearElem>,
    ) {
        let row_size = output.size();
        if row_size == 0 {
            return;
        }
        let col_size = matrix.size() / row_size;
        assert_eq!(matrix.size(), col_size * row_size);
        let rows = u32::try_from(row_size).expect("hash_rows: too many rows");
        let cols = u32::try_from(col_size).expect("hash_rows: too many columns");
        let args = &[
            output.as_arg(),
            matrix.as_arg(),
            KernelArg::Integer(rows),
            KernelArg::Integer(cols),
        ];
        hal.launch("sha_rows", row_size as u64, args);
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

pub struct MetalHal<MH: MetalHash> {
    pub device: Device,
    pub cmd_queue: CommandQueue,
    rt: Arc<Runtime>,
    kernels: &'static MetalKernels,
    hash: Option<Box<MH>>,
    ntt: ntt::NttState,
}

impl<MH: MetalHash> Debug for MetalHal<MH> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalHal")
            .field("device", &self.device.name())
            .finish_non_exhaustive()
    }
}

pub type MetalHalSha256 = MetalHal<MetalHashSha256>;
pub type MetalHalPoseidon2 = MetalHal<MetalHashPoseidon2>;
pub type MetalHalPoseidon254 = MetalHal<MetalHashPoseidon254>;

impl<MH: MetalHash> Default for MetalHal<MH> {
    fn default() -> Self {
        Self::new()
    }
}

impl<MH: MetalHash> MetalHal<MH> {
    /// A HAL on the system default device, with its own command queue and batch. The zkp PSOs are
    /// built on the first call in the process and shared afterwards. No lock is kept.
    pub fn new() -> Self {
        let _lock = exclusive();
        let device = Device::system_default().expect("no Metal device found");
        let kernels = zkp_kernels(&device);
        let rt = Arc::new(Runtime::new(device.clone()));
        let cmd_queue = rt.queue.clone();
        let mut hal = Self {
            device,
            cmd_queue,
            rt,
            kernels,
            hash: None,
            ntt: ntt::NttState::new(),
        };
        hal.hash = Some(Box::new(MH::new(&hal)));
        hal
    }

    fn hash(&self) -> &MH {
        self.hash.as_ref().unwrap()
    }

    /// Load a circuit metallib and build a PSO for each of `names`; panics on a missing name.
    pub fn load_library(&self, bytes: &[u8], names: &[&str]) -> MetalKernels {
        self.try_load_library(bytes, names)
            .unwrap_or_else(|e| panic!("{e:#}"))
    }

    /// [Self::load_library], returning the error instead of panicking.
    pub fn try_load_library(&self, bytes: &[u8], names: &[&str]) -> Result<MetalKernels> {
        MetalKernels::new(&self.device, bytes, names)
    }

    /// The PSO of a zkp kernel.
    pub fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.kernels.pipeline(name)
    }

    /// Encode one dispatch into the open batch. Argument `i` binds index `i`; `tg_bytes[i]` sets
    /// threadgroup memory index `i` (rounded up to 16; the total, with the PSO's static usage,
    /// must fit the device's 32 KB). Empty grids are skipped.
    pub fn dispatch(
        &self,
        pso: &ComputePipelineStateRef,
        args: &[KernelArg],
        grid: Grid,
        tg_bytes: &[u64],
    ) {
        self.rt.dispatch(pso, args, grid, tg_bytes, None);
    }

    /// 1-D dispatch of a zkp kernel over `count` threads.
    pub fn launch(&self, name: &str, count: u64, args: &[KernelArg]) {
        self.dispatch(self.pipeline(name), args, Grid::Threads(count), &[]);
    }

    /// Encode a blit fill of `len` bytes at `offset` with `byte`.
    pub fn blit_fill(&self, buf: &BufferRef, offset: u64, len: u64, byte: u8) {
        self.rt.blit_fill(buf, offset, len, byte);
    }

    /// A GPU-only (`StorageModePrivate`) buffer, not pooled. Panics if allocation fails after
    /// draining the pool.
    pub fn alloc_private(&self, bytes: u64) -> MtlBuffer {
        self.rt
            .alloc_private(bytes)
            .unwrap_or_else(|e| panic!("{e:#}"))
    }

    /// A pooled shared buffer of `size` elements of any `T`.
    pub fn alloc_buffer<T>(&self, name: &'static str, size: usize) -> BufferImpl<T> {
        BufferImpl::new(&self.rt, name, size)
    }

    /// A pooled shared buffer initialised from `slice`.
    pub fn copy_from_slice<T>(&self, name: &'static str, slice: &[T]) -> BufferImpl<T> {
        BufferImpl::copy_from(&self.rt, name, slice)
    }

    /// End encoding and commit the open batch, without waiting.
    pub fn flush(&self) {
        self.rt.flush();
    }

    /// Flush and wait for all committed work; panics if a command buffer reported an error.
    pub fn sync(&self) {
        self.rt.sync();
    }

    /// [Self::sync], returning a command-buffer error instead of panicking (plan 2.9).
    pub fn try_sync(&self) -> Result<()> {
        self.rt.try_sync()
    }

    /// Escape hatch: run `f` on the open batch's compute encoder (counts as one dispatch). `f`
    /// must not call back into this HAL, and buffers it binds are not tracked by the debug check.
    /// Document every use.
    pub fn with_encoder(&self, f: impl FnOnce(&ComputeCommandEncoderRef)) {
        self.rt.with_compute(|enc, bound, _| {
            f(enc);
            // `f` may have changed any encoder state.
            *bound = Default::default();
        });
    }

    /// Commit a batch after this many dispatches (default [DEFAULT_BATCH_DISPATCHES]).
    pub fn set_batch_limit(&self, dispatches: usize) {
        self.rt.set_batch_limit(dispatches);
    }

    /// The working-set limit in bytes: `RISC0_METAL_MAX_WORKING_SET_BYTES` (or
    /// [Self::set_working_set_limit]) when set and smaller, else `recommendedMaxWorkingSetSize`.
    /// `gpu_total_memory` reports it.
    pub fn working_set_limit(&self) -> u64 {
        self.rt.working_set_limit()
    }

    /// Set (`Some`) or clear (`None`) an explicit working-set budget for this HAL, overriding
    /// `RISC0_METAL_MAX_WORKING_SET_BYTES`: allocations that would take the device's
    /// `currentAllocatedSize` above it fail early (after draining the pool).
    pub fn set_working_set_limit(&self, bytes: Option<u64>) {
        self.rt.set_working_set_limit(bytes);
    }

    /// What an explicit budget is checked against ([BUDGET_SCOPE_ENV], default
    /// [BudgetScope::Device]).
    pub fn budget_scope(&self) -> BudgetScope {
        self.rt.budget_scope()
    }

    /// Override [BUDGET_SCOPE_ENV] for this HAL.
    pub fn set_budget_scope(&self, scope: BudgetScope) {
        self.rt.set_budget_scope(scope);
    }

    /// Bytes this HAL allocated and has not freed (live and pooled shared buffers, plus private
    /// buffers until the HAL drops): what a [BudgetScope::Hal] budget is checked against.
    pub fn owned_allocated_size(&self) -> u64 {
        self.rt.owned_bytes()
    }

    /// The device's `currentAllocatedSize` (all Metal memory of this process on the device).
    pub fn allocated_size(&self) -> u64 {
        self.rt.allocated_size()
    }

    /// Highest `currentAllocatedSize` observed after any Metal allocation in this process since
    /// the last [Self::reset_peak_allocated_size] (process-wide, covers every HAL).
    pub fn peak_allocated_size(&self) -> u64 {
        Runtime::peak_allocated()
    }

    /// Restart [Self::peak_allocated_size] from the current allocation level.
    pub fn reset_peak_allocated_size(&self) {
        self.rt.reset_peak_allocated();
    }

    /// Turn the `RISC0_METAL_DEBUG` host-access check on or off for this HAL.
    pub fn set_debug(&self, on: bool) {
        self.rt.set_debug(on);
    }

    /// Source-compatible: dispatch a zkp kernel by name over `count` threads (encode only).
    pub fn dispatch_by_name(&self, name: &str, args: &[KernelArg], count: u64) {
        self.launch(name, count, args);
    }

    /// Source-compatible with the pre-batch HAL: encodes `callback` (e.g. `use_resource`) and the
    /// dispatch, then syncs before returning. `opts` = (threadgroups, threads per threadgroup);
    /// `None` dispatches `count` threads.
    ///
    /// `callback` runs while this HAL's internal state lock is held: it must only call methods
    /// on the encoder (e.g. `use_resource`) and must not call back into this HAL (flush, sync,
    /// dispatch, alloc, host reads), or it deadlocks.
    #[deprecated(note = "use dispatch(); this syncs on every call")]
    pub fn dispatch_with_resources<K, F>(
        &self,
        kernel: &K,
        args: &[KernelArg],
        count: u64,
        opts: Option<(MTLSize, MTLSize)>,
        callback: F,
    ) where
        K: PipelineSource + ?Sized,
        F: Fn(&ComputeCommandEncoderRef),
    {
        let grid = match opts {
            Some((groups, threads)) => Grid::Groups { groups, threads },
            None => Grid::Threads(count),
        };
        kernel.with_pso(&self.device, &mut |pso| {
            self.rt.dispatch(pso, args, grid, &[], Some(&callback));
        });
        self.sync();
    }
}

impl<MH: MetalHash> Hal for MetalHal<MH> {
    type Elem = BabyBearElem;
    type ExtElem = BabyBearExtElem;
    type Field = BabyBear;
    type Buffer<T: Clone + Debug + PartialEq> = BufferImpl<T>;

    fn alloc_elem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::Elem> {
        self.alloc_buffer(name, size)
    }

    fn alloc_elem_init(
        &self,
        name: &'static str,
        size: usize,
        value: Self::Elem,
    ) -> Self::Buffer<Self::Elem> {
        // Z14: a buffer `prefill_elem` set aside (its fill encoded in a row hash, or encoded now).
        let word = value.as_u32_montgomery();
        let bytes = (size * std::mem::size_of::<Self::Elem>()) as u64;
        if let Some((buf, filled)) = self.rt.prefill_take(bytes, word) {
            let raw = buffer::RawBuffer::from_pooled(&self.rt, name, buf, bytes);
            let out = BufferImpl::from_raw(raw, size);
            if !filled {
                self.fill_u32(&out, word);
            }
            return out;
        }
        self.small_alloc_elem_init(name, size, value)
    }

    /// Z14: set aside a pooled buffer of `size` elements (one just freed, never a new
    /// allocation) for an upcoming `alloc_elem_init(.., size, value)`; its fill rides on the next
    /// commit row hash (`pv_rows_side_k_z9`) instead of a blit. Off with
    /// `RISC0_METAL_PREFILL=0`, with a hash that does not carry fills, and beyond an eighth of
    /// the working-set limit in set-aside bytes (a pool drain releases them).
    fn prefill_elem(&self, size: usize, value: Self::Elem) {
        let bytes = (size * std::mem::size_of::<Self::Elem>()) as u64;
        if !prefill_enabled()
            || !self.hash().carries_fills()
            || bytes == 0
            || !bytes.is_multiple_of(16)
        {
            return;
        }
        let max = self.rt.working_set_limit() / 8;
        self.rt.prefill_reserve(bytes, value.as_u32_montgomery(), max);
    }

    fn alloc_extelem_zeroed(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem> {
        self.small_alloc_extelem_zeroed(name, size)
    }

    fn copy_from_elem(&self, name: &'static str, slice: &[Self::Elem]) -> Self::Buffer<Self::Elem> {
        self.copy_from_slice(name, slice)
    }

    fn alloc_extelem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem> {
        self.alloc_buffer(name, size)
    }

    fn copy_from_extelem(
        &self,
        name: &'static str,
        slice: &[Self::ExtElem],
    ) -> Self::Buffer<Self::ExtElem> {
        self.copy_from_slice(name, slice)
    }

    fn alloc_digest(&self, name: &'static str, size: usize) -> Self::Buffer<Digest> {
        self.alloc_buffer(name, size)
    }

    fn copy_from_digest(&self, name: &'static str, slice: &[Digest]) -> Self::Buffer<Digest> {
        self.copy_from_slice(name, slice)
    }

    fn alloc_u32(&self, name: &'static str, size: usize) -> Self::Buffer<u32> {
        self.alloc_buffer(name, size)
    }

    fn copy_from_u32(&self, name: &'static str, slice: &[u32]) -> Self::Buffer<u32> {
        self.copy_from_slice(name, slice)
    }

    fn batch_bit_reverse(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        self.ntt_batch_bit_reverse(io, count);
    }

    fn batch_expand_into_evaluate_ntt(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
        count: usize,
        _expand_bits: usize,
    ) {
        self.ntt_batch_expand_into_evaluate(output, input, count);
    }

    fn batch_interpolate_ntt(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        self.ntt_batch_interpolate(io, count);
    }

    fn batch_interpolate_ntt_zk_shift(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        self.ntt_batch_interpolate_zk_shift(io, count);
    }

    fn batch_interpolate_ntt_zk_shift_from(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        count: usize,
    ) {
        self.ntt_batch_interpolate_zk_shift_from(dst, src, count);
    }

    fn has_commit_coeffs_lde(&self, po2: usize) -> bool {
        self.ntt_has_commit_coeffs_lde(po2)
    }

    fn commit_coeffs_lde(
        &self,
        coeffs: &Self::Buffer<Self::Elem>,
        evaluated: &Self::Buffer<Self::Elem>,
        witness: &Self::Buffer<Self::Elem>,
        count: usize,
    ) {
        self.ntt_commit_coeffs_lde(coeffs, evaluated, witness, count);
    }

    fn batch_evaluate_any(
        &self,
        coeffs: &Self::Buffer<Self::Elem>,
        poly_count: usize,
        which: &Self::Buffer<u32>,
        xs: &Self::Buffer<Self::ExtElem>,
        out: &Self::Buffer<Self::ExtElem>,
    ) {
        scope!("batch_evaluate_any");
        eval::batch_evaluate_any(self, coeffs, poly_count, which, xs, out, None);
    }

    fn eltwise_add_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input1: &Self::Buffer<Self::Elem>,
        input2: &Self::Buffer<Self::Elem>,
    ) {
        assert_eq!(output.size(), input1.size());
        assert_eq!(output.size(), input2.size());
        let count = output.size() as u64;
        let args = &[output.as_arg(), input1.as_arg(), input2.as_arg()];
        self.launch("eltwise_add_fp", count, args);
    }

    fn eltwise_sum_extelem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::ExtElem>,
    ) {
        self.small_eltwise_sum_extelem(output, input);
    }

    fn eltwise_copy_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
    ) {
        assert_eq!(output.size(), input.size());
        let count = output.size() as u64;
        let args = &[output.as_arg(), input.as_arg()];
        self.launch("eltwise_copy_fp", count, args);
    }

    fn fri_fold(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
        mix: &Self::ExtElem,
    ) {
        let count = output.size() / Self::ExtElem::EXT_SIZE;
        assert_eq!(output.size(), count * Self::ExtElem::EXT_SIZE);
        assert_eq!(input.size(), output.size() * FRI_FOLD);
        let count32 = u32::try_from(count).expect("fri_fold: too many outputs");
        let mix = [*mix];
        let args = &[
            output.as_arg(),
            input.as_arg(),
            KernelArg::Bytes(as_bytes(&mix)),
            KernelArg::Integer(count32),
        ];
        self.launch("fri_fold", count as u64, args);
    }

    fn mix_poly_coeffs(
        &self,
        output: &Self::Buffer<Self::ExtElem>,
        mix_start: &Self::ExtElem,
        mix: &Self::ExtElem,
        input: &Self::Buffer<Self::Elem>,
        combos: &Self::Buffer<u32>,
        input_size: usize,
        count: usize,
    ) {
        tracing::debug!(
            "output: {}, input: {}, combos: {}, input_size: {input_size}, count: {count}",
            output.size(),
            input.size(),
            combos.size(),
        );

        eval::mix_poly_coeffs(
            self, output, mix_start, mix, input, combos, input_size, count, None,
        );
    }

    fn has_mix_poly_coeffs_fused(&self) -> bool {
        mix::mix_fused_enabled()
    }

    #[allow(clippy::type_complexity)]
    fn mix_poly_coeffs_fused(
        &self,
        out: &Self::Buffer<Self::ExtElem>,
        mix: &Self::ExtElem,
        inputs: &[(&Self::Buffer<Self::Elem>, &[u32])],
        count: usize,
    ) {
        mix::mix_poly_coeffs_fused(self, out, mix, inputs, count, !mix::mix_fused_enabled());
    }

    fn hash_fold(&self, io: &Self::Buffer<Digest>, input_size: usize, output_size: usize) {
        // The CPU HAL's checks: the kernels read io[input_size..2 * input_size].
        assert!(
            io.size() >= 2 * input_size,
            "hash_fold: io holds {} digests, input_size {input_size} needs {}",
            io.size(),
            2 * input_size
        );
        assert_eq!(input_size, 2 * output_size);
        self.hash().hash_fold(self, io, output_size);
        self.flush();
    }

    fn hash_fold_tree(&self, io: &Self::Buffer<Digest>, layers: usize) {
        assert!(
            layers < usize::BITS as usize - 1 && io.size() >= 2 << layers,
            "hash_fold_tree: {} digests for {layers} layers",
            io.size()
        );
        self.hash().hash_fold_tree(self, io, layers);
        self.flush();
    }

    fn hash_rows(&self, output: &Self::Buffer<Digest>, matrix: &Self::Buffer<Self::Elem>) {
        self.hash().hash_rows(self, output, matrix);
        self.flush();
    }

    fn fuses_bit_reverse_into_hash(&self, rows: usize, cols: usize, lg: usize) -> bool {
        self.hash().fuses_bit_reverse(rows, cols, lg)
    }

    fn hash_rows_bit_reverse(
        &self,
        output: &Self::Buffer<Digest>,
        matrix: &Self::Buffer<Self::Elem>,
        coeffs: &Self::Buffer<Self::Elem>,
        count: usize,
    ) {
        if !self
            .hash()
            .hash_rows_bit_reverse(self, output, matrix, coeffs, count)
        {
            self.hash().hash_rows(self, output, matrix);
            self.ntt_batch_bit_reverse(coeffs, count);
        }
        self.flush();
    }

    fn commit_coeffs_lde_bit_reversed(
        &self,
        coeffs: &Self::Buffer<Self::Elem>,
        evaluated: &Self::Buffer<Self::Elem>,
        witness: &Self::Buffer<Self::Elem>,
        count: usize,
    ) {
        self.ntt_commit_coeffs_lde_impl(coeffs, evaluated, witness, count, false);
    }

    fn zk_shift(&self, io: &Self::Buffer<Self::Elem>, poly_count: usize) {
        self.ntt_zk_shift(io, poly_count);
    }

    fn gather_sample(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        idx: usize,
        size: usize,
        stride: usize,
    ) {
        // The CPU HAL panics on these; the kernel reads src[gid * stride + idx] (64-bit index math)
        // with no bounds check.
        assert!(
            dst.size() >= size,
            "gather_sample: dst holds {} elements, size {size}",
            dst.size()
        );
        if size == 0 {
            return;
        }
        let last = (size - 1)
            .checked_mul(stride)
            .and_then(|o| o.checked_add(idx))
            .expect("gather_sample: index overflow");
        assert!(
            last < src.size(),
            "gather_sample: reads src[{last}] of {} elements (idx {idx}, size {size}, stride \
             {stride})",
            src.size()
        );
        // The kernel computes the src index in 64 bits (Z8a), so `last` is bounded only by
        // src.size(); idx and stride are still passed as 32-bit kernel arguments.
        assert!(
            idx <= u32::MAX as usize && (size == 1 || stride <= u32::MAX as usize),
            "gather_sample: idx {idx} / stride {stride} exceed the kernel's 32-bit arguments"
        );
        let args = &[
            dst.as_arg(),
            src.as_arg(),
            KernelArg::Integer(idx as u32),
            KernelArg::Integer(stride as u32),
        ];
        self.launch("gather_sample", size as u64, args);
    }

    fn has_unified_memory(&self) -> bool {
        self.device.has_unified_memory()
    }

    fn batch_get_digest_at(&self, buf: &Self::Buffer<Digest>, indices: &[usize]) -> Vec<Digest> {
        query::batch_get_digest_at(buf, indices)
    }

    fn gather_multi(
        &self,
        jobs: &[super::GatherJob<'_, Self>],
    ) -> Option<Vec<(Vec<Self::Elem>, Vec<Digest>)>> {
        if !query::query_par_enabled() {
            return None;
        }
        let jobs: Vec<query::HostGather<'_, BabyBearElem>> = jobs
            .iter()
            .map(|j| query::HostGather {
                matrix: j.matrix,
                col_size: j.col_size,
                row_size: j.row_size,
                rows: j.rows,
                nodes: j.nodes,
                node_indices: j.node_indices,
            })
            .collect();
        Some(query::gather_multi(&jobs))
    }

    fn trim_device_memory(&self) {
        let released = self.rt.drain_pool();
        tracing::debug!(
            "trim_device_memory: released {:.2} GiB of pooled Metal buffers",
            released as f64 / (1u64 << 30) as f64
        );
    }

    fn gpu_total_memory(&self) -> usize {
        self.rt.working_set_limit() as usize
    }

    fn gpu_free_memory(&self) -> usize {
        self.rt.free_size() as usize
    }

    fn get_hash_suite(&self) -> &HashSuite<Self::Field> {
        self.hash().get_hash_suite()
    }

    fn prefix_products(&self, io: &Self::Buffer<Self::ExtElem>) {
        io.view_mut(|io| {
            for i in 1..io.len() {
                io[i] *= io[i - 1];
            }
        });
    }

    fn eltwise_zeroize_elem(&self, elems: &Self::Buffer<Self::Elem>) {
        self.small_eltwise_zeroize_elem(elems);
    }

    fn scatter(
        &self,
        into: &Self::Buffer<Self::Elem>,
        index: &[u32],
        offsets: &[u32],
        values: &[Self::Elem],
    ) {
        self.small_scatter(into, index, offsets, values);
    }

    fn scatter_bits(&self, into: &Self::Buffer<Self::Elem>, bit_data: &[u32], cycles: u32) {
        self.small_scatter_bits(into, bit_data, cycles);
    }

    fn combos_prepare(
        &self,
        combos: &Self::Buffer<Self::ExtElem>,
        coeff_u: &[Self::ExtElem],
        combo_count: usize,
        cycles: usize,
        reg_sizes: &[u32],
        reg_combo_ids: &[u32],
        mix: &Self::ExtElem,
    ) {
        self.small_combos_prepare(
            combos,
            coeff_u,
            combo_count,
            cycles,
            reg_sizes,
            reg_combo_ids,
            mix,
        );
    }

    fn eltwise_copy_elem_slice(
        &self,
        into: &Self::Buffer<Self::Elem>,
        from: &[Self::Elem],
        from_rows: usize,
        from_cols: usize,
        from_offset: usize,
        from_stride: usize,
        into_offset: usize,
        into_stride: usize,
    ) {
        self.small_eltwise_copy_elem_slice(
            into,
            from,
            from_rows,
            from_cols,
            from_offset,
            from_stride,
            into_offset,
            into_stride,
        );
    }

    fn combos_divide(
        &self,
        combos: &Self::Buffer<Self::ExtElem>,
        chunks: Vec<(usize, Vec<Self::ExtElem>)>,
        cycles: usize,
    ) {
        scope!("combos_divide");
        let remainders = self.mdiv_combos_divide(combos, &chunks, cycles);
        super::report_nonzero_remainders(&chunks, &remainders);
    }
}
