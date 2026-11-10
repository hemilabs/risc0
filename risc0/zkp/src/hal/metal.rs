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

mod buffer;
mod kernels;
mod mdiv;
mod runtime;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_z5;

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
use rayon::prelude::*;
use risc0_core::{
    field::{
        baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
        Elem, ExtElem, RootsOfUnity,
    },
    scope,
};

use self::runtime::Runtime;
pub use self::{
    buffer::BufferImpl,
    kernels::MetalKernels,
    runtime::{as_bytes, Grid, KernelArg, DEFAULT_BATCH_DISPATCHES, MAX_SET_BYTES},
};
use super::{Buffer as _, Hal};
use crate::{
    core::{
        digest::Digest,
        hash::{
            poseidon2::{self, Poseidon2HashSuite},
            sha::Sha256HashSuite,
            HashSuite,
        },
        log2_ceil,
    },
    FRI_FOLD,
};

const METAL_LIB: &[u8] = include_bytes!(env!("ZKP_METAL_PATH"));

/// Every kernel of the zkp metallib; a PSO is built for each when the first HAL is created.
pub const KERNEL_NAMES: &[&str] = &[
    "batch_expand",
    "eltwise_add_fp",
    "eltwise_copy_fp",
    "eltwise_mul_factor_fp",
    "eltwise_sum_fpext",
    "eltwise_zeroize_fp",
    "eltwise_zeroize_fpext",
    "fri_fold",
    "gather_sample",
    "mix_poly_coeffs",
    "multi_bit_reverse",
    "multi_ntt_fwd_step",
    "multi_ntt_rev_step",
    "multi_poly_eval",
    "poseidon2_fold",
    "poseidon2_rows",
    "sha_fold",
    "sha_rows",
    "zk_shift",
    // Z5: combos_divide (mdiv.metal)
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
];

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

/// The zkp PSOs, built once per process.
fn zkp_kernels(device: &Device) -> &'static MetalKernels {
    static ONCE: OnceLock<MetalKernels> = OnceLock::new();
    ONCE.get_or_init(|| {
        MetalKernels::new(device, METAL_LIB, KERNEL_NAMES)
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
        let args = &[
            io.as_arg_with_offset(output_size),
            io.as_arg_with_offset(output_size * 2),
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
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);
        // sha.metal declares `count`/`col_size` as writable `device uint32_t&`, and the
        // validation layer rejects `set_bytes` bound to a writable argument, so they go in a
        // buffer until the kernel takes them as `constant`/`const device` (zkp-hal unit).
        let params = hal.copy_from_u32("sha_rows_params", &[row_size as u32, col_size as u32]);
        let args = &[
            output.as_arg(),
            matrix.as_arg(),
            params.as_arg_with_offset(0),
            params.as_arg_with_offset(1),
        ];
        hal.launch("sha_rows", row_size as u64, args);
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

pub struct MetalHashPoseidon2 {
    suite: HashSuite<BabyBear>,
    round_constants: BufferImpl<BabyBearElem>,
    m_int_diag: BufferImpl<BabyBearElem>,
}

impl MetalHash for MetalHashPoseidon2 {
    fn new(hal: &MetalHal<Self>) -> Self {
        let round_constants =
            hal.copy_from_elem("round_constants", poseidon2::consts::ROUND_CONSTANTS);
        let m_int_diag = hal.copy_from_elem("m_int_diag", poseidon2::consts::M_INT_DIAG_HZN);
        MetalHashPoseidon2 {
            suite: Poseidon2HashSuite::new_suite(),
            round_constants,
            m_int_diag,
        }
    }

    fn hash_fold(&self, hal: &MetalHal<Self>, io: &BufferImpl<Digest>, output_size: usize) {
        let args = &[
            self.round_constants.as_arg(),
            self.m_int_diag.as_arg(),
            io.as_arg_with_offset(output_size),
            io.as_arg_with_offset(output_size * 2),
        ];
        hal.launch("poseidon2_fold", output_size as u64, args);
    }

    fn hash_rows(
        &self,
        hal: &MetalHal<Self>,
        output: &BufferImpl<Digest>,
        matrix: &BufferImpl<BabyBearElem>,
    ) {
        let row_size = output.size();
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);
        let args = &[
            self.round_constants.as_arg(),
            self.m_int_diag.as_arg(),
            output.as_arg(),
            matrix.as_arg(),
            KernelArg::Integer(row_size as u32),
            KernelArg::Integer(col_size as u32),
        ];
        hal.launch("poseidon2_rows", row_size as u64, args);
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
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << bits);
        tracing::debug!(
            "io: {}, count: {count}, row_size: {row_size}, bits: {bits}",
            io.size()
        );
        let args = &[io.as_arg(), KernelArg::Integer(bits as u32)];
        self.launch("multi_bit_reverse", row_size as u64 * count as u64, args);
        // CUDA ends this method with sync_stream(): close the batch (plan 2.6).
        self.flush();
    }

    fn batch_expand_into_evaluate_ntt(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
        count: usize,
        expand_bits: usize,
    ) {
        scope!("batch_expand", {
            tracing::debug!(
                "output: {}, input: {}, count: {count}",
                output.size(),
                input.size()
            );
            let out_size = output.size() / count;
            let in_size = input.size() / count;
            let expand_bits = log2_ceil(out_size / in_size);
            assert_eq!(output.size(), out_size * count);
            assert_eq!(input.size(), in_size * count);
            assert_eq!(out_size, in_size * (1 << expand_bits));
            let args = &[
                output.as_arg(),
                input.as_arg(),
                KernelArg::Integer(count as u32),
                KernelArg::Integer(out_size as u32),
                KernelArg::Integer(in_size as u32),
                KernelArg::Integer(expand_bits as u32),
            ];
            self.launch("batch_expand", out_size as u64, args);
        });

        scope!("batch_evaluate_ntt", {
            tracing::debug!(
                "output: {}, count: {count}, expand_bits: {expand_bits}",
                output.size()
            );
            let row_size = output.size() / count;
            assert_eq!(row_size * count, output.size());
            let n_bits = log2_ceil(row_size);
            assert_eq!(row_size, 1 << n_bits);
            assert!(n_bits >= expand_bits);
            assert!(n_bits < Self::Elem::MAX_ROU_PO2);
            let rou = as_bytes(Self::Elem::ROU_FWD);
            let kernel = self.pipeline("multi_ntt_fwd_step");
            for s_bits in 1 + expand_bits..=n_bits {
                let args = &[
                    output.as_arg(),
                    KernelArg::Bytes(rou),
                    KernelArg::Integer(n_bits as u32),
                    KernelArg::Integer(s_bits as u32),
                    KernelArg::Integer(count as u32),
                ];
                let grid = compute_launch_params(n_bits as u32, s_bits as u32, count as u32);
                self.dispatch(kernel, args, grid, &[]);
            }
        });
        self.flush();
    }

    fn batch_interpolate_ntt(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        scope!("batch_interpolate_ntt");
        tracing::debug!("io: {}, count: {count}", io.size());
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let n_bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << n_bits);
        assert!(n_bits < Self::Elem::MAX_ROU_PO2);

        let rou = as_bytes(Self::Elem::ROU_REV);
        let kernel = self.pipeline("multi_ntt_rev_step");
        for s_bits in (1..=n_bits).rev() {
            let args = &[
                io.as_arg(),
                KernelArg::Bytes(rou),
                KernelArg::Integer(n_bits as u32),
                KernelArg::Integer(s_bits as u32),
                KernelArg::Integer(count as u32),
            ];
            let grid = compute_launch_params(n_bits as u32, s_bits as u32, count as u32);
            self.dispatch(kernel, args, grid, &[]);
        }

        let norm = [Self::Elem::new(row_size as u32).inv()];
        let args = &[io.as_arg(), KernelArg::Bytes(as_bytes(&norm))];
        self.launch("eltwise_mul_factor_fp", io.size() as u64, args);
        self.flush();
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
        let po2 = log2_ceil(coeffs.size() / poly_count);
        let count = 1 << po2;
        assert_eq!(poly_count * count, coeffs.size());
        let eval_count = which.size();
        assert_eq!(xs.size(), eval_count);
        assert_eq!(out.size(), eval_count);
        let args = &[
            out.as_arg(),
            coeffs.as_arg(),
            which.as_arg(),
            xs.as_arg(),
            KernelArg::Integer(count as u32),
        ];
        // One threadgroup of 256 threads per evaluation (the kernel's reduction assumes 256).
        let grid = Grid::Groups {
            groups: MTLSize::new(out.size() as u64, 1, 1),
            threads: MTLSize::new(256, 1, 1),
        };
        self.dispatch(self.pipeline("multi_poly_eval"), args, grid, &[]);
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
        let count = output.size() / Self::ExtElem::EXT_SIZE;
        let to_add = input.size() / count;
        assert_eq!(output.size(), count * Self::ExtElem::EXT_SIZE);
        assert_eq!(input.size(), count * to_add);
        let args = &[
            output.as_arg(),
            input.as_arg(),
            KernelArg::Integer(count as u32),
            KernelArg::Integer(to_add as u32),
        ];
        self.launch("eltwise_sum_fpext", count as u64, args);
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
        let mix = [*mix];
        let args = &[
            output.as_arg(),
            input.as_arg(),
            KernelArg::Bytes(as_bytes(&mix)),
            KernelArg::Integer(count as u32),
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

        let mix_start = [*mix_start];
        let mix = [*mix];
        let args = &[
            output.as_arg(),
            input.as_arg(),
            combos.as_arg(),
            KernelArg::Bytes(as_bytes(&mix_start)),
            KernelArg::Bytes(as_bytes(&mix)),
            KernelArg::Integer(input_size as u32),
            KernelArg::Integer(count as u32),
        ];
        self.launch("mix_poly_coeffs", count as u64, args);
    }

    fn hash_fold(&self, io: &Self::Buffer<Digest>, input_size: usize, output_size: usize) {
        assert_eq!(input_size, 2 * output_size);
        self.hash().hash_fold(self, io, output_size);
        self.flush();
    }

    fn hash_fold_tree(&self, io: &Self::Buffer<Digest>, layers: usize) {
        self.hash().hash_fold_tree(self, io, layers);
        self.flush();
    }

    fn hash_rows(&self, output: &Self::Buffer<Digest>, matrix: &Self::Buffer<Self::Elem>) {
        self.hash().hash_rows(self, output, matrix);
        self.flush();
    }

    fn zk_shift(&self, io: &Self::Buffer<Self::Elem>, poly_count: usize) {
        let bits = log2_ceil(io.size() / poly_count);
        let count = io.size();
        assert_eq!(io.size(), poly_count * (1 << bits));
        let args = &[io.as_arg(), KernelArg::Integer(bits as u32)];
        self.launch("zk_shift", count as u64, args);
        // CUDA ends this method with sync_stream(): close the batch (plan 2.6).
        self.flush();
    }

    fn gather_sample(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        idx: usize,
        size: usize,
        stride: usize,
    ) {
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

    fn trim_device_memory(&self) {
        let released = self.rt.drain_pool();
        tracing::debug!(
            "trim_device_memory: released {:.2} GiB of pooled Metal buffers",
            released as f64 / (1u64 << 30) as f64
        );
    }

    fn gpu_total_memory(&self) -> usize {
        self.device.recommended_max_working_set_size() as usize
    }

    fn gpu_free_memory(&self) -> usize {
        (self.device.recommended_max_working_set_size() as usize)
            .saturating_sub(self.device.current_allocated_size() as usize)
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
        elems.view_mut(|slice| {
            slice.par_iter_mut().for_each(|elem| {
                *elem = elem.valid_or_zero();
            });
        })
    }

    fn scatter(
        &self,
        into: &Self::Buffer<Self::Elem>,
        index: &[u32],
        offsets: &[u32],
        values: &[Self::Elem],
    ) {
        into.view_mut(|into| {
            for cycle in 0..index.len() - 1 {
                for idx in index[cycle]..index[cycle + 1] {
                    into[offsets[idx as usize] as usize] = values[idx as usize];
                }
            }
        });
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
        into.view_mut(|into| {
            for row in 0..from_rows {
                for col in 0..from_cols {
                    into[into_offset + row * into_stride + col] =
                        from[from_offset + row * from_stride + col];
                }
            }
        });
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

fn compute_launch_params(n_bits: u32, s_bits: u32, c_size: u32) -> Grid {
    let s_size = 1 << (s_bits - 1);
    let g_size = 1 << (n_bits - s_bits);

    let mut grid = MTLSize::new(1, 1, 1);
    let mut block = MTLSize::new(1, 1, 1);

    let mut threads = 128;
    // First thread over S
    block.width = threads.min(s_size);
    threads /= block.width;
    // Next thread over G
    block.height = threads.min(g_size);
    // Don't bother threading over C
    let mut grids = 32;
    // First grid over S
    grid.width = grids.min(s_size / block.width);
    grids /= grid.width;
    // Next grid over G
    grid.height = grids.min(g_size / block.height);
    grids /= grid.height;
    // Next grid over C
    grid.depth = grids.min(c_size as u64);
    Grid::Groups {
        groups: grid,
        threads: block,
    }
}
