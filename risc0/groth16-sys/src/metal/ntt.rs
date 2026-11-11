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

//! NTT v1 over BN254 Fr on the Apple GPU (item G5; kernels `kernels/metal/{bit_rev,ntt}.metal`).
//!
//! [NttPlan] holds, for one domain size n = 2^lg, the twiddle table (w^j for j < n/2, w the
//! primitive n-th root; inverse twiddles are read from the same table, see `ntt.metal`) and the
//! coset tables for the shift the CUDA prover's LDE uses (`forward_roots_of_unity[lg + 1]`, the
//! primitive 2n-th root, `kernels/cuda/groth16_prover.cuh` `prover_init`). The base roots come
//! from ark-ff ([FftField::get_root_of_unity]); a test checks them word for word against the
//! sppark tables the CUDA and Intel provers use (`kernels/intel/ntt_params.hpp`). The tables are
//! expanded on the GPU (`ntt_powers`) into private buffers.
//!
//! Every transform is encode-only into the caller's compute encoder (one encoder, serial
//! dispatch type, tracked buffers: dependent dispatches are ordered). Elements are 8 x 32-bit
//! little-endian Montgomery limbs with R = 2^256 (`bn254.h`), the same bytes as ark's
//! `Fp256` representation ([fr_to_words], [fr_from_words]).
//!
//! Transforms (all in place on n elements at `offset` bytes of a buffer):
//! - [NttPlan::encode_forward]: coefficients -> evaluations over the domain (natural order).
//! - [NttPlan::encode_inverse]: evaluations -> coefficients, scaled by 1/n (natural order).
//! - [NttPlan::encode_coset_forward]: coefficients -> evaluations over `shift * domain`.
//! - [NttPlan::encode_to_coset]: evaluations over the domain -> evaluations over
//!   `shift * domain`; the CUDA prover's `NTT_sequence` (inverse NTT natural -> bit-reversed,
//!   times shift^k / n in bit-reversed position, forward NTT bit-reversed -> natural; no
//!   bit-reversal pass).

use std::ffi::c_void;

use anyhow::{anyhow, ensure, Result};
use ark_bn254::Fr;
use ark_ff::{BigInt, FftField, Field, One};
use metal::{
    objc::{msg_send, sel, sel_impl},
    Buffer, BufferRef, CommandQueueRef, ComputeCommandEncoderRef, ComputePipelineState, DeviceRef,
    LibraryRef, MTLCommandBufferStatus, MTLSize,
};

/// Bytes per Fr element.
pub const FR_BYTES: u64 = 32;
/// Device limit on threadgroup memory (D0): larger allocations fail silently, so it is asserted.
pub const MAX_THREADGROUP_BYTES: u64 = 32768;
/// log2 of the elements one `ntt_fused_stages` threadgroup holds (512 Fr = 16 KB).
pub const FUSED_LOG: u32 = 9;
/// Largest supported domain: the coset shift is the 2^(lg+1)-th root and Fr's two-adicity is 28.
pub const MAX_LG: u32 = 27;
/// log2 of the low window of the coset power tables (shift^k = lo[k mod 2^12] * hi[k >> 12]).
const LG_LO: u32 = 12;

/// Buffer indices of the kernels (mirrors the comment blocks of `bit_rev.metal`/`ntt.metal`).
pub mod buf {
    /// `bit_reverse`, `ntt_stage`, `ntt_fused_stages`, `ntt_mul_powers`, `ntt_scale`: the data.
    pub const X: u64 = 0;
    /// `bit_reverse`, `ntt_scale`: params.
    pub const BIT_REV_PARAMS: u64 = 1;
    /// `ntt_scale`: params.
    pub const SCALE_PARAMS: u64 = 1;
    /// `ntt_stage`, `ntt_fused_stages`: twiddles.
    pub const TW: u64 = 1;
    /// `ntt_stage`, `ntt_fused_stages`: params.
    pub const STAGE_PARAMS: u64 = 2;
    /// `ntt_powers`: output.
    pub const POWERS_OUT: u64 = 0;
    /// `ntt_powers`: params.
    pub const POWERS_PARAMS: u64 = 1;
    /// `ntt_mul_powers`: low window table.
    pub const MUL_LO: u64 = 1;
    /// `ntt_mul_powers`: high window table.
    pub const MUL_HI: u64 = 2;
    /// `ntt_mul_powers`: params.
    pub const MUL_PARAMS: u64 = 3;
}

const NTT_INVERSE: u32 = 1;
const NTT_DIF: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct BitRevParams {
    lg: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct StageParams {
    lg: u32,
    q: u32,
    flags: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FusedParams {
    lg: u32,
    q0: u32,
    ns: u32,
    cols: u32,
    flags: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PowersParams {
    count: u32,
    _pad: [u32; 7],
    c: [u32; 8],
    sq: [[u32; 8]; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MulPowersParams {
    lg: u32,
    lg_lo: u32,
    bitrev: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ScaleParams {
    count: u32,
    _pad: [u32; 7],
    c: [u32; 8],
}

// Sizes of the MSL structs (all members are uint32_t; no implicit padding on either side).
const _: () = assert!(std::mem::size_of::<BitRevParams>() == 16);
const _: () = assert!(std::mem::size_of::<StageParams>() == 16);
const _: () = assert!(std::mem::size_of::<FusedParams>() == 32);
const _: () = assert!(std::mem::size_of::<PowersParams>() == 32 + 32 + 32 * 32);
const _: () = assert!(std::mem::size_of::<MulPowersParams>() == 16);
const _: () = assert!(std::mem::size_of::<ScaleParams>() == 64);

/// The Montgomery limbs of `x` as the GPU stores them (8 little-endian 32-bit words).
pub fn fr_to_words(x: &Fr) -> [u32; 8] {
    let l = (x.0).0;
    let mut w = [0u32; 8];
    for i in 0..4 {
        w[2 * i] = l[i] as u32;
        w[2 * i + 1] = (l[i] >> 32) as u32;
    }
    w
}

/// The element whose Montgomery limbs are `w` (no reduction; a GPU output is canonical).
pub fn fr_from_words(w: &[u32]) -> Fr {
    let mut l = [0u64; 4];
    for i in 0..4 {
        l[i] = w[2 * i] as u64 | ((w[2 * i + 1] as u64) << 32);
    }
    Fr::new_unchecked(BigInt::new(l))
}

/// The pipelines of `bit_rev.metal` and `ntt.metal`.
pub struct NttKernels {
    bit_reverse: ComputePipelineState,
    stage: ComputePipelineState,
    fused: ComputePipelineState,
    powers: ComputePipelineState,
    mul_powers: ComputePipelineState,
    scale: ComputePipelineState,
    /// log2 of the largest fused block: [FUSED_LOG], less if the fused pipeline cannot run
    /// 2^(FUSED_LOG - 1) threads per threadgroup.
    fused_log: u32,
}

impl NttKernels {
    /// Build the pipelines from the groth16 metallib (`crate::METAL_LIB`) loaded as `lib`.
    pub fn new(device: &DeviceRef, lib: &LibraryRef) -> Result<Self> {
        let pso = |name: &str| -> Result<ComputePipelineState> {
            let f = lib
                .get_function(name, None)
                .map_err(|e| anyhow!("groth16 metallib has no {name}: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| anyhow!("pipeline {name}: {e}"))
        };
        let fused = pso("ntt_fused_stages")?;
        let max_threads = fused.max_total_threads_per_threadgroup().max(1);
        let fused_log = FUSED_LOG.min(63 - max_threads.leading_zeros() + 1);
        Ok(Self {
            bit_reverse: pso("bit_reverse")?,
            stage: pso("ntt_stage")?,
            fused,
            powers: pso("ntt_powers")?,
            mul_powers: pso("ntt_mul_powers")?,
            scale: pso("ntt_scale")?,
            fused_log,
        })
    }
}

/// How the butterfly levels are dispatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NttSchedule {
    /// One `ntt_stage` dispatch per level (lg dispatches per transform).
    Radix2,
    /// `ntt_fused_stages`: levels 0..9 in one contiguous 512-element pass, then passes of
    /// `9 - cols` levels over 2^cols adjacent columns (so loads stay contiguous over 2^cols Fr).
    Fused { cols: u32 },
}

impl Default for NttSchedule {
    /// The fastest schedule measured at lg 23 on the dev machine (docs/metal-port/notes/G5.md).
    fn default() -> Self {
        NttSchedule::Fused { cols: 2 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    Stage { q: u32 },
    Fused { q0: u32, ns: u32, cols: u32 },
}

/// Twiddle and coset tables for one domain size; see the module docs.
pub struct NttPlan {
    lg: u32,
    n: u64,
    schedule: NttSchedule,
    kernels: NttKernels,
    /// w^j, j < max(n/2, 1).
    tw: Buffer,
    /// shift^k for k < 2^lg_lo.
    coset_lo: Buffer,
    /// shift^(k 2^lg_lo) for k < 2^(lg - lg_lo).
    coset_hi: Buffer,
    /// shift^(k 2^lg_lo) / n.
    coset_hi_ninv: Buffer,
    lg_lo: u32,
    ninv: Fr,
    shift: Fr,
}

impl NttPlan {
    /// A plan for domain size 2^lg with the default schedule. Builds the tables on `queue` and
    /// waits for them.
    pub fn new(
        device: &DeviceRef,
        queue: &CommandQueueRef,
        lib: &LibraryRef,
        lg: u32,
    ) -> Result<Self> {
        Self::with_schedule(device, queue, lib, lg, NttSchedule::default())
    }

    /// A plan with an explicit level schedule (tests and benchmarks).
    pub fn with_schedule(
        device: &DeviceRef,
        queue: &CommandQueueRef,
        lib: &LibraryRef,
        lg: u32,
        schedule: NttSchedule,
    ) -> Result<Self> {
        ensure!(
            lg <= MAX_LG,
            "groth16-metal: NTT domain 2^{lg} too large (max 2^{MAX_LG})"
        );
        if let NttSchedule::Fused { cols } = schedule {
            ensure!(
                cols < FUSED_LOG,
                "groth16-metal: NTT fused cols {cols} >= {FUSED_LOG}"
            );
        }
        let kernels = NttKernels::new(device, lib)?;
        let n = 1u64 << lg;
        let lg_lo = lg.min(LG_LO);
        let w = Fr::get_root_of_unity(n).ok_or_else(|| anyhow!("no 2^{lg}-th root of unity"))?;
        let shift = Fr::get_root_of_unity(2 * n)
            .ok_or_else(|| anyhow!("no 2^{}-th root of unity", lg + 1))?;
        let ninv = Fr::from(n).inverse().unwrap();
        let shift_hi = shift.pow([1u64 << lg_lo]);

        // Nil-checked (G8): a refused allocation is an error, not a null `Buffer`.
        let private = |count: u64| {
            super::new_buffer_checked(device, count.max(1) * FR_BYTES, true, "the NTT tables")
        };
        let tw = private(n / 2)?;
        let coset_lo = private(1 << lg_lo)?;
        let coset_hi = private(1 << (lg - lg_lo))?;
        let coset_hi_ninv = private(1 << (lg - lg_lo))?;

        submit_and_wait(queue, |enc| {
            encode_powers(&kernels, enc, &tw, (n / 2).max(1), Fr::one(), w);
            encode_powers(&kernels, enc, &coset_lo, 1 << lg_lo, Fr::one(), shift);
            encode_powers(
                &kernels,
                enc,
                &coset_hi,
                1 << (lg - lg_lo),
                Fr::one(),
                shift_hi,
            );
            encode_powers(
                &kernels,
                enc,
                &coset_hi_ninv,
                1 << (lg - lg_lo),
                ninv,
                shift_hi,
            );
        })?;

        Ok(Self {
            lg,
            n,
            schedule,
            kernels,
            tw,
            coset_lo,
            coset_hi,
            coset_hi_ninv,
            lg_lo,
            ninv,
            shift,
        })
    }

    /// log2 of the domain size.
    pub fn lg(&self) -> u32 {
        self.lg
    }

    /// The domain size.
    pub fn n(&self) -> u64 {
        self.n
    }

    /// The coset shift (the primitive 2^(lg+1)-th root, as in the CUDA prover).
    pub fn shift(&self) -> Fr {
        self.shift
    }

    /// The schedule in use.
    pub fn schedule(&self) -> NttSchedule {
        self.schedule
    }

    /// Bytes of GPU memory the plan's tables hold.
    pub fn table_bytes(&self) -> u64 {
        self.tw.length()
            + self.coset_lo.length()
            + self.coset_hi.length()
            + self.coset_hi_ninv.length()
    }

    /// The twiddle table (w^j, j < n/2), for tests.
    pub fn twiddles(&self) -> &BufferRef {
        &self.tw
    }

    /// Coefficients -> evaluations over the domain, natural order in and out.
    pub fn encode_forward(&self, enc: &ComputeCommandEncoderRef, x: &BufferRef, offset: u64) {
        self.check(x, offset);
        self.encode_bit_reverse(enc, x, offset);
        self.encode_levels(enc, x, offset, 0);
    }

    /// Evaluations over the domain -> coefficients (scaled by 1/n), natural order in and out.
    pub fn encode_inverse(&self, enc: &ComputeCommandEncoderRef, x: &BufferRef, offset: u64) {
        self.check(x, offset);
        self.encode_bit_reverse(enc, x, offset);
        self.encode_levels(enc, x, offset, NTT_INVERSE);
        let p = ScaleParams {
            count: self.n as u32,
            c: fr_to_words(&self.ninv),
            ..Default::default()
        };
        enc.set_compute_pipeline_state(&self.kernels.scale);
        enc.set_buffer(buf::X, Some(x), offset);
        set_params(enc, buf::SCALE_PARAMS, &p);
        dispatch_1d(enc, &self.kernels.scale, self.n);
    }

    /// Coefficients -> evaluations over `shift * domain`, natural order in and out.
    pub fn encode_coset_forward(&self, enc: &ComputeCommandEncoderRef, x: &BufferRef, offset: u64) {
        self.check(x, offset);
        self.encode_mul_powers(enc, x, offset, &self.coset_hi, false);
        self.encode_bit_reverse(enc, x, offset);
        self.encode_levels(enc, x, offset, 0);
    }

    /// Evaluations over the domain -> evaluations over `shift * domain`, natural order in and
    /// out: the CUDA prover's `NTT_sequence` (DIF inverse, coset powers / n in bit-reversed
    /// position, DIT forward).
    pub fn encode_to_coset(&self, enc: &ComputeCommandEncoderRef, x: &BufferRef, offset: u64) {
        self.check(x, offset);
        self.encode_levels(enc, x, offset, NTT_INVERSE | NTT_DIF);
        self.encode_mul_powers(enc, x, offset, &self.coset_hi_ninv, true);
        self.encode_levels(enc, x, offset, 0);
    }

    fn check(&self, x: &BufferRef, offset: u64) {
        assert!(
            offset.is_multiple_of(4),
            "NTT buffer offset {offset} not 4-byte aligned"
        );
        assert!(
            x.length() >= offset + self.n * FR_BYTES,
            "NTT buffer of {} bytes too small for 2^{} Fr at offset {offset}",
            x.length(),
            self.lg
        );
    }

    fn encode_bit_reverse(&self, enc: &ComputeCommandEncoderRef, x: &BufferRef, offset: u64) {
        if self.lg == 0 {
            return;
        }
        let p = BitRevParams {
            lg: self.lg,
            ..Default::default()
        };
        enc.set_compute_pipeline_state(&self.kernels.bit_reverse);
        enc.set_buffer(buf::X, Some(x), offset);
        set_params(enc, buf::BIT_REV_PARAMS, &p);
        dispatch_1d(enc, &self.kernels.bit_reverse, self.n);
    }

    fn encode_mul_powers(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &BufferRef,
        offset: u64,
        hi: &BufferRef,
        bitrev: bool,
    ) {
        let p = MulPowersParams {
            lg: self.lg,
            lg_lo: self.lg_lo,
            bitrev: bitrev as u32,
            _pad: 0,
        };
        enc.set_compute_pipeline_state(&self.kernels.mul_powers);
        enc.set_buffer(buf::X, Some(x), offset);
        enc.set_buffer(buf::MUL_LO, Some(&self.coset_lo), 0);
        enc.set_buffer(buf::MUL_HI, Some(hi), 0);
        set_params(enc, buf::MUL_PARAMS, &p);
        dispatch_1d(enc, &self.kernels.mul_powers, self.n);
    }

    /// The level passes in DIT (ascending) order.
    fn passes(&self) -> Vec<Pass> {
        let lg = self.lg;
        match self.schedule {
            NttSchedule::Radix2 => (0..lg).map(|q| Pass::Stage { q }).collect(),
            NttSchedule::Fused { cols } => {
                let max = self.kernels.fused_log;
                let cols = cols.min(max - 1);
                let mut out = Vec::new();
                let mut q0 = 0;
                while q0 < lg {
                    let (ns, c) = if q0 == 0 {
                        (lg.min(max), 0)
                    } else {
                        let ns = (max - cols).min(lg - q0);
                        (ns, (max - ns).min(q0))
                    };
                    out.push(Pass::Fused { q0, ns, cols: c });
                    q0 += ns;
                }
                out
            }
        }
    }

    /// All lg butterfly levels: ascending (DIT, bit-reversed in) or, with [NTT_DIF], descending
    /// (natural in, bit-reversed out).
    fn encode_levels(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &BufferRef,
        offset: u64,
        flags: u32,
    ) {
        let mut passes = self.passes();
        if flags & NTT_DIF != 0 {
            passes.reverse();
        }
        for pass in passes {
            match pass {
                Pass::Stage { q } => {
                    let p = StageParams {
                        lg: self.lg,
                        q,
                        flags,
                        _pad: 0,
                    };
                    enc.set_compute_pipeline_state(&self.kernels.stage);
                    enc.set_buffer(buf::X, Some(x), offset);
                    enc.set_buffer(buf::TW, Some(&self.tw), 0);
                    set_params(enc, buf::STAGE_PARAMS, &p);
                    dispatch_1d(enc, &self.kernels.stage, self.n / 2);
                }
                Pass::Fused { q0, ns, cols } => {
                    let log_m = ns + cols;
                    let tg_bytes = (1u64 << log_m) * FR_BYTES;
                    assert!(
                        tg_bytes <= MAX_THREADGROUP_BYTES,
                        "ntt_fused_stages needs {tg_bytes} bytes of threadgroup memory"
                    );
                    let threads = 1u64 << (log_m - 1);
                    assert!(threads <= self.kernels.fused.max_total_threads_per_threadgroup());
                    let p = FusedParams {
                        lg: self.lg,
                        q0,
                        ns,
                        cols,
                        flags,
                        _pad: [0; 3],
                    };
                    enc.set_compute_pipeline_state(&self.kernels.fused);
                    enc.set_buffer(buf::X, Some(x), offset);
                    enc.set_buffer(buf::TW, Some(&self.tw), 0);
                    set_params(enc, buf::STAGE_PARAMS, &p);
                    enc.set_threadgroup_memory_length(0, tg_bytes);
                    enc.dispatch_thread_groups(
                        MTLSize::new(self.n >> log_m, 1, 1),
                        MTLSize::new(threads, 1, 1),
                    );
                }
            }
        }
    }
}

/// `out[j] = c * base^j` for j < count.
fn encode_powers(
    k: &NttKernels,
    enc: &ComputeCommandEncoderRef,
    out: &BufferRef,
    count: u64,
    c: Fr,
    base: Fr,
) {
    let mut p = PowersParams {
        count: count as u32,
        _pad: [0; 7],
        c: fr_to_words(&c),
        sq: [[0; 8]; 32],
    };
    let mut s = base;
    for sq in p.sq.iter_mut() {
        *sq = fr_to_words(&s);
        s.square_in_place();
    }
    enc.set_compute_pipeline_state(&k.powers);
    enc.set_buffer(buf::POWERS_OUT, Some(out), 0);
    set_params(enc, buf::POWERS_PARAMS, &p);
    dispatch_1d(enc, &k.powers, count);
}

fn set_params<T: Copy>(enc: &ComputeCommandEncoderRef, index: u64, p: &T) {
    enc.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        p as *const T as *const c_void,
    );
}

fn dispatch_1d(enc: &ComputeCommandEncoderRef, pso: &ComputePipelineState, count: u64) {
    if count == 0 {
        return;
    }
    let tg = pso.max_total_threads_per_threadgroup().min(256);
    enc.dispatch_threads(MTLSize::new(count, 1, 1), MTLSize::new(tg, 1, 1));
}

/// Encode `f` into one compute encoder of a new command buffer, commit, wait, and return the
/// command buffer's GPU time in seconds; errors if the command buffer failed.
pub fn submit_and_wait(
    queue: &CommandQueueRef,
    f: impl FnOnce(&ComputeCommandEncoderRef),
) -> Result<f64> {
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    f(enc);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    ensure!(
        cb.status() == MTLCommandBufferStatus::Completed,
        "groth16-metal: NTT command buffer failed ({:?})",
        cb.status()
    );
    Ok(gpu_seconds(cb))
}

/// GPUEndTime - GPUStartTime of a completed command buffer (not wrapped by metal-rs 0.29).
// objc's `sel_impl!` checks `feature = "cargo-clippy"`, unknown to this crate's check-cfg.
#[allow(unexpected_cfgs)]
fn gpu_seconds(cb: &metal::CommandBufferRef) -> f64 {
    let (start, end): (f64, f64) =
        unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
    end - start
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ark_ff::{PrimeField, UniformRand, Zero};
    use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
    use metal::{CommandQueue, Device, Library, MTLResourceOptions};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    struct Ctx {
        device: Device,
        queue: CommandQueue,
        lib: Library,
        /// The shared GPU test lock (`super::super::test_lock`), held for the test.
        _lock: std::sync::RwLockReadGuard<'static, ()>,
    }

    fn ctx() -> Option<Ctx> {
        let _lock = super::super::test_lock::shared();
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping");
            return None;
        };
        let lib = device.new_library_with_data(crate::METAL_LIB).unwrap();
        let queue = device.new_command_queue();
        Some(Ctx {
            device,
            queue,
            lib,
            _lock,
        })
    }

    fn upload(c: &Ctx, v: &[Fr]) -> Buffer {
        let b = c.device.new_buffer(
            (v.len().max(1) as u64) * FR_BYTES,
            MTLResourceOptions::StorageModeShared,
        );
        let w = b.contents() as *mut u32;
        for (i, x) in v.iter().enumerate() {
            let words = fr_to_words(x);
            unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), w.add(8 * i), 8) };
        }
        b
    }

    fn download(b: &BufferRef, n: usize) -> Vec<Fr> {
        let w = unsafe { std::slice::from_raw_parts(b.contents() as *const u32, 8 * n) };
        w.chunks_exact(8).map(fr_from_words).collect()
    }

    fn random(n: usize, seed: u64) -> Vec<Fr> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n).map(|_| Fr::rand(&mut rng)).collect()
    }

    fn assert_same(what: &str, got: &[Fr], want: &[Fr]) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
        assert!(
            bad.is_empty(),
            "{what}: {} of {} differ, first at {}",
            bad.len(),
            got.len(),
            bad[0]
        );
    }

    /// Rows of a `static constexpr uint32_t NAME[29][8]` table of `ntt_params.hpp`.
    fn sppark_table(name: &str) -> Vec<[u32; 8]> {
        let text = include_str!("../../kernels/intel/ntt_params.hpp");
        let start = text.find(&format!("{name}[")).unwrap();
        let body = &text[start..];
        let body = &body[..body.find("};").unwrap()];
        body.lines()
            .filter(|l| l.trim_start().starts_with("{0x"))
            .map(|l| {
                let v: Vec<u32> = l
                    .split(|ch: char| !ch.is_ascii_alphanumeric())
                    .filter(|t| t.starts_with("0x"))
                    .map(|t| u32::from_str_radix(t[2..].trim_end_matches('u'), 16).unwrap())
                    .collect();
                v.try_into().unwrap()
            })
            .collect()
    }

    /// The ark-ff roots and 1/n are the sppark tables the CUDA/Intel provers use, word for word
    /// (so coset evaluations line up with the proving key's H points).
    #[test]
    fn roots_match_sppark_tables() {
        let fwd = sppark_table("FORWARD_ROOTS_OF_UNITY");
        let inv = sppark_table("INVERSE_ROOTS_OF_UNITY");
        let dinv = sppark_table("DOMAIN_SIZE_INVERSE");
        assert_eq!(fwd.len(), 29);
        assert_eq!(inv.len(), 29);
        assert_eq!(dinv.len(), 29);
        for lg in 0..=28u32 {
            let n = 1u64 << lg;
            let w = Fr::get_root_of_unity(n).unwrap();
            assert_eq!(fr_to_words(&w), fwd[lg as usize], "forward root lg {lg}");
            assert_eq!(
                fr_to_words(&w.inverse().unwrap()),
                inv[lg as usize],
                "inverse root lg {lg}"
            );
            let ni = Fr::from(n).inverse().unwrap();
            assert_eq!(fr_to_words(&ni), dinv[lg as usize], "1/n lg {lg}");
        }
        // Word conversion roundtrip and Montgomery layout (1 in Montgomery form is R mod r).
        let one = fr_to_words(&Fr::one());
        assert_eq!(one, fwd[0]);
        let x = Fr::from(123456789u64);
        assert_eq!(fr_from_words(&fr_to_words(&x)), x);
        assert_eq!(x.into_bigint().0[0], 123456789);
    }

    #[test]
    fn tables_match_ark() {
        let Some(c) = ctx() else { return };
        for lg in [1u32, 5, 10, 13] {
            let plan = NttPlan::new(&c.device, &c.queue, &c.lib, lg).unwrap();
            let n = 1usize << lg;
            // Copy the private twiddle table into a shared buffer.
            let out = upload(&c, &vec![Fr::zero(); n / 2]);
            let cb = c.queue.new_command_buffer();
            let blit = cb.new_blit_command_encoder();
            blit.copy_from_buffer(plan.twiddles(), 0, &out, 0, (n as u64 / 2) * FR_BYTES);
            blit.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let w = Fr::get_root_of_unity(n as u64).unwrap();
            let mut want = Vec::with_capacity(n / 2);
            let mut acc = Fr::one();
            for _ in 0..n / 2 {
                want.push(acc);
                acc *= w;
            }
            assert_same(&format!("twiddles lg {lg}"), &download(&out, n / 2), &want);
            assert_eq!(plan.shift(), Fr::get_root_of_unity(2 * n as u64).unwrap());
        }
    }

    /// One transform on a copy of `input`; returns the output and the GPU seconds.
    fn run(
        c: &Ctx,
        plan: &NttPlan,
        input: &[Fr],
        f: impl Fn(&NttPlan, &ComputeCommandEncoderRef, &BufferRef, u64),
    ) -> (Vec<Fr>, f64) {
        let b = upload(c, input);
        let t = submit_and_wait(&c.queue, |enc| f(plan, enc, &b, 0)).unwrap();
        (download(&b, input.len()), t)
    }

    /// Forward, inverse, roundtrip, coset forward and to_coset against ark-poly.
    fn check_lg(c: &Ctx, lg: u32, schedules: &[NttSchedule], report: bool) {
        let n = 1usize << lg;
        let t0 = Instant::now();
        let coeffs = random(n, 1000 + lg as u64);
        let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        let shift = Fr::get_root_of_unity(2 * n as u64).unwrap();
        let coset = domain.get_coset(shift).unwrap();
        let evals = domain.fft(&coeffs);
        let coset_evals = coset.fft(&coeffs);
        let interp = domain.ifft(&coeffs); // inverse of a random vector
        if report {
            eprintln!("lg {lg}: ark reference {:.1} s", t0.elapsed().as_secs_f64());
        }

        for &s in schedules {
            let base = c.device.current_allocated_size();
            let plan = NttPlan::with_schedule(&c.device, &c.queue, &c.lib, lg, s).unwrap();
            let what = |op: &str| format!("lg {lg} {s:?} {op}");

            let (fwd, t_fwd) = run(c, &plan, &coeffs, NttPlan::encode_forward);
            assert_same(&what("forward"), &fwd, &evals);
            let (inv, t_inv) = run(c, &plan, &coeffs, NttPlan::encode_inverse);
            assert_same(&what("inverse"), &inv, &interp);
            let (back, _) = run(c, &plan, &fwd, NttPlan::encode_inverse);
            assert_same(&what("roundtrip"), &back, &coeffs);
            let (cf, t_cf) = run(c, &plan, &coeffs, NttPlan::encode_coset_forward);
            assert_same(&what("coset_forward"), &cf, &coset_evals);
            let (tc, t_tc) = run(c, &plan, &evals, NttPlan::encode_to_coset);
            assert_same(&what("to_coset"), &tc, &coset_evals);

            // Peak while a data buffer and the plan are alive (current_allocated_size is the
            // device total, so other processes' or tests' allocations can show up here).
            let data = upload(c, &coeffs);
            let peak = c.device.current_allocated_size();
            drop(data);
            if report {
                eprintln!(
                    "lg {lg} {s:?}: GPU forward {:.2} ms, inverse {:.2} ms, coset_forward {:.2} ms, \
                     to_coset {:.2} ms; tables {:.1} MiB, data {:.1} MiB, device allocated +{:.1} MiB",
                    t_fwd * 1e3,
                    t_inv * 1e3,
                    t_cf * 1e3,
                    t_tc * 1e3,
                    plan.table_bytes() as f64 / (1 << 20) as f64,
                    (n as u64 * FR_BYTES) as f64 / (1 << 20) as f64,
                    peak.saturating_sub(base) as f64 / (1 << 20) as f64,
                );
            }
        }
    }

    const ALL: &[NttSchedule] = &[
        NttSchedule::Radix2,
        NttSchedule::Fused { cols: 0 },
        NttSchedule::Fused { cols: 2 },
        NttSchedule::Fused { cols: 5 },
    ];

    #[test]
    fn ntt_small_sizes() {
        let Some(c) = ctx() else { return };
        for lg in 0..=14 {
            check_lg(&c, lg, ALL, false);
        }
        check_lg(&c, 19, ALL, false);
    }

    #[test]
    fn ntt_lg10() {
        let Some(c) = ctx() else { return };
        check_lg(&c, 10, ALL, true);
    }

    #[test]
    fn ntt_lg16() {
        let Some(c) = ctx() else { return };
        check_lg(&c, 16, ALL, true);
    }

    #[test]
    fn ntt_lg23() {
        let Some(c) = ctx() else { return };
        check_lg(&c, 23, &[NttSchedule::default(), NttSchedule::Radix2], true);
    }

    /// Two transforms at different offsets of one buffer, in one command buffer.
    #[test]
    fn ntt_offsets_and_batching() {
        let Some(c) = ctx() else { return };
        let lg = 12;
        let n = 1usize << lg;
        let plan = NttPlan::new(&c.device, &c.queue, &c.lib, lg).unwrap();
        let a = random(n, 1);
        let b = random(n, 2);
        let mut both = a.clone();
        both.extend_from_slice(&b);
        let buf = upload(&c, &both);
        submit_and_wait(&c.queue, |enc| {
            plan.encode_to_coset(enc, &buf, 0);
            plan.encode_to_coset(enc, &buf, n as u64 * FR_BYTES);
            plan.encode_inverse(enc, &buf, n as u64 * FR_BYTES);
        })
        .unwrap();
        let out = download(&buf, 2 * n);
        let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        let coset = domain.get_coset(plan.shift()).unwrap();
        assert_same("a to_coset", &out[..n], &coset.fft(&domain.ifft(&a)));
        // Interpreting coset evaluations as domain evaluations gives the coefficients p_k s^k.
        let pb = domain.ifft(&b);
        let want: Vec<Fr> = pb
            .iter()
            .enumerate()
            .map(|(k, x)| *x * plan.shift().pow([k as u64]))
            .collect();
        assert_same("b to_coset+inverse", &out[n..], &want);
    }

    /// Sweep of the schedules at lg 23 (timing only; run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn ntt_bench_schedules() {
        let Some(c) = ctx() else { return };
        let lg = 23;
        let n = 1usize << lg;
        let x = random(n, 7);
        let b = upload(&c, &x);
        let mut schedules = vec![NttSchedule::Radix2];
        schedules.extend((0..FUSED_LOG).map(|cols| NttSchedule::Fused { cols }));
        for s in schedules {
            let plan = NttPlan::with_schedule(&c.device, &c.queue, &c.lib, lg, s).unwrap();
            let mut best = [f64::MAX; 2];
            for _ in 0..5 {
                best[0] = best[0]
                    .min(submit_and_wait(&c.queue, |enc| plan.encode_forward(enc, &b, 0)).unwrap());
                best[1] = best[1].min(
                    submit_and_wait(&c.queue, |enc| plan.encode_to_coset(enc, &b, 0)).unwrap(),
                );
            }
            eprintln!(
                "lg {lg} {s:?} passes {:?}: forward {:.2} ms, to_coset {:.2} ms",
                plan.passes().len(),
                best[0] * 1e3,
                best[1] * 1e3
            );
        }
    }
}
