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

#[cfg(feature = "cuda")]
use cust::memory::DevicePointer;

use derive_more::Debug;
use risc0_core::field::baby_bear::{BabyBearElem, BabyBearExtElem};
#[cfg(feature = "rocm")]
use risc0_sys::cuda::DevicePointer;

#[derive(Clone, Debug, PartialEq)]
#[repr(C)]
pub struct RawMemoryTransaction {
    #[debug("{addr:#010x}")]
    pub addr: u32,
    pub cycle: u32,
    #[debug("{word:#010x}")]
    pub word: u32,
    pub prev_cycle: u32,
    #[debug("{word:#010x}")]
    pub prev_word: u32,
}

#[derive(Clone, Debug, PartialEq)]
#[repr(C)]
pub struct RawPreflightCycle {
    pub state: u32,
    #[debug("{pc:#010x}")]
    pub pc: u32,
    pub major: u8,
    pub minor: u8,
    pub machine_mode: u8,
    #[debug(skip)]
    pub padding: u8,
    pub user_cycle: u32,
    pub txn_idx: u32,
    pub paging_idx: u32,
    pub bigint_idx: u32,
    pub diff_count: [u32; 2],
}

#[repr(C)]
pub struct RawPreflightTrace {
    pub cycles: *const RawPreflightCycle,
    pub txns: *const RawMemoryTransaction,
    pub bigint_bytes: *const u8,
    pub txns_len: u32,
    pub bigint_bytes_len: u32,
    pub table_split_cycle: u32,
}

#[repr(C)]
pub struct RawBuffer {
    pub buf: *const BabyBearElem,
    pub rows: usize,
    pub cols: usize,
    pub checked: bool,
}

#[repr(C)]
pub struct RawExecBuffers {
    pub global: RawBuffer,
    pub data: RawBuffer,
    pub pre_data: RawBuffer,
}

#[repr(C)]
pub struct RawAccumBuffers {
    pub data: RawBuffer,
    pub accum: RawBuffer,
    pub global: RawBuffer,
    pub mix: RawBuffer,
}

extern "C" {
    pub fn risc0_circuit_rv32im_cpu_witgen(
        mode: u32,
        buffers: *const RawExecBuffers,
        preflight: *const RawPreflightTrace,
        cycles: u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_rv32im_cpu_accum(
        buffers: *const RawAccumBuffers,
        preflight: *const RawPreflightTrace,
        cycles: u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_rv32im_cpu_poly_fp(
        cycle: usize,
        steps: usize,
        poly_mixs: *const BabyBearExtElem,
        args_ptr: *const *const BabyBearElem,
        result: *mut BabyBearExtElem,
    ) -> *const std::os::raw::c_char;
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
extern "C" {
    pub fn risc0_circuit_rv32im_cuda_witgen(
        mode: u32,
        buffers: *const RawExecBuffers,
        preflight: *const RawPreflightTrace,
        cycles: u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_rv32im_cuda_accum(
        buffers: *const RawAccumBuffers,
        preflight: *const RawPreflightTrace,
        cycles: u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_rv32im_cuda_eval_check(
        check: DevicePointer<u8>,
        ctrl: DevicePointer<u8>,
        data: DevicePointer<u8>,
        accum: DevicePointer<u8>,
        mix: DevicePointer<u8>,
        out: DevicePointer<u8>,
        rou: *const BabyBearElem,
        po2: u32,
        domain: u32,
        poly_mix_pows: *const u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_rv32im_cuda_warmup() -> *const std::os::raw::c_char;

    /// 1 when `risc0_circuit_rv32im_cuda_accum` also zeroizes (INVALID -> 0) the
    /// accum buffer (fused into its finalize kernel on CUDA), so the caller skips its own zeroize.
    pub fn risc0_circuit_rv32im_cuda_accum_zeroizes() -> std::os::raw::c_int;
    pub fn risc0_circuit_rv32im_cuda_warmup_eval_check() -> *const std::os::raw::c_char;

    /// Make the persistent stream wait for the eval_check stream to complete.
    /// Call this before operations on the persistent stream that read eval_check
    /// output (e.g., iNTT on check_poly). GPU-side dependency only — CPU returns
    /// immediately.
    pub fn risc0_circuit_rv32im_cuda_eval_check_dep() -> *const std::os::raw::c_char;
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
extern "C" {
    /// Witgen with an optional pinned staging buffer holding cycles|txns|bigint_bytes back to
    /// back (`staged`, kept alive by the caller until the witgen's stream work is done), and
    /// optional device-side injection of the five per-cycle columns (`cycle_cols`: cycle,
    /// next_pc_low, next_pc_high, next_state, next_machine_mode).
    pub fn risc0_circuit_rv32im_cuda_witgen_v2(
        mode: u32,
        buffers: *const RawExecBuffers,
        preflight: *const RawPreflightTrace,
        cycles: u32,
        staged: *const u8,
        cycle_cols: *const u32,
        staged_id: u64,
    ) -> *const std::os::raw::c_char;

    /// Copy a staged trace to the device on a side stream (see ffi.cu).
    pub fn risc0_circuit_rv32im_cuda_prefetch(
        staged: *const u8,
        bytes: usize,
        id: u64,
        out_ev: *mut *mut std::ffi::c_void,
    ) -> *const std::os::raw::c_char;

    /// Wait for and destroy an event returned by `risc0_circuit_rv32im_cuda_prefetch`.
    pub fn risc0_circuit_rv32im_cuda_event_wait(
        ev: *mut std::ffi::c_void,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_rv32im_cuda_host_alloc(
        bytes: usize,
        out: *mut *mut std::ffi::c_void,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_rv32im_cuda_host_free(
        ptr: *mut std::ffi::c_void,
    ) -> *const std::os::raw::c_char;

    /// Free the device buffers of the staged-trace slots not in use (see ffi.cu).
    pub fn risc0_circuit_rv32im_cuda_release_slots() -> *const std::os::raw::c_char;

    /// 1 once staging is off because a thread proved with another device current.
    pub fn risc0_circuit_rv32im_cuda_staging_disabled() -> i32;
}

#[cfg(feature = "intel")]
extern "C" {
    pub fn risc0_circuit_rv32im_intel_eval_check(
        queue: *mut std::os::raw::c_void,
        check: *mut std::os::raw::c_void,
        data: *const std::os::raw::c_void,
        accum: *const std::os::raw::c_void,
        out: *const std::os::raw::c_void,
        mix: *const std::os::raw::c_void,
        poly_mix: *const std::os::raw::c_void,
        rou: u32,
        po2: u32,
        domain: u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_rv32im_intel_eval_check_sync(
        queue: *mut std::os::raw::c_void,
    ) -> *const std::os::raw::c_char;

    /// Frees eval_check's scratch slab (waits for the queue first).
    pub fn risc0_circuit_rv32im_intel_eval_check_release() -> *const std::os::raw::c_char;

    /// The number of parts of the degree-split eval_check (3: risc0_rv32im_intel_hi, _mid and
    /// _lo; 2: _hi and _lo) when available and on for the queue's GPU (Xe2, or
    /// RISC0_INTEL_EVAL_SPLIT=1), else 0. Decided on the first call.
    pub fn risc0_circuit_rv32im_intel_eval_check_parts_available(
        queue: *mut std::os::raw::c_void,
    ) -> std::os::raw::c_int;

    /// The degree-split eval_check on the `domain`-point coset: the constraint terms of the
    /// highest degrees on every row into `check` (4 * domain); those of degree <= 3 on its even
    /// rows into `check_lo` (4 * domain / 2); with three parts, those of degree 4 on cosets 0, 1, 2
    /// of the trace domain into `check_mid` (4 * 3 * domain / 4, coset-major; null with two
    /// parts). The caller recovers the rest of lo and mid and adds them.
    pub fn risc0_circuit_rv32im_intel_eval_check_parts(
        queue: *mut std::os::raw::c_void,
        check: *mut std::os::raw::c_void,
        check_lo: *mut std::os::raw::c_void,
        check_mid: *mut std::os::raw::c_void,
        data: *const std::os::raw::c_void,
        accum: *const std::os::raw::c_void,
        out: *const std::os::raw::c_void,
        mix: *const std::os::raw::c_void,
        poly_mix: *const std::os::raw::c_void,
        rou: u32,
        po2: u32,
        domain: u32,
    ) -> *const std::os::raw::c_char;

    /// Adds `n` cosets of the trace domain, stored coset-major in `src` (4 limbs * n * domain / 4),
    /// into their rows 4 m + c0 + c of `check` (4 * domain).
    pub fn risc0_circuit_rv32im_intel_eval_check_add_cosets(
        queue: *mut std::os::raw::c_void,
        check: *mut std::os::raw::c_void,
        src: *const std::os::raw::c_void,
        domain: u32,
        c0: u32,
        n: u32,
    ) -> *const std::os::raw::c_char;

    /// The degree-4 part's coefficients on coset 3 of the trace domain (4 * 2^po2, bit-reversed
    /// order) from those on cosets 0, 1, 2 (`coeffs`, batch_interpolate_ntt of check_mid).
    pub fn risc0_circuit_rv32im_intel_eval_check_coset3(
        queue: *mut std::os::raw::c_void,
        out: *mut std::os::raw::c_void,
        coeffs: *const std::os::raw::c_void,
        rou: u32,
        po2: u32,
    ) -> *const std::os::raw::c_char;

    // 2-way multi-pass eval_check (separate .so files per pass)
    pub fn risc0_circuit_rv32im_intel_eval_check_pass1(
        queue: *mut std::os::raw::c_void,
        inter_fp: *mut std::os::raw::c_void,
        inter_ext: *mut std::os::raw::c_void,
        data: *const std::os::raw::c_void,
        accum: *const std::os::raw::c_void,
        out: *const std::os::raw::c_void,
        mix: *const std::os::raw::c_void,
        poly_mix: *const std::os::raw::c_void,
        domain: u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_rv32im_intel_eval_check_pass2(
        queue: *mut std::os::raw::c_void,
        check: *mut std::os::raw::c_void,
        inter_fp: *const std::os::raw::c_void,
        inter_ext: *const std::os::raw::c_void,
        data: *const std::os::raw::c_void,
        accum: *const std::os::raw::c_void,
        out: *const std::os::raw::c_void,
        mix: *const std::os::raw::c_void,
        poly_mix: *const std::os::raw::c_void,
        rou: u32,
        po2: u32,
        domain: u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_rv32im_intel_witgen(
        queue: *mut std::os::raw::c_void,
        mode: u32,
        d_data: *mut std::os::raw::c_void,
        data_rows: u32,
        data_cols: u32,
        d_pre_data: *mut std::os::raw::c_void,
        d_global: *mut std::os::raw::c_void,
        global_cols: u32,
        h_cycles: *const RawPreflightCycle,
        cycles_len: u32,
        h_txns: *const RawMemoryTransaction,
        txns_len: u32,
        h_bigint: *const u8,
        bigint_len: u32,
        table_split_cycle: u32,
        last_cycle: u32,
        // witgen::CYCLE_COLS (5) when the host injector left those columns out, else null.
        cycle_cols: *const u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_rv32im_intel_accum(
        queue: *mut std::os::raw::c_void,
        d_data: *mut std::os::raw::c_void,
        data_rows: u32,
        data_cols: u32,
        d_accum: *mut std::os::raw::c_void,
        accum_rows: u32,
        accum_cols: u32,
        d_global: *mut std::os::raw::c_void,
        global_cols: u32,
        d_mix: *mut std::os::raw::c_void,
        mix_cols: u32,
        h_cycles: *const RawPreflightCycle,
        cycles_len: u32,
        h_txns: *const RawMemoryTransaction,
        txns_len: u32,
        h_bigint: *const u8,
        bigint_len: u32,
        table_split_cycle: u32,
        last_cycle: u32,
    ) -> *const std::os::raw::c_char;
}
