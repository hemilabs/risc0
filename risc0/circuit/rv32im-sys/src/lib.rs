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

#[cfg(feature = "rocm")]
use risc0_sys::cuda::DevicePointer;
use derive_more::Debug;
use risc0_core::field::baby_bear::{BabyBearElem, BabyBearExtElem};

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
    pub fn risc0_circuit_rv32im_cuda_warmup_eval_check() -> *const std::os::raw::c_char;

    /// Make the persistent stream wait for the eval_check stream to complete.
    /// Call this before operations on the persistent stream that read eval_check
    /// output (e.g., iNTT on check_poly). GPU-side dependency only — CPU returns
    /// immediately.
    pub fn risc0_circuit_rv32im_cuda_eval_check_dep() -> *const std::os::raw::c_char;
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
