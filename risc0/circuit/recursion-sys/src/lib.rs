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

use risc0_core::field::baby_bear::{BabyBearElem, BabyBearExtElem};

#[repr(C)]
pub struct RawExecBuffers {
    pub ctrl: *const BabyBearElem,
    pub data: *const BabyBearElem,
    pub global: *const BabyBearElem,
}

#[repr(C)]
pub struct RawAccumBuffers {
    pub ctrl: *const BabyBearElem,
    pub global: *const BabyBearElem,
    pub data: *const BabyBearElem,
    pub mix: *const BabyBearElem,
    pub accum: *const BabyBearElem,
}

#[derive(Clone, Debug, PartialEq)]
#[repr(C)]
pub struct RawPreflightCycle {
    pub iop_idx: u32,
    pub is_par_safe: u32,
}

#[repr(C)]
pub struct RawPreflightTrace {
    pub wom: *const BabyBearExtElem,
    pub cycles: *const RawPreflightCycle,
    pub iops: *const BabyBearExtElem,
    pub num_woms: u32,
    pub num_cycles: u32,
    pub num_iops: u32,
}

/// Rows of host-executed witgen chains, uploaded and scattered by the CUDA witgen.
#[repr(C)]
pub struct RawHostChains {
    pub heads: *const u32,
    pub num_heads: u32,
    pub cycles: *const u32,
    pub rows: *const u32,
    pub wom_idx: *const u32,
    pub wom_rows: *const u32,
    pub num_rows: u32,
    pub data_cols: u32,
}

#[derive(Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum StepMode {
    Parallel,
    SeqForward,
    SeqReverse,
}

extern "C" {
    pub fn risc0_circuit_recursion_cpu_witgen(
        mode: StepMode,
        buffers: *const RawExecBuffers,
        preflight: *const RawPreflightTrace,
        total_cycles: u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_recursion_cpu_accum(
        buffers: *const RawAccumBuffers,
        work_cycles: u32,
        total_cycles: u32,
    ) -> *const std::os::raw::c_char;

    /// Run the given (start, len) chains of the CPU witgen on the host; see ffi.cpp.
    pub fn risc0_circuit_recursion_cpu_chains(
        code: *const BabyBearElem,
        ctrl_cols: u32,
        preflight: *const RawPreflightTrace,
        steps: u32,
        data_cols: u32,
        chains: *const u32,
        num_chains: u32,
        out_cycles: *mut u32,
        out_rows: *mut u32,
        out_wom_idx: *mut u32,
        out_wom_rows: *mut u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_recursion_cpu_eval_check(
        buffers: *const RawAccumBuffers,
        poly_mix: *const BabyBearExtElem,
        check: *const BabyBearElem,
        rou: BabyBearElem,
        po2: u32,
        steps: u32,
    ) -> *const std::os::raw::c_char;
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
extern "C" {
    pub fn risc0_circuit_recursion_cuda_witgen(
        mode: StepMode,
        buffers: *const RawExecBuffers,
        preflight: *const RawPreflightTrace,
        steps: u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_recursion_cuda_warmup() -> *const std::os::raw::c_char;

    pub fn risc0_circuit_recursion_cuda_warmup_eval_check() -> *const std::os::raw::c_char;

    pub fn risc0_circuit_recursion_cuda_witgen_chains(
        mode: StepMode,
        buffers: *const RawExecBuffers,
        preflight: *const RawPreflightTrace,
        steps: u32,
        host_chains: *const RawHostChains,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_recursion_cuda_accum(
        buffers: *const RawAccumBuffers,
        steps: u32,
        cycles: u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_recursion_cuda_eval_check(
        check: *const BabyBearElem,
        ctrl: *const BabyBearElem,
        data: *const BabyBearElem,
        accum: *const BabyBearElem,
        mix: *const BabyBearElem,
        out: *const BabyBearElem,
        rou: *const BabyBearElem,
        po2: u32,
        domain: u32,
        poly_mix: *const BabyBearExtElem,
    ) -> *const std::os::raw::c_char;
}

#[cfg(feature = "intel")]
extern "C" {
    pub fn risc0_circuit_recursion_intel_eval_check(
        queue: *mut std::os::raw::c_void,
        check: *mut std::os::raw::c_void,
        ctrl: *const std::os::raw::c_void,
        data: *const std::os::raw::c_void,
        accum: *const std::os::raw::c_void,
        mix: *const std::os::raw::c_void,
        out: *const std::os::raw::c_void,
        poly_mix: *const std::os::raw::c_void,
        rou: u32,
        po2: u32,
        domain: u32,
    ) -> *const std::os::raw::c_char;

    /// GPU accum (step_compute_accum, prefix product, step_verify_accum) on
    /// the device buffers; enqueued on `queue` without waiting.
    pub fn risc0_circuit_recursion_intel_accum(
        queue: *mut std::os::raw::c_void,
        ctrl: *mut std::os::raw::c_void,
        global: *mut std::os::raw::c_void,
        data: *mut std::os::raw::c_void,
        mix: *mut std::os::raw::c_void,
        accum: *mut std::os::raw::c_void,
        work: u32,
        total: u32,
    ) -> *const std::os::raw::c_char;
}
