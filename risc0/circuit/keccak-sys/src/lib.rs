// Copyright 2024 RISC Zero, Inc.
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

#[cfg(all(
    feature = "metal",
    any(feature = "cuda", feature = "rocm", feature = "intel")
))]
compile_error!(
    "Feature 'metal' is mutually exclusive with 'cuda', 'rocm' and 'intel'. Enable only one GPU backend."
);

#[cfg(all(feature = "metal", not(any(target_os = "macos", target_os = "ios"))))]
compile_error!("Feature 'metal' is only supported on macOS and iOS targets.");

/// The unit's Metal kernels (`kernels/metal`), compiled by build.rs into a metallib.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub const METAL_LIB: &[u8] = include_bytes!(env!("KECCAK_METAL_PATH"));

/// Every kernel entry point in [METAL_LIB]; the HAL fails fast if one is missing.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub const METAL_KERNEL_NAMES: &[&str] = &METAL_KERNEL_NAMES_ARRAY;

/// The hand-written kernels of [METAL_LIB] (`kernels/metal/*.metal`).
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
const METAL_HAND_KERNEL_NAMES: [&str; 1] = ["scatter_preflight"];

/// The hand-written kernels, then the eval_check stages ([metal_eval::EVAL_STAGE_NAMES]).
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
const METAL_KERNEL_NAMES_ARRAY: [&str; METAL_HAND_KERNEL_NAMES.len() + metal_eval::EVAL_STAGES] =
    concat_names(METAL_HAND_KERNEL_NAMES, metal_eval::EVAL_STAGE_NAMES);

/// `a` followed by `b` (`N` must be `A + B`), for building [METAL_KERNEL_NAMES] at compile time.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
const fn concat_names<const A: usize, const B: usize, const N: usize>(
    a: [&'static str; A],
    b: [&'static str; B],
) -> [&'static str; N] {
    assert!(A + B == N);
    let mut out = [""; N];
    let mut i = 0;
    while i < A {
        out[i] = a[i];
        i += 1;
    }
    while i < N {
        out[i] = b[i - A];
        i += 1;
    }
    out
}

/// The staged eval_check kernels' names and sizes, written by the translator
/// (`build_metal_eval.rs`) next to the stage sources: see `kernels/metal/eval_check_prelude.h`.
/// Under `RISC0_SKIP_BUILD_KERNELS` (empty metallib) build.rs writes a stub with no stages, so
/// this crate and the HAL still compile; loading the empty metallib fails as before.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod metal_eval {
    include!(env!("KECCAK_METAL_EVAL_RS"));
}

#[cfg(feature = "cuda")]
use cust::memory::DevicePointer;

#[cfg(feature = "rocm")]
use risc0_sys::cuda::DevicePointer;

use risc0_core::field::baby_bear::{BabyBearElem, BabyBearExtElem};

/// Where to put the preflight's packed words in the witness: see `scatter_preflight` in
/// `kernels/cuda/ffi.cu` and `kernels/metal/scatter.metal`, which mirror this layout.
#[derive(Debug)]
#[repr(C)]
pub struct ScatterInfo {
    pub offset: u32,
    pub row: u32,
    pub col: u16,
    pub count: u16,
    pub bits: u32,
}

// The GPU kernels read `ScatterInfo` arrays as 16-byte structs with these field offsets
// (kernels/metal/scatter.metal static_asserts the size on its side).
const _: () = {
    assert!(std::mem::size_of::<ScatterInfo>() == 16);
    assert!(std::mem::align_of::<ScatterInfo>() == 4);
    assert!(std::mem::offset_of!(ScatterInfo, offset) == 0);
    assert!(std::mem::offset_of!(ScatterInfo, row) == 4);
    assert!(std::mem::offset_of!(ScatterInfo, col) == 8);
    assert!(std::mem::offset_of!(ScatterInfo, count) == 10);
    assert!(std::mem::offset_of!(ScatterInfo, bits) == 12);
};

#[repr(C)]
pub struct RawPreflightTrace {
    pub preimages: *const [u64; 25],
    pub preimages_count: u32,
    pub preimage_idxs: *const u32,
    pub run_order: *const u32,
}

#[repr(C)]
pub struct RawExecBuffers {
    pub global: RawBuffer,
    pub data: RawBuffer,
}

#[repr(C)]
pub struct RawBuffer {
    pub buf: *const BabyBearElem,
    pub rows: usize,
    pub cols: usize,
    pub checked_reads: bool,
}

extern "C" {
    pub fn risc0_circuit_keccak_cpu_witgen(
        mode: u32,
        buffers: *const RawExecBuffers,
        preflight: *const RawPreflightTrace,
        cycles: u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_keccak_cpu_poly_fp(
        cycle: usize,
        steps: usize,
        poly_mixs: *const BabyBearExtElem,
        args_ptr: *const *const BabyBearElem,
        result: *mut BabyBearExtElem,
    ) -> *const std::os::raw::c_char;
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
extern "C" {
    pub fn risc0_circuit_keccak_cuda_scatter(
        into: DevicePointer<u8>,
        infos: *const ScatterInfo,
        from: DevicePointer<u8>,
        rows: u32,
        count: u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_keccak_cuda_witgen(
        mode: u32,
        buffers: *const RawExecBuffers,
        preflight: *const RawPreflightTrace,
        cycles: u32,
    ) -> *const std::os::raw::c_char;

    pub fn risc0_circuit_keccak_cuda_eval_check(
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

    pub fn risc0_circuit_keccak_cuda_reset() -> *const std::os::raw::c_char;
}

#[cfg(feature = "intel")]
extern "C" {
    pub fn risc0_circuit_keccak_intel_eval_check(
        queue: *mut std::ffi::c_void,
        check: *mut u32,
        ctrl: *const u32,
        data: *const u32,
        accum: *const u32,
        mix: *const u32,
        out: *const u32,
        rou: u32,
        po2: u32,
        domain: u32,
        poly_mix_pows: *const u32,
    ) -> *const std::os::raw::c_char;
}
