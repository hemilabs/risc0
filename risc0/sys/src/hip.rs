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

//! Raw HIP runtime FFI bindings for ROCm GPU support.
//! Replaces the `cust` crate which only supports CUDA.

use std::ffi::c_void;
use std::os::raw::c_int;

pub const HIP_SUCCESS: i32 = 0;
pub const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;

// hipDeviceAttribute_t values we need
pub const HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 56;

extern "C" {
    pub fn hipInit(flags: u32) -> c_int;
    pub fn hipGetDevice(device: *mut c_int) -> c_int;
    pub fn hipSetDevice(device: c_int) -> c_int;
    pub fn hipGetDeviceCount(count: *mut c_int) -> c_int;
    pub fn hipDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
    pub fn hipDeviceSynchronize() -> c_int;

    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    pub fn hipFree(ptr: *mut c_void) -> c_int;
    pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> c_int;
    pub fn hipMemset(dst: *mut c_void, value: c_int, size: usize) -> c_int;
    pub fn hipMemsetD32(dst: *mut c_void, value: c_int, count: usize) -> c_int;

    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;

    pub fn hipGetErrorString(error: c_int) -> *const std::os::raw::c_char;
    pub fn hipGetLastError() -> c_int;
}

/// Check a HIP return code and panic with the error string if it failed.
#[inline]
pub fn hip_check(err: c_int) {
    if err != HIP_SUCCESS {
        let msg = unsafe {
            let ptr = hipGetErrorString(err);
            if ptr.is_null() {
                "unknown HIP error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_str()
                    .unwrap_or("invalid error string")
                    .to_string()
            }
        };
        panic!("HIP error {err}: {msg}");
    }
}

/// RAII wrapper for HIP device memory, replacing cust::DeviceBuffer<u8>.
pub struct HipDeviceBuffer {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for HipDeviceBuffer {}

impl HipDeviceBuffer {
    /// Allocate uninitialized device memory.
    pub fn uninitialized(size: usize) -> anyhow::Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { hipMalloc(&mut ptr, size) };
        if err != HIP_SUCCESS {
            anyhow::bail!("hipMalloc failed for {size} bytes (error {err})");
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            len: size,
        })
    }

    /// Copy host data to device.
    pub fn copy_from(&mut self, data: &[u8]) -> anyhow::Result<()> {
        assert!(data.len() <= self.len);
        hip_check(unsafe {
            hipMemcpy(
                self.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                data.len(),
                HIP_MEMCPY_HOST_TO_DEVICE,
            )
        });
        Ok(())
    }

    /// Copy device data to a new host Vec.
    pub fn as_host_vec(&self) -> anyhow::Result<Vec<u8>> {
        let mut vec = vec![0u8; self.len];
        hip_check(unsafe {
            hipMemcpy(
                vec.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                self.len,
                HIP_MEMCPY_DEVICE_TO_HOST,
            )
        });
        Ok(vec)
    }

    /// Copy a range of device data to a new host Vec.
    /// `offset` and `len` are in bytes.
    pub fn as_host_vec_range(&self, offset: usize, len: usize) -> anyhow::Result<Vec<u8>> {
        assert!(offset + len <= self.len, "range [{offset}..{}] exceeds buffer size {}", offset + len, self.len);
        let mut vec = vec![0u8; len];
        hip_check(unsafe {
            hipMemcpy(
                vec.as_mut_ptr() as *mut c_void,
                self.ptr.add(offset) as *const c_void,
                len,
                HIP_MEMCPY_DEVICE_TO_HOST,
            )
        });
        Ok(vec)
    }

    /// Set all 32-bit words to `value`. Equivalent to cust's set_32().
    pub fn set_32(&mut self, value: u32) {
        if value == 0 {
            hip_check(unsafe { hipMemset(self.ptr as *mut c_void, 0, self.len) });
        } else {
            let count = self.len / 4;
            hip_check(unsafe {
                hipMemsetD32(self.ptr as *mut c_void, value as c_int, count)
            });
        }
    }

    /// Get the device pointer.
    pub fn as_device_ptr(&self) -> super::cuda::DevicePointer<u8> {
        super::cuda::DevicePointer(self.ptr)
    }

    /// Get the buffer length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for HipDeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hipFree(self.ptr as *mut c_void) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
