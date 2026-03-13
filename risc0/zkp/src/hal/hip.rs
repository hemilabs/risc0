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

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    fmt::Debug,
    marker::PhantomData,
    mem::ManuallyDrop,
    rc::Rc,
    sync::{Once, OnceLock},
};

use anyhow::{bail, Context as _, Result};
use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
use risc0_core::{
    field::{
        baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
        Elem, ExtElem, RootsOfUnity,
    },
    scope,
};
use risc0_sys::{
    cuda::{DevicePointer, *},
    ffi_wrap,
    hip::{
        hip_check, hipDeviceGetAttribute, hipGetDeviceCount, hipInit, hipSetDevice,
        HipDeviceBuffer, HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    },
};

use super::{tracker, Buffer, Hal};
use crate::{
    core::{
        digest::Digest,
        hash::{
            poseidon2::Poseidon2HashSuite, poseidon_254::Poseidon254HashSuite,
            sha::Sha256HashSuite, HashSuite,
        },
        log2_ceil,
    },
    FRI_FOLD,
};

/// Thread-local pool of HIP device buffers, keyed by byte size.
/// Reuses freed buffers to avoid hipMalloc/hipFree overhead.
struct BufferPool {
    cache: HashMap<usize, Vec<HipDeviceBuffer>>,
    total_cached: usize,
}

const BUFFER_POOL_MAX_BYTES: usize = 16 << 30; // 16 GB
const BUFFER_POOL_SMALL_THRESHOLD: usize = 64 << 10; // 64 KB - always pool small buffers

impl BufferPool {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
            total_cached: 0,
        }
    }

    fn pop(&mut self, size: usize) -> Option<HipDeviceBuffer> {
        if let Some(bufs) = self.cache.get_mut(&size) {
            if let Some(buf) = bufs.pop() {
                self.total_cached -= size;
                return Some(buf);
            }
        }
        None
    }

    fn push(&mut self, size: usize, buf: HipDeviceBuffer) {
        // Always pool small buffers (hipFree overhead > storage cost).
        // Only enforce cap for large buffers.
        if size >= BUFFER_POOL_SMALL_THRESHOLD
            && self.total_cached + size > BUFFER_POOL_MAX_BYTES
        {
            drop(buf);
            return;
        }
        self.total_cached += size;
        self.cache.entry(size).or_default().push(buf);
    }
}

thread_local! {
    static BUFFER_POOL: RefCell<BufferPool> = RefCell::new(BufferPool::new());
    /// Which HIP device this thread should use (default 0).
    static THREAD_DEVICE: Cell<i32> = Cell::new(0);
}

/// Auto-configure HSA_ENABLE_SDMA based on detected GPU architecture.
///
/// RDNA4 (gfx12xx) has buggy SDMA and benefits from HSA_ENABLE_SDMA=0.
/// RDNA3 (gfx11xx) has working SDMA and is ~7% slower with it disabled.
/// This must run BEFORE hipInit() since HSA reads the env var during init.
///
/// Reads GPU architectures from Linux sysfs without requiring HIP runtime.
/// Respects any user-set HSA_ENABLE_SDMA value.
fn auto_configure_sdma() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Don't override if user explicitly set it.
        if std::env::var("HSA_ENABLE_SDMA").is_ok() {
            return;
        }

        // Parse HIP_VISIBLE_DEVICES to know which GPUs are active.
        let visible: Option<Vec<usize>> =
            std::env::var("HIP_VISIBLE_DEVICES").ok().map(|val| {
                val.split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect()
            });

        // Read GPU architectures from sysfs (works without HIP runtime).
        // /sys/class/kfd/kfd/topology/nodes/*/properties has gfx_target_version.
        // Node 0 is typically the CPU, GPU nodes start at 1+.
        let mut gpu_index: usize = 0;
        let mut all_rdna4 = true;
        let mut found_gpu = false;

        let Ok(entries) = std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes") else {
            return;
        };
        let mut node_dirs: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        node_dirs.sort_by_key(|e| e.file_name());

        for entry in node_dirs {
            let props_path = entry.path().join("properties");
            let Ok(content) = std::fs::read_to_string(&props_path) else {
                continue;
            };

            // Find gfx_target_version line. CPU nodes have version 0.
            let gfx_version = content.lines().find_map(|line| {
                let line = line.trim();
                if line.starts_with("gfx_target_version") {
                    line.split_whitespace().nth(1)?.parse::<u32>().ok()
                } else {
                    None
                }
            });

            let Some(ver) = gfx_version else { continue };
            if ver == 0 {
                continue; // CPU node
            }

            // This is a GPU node. Check if it's in the visible set.
            let is_visible = match &visible {
                Some(vis) => vis.contains(&gpu_index),
                None => true,
            };

            if is_visible {
                found_gpu = true;
                // gfx12xx = 120000..129999 (RDNA4)
                if ver < 120000 || ver >= 130000 {
                    all_rdna4 = false;
                }
            }

            gpu_index += 1;
        }

        if found_gpu && all_rdna4 {
            tracing::info!("Auto-setting HSA_ENABLE_SDMA=0 for RDNA4 GPU(s)");
            std::env::set_var("HSA_ENABLE_SDMA", "0");
        }
    });
}

/// Set the HIP device ID for the current thread.
/// Must be called BEFORE creating any HipHal instance on this thread.
pub fn set_device_for_thread(device_id: i32) {
    THREAD_DEVICE.with(|d| d.set(device_id));
}

/// Switch the current thread to a different HIP device.
/// Calls both the Rust thread-local and the HIP runtime hipSetDevice.
pub fn switch_to_device(device_id: i32) {
    set_device_for_thread(device_id);
    hip_check(unsafe { hipSetDevice(device_id) });
}

/// Get the HIP device ID for the current thread.
pub fn get_device_for_thread() -> i32 {
    THREAD_DEVICE.with(|d| d.get())
}

/// Return the number of HIP devices available.
pub fn device_count() -> i32 {
    let mut count: i32 = 0;
    let err = unsafe { hipGetDeviceCount(&mut count) };
    if err == risc0_sys::hip::HIP_SUCCESS {
        count
    } else {
        1
    }
}

/// Return the secondary device ID for recursion/Groth16, or None if single-GPU.
/// Reads RISC0_MULTI_GPU env var. Set to "1" to enable multi-GPU.
pub fn recursion_device() -> Option<i32> {
    static CACHE: OnceLock<Option<i32>> = OnceLock::new();
    *CACHE.get_or_init(|| {
        if std::env::var("RISC0_MULTI_GPU").is_ok() && device_count() >= 2 {
            // Use whatever device is NOT the main STARK device (device 0).
            Some(1)
        } else {
            None
        }
    })
}

/// Release all cached GPU buffers back to the HIP runtime.
///
/// The HAL keeps a thread-local pool of freed GPU buffers for reuse.
/// Call this before operations that need the maximum available VRAM
/// (e.g. Groth16 BN254 proving via sppark which uses its own allocator).
pub fn clear_buffer_pool() {
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        pool.cache.clear();
        pool.total_cached = 0;
    });
}

// Per-device locks: prevents concurrent provers on the SAME device.
// Different devices can run concurrently.
pub fn singleton() -> &'static ReentrantMutex<()> {
    singleton_for_device(0)
}

pub fn singleton_for_device(device: i32) -> &'static ReentrantMutex<()> {
    static ONCE: OnceLock<Vec<ReentrantMutex<()>>> = OnceLock::new();
    let locks = ONCE.get_or_init(|| {
        (0..16).map(|_| ReentrantMutex::new(())).collect()
    });
    &locks[device as usize]
}

pub trait HipHash {
    /// Create a hash implementation
    fn new() -> Self
    where
        Self: Sized;

    /// Run the hash_fold function
    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize);

    /// Run the hash_fold_tree function (fold all layers in a single FFI call)
    fn hash_fold_tree(&self, io: &BufferImpl<Digest>, layers: usize) {
        // Default: fall back to per-level hash_fold
        for i in (0..layers).rev() {
            let layer_size = 1 << i;
            self.hash_fold(io, layer_size);
        }
    }

    /// Run the hash_rows function
    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>);

    /// Return the HashSuite
    fn get_hash_suite(&self) -> &HashSuite<BabyBear>;
}

pub struct HipHashSha256 {
    suite: HashSuite<BabyBear>,
}

impl HipHash for HipHashSha256 {
    fn new() -> Self {
        HipHashSha256 {
            suite: Sha256HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        let input = io.as_device_ptr_with_offset(2 * output_size);
        let output = io.as_device_ptr_with_offset(output_size);

        extern "C" {
            fn risc0_zkp_cuda_sha_fold(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe { risc0_zkp_cuda_sha_fold(output, input, output_size as u32) }).unwrap();
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        let row_size = output.size();
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);

        extern "C" {
            fn risc0_zkp_cuda_sha_rows(
                output: DevicePointer<u8>,
                matrix: DevicePointer<u8>,
                row_size: u32,
                col_size: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_sha_rows(
                output.as_device_ptr(),
                matrix.as_device_ptr(),
                row_size as u32,
                col_size as u32,
            )
        })
        .unwrap();
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

pub struct HipHashPoseidon2 {
    suite: HashSuite<BabyBear>,
}

impl HipHash for HipHashPoseidon2 {
    fn new() -> Self {
        HipHashPoseidon2 {
            suite: Poseidon2HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        let err = unsafe {
            let input = io.as_device_ptr_with_offset(2 * output_size);
            let output = io.as_device_ptr_with_offset(output_size);
            sppark_poseidon2_fold(output, input, output_size)
        };
        if err.code != 0 {
            panic!("Failure during hash_fold: {err}");
        }
    }

    fn hash_fold_tree(&self, io: &BufferImpl<Digest>, layers: usize) {
        let err = unsafe {
            sppark_poseidon2_fold_tree(io.as_device_ptr(), layers as u32)
        };
        if err.code != 0 {
            panic!("Failure during hash_fold_tree: {err}");
        }
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        let row_size = output.size();
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);

        let err = unsafe {
            sppark_poseidon2_rows(
                output.as_device_ptr(),
                matrix.as_device_ptr(),
                row_size.try_into().unwrap(),
                col_size.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during hash_rows: {err}");
        }
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

pub struct HipHashPoseidon254 {
    suite: HashSuite<BabyBear>,
}

impl HipHash for HipHashPoseidon254 {
    fn new() -> Self {
        HipHashPoseidon254 {
            suite: Poseidon254HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        let err = unsafe {
            let input = io.as_device_ptr_with_offset(2 * output_size);
            let output = io.as_device_ptr_with_offset(output_size);
            sppark_poseidon254_fold(output, input, output_size)
        };
        if err.code != 0 {
            panic!("Failure during hash_fold: {err}");
        }
    }

    fn hash_fold_tree(&self, io: &BufferImpl<Digest>, layers: usize) {
        let err = unsafe {
            sppark_poseidon254_fold_tree(io.as_device_ptr(), layers as u32)
        };
        if err.code != 0 {
            panic!("Failure during hash_fold_tree: {err}");
        }
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        let row_size = output.size();
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);

        let err = unsafe {
            sppark_poseidon254_rows(
                output.as_device_ptr(),
                matrix.as_device_ptr(),
                row_size,
                col_size.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during hash_rows 254: {err}");
        }
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

pub struct HipHal<Hash: HipHash + ?Sized> {
    pub max_threads: u32,
    hash: Option<Box<Hash>>,
    _lock: ReentrantMutexGuard<'static, ()>,
}

pub type HipHalSha256 = HipHal<HipHashSha256>;
pub type HipHalPoseidon2 = HipHal<HipHashPoseidon2>;
pub type HipHalPoseidon254 = HipHal<HipHashPoseidon254>;

struct RawBuffer {
    name: &'static str,
    buf: ManuallyDrop<HipDeviceBuffer>,
}

impl RawBuffer {
    pub fn new(name: &'static str, size: usize) -> Self {
        tracing::trace!("alloc: {size} bytes, {name}");
        tracker().lock().unwrap().alloc(size);
        let buf = BUFFER_POOL
            .with(|pool| pool.borrow_mut().pop(size))
            .unwrap_or_else(|| {
                HipDeviceBuffer::uninitialized(size).unwrap_or_else(|_| {
                    // OOM: clear the buffer pool to free cached GPU memory and retry.
                    clear_buffer_pool();
                    // Clear the stale HIP error from the failed hipMalloc above.
                    // Without this, sppark's cudaGetLastError() picks up the old
                    // "out of memory" error and panics even though the retry succeeds.
                    unsafe { risc0_sys::hip::hipGetLastError(); }
                    HipDeviceBuffer::uninitialized(size)
                        .context(format!("allocation failed on {name}: {size} bytes"))
                        .unwrap()
                })
            });
        Self {
            name,
            buf: ManuallyDrop::new(buf),
        }
    }

    #[allow(dead_code)]
    pub fn set_u32(&mut self, value: u32) {
        self.buf.set_32(value);
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        let size = self.buf.len();
        tracing::trace!("free: {size} bytes, {}", self.name);
        tracker().lock().unwrap().free(size);
        // Cache the buffer for reuse instead of calling hipFree.
        // Safety: self.buf is not accessed after take() since we're in Drop,
        // and ManuallyDrop's own drop is a no-op.
        let buf = unsafe { ManuallyDrop::take(&mut self.buf) };
        // During thread shutdown, TLS may already be destroyed. Use try_with
        // to gracefully handle this — if the pool is gone, just drop the buffer
        // directly (hipFree via HipDeviceBuffer's own Drop).
        match BUFFER_POOL.try_with(|pool| pool.borrow_mut().push(size, buf)) {
            Ok(()) => {}
            Err(_) => {
                // TLS destroyed; closure was never called, so `buf` (captured
                // by the closure) drops here via HipDeviceBuffer::drop (hipFree).
            }
        }
    }
}

#[derive(Clone)]
pub struct BufferImpl<T> {
    buffer: Rc<RefCell<RawBuffer>>,
    size: usize,
    offset: usize,
    marker: PhantomData<T>,
}

#[inline]
fn unchecked_cast<A, B>(a: &[A]) -> &[B] {
    let new_len = std::mem::size_of_val(a) / std::mem::size_of::<B>();
    unsafe { std::slice::from_raw_parts(a.as_ptr() as *const B, new_len) }
}

#[inline]
fn unchecked_cast_mut<A, B>(a: &mut [A]) -> &mut [B] {
    let new_len = std::mem::size_of_val(a) / std::mem::size_of::<B>();
    unsafe { std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut B, new_len) }
}

impl<T> BufferImpl<T> {
    fn new(name: &'static str, size: usize) -> Self {
        let bytes_len = std::mem::size_of::<T>() * size;
        assert!(bytes_len > 0);
        BufferImpl {
            buffer: Rc::new(RefCell::new(RawBuffer::new(name, bytes_len))),
            size,
            offset: 0,
            marker: PhantomData,
        }
    }

    pub fn copy_from(name: &'static str, slice: &[T]) -> Self {
        // scope!("copy_from");
        let bytes_len = std::mem::size_of_val(slice);
        assert!(bytes_len > 0);
        let buffer = RawBuffer::new(name, bytes_len);
        // Use async H2D copy on persistent stream to avoid device-wide sync from hipMemcpy.
        extern "C" {
            fn risc0_zkp_cuda_memcpy_h2d(
                dst: *mut std::os::raw::c_void,
                src: *const std::os::raw::c_void,
                size: usize,
            ) -> *const std::os::raw::c_char;
        }
        let bytes = unchecked_cast(slice);
        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_memcpy_h2d(
                buffer.buf.as_device_ptr().0 as *mut std::os::raw::c_void,
                bytes.as_ptr() as *const std::os::raw::c_void,
                bytes.len(),
            )
        })
        .unwrap();

        BufferImpl {
            buffer: Rc::new(RefCell::new(buffer)),
            size: slice.len(),
            offset: 0,
            marker: PhantomData,
        }
    }

    pub fn as_device_ptr(&self) -> DevicePointer<u8> {
        let ptr = self.buffer.borrow_mut().buf.as_device_ptr();
        let offset = self.offset * std::mem::size_of::<T>();
        unsafe { ptr.offset(offset.try_into().unwrap()) }
    }

    pub fn as_device_ptr_with_offset(&self, offset: usize) -> DevicePointer<u8> {
        let ptr = self.buffer.borrow_mut().buf.as_device_ptr();
        let offset = (self.offset + offset) * std::mem::size_of::<T>();
        unsafe { ptr.offset(offset.try_into().unwrap()) }
    }
}

impl<T: Clone> Buffer<T> for BufferImpl<T> {
    fn name(&self) -> &'static str {
        self.buffer.borrow().name
    }

    fn size(&self) -> usize {
        self.size
    }

    fn slice(&self, offset: usize, size: usize) -> BufferImpl<T> {
        assert!(offset + size <= self.size());
        BufferImpl {
            buffer: self.buffer.clone(),
            size,
            offset: self.offset + offset,
            marker: PhantomData,
        }
    }

    fn get_at(&self, idx: usize) -> T {
        let item_size = std::mem::size_of::<T>();
        let buf = self.buffer.borrow_mut();
        let offset = (self.offset + idx) * item_size;
        // Read just the single element from device memory (ranged D2H)
        let host_buf = buf.buf.as_host_vec_range(offset, item_size).unwrap();
        let slice: &[T] = unchecked_cast(&host_buf[..]);
        slice[0].clone()
    }

    fn view<F: FnOnce(&[T])>(&self, f: F) {
        scope!("view");
        let item_size = std::mem::size_of::<T>();
        let buf = self.buffer.borrow_mut();
        let offset = self.offset * item_size;
        let len = self.size * item_size;
        // Use ranged D2H copy: only transfer the needed slice, not the full buffer.
        let host_buf = buf.buf.as_host_vec_range(offset, len).unwrap();
        let slice: &[T] = unchecked_cast(&host_buf[..]);
        f(slice);
    }

    fn view_mut<F: FnOnce(&mut [T])>(&self, f: F) {
        scope!("view_mut");
        let mut buf = self.buffer.borrow_mut();
        let mut host_buf = buf.buf.as_host_vec().unwrap();
        let slice = unchecked_cast_mut(&mut host_buf);
        f(&mut slice[self.offset..]);
        buf.buf.copy_from(&host_buf).unwrap();
    }

    fn to_vec(&self) -> Vec<T> {
        let item_size = std::mem::size_of::<T>();
        let buf = self.buffer.borrow_mut();
        let offset = self.offset * item_size;
        let len = self.size * item_size;
        // Use ranged D2H copy: only transfer the needed slice, not the full buffer.
        let host_buf = buf.buf.as_host_vec_range(offset, len).unwrap();
        let slice: &[T] = unchecked_cast(&host_buf[..]);
        slice.to_vec()
    }
}

impl<HH: HipHash> Default for HipHal<HH> {
    fn default() -> Self {
        Self::new()
    }
}

impl<HH: HipHash + ?Sized> HipHal<HH> {
    pub fn new() -> Self
    where
        HH: Sized,
    {
        Self::new_from_hash(Box::new(HH::new()))
    }

    fn new_from_hash(hash: Box<HH>) -> Self {
        let device = get_device_for_thread();
        let _lock = singleton_for_device(device).lock();

        // Auto-configure SDMA before HIP runtime init reads env vars.
        auto_configure_sdma();

        // Set device BEFORE sppark_init so NTT tables land on the right GPU.
        hip_check(unsafe { hipInit(0) });
        hip_check(unsafe { hipSetDevice(device) });

        let err = unsafe { sppark_init() };
        if err.code != 0 {
            panic!("Failure during sppark_init: {err}");
        }

        let mut max_threads: i32 = 0;
        hip_check(unsafe {
            hipDeviceGetAttribute(
                &mut max_threads,
                HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                device,
            )
        });

        let mut hal = Self {
            max_threads: max_threads as u32,
            hash: None,
            _lock,
        };
        hal.hash = Some(hash);
        hal
    }

    /// Synchronize the risc0 persistent CUDA/HIP stream.
    /// Must be called before sppark operations that read from buffers
    /// last written by risc0 kernels (which use a different stream).
    fn sync_stream() {
        extern "C" {
            fn risc0_zkp_cuda_sync_stream() -> *const std::os::raw::c_char;
        }
        ffi_wrap(|| unsafe { risc0_zkp_cuda_sync_stream() }).unwrap();
    }

    #[allow(dead_code)]
    fn poly_divide(
        &self,
        polynomial: &BufferImpl<BabyBearExtElem>,
        pow: BabyBearExtElem,
    ) -> BabyBearExtElem {
        let mut remainder = BabyBearExtElem::ZERO;
        let poly_size = polynomial.size();
        let pow = pow.to_u32_words();

        let err = unsafe {
            supra_poly_divide(
                polynomial.as_device_ptr(),
                poly_size,
                &mut remainder as *mut _ as *mut u32,
                pow.as_ptr(),
            )
        };

        if err.code != 0 {
            panic!("Failure during supra_poly_divide: {err}");
        }

        remainder
    }

    #[allow(dead_code)]
    fn poly_divide_batch(
        &self,
        polynomial: &BufferImpl<BabyBearExtElem>,
        pows: &[BabyBearExtElem],
    ) -> Vec<BabyBearExtElem> {
        let num_divides = pows.len();
        let poly_size = polynomial.size();
        let mut remainders = vec![BabyBearExtElem::ZERO; num_divides];
        let pows_words: Vec<u32> = pows.iter().flat_map(|p| p.to_u32_words()).collect();
        Self::sync_stream();

        let err = unsafe {
            supra_poly_divide_batch(
                polynomial.as_device_ptr(),
                poly_size,
                remainders.as_mut_ptr() as *mut u32,
                pows_words.as_ptr(),
                num_divides as u32,
            )
        };

        if err.code != 0 {
            panic!("Failure during supra_poly_divide_batch: {err}");
        }

        remainders
    }
}

impl HipHal<dyn HipHash> {
    pub fn new_from_hash_suite(hash_suite: HashSuite<BabyBear>) -> Result<Self> {
        let hash_suite_box = match &hash_suite.name[..] {
            "poseidon2" => Box::new(HipHashPoseidon2::new()) as Box<dyn HipHash>,
            "poseidon254" => Box::new(HipHashPoseidon254::new()) as Box<dyn HipHash>,
            "sha-256" => Box::new(HipHashSha256::new()) as Box<dyn HipHash>,
            other => bail!("unsupported hash_fn {other}"),
        };
        Ok(Self::new_from_hash(hash_suite_box))
    }
}

impl<HH: HipHash + ?Sized> Hal for HipHal<HH> {
    type Field = BabyBear;
    type Elem = BabyBearElem;
    type ExtElem = BabyBearExtElem;
    type Buffer<T: Clone + Debug + PartialEq> = BufferImpl<T>;

    fn alloc_elem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::Elem> {
        BufferImpl::new(name, size)
    }

    fn alloc_elem_init(
        &self,
        name: &'static str,
        size: usize,
        value: Self::Elem,
    ) -> Self::Buffer<Self::Elem> {
        let buffer = self.alloc_elem(name, size);
        // Use async fill on persistent stream to avoid device-wide sync from hipMemsetD32.
        extern "C" {
            fn risc0_zkp_cuda_fill_u32(
                buf: DevicePointer<u8>,
                value: u32,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }
        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_fill_u32(
                buffer.as_device_ptr(),
                value.as_u32_montgomery(),
                size as u32,
            )
        })
        .unwrap();
        buffer
    }

    fn copy_from_elem(&self, name: &'static str, slice: &[Self::Elem]) -> Self::Buffer<Self::Elem> {
        BufferImpl::copy_from(name, slice)
    }

    fn alloc_extelem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem> {
        BufferImpl::new(name, size)
    }

    fn alloc_extelem_zeroed(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem> {
        let buffer = self.alloc_extelem(name, size);
        // Use async fill on persistent stream to avoid device-wide sync from hipMemset.
        extern "C" {
            fn risc0_zkp_cuda_fill_u32(
                buf: DevicePointer<u8>,
                value: u32,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }
        let ext_count = size * std::mem::size_of::<Self::ExtElem>() / 4;
        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_fill_u32(buffer.as_device_ptr(), 0, ext_count as u32)
        })
        .unwrap();
        buffer
    }

    fn copy_from_extelem(
        &self,
        name: &'static str,
        slice: &[Self::ExtElem],
    ) -> Self::Buffer<Self::ExtElem> {
        BufferImpl::copy_from(name, slice)
    }

    fn alloc_digest(&self, name: &'static str, size: usize) -> Self::Buffer<Digest> {
        BufferImpl::new(name, size)
    }

    fn copy_from_digest(&self, name: &'static str, slice: &[Digest]) -> Self::Buffer<Digest> {
        BufferImpl::copy_from(name, slice)
    }

    fn alloc_u32(&self, name: &'static str, size: usize) -> Self::Buffer<u32> {
        BufferImpl::new(name, size)
    }

    fn copy_from_u32(&self, name: &'static str, slice: &[u32]) -> Self::Buffer<u32> {
        BufferImpl::copy_from(name, slice)
    }

    fn batch_expand_into_evaluate_ntt(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
        poly_count: usize,
        _expand_bits: usize,
    ) {
        let out_size = output.size() / poly_count;
        let in_size = input.size() / poly_count;
        let expand_bits = log2_ceil(out_size / in_size);
        assert_eq!(output.size(), out_size * poly_count);
        assert_eq!(input.size(), in_size * poly_count);
        assert_eq!(out_size, in_size * (1 << expand_bits));
        let in_bits = log2_ceil(in_size);

        let row_size = output.size() / poly_count;
        assert_eq!(row_size * poly_count, output.size());
        let n_bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << n_bits);
        assert!(n_bits >= expand_bits);
        assert!(n_bits < Self::Elem::MAX_ROU_PO2);
        Self::sync_stream();

        let err = unsafe {
            sppark_batch_expand_NTT(
                output.as_device_ptr(),
                input.as_device_ptr(),
                in_bits.try_into().unwrap(),
                expand_bits.try_into().unwrap(),
                poly_count.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during batch_expand_NTT: {err}");
        }
    }

    fn batch_interpolate_ntt(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let n_bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << n_bits);
        assert!(n_bits < Self::Elem::MAX_ROU_PO2);
        Self::sync_stream();

        let err = unsafe {
            sppark_batch_iNTT(
                io.as_device_ptr(),
                n_bits.try_into().unwrap(),
                count.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during batch_interpolate_ntt: {err}");
        }

    }

    fn batch_bit_reverse(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << bits);

        // Sync persistent stream before sppark operates on the buffer.
        Self::sync_stream();

        let err = unsafe {
            sppark_batch_bit_reverse(io.as_device_ptr(), bits as u32, count as u32)
        };
        if err.code != 0 {
            panic!("Failure during sppark_batch_bit_reverse: {err}");
        }
    }

    fn batch_evaluate_any(
        &self,
        coeffs: &Self::Buffer<Self::Elem>,
        poly_count: usize,
        which: &Self::Buffer<u32>,
        xs: &Self::Buffer<Self::ExtElem>,
        out: &Self::Buffer<Self::ExtElem>,
    ) {
        let po2 = log2_ceil(coeffs.size() / poly_count);
        let count = 1 << po2;
        assert_eq!(poly_count * count, coeffs.size());
        let eval_count = which.size();
        assert_eq!(xs.size(), eval_count);
        assert_eq!(out.size(), eval_count);

        let threads_per_block = self.max_threads / 4;
        const BYTES_PER_WORD: u32 = 4;
        const WORDS_PER_FPEXT: u32 = 4;
        let shared_size = threads_per_block * BYTES_PER_WORD * WORDS_PER_FPEXT;
        let kernel_count = out.size() * threads_per_block as usize;

        extern "C" {
            fn risc0_zkp_cuda_batch_evaluate_any(
                output: DevicePointer<u8>,
                coeffs: DevicePointer<u8>,
                which: DevicePointer<u8>,
                xs: DevicePointer<u8>,
                shared_size: u32,
                kernel_count: u32,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_batch_evaluate_any(
                out.as_device_ptr(),
                coeffs.as_device_ptr(),
                which.as_device_ptr(),
                xs.as_device_ptr(),
                shared_size,
                kernel_count as u32,
                count as u32,
            )
        })
        .unwrap();
    }

    fn gather_sample(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        idx: usize,
        size: usize,
        stride: usize,
    ) {
        extern "C" {
            fn risc0_zkp_cuda_gather_sample(
                dst: DevicePointer<u8>,
                src: DevicePointer<u8>,
                idx: u32,
                size: u32,
                stride: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_gather_sample(
                dst.as_device_ptr(),
                src.as_device_ptr(),
                idx as u32,
                size as u32,
                stride as u32,
            )
        })
        .unwrap();
    }

    fn has_unified_memory(&self) -> bool {
        false
    }

    fn gpu_free_memory(&self) -> usize {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let err = unsafe { risc0_sys::hip::hipMemGetInfo(&mut free, &mut total) };
        if err == risc0_sys::hip::HIP_SUCCESS {
            // Add pool contents to free count, since those are reclaimable.
            let pool_bytes = BUFFER_POOL.with(|pool| pool.borrow().total_cached);
            free + pool_bytes
        } else {
            usize::MAX
        }
    }

    fn gpu_total_memory(&self) -> usize {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let err = unsafe { risc0_sys::hip::hipMemGetInfo(&mut free, &mut total) };
        if err == risc0_sys::hip::HIP_SUCCESS {
            total
        } else {
            usize::MAX
        }
    }

    fn batch_get_digest_at(&self, buf: &Self::Buffer<Digest>, indices: &[usize]) -> Vec<Digest> {
        if indices.is_empty() {
            return Vec::new();
        }
        let count = indices.len();
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let indices_buf = self.copy_from_u32("gather_indices", &indices_u32);
        let output_buf: BufferImpl<Digest> = BufferImpl::new("gather_output", count);

        extern "C" {
            fn risc0_zkp_cuda_gather_digests(
                dst: DevicePointer<u8>,
                src: DevicePointer<u8>,
                indices: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_gather_digests(
                output_buf.as_device_ptr(),
                buf.as_device_ptr(),
                indices_buf.as_device_ptr(),
                count as u32,
            )
        })
        .unwrap();

        let mut result = Vec::new();
        output_buf.view(|view| {
            result = view.to_vec();
        });
        result
    }

    fn zk_shift(&self, io: &Self::Buffer<Self::Elem>, poly_count: usize) {
        let bits = log2_ceil(io.size() / poly_count);
        assert_eq!(io.size(), poly_count * (1 << bits));

        let err = unsafe {
            sppark_batch_zk_shift(
                io.as_device_ptr(),
                bits.try_into().unwrap(),
                poly_count.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during zk_shift: {err}");
        }
    }

    fn batch_interpolate_ntt_zk_shift(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let n_bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << n_bits);
        assert!(n_bits < Self::Elem::MAX_ROU_PO2);
        Self::sync_stream();

        let err = unsafe {
            sppark_batch_iNTT_zk_shift(
                io.as_device_ptr(),
                n_bits.try_into().unwrap(),
                count.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during batch_iNTT_zk_shift: {err}");
        }
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
        let mix_start = self.copy_from_extelem("mix_start", &[*mix_start]);
        let mix = self.copy_from_extelem("mix", &[*mix]);

        extern "C" {
            fn risc0_zkp_cuda_mix_poly_coeffs(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                combos: DevicePointer<u8>,
                mix_start: DevicePointer<u8>,
                mix: DevicePointer<u8>,
                input_size: u32,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_mix_poly_coeffs(
                output.as_device_ptr(),
                input.as_device_ptr(),
                combos.as_device_ptr(),
                mix_start.as_device_ptr(),
                mix.as_device_ptr(),
                input_size as u32,
                count as u32,
            )
        })
        .unwrap();
    }

    fn eltwise_add_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input1: &Self::Buffer<Self::Elem>,
        input2: &Self::Buffer<Self::Elem>,
    ) {
        assert_eq!(output.size(), input1.size());
        assert_eq!(output.size(), input2.size());
        let count = output.size();

        extern "C" {
            fn risc0_zkp_cuda_eltwise_add_fp(
                out: DevicePointer<u8>,
                x: DevicePointer<u8>,
                y: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_eltwise_add_fp(
                output.as_device_ptr(),
                input1.as_device_ptr(),
                input2.as_device_ptr(),
                count as u32,
            )
        })
        .unwrap();
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

        extern "C" {
            fn risc0_zkp_cuda_eltwise_sum_fpext(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                to_add: u32,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_eltwise_sum_fpext(
                output.as_device_ptr(),
                input.as_device_ptr(),
                to_add as u32,
                count as u32,
            )
        })
        .unwrap();
    }

    fn eltwise_copy_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
    ) {
        let count = output.size();
        assert_eq!(count, input.size());

        extern "C" {
            fn risc0_zkp_cuda_eltwise_copy_fp(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_eltwise_copy_fp(
                output.as_device_ptr(),
                input.as_device_ptr(),
                count as u32,
            )
        })
        .unwrap();
    }

    fn eltwise_zeroize_elem(&self, elems: &Self::Buffer<Self::Elem>) {
        extern "C" {
            fn risc0_zkp_cuda_eltwise_zeroize_fp(
                elems: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_eltwise_zeroize_fp(elems.as_device_ptr(), elems.size() as u32)
        })
        .unwrap();
    }

    fn scatter(
        &self,
        into: &Self::Buffer<Self::Elem>,
        index: &[u32],
        offsets: &[u32],
        values: &[Self::Elem],
    ) {
        if index.is_empty() {
            return;
        }

        let count = index.len() - 1;
        if count == 0 {
            return;
        }

        extern "C" {
            fn risc0_zkp_cuda_scatter_from_host(
                into: DevicePointer<u8>,
                h_index: *const u32,
                index_count: u32,
                h_offsets: *const u32,
                offsets_count: u32,
                h_values: *const u8,
                values_count: u32,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_scatter_from_host(
                into.as_device_ptr(),
                index.as_ptr(),
                index.len() as u32,
                offsets.as_ptr(),
                offsets.len() as u32,
                values.as_ptr() as *const u8,
                values.len() as u32,
                count as u32,
            )
        })
        .unwrap();
    }

    fn scatter_bits(
        &self,
        into: &Self::Buffer<Self::Elem>,
        bit_data: &[u32],
        cycles: u32,
    ) {
        let count = bit_data.len() / 3;
        if count == 0 {
            return;
        }

        extern "C" {
            fn risc0_zkp_cuda_scatter_bits_from_host(
                into: DevicePointer<u8>,
                h_data: *const u32,
                triplet_count: u32,
                cycles: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_scatter_bits_from_host(
                into.as_device_ptr(),
                bit_data.as_ptr(),
                count as u32,
                cycles,
            )
        })
        .unwrap();
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
        let from = self.copy_from_elem("from", from);

        extern "C" {
            fn risc0_zkp_cuda_eltwise_copy_fp_region(
                into: DevicePointer<u8>,
                from: DevicePointer<u8>,
                from_rows: u32,
                from_cols: u32,
                from_offset: u32,
                from_stride: u32,
                into_offset: u32,
                into_stride: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_eltwise_copy_fp_region(
                into.as_device_ptr(),
                from.as_device_ptr(),
                from_rows as u32,
                from_cols as u32,
                from_offset as u32,
                from_stride as u32,
                into_offset as u32,
                into_stride as u32,
            )
        })
        .unwrap();
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
        let mix = self.copy_from_extelem("mix", &[*mix]);

        extern "C" {
            fn risc0_zkp_cuda_fri_fold(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                mix: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_fri_fold(
                output.as_device_ptr(),
                input.as_device_ptr(),
                mix.as_device_ptr(),
                count as u32,
            )
        })
        .unwrap();
    }

    fn hash_fold(&self, io: &Self::Buffer<Digest>, input_size: usize, output_size: usize) {
        assert_eq!(input_size, 2 * output_size);
        self.hash.as_ref().unwrap().hash_fold(io, output_size);
    }

    fn hash_fold_tree(&self, io: &Self::Buffer<Digest>, layers: usize) {
        self.hash.as_ref().unwrap().hash_fold_tree(io, layers);
    }

    fn hash_rows(&self, output: &Self::Buffer<Digest>, matrix: &Self::Buffer<Self::Elem>) {
        self.hash.as_ref().unwrap().hash_rows(output, matrix);
    }

    fn get_hash_suite(&self) -> &HashSuite<Self::Field> {
        self.hash.as_ref().unwrap().get_hash_suite()
    }

    fn prefix_products(&self, io: &Self::Buffer<Self::ExtElem>) {
        io.view_mut(|io| {
            for i in 1..io.len() {
                io[i] *= io[i - 1];
            }
        });
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
        let coeff_u = self.copy_from_extelem("coeff_u", coeff_u);
        let combo_count = combo_count as u32;
        let cycles = cycles as u32;
        let regs_count = reg_sizes.len() as u32;
        let reg_sizes = self.copy_from_u32("reg_sizes", reg_sizes);
        let reg_combo_ids = self.copy_from_u32("reg_combo_ids", reg_combo_ids);
        let mix = self.copy_from_extelem("mix", &[*mix]);

        extern "C" {
            fn risc0_zkp_cuda_combos_prepare(
                combos: DevicePointer<u8>,
                coeff_u: DevicePointer<u8>,
                combo_count: u32,
                cycles: u32,
                regs_count: u32,
                reg_sizes: DevicePointer<u8>,
                reg_combo_ids: DevicePointer<u8>,
                checkSize: u32,
                mix: DevicePointer<u8>,
            ) -> *const std::os::raw::c_char;
        }

        ffi_wrap(|| unsafe {
            risc0_zkp_cuda_combos_prepare(
                combos.as_device_ptr(),
                coeff_u.as_device_ptr(),
                combo_count,
                cycles,
                regs_count,
                reg_sizes.as_device_ptr(),
                reg_combo_ids.as_device_ptr(),
                Self::CHECK_SIZE as u32,
                mix.as_device_ptr(),
            )
        })
        .unwrap();
    }

    fn combos_divide(
        &self,
        combos: &Self::Buffer<Self::ExtElem>,
        chunks: Vec<(usize, Vec<Self::ExtElem>)>,
        cycles: usize,
    ) {
        scope!("combos_divide");
        let combo_indices: Vec<u32> = chunks.iter().map(|(i, _)| *i as u32).collect();
        let pows_per_combo: Vec<u32> = chunks.iter().map(|(_, pows)| pows.len() as u32).collect();
        let all_pows_flat: Vec<u32> = chunks
            .iter()
            .flat_map(|(_, pows)| pows.iter().flat_map(|p| p.to_u32_words()))
            .collect();

        Self::sync_stream();
        let err = unsafe {
            supra_poly_divide_multi(
                combos.as_device_ptr(),
                cycles,
                combo_indices.as_ptr(),
                pows_per_combo.as_ptr(),
                all_pows_flat.as_ptr(),
                chunks.len() as u32,
            )
        };
        if err.code != 0 {
            panic!("Failure during supra_poly_divide_multi: {err}");
        }
    }
}

#[cfg(test)]
mod tests {
    use test_log::test;

    use super::{HipHalPoseidon2, HipHalSha256};
    use crate::hal::testutil;

    #[test]
    #[should_panic]
    fn check_req() {
        testutil::check_req(HipHalSha256::new());
    }

    #[test]
    fn eltwise_add_elem() {
        testutil::eltwise_add_elem(HipHalSha256::new());
    }

    #[test]
    fn eltwise_copy_elem() {
        testutil::eltwise_copy_elem(HipHalSha256::new());
    }

    #[test]
    fn eltwise_sum_extelem() {
        testutil::eltwise_sum_extelem(HipHalSha256::new());
    }

    #[test]
    fn hash_rows_sha256() {
        testutil::hash_rows(HipHalSha256::new());
    }

    #[test]
    fn hash_fold_sha256() {
        testutil::hash_fold(HipHalSha256::new());
    }

    #[test]
    fn hash_rows_poseidon2() {
        testutil::hash_rows(HipHalPoseidon2::new());
    }

    #[test]
    fn hash_fold_poseidon2() {
        testutil::hash_fold(HipHalPoseidon2::new());
    }

    #[test]
    fn fri_fold() {
        testutil::fri_fold(HipHalSha256::new());
    }

    #[test]
    fn batch_expand_into_evaluate_ntt() {
        testutil::batch_expand_into_evaluate_ntt(HipHalSha256::new());
    }

    #[test]
    fn batch_interpolate_ntt() {
        testutil::batch_interpolate_ntt(HipHalSha256::new());
    }

    #[test]
    fn batch_bit_reverse() {
        testutil::batch_bit_reverse(HipHalSha256::new());
    }

    #[test]
    fn batch_evaluate_any() {
        testutil::batch_evaluate_any(HipHalSha256::new());
    }

    #[test]
    fn gather_sample() {
        testutil::gather_sample(HipHalSha256::new());
    }

    #[test]
    fn zk_shift() {
        testutil::zk_shift(HipHalSha256::new());
    }

    #[test]
    fn mix_poly_coeffs() {
        testutil::mix_poly_coeffs(HipHalSha256::new());
    }
}
