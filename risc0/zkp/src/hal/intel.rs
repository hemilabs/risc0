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

//! Hardware Abstraction Layer (HAL) for Intel GPU acceleration via SYCL/ESIMD.

use std::{
    cell::RefCell,
    fmt::Debug,
    marker::PhantomData,
    mem::ManuallyDrop,
    rc::Rc,
    sync::OnceLock,
};

use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
use risc0_core::{
    field::{
        baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
        ExtElem, RootsOfUnity,
    },
    scope,
};
use risc0_sys::intel::{
    self, esimd_check, get_queue, DevicePointer, IntelDeviceBuffer,
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

// ============================================================================
// Singleton lock (one IntelHal per process)
// ============================================================================

pub fn singleton() -> &'static ReentrantMutex<()> {
    static ONCE: OnceLock<ReentrantMutex<()>> = OnceLock::new();
    ONCE.get_or_init(|| ReentrantMutex::new(()))
}

// ============================================================================
// IntelHash trait and implementations
// ============================================================================

pub trait IntelHash {
    /// Create a hash implementation
    fn new() -> Self
    where
        Self: Sized;

    /// Run the hash_fold function
    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize);

    /// Run the hash_rows function
    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>);

    /// Return the HashSuite
    fn get_hash_suite(&self) -> &HashSuite<BabyBear>;
}

// ---- Poseidon2 (ESIMD GPU) ----

pub struct IntelHashPoseidon2 {
    suite: HashSuite<BabyBear>,
}

impl IntelHash for IntelHashPoseidon2 {
    fn new() -> Self {
        IntelHashPoseidon2 {
            suite: Poseidon2HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        let input_ptr = io.as_device_ptr_with_offset(2 * output_size);
        let output_ptr = io.as_device_ptr_with_offset(output_size);
        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_poseidon2_hash_fold(
                queue,
                output_ptr.0 as *mut std::ffi::c_void,
                input_ptr.0 as *const std::ffi::c_void,
                output_size as u32,
            )
        });
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        let row_size = output.size();
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);
        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_poseidon2_hash_rows(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                matrix.as_device_ptr().0 as *const std::ffi::c_void,
                row_size as u32,
                col_size as u32,
            )
        });
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

// ---- SHA-256 (CPU fallback) ----

pub struct IntelHashSha256 {
    suite: HashSuite<BabyBear>,
}

impl IntelHash for IntelHashSha256 {
    fn new() -> Self {
        IntelHashSha256 {
            suite: Sha256HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        // CPU fallback with rayon parallelism
        io.view_mut(|buf| {
            let hashfn: &dyn crate::core::hash::HashFn<BabyBear> = self.suite.hashfn.as_ref();
            let (_, rest) = buf.split_at_mut(output_size);
            let (out_slice, in_slice) = rest.split_at_mut(output_size);
            use rayon::prelude::*;
            out_slice.par_iter_mut().enumerate().for_each(|(idx, out)| {
                *out = *hashfn.hash_pair(&in_slice[idx * 2], &in_slice[idx * 2 + 1]);
            });
        });
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        let row_size = output.size();
        let col_size = matrix.size() / row_size;
        assert_eq!(matrix.size(), col_size * row_size);

        // CPU fallback with rayon parallelism
        matrix.view(|matrix_data| {
            output.view_mut(|out_data| {
                let hashfn: &dyn crate::core::hash::HashFn<BabyBear> = self.suite.hashfn.as_ref();
                use rayon::prelude::*;
                out_data.par_iter_mut().enumerate().for_each(|(row, out)| {
                    let mut row_elems = Vec::with_capacity(col_size);
                    for col in 0..col_size {
                        row_elems.push(matrix_data[col * row_size + row]);
                    }
                    *out = *hashfn.hash_elem_slice(&row_elems);
                });
            });
        });
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

// ---- Poseidon-254 (GPU ESIMD) ----

pub struct IntelHashPoseidon254 {
    suite: HashSuite<BabyBear>,
}

impl IntelHash for IntelHashPoseidon254 {
    fn new() -> Self {
        IntelHashPoseidon254 {
            suite: Poseidon254HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        let input_ptr = io.as_device_ptr_with_offset(2 * output_size);
        let output_ptr = io.as_device_ptr_with_offset(output_size);
        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_poseidon254_hash_fold(
                queue,
                output_ptr.0 as *mut std::ffi::c_void,
                input_ptr.0 as *const std::ffi::c_void,
                output_size as u32,
            )
        });
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        let row_size = output.size();
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);
        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_poseidon254_hash_rows(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                matrix.as_device_ptr().0 as *const std::ffi::c_void,
                row_size as u32,
                col_size as u32,
            )
        });
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

// ============================================================================
// IntelHal struct
// ============================================================================

pub struct IntelHal<Hash: IntelHash + ?Sized> {
    hash: Option<Box<Hash>>,
    _lock: ReentrantMutexGuard<'static, ()>,
}

pub type IntelHalPoseidon2 = IntelHal<IntelHashPoseidon2>;
pub type IntelHalSha256 = IntelHal<IntelHashSha256>;
pub type IntelHalPoseidon254 = IntelHal<IntelHashPoseidon254>;

// ============================================================================
// RawBuffer - RAII device allocation with tracking
// ============================================================================

struct RawBuffer {
    name: &'static str,
    buf: ManuallyDrop<IntelDeviceBuffer>,
}

impl RawBuffer {
    pub fn new(name: &'static str, size: usize) -> Self {
        tracing::trace!("alloc: {size} bytes, {name}");
        tracker().lock().unwrap().alloc(size);
        let buf = IntelDeviceBuffer::uninitialized(size)
            .unwrap_or_else(|e| panic!("Intel GPU allocation failed on {name}: {size} bytes: {e}"));
        Self {
            name,
            buf: ManuallyDrop::new(buf),
        }
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        let size = self.buf.len();
        tracing::trace!("free: {size} bytes, {}", self.name);
        tracker().lock().unwrap().free(size);
        // Safety: self.buf is not accessed after take() since we're in Drop,
        // and ManuallyDrop's own drop is a no-op.
        unsafe { ManuallyDrop::drop(&mut self.buf) };
    }
}

// ============================================================================
// BufferImpl<T> - typed view over RawBuffer
// ============================================================================

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
        let bytes_len = std::mem::size_of_val(slice);
        assert!(bytes_len > 0);
        let mut buffer = RawBuffer::new(name, bytes_len);
        let bytes: &[u8] = unchecked_cast(slice);
        buffer.buf.copy_from(bytes).unwrap();
        BufferImpl {
            buffer: Rc::new(RefCell::new(buffer)),
            size: slice.len(),
            offset: 0,
            marker: PhantomData,
        }
    }

    pub fn as_device_ptr(&self) -> DevicePointer<u8> {
        let ptr = self.buffer.borrow().buf.as_device_ptr();
        let offset = self.offset * std::mem::size_of::<T>();
        unsafe { ptr.offset(offset.try_into().unwrap()) }
    }

    pub fn as_device_ptr_with_offset(&self, offset: usize) -> DevicePointer<u8> {
        let ptr = self.buffer.borrow().buf.as_device_ptr();
        let offset = (self.offset + offset) * std::mem::size_of::<T>();
        unsafe { ptr.offset(offset.try_into().unwrap()) }
    }

    /// Upload host data directly to the GPU buffer without downloading first.
    /// More efficient than view_mut() when the entire buffer is being overwritten,
    /// since view_mut() does a needless D2H transfer before the H2D upload.
    pub fn copy_from_host(&self, data: &[T]) {
        assert_eq!(data.len(), self.size, "copy_from_host: size mismatch");
        let item_size = std::mem::size_of::<T>();
        let byte_offset = self.offset * item_size;
        let byte_len = self.size * item_size;
        let bytes: &[u8] = unchecked_cast(data);
        let buf = self.buffer.borrow();
        let queue = get_queue();
        let ptr = buf.buf.as_device_ptr();
        unsafe {
            intel::esimd_memcpy_htod(
                queue,
                (ptr.0 as *mut u8).add(byte_offset) as *mut std::ffi::c_void,
                bytes.as_ptr() as *const std::ffi::c_void,
                byte_len,
            );
        }
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
        let buf = self.buffer.borrow();
        let offset = (self.offset + idx) * item_size;
        let host_buf = buf.buf.as_host_vec_range(offset, item_size).unwrap();
        let slice: &[T] = unchecked_cast(&host_buf[..]);
        slice[0].clone()
    }

    fn view<F: FnOnce(&[T])>(&self, f: F) {
        scope!("view");
        let item_size = std::mem::size_of::<T>();
        let buf = self.buffer.borrow();
        let offset = self.offset * item_size;
        let len = self.size * item_size;
        let host_buf = buf.buf.as_host_vec_range(offset, len).unwrap();
        let slice: &[T] = unchecked_cast(&host_buf[..]);
        f(slice);
    }

    fn view_mut<F: FnOnce(&mut [T])>(&self, f: F) {
        scope!("view_mut");
        let item_size = std::mem::size_of::<T>();
        let byte_offset = self.offset * item_size;
        let byte_len = self.size * item_size;
        let mut buf = self.buffer.borrow_mut();
        let total_bytes = buf.buf.len();

        if byte_offset == 0 && byte_len == total_bytes {
            // Fast path: BufferImpl covers the entire RawBuffer.
            // Download only our region, modify, upload only our region.
            let mut host_buf = buf.buf.as_host_vec().unwrap();
            let slice = unchecked_cast_mut(&mut host_buf);
            f(slice);
            buf.buf.copy_from(&host_buf).unwrap();
        } else {
            // Slice path: download just the slice region, modify, upload just the region.
            let mut host_buf = buf.buf.as_host_vec_range(byte_offset, byte_len).unwrap();
            let slice: &mut [T] = unchecked_cast_mut(&mut host_buf);
            f(slice);
            // Upload just the modified region back to the correct offset in device memory.
            let queue = get_queue();
            let ptr = buf.buf.as_device_ptr();
            unsafe {
                intel::esimd_memcpy_htod(
                    queue,
                    (ptr.0 as *mut u8).add(byte_offset) as *mut std::ffi::c_void,
                    host_buf.as_ptr() as *const std::ffi::c_void,
                    byte_len,
                );
            }
        }
    }

    fn to_vec(&self) -> Vec<T> {
        let item_size = std::mem::size_of::<T>();
        let buf = self.buffer.borrow();
        let offset = self.offset * item_size;
        let len = self.size * item_size;
        let host_buf = buf.buf.as_host_vec_range(offset, len).unwrap();
        let slice: &[T] = unchecked_cast(&host_buf[..]);
        slice.to_vec()
    }
}

// ============================================================================
// IntelHal construction
// ============================================================================

impl<IH: IntelHash> Default for IntelHal<IH> {
    fn default() -> Self {
        Self::new()
    }
}

impl<IH: IntelHash + ?Sized> IntelHal<IH> {
    pub fn new() -> Self
    where
        IH: Sized,
    {
        Self::new_from_hash(Box::new(IH::new()))
    }

    fn new_from_hash(hash: Box<IH>) -> Self {
        let _lock = singleton().lock();

        // Ensure the SYCL queue is initialized (creates on first call)
        let queue = get_queue();

        // Pre-warm twiddle tables + SYCL runtime (only on first call)
        static WARMUP_DONE: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !WARMUP_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
            unsafe { intel::esimd_warmup(queue, 22) };
        }

        let mut hal = Self {
            hash: None,
            _lock,
        };
        hal.hash = Some(hash);
        hal
    }
}

// ============================================================================
// Hal trait implementation for IntelHal
// ============================================================================

impl<IH: IntelHash + ?Sized> Hal for IntelHal<IH> {
    type Field = BabyBear;
    type Elem = BabyBearElem;
    type ExtElem = BabyBearExtElem;
    type Buffer<T: Clone + Debug + PartialEq> = BufferImpl<T>;

    fn has_unified_memory(&self) -> bool {
        false
    }

    fn get_hash_suite(&self) -> &HashSuite<Self::Field> {
        self.hash.as_ref().unwrap().get_hash_suite()
    }

    // ---- Allocation ----

    fn alloc_elem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::Elem> {
        BufferImpl::new(name, size)
    }

    fn alloc_extelem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem> {
        BufferImpl::new(name, size)
    }

    fn alloc_digest(&self, name: &'static str, size: usize) -> Self::Buffer<Digest> {
        BufferImpl::new(name, size)
    }

    fn alloc_u32(&self, name: &'static str, size: usize) -> Self::Buffer<u32> {
        BufferImpl::new(name, size)
    }

    /// Override the default alloc_elem_init to avoid the expensive
    /// download-fill-upload round-trip. Uses GPU-side memset/set_32 instead.
    ///
    /// The default impl does: alloc -> download (uninitialized!) -> fill on CPU -> upload.
    /// For large buffers (e.g. data=844MB at po2=20), this wastes seconds on PCIe transfers.
    /// Instead, we use GPU-side memset for common patterns:
    /// - INVALID (0xFFFFFFFF): memset(0xFF) -- all bytes are 0xFF
    /// - ZERO (0x00000000): memset(0)
    /// - Other values: set_32 (uploads a host buffer)
    fn alloc_elem_init(
        &self,
        name: &'static str,
        size: usize,
        value: Self::Elem,
    ) -> Self::Buffer<Self::Elem> {
        let buf = BufferImpl::new(name, size);
        let bytes_len = size * std::mem::size_of::<Self::Elem>();
        // Access the raw Montgomery representation directly.
        // SAFETY: Elem is #[repr(transparent)] wrapping u32 and derives NoUninit,
        // so transmuting to u32 is guaranteed safe.
        let raw_bits: u32 = unsafe { std::mem::transmute(value) };
        if raw_bits == 0xFFFFFFFF {
            // INVALID sentinel: all bytes are 0xFF
            buf.buffer.borrow_mut().buf.memset(0xFF_u8 as i32, bytes_len);
        } else if raw_bits == 0 {
            // ZERO: all bytes are 0x00
            buf.buffer.borrow_mut().buf.memset(0, bytes_len);
        } else {
            // General case: use set_32
            buf.buffer.borrow_mut().buf.set_32(raw_bits);
        }
        buf
    }

    /// Override the default alloc_extelem_zeroed to use GPU-side memset(0)
    /// instead of download-fill-upload round-trip.
    fn alloc_extelem_zeroed(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem> {
        let buf = BufferImpl::new(name, size);
        let bytes_len = size * std::mem::size_of::<Self::ExtElem>();
        buf.buffer.borrow_mut().buf.memset(0, bytes_len);
        buf
    }

    // ---- Copy-from host ----

    fn copy_from_elem(&self, name: &'static str, slice: &[Self::Elem]) -> Self::Buffer<Self::Elem> {
        BufferImpl::copy_from(name, slice)
    }

    fn copy_from_extelem(
        &self,
        name: &'static str,
        slice: &[Self::ExtElem],
    ) -> Self::Buffer<Self::ExtElem> {
        BufferImpl::copy_from(name, slice)
    }

    fn copy_from_digest(&self, name: &'static str, slice: &[Digest]) -> Self::Buffer<Digest> {
        BufferImpl::copy_from(name, slice)
    }

    fn copy_from_u32(&self, name: &'static str, slice: &[u32]) -> Self::Buffer<u32> {
        BufferImpl::copy_from(name, slice)
    }

    // ---- NTT operations ----

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

        let lg_out = log2_ceil(out_size);
        assert_eq!(out_size, 1 << lg_out);
        assert!(lg_out < Self::Elem::MAX_ROU_PO2);

        let queue = get_queue();

        // Step 1: GPU expand — fused zero+scatter in one pass.
        // Uses stride-scatter pattern matching the CPU's DIF expand_bits skip.
        esimd_check(unsafe {
            intel::esimd_batch_expand_ffi(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                input.as_device_ptr().0 as *const std::ffi::c_void,
                in_size as u32,
                out_size as u32,
                poly_count as u32,
                expand_bits as u32,
            )
        });

        // Step 2: Batch forward NTT — all polynomials submitted without intermediate sync.
        esimd_check(unsafe {
            intel::esimd_batch_forward_ntt(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                lg_out as u32,
                poly_count as u32,
                out_size as u32,
            )
        });
    }

    fn batch_interpolate_ntt(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let n_bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << n_bits);
        assert!(n_bits < Self::Elem::MAX_ROU_PO2);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_batch_inverse_ntt(
                queue,
                io.as_device_ptr().0 as *mut std::ffi::c_void,
                n_bits as u32,
                count as u32,
                row_size as u32,
            )
        });
    }

    fn batch_bit_reverse(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << bits);

        let queue = get_queue();
        // count parameter to FFI is total elements, not poly count
        esimd_check(unsafe {
            intel::esimd_batch_bit_reverse_ffi(
                queue,
                io.as_device_ptr().0 as *mut std::ffi::c_void,
                bits as u32,
                io.size() as u32,
            )
        });
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

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_batch_evaluate_any_ffi(
                queue,
                out.as_device_ptr().0 as *mut std::ffi::c_void,
                coeffs.as_device_ptr().0 as *const std::ffi::c_void,
                which.as_device_ptr().0 as *const std::ffi::c_void,
                xs.as_device_ptr().0 as *const std::ffi::c_void,
                eval_count as u32,
                count as u32,
            )
        });
    }

    fn zk_shift(&self, io: &Self::Buffer<Self::Elem>, poly_count: usize) {
        let bits = log2_ceil(io.size() / poly_count);
        assert_eq!(io.size(), poly_count * (1 << bits));

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_batch_zk_shift(
                queue,
                io.as_device_ptr().0 as *mut std::ffi::c_void,
                bits as u32,
                poly_count as u32,
            )
        });
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
        let mix_start_buf = self.copy_from_extelem("mix_start", &[*mix_start]);
        let mix_buf = self.copy_from_extelem("mix", &[*mix]);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_mix_poly_coeffs_ffi(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                input.as_device_ptr().0 as *const std::ffi::c_void,
                combos.as_device_ptr().0 as *const std::ffi::c_void,
                mix_start_buf.as_device_ptr().0 as *const std::ffi::c_void,
                mix_buf.as_device_ptr().0 as *const std::ffi::c_void,
                input_size as u32,
                count as u32,
            )
        });
    }

    // ---- Element-wise operations ----

    fn eltwise_add_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input1: &Self::Buffer<Self::Elem>,
        input2: &Self::Buffer<Self::Elem>,
    ) {
        assert_eq!(output.size(), input1.size());
        assert_eq!(output.size(), input2.size());
        let count = output.size();

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_eltwise_add_fp_ffi(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                input1.as_device_ptr().0 as *const std::ffi::c_void,
                input2.as_device_ptr().0 as *const std::ffi::c_void,
                count as u32,
            )
        });
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

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_eltwise_sum_fpext_ffi(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                input.as_device_ptr().0 as *const std::ffi::c_void,
                to_add as u32,
                count as u32,
            )
        });
    }

    fn eltwise_copy_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
    ) {
        let count = output.size();
        assert_eq!(count, input.size());

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_eltwise_copy_fp_ffi(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                input.as_device_ptr().0 as *const std::ffi::c_void,
                count as u32,
            )
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
        let from_buf = self.copy_from_elem("from", from);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_eltwise_copy_fp_region_ffi(
                queue,
                into.as_device_ptr().0 as *mut std::ffi::c_void,
                from_buf.as_device_ptr().0 as *const std::ffi::c_void,
                from_rows as u32,
                from_cols as u32,
                from_offset as u32,
                from_stride as u32,
                into_offset as u32,
                into_stride as u32,
            )
        });
    }

    fn eltwise_zeroize_elem(&self, elems: &Self::Buffer<Self::Elem>) {
        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_eltwise_zeroize_fp_ffi(
                queue,
                elems.as_device_ptr().0 as *mut std::ffi::c_void,
                elems.size() as u32,
            )
        });
    }

    // ---- FRI ----

    fn fri_fold(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
        mix: &Self::ExtElem,
    ) {
        let count = output.size() / Self::ExtElem::EXT_SIZE;
        assert_eq!(output.size(), count * Self::ExtElem::EXT_SIZE);
        assert_eq!(input.size(), output.size() * FRI_FOLD);
        let mix_buf = self.copy_from_extelem("mix", &[*mix]);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_fri_fold_ffi(
                queue,
                output.as_device_ptr().0 as *mut std::ffi::c_void,
                input.as_device_ptr().0 as *const std::ffi::c_void,
                mix_buf.as_device_ptr().0 as *const std::ffi::c_void,
                count as u32,
            )
        });
    }

    // ---- Hashing ----

    fn hash_rows(&self, output: &Self::Buffer<Digest>, matrix: &Self::Buffer<Self::Elem>) {
        self.hash.as_ref().unwrap().hash_rows(output, matrix);
    }

    fn hash_fold(&self, io: &Self::Buffer<Digest>, input_size: usize, output_size: usize) {
        assert_eq!(input_size, 2 * output_size);
        self.hash.as_ref().unwrap().hash_fold(io, output_size);
    }

    // ---- Gather / Scatter ----

    fn gather_sample(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        idx: usize,
        size: usize,
        stride: usize,
    ) {
        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_gather_sample_fp_ffi(
                queue,
                dst.as_device_ptr().0 as *mut std::ffi::c_void,
                src.as_device_ptr().0 as *const std::ffi::c_void,
                idx as u32,
                size as u32,
                stride as u32,
            )
        });
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

        // Upload index, offsets, values to device then call ESIMD scatter
        let index_buf = self.copy_from_u32("scatter_index", index);
        let offsets_buf = self.copy_from_u32("scatter_offsets", offsets);
        let values_buf = self.copy_from_elem("scatter_values", values);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_scatter_fp_ffi(
                queue,
                into.as_device_ptr().0 as *mut std::ffi::c_void,
                index_buf.as_device_ptr().0 as *const std::ffi::c_void,
                offsets_buf.as_device_ptr().0 as *const std::ffi::c_void,
                values_buf.as_device_ptr().0 as *const std::ffi::c_void,
                count as u32,
            )
        });
    }

    // ---- prefix_products: CPU fallback ----

    fn prefix_products(&self, io: &Self::Buffer<Self::ExtElem>) {
        io.view_mut(|io| {
            for i in 1..io.len() {
                io[i] *= io[i - 1];
            }
        });
    }

    // combos_divide uses default CPU fallback. GPU implementation is sequential
    // (4M iterations per division) and even when parallelized across combos via
    // parallel_for, scalar GPU execution is much slower than vectorized CPU.
    // Tested 7135ms GPU vs 234ms CPU (30x slower).

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
        scope!("combos_prepare");
        // Upload small parameter buffers to GPU
        let coeff_u_buf = self.copy_from_extelem("coeff_u", coeff_u);
        let reg_sizes_buf = self.copy_from_u32("reg_sizes", reg_sizes);
        let reg_combo_ids_buf = self.copy_from_u32("reg_combo_ids", reg_combo_ids);
        let mix_buf = self.copy_from_extelem("mix", &[*mix]);

        let queue = get_queue();
        esimd_check(unsafe {
            intel::esimd_combos_prepare_ffi(
                queue,
                combos.as_device_ptr().0 as *mut std::ffi::c_void,
                coeff_u_buf.as_device_ptr().0 as *const std::ffi::c_void,
                combo_count as u32,
                cycles as u32,
                reg_sizes.len() as u32,
                reg_sizes_buf.as_device_ptr().0 as *const std::ffi::c_void,
                reg_combo_ids_buf.as_device_ptr().0 as *const std::ffi::c_void,
                Self::CHECK_SIZE as u32,
                mix_buf.as_device_ptr().0 as *const std::ffi::c_void,
            )
        });
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use test_log::test;

    use super::{IntelHalPoseidon2, IntelHalSha256};
    use crate::hal::testutil;

    #[test]
    #[should_panic]
    fn check_req() {
        testutil::check_req(IntelHalSha256::new());
    }

    #[test]
    fn eltwise_add_elem() {
        testutil::eltwise_add_elem(IntelHalSha256::new());
    }

    #[test]
    fn eltwise_copy_elem() {
        testutil::eltwise_copy_elem(IntelHalSha256::new());
    }

    #[test]
    fn eltwise_sum_extelem() {
        testutil::eltwise_sum_extelem(IntelHalSha256::new());
    }

    #[test]
    fn hash_rows_sha256() {
        testutil::hash_rows(IntelHalSha256::new());
    }

    #[test]
    fn hash_fold_sha256() {
        testutil::hash_fold(IntelHalSha256::new());
    }

    #[test]
    fn hash_rows_poseidon2() {
        testutil::hash_rows(IntelHalPoseidon2::new());
    }

    #[test]
    fn hash_fold_poseidon2() {
        testutil::hash_fold(IntelHalPoseidon2::new());
    }

    #[test]
    fn fri_fold() {
        testutil::fri_fold(IntelHalSha256::new());
    }

    #[test]
    fn batch_expand_into_evaluate_ntt() {
        testutil::batch_expand_into_evaluate_ntt(IntelHalSha256::new());
    }

    #[test]
    fn batch_interpolate_ntt() {
        testutil::batch_interpolate_ntt(IntelHalSha256::new());
    }

    #[test]
    fn batch_bit_reverse() {
        testutil::batch_bit_reverse(IntelHalSha256::new());
    }

    #[test]
    fn batch_evaluate_any() {
        testutil::batch_evaluate_any(IntelHalSha256::new());
    }

    #[test]
    fn gather_sample() {
        testutil::gather_sample(IntelHalSha256::new());
    }

    #[test]
    fn zk_shift() {
        testutil::zk_shift(IntelHalSha256::new());
    }

    #[test]
    fn mix_poly_coeffs() {
        testutil::mix_poly_coeffs(IntelHalSha256::new());
    }
}
