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

//! `BufferImpl`: pooled `StorageModeShared` buffers with lazy-sync host access (plan 2.6).

use std::{
    fmt,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    slice,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use metal::{Buffer as MtlBuffer, BufferRef};

use super::{runtime::Runtime, KernelArg};
use crate::hal::{tracker, Buffer};

/// One device allocation. On drop it goes back to its HAL's pool (deferred until the batches
/// that may use it complete) unless `set_bypass_pool` was called.
pub(crate) struct RawBuffer {
    name: &'static str,
    buf: ManuallyDrop<MtlBuffer>,
    bytes: u64,
    bypass_pool: AtomicBool,
    rt: Arc<Runtime>,
}

impl RawBuffer {
    /// Z14: wrap a buffer of `bytes` this runtime's pool handed out (`Runtime::prefill_take`).
    pub(crate) fn from_pooled(
        rt: &Arc<Runtime>,
        name: &'static str,
        buf: MtlBuffer,
        bytes: u64,
    ) -> Self {
        tracker().lock().unwrap().alloc(bytes as usize);
        Self {
            name,
            buf: ManuallyDrop::new(buf),
            bytes,
            bypass_pool: AtomicBool::new(false),
            rt: rt.clone(),
        }
    }

    pub(crate) fn new(rt: &Arc<Runtime>, name: &'static str, bytes: usize) -> Self {
        // Metal returns nil for a zero-length buffer; keep every allocation addressable.
        let bytes = (bytes as u64).max(16);
        let buf = rt
            .alloc_shared(name, bytes)
            .unwrap_or_else(|e| panic!("{e:#}"));
        tracker().lock().unwrap().alloc(bytes as usize);
        Self {
            name,
            buf: ManuallyDrop::new(buf),
            bytes,
            bypass_pool: AtomicBool::new(false),
            rt: rt.clone(),
        }
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        tracker().lock().unwrap().free(self.bytes as usize);
        // SAFETY: `buf` is not used again after this point.
        let buf = unsafe { ManuallyDrop::take(&mut self.buf) };
        if self.bypass_pool.load(Ordering::Relaxed) {
            drop(buf);
            self.rt.freed(self.bytes);
        } else {
            self.rt.release(self.bytes, buf);
        }
    }
}

/// A typed view (`offset`, `size` in elements of `T`) of a shared Metal buffer.
pub struct BufferImpl<T> {
    raw: Arc<RawBuffer>,
    offset: usize,
    size: usize,
    marker: PhantomData<T>,
}

impl<T> Clone for BufferImpl<T> {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            offset: self.offset,
            size: self.size,
            marker: PhantomData,
        }
    }
}

impl<T> fmt::Debug for BufferImpl<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferImpl")
            .field("name", &self.raw.name)
            .field("offset", &self.offset)
            .field("size", &self.size)
            .field("bytes", &self.raw.bytes)
            .finish()
    }
}

impl<T> BufferImpl<T> {
    pub(crate) fn new(rt: &Arc<Runtime>, name: &'static str, size: usize) -> Self {
        Self {
            raw: Arc::new(RawBuffer::new(rt, name, size * mem::size_of::<T>())),
            offset: 0,
            size,
            marker: PhantomData,
        }
    }

    /// Z14: a buffer of `size` elements over `raw` (`Runtime::prefill_take`'s buffer).
    pub(crate) fn from_raw(raw: RawBuffer, size: usize) -> Self {
        assert!((size * mem::size_of::<T>()) as u64 <= raw.bytes);
        Self {
            raw: Arc::new(raw),
            offset: 0,
            size,
            marker: PhantomData,
        }
    }

    /// A new buffer holding a copy of `slice`. The memory is either new or came back from the pool
    /// after every batch that used it completed, so the copy needs no sync.
    pub(crate) fn copy_from(rt: &Arc<Runtime>, name: &'static str, slice: &[T]) -> Self {
        let this = Self::new(rt, name, slice.len());
        // SAFETY: the allocation holds at least `size_of_val(slice)` bytes and no GPU work uses it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                slice.as_ptr() as *const u8,
                this.raw.buf.contents() as *mut u8,
                mem::size_of_val(slice),
            );
        }
        this
    }

    fn byte_offset(&self) -> u64 {
        (self.offset * mem::size_of::<T>()) as u64
    }

    /// The underlying Metal buffer (whole allocation; see [Self::as_arg] for the view's offset).
    pub fn as_buf(&self) -> &BufferRef {
        &self.raw.buf
    }

    /// Element offset of this view into the allocation.
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn as_arg(&self) -> KernelArg<'_> {
        KernelArg::Buffer {
            buffer: &self.raw.buf,
            offset: self.byte_offset(),
        }
    }

    /// A kernel argument `off` elements past the start of this view (`off <= size`; binding at
    /// the end is allowed, past it panics: the kernel would get an out-of-bounds pointer).
    pub fn as_arg_with_offset(&self, off: usize) -> KernelArg<'_> {
        assert!(
            off <= self.size,
            "as_arg_with_offset({off}) past the end of '{}' ({} elements)",
            self.raw.name,
            self.size
        );
        KernelArg::Buffer {
            buffer: &self.raw.buf,
            offset: ((self.offset + off) * mem::size_of::<T>()) as u64,
        }
    }

    /// Host pointer to the first element of this view. Not synchronised: valid to read GPU
    /// results only after `sync()`. With `RISC0_METAL_DEBUG=1` it panics if a batch that has
    /// not been observed complete uses this buffer.
    pub fn as_ptr(&self) -> *const T {
        self.raw
            .rt
            .check_host_access(&self.raw.buf, self.raw.name, "pointer");
        // SAFETY: the offset lies within the allocation.
        unsafe {
            (self.raw.buf.contents() as *const u8).add(self.byte_offset() as usize) as *const T
        }
    }

    /// Mutable host pointer to the first element of this view; same rules as [Self::as_ptr].
    pub fn as_mut_ptr(&self) -> *mut T {
        self.as_ptr() as *mut T
    }

    /// GPU virtual address of the first element of this view.
    pub fn as_device_ptr(&self) -> u64 {
        self.raw.buf.gpu_address() + self.byte_offset()
    }

    /// GPU virtual address `off` elements past the start of this view (`off <= size`).
    pub fn gpu_address_at(&self, off: usize) -> u64 {
        assert!(
            off <= self.size,
            "gpu_address_at({off}) past the end of '{}' ({} elements)",
            self.raw.name,
            self.size
        );
        self.raw.buf.gpu_address() + ((self.offset + off) * mem::size_of::<T>()) as u64
    }

    fn host_slice(&self, what: &str) -> *mut T {
        self.raw
            .rt
            .sync_for_host(&self.raw.buf, self.raw.name, what);
        // SAFETY: as in `as_ptr`.
        unsafe { (self.raw.buf.contents() as *mut u8).add(self.byte_offset() as usize) as *mut T }
    }
}

impl<T: Clone> Buffer<T> for BufferImpl<T> {
    fn name(&self) -> &'static str {
        self.raw.name
    }

    fn size(&self) -> usize {
        self.size
    }

    fn slice(&self, offset: usize, size: usize) -> BufferImpl<T> {
        assert!(offset + size <= self.size);
        BufferImpl {
            raw: self.raw.clone(),
            offset: self.offset + offset,
            size,
            marker: PhantomData,
        }
    }

    fn get_at(&self, idx: usize) -> T {
        assert!(
            idx < self.size,
            "get_at({idx}) on a buffer of {}",
            self.size
        );
        let ptr = self.host_slice("read");
        // SAFETY: idx < size, synced.
        unsafe { (*ptr.add(idx)).clone() }
    }

    fn view<F: FnOnce(&[T])>(&self, f: F) {
        let ptr = self.host_slice("view");
        // SAFETY: synced; the view lies within the allocation.
        f(unsafe { slice::from_raw_parts(ptr, self.size) });
    }

    fn view_mut<F: FnOnce(&mut [T])>(&self, f: F) {
        let ptr = self.host_slice("view_mut");
        // SAFETY: synced; the view lies within the allocation. Shared storage needs no
        // didModifyRange.
        f(unsafe { slice::from_raw_parts_mut(ptr, self.size) });
    }

    fn to_vec(&self) -> Vec<T> {
        let ptr = self.host_slice("to_vec");
        // SAFETY: synced; the view lies within the allocation.
        unsafe { slice::from_raw_parts(ptr, self.size) }.to_vec()
    }

    fn set_bypass_pool(&self) {
        self.raw.bypass_pool.store(true, Ordering::Relaxed);
    }
}
