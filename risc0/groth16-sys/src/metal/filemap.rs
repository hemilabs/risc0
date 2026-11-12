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

//! Read-only file mappings wrapped as Metal buffers without a copy (item G10), and the process
//! memory counters the G10 measurements report.
//!
//! `newBufferWithBytesNoCopy:length:options:deallocator:` takes a page-aligned pointer and a
//! length that is a multiple of the page size (16 KiB on Apple silicon). The zkey's point sections
//! start at arbitrary 4-byte-aligned offsets, so [FileMap::map] maps the whole pages that hold a
//! byte range, and the buffer is bound with the section's offset inside the first page
//! ([FileMap::inner_offset]). The pages are the file's page cache (clean, file backed), not
//! anonymous memory: they do not count toward the process' physical footprint, the kernel can
//! drop them under memory pressure and read them back, and creating the buffer copies nothing.
//! The mapping is `PROT_READ`/`MAP_PRIVATE`: nothing (host or GPU) writes it; the kernels bind
//! the points as `const device`.

use std::{
    ffi::c_void,
    fs::File,
    os::{fd::AsRawFd as _, raw::c_int},
    path::Path,
};

use anyhow::{anyhow, ensure, Context as _, Result};
use metal::{
    foreign_types::{ForeignType as _, ForeignTypeRef as _},
    objc::{msg_send, runtime::Object, sel, sel_impl},
    Buffer, BufferRef, DeviceRef, MTLBuffer, MTLResourceOptions,
};

extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        off: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
    fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    fn getpagesize() -> c_int;
    fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *mut c_void) -> c_int;
}

const PROT_READ: c_int = 1;
const MAP_PRIVATE: c_int = 2;
const MADV_WILLNEED: c_int = 3;

/// The VM page size (16 KiB on Apple silicon).
pub fn page_size() -> u64 {
    // SAFETY: no arguments, no side effects.
    (unsafe { getpagesize() }) as u64
}

/// A read-only `MAP_PRIVATE` mapping of whole pages of a file, wrapped in a no-copy
/// `StorageModeShared` Metal buffer.
pub struct FileMap {
    /// Released before the mapping is unmapped ([Drop]). Metal itself keeps a buffer alive while a command buffer that
    /// uses it is in flight, so owners must not drop a [FileMap] before their GPU work completed
    /// (the prover waits for every command buffer of a proof).
    buffer: std::mem::ManuallyDrop<Buffer>,
    ptr: *mut c_void,
    len: usize,
    inner: u64,
    payload: u64,
}

// SAFETY: the mapping is read-only and owned by this value; the Metal buffer is thread safe.
unsafe impl Send for FileMap {}
unsafe impl Sync for FileMap {}

impl FileMap {
    /// Map the bytes `[offset, offset + len)` of `file` (whole pages around them) and wrap them in
    /// a no-copy Metal buffer. `advise_willneed`: start reading the pages in the background
    /// (`madvise(MADV_WILLNEED)`), so the first GPU use does not fault them in one by one.
    pub fn map(
        device: &DeviceRef,
        file: &File,
        path: &Path,
        offset: u64,
        len: u64,
        what: &str,
        advise_willneed: bool,
    ) -> Result<Self> {
        let page = page_size();
        let start = offset / page * page;
        let end = (offset + len.max(1)).div_ceil(page) * page;
        let map_len = (end - start) as usize;
        let max = device.max_buffer_length();
        ensure!(
            map_len as u64 <= max,
            "{what}: {map_len} bytes exceed this GPU's max_buffer_length {max}"
        );
        // A mapping past the end of the file is valid up to the last page that holds file
        // bytes; the file must hold the whole range.
        let file_len = file
            .metadata()
            .with_context(|| format!("stat(\"{}\")", path.display()))?
            .len();
        ensure!(
            offset + len <= file_len,
            "{what}: bytes {offset}..{} past the end of \"{}\" ({file_len} bytes)",
            offset + len,
            path.display()
        );
        // SAFETY: a fresh read-only private mapping of an open file; checked for MAP_FAILED.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                map_len,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                start as i64,
            )
        };
        if ptr as isize == -1 || ptr.is_null() {
            return Err(anyhow!(
                "mmap of {map_len} bytes of \"{}\" for {what} failed: {}",
                path.display(),
                std::io::Error::last_os_error()
            ));
        }
        if advise_willneed {
            // SAFETY: the range is the mapping just created. Advisory only; errors ignored.
            unsafe { madvise(ptr, map_len, MADV_WILLNEED) };
        }
        let Some(buffer) = no_copy_buffer(device, ptr, map_len as u64) else {
            // SAFETY: the mapping created above, not referenced by anything else.
            unsafe { munmap(ptr, map_len) };
            return Err(anyhow!(
                "Metal allocation failed on {what}: newBufferWithBytesNoCopy of {map_len} mapped \
                 bytes of \"{}\" returned nil",
                path.display()
            ));
        };
        Ok(Self {
            buffer: std::mem::ManuallyDrop::new(buffer),
            ptr,
            len: map_len,
            inner: offset - start,
            payload: len,
        })
    }

    /// The buffer (bind it with [Self::inner_offset]).
    pub fn buffer(&self) -> &BufferRef {
        &self.buffer
    }

    /// Byte offset of the mapped range's first byte inside [Self::buffer].
    pub fn inner_offset(&self) -> u64 {
        self.inner
    }

    /// The mapped range's bytes.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping holds [inner, inner + payload) (checked against the file length at
        // map time), is read only and lives as long as `self`.
        unsafe {
            std::slice::from_raw_parts(
                (self.ptr as *const u8).add(self.inner as usize),
                self.payload as usize,
            )
        }
    }

    /// Bytes of the mapping (whole pages).
    pub fn mapped_len(&self) -> u64 {
        self.len as u64
    }
}

impl Drop for FileMap {
    fn drop(&mut self) {
        // SAFETY: the buffer is released first and never used again; then the mapping it wrapped
        // (owned by this value) is unmapped.
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.buffer);
            munmap(self.ptr, self.len);
        }
    }
}

/// A read-only `MAP_PRIVATE` mapping of a whole file for the host only (no Metal buffer).
pub struct HostMap {
    ptr: *mut c_void,
    len: usize,
}

// SAFETY: read-only mapping owned by this value.
unsafe impl Send for HostMap {}
unsafe impl Sync for HostMap {}

impl HostMap {
    /// Map all of `file` (`len` > 0 bytes).
    pub fn map(file: &File, path: &Path) -> Result<Self> {
        let len = file
            .metadata()
            .with_context(|| format!("stat(\"{}\")", path.display()))?
            .len() as usize;
        ensure!(len > 0, "\"{}\" is empty", path.display());
        // SAFETY: a fresh read-only private mapping of an open file; checked for MAP_FAILED.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr as isize == -1 || ptr.is_null() {
            return Err(anyhow!(
                "mmap of \"{}\" failed: {}",
                path.display(),
                std::io::Error::last_os_error()
            ));
        }
        Ok(Self { ptr, len })
    }

    /// The file's bytes (page-aligned start).
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the live read-only mapping of `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

impl Drop for HostMap {
    fn drop(&mut self) {
        // SAFETY: the mapping owned by this value.
        unsafe { munmap(self.ptr, self.len) };
    }
}

/// `newBufferWithBytesNoCopy:length:options:deallocator:` (no deallocator: [FileMap] unmaps) with a
/// nil check.
// objc's `msg_send!` expands a `cfg(feature = "cargo-clippy")` this crate does not declare.
#[allow(unexpected_cfgs)]
fn no_copy_buffer(device: &DeviceRef, ptr: *mut c_void, len: u64) -> Option<Buffer> {
    let options = MTLResourceOptions::StorageModeShared;
    let dealloc: *const c_void = std::ptr::null();
    // SAFETY: page-aligned pointer and page-multiple length of a live mapping; the result is a
    // +1 reference or nil, only a non-nil result is wrapped.
    let raw: *mut MTLBuffer = unsafe {
        msg_send![device.as_ptr() as *mut Object, newBufferWithBytesNoCopy: ptr
            length: len options: options deallocator: dealloc]
    };
    if raw.is_null() {
        None
    } else {
        Some(unsafe { Buffer::from_ptr(raw) })
    }
}

/// This process' memory counters (`proc_pid_rusage`, `RUSAGE_INFO_V0`), bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessMemory {
    /// Physical footprint (what Activity Monitor's "Memory" and jetsam count: anonymous and
    /// dirty memory, plus GPU allocations charged to the process; not clean file-backed pages).
    pub phys_footprint: u64,
    pub resident: u64,
    pub wired: u64,
}

/// [ProcessMemory] of this process (zeros if the call fails).
pub fn process_memory() -> ProcessMemory {
    #[repr(C)]
    struct RusageInfoV0 {
        uuid: [u8; 16],
        user_time: u64,
        system_time: u64,
        pkg_idle_wkups: u64,
        interrupt_wkups: u64,
        pageins: u64,
        wired_size: u64,
        resident_size: u64,
        phys_footprint: u64,
        proc_start_abstime: u64,
        proc_exit_abstime: u64,
    }
    let mut info = std::mem::MaybeUninit::<RusageInfoV0>::zeroed();
    // SAFETY: RUSAGE_INFO_V0 (0) fills a rusage_info_v0, which this struct mirrors.
    let rc = unsafe { proc_pid_rusage(std::process::id() as c_int, 0, info.as_mut_ptr().cast()) };
    if rc != 0 {
        return ProcessMemory::default();
    }
    // SAFETY: filled by the successful call (and zeroed before).
    let info = unsafe { info.assume_init() };
    ProcessMemory {
        phys_footprint: info.phys_footprint,
        resident: info.resident_size,
        wired: info.wired_size,
    }
}

#[cfg(test)]
#[path = "g10_tests.rs"]
mod g10_tests;
