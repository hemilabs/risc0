// C2-R2: page-locked staging of the recursion witgen inputs, filled on the prepare thread.
//
// The GPU-thread uploads of the preflight trace (wom, cycles, iops) and the host-chain rows are
// pageable cudaMemcpys: ~3.5 ms per join after the last segment and ~13 ms per join between
// segments, where the driver's bounce-buffer copies compete with the next segment's host work.
// From page-locked memory they run at full PCIe rate. Buffers come from a small process-wide pool
// and go back to it when the preflight that owns them is dropped (after its witgen call, which
// synchronizes, so no copy from them can still be in flight).

use std::sync::Mutex;

use risc0_circuit_recursion_sys::{
    risc0_circuit_recursion_cuda_host_alloc, risc0_circuit_recursion_cuda_host_free,
    RawPreflightCycle,
};
use risc0_zkp::field::baby_bear::BabyBearExtElem;

pub(crate) struct Pinned {
    ptr: usize,
    cap: usize,
}

static POOL: Mutex<Vec<Pinned>> = Mutex::new(Vec::new());
const POOL_CAP: usize = 6;

impl Pinned {
    fn take(bytes: usize) -> Option<Pinned> {
        {
            let mut pool = POOL.lock().unwrap();
            if let Some(i) = pool
                .iter()
                .enumerate()
                .filter(|(_, b)| b.cap >= bytes)
                .min_by_key(|(_, b)| b.cap)
                .map(|(i, _)| i)
            {
                return Some(pool.swap_remove(i));
            }
        }
        // Headroom so the next job of a similar size reuses it.
        let cap = (bytes + bytes / 4 + (1 << 20)) & !((1 << 20) - 1);
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        risc0_sys::ffi_wrap(|| unsafe { risc0_circuit_recursion_cuda_host_alloc(cap, &mut p) }).ok()?;
        Some(Pinned { ptr: p as usize, cap })
    }

    fn base(&self) -> *mut u8 {
        self.ptr as *mut u8
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        let mut pool = POOL.lock().unwrap();
        if pool.len() < POOL_CAP {
            pool.push(Pinned { ptr: self.ptr, cap: self.cap });
        } else {
            let p = self.ptr as *mut std::ffi::c_void;
            let _ = risc0_sys::ffi_wrap(|| unsafe { risc0_circuit_recursion_cuda_host_free(p) });
        }
        self.ptr = 0;
    }
}

/// Byte offsets of each array in the buffer.
pub(crate) struct PinnedTrace {
    buf: Pinned,
    off: [usize; 8],
}

fn copy_par(dst: *mut u8, src: &[u8]) {
    use rayon::prelude::*;
    const CH: usize = 1 << 20;
    let dst = dst as usize;
    src.par_chunks(CH).enumerate().for_each(|(i, c)| unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), (dst + i * CH) as *mut u8, c.len());
    });
}

impl PinnedTrace {
    /// wom, cycles, iops, then (if any) chain heads, cycles, rows, wom_idx, wom_rows.
    pub(crate) fn new(
        wom: &[BabyBearExtElem],
        cycles: &[RawPreflightCycle],
        iops: &[BabyBearExtElem],
        chains: Option<&super::witgen::HostChains>,
    ) -> Option<Self> {
        let parts: [&[u8]; 8] = {
            let e: &[u32] = &[];
            let (h, c, r, wi, wr) = match chains {
                Some(hc) => (&hc.heads[..], &hc.cycles[..], &hc.rows[..], &hc.wom_idx[..], &hc.wom_rows[..]),
                None => (e, e, e, e, e),
            };
            unsafe {
                [
                    std::slice::from_raw_parts(wom.as_ptr() as *const u8, std::mem::size_of_val(wom)),
                    std::slice::from_raw_parts(cycles.as_ptr() as *const u8, std::mem::size_of_val(cycles)),
                    std::slice::from_raw_parts(iops.as_ptr() as *const u8, std::mem::size_of_val(iops)),
                    bytemuck::cast_slice(h),
                    bytemuck::cast_slice(c),
                    bytemuck::cast_slice(r),
                    bytemuck::cast_slice(wi),
                    bytemuck::cast_slice(wr),
                ]
            }
        };
        let mut off = [0usize; 8];
        let mut total = 0usize;
        for (i, p) in parts.iter().enumerate() {
            off[i] = total;
            total += (p.len() + 63) & !63;
        }
        let buf = Pinned::take(total.max(64))?;
        for (i, p) in parts.iter().enumerate() {
            if !p.is_empty() {
                copy_par(unsafe { buf.base().add(off[i]) }, p);
            }
        }
        Some(PinnedTrace { buf, off })
    }

    pub(crate) fn ptr<T>(&self, i: usize) -> *const T {
        unsafe { self.buf.base().add(self.off[i]) as *const T }
    }
}
