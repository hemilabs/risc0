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

//! rv32im accumulation on the Metal HAL (docs/metal-port/PLAN.md 4.2, item R9).
//!
//! Mirrors `risc0_circuit_rv32im_cuda_accum` (rv32im-sys `kernels/cuda/ffi.cu`):
//!
//! 1. **perm** (CPU): the cycles grouped by `major` within 2^win-row windows, the order CUDA's
//!    `majorKeys` + stable `cub::DeviceRadixSort` produces (keys `(i >> win) << 8 | major`), built
//!    here with one counting sort per window. Same output with or without it: each cycle runs once
//!    and writes only its own row. `RISC0_METAL_ACCUM_SORT=0` runs without a perm (the kernel's
//!    `hasPerm = 0` path); `4..=24` picks the window (default [DEFAULT_SORT_WIN]).
//! 2. **GPU**: zero the fault word (blit fill), `stepAccumA` (accum_tu.metal, R4) over
//!    `lastCycle` threads.
//! 3. **GPU scans** (R13): inclusive Fp prefix sums of the last 4 accum columns over `lastCycle`
//!    rows (CUDA: 4 `cub::DeviceScan::InclusiveScan`; CPU: `std::inclusive_scan`) with misc.metal's
//!    `fpScanReduce`, `fpScanBlockSums`, `fpScanApply` (reduce-then-scan, the 4 columns in each
//!    dispatch). Fp addition of canonical words is exact and associative, so the sums equal both,
//!    bit for bit; a non-canonical input sets fault bit 0x100 instead.
//! 4. **GPU**: `finalizeZeroizeAccum` (misc.metal, R1) over all rows: the machine columns get
//!    the previous row's running sums, every INVALID cell becomes 0. So
//!    [CircuitAccumulator::step_accum_zeroizes] is true, as on CUDA.
//!
//! Steps 2-4 are encoded back to back and waited for once, at the end of the phase; the fault
//! word is read after that wait. `RISC0_METAL_ACCUM_SCAN=cpu` keeps R9's sequence instead: wait
//! after step 2 (and check the fault word), scan on the CPU ([fp_inclusive_scan]), then encode
//! step 4 and wait.
//!
//! `RISC0_METAL_RV32IM_CPU_PHASES` (comma list of `witgen,accum,eval`) naming `accum` keeps the
//! R7 path instead: the CPU accum kernel over host views of the Metal buffers (the prover then
//! zeroizes).
//!
//! `CircuitHal::accumulate` stays the empty no-op it is in `hal/cuda.rs` (PLAN 8.4); it lives in
//! `eval_check.rs` with the rest of `CircuitHal`.

use std::sync::OnceLock;

use anyhow::{bail, ensure, Result};
use metal::MTLSize;
use rayon::prelude::*;
use risc0_circuit_rv32im_sys::{
    risc0_circuit_rv32im_cpu_accum, RawAccumBuffers, RawBuffer, RawPreflightCycle,
    RawPreflightTrace,
};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::hal::{
    metal::{as_bytes, BufferImpl, Grid, KernelArg, MetalHal, MetalHash},
    Buffer as _,
};

use super::{
    count, cpu_phase, grow, host_slice, phase_mark, MetalCircuitHal, Phase, CPU_PHASES_ENV,
};
use crate::{
    prove::hal::{CircuitAccumulator, MetaBuffer, PreflightTrace},
    zirgen::circuit::LAYOUT_TOP_ACCUM,
};

/// Accum sort window (log2 rows); `0`/`off` disables the perm.
pub(crate) const ACCUM_SORT_ENV: &str = "RISC0_METAL_ACCUM_SORT";

/// Default sort window: CUDA's `s_win = 14` (ffi.cu, `risc0_circuit_rv32im_cuda_accum`). The
/// window sweep 11..14 is recorded in docs/metal-port/notes/R9.md.
pub(crate) const DEFAULT_SORT_WIN: u32 = 14;

/// CUDA `kUserAccumSplit` = `kLayout_TopAccum.columns[0].col` (23 today): the first machine
/// column of the accum group; stepAccumA's `zeroBack` and finalizeZeroizeAccum's split.
pub(crate) fn user_accum_split() -> u32 {
    LAYOUT_TOP_ACCUM.columns[0].offset as u32
}

/// BabyBear modulus.
const P: u32 = 15 * (1 << 27) + 1;

/// Fp addition on raw (Montgomery-form) words, exactly as `fp.h` `Fp::add` (also on INVALID).
#[inline(always)]
fn fp_add(a: u32, b: u32) -> u32 {
    let r = a.wrapping_add(b);
    if r >= P {
        r - P
    } else {
        r
    }
}

/// In-place inclusive prefix sum over Fp words.
pub(crate) fn fp_inclusive_scan(col: &mut [u32]) {
    let mut acc = match col.first() {
        Some(&v) => v,
        None => return,
    };
    for v in col.iter_mut().skip(1) {
        acc = fp_add(acc, *v);
        *v = acc;
    }
}

/// Where step 3 (the 4 prefix scans) runs: `gpu` (default, R13) or `cpu` (R9's host scans with
/// a wait before them).
pub(crate) const ACCUM_SCAN_ENV: &str = "RISC0_METAL_ACCUM_SCAN";

/// Step 3's implementation (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScanMode {
    /// misc.metal's scan kernels, in the same batch as stepAccumA and finalizeZeroizeAccum.
    Gpu,
    /// R9: wait for stepAccumA, scan on the host, then finalize.
    Cpu,
}

/// Parse an `RISC0_METAL_ACCUM_SCAN` value (unset or unknown: [ScanMode::Gpu]).
pub(crate) fn parse_scan_mode(v: Option<&str>) -> ScanMode {
    match v.map(str::trim) {
        None | Some("") => ScanMode::Gpu,
        Some(s) if s.eq_ignore_ascii_case("gpu") => ScanMode::Gpu,
        Some(s) if s.eq_ignore_ascii_case("cpu") => ScanMode::Cpu,
        Some(s) => {
            tracing::error!("{ACCUM_SCAN_ENV}={s:?}: expected `gpu` or `cpu`; using gpu");
            ScanMode::Gpu
        }
    }
}

fn scan_mode() -> ScanMode {
    static MODE: OnceLock<ScanMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = parse_scan_mode(std::env::var(ACCUM_SCAN_ENV).ok().as_deref());
        if mode == ScanMode::Cpu {
            tracing::info!("rv32im Metal accum: CPU prefix scans ({ACCUM_SCAN_ENV})");
        }
        mode
    })
}

/// `RISC0_METAL_RV32IM_CPU_PHASES` names `accum` (read once per process).
fn accum_on_cpu() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = cpu_phase("accum");
        if on {
            tracing::info!("rv32im Metal accum: CPU kernels ({CPU_PHASES_ENV})");
        }
        on
    })
}

/// Parse an `RISC0_METAL_ACCUM_SORT` value: `None` = no perm.
pub(crate) fn parse_sort_win(v: Option<&str>) -> Option<u32> {
    match v.map(str::trim) {
        None | Some("") => Some(DEFAULT_SORT_WIN),
        Some(s) if s.eq_ignore_ascii_case("off") => None,
        Some(s) => match s.parse::<u32>() {
            Ok(0) => None,
            Ok(w) => Some(w.clamp(4, 24)),
            Err(_) => {
                tracing::error!("{ACCUM_SORT_ENV}={s:?} is not a number; using {DEFAULT_SORT_WIN}");
                Some(DEFAULT_SORT_WIN)
            }
        },
    }
}

fn sort_win() -> Option<u32> {
    static WIN: OnceLock<Option<u32>> = OnceLock::new();
    *WIN.get_or_init(|| parse_sort_win(std::env::var(ACCUM_SORT_ENV).ok().as_deref()))
}

/// The cycles `[0, cycles.len())` grouped by `major` within 2^win-row windows, rows ascending
/// within a major: the permutation of CUDA's stable radix sort over `majorKeys`' keys
/// `(i >> win) << 8 | major`. One counting sort per window (windows in parallel).
pub(crate) fn build_perm(cycles: &[RawPreflightCycle], win: u32, perm: &mut [u32]) {
    let n = cycles.len();
    assert!(perm.len() >= n);
    let w = 1usize << win.min(31);
    perm[..n]
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(wi, out)| {
            let base = wi * w;
            let rows = &cycles[base..base + out.len()];
            let mut next = [0u32; 256];
            for c in rows {
                next[c.major as usize] += 1;
            }
            let mut sum = 0;
            for slot in next.iter_mut() {
                let c = *slot;
                *slot = sum;
                sum += c;
            }
            for (i, c) in rows.iter().enumerate() {
                let slot = &mut next[c.major as usize];
                out[*slot as usize] = (base + i) as u32;
                *slot += 1;
            }
        });
}

/// stepAccumA's `AccumParams` (accum_tu.metal, 48 bytes, static_assert'ed there).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct AccumParams {
    count: u32,
    has_perm: u32,
    data_rows: u32,
    data_cols: u32,
    accum_rows: u32,
    accum_cols: u32,
    global_rows: u32,
    global_cols: u32,
    mix_rows: u32,
    mix_cols: u32,
    _pad: [u32; 2],
}
const _: () = assert!(std::mem::size_of::<AccumParams>() == 48);
// SAFETY: #[repr(C)], only u32 fields, no padding.
unsafe impl bytemuck::Zeroable for AccumParams {}
unsafe impl bytemuck::Pod for AccumParams {}

/// finalizeZeroizeAccum's `FinalizeAccumParams` (misc.metal, 16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct FinalizeAccumParams {
    rows: u32,
    cols: u32,
    last_cycle: u32,
    user_accum_split: u32,
}
const _: () = assert!(std::mem::size_of::<FinalizeAccumParams>() == 16);
// SAFETY: as above.
unsafe impl bytemuck::Zeroable for FinalizeAccumParams {}
unsafe impl bytemuck::Pod for FinalizeAccumParams {}

/// The scan kernels' `FpScanParams` (misc.metal, 16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct FpScanParams {
    n: u32,
    rows: u32,
    col0: u32,
    blocks: u32,
}
const _: () = assert!(std::mem::size_of::<FpScanParams>() == 16);
// SAFETY: as above.
unsafe impl bytemuck::Zeroable for FpScanParams {}
unsafe impl bytemuck::Pod for FpScanParams {}

/// misc.metal `FP_SCAN_PER_THREAD`: rows per thread of a scan block.
const FP_SCAN_PER_THREAD: u32 = 4;

/// Default scan threadgroup size (capped by the PSOs, rounded to the SIMD width).
const FP_SCAN_TG: u64 = 256;

/// Simdgroups per scan threadgroup the host allows (threadgroup memory index 1 holds one word
/// each; the kernels scan them within one simdgroup, so also at most the SIMD width).
const FP_SCAN_MAX_SIMDGROUPS: u64 = 32;

/// The scan kernels (misc.metal), in dispatch order.
pub(crate) const SCAN_KERNELS: [&str; 3] = ["fpScanReduce", "fpScanBlockSums", "fpScanApply"];

/// Accum fault bits (rv32im-sys kernels/metal/cuda_compat.h `RV32IM_FAULT_*`; R17 gave
/// `RV32IM_FAULT_ACCUM_PERM` its own bit, 0x40, distinct from witgen's cycle-index 0x20;
/// `RV32IM_FAULT_ACCUM_SCAN` is defined in misc.metal next to fpScanReduce).
const FAULT_ASSERT: u32 = 0x10;
const FAULT_ACCUM_PERM: u32 = 0x40;
const FAULT_ACCUM_SCAN: u32 = 0x100;

/// An error naming the bits of a nonzero accum fault word.
fn check_fault(bits: u32) -> Result<()> {
    if bits == 0 {
        return Ok(());
    }
    let mut what = Vec::new();
    if bits & FAULT_ASSERT != 0 {
        what.push("a circuit assert failed");
    }
    if bits & FAULT_ACCUM_PERM != 0 {
        what.push("a perm entry was out of range");
    }
    if bits & FAULT_ACCUM_SCAN != 0 {
        what.push("a scanned cell was not a canonical Fp word");
    }
    bail!(
        "rv32im Metal accum: kernels reported fault bits {bits:#x} ({})",
        what.join(", ")
    )
}

/// Test-only phase timestamps (R13 profile; `accum/tests.rs` `accum_ab`). No-op otherwise.
#[cfg(test)]
pub(crate) mod marks {
    use std::{cell::RefCell, time::Instant};
    thread_local! {
        pub(crate) static MARKS: RefCell<Vec<(&'static str, Instant)>> =
            const { RefCell::new(Vec::new()) };
    }
    pub(crate) fn take() -> Vec<(&'static str, Instant)> {
        MARKS.with(|m| std::mem::take(&mut *m.borrow_mut()))
    }
}
#[inline(always)]
fn test_mark(_label: &'static str) {
    #[cfg(test)]
    marks::MARKS.with(|m| m.borrow_mut().push((_label, std::time::Instant::now())));
}

fn u32_of(what: &str, v: usize) -> Result<u32> {
    u32::try_from(v).map_err(|_| anyhow::anyhow!("rv32im Metal accum: {what} = {v} exceeds u32"))
}

impl<MH: MetalHash> MetalCircuitHal<MH> {
    /// The Metal accum path (module docs); `win`: the perm's sort window, `None` for no perm;
    /// `scan`: where step 3 runs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_accum_metal(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<MetalHal<MH>>,
        accum: &MetaBuffer<MetalHal<MH>>,
        global: &MetaBuffer<MetalHal<MH>>,
        mix: &MetaBuffer<MetalHal<MH>>,
        win: Option<u32>,
        scan: ScanMode,
    ) -> Result<()> {
        scope!("metal_accumulate");
        self.ctx.require(Phase::Accum)?;
        let hal = self.hal.as_ref();
        let n = preflight.cycles.len();
        let split = user_accum_split() as usize;
        tracing::debug!("accumulate: {n} (Metal)");

        // Host checks (CUDA throws on non-power-of-two rows; the 32-bit kernel indexing needs
        // rows * cols < 2^32, R1).
        let pow2 = |r: usize| r != 0 && r.is_power_of_two();
        ensure!(
            pow2(data.rows) && pow2(accum.rows),
            "rv32im accum: rows must be a power of two (data {}, accum {})",
            data.rows,
            accum.rows
        );
        ensure!(
            n <= accum.rows && n <= data.rows,
            "rv32im accum: {n} cycles exceed the trace rows (data {}, accum {})",
            data.rows,
            accum.rows
        );
        ensure!(
            accum.cols >= split + 4,
            "rv32im accum: {} accum columns, the layout needs at least {}",
            accum.cols,
            split + 4
        );
        for (what, m) in [
            ("data", data),
            ("accum", accum),
            ("global", global),
            ("mix", mix),
        ] {
            ensure!(
                m.buf.size() >= m.rows * m.cols,
                "rv32im accum: {what} buffer holds {} < {} x {} elements",
                m.buf.size(),
                m.rows,
                m.cols
            );
            u32_of(what, m.rows * m.cols)?;
        }
        // The fault word still to be read after the final wait (GPU scans).
        let pending_fault = if n > 0 {
            self.accum_steps(preflight, data, accum, global, mix, win, scan)?
        } else {
            None
        };

        // 4. finalizeZeroizeAccum over every row (with n = 0 it only zeroizes).
        let fin = FinalizeAccumParams {
            rows: accum.rows as u32,
            cols: accum.cols as u32,
            last_cycle: n as u32,
            user_accum_split: split as u32,
        };
        hal.dispatch(
            self.ctx.pipeline("finalizeZeroizeAccum"),
            &[accum.buf.as_arg(), KernelArg::Bytes(as_bytes(&[fin]))],
            Grid::Threads(accum.rows as u64),
            &[],
        );
        // End of the phase (PLAN 2.6): close the batch; errors are attributed to accum.
        scope!("accum_finalize_wait", hal.try_sync())?;
        test_mark("waited");
        if let Some(fault) = pending_fault {
            check_fault(fault.to_vec()[0])?;
        }
        tracing::debug!(
            "accumulate: done, GPU allocated {} MB",
            hal.allocated_size() >> 20
        );
        Ok(())
    }

    /// Steps 1-3 (module docs) for `n > 0` cycles. Returns the fault word when it is still to be
    /// checked after the phase's final wait (GPU scans); with CPU scans it was checked here.
    #[allow(clippy::too_many_arguments)]
    fn accum_steps(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<MetalHal<MH>>,
        accum: &MetaBuffer<MetalHal<MH>>,
        global: &MetaBuffer<MetalHal<MH>>,
        mix: &MetaBuffer<MetalHal<MH>>,
        win: Option<u32>,
        scan: ScanMode,
    ) -> Result<Option<BufferImpl<u32>>> {
        test_mark("start");
        let hal = self.hal.as_ref();
        let n = preflight.cycles.len();
        // 1. perm. The scratch perm may still be bound by in-flight work of a previous call.
        let win = win.filter(|_| n > 1);
        let perm: Option<BufferImpl<u32>> = match win {
            Some(win) => {
                hal.try_sync()?;
                let buf = {
                    let mut s = self.ctx.scratch();
                    grow(hal, &mut s.perm, "accum_perm", n)
                };
                // SAFETY: synced; nothing that uses `buf` is encoded until this returns.
                let host = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr(), n) };
                test_mark("perm_synced");
                scope!("accum_perm", build_perm(&preflight.cycles, win, host));
                test_mark("perm_built");
                Some(buf)
            }
            None => None,
        };

        // 2. [fill fault, stepAccumA].
        let fault = hal.alloc_buffer::<u32>("accum_fault", 1);
        hal.blit_fill(fault.as_buf(), 0, 4, 0);
        let params = AccumParams {
            count: n as u32,
            has_perm: perm.is_some() as u32,
            data_rows: data.rows as u32,
            data_cols: data.cols as u32,
            accum_rows: accum.rows as u32,
            accum_cols: accum.cols as u32,
            global_rows: global.rows as u32,
            global_cols: global.cols as u32,
            mix_rows: mix.rows as u32,
            mix_cols: mix.cols as u32,
            _pad: [0; 2],
        };
        hal.dispatch(
            self.ctx.pipeline("stepAccumA"),
            &[
                data.buf.as_arg(),
                accum.buf.as_arg(),
                global.buf.as_arg(),
                mix.buf.as_arg(),
                // perm is read only when hasPerm; any buffer is valid otherwise.
                perm.as_ref()
                    .map_or_else(|| accum.buf.as_arg(), |p| p.as_arg()),
                fault.as_arg(),
                KernelArg::Bytes(as_bytes(&[params])),
            ],
            Grid::Threads(n as u64),
            &[],
        );
        if scan == ScanMode::Gpu {
            // 3. GPU scans of the last 4 columns over the first n rows, same batch.
            self.encode_fp_scan(&accum.buf, accum.rows, accum.cols - 4, 4, n, &fault, None)?;
            test_mark("encoded");
            return Ok(Some(fault));
        }
        scope!("accum_step_wait", hal.try_sync())?;
        test_mark("step_waited");
        check_fault(fault.to_vec()[0])?;

        // 3. CPU scans of the last 4 columns over the first n rows.
        {
            scope!("accum_scans");
            let rows = accum.rows;
            let cols = accum.cols;
            // SAFETY: synced above and nothing is encoded until the scans are done.
            let all = unsafe {
                std::slice::from_raw_parts_mut(accum.buf.as_mut_ptr() as *mut u32, rows * cols)
            };
            all[(cols - 4) * rows..]
                .par_chunks_mut(rows)
                .for_each(|col| fp_inclusive_scan(&mut col[..n]));
        }
        test_mark("scanned");

        Ok(None)
    }

    /// The scan threadgroup size: `want` (default [FP_SCAN_TG]) capped by every scan PSO's
    /// maximum and rounded down to a multiple of the SIMD width, so the block size of
    /// fpScanReduce and fpScanApply agree and each SIMD group is full. Also checks that the
    /// simdgroup sums fit one simdgroup (kernel assumption).
    pub(crate) fn fp_scan_tg(&self, want: Option<u64>) -> Result<u64> {
        let mut max = want.unwrap_or(FP_SCAN_TG);
        let mut width = 1;
        for name in SCAN_KERNELS {
            let pso = self.ctx.pipeline(name);
            max = max.min(pso.max_total_threads_per_threadgroup());
            width = width.max(pso.thread_execution_width());
        }
        let tg = max / width * width;
        ensure!(
            tg >= width && tg / width <= width.min(FP_SCAN_MAX_SIMDGROUPS),
            "rv32im Metal accum scan: no usable threadgroup size (max {max}, SIMD width {width})"
        );
        Ok(tg)
    }

    /// Encode (no wait) the in-place inclusive Fp prefix scan of rows `[0, n)` of columns
    /// `col0 .. col0 + ncols` of the column-major `rows`-row buffer `buf` (misc.metal's three
    /// scan kernels). A non-canonical input ORs [FAULT_ACCUM_SCAN] into `fault`. `tg`: the
    /// threadgroup size (tests; default [Self::fp_scan_tg]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_fp_scan<T: Clone>(
        &self,
        buf: &BufferImpl<T>,
        rows: usize,
        col0: usize,
        ncols: usize,
        n: usize,
        fault: &BufferImpl<u32>,
        tg: Option<u64>,
    ) -> Result<()> {
        ensure!(
            n <= rows && (col0 + ncols) * rows <= buf.size(),
            "rv32im Metal scan: {n} rows of columns {col0}..{} exceed the buffer ({} x {rows})",
            col0 + ncols,
            buf.size() / rows.max(1)
        );
        u32_of("scanned cells", (col0 + ncols) * rows)?;
        if n == 0 || ncols == 0 {
            return Ok(());
        }
        let hal = self.hal.as_ref();
        let tg = match tg {
            Some(t) => t,
            None => self.fp_scan_tg(None)?,
        };
        let block = tg as usize * FP_SCAN_PER_THREAD as usize;
        let blocks = n.div_ceil(block);
        let params = FpScanParams {
            n: n as u32,
            rows: rows as u32,
            col0: col0 as u32,
            blocks: blocks as u32,
        };
        let sums = hal.alloc_buffer::<u32>("accum_scan_sums", blocks * ncols);
        let size = |x: usize, y: usize| MTLSize::new(x as u64, y as u64, 1);
        // One word per simdgroup; fp_scan_tg guarantees at most FP_SCAN_MAX_SIMDGROUPS.
        let wsum_bytes = 4 * FP_SCAN_MAX_SIMDGROUPS;
        let threads = size(tg as usize, 1);
        let p = [params];
        let pbytes = as_bytes(&p);
        let [reduce, block_sums, apply] = SCAN_KERNELS;
        hal.dispatch(
            self.ctx.pipeline(reduce),
            &[
                buf.as_arg(),
                sums.as_arg(),
                fault.as_arg(),
                KernelArg::Bytes(pbytes),
            ],
            Grid::Groups {
                groups: size(blocks, ncols),
                threads,
            },
            &[0, wsum_bytes],
        );
        hal.dispatch(
            self.ctx.pipeline(block_sums),
            &[sums.as_arg(), KernelArg::Bytes(pbytes)],
            Grid::Groups {
                groups: size(1, ncols),
                threads,
            },
            &[0, wsum_bytes],
        );
        hal.dispatch(
            self.ctx.pipeline(apply),
            &[buf.as_arg(), sums.as_arg(), KernelArg::Bytes(pbytes)],
            Grid::Groups {
                groups: size(blocks, ncols),
                threads,
            },
            &[block as u64 * 4, wsum_bytes],
        );
        Ok(())
    }

    /// R7's path: the CPU accum kernel over host views of the Metal buffers.
    pub(crate) fn step_accum_cpu(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<MetalHal<MH>>,
        accum: &MetaBuffer<MetalHal<MH>>,
        global: &MetaBuffer<MetalHal<MH>>,
        mix: &MetaBuffer<MetalHal<MH>>,
    ) -> Result<()> {
        scope!("metal_accumulate(cpu)");
        let cycles = preflight.cycles.len();
        tracing::debug!("accumulate: {cycles} (CPU kernels over Metal buffers)");
        self.sync()?;
        let raw = |m: &MetaBuffer<MetalHal<MH>>| RawBuffer {
            // SAFETY: synced, and nothing is encoded until this call returns.
            buf: unsafe { host_slice(&m.buf) }.as_ptr(),
            rows: m.rows,
            cols: m.cols,
            checked: m.checked,
        };
        let buffers = RawAccumBuffers {
            data: raw(data),
            accum: raw(accum),
            global: raw(global),
            mix: raw(mix),
        };
        let preflight = RawPreflightTrace {
            cycles: preflight.cycles.as_ptr(),
            txns: preflight.txns.as_ptr(),
            bigint_bytes: preflight.bigint_bytes.as_ptr(),
            txns_len: preflight.txns.len() as u32,
            bigint_bytes_len: preflight.bigint_bytes.len() as u32,
            table_split_cycle: preflight.table_split_cycle,
        };
        ffi_wrap(|| unsafe { risc0_circuit_rv32im_cpu_accum(&buffers, &preflight, cycles as u32) })
    }
}

impl<MH: MetalHash> CircuitAccumulator<MetalHal<MH>> for MetalCircuitHal<MH> {
    fn step_accum(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<MetalHal<MH>>,
        accum: &MetaBuffer<MetalHal<MH>>,
        global: &MetaBuffer<MetalHal<MH>>,
        mix: &MetaBuffer<MetalHal<MH>>,
    ) -> Result<()> {
        count(Phase::Accum);
        phase_mark(&self.hal, "accum start");
        let res = if accum_on_cpu() {
            self.step_accum_cpu(preflight, data, accum, global, mix)
        } else {
            self.step_accum_metal(preflight, data, accum, global, mix, sort_win(), scan_mode())
        };
        phase_mark(&self.hal, "accum end");
        res
    }

    /// finalizeZeroizeAccum zeroizes (as CUDA); the CPU fallback leaves INVALID cells.
    fn step_accum_zeroizes(&self) -> bool {
        !accum_on_cpu()
    }
}

#[cfg(test)]
mod tests;
