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

//! Recursion witness generation on the Metal HAL (docs/metal-port/PLAN.md 4.3, item C6).
//!
//! Parallel mode mirrors the CUDA witgen (`recursion-sys/kernels/cuda/ffi.cu`,
//! `HostExecContext` + `risc0_circuit_recursion_cuda_witgen_chains`, lines ~1060-1300 and
//! 1593-1617) with the host work of plan 4.3 on the CPU over the HAL's shared buffers:
//!
//! 1. encode: zero the fault word, fill `womRows` with the invalid pattern and `womIndex` with 0
//!    (CUDA `cudaMemset`s) -> sync;
//! 2. host: copy `wom`, `cycles` (the kernels advance `iopIdx`, so the caller's trace stays
//!    pristine) and `iops` into HAL scratch; with host chains, mark their heads
//!    (`isParSafe = 2`) and scatter their rows (CUDA `markHostChains` / `scatterHostChains`);
//! 3. `parStepExec` -> wait -> fault check;
//! 4. host: sort the WOM rows exactly as `std::sort` would (a port of `cxx/ffi.cpp`
//!    `sortWomFast`, with the full sort as fallback) and exclusive-scan `womIndex`
//!    (CUDA: `thrust::sort` + `thrust::exclusive_scan`);
//! 5. `injectWomBacks` -> `parStepVerifyWom` -> wait -> fault check.
//!
//! The sequential modes (`SeqForward`, `SeqReverse`) are not ported (plan 4.3): they run the CPU
//! witgen (`risc0_circuit_recursion_cpu_witgen`) over host views of the shared buffers.
//! `upload_host_witness` (errata 8.4) copies a host witness into the shared `data` / `global`.

use std::{
    cell::Cell,
    sync::{
        atomic::{AtomicUsize, Ordering},
        LazyLock,
    },
};

use anyhow::{bail, ensure, Result};
use rayon::prelude::*;
use risc0_circuit_recursion_sys::{
    risc0_circuit_recursion_cpu_witgen, RawExecBuffers, RawPreflightCycle, RawPreflightTrace,
    StepMode,
};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    field::baby_bear::BabyBearElem,
    hal::{
        metal::{as_bytes, BufferImpl, Grid, KernelArg, MetalHal, MetalHash},
        Buffer as _,
    },
};

use super::{grow, host_slice, MetalCircuitHal, Phase};
use crate::{
    prove::{hal::CircuitWitnessGenerator, witgen::HostChains},
    CircuitImpl, CIRCUIT,
};

const CYCLE_WORDS: usize = std::mem::size_of::<RawPreflightCycle>() / 4;
const _: () = assert!(std::mem::size_of::<RawPreflightCycle>() == 8);

/// `kMaxWomRowsPerCycle` (context.h).
pub(crate) const MAX_WOM_ROWS_PER_CYCLE: usize = 9;
/// Words of one `WomArgumentRow` (`Fp addr; FpExt value;`, 20 bytes, raw Montgomery words).
pub(crate) const WOM_ROW_WORDS: usize = 5;
/// `kInvalidPattern`: the fill of unwritten WOM rows (and of INVALID field elements).
const INVALID: u32 = 0xffff_ffff;

/// One WOM argument row as the kernels store it: raw (Montgomery) words of addr, value[0..4].
pub(crate) type WomRow = [u32; WOM_ROW_WORDS];

/// `RecExecParams` (recursion-sys/kernels/metal/context.h), bound with `set_bytes`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RecExecParams {
    pub total_cycles: u32,
    pub num_cycles: u32,
    pub num_woms: u32,
    pub num_iops: u32,
    pub num_wom_rows: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}
const _: () = assert!(std::mem::size_of::<RecExecParams>() == 32);
// SAFETY: repr(C), eight u32 fields, no padding (size asserted above).
unsafe impl bytemuck::NoUninit for RecExecParams {}

/// The witgen's HAL-owned scratch (plan 2.6; the cycles copy is [super::ScratchBuffers::cycles]).
#[derive(Default)]
pub(crate) struct WitgenScratch {
    /// `trace.wom` (FpExt, 4 words each).
    pub wom: Option<BufferImpl<u32>>,
    /// `trace.iops` (FpExt, 4 words each).
    pub iops: Option<BufferImpl<u32>>,
    /// `numCycles * 9` WOM argument rows (5 words each).
    pub wom_rows: Option<BufferImpl<u32>>,
    /// Per cycle: rows written (after exec), then their exclusive scan (before verify).
    pub wom_index: Option<BufferImpl<u32>>,
}

/// Witgen calls counted for tests and logs.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Call {
    /// Served by the Metal Parallel path.
    Parallel = 0,
    /// Served by the CPU witgen (sequential modes).
    SeqCpu = 1,
    /// Witgens that applied host chains.
    Chain = 2,
}

/// Process-wide call counts, indexed by [Call].
static CALLS: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];

thread_local! {
    /// The same counts for the calling thread only. The witgen runs on the thread that called
    /// the prover, so a test asserts on the calls its own thread made: the process-wide counts
    /// also move with every other test of the parallel harness (PLAN 8.7, C10 fix round 1).
    static THREAD_CALLS: Cell<[usize; 3]> = const { Cell::new([0; 3]) };
}

fn count(call: Call) {
    CALLS[call as usize].fetch_add(1, Ordering::Relaxed);
    THREAD_CALLS.with(|c| {
        let mut v = c.get();
        v[call as usize] += 1;
        c.set(v);
    });
}

/// Calls of kind `call` in this process.
#[allow(dead_code)]
pub(crate) fn process_calls(call: Call) -> usize {
    CALLS[call as usize].load(Ordering::Relaxed)
}

/// Calls of kind `call` made on the current thread.
#[allow(dead_code)]
pub(crate) fn thread_calls(call: Call) -> usize {
    THREAD_CALLS.with(|c| c.get()[call as usize])
}

/// `RISC0_RECURSION_FAST_SORT=0` disables the counting sort (as for the CPU witgen).
static FAST_SORT: LazyLock<bool> =
    LazyLock::new(|| std::env::var("RISC0_RECURSION_FAST_SORT").as_deref() != Ok("0"));
/// `RISC0_RECURSION_FAST_SORT_CHECK`: compare the counting sort with the full sort.
static FAST_SORT_CHECK: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("RISC0_RECURSION_FAST_SORT_CHECK").is_some());

/// `RISC0_METAL_REC_WITGEN_CHECK=1`: after every Parallel Metal witgen, run the CPU witgen on a
/// host snapshot of the inputs and fail on the first differing element (validation for real
/// lift/join traces; costs a host copy of `data` and the CPU witgen per call).
static WITGEN_CHECK: LazyLock<bool> =
    LazyLock::new(|| std::env::var("RISC0_METAL_REC_WITGEN_CHECK").as_deref() == Ok("1"));
/// Witgens compared by `RISC0_METAL_REC_WITGEN_CHECK`.
pub(crate) static WITGEN_CHECKED: AtomicUsize = AtomicUsize::new(0);

/// `RISC0_METAL_REC_ROW_PASSES` (read once): `1` turns on the witness-buffer row passes (C11
/// sub-track; see `has_row_passes`). Off by default: C11's interleaved A/B found no change.
pub(crate) const ROW_PASSES_ENV: &str = "RISC0_METAL_REC_ROW_PASSES";

thread_local! {
    /// Test override of [ROW_PASSES_ENV].
    static ROW_PASSES_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

fn row_passes_enabled() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var(ROW_PASSES_ENV).as_deref() == Ok("1"));
    ROW_PASSES_OVERRIDE.with(|c| c.get()).unwrap_or(*ON)
}

/// Run `f` with the row passes forced on or off on this thread (tests).
#[allow(dead_code)]
pub(crate) fn with_row_passes<R>(on: bool, f: impl FnOnce() -> R) -> R {
    let old = ROW_PASSES_OVERRIDE.with(|c| c.replace(Some(on)));
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            ROW_PASSES_OVERRIDE.with(|c| c.set(self.0));
        }
    }
    let _r = Restore(old);
    f()
}

/// `RecRowParams` (recursion-sys/kernels/metal/context.h), bound with `set_bytes`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RecRowParams {
    pub log_rows: u32,
    pub live: u32,
    pub cols: u32,
    pub zk: u32,
    pub value: u32,
    pub _pad: [u32; 3],
}
const _: () = assert!(std::mem::size_of::<RecRowParams>() == 32);

/// (log2 rows, columns) of a column-major buffer of `rows` (a power of two, >= 4) rows per column
/// (as hip.rs; the kernels work on row quads).
fn row_shape(buf: &BufferImpl<BabyBearElem>, rows: usize) -> Result<(u32, u32)> {
    ensure!(
        rows.is_power_of_two() && rows >= 4 && buf.size().is_multiple_of(rows),
        "row passes: {} elements in rows of {rows}",
        buf.size()
    );
    let cols = buf.size() / rows;
    ensure!(
        cols <= u32::MAX as usize && buf.size() / 4 <= u32::MAX as usize,
        "row passes: buffer too large"
    );
    Ok((rows.trailing_zeros(), cols as u32))
}

/// The canonical value of a raw (Montgomery) word, as CUDA's `Fp::asUInt32()` and the CPU
/// `WomArgumentRow::operator<` (canonical `addr`, `Fp::operator<` on the value) compare them.
#[inline]
fn canon(raw: u32) -> u32 {
    BabyBearElem::new_raw(raw).as_u32()
}

/// The `operator<` key of a row: (addr, value[0], .., value[3]) as canonical values.
#[inline]
fn sort_key(r: &WomRow) -> [u32; WOM_ROW_WORDS] {
    r.map(canon)
}

/// Sort the written WOM rows and exclusive-scan the per-cycle counts, in place.
///
/// `rows` holds `counts.len() * 9` rows; cycle `c` wrote slots `[c * 9, c * 9 + counts[c])` and
/// every other slot holds the invalid pattern. On return rows `[0, nv)` (nv = the sum of the
/// counts) are the written rows ordered by `operator<` on canonical values (exactly what
/// `std::sort` / `thrust::sort` of all rows gives: the invalid rows, address 0xffffffff, sort
/// last), every other row is invalid, and `counts` holds the exclusive scan. Returns nv.
///
/// Port of `cxx/ffi.cpp` `sortWomFast`: a counting sort on the address over the written rows,
/// runs of equal addresses that are not all identical finished by the full comparison; when an
/// address is >= nv (not dense) it falls back to a full sort.
pub(crate) fn sort_wom_rows(rows: &mut [WomRow], counts: &mut [u32]) -> Result<usize> {
    let num_cycles = counts.len();
    ensure!(
        rows.len() >= num_cycles * MAX_WOM_ROWS_PER_CYCLE,
        "WOM rows: {} rows for {num_cycles} cycles",
        rows.len()
    );
    let mut nv = 0usize;
    let mut max_addr = 0u32;
    for (c, &n) in counts.iter().enumerate() {
        if n as usize > MAX_WOM_ROWS_PER_CYCLE {
            bail!("recursion Metal Witgen: cycle {c} wrote {n} WOM rows (max 9)");
        }
        nv += n as usize;
        let base = c * MAX_WOM_ROWS_PER_CYCLE;
        for r in &rows[base..base + n as usize] {
            max_addr = max_addr.max(canon(r[0]));
        }
    }
    let written = |c: usize| {
        let base = c * MAX_WOM_ROWS_PER_CYCLE;
        base..base + counts[c] as usize
    };

    let full_sort = |rows: &[WomRow]| -> Vec<WomRow> {
        let mut v: Vec<WomRow> = Vec::with_capacity(nv);
        for c in 0..num_cycles {
            v.extend_from_slice(&rows[written(c)]);
        }
        // Rows with equal keys are bit-identical (canonical values < P have one raw form), so an
        // unstable sort gives the unique std::sort result.
        v.par_sort_unstable_by_key(sort_key);
        v
    };

    let sorted = if *FAST_SORT && (max_addr as usize) < nv {
        let mut cnt = vec![0u32; max_addr as usize + 2];
        for c in 0..num_cycles {
            for r in &rows[written(c)] {
                cnt[canon(r[0]) as usize + 1] += 1;
            }
        }
        for d in 1..cnt.len() {
            cnt[d] += cnt[d - 1];
        }
        let mut out = vec![[INVALID; WOM_ROW_WORDS]; nv];
        for c in 0..num_cycles {
            for r in &rows[written(c)] {
                let a = canon(r[0]) as usize;
                out[cnt[a] as usize] = *r;
                cnt[a] += 1;
            }
        }
        let mut i = 0;
        while i < nv {
            let mut j = i + 1;
            let mut same = true;
            while j < nv && out[j][0] == out[i][0] {
                same &= out[j] == out[i];
                j += 1;
            }
            if j - i > 1 && !same {
                out[i..j].sort_unstable_by_key(sort_key);
            }
            i = j;
        }
        if *FAST_SORT_CHECK {
            let reference = full_sort(rows);
            if reference != out {
                bail!("recursion Metal Witgen: fast WOM sort differs from the full sort");
            }
        }
        out
    } else {
        full_sort(rows)
    };

    // Invalidate the written slots outside [0, nv) (all other slots are invalid already), then
    // place the sorted rows.
    for c in 0..num_cycles {
        let range = written(c);
        let lo = range.start.max(nv);
        if lo < range.end {
            rows[lo..range.end].fill([INVALID; WOM_ROW_WORDS]);
        }
    }
    rows[..nv].copy_from_slice(&sorted);

    // Exclusive scan of the counts (std::exclusive_scan / thrust::exclusive_scan).
    let mut acc = 0u32;
    for n in counts.iter_mut() {
        let v = *n;
        *n = acc;
        acc += v;
    }
    Ok(nv)
}

/// A raw pointer that may be shared by rayon tasks writing disjoint elements.
#[derive(Clone, Copy)]
struct SharedPtr(*mut u32);
// SAFETY: only used for writes to disjoint indices while the owner is borrowed.
unsafe impl Send for SharedPtr {}
unsafe impl Sync for SharedPtr {}

/// Host-side `markHostChains` + `scatterHostChains` (ffi.cu:687-702) into the shared buffers.
/// Errors (instead of clamping) on an inconsistent chain set, before writing anything; the
/// witgen then fails like a kernel fault.
#[allow(clippy::too_many_arguments)]
fn apply_host_chains(
    hc: &HostChains,
    total_cycles: usize,
    num_cycles: usize,
    data_cols_total: usize,
    cycles: &mut [u32],
    data: &mut [u32],
    wom_rows: &mut [u32],
    wom_index: &mut [u32],
) -> Result<()> {
    let n = hc.cycles.len();
    ensure!(
        n > 0 && hc.rows.len().is_multiple_of(n),
        "host chains: bad row block"
    );
    let data_cols = hc.rows.len() / n;
    ensure!(
        data_cols <= data_cols_total
            && hc.wom_idx.len() == n
            && hc.wom_rows.len() == n * MAX_WOM_ROWS_PER_CYCLE * WOM_ROW_WORDS,
        "host chains: inconsistent sizes ({data_cols} data columns of {data_cols_total})"
    );
    ensure!(
        data_cols >= MAX_WOM_ROWS_PER_CYCLE,
        "host chains: {data_cols} data columns"
    );
    for &h in &hc.heads {
        ensure!(
            (h as usize) < num_cycles,
            "host chains: head {h} out of range"
        );
    }
    for &c in &hc.cycles {
        ensure!(
            (c as usize) < num_cycles,
            "host chains: cycle {c} out of range"
        );
    }
    for &h in &hc.heads {
        cycles[h as usize * CYCLE_WORDS + 1] = 2; // isParSafe = 2: the device skips the chain
    }
    for (i, &c) in hc.cycles.iter().enumerate() {
        let c = c as usize;
        wom_index[c] = hc.wom_idx[i];
        let src = i * MAX_WOM_ROWS_PER_CYCLE * WOM_ROW_WORDS;
        let dst = c * MAX_WOM_ROWS_PER_CYCLE * WOM_ROW_WORDS;
        let len = MAX_WOM_ROWS_PER_CYCLE * WOM_ROW_WORDS;
        wom_rows[dst..dst + len].copy_from_slice(&hc.wom_rows[src..src + len]);
    }
    // Column-major data: one task per column, writing rows of disjoint columns.
    let ptr = SharedPtr(data.as_mut_ptr());
    let rows = &hc.rows;
    let cyc = &hc.cycles;
    (0..data_cols).into_par_iter().for_each(|k| {
        let p = ptr;
        let col = k * total_cycles;
        for (i, &c) in cyc.iter().enumerate() {
            // SAFETY: k < data_cols <= the data buffer's columns and c < num_cycles <=
            // total_cycles, so the index is in bounds; columns are disjoint per task.
            unsafe { *p.0.add(col + c as usize) = rows[i * data_cols + k] };
        }
    });
    Ok(())
}

/// The buffer shapes every witgen path relies on (C10): the step kernels and the CPU witgen
/// write every data column and read every ctrl column at `col * total_cycles + (row & mask)`,
/// and the globals at fixed indices, without further bounds checks. A short buffer would be an
/// out-of-bounds GPU write into whatever follows it, so it is an error before anything is encoded.
fn check_witgen_shapes(
    total_cycles: u32,
    preflight: &RawPreflightTrace,
    ctrl: &BufferImpl<BabyBearElem>,
    data: &BufferImpl<BabyBearElem>,
    global: &BufferImpl<BabyBearElem>,
) -> Result<()> {
    use risc0_zkp::adapter::{CircuitInfo as _, TapsProvider as _};
    let steps = total_cycles as usize;
    ensure!(
        steps.is_power_of_two() && preflight.num_cycles as usize <= steps,
        "recursion Metal Witgen: {} cycles in {steps} rows",
        preflight.num_cycles
    );
    for (name, have, need) in [
        ("ctrl", ctrl.size(), CIRCUIT.ctrl_size() * steps),
        ("data", data.size(), CIRCUIT.data_size() * steps),
        ("global", global.size(), CircuitImpl::OUTPUT_SIZE),
    ] {
        ensure!(
            have >= need,
            "recursion Metal Witgen: {name} has {have} elements, needs {need} for {steps} rows"
        );
    }
    Ok(())
}

impl<MH: MetalHash> MetalCircuitHal<MH> {
    /// The sequential modes: the CPU witgen over host views (C5's path).
    fn witgen_cpu(
        &self,
        mode: StepMode,
        total_cycles: u32,
        preflight: &RawPreflightTrace,
        ctrl: &BufferImpl<BabyBearElem>,
        data: &BufferImpl<BabyBearElem>,
        global: &BufferImpl<BabyBearElem>,
    ) -> Result<()> {
        scope!("metal_witgen(cpu)");
        count(Call::SeqCpu);
        let num_cycles = preflight.num_cycles as usize;
        let cycles = grow(
            &self.hal,
            &mut self.scratch().cycles,
            "rec_cycles",
            (num_cycles * CYCLE_WORDS).max(1),
        );
        self.record(Phase::Witgen);
        self.sync()?;
        // SAFETY: synced, nothing is encoded until this call returns; `cycles` holds
        // `num_cycles` RawPreflightCycle (2 x u32, size asserted) and the source as many.
        unsafe {
            if num_cycles > 0 {
                std::ptr::copy_nonoverlapping(
                    preflight.cycles as *const u32,
                    cycles.as_mut_ptr(),
                    num_cycles * CYCLE_WORDS,
                );
            }
        }
        let (ctrl, data, global) =
            unsafe { (host_slice(ctrl), host_slice(data), host_slice(global)) };
        let buffers = RawExecBuffers {
            ctrl: ctrl.as_ptr(),
            data: data.as_ptr(),
            global: global.as_ptr(),
        };
        let trace = RawPreflightTrace {
            wom: preflight.wom,
            cycles: cycles.as_ptr() as *const RawPreflightCycle,
            iops: preflight.iops,
            num_woms: preflight.num_woms,
            num_cycles: preflight.num_cycles,
            num_iops: preflight.num_iops,
        };
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_cpu_witgen(mode, &buffers, &trace, total_cycles)
        })
    }

    /// The Parallel witgen on the GPU (module doc).
    #[allow(clippy::too_many_arguments)]
    fn witgen_parallel(
        &self,
        total_cycles: u32,
        preflight: &RawPreflightTrace,
        host_chains: Option<&HostChains>,
        ctrl: &BufferImpl<BabyBearElem>,
        data: &BufferImpl<BabyBearElem>,
        global: &BufferImpl<BabyBearElem>,
    ) -> Result<()> {
        scope!("metal_witgen");
        count(Call::Parallel);
        let t0 = std::time::Instant::now();
        let steps = total_cycles as usize;
        let num_cycles = preflight.num_cycles as usize;
        let num_woms = preflight.num_woms as usize;
        let num_iops = preflight.num_iops as usize;
        ensure!(
            steps.is_power_of_two() && num_cycles <= steps,
            "recursion Metal Witgen: {num_cycles} cycles in {steps} rows"
        );
        ensure!(
            data.size().is_multiple_of(steps) && data.size() / steps >= WOM_ROW_WORDS,
            "recursion Metal Witgen: data holds {} elements for {steps} rows",
            data.size()
        );
        if num_cycles == 0 {
            return Ok(());
        }
        let num_wom_rows = num_cycles * MAX_WOM_ROWS_PER_CYCLE;
        ensure!(
            num_wom_rows * WOM_ROW_WORDS <= u32::MAX as usize,
            "recursion Metal Witgen: too many cycles ({num_cycles})"
        );

        let (cycles, wom, iops, wom_rows, wom_index) = {
            let mut s = self.scratch();
            let cycles = grow(
                &self.hal,
                &mut s.cycles,
                "rec_cycles",
                num_cycles * CYCLE_WORDS,
            );
            let w = &mut s.witgen;
            (
                cycles,
                grow(&self.hal, &mut w.wom, "rec_wom", (num_woms * 4).max(1)),
                grow(&self.hal, &mut w.iops, "rec_iops", (num_iops * 4).max(1)),
                grow(
                    &self.hal,
                    &mut w.wom_rows,
                    "rec_wom_rows",
                    num_wom_rows * WOM_ROW_WORDS,
                ),
                grow(&self.hal, &mut w.wom_index, "rec_wom_index", num_cycles),
            )
        };
        let fault = self.fault_word();
        // CUDA: cudaMemset(womRows, kInvalidPattern) and cudaMemset(womIndex, 0).
        self.hal.blit_fill(
            wom_rows.as_buf(),
            (wom_rows.offset() * 4) as u64,
            (num_wom_rows * WOM_ROW_WORDS * 4) as u64,
            0xff,
        );
        self.hal.blit_fill(
            wom_index.as_buf(),
            (wom_index.offset() * 4) as u64,
            (num_cycles * 4) as u64,
            0,
        );
        self.record(Phase::Witgen);
        // Host writes below (trace copies, host chains into `data`) follow every encoded use.
        self.sync()?;
        let t_fill = t0.elapsed();

        // SAFETY (this block): synced, nothing is encoded until the dispatches below; every
        // destination holds at least the copied element count (grow sizes above) and the
        // sources are the caller's trace arrays of the stated lengths.
        unsafe {
            std::ptr::copy_nonoverlapping(
                preflight.cycles as *const u32,
                cycles.as_mut_ptr(),
                num_cycles * CYCLE_WORDS,
            );
            if num_woms > 0 {
                std::ptr::copy_nonoverlapping(
                    preflight.wom as *const u32,
                    wom.as_mut_ptr(),
                    num_woms * 4,
                );
            }
            if num_iops > 0 {
                std::ptr::copy_nonoverlapping(
                    preflight.iops as *const u32,
                    iops.as_mut_ptr(),
                    num_iops * 4,
                );
            }
        }
        if let Some(hc) = host_chains.filter(|hc| !hc.heads.is_empty()) {
            scope!("hostChains");
            // SAFETY: synced; distinct buffers; lengths are the views' sizes.
            let (cyc, d, wr, wi) = unsafe {
                (
                    std::slice::from_raw_parts_mut(cycles.as_mut_ptr(), cycles.size()),
                    std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u32, data.size()),
                    std::slice::from_raw_parts_mut(wom_rows.as_mut_ptr(), wom_rows.size()),
                    std::slice::from_raw_parts_mut(wom_index.as_mut_ptr(), wom_index.size()),
                )
            };
            apply_host_chains(hc, steps, num_cycles, data.size() / steps, cyc, d, wr, wi)?;
            count(Call::Chain);
        }
        let t_host = t0.elapsed();

        let params = RecExecParams {
            total_cycles,
            num_cycles: num_cycles as u32,
            num_woms: num_woms as u32,
            num_iops: num_iops as u32,
            num_wom_rows: num_wom_rows as u32,
            ..Default::default()
        };
        let params_bytes = as_bytes(std::slice::from_ref(&params));
        let exec_args = [
            ctrl.as_arg(),
            data.as_arg(),
            global.as_arg(),
            wom.as_arg(),
            cycles.as_arg(),
            iops.as_arg(),
            wom_rows.as_arg(),
            wom_index.as_arg(),
            fault.as_arg(),
            KernelArg::Bytes(params_bytes),
        ];
        let k = self.kernels();
        self.hal.dispatch(
            k.pipeline("parStepExec"),
            &exec_args,
            super::grid_1d(
                super::TgKind::Step,
                k.pipeline("parStepExec"),
                num_cycles as u64,
            ),
            &[],
        );
        self.check_fault(&fault, Phase::Witgen)?;
        let t_exec = t0.elapsed();

        {
            scope!("sortWom");
            // SAFETY: synced by check_fault; nothing is encoded until the dispatches below.
            let (rows, counts) = unsafe {
                (
                    std::slice::from_raw_parts_mut(
                        wom_rows.as_mut_ptr() as *mut WomRow,
                        num_wom_rows,
                    ),
                    std::slice::from_raw_parts_mut(wom_index.as_mut_ptr(), num_cycles),
                )
            };
            sort_wom_rows(rows, counts)?;
        }
        let t_sort = t0.elapsed();

        let fault = self.fault_word();
        let inject_args = [
            data.as_arg(),
            wom_index.as_arg(),
            wom_rows.as_arg(),
            fault.as_arg(),
            KernelArg::Bytes(params_bytes),
        ];
        self.hal.dispatch(
            k.pipeline("injectWomBacks"),
            &inject_args,
            Grid::Threads(num_cycles as u64),
            &[],
        );
        let verify_args = [
            ctrl.as_arg(),
            data.as_arg(),
            global.as_arg(),
            wom.as_arg(),
            cycles.as_arg(),
            iops.as_arg(),
            wom_rows.as_arg(),
            wom_index.as_arg(),
            fault.as_arg(),
            KernelArg::Bytes(params_bytes),
        ];
        self.hal.dispatch(
            k.pipeline("parStepVerifyWom"),
            &verify_args,
            super::grid_1d(
                super::TgKind::Step,
                k.pipeline("parStepVerifyWom"),
                num_cycles as u64,
            ),
            &[],
        );
        self.check_fault(&fault, Phase::Witgen)?;
        self.record(Phase::Witgen);
        let t_all = t0.elapsed();
        super::profile::record("witgen.fill", ms(t_fill));
        super::profile::record("witgen.host", ms(t_host - t_fill));
        super::profile::record("witgen.exec", ms(t_exec - t_host));
        super::profile::record("witgen.sort", ms(t_sort - t_exec));
        super::profile::record("witgen.verify", ms(t_all - t_sort));
        tracing::debug!(
            "metal witgen: {num_cycles} cycles of {steps}, chains={}: fill {:.2}ms, host {:.2}ms, \
             exec {:.2}ms, sort+scan {:.2}ms, inject+verify {:.2}ms, total {:.2}ms",
            host_chains.map_or(0, |hc| hc.heads.len()),
            ms(t_fill),
            ms(t_host - t_fill),
            ms(t_exec - t_host),
            ms(t_sort - t_exec),
            ms(t_all - t_sort),
            ms(t_all)
        );
        Ok(())
    }
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// `RISC0_METAL_REC_WITGEN_CHECK`: the CPU witgen (Parallel, no host chains) from the snapshot
/// `data0` / `global0` must give exactly the Metal witness now in `data` / `global`.
fn check_against_cpu(
    total_cycles: u32,
    preflight: &RawPreflightTrace,
    ctrl: &BufferImpl<BabyBearElem>,
    data: &BufferImpl<BabyBearElem>,
    global: &BufferImpl<BabyBearElem>,
    mut data0: Vec<BabyBearElem>,
    mut global0: Vec<BabyBearElem>,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    let ctrl = ctrl.to_vec();
    let num_cycles = preflight.num_cycles as usize;
    // SAFETY: the caller's trace holds num_cycles cycles; the copy keeps it pristine.
    let mut cycles: Vec<RawPreflightCycle> =
        unsafe { std::slice::from_raw_parts(preflight.cycles, num_cycles) }.to_vec();
    let trace = RawPreflightTrace {
        cycles: cycles.as_mut_ptr(),
        ..*preflight
    };
    let buffers = RawExecBuffers {
        ctrl: ctrl.as_ptr(),
        data: data0.as_mut_ptr(),
        global: global0.as_mut_ptr(),
    };
    ffi_wrap(|| unsafe {
        risc0_circuit_recursion_cpu_witgen(StepMode::Parallel, &buffers, &trace, total_cycles)
    })?;
    let steps = total_cycles as usize;
    let differ = |a: &[BabyBearElem], b: &[BabyBearElem]| {
        a.iter()
            .zip(b)
            .position(|(x, y)| x.as_u32_montgomery() != y.as_u32_montgomery())
    };
    let (d, g) = (data.to_vec(), global.to_vec());
    if let Some(i) = differ(&d, &data0) {
        bail!(
            "RISC0_METAL_REC_WITGEN_CHECK: data differs at col {} row {}: metal {:#x} cpu {:#x}",
            i / steps,
            i % steps,
            d[i].as_u32_montgomery(),
            data0[i].as_u32_montgomery()
        );
    }
    if let Some(i) = differ(&g, &global0) {
        bail!("RISC0_METAL_REC_WITGEN_CHECK: global differs at {i}");
    }
    let n = WITGEN_CHECKED.fetch_add(1, Ordering::Relaxed) + 1;
    eprintln!(
        "[recursion_metal_witgen_check] #{n}: {num_cycles} cycles of {steps}: metal == cpu ({:.1} ms)",
        ms(t0.elapsed())
    );
    Ok(())
}

impl<MH: MetalHash> CircuitWitnessGenerator<MetalHal<MH>> for MetalCircuitHal<MH> {
    fn generate_witness(
        &self,
        mode: StepMode,
        total_cycles: u32,
        preflight: &RawPreflightTrace,
        ctrl: &BufferImpl<BabyBearElem>,
        data: &BufferImpl<BabyBearElem>,
        global: &BufferImpl<BabyBearElem>,
    ) -> Result<()> {
        self.generate_witness_with_chains(mode, total_cycles, preflight, None, ctrl, data, global)
    }

    fn generate_witness_with_chains(
        &self,
        mode: StepMode,
        total_cycles: u32,
        preflight: &RawPreflightTrace,
        host_chains: Option<&HostChains>,
        ctrl: &BufferImpl<BabyBearElem>,
        data: &BufferImpl<BabyBearElem>,
        global: &BufferImpl<BabyBearElem>,
    ) -> Result<()> {
        // Shapes first: a rejected call is not counted as a served witgen phase (C10).
        check_witgen_shapes(total_cycles, preflight, ctrl, data, global)?;
        self.enter(Phase::Witgen);
        match mode {
            // As CUDA: host chains apply to the Parallel mode only.
            StepMode::Parallel if *WITGEN_CHECK => {
                self.sync()?;
                let (data0, global0) = (data.to_vec(), global.to_vec());
                self.witgen_parallel(total_cycles, preflight, host_chains, ctrl, data, global)?;
                check_against_cpu(total_cycles, preflight, ctrl, data, global, data0, global0)
            }
            StepMode::Parallel => {
                self.witgen_parallel(total_cycles, preflight, host_chains, ctrl, data, global)?;
                self.prof_mark("witgen");
                Ok(())
            }
            StepMode::SeqForward | StepMode::SeqReverse => {
                self.witgen_cpu(mode, total_cycles, preflight, ctrl, data, global)
            }
        }
    }

    /// C11 row passes (CUDA/HIP `recInitRows`/`recFillTail`/`recZeroizeRows`, C1's kernels):
    /// on under [ROW_PASSES_ENV]. The generic path fills all of `data` with INVALID and zeroizes
    /// all of it; the row passes write zero (not INVALID) past the live rows and zeroize only
    /// the live rows. Validated by `RISC0_HIP_REC_ZCHECK=1` (prove/witgen.rs) and `c11_tests`.
    fn has_row_passes(&self) -> bool {
        row_passes_enabled()
    }

    fn init_rows(&self, buf: &BufferImpl<BabyBearElem>, rows: usize, live: usize) -> Result<()> {
        let (log_rows, cols) = row_shape(buf, rows)?;
        let p = RecRowParams {
            log_rows,
            live: live.min(rows) as u32,
            cols,
            ..Default::default()
        };
        let quads = (buf.size() / 4) as u64;
        self.hal.dispatch(
            self.kernels().pipeline("recInitRows"),
            &[buf.as_arg(), KernelArg::Bytes(bytemuck::bytes_of(&p))],
            Grid::Threads(quads),
            &[],
        );
        Ok(())
    }

    fn fill_tail(
        &self,
        buf: &BufferImpl<BabyBearElem>,
        rows: usize,
        zk: usize,
        value: BabyBearElem,
    ) -> Result<()> {
        let (log_rows, cols) = row_shape(buf, rows)?;
        ensure!(zk <= rows, "fill_tail: {zk} rows of {rows}");
        let p = RecRowParams {
            log_rows,
            cols,
            zk: zk as u32,
            value: value.as_u32_montgomery(),
            ..Default::default()
        };
        self.hal.dispatch(
            self.kernels().pipeline("recFillTail"),
            &[buf.as_arg(), KernelArg::Bytes(bytemuck::bytes_of(&p))],
            Grid::Threads(zk as u64 * cols as u64),
            &[],
        );
        Ok(())
    }

    fn zeroize_rows(&self, buf: &BufferImpl<BabyBearElem>, rows: usize, live: usize) -> Result<()> {
        let (log_rows, cols) = row_shape(buf, rows)?;
        let live = live.min(rows) as u32;
        let p = RecRowParams {
            log_rows,
            live,
            cols,
            ..Default::default()
        };
        self.hal.dispatch(
            self.kernels().pipeline("recZeroizeRows"),
            &[buf.as_arg(), KernelArg::Bytes(bytemuck::bytes_of(&p))],
            Grid::Threads2D(live.div_ceil(4) as u64, cols as u64),
            &[],
        );
        Ok(())
    }

    /// Errata 8.4: a host-generated witness is copied into the shared `data` / `global` buffers
    /// (unified memory: a memcpy after the pending GPU work on them completes).
    fn upload_host_witness(
        &self,
        data: &BufferImpl<BabyBearElem>,
        global: &BufferImpl<BabyBearElem>,
        host_data: Vec<BabyBearElem>,
        host_global: Vec<BabyBearElem>,
    ) -> Result<()> {
        ensure!(
            host_data.len() == data.size() && host_global.len() == global.size(),
            "upload_host_witness: host witness {} + {} elements for buffers of {} + {}",
            host_data.len(),
            host_global.len(),
            data.size(),
            global.size()
        );
        self.sync()?;
        data.view_mut(|v| v.copy_from_slice(&host_data));
        global.view_mut(|v| v.copy_from_slice(&host_global));
        Ok(())
    }
}

#[cfg(all(test, feature = "test"))]
mod c6_tests;
