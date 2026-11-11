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

//! rv32im witness generation on the Metal HAL (docs/metal-port/PLAN.md 4.2, item R8).
//!
//! Host side of CUDA `risc0_circuit_rv32im_cuda_witgen_v2` (`rv32im-sys/kernels/cuda/ffi.cu`
//! `setup_exec` + `witgen_v2`), over the kernels of `misc.metal` (`injectCycleCols`) and the
//! generated `witgen_tu.metal` (`par_stepExecW`, `fwd_stepExec`, `rev_stepExec`; bindings in its
//! header, mirrored by [WitgenParams] and [SeqParams] here):
//!
//! 1. The preflight trace (cycles | txns | bigint bytes, 36/20-byte structs asserted in `mod.rs`)
//!    is memcpy'd into the HAL-owned shared buffer (`MetalRv32imCtx::upload_preflight`): the
//!    kernels increment `cycles[].txnIdx` in place, the host trace stays pristine.
//! 2. One batch (command buffer): zero the u8/u16 lookup tables and the fault word (blit fills,
//!    CUDA `cudaMemsetAsync`) → `injectCycleCols` when the trace deferred the five per-cycle
//!    columns → in parallel mode a `pre_data` snapshot of `data` (see below) and `par_stepExecW`
//!    over `[0, split)` then `[split, lastCycle)` (CUDA's two launches); in the sequential modes
//!    one 1x1 `fwd_stepExec` / `rev_stepExec` dispatch per `RV32IM_SEQ_CHUNK` positions (4096),
//!    so no single dispatch runs long enough for the GPU watchdog.
//! 3. `sync()` (phase end, PLAN 2.6), then the fault word: any `RV32IM_FAULT_*` bit → `Err`.
//!
//! `pre_data` (PLAN 4.2: a real buffer by default until single-buffer bit-exactness is proven):
//! the prover passes `data` itself unless `RISC0_USE_PRE_DATA` is set. Then a copy of `data`
//! taken after `injectCycleCols` is bound at index 1, so `par_stepExecW`'s back-row loads read the
//! scattered rows and never a row a neighbouring thread is writing.
//! `RISC0_METAL_WITGEN_SINGLE_BUFFER=1` binds `data` itself instead (CUDA's default
//! `pre_data == data`, saving `rows * 211 * 4` bytes). A separate `pre_data` from the prover
//! (`RISC0_USE_PRE_DATA`) already holds the scattered rows and cycle columns and is bound as is.
//!
//! **The GPU path is the default** (again since R17). `RISC0_METAL_RV32IM_CPU_PHASES` containing
//! `witgen` is the kill switch: witgen then runs the CPU kernel over host views of the shared
//! buffers (R7's delegation), with the CPU HAL's flags. `RISC0_METAL_RV32IM_GPU_PHASES=witgen`, the
//! opt-in of R8-R16, is still accepted and changes nothing.
//!
//! From R8 to R16 the GPU path was opt-in: the generated kernel gave wrong witnesses on identical
//! inputs from time to time (notes/R8.md, R11.md). R17 traced that to the Apple GPU losing a
//! private-array store in the code shape of CUDA's `map()` lowering (a default-constructed array
//! written in a loop that calls a noinline function). `witgen_tu_gen.py` now unrolls every
//! map/reduce and refuses any loop in the call graph (docs/metal-port/notes/R17.md: mechanism,
//! gate, and how to re-verify after a macOS or Metal toolchain update with
//! `witgen_segment_file`, `RISC0_METAL_WITGEN_PERM=sorted` and a `--map-lowering loop` baseline).

use std::sync::LazyLock;

use anyhow::{bail, ensure, Result};
use risc0_circuit_rv32im_sys::{
    risc0_circuit_rv32im_cpu_witgen, RawBuffer, RawExecBuffers, RawPreflightTrace,
};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::hal::{
    metal::{as_bytes, BufferImpl, Grid, KernelArg, MetalHal, MetalHash},
    Buffer as _, Hal as _,
};

use super::{
    count, host_slice, phase_mark, MetalCircuitHal, Phase, TABLE_U16_CELLS, TABLE_U8_CELLS,
};
use crate::prove::{
    hal::{CircuitWitnessGenerator, MetaBuffer, PreflightTrace, StepMode},
    witgen::CYCLE_COLS,
};

/// `WitgenParams` of `witgen_tu.metal` (48 bytes, size and offsets static_asserted in MSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::NoUninit)]
pub(crate) struct WitgenParams {
    pub rows: u32,
    pub data_cols: u32,
    pub global_rows: u32,
    pub global_cols: u32,
    pub cycles_len: u32,
    pub txns_len: u32,
    pub bigint_bytes_len: u32,
    pub table_split_cycle: u32,
    pub start: u32,
    pub count: u32,
    pub use_perm: u32,
    pub _pad: u32,
}
const _: () = assert!(std::mem::size_of::<WitgenParams>() == 48);

/// `SeqParams` of `seq_exec.h` (16 bytes): positions `[begin, end)` of the sequential order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::NoUninit)]
pub(crate) struct SeqParams {
    pub begin: u32,
    pub end: u32,
    pub split: u32,
    pub last_cycle: u32,
}
const _: () = assert!(std::mem::size_of::<SeqParams>() == 16);

/// `InjectParams` of `misc.metal` (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::NoUninit)]
pub(crate) struct InjectParams {
    pub n: u32,
    pub rows: u32,
    pub c_cycle: u32,
    pub c_pc_low: u32,
    pub c_pc_high: u32,
    pub c_state: u32,
    pub c_mode: u32,
    pub _pad: u32,
}
const _: () = assert!(std::mem::size_of::<InjectParams>() == 32);

/// `RV32IM_SEQ_CHUNK` (seq_exec.h): positions per sequential dispatch.
pub(crate) const SEQ_CHUNK: u32 = 4096;

/// Fault bits of the witgen kernels (`cuda_compat.h` `RV32IM_FAULT_*`; 0x40 is accum's
/// `RV32IM_FAULT_ACCUM_PERM`, never set by witgen).
const FAULT_BITS: &[(u32, &str)] = &[
    (0x1, "lookup into an unknown table"),
    (0x2, "lookup index out of range"),
    (0x4, "memory transaction index out of range"),
    (0x8, "bigint byte index out of range"),
    (0x10, "assertion failed"),
    (0x20, "cycle index out of range"),
    (0x80, "getDiffCount cycle out of range"),
];

pub(crate) fn describe_fault(fault: u32) -> String {
    let mut parts: Vec<&str> = FAULT_BITS
        .iter()
        .filter(|(bit, _)| fault & bit != 0)
        .map(|(_, what)| *what)
        .collect();
    if fault & !FAULT_BITS.iter().fold(0, |m, (bit, _)| m | bit) != 0 {
        parts.push("unknown bits");
    }
    format!("{fault:#x} ({})", parts.join(", "))
}

/// Runs an allocating closure, turning the HAL's allocation panic (working-set budget
/// `RISC0_METAL_MAX_WORKING_SET_BYTES` exceeded, or a failed allocation) into this phase's error.
fn no_panic<T>(what: &str, f: impl FnOnce() -> T) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|e| {
        let msg = e
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| e.downcast_ref::<&str>().copied())
            .unwrap_or("allocation failed");
        anyhow::anyhow!("rv32im Metal witgen: allocating {what}: {msg}")
    })
}

/// Whether `RISC0_METAL_RV32IM_CPU_PHASES` (comma list of `witgen,accum,eval`) names `phase`.
pub(crate) fn cpu_phase(phase: &str) -> bool {
    super::cpu_phase(phase)
}

/// Witgen runs the CPU kernel: only when `RISC0_METAL_RV32IM_CPU_PHASES` names `witgen` (the kill
/// switch; the GPU kernel is the default, module doc). Read once per process.
static CPU_WITGEN: LazyLock<bool> = LazyLock::new(|| cpu_phase("witgen"));

/// The effective [CPU_WITGEN] (logged by `segment_prover`).
pub(crate) fn cpu_witgen() -> bool {
    *CPU_WITGEN
}

/// `RISC0_METAL_WITGEN_SINGLE_BUFFER=1`: bind `data` as `pre_data` (CUDA's default).
fn single_buffer() -> bool {
    matches!(
        std::env::var("RISC0_METAL_WITGEN_SINGLE_BUFFER").as_deref(),
        Ok("1")
    )
}

/// Whether the Metal witgen writes the five per-cycle columns itself, so preflight can leave them
/// out of the host injector (`witgen::DEFER_CYCLE_COLS`).
pub(crate) fn defers_cycle_cols() -> bool {
    risc0_zkp::hal::metal::enabled("rv32im") && !*CPU_WITGEN
}

/// Debug: `RISC0_METAL_WITGEN_SYNC_STEPS=1` waits for the GPU after every encoded step (each
/// dispatch, and after `injectCycleCols` and the `pre_data` snapshot), to rule out ordering
/// hazards inside the batch (it does not change the nondeterminism of notes/R8.md).
static SYNC_STEPS: LazyLock<bool> = LazyLock::new(|| {
    matches!(
        std::env::var("RISC0_METAL_WITGEN_SYNC_STEPS").as_deref(),
        Ok("1")
    )
});

/// What parallel witgen binds as `pre_data` (tests force a choice per call).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreData {
    /// The default: a snapshot of `data` when the prover passes `data` itself as `pre_data`.
    Auto,
    /// Bind `data` itself (single-buffer mode).
    Single,
}

impl<MH: MetalHash> CircuitWitnessGenerator<MetalHal<MH>> for MetalCircuitHal<MH> {
    fn writes_cycle_cols(&self) -> bool {
        !*CPU_WITGEN
    }

    fn takes_zeroed_data(&self) -> bool {
        // The Metal kernels' sets are unchecked; the CPU kernel's checked sets expect INVALID.
        !*CPU_WITGEN
    }

    fn generate_witness(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        global: &MetaBuffer<MetalHal<MH>>,
        data: &MetaBuffer<MetalHal<MH>>,
        pre_data: &MetaBuffer<MetalHal<MH>>,
    ) -> Result<()> {
        count(Phase::Witgen);
        phase_mark(&self.hal, "witgen start");
        let res = if *CPU_WITGEN {
            self.cpu_witgen(mode, preflight, global, data)
        } else {
            let pre = if single_buffer() {
                PreData::Single
            } else {
                PreData::Auto
            };
            self.metal_witgen(mode, preflight, global, data, pre_data, pre)
        };
        phase_mark(&self.hal, "witgen end");
        res
    }
}

impl<MH: MetalHash> MetalCircuitHal<MH> {
    /// The Metal witgen (see the module doc).
    pub(crate) fn metal_witgen(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        global: &MetaBuffer<MetalHal<MH>>,
        data: &MetaBuffer<MetalHal<MH>>,
        pre_data: &MetaBuffer<MetalHal<MH>>,
        pre: PreData,
    ) -> Result<()> {
        scope!("metal_witgen");
        let ctx = &self.ctx;
        ctx.require(Phase::Witgen)?;

        let cycles = preflight.cycles.len();
        tracing::debug!("witgen: {cycles} (Metal)");
        // CUDA asserts the same (hal/cuda.rs): one preflight cycle per data row.
        ensure!(
            cycles == data.rows,
            "rv32im Metal witgen: {cycles} preflight cycles for {} data rows",
            data.rows
        );
        // RISC0_ROWS_POW2 is compiled into witgen_tu.metal (CUDA falls back to steps.cu
        // otherwise; Metal has no fallback) and cells are indexed with 32 bits.
        ensure!(
            data.rows.is_power_of_two(),
            "rv32im Metal witgen: data rows {} not a power of two",
            data.rows
        );
        ensure!(
            (data.rows as u64) * (data.cols as u64) < 1 << 32,
            "rv32im Metal witgen: {} x {} data cells exceed 32-bit indexing",
            data.rows,
            data.cols
        );
        ensure!(data.buf.size() >= data.rows * data.cols && global.buf.size() >= global.cols);
        let split = preflight.table_split_cycle;
        ensure!(
            split as usize <= cycles,
            "rv32im Metal witgen: table split {split} beyond {cycles} cycles"
        );
        ensure!(
            preflight.txns.len() < 1 << 32 && preflight.bigint_bytes.len() < 1 << 32,
            "rv32im Metal witgen: preflight trace too large"
        );

        let aliased = pre_data.buf.as_device_ptr() == data.buf.as_device_ptr();
        if !aliased {
            ensure!(
                pre_data.rows == data.rows
                    && pre_data.cols == data.cols
                    && pre_data.buf.size() >= data.rows * data.cols,
                "rv32im Metal witgen: pre_data shape differs from data"
            );
        }

        let hal = &self.hal;
        let (pf, layout) = no_panic("the preflight copy", || ctx.upload_preflight(preflight))??;
        let (table_u8, table_u16) = no_panic("the lookup tables", || ctx.tables())?;
        let fault: BufferImpl<u32> =
            no_panic("the fault word", || hal.alloc_buffer("witgen_fault", 1))?;

        // A binding as (buffer, byte offset), so it can be bound by several dispatches.
        type Bind<'a> = (&'a metal::BufferRef, u64);
        fn bind<T>(buf: &BufferImpl<T>) -> Bind<'_> {
            (
                buf.as_buf(),
                (buf.offset() * std::mem::size_of::<T>()) as u64,
            )
        }
        // Bindings 3..5: an empty section is bound at the start of the buffer instead (any valid
        // buffer; the kernels clamp against the lengths).
        let section = |off: usize, len: usize| -> Bind<'_> {
            let (buf, base) = bind(&pf);
            (buf, if len == 0 { base } else { base + off as u64 })
        };
        fn arg((buffer, offset): Bind<'_>) -> KernelArg<'_> {
            KernelArg::Buffer { buffer, offset }
        }

        // 1. Tables and fault word (CUDA setup_exec memsets).
        hal.blit_fill(table_u8.as_buf(), 0, (TABLE_U8_CELLS * 4) as u64, 0);
        hal.blit_fill(table_u16.as_buf(), 0, (TABLE_U16_CELLS * 4) as u64, 0);
        hal.blit_fill(fault.as_buf(), bind(&fault).1, 4, 0);

        // 2. The deferred per-cycle columns (CUDA witgen_v2: cycle_cols != nullptr).
        if preflight.cycle_cols_deferred {
            let p = [InjectParams {
                n: cycles as u32,
                rows: data.rows as u32,
                c_cycle: CYCLE_COLS[0],
                c_pc_low: CYCLE_COLS[1],
                c_pc_high: CYCLE_COLS[2],
                c_state: CYCLE_COLS[3],
                c_mode: CYCLE_COLS[4],
                _pad: 0,
            }];
            hal.dispatch(
                ctx.pipeline("injectCycleCols"),
                &[
                    data.buf.as_arg(),
                    arg(section(layout.cycles_offset, layout.cycles_len)),
                    KernelArg::Bytes(as_bytes(&p)),
                ],
                Grid::Threads(cycles as u64),
                &[],
            );
        }
        if *SYNC_STEPS {
            hal.try_sync()?;
        }

        // 3. pre_data (parallel mode only; the sequential kernels ignore binding 1).
        let snapshot = if mode == StepMode::Parallel && aliased && pre == PreData::Auto {
            // The HAL panics when an allocation exceeds the working-set budget
            // (RISC0_METAL_MAX_WORKING_SET_BYTES); report that as this phase's error.
            let size = data.buf.size();
            let snap = no_panic("the pre_data snapshot", || {
                hal.alloc_elem("witgen_pre_data", size)
            })
            .map_err(|e| e.context("RISC0_METAL_WITGEN_SINGLE_BUFFER=1 runs without it"))?;
            hal.eltwise_copy_elem(&snap, &data.buf);
            if *SYNC_STEPS {
                hal.try_sync()?;
            }
            Some(snap)
        } else {
            None
        };
        let pre_bind = match &snapshot {
            Some(snap) => bind(snap),
            None => bind(&pre_data.buf),
        };
        let binds: [Bind<'_>; 9] = [
            bind(&data.buf),
            pre_bind,
            bind(&global.buf),
            section(layout.cycles_offset, layout.cycles_len),
            section(layout.txns_offset, layout.txns_len),
            section(layout.bigint_offset, layout.bigint_len),
            bind(&table_u8),
            bind(&table_u16),
            bind(&fault),
        ];
        let base = WitgenParams {
            rows: data.rows as u32,
            data_cols: data.cols as u32,
            global_rows: global.rows as u32,
            global_cols: global.cols as u32,
            cycles_len: cycles as u32,
            txns_len: layout.txns_len as u32,
            bigint_bytes_len: layout.bigint_len as u32,
            table_split_cycle: split,
            ..Default::default()
        };
        // Bindings 0..9, then `extra` at 10.
        let encode = |pso: &metal::ComputePipelineStateRef,
                      params: &WitgenParams,
                      extra: KernelArg<'_>,
                      grid: Grid| {
            let mut args: Vec<KernelArg<'_>> = binds.iter().map(|&b| arg(b)).collect();
            args.push(KernelArg::Bytes(as_bytes(std::slice::from_ref(params))));
            args.push(extra);
            hal.dispatch(pso, &args, grid, &[]);
            if *SYNC_STEPS {
                hal.try_sync()
            } else {
                Ok(())
            }
        };

        let last_cycle = cycles as u32;
        let mut perm_bufs = Vec::new();
        match mode {
            StepMode::Parallel => {
                // CUDA: par_stepExec over [0, split), then [split, lastCycle).
                let pso = ctx.pipeline("par_stepExecW");
                for (start, count) in [(0, split), (split, last_cycle - split)] {
                    // Tests only: a launch order other than start + i (`test_perm`).
                    if let Some(p) = test_perm(preflight, start, count) {
                        let params = WitgenParams {
                            start,
                            count: p.len() as u32,
                            use_perm: 1,
                            ..base
                        };
                        let pb = hal.copy_from_u32("witgen_perm", &p);
                        encode(pso, &params, arg(bind(&pb)), Grid::Threads(p.len() as u64))?;
                        perm_bufs.push(pb);
                        continue;
                    }
                    let params = WitgenParams {
                        start,
                        count,
                        use_perm: 0,
                        ..base
                    };
                    // perm (binding 10) is not read with use_perm = 0: bind the fault word.
                    encode(pso, &params, arg(bind(&fault)), Grid::Threads(count as u64))?;
                }
            }
            StepMode::SeqForward | StepMode::SeqReverse => {
                let pso = ctx.pipeline(if mode == StepMode::SeqForward {
                    "fwd_stepExec"
                } else {
                    "rev_stepExec"
                });
                let one = metal::MTLSize::new(1, 1, 1);
                let mut begin = 0;
                while begin < last_cycle {
                    let end = last_cycle.min(begin + SEQ_CHUNK);
                    let sp = [SeqParams {
                        begin,
                        end,
                        split,
                        last_cycle,
                    }];
                    let grid = Grid::Groups {
                        groups: one,
                        threads: one,
                    };
                    encode(pso, &base, KernelArg::Bytes(as_bytes(&sp)), grid)?;
                    begin = end;
                }
            }
        }

        // Phase end: wait, then the fault word.
        hal.try_sync()?;
        drop(perm_bufs);
        drop(snapshot);
        // SAFETY: synced above; nothing using `fault` is encoded.
        let fault = unsafe { host_slice(&fault) }[0];
        if fault != 0 {
            bail!(
                "rv32im Metal witgen ({}): kernel fault {}",
                match mode {
                    StepMode::Parallel => "parallel",
                    StepMode::SeqForward => "forward",
                    StepMode::SeqReverse => "reverse",
                },
                describe_fault(fault)
            );
        }
        Ok(())
    }

    /// CPU fallback (`RISC0_METAL_RV32IM_CPU_PHASES=witgen`): the CPU witgen kernel over host views
    /// of the Metal buffers, as `hal/cpu.rs` calls it.
    pub(crate) fn cpu_witgen(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        global: &MetaBuffer<MetalHal<MH>>,
        data: &MetaBuffer<MetalHal<MH>>,
    ) -> Result<()> {
        scope!("metal_witgen(cpu)");
        let cycles = preflight.cycles.len();
        tracing::debug!("witgen: {cycles} (CPU kernels over Metal buffers)");
        self.sync()?;
        // SAFETY: synced, and nothing is encoded until this call returns.
        let (global_buf, data_buf) = unsafe { (host_slice(&global.buf), host_slice(&data.buf)) };
        let buffers = RawExecBuffers {
            global: RawBuffer {
                buf: global_buf.as_ptr(),
                rows: global.rows,
                cols: global.cols,
                checked: global.checked,
            },
            data: RawBuffer {
                buf: data_buf.as_ptr(),
                rows: data.rows,
                cols: data.cols,
                checked: data.checked,
            },
            // CPU runs sequentially; no race condition, so pre_data aliases data (as hal/cpu.rs).
            pre_data: RawBuffer {
                buf: data_buf.as_ptr(),
                rows: data.rows,
                cols: data.cols,
                checked: false,
            },
        };
        let preflight = RawPreflightTrace {
            cycles: preflight.cycles.as_ptr(),
            txns: preflight.txns.as_ptr(),
            bigint_bytes: preflight.bigint_bytes.as_ptr(),
            txns_len: preflight.txns.len() as u32,
            bigint_bytes_len: preflight.bigint_bytes.len() as u32,
            table_split_cycle: preflight.table_split_cycle,
        };
        ffi_wrap(|| unsafe {
            risc0_circuit_rv32im_cpu_witgen(mode as u32, &buffers, &preflight, cycles as u32)
        })
    }
}

/// The cycles of a parallel launch `[start, start + count)` in the order `par_stepExecW` runs
/// them (`WitgenParams::usePerm`), or `None` for `start + i`. Production always runs `start + i`;
/// tests can pick another order (`tests::launch_perm`, e.g. the R17 `sorted` amplifier).
#[cfg(test)]
fn test_perm(preflight: &PreflightTrace, start: u32, count: u32) -> Option<Vec<u32>> {
    tests::launch_perm(preflight, start, count)
}

#[cfg(not(test))]
fn test_perm(_: &PreflightTrace, _: u32, _: u32) -> Option<Vec<u32>> {
    None
}

#[cfg(test)]
#[path = "witgen_tests.rs"]
mod tests;
