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
//! `pre_data`: the prover passes `data` itself unless `RISC0_USE_PRE_DATA` is set. Since R15w the
//! default binds `data` itself (single-buffer mode, CUDA's default `pre_data == data`), saving
//! `rows * 211 * 4` bytes (885 MB at po2 20). That is bit-exact because every row a cycle reads
//! back holds its injected value, which its own cycle writes again unchanged, so a back load reads
//! the same word before or after its writer ran (HIP's ffi.cu argues the same; notes/R15w.md has
//! the argument and the gate: 600 calls on 5 segments, sorted and production launch orders, 0
//! bad, with a same-session baseline that fails). `RISC0_METAL_WITGEN_SINGLE_BUFFER=0` restores
//! R8's snapshot: a copy of `data` taken after `injectCycleCols`, bound at index 1, so back loads
//! never read a row a neighbouring thread is writing. A separate `pre_data` from the prover
//! (`RISC0_USE_PRE_DATA`) already holds the scattered rows and cycle columns and is bound as is.
//!
//! Launch order and threadgroup size (R15w, [WitgenTune]): the first parallel launch (cycles
//! before the table split) runs its cycles grouped by major within 2^14-row windows
//! (`RISC0_METAL_WITGEN_SORT`, 0 = cycle order), HIP's order: a SIMD group then mostly runs one
//! arm of step_Top's mux. The permutation is built on the host from the pristine preflight cycles
//! (accum's counting sort, the same order HIP's `wmsSort` builds on the GPU) and bound as the
//! kernel's `perm`. `RISC0_METAL_WITGEN_TG` overrides the threadgroup size (default: the
//! runtime's `min(256, PSO maximum)`).
//!
//! **The GPU path is the default** (again since R17). `RISC0_METAL_RV32IM_CPU_PHASES` containing
//! `witgen` is the kill switch: witgen then runs the CPU kernel over host views of the shared
//! buffers (R7's delegation), with the CPU HAL's flags. `RISC0_METAL_RV32IM_GPU_PHASES=witgen`, the
//! opt-in of R8-R16, is still accepted and changes nothing.
//!
//! From R8 to R16 the GPU path was opt-in: the generated kernel gave wrong witnesses on identical
//! inputs from time to time (notes/R8.md, R11.md). R17 traced that to an Apple GPU defect: a
//! thread-private aggregate element written late (after an out-of-line call) can read back its
//! earlier value. `witgen_tu_gen.py` avoids every known trigger shape in the witgen TU
//! (docs/metal-port/notes/R17.md has the mechanism and the gates):
//! - R17 round 0: every map/reduce is unrolled into straight-line code and the call graph may not
//!   contain a loop (CUDA's lowering writes a default-constructed array in a loop that calls a
//!   noinline step function; `--scan-loops` checks the preprocessed TU);
//! - rounds 1-2: the TU compiles its own `witgen_fpext.h` (no element loops, every FpExt function
//!   always_inline: an out-of-line `operator*(FpExt lhs, Fp rhs) { lhs *= rhs; return lhs; }`
//!   lost elems[3] of PoseidonState.zcheck at -O0/-Oz; `--scan-ir` checks the compiled IR);
//! - R17b: its own `witgen_fp.h` (every Fp function always_inline), the FpExt helpers of
//!   witgen.h / ef_fpext.h always_inline, no default-constructed aggregate (statement-expression
//!   map lowering, mux results initialized in place), and internal linkage for everything but the
//!   three kernels, so the metallib linker cannot bind a call to another TU's (or a stale .air's)
//!   copy. `--scan-ir` also checks the linkage, `tools/metal_audit/ir_scan.py` finds no defect
//!   shape at -O2/-O0/-Oz/compiler-decided inlining, and `--scan-metallib` checks that the linked
//!   kernels contain only the TU's own functions (tests/metal_generators.rs).
//!
//! Re-verifying after a macOS or Metal toolchain update (R17.md, "Re-verifying"): run the
//! rv32im-sys generator tests, `witgen_sorted_amplifier`, and `witgen_segment_file` with
//! `RISC0_METAL_WITGEN_PERM=sorted` on the R17 segments, against same-session baselines that must
//! still fail (`--map-lowering loop`, and the R17 round-1 copy via `--fpext-calls outline` at -O0).

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

/// Parallel-witgen tuning (R15w, notes/R15w.md): the env knobs below, read once per process
/// ([tune]); tests override them per thread (`tests::with_tune`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WitgenTune {
    /// `par_stepExecW` threadgroup size (`RISC0_METAL_WITGEN_TG`); 0: the runtime's default
    /// (`min(256, PSO maximum)`, `Grid::Threads`). Otherwise capped by the PSO maximum and rounded
    /// down to a multiple of the SIMD width.
    pub tg: u64,
    /// The first launch (cycles before the table split) runs its cycles grouped by major within
    /// 2^sort_win-row windows (`RISC0_METAL_WITGEN_SORT`, HIP `RISC0_HIP_WITGEN_SORT`); 0: cycle
    /// order.
    pub sort_win: u32,
    /// Group by (min(major, 31), minor & 7) instead of the major (`RISC0_METAL_WITGEN_SORT_MINOR=1`,
    /// HIP `RISC0_HIP_WITGEN_MINOR`).
    pub minor: bool,
    /// Bind `data` itself as `pre_data` when the prover passes `data` (CUDA's default; no
    /// snapshot): `RISC0_METAL_WITGEN_SINGLE_BUFFER=1`, `0` takes the R8 snapshot.
    pub single: bool,
}

/// Defaults of [WitgenTune] (notes/R15w.md has the measurements behind each): the runtime's
/// threadgroup size (no other size measured faster), the first launch grouped by major in 2^14-row
/// windows (HIP's gfx11 default; kernels ~5 ms faster per po2 20 segment, 165 of 229 pairs), no
/// minor key (no gain measured), and single-buffer mode (no `rows * 211 * 4`-byte snapshot: −885 MB
/// at po2 20, ~7 ms less per call; bit-exact over 600 gate calls on 5 segments).
/// `RISC0_METAL_WITGEN_SORT=0 RISC0_METAL_WITGEN_SINGLE_BUFFER=0` restores the R8-R17 behaviour.
pub(crate) const DEFAULT_TUNE: WitgenTune = WitgenTune {
    tg: 0,
    sort_win: 14,
    minor: false,
    single: true,
};

/// Parse one knob: unset/empty -> `default`; unparsable -> `default` with an error log.
fn knob<T: std::str::FromStr + Copy>(name: &str, v: Option<&str>, default: T) -> T {
    match v.map(str::trim) {
        None | Some("") => default,
        Some(s) => s.parse().unwrap_or_else(|_| {
            tracing::error!("{name}={s:?} is not a valid value; using the default");
            default
        }),
    }
}

/// [WitgenTune] from the environment lookup `get` (`std::env::var` in production).
pub(crate) fn parse_tune(get: impl Fn(&str) -> Option<String>) -> WitgenTune {
    let flag = |name: &str, default: bool| -> bool {
        match get(name).as_deref().map(str::trim) {
            None | Some("") => default,
            Some("1") => true,
            Some("0") => false,
            Some(s) => {
                tracing::error!("{name}={s:?}: expected 0 or 1; using the default");
                default
            }
        }
    };
    WitgenTune {
        tg: knob(
            "RISC0_METAL_WITGEN_TG",
            get("RISC0_METAL_WITGEN_TG").as_deref(),
            DEFAULT_TUNE.tg,
        ),
        sort_win: knob(
            "RISC0_METAL_WITGEN_SORT",
            get("RISC0_METAL_WITGEN_SORT").as_deref(),
            DEFAULT_TUNE.sort_win,
        )
        .min(24),
        minor: flag("RISC0_METAL_WITGEN_SORT_MINOR", DEFAULT_TUNE.minor),
        single: flag("RISC0_METAL_WITGEN_SINGLE_BUFFER", DEFAULT_TUNE.single),
    }
}

/// The process's [WitgenTune] (tests: the thread's override, if any).
fn tune() -> WitgenTune {
    static TUNE: LazyLock<WitgenTune> = LazyLock::new(|| {
        let t = parse_tune(|k| std::env::var(k).ok());
        if t != DEFAULT_TUNE {
            tracing::info!("rv32im Metal witgen tuning: {t:?}");
        }
        t
    });
    test_tune().unwrap_or(*TUNE)
}

/// `par_stepExecW`'s threadgroup size for `want` (0: none, the runtime picks), as [WitgenTune::tg].
pub(crate) fn witgen_tg(pso: &metal::ComputePipelineStateRef, want: u64) -> Option<u64> {
    if want == 0 {
        return None;
    }
    let width = pso.thread_execution_width().max(1);
    let max = pso.max_total_threads_per_threadgroup();
    Some((want.min(max) / width * width).max(width.min(max)))
}

/// The major-grouped order of `[0, n)` (cycles before the table split), as HIP's `wmsSort` makes
/// it for par_stepExec: rows grouped by key within 2^win-row windows, ascending within a key.
pub(crate) fn witgen_sort_perm(
    cycles: &[risc0_circuit_rv32im_sys::RawPreflightCycle],
    win: u32,
    minor: bool,
) -> Vec<u32> {
    let mut perm = vec![0u32; cycles.len()];
    if minor {
        super::accum::build_perm_by(cycles, win, &mut perm, |c| {
            (c.major.min(31) << 3) | (c.minor & 7)
        });
    } else {
        super::accum::build_perm(cycles, win, &mut perm);
    }
    perm
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

/// What parallel witgen binds as `pre_data` when the prover passes `data` itself (tests force a
/// choice per call).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreData {
    /// [WitgenTune::single] decides (the production path).
    Auto,
    /// A snapshot of `data` taken after `injectCycleCols` (R8).
    Snapshot,
    /// Bind `data` itself (single-buffer mode, CUDA's default).
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
            self.metal_witgen(mode, preflight, global, data, pre_data, PreData::Auto)
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
        let tune = tune();
        let pre = match pre {
            PreData::Auto if tune.single => PreData::Single,
            PreData::Auto => PreData::Snapshot,
            p => p,
        };
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
        let snapshot = if mode == StepMode::Parallel && aliased && pre == PreData::Snapshot {
            // The HAL panics when an allocation exceeds the working-set budget
            // (RISC0_METAL_MAX_WORKING_SET_BYTES); report that as this phase's error.
            let size = data.buf.size();
            let snap = no_panic("the pre_data snapshot", || {
                hal.alloc_elem("witgen_pre_data", size)
            })
            .map_err(|e| e.context("RISC0_METAL_WITGEN_SINGLE_BUFFER=1 runs without it"))?;
            timing_mark("snapshot");
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
                let tg = witgen_tg(pso, tune.tg);
                let grid = |n: u32| match tg {
                    None => Grid::Threads(n as u64),
                    Some(t) => Grid::Groups {
                        groups: metal::MTLSize::new((n as u64).div_ceil(t), 1, 1),
                        threads: metal::MTLSize::new(t, 1, 1),
                    },
                };
                // HIP: the first launch's cycles grouped by major (R15w; [WitgenTune::sort_win]),
                // built on the host from the pristine preflight cycles before anything is encoded.
                // Same witness: each cycle still runs once and writes only its own row; the rows
                // it reads back hold their injected values (pre_data), or, single-buffer, the
                // same values their own cycles write again (notes/R15w.md).
                let mut sorted = (tune.sort_win > 0 && split > 1).then(|| {
                    scope!("witgen_sort");
                    witgen_sort_perm(
                        &preflight.cycles[..split as usize],
                        tune.sort_win,
                        tune.minor,
                    )
                });
                if timing_on() {
                    hal.try_sync()?;
                }
                timing_mark("encode");
                for (start, count) in [(0, split), (split, last_cycle - split)] {
                    // Tests only: a launch order other than start + i (`test_perm`).
                    let order = test_perm(preflight, start, count).or_else(|| {
                        if start == 0 {
                            sorted.take()
                        } else {
                            None
                        }
                    });
                    if let Some(p) = order {
                        let params = WitgenParams {
                            start,
                            count: p.len() as u32,
                            use_perm: 1,
                            ..base
                        };
                        let pb =
                            no_panic("the launch order", || hal.copy_from_u32("witgen_perm", &p))?;
                        encode(pso, &params, arg(bind(&pb)), grid(p.len() as u32))?;
                        perm_bufs.push(pb);
                        continue;
                    }
                    if count == 0 {
                        continue;
                    }
                    let params = WitgenParams {
                        start,
                        count,
                        use_perm: 0,
                        ..base
                    };
                    // perm (binding 10) is not read with use_perm = 0: bind the fault word.
                    encode(pso, &params, arg(bind(&fault)), grid(count))?;
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
        timing_mark("done");
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
fn test_tune() -> Option<WitgenTune> {
    tests::thread_tune()
}

#[cfg(not(test))]
fn test_tune() -> Option<WitgenTune> {
    None
}

/// Tests only (`tests::timing`): a timestamp per step of the call. With timing on, the GPU is
/// also waited for before the kernels are encoded, so "encode" -> "done" is their own time.
#[cfg(test)]
fn timing_mark(label: &'static str) {
    tests::timing_mark(label)
}

#[cfg(not(test))]
#[inline(always)]
fn timing_mark(_: &'static str) {}

#[cfg(test)]
fn timing_on() -> bool {
    tests::timing_on()
}

#[cfg(not(test))]
#[inline(always)]
fn timing_on() -> bool {
    false
}

#[cfg(test)]
#[path = "witgen_tests.rs"]
mod tests;
