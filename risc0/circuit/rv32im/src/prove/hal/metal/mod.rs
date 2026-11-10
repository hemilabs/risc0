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

//! Metal HAL for the rv32im circuit (docs/metal-port/PLAN.md 4.2, item R7 skeleton).
//!
//! Structure mirrors `hal/cuda.rs`: a [MetalCircuitHal] that keeps an `Arc` of the zkp
//! [MetalHal] it works through, and a [segment_prover] factory. The circuit-specific state lives
//! in [MetalRv32imCtx]: the rv32im metallib (`risc0_circuit_rv32im_sys::METAL_LIB`) with a PSO per
//! present kernel, and the HAL-owned scratch buffers (lookup tables, the preflight copy, perm,
//! scratch, poly_mix) that the Metal witgen/accum/eval_check paths bind.
//!
//! Until R8 (witgen), R9 (accum) and R10 (eval_check) land, every circuit phase runs the CPU
//! circuit kernels (`risc0_circuit_rv32im_cpu_*`) on host views of the Metal HAL's shared
//! buffers, after a `sync()`: the zkp side (NTT, hashing, FRI) runs on the GPU and the segment
//! proof is complete and verifiable. The phase submodules (`witgen`, `accum`, `eval_check`)
//! contain only that delegation.
//!
//! Kernel loading tolerates kernels that later items have not added yet: [EXPECTED_KERNELS] lists
//! the entry points each phase will need; the ones missing from the metallib are recorded, and
//! [MetalRv32imCtx::require] reports them only when that phase asks for its kernels.

mod accum;
mod eval_check;
mod witgen;

use std::{
    collections::HashSet,
    mem::size_of,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, MutexGuard, OnceLock,
    },
};

use anyhow::{anyhow, bail, Result};
use metal::ComputePipelineState;
use risc0_circuit_rv32im_sys::{
    RawMemoryTransaction, RawPreflightCycle, METAL_KERNEL_NAMES, METAL_LIB,
};
use risc0_zkp::hal::{
    metal::{BufferImpl, MetalHal, MetalHalPoseidon2, MetalHash, MetalHashPoseidon2, MetalKernels},
    Buffer as _,
};

use super::{PreflightTrace, SegmentProverImpl};
use crate::{prove::SegmentProver, zirgen::circuit::ExtVal};

/// Size of a preflight cycle as the kernels see it (`PreflightCycle`, rv32im-sys
/// `kernels/metal/preflight.h` static_asserts the same).
pub(crate) const PREFLIGHT_CYCLE_BYTES: usize = 36;
/// Size of a memory transaction as the kernels see it (`MemoryTransaction`).
pub(crate) const MEMORY_TXN_BYTES: usize = 20;
const _: () = assert!(size_of::<RawPreflightCycle>() == PREFLIGHT_CYCLE_BYTES);
const _: () = assert!(size_of::<RawMemoryTransaction>() == MEMORY_TXN_BYTES);

/// Lookup table sizes in cells (`LookupCell` = `uint32_t`), as CUDA `ffi.cu` allocates them.
pub(crate) const TABLE_U8_CELLS: usize = 1 << 8;
pub(crate) const TABLE_U16_CELLS: usize = 1 << 16;

/// Byte alignment of each section (cycles | txns | bigint) in the preflight buffer, so each can be
/// bound with a buffer offset.
const PF_ALIGN: usize = 256;

/// A circuit phase that runs Metal kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Phase {
    Witgen,
    Accum,
    EvalCheck,
}

/// The entry points each phase needs (CUDA `__global__` names kept 1:1, PLAN 2.4). Later items
/// edit these lists as they add kernels; a name may be listed before the metallib has it.
pub(crate) const EXPECTED_KERNELS: &[(Phase, &[&str])] = &[
    // R8 (witgen_tu.metal from witgen_tu_gen.py --metal, misc.metal).
    (
        Phase::Witgen,
        &[
            "injectCycleCols",
            "par_stepExecW",
            "fwd_stepExec",
            "rev_stepExec",
        ],
    ),
    // R9 (accum_tu.metal from accum_tu_gen.py --metal, misc.metal).
    (Phase::Accum, &["stepAccumA", "finalizeZeroizeAccum"]),
    // R10 (baseline eval_check.metal).
    (Phase::EvalCheck, &["eval_check"]),
];

/// The rv32im metallib with a PSO for every [METAL_KERNEL_NAMES] entry and every
/// [EXPECTED_KERNELS] entry the library contains. Built once per process (one Metal device),
/// like the zkp PSO cache.
pub(crate) struct Rv32imKernels {
    kernels: MetalKernels,
    /// Expected kernels the library does not have, by phase.
    missing: Vec<(Phase, &'static str)>,
}

impl Rv32imKernels {
    fn load_from<MH: MetalHash>(
        hal: &MetalHal<MH>,
        bytes: &[u8],
        listed: &[&str],
        expected: &[(Phase, &[&'static str])],
    ) -> Result<Self> {
        // First load without PSOs to see which entry points the library has.
        let available: HashSet<String> = hal
            .try_load_library(bytes, &[])?
            .library
            .function_names()
            .into_iter()
            .collect();
        let mut wanted: Vec<&str> = listed.to_vec();
        let mut missing = Vec::new();
        for &(phase, names) in expected {
            for &name in names {
                if !available.contains(name) {
                    missing.push((phase, name));
                } else if !wanted.contains(&name) {
                    wanted.push(name);
                }
            }
        }
        // Names listed by rv32im-sys must be present (MetalKernels::new fails fast otherwise).
        let kernels = hal.try_load_library(bytes, &wanted)?;
        Ok(Self { kernels, missing })
    }

    /// The process-wide instance, loaded through `hal` on first use.
    fn get<MH: MetalHash>(hal: &MetalHal<MH>) -> Result<&'static Self> {
        static KERNELS: OnceLock<std::result::Result<Rv32imKernels, String>> = OnceLock::new();
        KERNELS
            .get_or_init(|| {
                Self::load_from(hal, METAL_LIB, METAL_KERNEL_NAMES, EXPECTED_KERNELS)
                    .map_err(|e| format!("{e:#}"))
            })
            .as_ref()
            .map_err(|e| anyhow!("loading the rv32im metallib: {e}"))
    }

    /// Fails, naming the missing kernels, if `phase` cannot run on Metal with this metallib.
    pub(crate) fn require(&self, phase: Phase) -> Result<()> {
        let missing: Vec<&str> = self
            .missing
            .iter()
            .filter(|(p, _)| *p == phase)
            .map(|(_, name)| *name)
            .collect();
        if !missing.is_empty() {
            bail!(
                "rv32im Metal {phase:?}: the metallib is missing kernel(s) {} \
                 (risc0-circuit-rv32im-sys METAL_LIB; see EXPECTED_KERNELS)",
                missing.join(", ")
            );
        }
        Ok(())
    }
}

/// Offsets (bytes) of the sections of the preflight copy in [ScratchBuffers::pf].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PreflightLayout {
    pub cycles_offset: usize,
    pub cycles_len: usize,
    pub txns_offset: usize,
    pub txns_len: usize,
    pub bigint_offset: usize,
    pub bigint_len: usize,
    pub bytes: usize,
}

impl PreflightLayout {
    pub fn new(cycles: usize, txns: usize, bigint_bytes: usize) -> Self {
        let align = |n: usize| n.div_ceil(PF_ALIGN) * PF_ALIGN;
        let cycles_offset = 0;
        let txns_offset = align(cycles_offset + cycles * PREFLIGHT_CYCLE_BYTES);
        let bigint_offset = align(txns_offset + txns * MEMORY_TXN_BYTES);
        let bytes = align(bigint_offset + bigint_bytes).max(PF_ALIGN);
        Self {
            cycles_offset,
            cycles_len: cycles,
            txns_offset,
            txns_len: txns,
            bigint_offset,
            bigint_len: bigint_bytes,
            bytes,
        }
    }
}

/// HAL-owned scratch (PLAN 2.6), grown with 25 % headroom and reused across segments.
/// `perm`, `scratch` and `poly_mix` are used from R9/R10 on.
#[allow(dead_code)]
#[derive(Default)]
pub(crate) struct ScratchBuffers {
    pub table_u8: Option<BufferImpl<u32>>,
    pub table_u16: Option<BufferImpl<u32>>,
    /// cycles | txns | bigint_bytes of the current segment (kernels mutate `txnIdx`/`iopIdx`, so
    /// they work on this copy and the host trace stays pristine).
    pub pf: Option<BufferImpl<u8>>,
    pub pf_layout: PreflightLayout,
    /// Accum cycle permutation (R9).
    pub perm: Option<BufferImpl<u32>>,
    /// General Fp scratch (R9 scans, R12 evalfast slab).
    pub scratch: Option<BufferImpl<u32>>,
    /// poly_mix powers for eval_check (R10).
    pub poly_mix: Option<BufferImpl<ExtVal>>,
}

/// A buffer of at least `size` elements: `cur` if large enough, else a new one with 25 % headroom.
fn grow<T, MH: MetalHash>(
    hal: &MetalHal<MH>,
    cur: &mut Option<BufferImpl<T>>,
    name: &'static str,
    size: usize,
) -> BufferImpl<T>
where
    T: Clone,
{
    match cur {
        Some(buf) if buf.size() >= size => buf.clone(),
        _ => {
            let buf = hal.alloc_buffer::<T>(name, (size + size / 4).max(1));
            *cur = Some(buf.clone());
            buf
        }
    }
}

/// The rv32im circuit's Metal state (PLAN 4.2 host glue).
pub struct MetalRv32imCtx<MH: MetalHash> {
    hal: Arc<MetalHal<MH>>,
    kernels: &'static Rv32imKernels,
    /// A Mutex rather than a RefCell (PLAN 2.6) so the circuit HAL is Send + Sync like the HAL.
    scratch: Mutex<ScratchBuffers>,
}

impl<MH: MetalHash> MetalRv32imCtx<MH> {
    pub fn new(hal: Arc<MetalHal<MH>>) -> Result<Self> {
        Ok(Self {
            kernels: Rv32imKernels::get(&hal)?,
            hal,
            scratch: Mutex::new(ScratchBuffers::default()),
        })
    }

    /// The loaded rv32im metallib.
    #[allow(dead_code)]
    pub fn library(&self) -> &metal::LibraryRef {
        &self.kernels.kernels.library
    }

    /// Fails, naming the missing kernels, if `phase` cannot run on Metal with this metallib.
    #[allow(dead_code)]
    pub(crate) fn require(&self, phase: Phase) -> Result<()> {
        self.kernels.require(phase)
    }

    /// The PSO of an rv32im kernel; call [Self::require] for its phase first.
    #[allow(dead_code)]
    pub(crate) fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.kernels.kernels.pipeline(name)
    }

    /// The u8 and u16 lookup tables (CUDA `d_tableU8`/`d_tableU16`); contents unspecified, the
    /// phase zero-fills them.
    #[allow(dead_code)]
    pub(crate) fn tables(&self) -> (BufferImpl<u32>, BufferImpl<u32>) {
        let mut s = self.scratch.lock().unwrap();
        let u8 = grow(&self.hal, &mut s.table_u8, "table_u8", TABLE_U8_CELLS);
        let u16 = grow(&self.hal, &mut s.table_u16, "table_u16", TABLE_U16_CELLS);
        (u8, u16)
    }

    /// Copy cycles | txns | bigint_bytes of `trace` into the HAL-owned preflight buffer. The
    /// buffer is reused (grown with 25 % headroom), so the caller must not have GPU work using it
    /// in flight: this syncs first.
    #[allow(dead_code)]
    pub(crate) fn upload_preflight(
        &self,
        trace: &PreflightTrace,
    ) -> Result<(BufferImpl<u8>, PreflightLayout)> {
        let layout = PreflightLayout::new(
            trace.cycles.len(),
            trace.txns.len(),
            trace.bigint_bytes.len(),
        );
        self.hal.try_sync()?;
        let mut s = self.scratch.lock().unwrap();
        let buf = grow(&self.hal, &mut s.pf, "preflight", layout.bytes);
        s.pf_layout = layout;
        buf.view_mut(|dst| {
            let mut copy = |off: usize, src: &[u8]| dst[off..off + src.len()].copy_from_slice(src);
            // SAFETY: both are #[repr(C)] plain-old-data structs of u32/u8 fields (sizes asserted).
            let cycles = unsafe {
                std::slice::from_raw_parts(
                    trace.cycles.as_ptr() as *const u8,
                    trace.cycles.len() * PREFLIGHT_CYCLE_BYTES,
                )
            };
            let txns = unsafe {
                std::slice::from_raw_parts(
                    trace.txns.as_ptr() as *const u8,
                    trace.txns.len() * MEMORY_TXN_BYTES,
                )
            };
            copy(layout.cycles_offset, cycles);
            copy(layout.txns_offset, txns);
            copy(layout.bigint_offset, &trace.bigint_bytes);
        });
        Ok((buf, layout))
    }

    /// Scratch buffers, for phases that need more than the helpers above.
    #[allow(dead_code)]
    pub(crate) fn scratch(&self) -> MutexGuard<'_, ScratchBuffers> {
        self.scratch.lock().unwrap()
    }
}

/// Per-phase counts of circuit-phase calls served by the Metal circuit HAL (tests and logs).
pub(crate) static PHASE_CALLS: [AtomicUsize; 3] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];

fn count(phase: Phase) {
    PHASE_CALLS[phase as usize].fetch_add(1, Ordering::Relaxed);
}

pub struct MetalCircuitHal<MH: MetalHash> {
    hal: Arc<MetalHal<MH>>, // retain a reference: buffers may only be used through their own HAL
    ctx: MetalRv32imCtx<MH>,
}

impl<MH: MetalHash> MetalCircuitHal<MH> {
    pub fn new(hal: Arc<MetalHal<MH>>) -> Result<Self> {
        let ctx = MetalRv32imCtx::new(hal.clone())?;
        Ok(Self { hal, ctx })
    }

    #[allow(dead_code)]
    pub(crate) fn ctx(&self) -> &MetalRv32imCtx<MH> {
        &self.ctx
    }

    /// Wait for all GPU work of the HAL, so that host views of its buffers are current.
    fn sync(&self) -> Result<()> {
        self.hal.try_sync()
    }
}

/// A host slice over a Metal buffer. Only valid after [MetalCircuitHal::sync] and while no GPU
/// work using the buffer is encoded.
unsafe fn host_slice<T: Clone>(buf: &BufferImpl<T>) -> &[T] {
    std::slice::from_raw_parts(buf.as_ptr(), buf.size())
}

pub type MetalCircuitHalPoseidon2 = MetalCircuitHal<MetalHashPoseidon2>;

/// Build the zkp PSOs and the rv32im metallib PSOs now, so the first segment does not pay for it.
pub fn warmup() -> Result<()> {
    let hal = Arc::new(MetalHalPoseidon2::new());
    MetalCircuitHalPoseidon2::new(hal)?;
    Ok(())
}

pub fn segment_prover() -> Result<Box<dyn SegmentProver>> {
    // Fail here, not inside the prover, if the metallib cannot be loaded.
    Rv32imKernels::get(&MetalHalPoseidon2::new())?;
    tracing::debug!(
        "rv32im segment prover: Metal HAL ({}), circuit kernels on CPU until R8/R9/R10",
        risc0_zkp::hal::metal::device_name().unwrap_or_default()
    );
    let hal_factory = || {
        let hal = Arc::new(MetalHalPoseidon2::new());
        let circuit_hal = Arc::new(
            MetalCircuitHalPoseidon2::new(hal.clone()).expect("rv32im metallib loaded above"),
        );
        (hal, circuit_hal)
    };
    Ok(Box::new(SegmentProverImpl::new(hal_factory)))
}

#[cfg(test)]
mod tests;
