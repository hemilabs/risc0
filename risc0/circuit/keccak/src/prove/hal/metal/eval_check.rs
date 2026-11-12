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

//! The keccak eval_check on Metal (item K4, docs/metal-port/PLAN.md 4.4): the runtime of the
//! staged kernels `eval_stage_0..S-1` that keccak-sys/build_metal_eval.rs (K3) translates from the
//! CUDA constraint code.
//!
//! Port of the Intel launcher (`keccak-sys/kernels/intel/kernels.h`,
//! `risc0_circuit_keccak_intel_eval_check` and `launch_eval_stage`), with the inputs of the CUDA
//! entry point (`hal/cuda.rs` `eval_check`):
//! - the cycles of the domain go through all `S` stages a chunk of `C` cycles at a time (Intel's
//!   `kScratchStride`): each stage is one dispatch of `count <= C` threads, thread `t` running cycle
//!   `first + t`; the next chunk reuses the same scratch and carry (the batch's serial compute
//!   encoder orders the dispatches);
//! - scratch (`EVAL_SCRATCH_ELEMS` Fp a thread, poly_fp's `Fp x13[8237]`) and carry
//!   (`EVAL_CARRY_WORDS` words a thread, the FpExt arguments between stages) are
//!   `StorageModePrivate` buffers of stride `C`, allocated once per eval_check (every chunk and
//!   stage of the call reuses them) and freed when it returns, so that a later, larger proof on the
//!   same prover has the memory for its commit phase under a tight working-set budget (K4 review).
//!   [MetalCircuitHal::set_keep_eval_scratch] keeps them cached across calls instead (tests, K7);
//!   the zkvm builds a new prover per keccak proof, so the cache would not survive a proof anyway;
//! - the last stage multiplies by the vanishing quotient, which depends only on `cycle & 3`: the
//!   host computes the four values (as Intel's launcher does) and passes them in `EvalParams` as raw
//!   Montgomery words;
//! - the polynomial mix powers (`map_pow(poly_mix, POLY_MIX_POWERS)`, as `hal/cuda.rs`) are a bound
//!   `const device FpExt*` buffer (34.8 KB, above `set_bytes`' 4 KB).
//!
//! K7 (docs/metal-port/notes/K7.md) knobs, each with the previous behaviour one setting away:
//! - `RISC0_KECCAK_METAL_EVAL_MACC=lazy|strict`: the stage kernels with the lazy
//!   multiply-accumulate (default; ~44% less eval_check time at po2 17, bit-identical) or K4's
//!   (`eval_strict_stage_<k>`, built on first use);
//! - `RISC0_KECCAK_METAL_EVAL_CHUNK` (default now 16384, K4's 32768).
//!
//! A tiled scratch layout and a runtime threadgroup size were measured and showed no change, so
//! they are not kept (K7.md).
//!
//! The kernels index with 32-bit math, as CUDA: the launcher asserts
//! `(EVAL_MAX_COLUMN + 1) * domain <= 2^32`, `stride <= EVAL_MAX_STRIDE` and `count <= stride`.
//! Bindings: `kernels/metal/eval_check_prelude.h` (mirrored by `EVAL_BUF_*`). Each stage binds only
//! the buffers its code may touch (`EVAL_STAGE_BINDINGS`, from build_metal_eval.rs), the rest are
//! left nil, so the Metal validation layer reports no unused bindings (no stage reads ctrl, mix or
//! accum today).

use std::{
    cell::{Cell, RefCell},
    time::{Duration, Instant},
};

use metal::Buffer as MtlBuffer;
use risc0_circuit_keccak_sys::metal_eval::{
    EVAL_BUF_ACCUM, EVAL_BUF_CARRY, EVAL_BUF_CHECK, EVAL_BUF_CTRL, EVAL_BUF_DATA, EVAL_BUF_MIX,
    EVAL_BUF_OUT, EVAL_BUF_PARAMS, EVAL_BUF_POLY_MIX, EVAL_BUF_SCRATCH, EVAL_CARRY_WORDS,
    EVAL_INV_RATE, EVAL_MAX_COLUMN, EVAL_MAX_STRIDE, EVAL_NUM_POLY_MIX_POWS, EVAL_PARAMS_SIZE,
    EVAL_SCRATCH_ELEMS, EVAL_STAGES, EVAL_STAGE_BINDINGS, EVAL_STAGE_NAMES,
    EVAL_STRICT_STAGE_NAMES,
};
use risc0_core::field::RootsOfUnity as _;
use risc0_zkp::{
    adapter::CircuitInfo as _,
    core::log2_ceil,
    field::{map_pow, Elem as _, ExtElem as _},
    hal::{
        metal::{as_bytes, BufferImpl as MetalBuffer, Grid, KernelArg, MetalHash},
        Buffer as _, Hal as _,
    },
    INV_RATE,
};

use super::MetalCircuitHal;
use crate::{
    prove::{CircuitImpl, GLOBAL_MIX, GLOBAL_OUT},
    zirgen::{
        circuit::{ExtVal, Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CODE, REGISTER_GROUP_DATA},
        info::POLY_MIX_POWERS,
        taps::TAPSET,
    },
};

/// Default cycles per chunk (the scratch/carry stride): 16384 cycles take 541 MB of scratch.
/// Why chunk at all: the scratch is ~33 KB per cycle, so a whole po2 18 domain (2^20 cycles)
/// would need ~34 GB; a fixed chunk keeps it constant from po2 12 up. Why 16384 (K7; K4 had
/// 32768): interleaved A/B on the M5 Max (40 cores) found 16384 as fast as 32768 and 65536 at
/// po2 17 (lazy kernels: medians 671 vs 706-719 ms, minima 647 vs 617-621 ms, within the A/A
/// spread) and faster at po2 14 (102 vs 132 ms: half the scratch to allocate and free per call),
/// with half the peak memory; 8192 was 30% slower (too few threads per dispatch to hide latency
/// on 40 cores). 16384 threads still fill a 10-core GPU many times over.
/// `RISC0_KECCAK_METAL_EVAL_CHUNK=32768` restores K4's value.
pub const DEFAULT_EVAL_CHUNK: u32 = 16384;

/// Overrides [DEFAULT_EVAL_CHUNK] (cycles per chunk; for the K7 sweep).
pub const EVAL_CHUNK_ENV: &str = "RISC0_KECCAK_METAL_EVAL_CHUNK";

/// The smallest chunk the launcher shrinks to when the scratch does not fit the working-set
/// budget (`MetalHal::working_set_limit`). 1024 cycles is ~34 MB of scratch; smaller chunks
/// would save little memory while the dispatch count (domain / chunk × stages: 59k at po2 18)
/// would dominate the time.
pub const MIN_EVAL_CHUNK: u32 = 1024;

/// Bytes of scratch plus carry for a stride of `stride` cycles.
pub(super) fn scratch_bytes(stride: u32) -> u64 {
    (EVAL_SCRATCH_ELEMS as u64 + EVAL_CARRY_WORDS as u64) * 4 * stride as u64
}

/// The chunk eval_check runs with when `want` cycles of scratch would take the device's allocation
/// (`allocated` bytes) above `limit`: `want` halved until it fits, but not below [MIN_EVAL_CHUNK]
/// (`want` itself if it fits or is already below that).
pub(super) fn fit_chunk(want: u32, allocated: u64, limit: u64) -> u32 {
    let fits = |c: u32| allocated + scratch_bytes(c) <= limit;
    let mut chunk = want;
    while !fits(chunk) && chunk / 2 >= MIN_EVAL_CHUNK {
        chunk /= 2;
    }
    chunk
}

/// Selects the stage kernels: `lazy` (default; K7's lazy multiply-accumulate, ~40% faster) or
/// `strict` (K3/K4's kernels, `EVAL_STRICT_STAGE_NAMES`, built on first use). Bit-identical.
pub const EVAL_MACC_ENV: &str = "RISC0_KECCAK_METAL_EVAL_MACC";

/// Whether [EVAL_MACC_ENV] selects the strict kernels.
fn strict_from_env() -> bool {
    match std::env::var(EVAL_MACC_ENV).as_deref() {
        Ok("strict") => true,
        Ok("lazy") | Ok("") | Err(_) => false,
        Ok(v) => {
            tracing::error!("{EVAL_MACC_ENV}={v:?} is not lazy or strict; using lazy");
            false
        }
    }
}

/// `EvalParams` of eval_check_prelude.h: count, first, domain, stride, quot[4] (raw words).
type EvalParams = [u32; 8];
const _: () = assert!(std::mem::size_of::<EvalParams>() == EVAL_PARAMS_SIZE);

/// The cached private buffers of the stage kernels.
struct EvalScratch {
    /// Cycles the buffers hold (the kernels' `stride`).
    stride: u32,
    scratch: MtlBuffer,
    carry: MtlBuffer,
}

/// The circuit HAL's eval_check state.
pub(super) struct EvalCheckState {
    /// Cycles per chunk ([DEFAULT_EVAL_CHUNK], [EVAL_CHUNK_ENV], [MetalCircuitHal::set_eval_chunk]).
    chunk: Cell<u32>,
    /// Keep the scratch cached after a call ([MetalCircuitHal::set_keep_eval_scratch]).
    keep: Cell<bool>,
    scratch: RefCell<Option<EvalScratch>>,
    /// Run the strict stage kernels ([EVAL_MACC_ENV], [MetalCircuitHal::set_eval_strict]).
    strict: Cell<bool>,
    /// Measurements only (K7 `k7_eval_ab`): stage PSOs from another metallib, used in place of
    /// the process-wide ones, so variants can be interleaved in one process.
    #[cfg(test)]
    stage_override: RefCell<Option<Vec<(metal::ComputePipelineState, u32)>>>,
}

impl EvalCheckState {
    pub(super) fn new() -> Self {
        let chunk = match std::env::var(EVAL_CHUNK_ENV) {
            Ok(text) => match text.trim().parse::<u32>() {
                Ok(c) if c > 0 => c,
                _ => {
                    tracing::error!("{EVAL_CHUNK_ENV}={text:?} is not a positive cycle count; using {DEFAULT_EVAL_CHUNK}");
                    DEFAULT_EVAL_CHUNK
                }
            },
            Err(_) => DEFAULT_EVAL_CHUNK,
        };
        Self {
            chunk: Cell::new(chunk.min(EVAL_MAX_STRIDE)),
            keep: Cell::new(false),
            scratch: RefCell::new(None),
            strict: Cell::new(strict_from_env()),
            #[cfg(test)]
            stage_override: RefCell::new(None),
        }
    }
}

/// The four values of the vanishing quotient `1 / ((3 * rou^cycle)^(2^po2) - 1)`, by `cycle & 3`,
/// as raw Montgomery words: `rou` (`ROU_FWD[po2 + 2]`) has order `2^(po2 + 2)`, so the power only
/// depends on `cycle mod 4` (Intel launcher; the CPU HAL computes it per cycle).
pub(super) fn vanishing_quotients(po2: usize) -> [u32; 4] {
    const EXP_PO2: usize = log2_ceil(INV_RATE);
    let rou = Val::ROU_FWD[po2 + EXP_PO2];
    assert_eq!(
        rou.pow(4 << po2),
        Val::ONE,
        "keccak eval_check: rou's order does not divide 2^(po2 + 2)"
    );
    std::array::from_fn(|k| {
        // As CpuCircuitHal: 3 is the coset shift of the evaluation domain.
        let y = (Val::new(3) * rou.pow(k)).pow(1 << po2);
        (y - Val::ONE).inv().as_u32_montgomery()
    })
}

impl<MH: MetalHash> MetalCircuitHal<MH> {
    /// Cycles per eval_check chunk (the scratch stride) for later calls; a smaller chunk takes
    /// less memory (33 KB a cycle) and more dispatches. Panics if 0 or above `EVAL_MAX_STRIDE`.
    // Tuning/memory knobs for tests, K7 and the miner; not called by the prover itself.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn set_eval_chunk(&self, cycles: u32) {
        assert!(
            cycles > 0 && cycles <= EVAL_MAX_STRIDE,
            "eval_check chunk {cycles} not in 1..={EVAL_MAX_STRIDE}"
        );
        self.eval.chunk.set(cycles);
    }

    /// Whether later calls run the strict stage kernels (K4's code) instead of the lazy ones.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn set_eval_strict(&self, strict: bool) {
        self.eval.strict.set(strict);
    }

    /// Cycles per eval_check chunk.
    // Tuning/memory knobs for tests, K7 and the miner; not called by the prover itself.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn eval_chunk(&self) -> u32 {
        self.eval.chunk.get()
    }

    /// Whether eval_check keeps its scratch and carry (about 0.5 GB at the default chunk) cached for
    /// the next call instead of freeing them on return (the default). Only for callers that run
    /// many eval_checks of one size on one HAL with memory to spare: the cached scratch counts
    /// against the working-set budget of everything that runs between the calls.
    // Tuning/memory knobs for tests, K7 and the miner; not called by the prover itself.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn set_keep_eval_scratch(&self, keep: bool) {
        self.eval.keep.set(keep);
        if !keep {
            self.release_eval_scratch();
        }
    }

    /// Free the cached eval_check scratch and carry (about 0.5 GB at the default chunk); the next
    /// eval_check allocates them again.
    // Tuning/memory knobs for tests, K7 and the miner; not called by the prover itself.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn release_eval_scratch(&self) {
        self.hal.sync();
        self.eval.scratch.borrow_mut().take();
    }

    /// Bytes of the cached eval_check scratch and carry (0 when none).
    // Tuning/memory knobs for tests, K7 and the miner; not called by the prover itself.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn eval_scratch_bytes(&self) -> u64 {
        self.eval
            .scratch
            .borrow()
            .as_ref()
            .map_or(0, |s| scratch_bytes(s.stride))
    }

    /// The private scratch/carry for chunks of `want` cycles (`want <= domain`): the cached
    /// buffers if they are large enough, else new ones. The chunk is capped so that the scratch
    /// fits the device's `maxBufferLength`. If the scratch would not fit the HAL's
    /// working-set limit, the chunk halves (down to [MIN_EVAL_CHUNK]) so that small-memory Macs
    /// and `RISC0_METAL_MAX_WORKING_SET_BYTES` budgets still run, with more dispatches; if even
    /// that does not fit, the allocation panics with the HAL's budget message.
    /// Returns (scratch, carry, stride, cycles per chunk).
    fn eval_scratch(&self, want: u32) -> (MtlBuffer, MtlBuffer, u32, u32) {
        let mut cached = self.eval.scratch.borrow_mut();
        if let Some(s) = cached.as_ref() {
            if s.stride >= want {
                return (s.scratch.clone(), s.carry.clone(), s.stride, want);
            }
        }
        // Too small: free it first (no batch uses it: every eval_check ends with a sync).
        cached.take();
        let max_chunk = self.hal.device.max_buffer_length()
            / (EVAL_SCRATCH_ELEMS.max(EVAL_CARRY_WORDS) as u64 * 4);
        assert!(
            max_chunk >= 1,
            "keccak eval_check: maxBufferLength below one cycle"
        );
        let want = want.min(max_chunk.min(u32::MAX as u64) as u32);
        let limit = self.hal.working_set_limit();
        let mut chunk = want;
        if self.hal.allocated_size() + scratch_bytes(want) > limit {
            self.hal.trim_device_memory();
            chunk = fit_chunk(want, self.hal.allocated_size(), limit);
            if chunk != want {
                tracing::warn!(
                    "keccak eval_check: chunk {want} cycles ({} MiB of scratch) does not fit the \
                     working-set limit of {} MiB ({} MiB allocated); using {chunk}",
                    scratch_bytes(want) >> 20,
                    limit >> 20,
                    self.hal.allocated_size() >> 20
                );
            }
        }
        let scratch = self
            .hal
            .alloc_private(EVAL_SCRATCH_ELEMS as u64 * 4 * chunk as u64);
        let carry = self
            .hal
            .alloc_private(EVAL_CARRY_WORDS as u64 * 4 * chunk as u64);
        *cached = Some(EvalScratch {
            stride: chunk,
            scratch: scratch.clone(),
            carry: carry.clone(),
        });
        (scratch, carry, chunk, chunk)
    }

    /// eval_check on the GPU: see the module doc. With `profile`, syncs after every dispatch and
    /// adds each stage's wall time (encode + GPU) to `profile[stage]` (for measurements only).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn eval_check_metal(
        &self,
        check: &MetalBuffer<Val>,
        groups: &[&MetalBuffer<Val>],
        globals: &[&MetalBuffer<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
        mut profile: Option<&mut Vec<Duration>>,
    ) {
        // Constant per build: false only for the RISC0_SKIP_BUILD_KERNELS stub.
        #[allow(clippy::assertions_on_constants, clippy::absurd_extreme_comparisons)]
        let has_stages = EVAL_STAGES > 0;
        assert!(
            has_stages,
            "the keccak metallib has no eval_check stages (built with RISC0_SKIP_BUILD_KERNELS?)"
        );
        assert_eq!(EVAL_INV_RATE as usize, INV_RATE);
        let ctrl = groups[REGISTER_GROUP_CODE];
        let data = groups[REGISTER_GROUP_DATA];
        let accum = groups[REGISTER_GROUP_ACCUM];
        let mix = globals[GLOBAL_MIX];
        let out = globals[GLOBAL_OUT];
        let domain = steps * INV_RATE;
        assert_eq!(steps, 1 << po2);
        tracing::debug!(
            "eval_check: po2 {po2}, domain {domain}, check {}, ctrl {}, data {}, accum {}, mix {}, out {}",
            check.size(),
            ctrl.size(),
            data.size(),
            accum.size(),
            mix.size(),
            out.size()
        );

        // The kernels read without bounds checks: the buffers must be the circuit's shapes.
        assert_eq!(check.size(), ExtVal::EXT_SIZE * domain, "check size");
        for (name, buf, group) in [
            ("ctrl", ctrl, REGISTER_GROUP_CODE),
            ("data", data, REGISTER_GROUP_DATA),
            ("accum", accum, REGISTER_GROUP_ACCUM),
        ] {
            assert!(
                buf.size() >= TAPSET.group_size(group) * domain,
                "eval_check: {name} has {} elements, the circuit needs {} x {domain}",
                buf.size(),
                TAPSET.group_size(group)
            );
        }
        assert!(mix.size() >= CircuitImpl::MIX_SIZE, "eval_check: mix size");
        assert!(
            out.size() >= CircuitImpl::OUTPUT_SIZE,
            "eval_check: out size"
        );
        // 32-bit indexing (K3): columns as `buf[c * domain + cycle]`, check as
        // `check[domain * i + cycle]` (i < 4 <= EVAL_MAX_COLUMN).
        assert!(
            (EVAL_MAX_COLUMN as u64 + 1) * domain as u64 <= 1 << 32,
            "keccak eval_check: domain {domain} too large for 32-bit indexing \
             ((EVAL_MAX_COLUMN + 1) * domain > 2^32)"
        );
        let domain = domain as u32;

        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);
        assert_eq!(poly_mix_pows.len(), EVAL_NUM_POLY_MIX_POWS as usize);
        let poly_mix_buf = self
            .hal
            .copy_from_slice::<ExtVal>("eval_poly_mix", &poly_mix_pows);
        let quot = vanishing_quotients(po2);

        let want = self.eval.chunk.get().min(domain);
        let (scratch, carry, stride, chunk) = self.eval_scratch(want);
        assert!(stride <= EVAL_MAX_STRIDE && chunk <= stride);
        assert!(EVAL_SCRATCH_ELEMS as u64 * stride as u64 <= 1 << 32);
        assert!(EVAL_CARRY_WORDS as u64 * stride as u64 <= 1 << 32);

        #[cfg(test)]
        let overridden = self.eval.stage_override.borrow().clone();
        #[cfg(not(test))]
        let overridden: Option<Vec<(metal::ComputePipelineState, u32)>> = None;
        // (PSO, bindings mask) of each stage, in order.
        let pipelines: Vec<(&metal::ComputePipelineStateRef, u32)> = match &overridden {
            Some(v) => v.iter().map(|(p, m)| (&**p, *m)).collect(),
            None if self.eval.strict.get() => {
                let strict = super::keccak_strict_kernels(&self.hal.device);
                EVAL_STRICT_STAGE_NAMES
                    .iter()
                    .zip(EVAL_STAGE_BINDINGS)
                    .map(|(name, mask)| (&**strict.pipeline(name), mask))
                    .collect()
            }
            None => EVAL_STAGE_NAMES
                .iter()
                .zip(EVAL_STAGE_BINDINGS)
                .map(|(name, mask)| (&**self.pipeline(name), mask))
                .collect(),
        };
        if let Some(p) = profile.as_mut() {
            p.resize(pipelines.len(), Duration::ZERO);
            self.hal.sync();
        }
        let mut first = 0u32;
        while first < domain {
            let count = chunk.min(domain - first);
            assert!(count <= stride);
            let params: EvalParams = [
                count, first, domain, stride, quot[0], quot[1], quot[2], quot[3],
            ];
            let mut all = [const { KernelArg::Null }; 10];
            all[EVAL_BUF_CHECK as usize] = check.as_arg();
            all[EVAL_BUF_CTRL as usize] = ctrl.as_arg();
            all[EVAL_BUF_OUT as usize] = out.as_arg();
            all[EVAL_BUF_DATA as usize] = data.as_arg();
            all[EVAL_BUF_MIX as usize] = mix.as_arg();
            all[EVAL_BUF_ACCUM as usize] = accum.as_arg();
            all[EVAL_BUF_POLY_MIX as usize] = poly_mix_buf.as_arg();
            all[EVAL_BUF_SCRATCH as usize] = KernelArg::Buffer {
                buffer: &scratch,
                offset: 0,
            };
            all[EVAL_BUF_CARRY as usize] = KernelArg::Buffer {
                buffer: &carry,
                offset: 0,
            };
            all[EVAL_BUF_PARAMS as usize] = KernelArg::Bytes(as_bytes(&params[..]));
            for (stage, &(pso, mask)) in pipelines.iter().enumerate() {
                // Only the stage's own bindings; the others stay nil (unread).
                let args: [KernelArg; 10] = std::array::from_fn(|i| {
                    if mask & (1 << i) == 0 {
                        return KernelArg::Null;
                    }
                    match &all[i] {
                        KernelArg::Null => KernelArg::Null,
                        KernelArg::Buffer { buffer, offset } => KernelArg::Buffer {
                            buffer,
                            offset: *offset,
                        },
                        KernelArg::Integer(v) => KernelArg::Integer(*v),
                        KernelArg::Bytes(b) => KernelArg::Bytes(b),
                    }
                });
                let t0 = Instant::now();
                self.hal
                    .dispatch(pso, &args, Grid::Threads(count as u64), &[]);
                if let Some(p) = profile.as_mut() {
                    self.hal.sync();
                    p[stage] += t0.elapsed();
                }
            }
            first += count;
        }
        // End of the phase (plan 2.6): wait, so GPU faults surface here, and so the cached
        // scratch is idle whenever eval_check is not running.
        if let Err(err) = self.hal.try_sync() {
            panic!("keccak eval_check on Metal failed: {err:#}");
        }
        // Free the scratch (the GPU is done with it) unless asked to keep it: see the module doc.
        if !self.eval.keep.get() {
            self.eval.scratch.borrow_mut().take();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        rc::Rc,
        time::{Duration, Instant},
    };

    use risc0_circuit_keccak_sys::{
        metal_eval::{EVAL_STAGES, EVAL_STAGE_NAMES},
        METAL_LIB,
    };
    use risc0_core::field::{baby_bear::BabyBear, Elem as _, RootsOfUnity as _};
    use risc0_zkp::{
        core::hash::sha::Sha256HashSuite,
        hal::{cpu::CpuHal, metal::MetalHalSha256, Buffer as _, Hal},
    };
    use test_log::test;

    use super::{
        fit_chunk, scratch_bytes, vanishing_quotients, MetalCircuitHal, DEFAULT_EVAL_CHUNK,
        MIN_EVAL_CHUNK,
    };
    use crate::{
        prove::{
            hal::cpu::CpuCircuitHal,
            testutil::{eval_check_impl, EvalCheckParams},
        },
        zirgen::circuit::Val,
    };

    /// The CPU HAL's check polynomial for `params` (the oracle).
    fn cpu_check(params: &EvalCheckParams) -> Vec<Val> {
        let cpu_hal: CpuHal<BabyBear> = CpuHal::new(Sha256HashSuite::new_suite());
        eval_check_impl(params, &cpu_hal, &CpuCircuitHal)
    }

    fn assert_same(what: &str, actual: &[Val], expected: &[Val]) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        let diffs: Vec<usize> = (0..actual.len())
            .filter(|&i| actual[i] != expected[i])
            .collect();
        assert!(
            diffs.is_empty(),
            "{what}: {} of {} check cells differ from the CPU (first at {:?}; stages {EVAL_STAGES})",
            diffs.len(),
            actual.len(),
            &diffs[..diffs.len().min(8)]
        );
    }

    /// The Metal check polynomial for `params`, with chunks of `chunk` cycles.
    fn metal_check(
        hal: &Rc<MetalHalSha256>,
        circuit_hal: &MetalCircuitHal<risc0_zkp::hal::metal::MetalHashSha256>,
        params: &EvalCheckParams,
        chunk: u32,
    ) -> Vec<Val> {
        circuit_hal.set_eval_chunk(chunk);
        eval_check_impl(params, hal.as_ref(), circuit_hal)
    }

    /// The four host quotients equal the CPU HAL's per-cycle `1 / ((3 x)^(2^po2) - 1)`.
    #[test]
    fn vanishing_quotients_match_cpu() {
        for po2 in [4, 8, 14, 17] {
            let quot = vanishing_quotients(po2);
            let rou = Val::ROU_FWD[po2 + 2];
            for cycle in [0usize, 1, 2, 3, 4, 5, 6, 7, 1001, (4 << po2) - 1] {
                let y = (Val::new(3) * rou.pow(cycle)).pow(1 << po2);
                let want = (y - Val::new(1)).inv();
                assert_eq!(
                    Val::new_raw(quot[cycle & 3]),
                    want,
                    "po2 {po2} cycle {cycle}"
                );
            }
        }
    }

    /// Bit-exact against the CPU HAL at `po2s`, at the default chunk (one chunk up to po2 12,
    /// 2^(po2 - 12) chunks above).
    fn matches_cpu(po2s: &[usize]) {
        let _gpu = super::super::tests::gpu_test_lock();
        let hal = Rc::new(MetalHalSha256::new());
        let circuit_hal = MetalCircuitHal::new(hal.clone());
        for &po2 in po2s {
            let params = EvalCheckParams::new(po2);
            let expected = cpu_check(&params);
            let t0 = Instant::now();
            let actual = metal_check(&hal, &circuit_hal, &params, DEFAULT_EVAL_CHUNK);
            println!(
                "po2 {po2}: {EVAL_STAGES} stages, domain {}, chunk {DEFAULT_EVAL_CHUNK}: Metal eval_check {:.3} s (incl. upload/readback)",
                params.domain,
                t0.elapsed().as_secs_f64()
            );
            assert_same(&format!("po2 {po2}"), &actual, &expected);
        }
    }

    #[test]
    fn eval_check_matches_cpu() {
        matches_cpu(&[4, 8, 14]);
    }

    /// K7: the strict stage kernels (no lazy multiply-accumulate, `RISC0_KECCAK_METAL_EVAL_MACC=
    /// strict`) and the lazy ones (default) give the CPU's check, and so each other's, bit for bit.
    #[test]
    fn eval_check_strict_and_lazy_match_cpu() {
        let _gpu = super::super::tests::gpu_test_lock();
        let hal = Rc::new(MetalHalSha256::new());
        let circuit_hal = MetalCircuitHal::new(hal.clone());
        for po2 in [8, 14] {
            let params = EvalCheckParams::new(po2);
            let expected = cpu_check(&params);
            for strict in [true, false] {
                circuit_hal.set_eval_strict(strict);
                let actual = metal_check(&hal, &circuit_hal, &params, 3000);
                assert_same(&format!("po2 {po2} strict {strict}"), &actual, &expected);
            }
        }
    }

    /// PO2 17 (domain 2^19: 32 chunks of 16384 at the default chunk); ~10 GB of host memory.
    #[test]
    #[ignore]
    fn eval_check_matches_cpu_po2_17() {
        matches_cpu(&[17]);
    }

    /// R17 audit: `R17_AUDIT_REPS` (default 10) Metal eval_checks of one `R17_AUDIT_PO2` (default
    /// 17) input, each compared bit for bit with the first and with the CPU HAL. Prints
    /// `R17AUDIT keccak eval_check ...: k/n` (all 58 stages run in every rep).
    #[test]
    #[ignore]
    fn r17_eval_check_repeat() {
        let _gpu = super::super::tests::gpu_test_lock();
        let reps: usize = std::env::var("R17_AUDIT_REPS").map_or(10, |v| v.parse().unwrap());
        let po2: usize = std::env::var("R17_AUDIT_PO2").map_or(17, |v| v.parse().unwrap());
        let hal = Rc::new(MetalHalSha256::new());
        let circuit_hal = MetalCircuitHal::new(hal.clone());
        let params = EvalCheckParams::new(po2);
        let expected = cpu_check(&params);
        let (mut vs_first, mut vs_cpu) = (0, 0);
        let mut first: Option<Vec<Val>> = None;
        let mut detail = vec![];
        let t0 = Instant::now();
        for rep in 0..reps {
            let actual = metal_check(&hal, &circuit_hal, &params, DEFAULT_EVAL_CHUNK);
            let n = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            if n > 0 {
                vs_cpu += 1;
                detail.push(format!("rep {rep}: {n} vs CPU"));
            }
            match &first {
                None => first = Some(actual),
                Some(f) => {
                    let n = actual.iter().zip(f).filter(|(a, b)| a != b).count();
                    if n > 0 {
                        vs_first += 1;
                        detail.push(format!("rep {rep}: {n} vs rep 0"));
                    }
                }
            }
        }
        println!(
            "R17AUDIT keccak eval_check po2 {po2} ({EVAL_STAGES} stages): {vs_first}/{reps} reps \
             differ from rep 0, {vs_cpu}/{reps} differ from CPU ({:.2} s/rep) {detail:?}",
            t0.elapsed().as_secs_f64() / reps as f64
        );
        assert_eq!(vs_first + vs_cpu, 0);
    }

    /// K7 interleaved A/B of runtime eval_check configurations at `K7_PO2` (default 17):
    /// `K7_CONFIGS` = `label:chunk=N,strict=0|1,lib=N;...` (unset keys keep the defaults), `K7_REPS`
    /// rounds (default 7) of one call per config in turn (A B C A B C ...), each call as the prover
    /// makes it (scratch allocated and freed, one sync). Prints median/min/max per config and
    /// asserts every output equals the first config's (and the CPU's with `K7_CPU=1`).
    #[test]
    #[ignore]
    fn k7_eval_ab() {
        let _gpu = super::super::tests::gpu_test_lock();
        let po2: usize = std::env::var("K7_PO2").map_or(17, |v| v.parse().unwrap());
        let reps: usize = std::env::var("K7_REPS").map_or(7, |v| v.parse().unwrap());
        let configs: Vec<(String, Vec<(String, u32)>)> = std::env::var("K7_CONFIGS")
            .unwrap_or_else(|_| "lazy:;strict:strict=1".into())
            .split(';')
            .filter(|c| !c.trim().is_empty())
            .map(|c| {
                let (label, kv) = c.split_once(':').unwrap_or((c, ""));
                let kv = kv
                    .split(',')
                    .filter(|x| !x.is_empty())
                    .map(|x| {
                        let (k, v) = x.split_once('=').unwrap();
                        (k.trim().to_string(), v.trim().parse().unwrap())
                    })
                    .collect();
                (label.trim().to_string(), kv)
            })
            .collect();
        let hal = Rc::new(MetalHalSha256::new());
        let circuit_hal = MetalCircuitHal::new(hal.clone());
        let params = EvalCheckParams::new(po2);
        let check = hal.alloc_elem("check", 4 * params.domain);
        let code = hal.copy_from_elem("code", &params.code);
        let data = hal.copy_from_elem("data", &params.data);
        let accum = hal.copy_from_elem("accum", &params.accum);
        let mix = hal.copy_from_elem("mix", &params.mix);
        let out = hal.copy_from_elem("out", &params.out);
        hal.sync();
        // K7_LIBS: comma list of `metallib[@keccak_metal_eval.rs]` (`lib=i` picks one; `lib`
        // unset = embedded). The optional constants file of the variant's build gives its stage
        // count and bindings (a build with another RISC0_KECCAK_METAL_EVAL_STAGES); without it
        // the variant has this build's stages.
        let libs: Vec<Vec<(metal::ComputePipelineState, u32)>> = std::env::var("K7_LIBS")
            .unwrap_or_default()
            .split(',')
            .filter(|p| !p.is_empty())
            .enumerate()
            .map(|(i, spec)| {
                let (path, consts) = spec.split_once('@').unwrap_or((spec, ""));
                let masks: Vec<u32> = if consts.is_empty() {
                    risc0_circuit_keccak_sys::metal_eval::EVAL_STAGE_BINDINGS.to_vec()
                } else {
                    let text = std::fs::read_to_string(consts).unwrap();
                    let line = text
                        .lines()
                        .find(|l| l.starts_with("pub const EVAL_STAGE_BINDINGS"))
                        .unwrap();
                    let list = &line[line.rfind('[').unwrap() + 1..line.rfind(']').unwrap()];
                    list.split(',')
                        .map(|x| {
                            u32::from_str_radix(x.trim().trim_start_matches("0x"), 16).unwrap()
                        })
                        .collect()
                };
                let names: Vec<String> = (0..masks.len())
                    .map(|k| format!("eval_stage_{k}"))
                    .collect();
                let names: Vec<&str> = names.iter().map(|n| n.as_str()).collect();
                let bytes = std::fs::read(path).unwrap();
                // The variant's own archive, next to it (never the shared cache dir).
                let dir = std::path::Path::new(path).parent().map(|d| d.join("arch"));
                let k = super::super::archive::ArchivedKernels::new_in(
                    dir,
                    &hal.device,
                    "keccak",
                    &bytes,
                    &names,
                )
                .unwrap();
                println!(
                    "K7 lib{i} {path}: {} stages, {} hits {} misses {:.1} s",
                    names.len(),
                    k.report.hits,
                    k.report.misses,
                    k.report.elapsed.as_secs_f64()
                );
                names
                    .iter()
                    .zip(masks)
                    .map(|(n, m)| (k.pipeline(n).clone(), m))
                    .collect()
            })
            .collect();
        let apply = |kv: &[(String, u32)]| {
            *circuit_hal.eval.stage_override.borrow_mut() = None;
            circuit_hal.set_eval_chunk(DEFAULT_EVAL_CHUNK);
            circuit_hal.set_eval_strict(false);
            for (k, v) in kv {
                match k.as_str() {
                    "chunk" => circuit_hal.set_eval_chunk(*v),
                    "strict" => circuit_hal.set_eval_strict(*v != 0),
                    "lib" => {
                        *circuit_hal.eval.stage_override.borrow_mut() =
                            Some(libs[*v as usize].clone())
                    }
                    _ => panic!("unknown K7 key {k}"),
                }
            }
        };
        // K7_QUIET=N: before each timed call, wait (up to 120 s) until the GPU's "Device
        // Utilization %" (ioreg; other processes share the GPU) is <= N, sampled 0.3 s apart.
        let quiet: Option<u32> = std::env::var("K7_QUIET").ok().map(|v| v.parse().unwrap());
        let wait_quiet = || {
            if let Some(q) = quiet {
                let t0 = Instant::now();
                while t0.elapsed() < Duration::from_secs(120) {
                    std::thread::sleep(Duration::from_millis(300));
                    if super::super::tests::gpu_utilization().is_some_and(|u| u <= q) {
                        break;
                    }
                }
            }
        };
        let run = || {
            let t0 = Instant::now();
            circuit_hal.eval_check_metal(
                &check,
                &[&accum, &code, &data],
                &[&mix, &out],
                params.poly_mix,
                params.po2,
                params.steps,
                None,
            );
            t0.elapsed()
        };
        let expected = std::env::var("K7_CPU")
            .is_ok_and(|v| v == "1")
            .then(|| cpu_check(&params));
        let mut reference: Option<Vec<Val>> = None;
        // Warm-up: one call per config, checked.
        for (label, kv) in &configs {
            apply(kv);
            run();
            let got = check.to_vec();
            if let Some(e) = &expected {
                assert_same(&format!("{label} vs CPU"), &got, e);
            }
            match &reference {
                None => reference = Some(got),
                Some(r) => assert_same(&format!("{label} vs {}", configs[0].0), &got, r),
            }
        }
        let mut times = vec![Vec::new(); configs.len()];
        for _ in 0..reps {
            for (i, (_, kv)) in configs.iter().enumerate() {
                apply(kv);
                wait_quiet();
                times[i].push(run().as_secs_f64() * 1e3);
            }
        }
        for (i, (label, _)) in configs.iter().enumerate() {
            let mut t = times[i].clone();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "K7AB po2 {po2} {EVAL_STAGES} stages {label}: median {:.1} ms min {:.1} max {:.1} (n {}) {:?}",
                t[t.len() / 2],
                t[0],
                t[t.len() - 1],
                t.len(),
                times[i].iter().map(|x| x.round() as u64).collect::<Vec<_>>()
            );
        }
        // The last round's output is still bit-exact.
        assert_same("last call", &check.to_vec(), reference.as_ref().unwrap());
    }

    /// Many chunks, a partial last chunk, and a cached stride larger than the chunk.
    #[test]
    fn eval_check_multi_chunk() {
        let _gpu = super::super::tests::gpu_test_lock();
        let hal = Rc::new(MetalHalSha256::new());
        let circuit_hal = MetalCircuitHal::new(hal.clone());
        circuit_hal.set_keep_eval_scratch(true);

        // PO2 8 (domain 1024): 11 chunks of 100 (the last of 24), then 1 chunk.
        let params = EvalCheckParams::new(8);
        let expected = cpu_check(&params);
        for chunk in [100, 1024, 4096] {
            let actual = metal_check(&hal, &circuit_hal, &params, chunk);
            assert_same(&format!("po2 8 chunk {chunk}"), &actual, &expected);
        }
        // The stride stayed 1024 (min(chunk, domain)): no regrowth for a smaller domain.
        assert_eq!(circuit_hal.eval_scratch_bytes(), scratch_bytes(1024));

        // PO2 14 (domain 65536): 22 chunks of 3000 (the last of 2536) in a stride of 3000, then
        // 66 chunks of 1000 reusing that stride (count < stride), then 2 chunks of 32768.
        let params = EvalCheckParams::new(14);
        let expected = cpu_check(&params);
        for chunk in [3000, 1000, 32768] {
            let actual = metal_check(&hal, &circuit_hal, &params, chunk);
            assert_same(&format!("po2 14 chunk {chunk}"), &actual, &expected);
            if chunk == 1000 {
                assert_eq!(circuit_hal.eval_scratch_bytes(), scratch_bytes(3000));
            }
        }
        circuit_hal.release_eval_scratch();
        assert_eq!(circuit_hal.eval_scratch_bytes(), 0);

        // By default every call frees its scratch on return (K4 review: a cached scratch held
        // memory a later, larger proof's commit needed under a 4 GB budget).
        circuit_hal.set_keep_eval_scratch(false);
        let params = EvalCheckParams::new(8);
        let expected = cpu_check(&params);
        let actual = metal_check(&hal, &circuit_hal, &params, DEFAULT_EVAL_CHUNK);
        assert_same("po2 8 without keep", &actual, &expected);
        assert_eq!(circuit_hal.eval_scratch_bytes(), 0);
    }

    /// Every stage binds params and scratch, the last one check, and only the bindings that
    /// exist; some stage reads `data` (all of the circuit's columns come from it).
    #[test]
    fn eval_stage_bindings() {
        use risc0_circuit_keccak_sys::metal_eval::{
            EVAL_BUF_CHECK, EVAL_BUF_DATA, EVAL_BUF_PARAMS, EVAL_BUF_SCRATCH, EVAL_STAGE_BINDINGS,
        };
        assert_eq!(EVAL_STAGE_BINDINGS.len(), EVAL_STAGES);
        let all = EVAL_STAGE_BINDINGS.iter().fold(0, |a, m| a | m);
        assert_eq!(all & !0x3ff, 0, "bindings above index 9");
        assert_ne!(all & (1 << EVAL_BUF_DATA), 0);
        for (k, m) in EVAL_STAGE_BINDINGS.iter().enumerate() {
            assert_ne!(m & (1 << EVAL_BUF_PARAMS), 0, "stage {k}");
            assert_ne!(m & (1 << EVAL_BUF_SCRATCH), 0, "stage {k}");
            assert_eq!(
                m & (1 << EVAL_BUF_CHECK) != 0,
                k + 1 == EVAL_STAGES,
                "stage {k}: check"
            );
        }
        println!("stage bindings: {EVAL_STAGE_BINDINGS:03x?}");
    }

    /// Under a working-set budget that does not fit the default chunk's 541 MB of scratch, the
    /// launcher shrinks the chunk and stays bit-exact.
    #[test]
    fn eval_check_shrinks_chunk_under_budget() {
        let _gpu = super::super::tests::gpu_test_lock();
        let hal = Rc::new(MetalHalSha256::new());
        let circuit_hal = MetalCircuitHal::new(hal.clone());
        let params = EvalCheckParams::new(14);
        let expected = cpu_check(&params);
        // Upload the inputs as eval_check_impl does, then cap the budget at what is allocated
        // plus room for a 4096-cycle scratch (and the 35 KB poly_mix buffer).
        let check = hal.alloc_elem("check", 4 * params.domain);
        let code = hal.copy_from_elem("code", &params.code);
        let data = hal.copy_from_elem("data", &params.data);
        let accum = hal.copy_from_elem("accum", &params.accum);
        let mix = hal.copy_from_elem("mix", &params.mix);
        let out = hal.copy_from_elem("out", &params.out);
        hal.sync();
        hal.trim_device_memory();
        let budget = hal.allocated_size() + scratch_bytes(4096) + (1 << 20);
        hal.set_working_set_limit(Some(budget));
        circuit_hal.set_eval_chunk(DEFAULT_EVAL_CHUNK);
        circuit_hal.set_keep_eval_scratch(true);
        use risc0_zkp::hal::CircuitHal as _;
        circuit_hal.eval_check(
            &check,
            &[&accum, &code, &data],
            &[&mix, &out],
            params.poly_mix,
            params.po2,
            params.steps,
        );
        hal.set_working_set_limit(None);
        // The budget is against the whole process's Metal memory: every test that allocates
        // Metal memory takes `gpu_test_lock` (K5, K8), so nothing else allocates or frees while
        // this runs and the chunk is exactly the one that fits, under the default parallel
        // harness too (K8: the lock-free `prove::tests` and archive tests used to race it).
        let stride = circuit_hal.eval_scratch_bytes() / scratch_bytes(1);
        println!("po2 14 under a budget: chunk {stride}");
        assert_eq!(stride, 4096);
        assert_same("po2 14 under budget", &check.to_vec(), &expected);
    }

    #[test]
    fn fit_chunk_halves_to_the_budget() {
        let base = 3 << 30;
        let limit = base + scratch_bytes(4096) + (1 << 20);
        assert_eq!(fit_chunk(32768, base, limit), 4096);
        assert_eq!(fit_chunk(32768, base, u64::MAX), 32768);
        assert_eq!(fit_chunk(3000, base, limit), 3000);
        assert_eq!(fit_chunk(32768, base, base), MIN_EVAL_CHUNK);
        assert_eq!(fit_chunk(100, base, base), 100);
        // The 4 GB minimum tier at po2 17 (commit phase ~3452 MiB allocated): 16384.
        assert_eq!(fit_chunk(32768, 3452 << 20, 4 << 30), 16384);
    }

    /// PSO creation and GPU time per stage (measurement, prints a table). `K4_PROFILE_PO2`
    /// (default 17) and `RISC0_KECCAK_METAL_EVAL_CHUNK` pick the shape. Each stage's time is the
    /// sum over the chunks of (encode + commit + wait) of its dispatch, synced one at a time.
    /// Run alone: `cargo test -p risc0-circuit-keccak --release --features metal --lib --
    /// eval_check_stage_profile --ignored --nocapture`
    #[test]
    #[ignore]
    fn eval_check_stage_profile() {
        let _gpu = super::super::tests::gpu_test_lock();
        let po2: usize = std::env::var("K4_PROFILE_PO2")
            .map(|s| s.parse().unwrap())
            .unwrap_or(17);

        // PSO creation, one stage at a time, on a fresh library (served by the OS shader cache
        // when this metallib's content was compiled before: see the K3 note).
        let device = metal::Device::system_default().unwrap();
        let library = device.new_library_with_data(METAL_LIB).unwrap();
        let mut pso_times = Vec::new();
        for name in EVAL_STAGE_NAMES {
            let f = library.get_function(name, None).unwrap();
            let t0 = Instant::now();
            let pso = device.new_compute_pipeline_state_with_function(&f).unwrap();
            pso_times.push(t0.elapsed());
            assert!(pso.max_total_threads_per_threadgroup() >= 32);
        }

        let hal = Rc::new(MetalHalSha256::new());
        let circuit_hal = MetalCircuitHal::new(hal.clone());
        // K4_PROFILE_KEEP=1: keep the scratch cached between the runs (else each run allocates
        // and frees it, as the prover does).
        circuit_hal.set_keep_eval_scratch(std::env::var("K4_PROFILE_KEEP").is_ok_and(|v| v == "1"));
        let params = EvalCheckParams::new(po2);
        let check = hal.alloc_elem("check", 4 * params.domain);
        let code = hal.copy_from_elem("code", &params.code);
        let data = hal.copy_from_elem("data", &params.data);
        let accum = hal.copy_from_elem("accum", &params.accum);
        let mix = hal.copy_from_elem("mix", &params.mix);
        let out = hal.copy_from_elem("out", &params.out);
        hal.sync();
        let run = |profile: Option<&mut Vec<Duration>>| {
            circuit_hal.eval_check_metal(
                &check,
                &[&accum, &code, &data],
                &[&mix, &out],
                params.poly_mix,
                params.po2,
                params.steps,
                profile,
            )
        };
        // Warm-up (allocates the scratch), then timed batched runs (best of 3), then the
        // per-stage run.
        run(None);
        let batched = (0..3)
            .map(|_| {
                let t0 = Instant::now();
                run(None);
                t0.elapsed()
            })
            .min()
            .unwrap();
        let mut stages = Vec::new();
        run(Some(&mut stages));
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        println!(
            "po2 {po2}: {EVAL_STAGES} stages, domain {}, chunk {} ({} chunks), scratch {:.1} MiB; \
             eval_check (batched, one sync) {:.3} s; per-stage synced sum {:.3} s; \
             PSO creation total {:.3} s",
            params.domain,
            circuit_hal.eval_chunk().min(params.domain as u32),
            params
                .domain
                .div_ceil(circuit_hal.eval_chunk().min(params.domain as u32) as usize),
            mib(scratch_bytes(
                circuit_hal.eval_chunk().min(params.domain as u32)
            )),
            batched.as_secs_f64(),
            stages.iter().sum::<Duration>().as_secs_f64(),
            pso_times.iter().sum::<Duration>().as_secs_f64(),
        );
        println!("stage | PSO s | GPU ms (all chunks)");
        for (i, name) in EVAL_STAGE_NAMES.iter().enumerate() {
            println!(
                "{name} | {:.3} | {:.2}",
                pso_times[i].as_secs_f64(),
                stages[i].as_secs_f64() * 1e3
            );
        }
    }
}
