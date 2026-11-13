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

//! Metal HAL for the recursion circuit (docs/metal-port/PLAN.md 4.3, item C5).
//!
//! Rewritten from `hal/hip.rs` / `hal/cuda.rs`: a [MetalCircuitHal] keeps an `Arc` of the zkp
//! [MetalHal] it works through (buffers may only be used through the HAL that allocated them),
//! the recursion metallib's PSOs ([RecursionKernels], `risc0_circuit_recursion_sys::METAL_LIB`)
//! and HAL-owned scratch buffers grown with 25 % headroom ([ScratchBuffers], PLAN 2.6).
//!
//! The circuit phases run on the GPU: `witgen` (C6; CPU WOM sort/scan between its two command
//! buffers, Seq modes on the CPU), `accum` (C7, one batch) and `eval_check` (C8, monolith or
//! split). NTT, Merkle hashing and FRI run through the zkp Metal HAL, for every hash suite:
//! `poseidon_254` (identity_p254) uses the zkp `MetalHashPoseidon254` since Z10
//! (`RISC0_METAL_REC_P254=cpu` reverts it to the CPU prover). `RISC0_METAL_REC_SEAL_CHECK=1`
//! re-proves every proof on the CPU with the same deterministic noise and requires identical
//! seals (C9 validation knob).
//!
//! PSOs: the base kernels (C1's hand-written ones and C2's four step wrappers) are built once per
//! process on first use; the eval_check PSOs are built lazily for the variant
//! `RISC0_METAL_REC_KERNELS` selects (split by default since C11), the first time a phase asks for
//! them or by [warmup] on a background thread. Never per proof. Each set builds on parallel
//! threads (C11, [Pipelines]): cold (no OS shader cache) base ~3 s and split ~2.2 s here, against
//! 5-6 s and 7 s sequential and 36 s for the monolith.
//!
//! Measurement aids (C11, off by default): `RISC0_METAL_REC_PROFILE=1` ([profile]) and
//! `RISC0_METAL_REC_CAPTURE=<dir>` ([capture]).

mod accum;
#[cfg(all(test, feature = "test"))]
mod c10_tests;
#[cfg(all(test, feature = "test"))]
mod c11_tests;
#[cfg(all(test, feature = "test"))]
mod c13_tests;
pub(crate) mod capture;
mod eval_check;
pub(crate) mod profile;
pub(crate) mod staging;
#[cfg(all(test, feature = "test"))]
mod tests;
mod witgen;
#[cfg(all(test, feature = "test"))]
mod z10_tests;

use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex, MutexGuard, OnceLock,
};

use anyhow::{anyhow, bail, Result};
use metal::{ComputePipelineState, DeviceRef};
use risc0_circuit_recursion_sys::{METAL_KERNEL_NAMES, METAL_LIB};
use risc0_zkp::{
    field::baby_bear::BabyBearExtElem,
    hal::{
        metal::{
            BufferImpl, Grid, MetalHal, MetalHalPoseidon2, MetalHalPoseidon254, MetalHalSha256,
            MetalHash, MetalHashPoseidon2, MetalHashPoseidon254, MetalHashSha256,
        },
        Buffer as _,
    },
};

use crate::prove::{
    witgen::with_noise_seed, Preflight, Program, RecursionProver, RecursionProverImpl,
    RecursionReceipt,
};

/// Environment variable selecting the eval_check kernels: `lz` (the split in lazy Fp form, default
/// since C12; uint4 mix-power loads since C13), `lz32` (the C12 four-load form, also
/// `RISC0_METAL_REC_LZ_UINT4=0`), `split` (the C11 default) or `monolith` (PLAN 2.0). C11 measured the split 31 %
/// faster per lift/join/resolve proof at po2 18 (interleaved A/B on captured zkvm proofs) and ~5x
/// cheaper to build cold (7.5 s vs 36 s); C12 measured the lazy Fp split 58.6 -> 47.8 ms per
/// eval_check call at po2 18 (see `docs/metal-port/notes/C12.md`).
pub(crate) const REC_KERNELS_ENV: &str = "RISC0_METAL_REC_KERNELS";

/// The monolithic eval_check entry point (C3, `cuda2metal_eval.py`).
const EVAL_MONOLITH: &[&str] = &["eval_check"];
/// The split eval_check entry points, in dispatch order (C3, `cuda2metal_eval.py --split`).
const EVAL_SPLIT: &[&str] = &[
    "ec_part_0",
    "ec_part_1",
    "ec_part_2",
    "ec_part_3",
    "ec_part_4",
    "ec_part_5",
];
/// The lazy Fp form of the split, in dispatch order (C12, `cuda2metal_eval.py --split-lz`): the
/// same parts and bindings as [EVAL_SPLIT], every part's Fp sums of products accumulated lazily.
const EVAL_SPLIT_LZ: &[&str] = &[
    "ec_lz_part_0",
    "ec_lz_part_1",
    "ec_lz_part_2",
    "ec_lz_part_3",
    "ec_lz_part_4",
    "ec_lz_part_5",
];

/// [EVAL_SPLIT_LZ] with the C12 four-load `rec_lz_add` (C13 made its mix-power read one 16-byte
/// uint4 load in [EVAL_SPLIT_LZ]); selected by `RISC0_METAL_REC_LZ_UINT4=0` (or
/// `RISC0_METAL_REC_KERNELS=lz32`).
const EVAL_SPLIT_LZ_U32: &[&str] = &[
    "ec_lz32_part_0",
    "ec_lz32_part_1",
    "ec_lz32_part_2",
    "ec_lz32_part_3",
    "ec_lz32_part_4",
    "ec_lz32_part_5",
];

/// Environment variable (read once): `0` reverts the lazy Fp split's uint4 mix-power loads (C13)
/// to the C12 four-load form ([EVAL_SPLIT_LZ_U32]). Only affects the default `lz` variant.
pub(crate) const REC_LZ_UINT4_ENV: &str = "RISC0_METAL_REC_LZ_UINT4";

/// Which eval_check kernels the recursion Metal HAL uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EvalKernels {
    Monolith,
    Split,
    SplitLz,
    SplitLzU32,
}

impl EvalKernels {
    /// Parse a `RISC0_METAL_REC_KERNELS` value; `None` for an unknown one.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "monolith" | "mono" | "eval_check" => Some(Self::Monolith),
            "split" | "ec_part" => Some(Self::Split),
            "" | "lz" | "split-lz" | "split_lz" | "ec_lz_part" => Some(Self::SplitLz),
            "lz32" | "lz-u32" | "lz_u32" | "ec_lz32_part" => Some(Self::SplitLzU32),
            _ => None,
        }
    }

    /// The process-wide selection (read once): `RISC0_METAL_REC_KERNELS`, default the lazy Fp
    /// split (C12; `split` reverts to the C11 default), with uint4 mix loads unless
    /// `RISC0_METAL_REC_LZ_UINT4=0` (C13).
    pub(crate) fn selected() -> Self {
        static SEL: OnceLock<EvalKernels> = OnceLock::new();
        *SEL.get_or_init(|| {
            let v = match std::env::var(REC_KERNELS_ENV) {
                Ok(v) => Self::parse(&v).unwrap_or_else(|| {
                    tracing::warn!(
                        "{REC_KERNELS_ENV}={v:?} is not lz|lz32|split|monolith; using lz"
                    );
                    Self::SplitLz
                }),
                Err(_) => Self::SplitLz,
            };
            Self::with_uint4(v, std::env::var(REC_LZ_UINT4_ENV).ok().as_deref())
        })
    }

    /// `v` with `RISC0_METAL_REC_LZ_UINT4` = `uint4` applied: `0` / `off` / `false` turns the
    /// default [Self::SplitLz] into [Self::SplitLzU32]; other variants are unchanged.
    pub(crate) fn with_uint4(v: Self, uint4: Option<&str>) -> Self {
        let off = uint4.is_some_and(|u| {
            matches!(
                u.trim().to_ascii_lowercase().as_str(),
                "0" | "off" | "false"
            )
        });
        if v == Self::SplitLz && off {
            Self::SplitLzU32
        } else {
            v
        }
    }

    /// The entry points of this variant, in dispatch order.
    pub(crate) fn names(self) -> &'static [&'static str] {
        match self {
            Self::Monolith => EVAL_MONOLITH,
            Self::Split => EVAL_SPLIT,
            Self::SplitLz => EVAL_SPLIT_LZ,
            Self::SplitLzU32 => EVAL_SPLIT_LZ_U32,
        }
    }

    /// Every variant (tests).
    #[allow(dead_code)]
    pub(crate) const ALL: [EvalKernels; 4] =
        [Self::Monolith, Self::Split, Self::SplitLz, Self::SplitLzU32];
}

/// Whether `name` is one of the eval_check entry points (any variant).
fn is_eval_kernel(name: &str) -> bool {
    EvalKernels::ALL.iter().any(|v| v.names().contains(&name))
}

/// Environment variable (read once): threads that build the recursion PSOs. Default
/// `available_parallelism()`; `1` is the pre-C11 sequential build. C11 measured the cold build
/// (OS shader cache off, 8 interleaved pairs, medians) of the base kernels 6.15 s -> 3.38 s and of
/// the split eval_check 7.33 s -> 2.00 s on parallel threads (each PSO compiles on its own thread).
pub(crate) const PSO_THREADS_ENV: &str = "RISC0_METAL_REC_PSO_THREADS";

fn pso_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var(PSO_THREADS_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()))
    })
}

/// A set of PSOs from one metallib, built on up to [pso_threads] threads (C11). The same API as
/// zkp's `MetalKernels` (which builds sequentially).
pub(crate) struct Pipelines {
    pipelines: std::collections::HashMap<String, ComputePipelineState>,
}

impl Pipelines {
    /// Load `bytes` once and build a PSO for each of `names` on up to `threads` threads. Fails on
    /// a missing name or a PSO the device rejects (the first error in `names` order).
    pub(crate) fn new(
        device: &DeviceRef,
        bytes: &[u8],
        names: &[&str],
        threads: usize,
    ) -> Result<Self> {
        let library = device
            .new_library_with_data(bytes)
            .map_err(|e| anyhow!("loading metallib ({} bytes): {e}", bytes.len()))?;
        let available = library.function_names();
        let missing: Vec<&str> = names
            .iter()
            .copied()
            .filter(|name| !available.iter().any(|a| a == name))
            .collect();
        if !missing.is_empty() {
            bail!("metallib is missing kernel(s): {}", missing.join(", "));
        }
        let build = |name: &str| -> Result<ComputePipelineState> {
            // Off the main thread: give each PSO its own autorelease pool (K5).
            metal::objc::rc::autoreleasepool(|| {
                let function = library
                    .get_function(name, None)
                    .map_err(|e| anyhow!("metallib function {name}: {e}"))?;
                device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|e| anyhow!("compute pipeline for {name}: {e}"))
            })
        };
        let threads = threads.clamp(1, names.len().max(1));
        let built: Vec<Result<ComputePipelineState>> = if threads == 1 {
            names.iter().map(|n| build(n)).collect()
        } else {
            let next = AtomicUsize::new(0);
            let slots: Mutex<Vec<Option<Result<ComputePipelineState>>>> =
                Mutex::new((0..names.len()).map(|_| None).collect());
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= names.len() {
                            break;
                        }
                        let r = build(names[i]);
                        slots.lock().unwrap_or_else(|e| e.into_inner())[i] = Some(r);
                    });
                }
            });
            slots
                .into_inner()
                .unwrap_or_else(|e| e.into_inner())
                .into_iter()
                .map(|r| r.unwrap_or_else(|| Err(anyhow!("PSO build thread panicked"))))
                .collect()
        };
        let mut pipelines = std::collections::HashMap::with_capacity(names.len());
        for (name, pso) in names.iter().zip(built) {
            pipelines.insert(name.to_string(), pso?);
        }
        Ok(Self { pipelines })
    }

    /// The PSO for `name`; panics (a programming error, plan 2.9) if it was not built.
    pub(crate) fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.get(name)
            .unwrap_or_else(|| panic!("recursion Metal kernel {name} was not loaded"))
    }

    pub(crate) fn get(&self, name: &str) -> Option<&ComputePipelineState> {
        self.pipelines.get(name)
    }

    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.pipelines.keys().map(String::as_str)
    }
}

/// The selected eval_check variant's PSOs (process-wide, independent of the base kernels so
/// [warmup] can build both at once).
static EVAL_KERNELS: OnceLock<std::result::Result<(EvalKernels, Pipelines), String>> =
    OnceLock::new();

/// The recursion metallib's PSOs: every [METAL_KERNEL_NAMES] entry except the eval_check ones
/// eagerly, the selected eval_check variant lazily ([EVAL_KERNELS]). One instance per process (one
/// Metal device), like the zkp PSO cache.
pub(crate) struct RecursionKernels {
    base: Pipelines,
}

impl RecursionKernels {
    fn load(device: &DeviceRef) -> Result<Self> {
        for names in [EVAL_MONOLITH, EVAL_SPLIT, EVAL_SPLIT_LZ, EVAL_SPLIT_LZ_U32] {
            for name in names {
                if !METAL_KERNEL_NAMES.contains(name) {
                    bail!("risc0-circuit-recursion-sys METAL_KERNEL_NAMES does not list {name}");
                }
            }
        }
        let base: Vec<&str> = METAL_KERNEL_NAMES
            .iter()
            .copied()
            .filter(|name| !is_eval_kernel(name))
            .collect();
        let t0 = std::time::Instant::now();
        let base = Pipelines::new(device, METAL_LIB, &base, pso_threads())?;
        tracing::debug!(
            "recursion Metal: {} base PSOs in {:.2}s",
            base.names().count(),
            t0.elapsed().as_secs_f64()
        );
        Ok(Self { base })
    }

    /// The process-wide instance, built on `device` on first use. Takes no GPU lock (PSO creation
    /// needs only the device), so a warmup thread can build it while a proof holds `exclusive()`.
    pub(crate) fn get(device: &DeviceRef) -> Result<&'static Self> {
        static KERNELS: OnceLock<std::result::Result<RecursionKernels, String>> = OnceLock::new();
        KERNELS
            .get_or_init(|| Self::load(device).map_err(|e| format!("{e:#}")))
            .as_ref()
            .map_err(|e| anyhow!("loading the recursion metallib: {e}"))
    }

    /// The PSO of a base (non-eval_check) kernel; panics if `name` is not one (programming error).
    #[allow(dead_code)]
    pub(crate) fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.base.pipeline(name)
    }

    /// The selected eval_check variant and its PSOs, built on first call (cold: tens of seconds
    /// for the monolith; warm the process with [warmup]).
    pub(crate) fn eval(&self, device: &DeviceRef) -> Result<(EvalKernels, &'static Pipelines)> {
        eval_kernels(device)
    }
}

/// The selected eval_check variant and its PSOs, built on first call (see
/// [RecursionKernels::eval]).
pub(crate) fn eval_kernels(device: &DeviceRef) -> Result<(EvalKernels, &'static Pipelines)> {
    let entry = EVAL_KERNELS.get_or_init(|| {
        let variant = EvalKernels::selected();
        let t0 = std::time::Instant::now();
        let kernels = Pipelines::new(device, METAL_LIB, variant.names(), pso_threads())
            .map_err(|e| format!("{e:#}"))?;
        tracing::debug!(
            "recursion Metal: eval_check {variant:?} PSOs in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        Ok((variant, kernels))
    });
    match entry {
        Ok((variant, kernels)) => Ok((*variant, kernels)),
        Err(e) => Err(anyhow!("building the recursion eval_check PSOs: {e}")),
    }
}

/// Which launch-width knob a dispatch follows (C11 threadgroup sweep).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TgKind {
    /// The eval_check dispatches: `RISC0_METAL_REC_EVAL_TG`.
    Eval = 0,
    /// The four step kernels (exec, verify WOM, compute/verify accum): `RISC0_METAL_REC_STEP_TG`.
    Step = 1,
}

const TG_ENVS: [&str; 2] = ["RISC0_METAL_REC_EVAL_TG", "RISC0_METAL_REC_STEP_TG"];

thread_local! {
    /// Test override of the knobs (sweeps in one process).
    static TG_OVERRIDE: std::cell::Cell<[Option<u64>; 2]> = const { std::cell::Cell::new([None; 2]) };
}

/// Run `f` with the threadgroup width of `kind` forced to `tg` (0 = default) on this thread.
#[allow(dead_code)]
pub(crate) fn with_tg<R>(kind: TgKind, tg: u64, f: impl FnOnce() -> R) -> R {
    let old = TG_OVERRIDE.with(|c| c.get());
    let mut v = old;
    v[kind as usize] = Some(tg);
    TG_OVERRIDE.with(|c| c.set(v));
    struct Restore([Option<u64>; 2]);
    impl Drop for Restore {
        fn drop(&mut self) {
            TG_OVERRIDE.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(old);
    f()
}

/// The 1-D grid of `n` threads for a dispatch of `kind`: the default `Grid::Threads` (threadgroups
/// of min(256, PSO max)) unless the knob or a test override names a width; that width is clamped
/// to the PSO's `maxTotalThreadsPerThreadgroup` and rounded down to whole SIMD groups (no core
/// count involved). Every kernel bounded this way checks `gid` against its thread count.
pub(crate) fn grid_1d(kind: TgKind, pso: &ComputePipelineState, n: u64) -> Grid {
    static ENV: OnceLock<[u64; 2]> = OnceLock::new();
    let env = ENV.get_or_init(|| {
        std::array::from_fn(|i| {
            std::env::var(TG_ENVS[i])
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0)
        })
    });
    let tg = TG_OVERRIDE
        .with(|c| c.get()[kind as usize])
        .unwrap_or(env[kind as usize]);
    if tg == 0 {
        return Grid::Threads(n);
    }
    let simd = pso.thread_execution_width().max(1);
    let tg = tg.min(pso.max_total_threads_per_threadgroup());
    let tg = (tg / simd * simd).max(simd);
    Grid::Groups {
        groups: metal::MTLSize::new(n.div_ceil(tg), 1, 1),
        threads: metal::MTLSize::new(tg, 1, 1),
    }
}

/// HAL-owned scratch (PLAN 2.6), grown with 25 % headroom and reused across proofs. C6/C7/C8 add
/// the buffers their kernels bind (womRows, womIndex, prefix totals, ...).
#[derive(Default)]
#[allow(dead_code)]
pub(crate) struct ScratchBuffers {
    /// Private copy of the preflight cycles (`RawPreflightCycle` = 2 words each): the witgen
    /// advances `iopIdx`, so it works on this copy and the host trace stays pristine.
    pub cycles: Option<BufferImpl<u32>>,
    /// The fault word of the current phase (PLAN 2.9; [MetalCircuitHal::fault_word]).
    pub fault: Option<BufferImpl<u32>>,
    /// poly_mix powers for eval_check (C8).
    pub poly_mix: Option<BufferImpl<BabyBearExtElem>>,
    /// The witgen's trace copies and WOM argument buffers (C6).
    pub witgen: witgen::WitgenScratch,
    /// Accum per-cycle FpExt products (C7; CUDA `g_accumCache.accum`), work_cycles elements.
    pub accum_tmp: Option<BufferImpl<BabyBearExtElem>>,
    /// Accum prefix-scan chunk totals (C7), ceil(work_cycles / 2048) elements.
    pub accum_totals: Option<BufferImpl<BabyBearExtElem>>,
}

/// A buffer of at least `size` elements: `cur` if large enough, else a new one with 25 % headroom
/// (the old one returns to the HAL's pool once no batch uses it).
pub(crate) fn grow<T: Clone, MH: MetalHash>(
    hal: &MetalHal<MH>,
    cur: &mut Option<BufferImpl<T>>,
    name: &'static str,
    size: usize,
) -> BufferImpl<T> {
    match cur {
        Some(buf) if buf.size() >= size => buf.slice(0, size),
        _ => {
            let buf = hal.alloc_buffer::<T>(name, (size + size / 4).max(1));
            *cur = Some(buf.clone());
            buf.slice(0, size)
        }
    }
}

/// A circuit phase of the recursion HAL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Witgen,
    Accum,
    EvalCheck,
}

/// Per-phase counts of calls served by the Metal circuit HAL (tests and logs).
pub(crate) static PHASE_CALLS: [AtomicUsize; 3] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];

/// Per-phase highest device `currentAllocatedSize` (bytes) seen by [MetalCircuitHal::record], for
/// the PLAN 9.1 peak-memory report (tests and `RUST_LOG=debug`).
pub(crate) static PHASE_ALLOC: [AtomicU64; 3] =
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// Poisoned scratch mutexes recovered by [MetalCircuitHal::scratch] (tests and logs).
pub(crate) static SCRATCH_POISON_RECOVERIES: AtomicUsize = AtomicUsize::new(0);

/// The calls served so far, `[witgen, accum, eval_check]`.
#[allow(dead_code)]
pub(crate) fn phase_calls() -> [usize; 3] {
    std::array::from_fn(|i| PHASE_CALLS[i].load(Ordering::Relaxed))
}

/// The highest allocation recorded per phase, `[witgen, accum, eval_check]`, in bytes.
#[allow(dead_code)]
pub(crate) fn phase_alloc() -> [u64; 3] {
    std::array::from_fn(|i| PHASE_ALLOC[i].load(Ordering::Relaxed))
}

pub(crate) struct MetalCircuitHal<MH: MetalHash> {
    hal: Arc<MetalHal<MH>>,
    kernels: &'static RecursionKernels,
    /// A Mutex rather than PLAN 2.6's RefCell, so the circuit HAL is Send + Sync like the HAL
    /// (as rv32im's R7 HAL; `Arc<RefCell<..>>` trips clippy::arc_with_non_send_sync).
    scratch: Mutex<ScratchBuffers>,
}

pub(crate) type MetalCircuitHalSha256 = MetalCircuitHal<MetalHashSha256>;
pub(crate) type MetalCircuitHalPoseidon2 = MetalCircuitHal<MetalHashPoseidon2>;
pub(crate) type MetalCircuitHalPoseidon254 = MetalCircuitHal<MetalHashPoseidon254>;

impl<MH: MetalHash> MetalCircuitHal<MH> {
    pub fn new(hal: Arc<MetalHal<MH>>) -> Result<Self> {
        Ok(Self {
            kernels: RecursionKernels::get(&hal.device)?,
            hal,
            scratch: Mutex::new(ScratchBuffers::default()),
        })
    }

    #[allow(dead_code)]
    pub(crate) fn hal(&self) -> &MetalHal<MH> {
        &self.hal
    }

    /// The recursion PSOs (base kernels; `eval` for eval_check).
    #[allow(dead_code)]
    pub(crate) fn kernels(&self) -> &'static RecursionKernels {
        self.kernels
    }

    /// The selected eval_check variant and its PSOs (built on first use).
    #[allow(dead_code)]
    pub(crate) fn eval_kernels(&self) -> Result<(EvalKernels, &'static Pipelines)> {
        self.kernels.eval(&self.hal.device)
    }

    /// The HAL-owned scratch buffers. Do not hold the guard across a call that borrows it again.
    ///
    /// Poison-tolerant (C10): a panic while the guard is held (in practice the zkp HAL's
    /// working-set budget panic inside [grow], M1) used to poison the mutex, and every later proof
    /// of this cached prover failed with `PoisonError`. The scratch holds no state a proof relies
    /// on across calls (every phase sizes, fills or zeroes what it binds), so a poisoned guard is
    /// recovered by dropping all scratch buffers (their memory goes back to the HAL's pool once no
    /// batch uses them; after an out-of-memory panic that is what we want anyway) and clearing the
    /// poison. The next proof regrows what it needs.
    pub(crate) fn scratch(&self) -> MutexGuard<'_, ScratchBuffers> {
        self.scratch.lock().unwrap_or_else(|poisoned| {
            tracing::warn!(
                "recursion Metal: scratch mutex poisoned by an earlier panic; dropping the \
                 scratch buffers and continuing"
            );
            let mut guard = poisoned.into_inner();
            *guard = ScratchBuffers::default();
            self.scratch.clear_poison();
            SCRATCH_POISON_RECOVERIES.fetch_add(1, Ordering::Relaxed);
            guard
        })
    }

    /// Count a call of `phase` and record the device allocation (PLAN 9.1 peak per phase).
    /// Under [PROFILE_ENV], also close the profile interval of the GPU work queued before it.
    pub(crate) fn enter(&self, phase: Phase) {
        PHASE_CALLS[phase as usize].fetch_add(1, Ordering::Relaxed);
        if profile::enabled() {
            match phase {
                Phase::Witgen => {
                    let _ = self.sync();
                    profile::start();
                }
                Phase::Accum => self.prof_mark("pre_accum"),
                Phase::EvalCheck => self.prof_mark("pre_eval"),
            }
        }
        self.record(phase);
    }

    /// Under [PROFILE_ENV]: sync, then record the wall time since the previous mark as `label`.
    pub(crate) fn prof_mark(&self, label: &'static str) {
        if profile::enabled() {
            let _ = self.sync();
            profile::mark(label);
        }
    }

    /// Record the device's current allocation as a candidate peak of `phase`.
    pub(crate) fn record(&self, phase: Phase) {
        let bytes = self.hal.allocated_size();
        PHASE_ALLOC[phase as usize].fetch_max(bytes, Ordering::Relaxed);
        tracing::debug!("recursion Metal {phase:?}: device allocation {bytes} B");
    }

    /// Wait for all GPU work of the HAL, so host views of its buffers are current; a failed
    /// command buffer becomes an error.
    pub(crate) fn sync(&self) -> Result<()> {
        self.hal.try_sync()
    }

    /// The phase's fault word (one `uint`, bound as `device atomic_uint* fault`), with a blit that
    /// zeroes it encoded into the open batch, so kernels encoded after this call see 0.
    #[allow(dead_code)]
    pub(crate) fn fault_word(&self) -> BufferImpl<u32> {
        let buf = grow(&self.hal, &mut self.scratch().fault, "rec_fault", 1);
        self.hal
            .blit_fill(buf.as_buf(), (buf.offset() * 4) as u64, 4, 0);
        buf
    }

    /// Sync, then fail if a kernel of `phase` reported a fault in `fault` (PLAN 2.9: corrupt trace
    /// or out-of-range index; the kernels clamp and run to completion first).
    #[allow(dead_code)]
    pub(crate) fn check_fault(&self, fault: &BufferImpl<u32>, phase: Phase) -> Result<()> {
        self.sync()
            .map_err(|e| e.context(format!("recursion Metal {phase:?}")))?;
        let word = fault.get_at(0);
        if word != 0 {
            bail!(
                "recursion Metal {phase:?}: kernel fault 0x{word:08x} (inconsistent trace or \
                 out-of-range index)"
            );
        }
        Ok(())
    }
}

/// A host slice over a Metal buffer view. Only valid after [MetalCircuitHal::sync] and while no
/// GPU work using the buffer is encoded; the CPU kernels write through it (the shared storage is
/// coherent, as `hal/cpu.rs` writes through `as_slice`).
pub(crate) unsafe fn host_slice<T: Clone>(buf: &BufferImpl<T>) -> &[T] {
    std::slice::from_raw_parts(buf.as_ptr(), buf.size())
}

/// Build the recursion PSOs (base and the selected eval_check variant) and the zkp PSOs, so the
/// first lift does not pay for them. Run it on a background thread: cold it takes tens of
/// seconds (C4 note). Takes the GPU lock only briefly (`MetalHal::new`), after the recursion PSOs.
pub(crate) fn warmup() -> Result<()> {
    let device = metal::Device::system_default().ok_or_else(|| anyhow!("no Metal device"))?;
    let t0 = std::time::Instant::now();
    // Base and eval_check PSOs at once (each set also builds on parallel threads, C11).
    std::thread::scope(|s| {
        let eval = s.spawn(|| eval_kernels(&device).map(|_| ()));
        let base = RecursionKernels::get(&device).map(|_| ());
        base.and(
            eval.join()
                .unwrap_or_else(|_| Err(anyhow!("eval PSO thread panicked"))),
        )
    })?;
    drop(MetalHalPoseidon2::new());
    tracing::debug!("recursion Metal warmup: {:.2}s", t0.elapsed().as_secs_f64());
    Ok(())
}

/// Environment variable: `1` makes every Metal recursion proof (poseidon2, sha-256) a validation
/// run (C9): the proof is made with deterministic ZK noise, then re-proved by the CPU prover with
/// the same noise, and the two seals must be bit-identical (else the proof fails). Doubles the
/// cost of a proof and removes zero knowledge: never set it in production.
pub(crate) const SEAL_CHECK_ENV: &str = "RISC0_METAL_REC_SEAL_CHECK";

/// Seal checks passed so far (tests).
pub(crate) static SEAL_CHECKS_OK: AtomicUsize = AtomicUsize::new(0);

/// Serializes the tests that set or assert process-wide prover state (the seal-check and
/// phase counters, the host chain check switch). Poison-tolerant, so one failing test does
/// not fail the rest.
#[cfg(test)]
pub(crate) fn global_state_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn seal_check_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var(SEAL_CHECK_ENV).as_deref() == Ok("1"))
}

/// The first index where `a` and `b` differ (or the shorter length), `None` if equal.
pub(crate) fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or((a.len() != b.len()).then(|| a.len().min(b.len())))
}

/// A copy of a preflight's trace and output (the parts a proof reads), for the CPU re-proof.
fn clone_preflight(p: &Preflight) -> Preflight {
    let mut c = Preflight::new(std::collections::VecDeque::new());
    c.trace.wom = p.trace.wom.clone();
    c.trace.cycles = p.trace.cycles.clone();
    c.trace.iops = p.trace.iops.clone();
    c.output = p.output.clone();
    c
}

/// Prove with `metal` and then with `cpu`, both under the same deterministic noise seed, and
/// require bit-identical seals and outputs ([SEAL_CHECK_ENV]).
pub(crate) fn prove_seal_checked(
    metal: &dyn RecursionProver,
    cpu: &dyn RecursionProver,
    program: &Arc<Program>,
    input: std::collections::VecDeque<u32>,
    preflight: Option<Preflight>,
) -> Result<RecursionReceipt> {
    let seed: u64 = rand::random();
    let cpu_preflight = preflight.as_ref().map(clone_preflight);
    let t0 = std::time::Instant::now();
    let m = with_noise_seed(seed, || {
        metal.prove_shared(program, input.clone(), preflight)
    })?;
    let metal_s = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let c = with_noise_seed(seed, || cpu.prove_shared(program, input, cpu_preflight))?;
    let cpu_s = t0.elapsed().as_secs_f64();
    if let Some(i) = first_diff(&m.output, &c.output) {
        bail!("{SEAL_CHECK_ENV}: output differs from the CPU prover at word {i}");
    }
    if let Some(i) = first_diff(&m.seal, &c.seal) {
        bail!(
            "{SEAL_CHECK_ENV}: seal differs from the CPU prover at word {i} of {} / {}",
            m.seal.len(),
            c.seal.len()
        );
    }
    let n = SEAL_CHECKS_OK.fetch_add(1, Ordering::Relaxed) + 1;
    eprintln!(
        "[{SEAL_CHECK_ENV}] #{n}: po2 {} ({} rows) seal == CPU seal ({} words); metal {metal_s:.2}s, \
         cpu {cpu_s:.2}s",
        program.po2,
        program.code_rows(),
        m.seal.len()
    );
    Ok(m)
}

/// The Metal prover under [SEAL_CHECK_ENV].
struct SealCheckProver {
    metal: Box<dyn RecursionProver>,
    cpu: Box<dyn RecursionProver>,
}

impl RecursionProver for SealCheckProver {
    fn prove(
        &self,
        program: &Program,
        input: std::collections::VecDeque<u32>,
        preflight: Option<Preflight>,
    ) -> Result<RecursionReceipt> {
        self.prove_shared(&Arc::new(program.clone()), input, preflight)
    }

    fn prewarm(&self, program: &Arc<Program>) -> Result<()> {
        self.metal.prewarm(program)
    }

    fn prove_shared(
        &self,
        program: &Arc<Program>,
        input: std::collections::VecDeque<u32>,
        preflight: Option<Preflight>,
    ) -> Result<RecursionReceipt> {
        prove_seal_checked(
            self.metal.as_ref(),
            self.cpu.as_ref(),
            program,
            input,
            preflight,
        )
    }
}

/// Environment variable (read once): `cpu` proves `poseidon_254` (identity_p254) with the CPU
/// recursion prover, as before Z10; `metal` (default) with the Metal HAL and
/// `MetalHashPoseidon254`.
pub(crate) const REC_P254_ENV: &str = "RISC0_METAL_REC_P254";

/// Parse a [REC_P254_ENV] value: `Some(true)` for Metal, `Some(false)` for the CPU, `None` for an
/// unknown value.
pub(crate) fn parse_rec_p254(v: &str) -> Option<bool> {
    match v.trim().to_ascii_lowercase().as_str() {
        "metal" | "gpu" | "" => Some(true),
        "cpu" => Some(false),
        _ => None,
    }
}

/// Whether `poseidon_254` proves on Metal ([REC_P254_ENV], read once; unknown values warn and
/// keep Metal).
pub(crate) fn p254_on_metal() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var(REC_P254_ENV) {
        Err(_) => true,
        Ok(v) => parse_rec_p254(&v).unwrap_or_else(|| {
            tracing::warn!("{REC_P254_ENV}={v}: unknown (cpu | metal), using metal");
            true
        }),
    })
}

pub(crate) fn recursion_prover(hashfn: &str) -> Result<Box<dyn RecursionProver>> {
    let metal = metal_recursion_prover(hashfn)?;
    let metal = match std::env::var_os(capture::CAPTURE_ENV) {
        Some(dir) if !dir.is_empty() => Box::new(capture::CaptureProver {
            inner: metal,
            dir: dir.into(),
            hashfn: hashfn.to_string(),
        }),
        _ => metal,
    };
    let cpu_p254 = hashfn == "poseidon_254" && !p254_on_metal();
    if seal_check_enabled() && !cpu_p254 {
        tracing::warn!("{SEAL_CHECK_ENV}=1: every recursion proof is re-proved on the CPU");
        return Ok(Box::new(SealCheckProver {
            metal,
            cpu: super::cpu::recursion_prover(hashfn)?,
        }));
    }
    Ok(metal)
}

fn metal_recursion_prover(hashfn: &str) -> Result<Box<dyn RecursionProver>> {
    match hashfn {
        "poseidon2" => {
            let hal = Arc::new(MetalHalPoseidon2::new());
            let circuit_hal = Arc::new(MetalCircuitHalPoseidon2::new(hal.clone())?);
            Ok(Box::new(RecursionProverImpl::new(hal, circuit_hal)))
        }
        "sha-256" => {
            let hal = Arc::new(MetalHalSha256::new());
            let circuit_hal = Arc::new(MetalCircuitHalSha256::new(hal.clone())?);
            Ok(Box::new(RecursionProverImpl::new(hal, circuit_hal)))
        }
        // identity_p254 (Z10): the Metal HAL with MetalHashPoseidon254, or the CPU prover under
        // RISC0_METAL_REC_P254=cpu (the pre-Z10 path).
        "poseidon_254" if p254_on_metal() => metal_p254_prover(),
        "poseidon_254" => super::cpu::recursion_prover(hashfn),
        _ => bail!("Unsupported hashfn: {hashfn}"),
    }
}

/// The Metal `poseidon_254` recursion prover (whatever [REC_P254_ENV] says).
pub(crate) fn metal_p254_prover() -> Result<Box<dyn RecursionProver>> {
    let hal = Arc::new(MetalHalPoseidon254::new());
    let circuit_hal = Arc::new(MetalCircuitHalPoseidon254::new(hal.clone())?);
    Ok(Box::new(RecursionProverImpl::new(hal, circuit_hal)))
}

#[cfg(all(test, feature = "test"))]
mod c5_tests {
    use std::{collections::VecDeque, sync::Arc};

    use risc0_core::field::baby_bear::BabyBear;
    use risc0_zkp::{
        core::hash::sha::Sha256HashSuite,
        hal::{
            cpu::CpuHal,
            metal::{MetalHalPoseidon2, MetalHalSha256},
            Buffer,
        },
    };
    use test_log::test;

    use super::*;
    use crate::prove::hal::cpu::CpuCircuitHal;

    #[test]
    fn eval_kernels_parse() {
        assert_eq!(EvalKernels::parse(""), Some(EvalKernels::SplitLz));
        assert_eq!(EvalKernels::parse("lz"), Some(EvalKernels::SplitLz));
        assert_eq!(EvalKernels::parse("monolith"), Some(EvalKernels::Monolith));
        assert_eq!(EvalKernels::parse(" Split "), Some(EvalKernels::Split));
        assert_eq!(EvalKernels::parse("bogus"), None);
        assert_eq!(EvalKernels::Monolith.names(), &["eval_check"]);
        assert_eq!(EvalKernels::Split.names().len(), 6);
        assert_eq!(EvalKernels::SplitLz.names().len(), 6);
        assert_eq!(EvalKernels::parse("lz32"), Some(EvalKernels::SplitLzU32));
        assert_eq!(EvalKernels::SplitLzU32.names().len(), 6);
        // C13: RISC0_METAL_REC_LZ_UINT4=0 reverts only the default variant.
        let u = |v, s| EvalKernels::with_uint4(v, s);
        assert_eq!(u(EvalKernels::SplitLz, None), EvalKernels::SplitLz);
        assert_eq!(u(EvalKernels::SplitLz, Some("1")), EvalKernels::SplitLz);
        assert_eq!(u(EvalKernels::SplitLz, Some("0")), EvalKernels::SplitLzU32);
        assert_eq!(
            u(EvalKernels::SplitLz, Some(" OFF")),
            EvalKernels::SplitLzU32
        );
        assert_eq!(u(EvalKernels::Split, Some("0")), EvalKernels::Split);
        assert_eq!(u(EvalKernels::Monolith, Some("0")), EvalKernels::Monolith);
    }

    #[test]
    fn base_kernels_loaded_once() {
        let hal = Arc::new(MetalHalPoseidon2::new());
        let a = MetalCircuitHal::new(hal.clone()).unwrap();
        let b = MetalCircuitHal::new(hal).unwrap();
        assert!(std::ptr::eq(a.kernels(), b.kernels()));
        let names: Vec<&str> = a.kernels().base.names().collect();
        let eval_names: usize = EvalKernels::ALL.iter().map(|v| v.names().len()).sum();
        assert_eq!(eval_names, 19);
        assert_eq!(names.len(), METAL_KERNEL_NAMES.len() - eval_names);
        for name in METAL_KERNEL_NAMES.iter().filter(|n| !is_eval_kernel(n)) {
            assert!(a.kernels().base.get(name).is_some(), "{name}");
        }
        assert!(a.kernels().base.get("eval_check").is_none());
    }

    #[test]
    fn scratch_growth_headroom() {
        let hal = Arc::new(MetalHalPoseidon2::new());
        let c = MetalCircuitHal::new(hal.clone()).unwrap();
        let mut s = ScratchBuffers::default();
        let a = grow(&hal, &mut s.cycles, "t", 100);
        assert_eq!(a.size(), 100);
        assert_eq!(s.cycles.as_ref().unwrap().size(), 125);
        let b = grow(&hal, &mut s.cycles, "t", 120);
        assert_eq!(b.size(), 120);
        assert_eq!(
            b.as_device_ptr(),
            a.as_device_ptr(),
            "reused within headroom"
        );
        let d = grow(&hal, &mut s.cycles, "t", 126);
        assert_eq!(d.size(), 126);
        assert_eq!(s.cycles.as_ref().unwrap().size(), 157);
        drop(c);
    }

    #[test]
    fn fault_word_and_check() {
        let hal = Arc::new(MetalHalPoseidon2::new());
        let c = MetalCircuitHal::new(hal.clone()).unwrap();
        let f = c.fault_word();
        c.check_fault(&f, Phase::Witgen).unwrap();
        // A kernel would OR into it; emulate with a host write after a sync.
        c.sync().unwrap();
        f.view_mut(|v| v[0] = 5);
        let err = c.check_fault(&f, Phase::Accum).unwrap_err();
        assert!(format!("{err:#}").contains("fault 0x00000005"), "{err:#}");
        // The next phase's fault word is zeroed by the encoded blit.
        let f = c.fault_word();
        c.check_fault(&f, Phase::EvalCheck).unwrap();
    }

    #[test]
    fn eval_check_matches_cpu() {
        const PO2: usize = 4;
        let cpu_hal: CpuHal<BabyBear> = CpuHal::new(Sha256HashSuite::new_suite());
        let gpu_hal = Arc::new(MetalHalSha256::new());
        let gpu_eval = MetalCircuitHal::new(gpu_hal.clone()).unwrap();
        crate::testutil::eval_check(&cpu_hal, CpuCircuitHal, gpu_hal.as_ref(), gpu_eval, PO2);
    }

    #[test]
    fn factory_hashfns() {
        let _guard = crate::prove::hal::metal::global_state_test_lock();
        for h in ["poseidon2", "sha-256", "poseidon_254"] {
            recursion_prover(h).unwrap();
        }
        assert!(recursion_prover("blake3").is_err());
    }

    /// The test_recursion_circuit.zkr program through the Metal HAL (CPU circuit phases): same
    /// output and globals as the CPU prover, every phase served by the Metal circuit HAL. Prints
    /// the peak device allocation per phase and overall (PLAN 9.1).
    fn prove_test_zkr(po2: usize) {
        use crate::prove::zkr::get_zkr;
        let program = get_zkr("test_recursion_circuit.zkr", po2).unwrap();
        // Two Poseidon2 digests (8 words each), as zkvm's `test_zkr` adds them.
        let input: VecDeque<u32> = (0..16u32).map(|i| i * 0x101 + 7).collect();
        let before = phase_calls();
        let metal = recursion_prover("poseidon2").unwrap();
        let probe = MetalHalPoseidon2::new();
        probe.reset_peak_allocated_size();
        let base = probe.allocated_size();
        for a in &PHASE_ALLOC {
            a.store(0, Ordering::Relaxed);
        }
        let t0 = std::time::Instant::now();
        let m = metal.prove(&program, input.clone(), None).unwrap();
        let metal_s = t0.elapsed().as_secs_f64();
        let peak = probe.peak_allocated_size();
        let after = phase_calls();
        let alloc = phase_alloc();
        let cpu = crate::prove::hal::cpu::recursion_prover("poseidon2").unwrap();
        let t0 = std::time::Instant::now();
        let c = cpu.prove(&program, input, None).unwrap();
        let cpu_s = t0.elapsed().as_secs_f64();
        eprintln!(
            "C5 po2 {po2}: metal-hal prove {metal_s:.2}s, cpu prove {cpu_s:.2}s; device alloc \
             before {base} B, at witgen {} B, accum {} B, eval_check {} B, peak {peak} B",
            alloc[0], alloc[1], alloc[2]
        );
        assert_eq!(m.output, c.output);
        assert_eq!(m.seal.len(), c.seal.len());
        // The header (globals) precedes the randomized part of the seal.
        let globals = <crate::CircuitImpl as risc0_zkp::adapter::CircuitInfo>::OUTPUT_SIZE;
        assert_eq!(m.seal[..globals], c.seal[..globals]);
        for i in 0..3 {
            assert!(
                after[i] > before[i],
                "phase {i} not served by the Metal HAL"
            );
        }
        // A second proof reuses the PSOs and the scratch.
        let t0 = std::time::Instant::now();
        let m2 = metal
            .prove(&program, (0..16u32).map(|i| i * 0x101 + 7).collect(), None)
            .unwrap();
        eprintln!(
            "C5 po2 {po2}: second metal-hal prove {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        assert_eq!(m2.output, m.output);
    }

    #[test]
    fn prove_test_zkr_po2_16_matches_cpu() {
        prove_test_zkr(16);
    }

    #[test]
    fn prove_test_zkr_po2_18_matches_cpu() {
        prove_test_zkr(18); // zkvm RECURSION_PO2
    }

    /// The warmup builds the base and the selected eval_check PSOs; afterwards they are instant.
    #[test]
    fn warmup_builds_eval_pipelines() {
        let t0 = std::time::Instant::now();
        warmup().unwrap();
        let cold = t0.elapsed().as_secs_f64();
        let hal = Arc::new(MetalHalSha256::new());
        let c = MetalCircuitHal::new(hal).unwrap();
        let t0 = std::time::Instant::now();
        let (variant, kernels) = c.eval_kernels().unwrap();
        let warm = t0.elapsed().as_secs_f64();
        assert_eq!(variant, EvalKernels::selected());
        for name in variant.names() {
            assert!(kernels.get(name).is_some(), "{name}");
        }
        eprintln!("C5 warmup ({variant:?}): {cold:.2}s; eval PSOs afterwards {warm:.4}s");
        assert!(warm < 1.0);
    }
}
