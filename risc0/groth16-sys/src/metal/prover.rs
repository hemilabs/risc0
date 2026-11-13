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

//! The Metal Groth16 prover's orchestration (item G7, docs/metal-port/PLAN.md 4.5).
//!
//! It follows the Intel prover (`kernels/intel/groth16_intel.hpp`, `Prover` constructor and
//! `Prover::prove`), which follows the CUDA prover step for step:
//!
//! | step | here | Intel / CUDA |
//! |---|---|---|
//! | key, coefficients, fuzzed results | [MetalProver::new]: [Srs::load], [Coeffs::load] + [Coeffs::validate], [FuzzedResults::load] | `Srs`, `Coeffs`, `read_fuzzed_results` |
//! | the ChaCha fuzz (once per key) | `chacha_fuzz` | `generate_fuzz` / `generate_random_scalars` |
//! | a, b = witness into the QAP polynomials; c = a * b | `witness_into_poly` x2, `fr_mul` | "witness to polynomials" |
//! | a, b, c to the coset; h = (a b - c).from_mont(); w += f | [NttPlan::encode_to_coset] x3, `fr_mul_sub_from_mont`, `fr_add` | "NTTs" (`NTT_sequence`) |
//! | H = MSM(H, h) | sort h, MSM G1 | "MSM H" |
//! | A, B1, C = MSM(., w + f), B2 = MSM(B2, w + f) | one sort of w + f, MSM G1 x3, MSM G2 | "witness sort", "MSM A/B1/B2/C" |
//! | remove the fuzz, blind with r, s | host, arkworks | host |
//! | `proof.json`, `public.json` | [Proof::write_files] | `write_proof_file`, CUDA `write_public_file` |
//!
//! Command buffers: one proof is 4 command buffers on one queue, committed back to back and
//! waited once (polynomials + NTTs; MSM H; the G1 MSMs of the witness; the G2 MSM), so the GPU
//! never idles between stages and each stage's GPU time is measured. With
//! [ProveOptions::profile] every stage gets its own command buffer (finer timings, a little
//! slower). Pipelines (PSOs) are built once per process ([kernels]) and survive
//! [crate::metal_release]; the NTT plan's pipelines are built with the plan (once per key load).
//! `RISC0_G16_METAL_PROFILE_MSM=1` (with the profile) also splits each MSM into its dispatches.
//!
//! Item G9 optimizations ([Tuning], every one on by default, each with a revert switch:
//! `RISC0_G16_METAL_G9=0` for the G7/G8 kernels, `RISC0_G16_METAL_G9_OFF=wide_mul,wide_ntt,booth`
//! for single ones): the MSM and NTT kernels over bn254_fast.h's 64-bit-accumulate Montgomery
//! multiplication, and Booth MSM digits (15 windows). All bit-exact (`g9_tuning_bit_exact`).
//!
//! Item G12 adds two MSM switches to [Tuning] (revert: `RISC0_G16_METAL_G12=0`, or
//! `RISC0_G16_METAL_G12_OFF=lenperm,lent_groups`): `lenperm`, the bucket threads take the pieces
//! of a window longest first (balanced SIMD lanes; -184 ms per proof), and `lent_groups`, the
//! window groups are sized by the lent a, b buffer, so MSM B2 is one group (-19 ms). Same
//! results, bit for bit.
//!
//! Memory (rzup component: 5,635,930 witness elements, domain 2^23; device allocation 3890 MiB
//! with the item G10 [MemOptions], 5624 MiB without): key points 2232 MiB (mapped from the zkey
//! without a copy: file-backed pages, outside the process' physical footprint), coefficients
//! 309 MiB compact (1375 MiB full), a, b 512 MiB (c lives in the MSM scratch's `sorted` buffer
//! during the NTT phase, and a, b hold the regenerated fuzz and the MSM pieces once MSM H's sort
//! has consumed h; without `alias` a, b, c 768 MiB + fuzz 172 MiB + pieces 240 MiB), witness
//! 172 MiB, NTT tables 128 MiB, MSM scratch 536 MiB. Everything is allocated at construction
//! (nothing during a proof), after one budget check of the whole footprint
//! ([super::check_working_set]); the smallest working-set budget that builds it is ~3.84 GiB.

use std::{
    fs::File,
    io::Read as _,
    path::{Path, PathBuf},
    sync::OnceLock,
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, ensure, Context as _, Result};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{AffineRepr as _, CurveGroup as _};
use ark_ff::{BigInt, PrimeField as _, Zero as _};
use metal::{
    Buffer, BufferRef, CommandBuffer, CommandBufferRef, CommandQueue, CommandQueueRef,
    ComputeCommandEncoder, ComputeCommandEncoderRef, Device, DeviceRef, Library,
    MTLCommandBufferStatus, NSRange,
};

use super::{
    check_working_set,
    coeffs::{Coeffs, FuzzedResults, FuzzedResultsArk},
    compact::CompactCoeffs,
    msm::{MsmBuf, MsmConfig, MsmCurve, MsmKernels, MsmScratch, MsmStage, MsmTune, G1, G2},
    ntt::{NttKernels, NttPlan, NttSchedule},
    pipelines,
    small::{self, BufSlice, SmallKernels},
    srs::{self, SharedBuf, Srs, VerifyingKeyArk},
};
use crate::SetupParams;

const FR_BYTES: u64 = 32;
/// Allowance for page rounding of ~20 buffers and the NTT plan's pipelines in the footprint
/// check (measured: the plan without it was 0.3 MiB below the real growth).
const PLAN_SLACK: u64 = 32 << 20;

/// The process-wide GPU objects: device, queue, the groth16 library and the pipelines of the
/// small kernels and the MSM (built once; cold creation of the MSM pipelines takes up to minutes
/// the first time a metallib is seen, the OS shader cache makes later processes fast). By default
/// they come through the binary archive ([super::pipelines], item G11).
pub struct Kernels {
    pub device: Device,
    pub queue: CommandQueue,
    pub library: Library,
    pub small: SmallKernels,
    pub msm: MsmKernels,
    /// The NTT pipelines every plan shares (G11, archive on); `None` with
    /// `RISC0_G16_METAL_ARCHIVE=0`, where each plan builds its own (the pre-G11 behaviour).
    pub ntt: Option<NttKernels>,
    /// What the archive did (`None` with `RISC0_G16_METAL_ARCHIVE=0`).
    pub archive: Option<risc0_zkp::hal::metal::archive::ArchiveReport>,
    /// Wall time of building the library and pipelines.
    pub build_time: Duration,
}

impl Kernels {
    /// An NTT plan for 2^lg with the default schedule, from the shared pipelines if there are.
    pub fn ntt_plan(&self, lg: u32) -> Result<NttPlan> {
        match &self.ntt {
            Some(k) => NttPlan::with_kernels(
                &self.device,
                &self.queue,
                k.clone(),
                lg,
                NttSchedule::default(),
            ),
            None => NttPlan::new(&self.device, &self.queue, &self.library, lg),
        }
    }
}

/// The process-wide [Kernels], built on first use.
pub fn kernels() -> Result<&'static Kernels> {
    static KERNELS: OnceLock<std::result::Result<Kernels, String>> = OnceLock::new();
    KERNELS
        .get_or_init(|| build_kernels().map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow!("{e}"))
}

fn build_kernels() -> Result<Kernels> {
    super::metal_supported()?;
    let t0 = Instant::now();
    let device = Device::system_default().ok_or_else(|| anyhow!("no Metal device"))?;
    if !pipelines::use_archive() {
        // `RISC0_G16_METAL_ARCHIVE=0`: the pre-G11 loader (serial plain builds; NTT per plan).
        let library = device
            .new_library_with_data(crate::METAL_LIB)
            .map_err(|e| anyhow!("loading the groth16 metallib: {e}"))?;
        let small = SmallKernels::new(&device, &library)?;
        let msm = MsmKernels::new(&device, &library)?;
        let queue = device.new_command_queue();
        return Ok(Kernels {
            device,
            queue,
            library,
            small,
            msm,
            ntt: None,
            archive: None,
            build_time: t0.elapsed(),
        });
    }
    let archived = pipelines::build_archived(
        &device,
        risc0_zkp::hal::metal::archive::archive_dir(),
        crate::METAL_LIB,
    )
    .context("loading the groth16 metallib")?;
    let library = archived.library().to_owned();
    let src = pipelines::PsoSource::archived(&device, &archived);
    let small = SmallKernels::from_source(&src)?;
    let msm = MsmKernels::from_source(&device, &library, &src)?;
    let ntt = NttKernels::from_source(&src)?;
    let queue = device.new_command_queue();
    if std::env::var_os("RISC0_VERBOSE").is_some() {
        eprintln!(
            "[groth16-metal] pipelines: {}",
            pipelines::describe(&archived.report)
        );
    }
    Ok(Kernels {
        device,
        queue,
        library,
        small,
        msm,
        ntt: Some(ntt),
        archive: Some(archived.report.clone()),
        build_time: t0.elapsed(),
    })
}

/// How [MetalProver::prove] runs.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProveOptions {
    /// One command buffer per stage (finer per-stage GPU times).
    pub profile: bool,
    /// Print the stage timings to stderr (`[groth16-metal]` lines).
    pub verbose: bool,
    /// Optimization switches (item G9); the default is every optimization on.
    pub tune: Tuning,
}

impl ProveOptions {
    /// `RISC0_VERBOSE` set: verbose; `RISC0_G16_METAL_PROFILE=1`: profile (and verbose);
    /// [Tuning::from_env] for the optimization switches.
    pub fn from_env() -> Self {
        let profile = std::env::var("RISC0_G16_METAL_PROFILE").as_deref() == Ok("1");
        Self {
            profile,
            verbose: profile || std::env::var_os("RISC0_VERBOSE").is_some(),
            tune: Tuning::from_env(),
        }
    }
}

/// The optimization switches of item G9, each with a revert knob so that a baseline can A/B
/// them. `RISC0_G16_METAL_G9=0` turns every one off (the G7/G8 code paths);
/// `RISC0_G16_METAL_G9_OFF=<name>,<name>` turns off single ones (names: [Tuning::NAMES]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    /// `wide_mul`: the MSM point kernels with the 64-bit-accumulate Montgomery multiplication
    /// (bn254_fast.h).
    pub wide_mul: bool,
    /// `wide_ntt`: the NTT kernels (`ntt_*_w`) with the same multiplication.
    pub wide_ntt: bool,
    /// `booth`: signed MSM digits of 17 bits, 15 windows (the scalars are canonical Fr).
    pub booth: bool,
    /// `lenperm` (item G12): MSM bucket threads take the pieces longest first
    /// ([MsmTune::lenperm]). Revert: `RISC0_G16_METAL_G12=0` or `RISC0_G16_METAL_G12_OFF=lenperm`
    /// (the switch names are also accepted in `RISC0_G16_METAL_G9_OFF`).
    pub lenperm: bool,
    /// `lent_groups` (item G12): MSM window groups sized by the lent pieces buffer (a, b) rather
    /// than the 256 MiB cap ([MsmTune::lent_groups]; MSM B2 in one group). Revert:
    /// `RISC0_G16_METAL_G12_OFF=lent_groups`.
    pub lent_groups: bool,
}

impl Default for Tuning {
    fn default() -> Self {
        Self::all()
    }
}

impl Tuning {
    /// Switch names for `RISC0_G16_METAL_G9_OFF`.
    pub const NAMES: &'static [&'static str] =
        &["wide_mul", "wide_ntt", "booth", "lenperm", "lent_groups"];

    /// The MSM switches.
    pub fn msm(&self) -> MsmTune {
        MsmTune {
            wide_mul: self.wide_mul,
            booth: self.booth,
            lenperm: self.lenperm,
            lent_groups: self.lent_groups,
        }
    }

    /// Every optimization on.
    pub fn all() -> Self {
        Self {
            wide_mul: true,
            wide_ntt: true,
            booth: true,
            lenperm: true,
            lent_groups: true,
        }
    }

    /// Every optimization off: the G7/G8 prover.
    pub fn none() -> Self {
        Self {
            wide_mul: false,
            wide_ntt: false,
            booth: false,
            lenperm: false,
            lent_groups: false,
        }
    }

    /// Turn the switch `name` on or off; false for an unknown name.
    pub fn set(&mut self, name: &str, on: bool) -> bool {
        match name {
            "wide_mul" => self.wide_mul = on,
            "wide_ntt" => self.wide_ntt = on,
            "booth" => self.booth = on,
            "lenperm" => self.lenperm = on,
            "lent_groups" => self.lent_groups = on,
            _ => return false,
        }
        true
    }

    /// `none()` for "base", `all()` for "all", else `all()` with "-name" items off and "+name"
    /// items on (comma separated, applied in order to `all()`, or to `none()` after "base,").
    pub fn parse(spec: &str) -> Option<Self> {
        let mut t = Self::all();
        for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match item {
                "base" => t = Self::none(),
                "all" => t = Self::all(),
                _ => {
                    let (on, name) = match item.as_bytes()[0] {
                        b'-' => (false, &item[1..]),
                        b'+' => (true, &item[1..]),
                        _ => (true, item),
                    };
                    if !t.set(name, on) {
                        return None;
                    }
                }
            }
        }
        Some(t)
    }

    /// The switches from the environment (see [Tuning]); unknown names are ignored with a
    /// warning on stderr.
    pub fn from_env() -> Self {
        if std::env::var("RISC0_G16_METAL_G9").as_deref() == Ok("0") {
            return Self::none();
        }
        let mut t = Self::all();
        for var in ["RISC0_G16_METAL_G9_OFF", "RISC0_G16_METAL_G12_OFF"] {
            if let Ok(off) = std::env::var(var) {
                for name in off.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    if !t.set(name, false) {
                        eprintln!("[groth16-metal] {var}: unknown switch {name}");
                    }
                }
            }
        }
        // Item G12's switches, all off.
        if std::env::var("RISC0_G16_METAL_G12").as_deref() == Ok("0") {
            t.lenperm = false;
            t.lent_groups = false;
        }
        t
    }
}

/// The memory switches of item G10 (how the prover is built), each with a revert knob:
/// `RISC0_G16_METAL_G10=0` turns every one off (the G9 layout), `RISC0_G16_METAL_G10_OFF=<name>,..`
/// single ones (names: [MemOptions::NAMES]). All keep the proof bit-exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemOptions {
    /// `compact`: the compact coefficients ([CompactCoeffs], disk cached): ~0.3 GB instead of
    /// the 1.44 GB `coeff_t` array.
    pub compact: bool,
    /// `mmap`: the zkey's points (and the compact cache file) as no-copy buffers over read-only
    /// file mappings (page cache, not anonymous memory; no copy at load).
    pub mmap: bool,
    /// `alias`: no persistent fuzz (regenerated per proof into a, b once MSM H's sort has
    /// consumed h) and the third polynomial c in the MSM scratch's `sorted` buffer during the NTT
    /// phase, and the MSM piece buffer in a, b (free once MSM H's sort has consumed h): -668 MiB
    /// at the rzup sizes.
    pub alias: bool,
}

impl Default for MemOptions {
    fn default() -> Self {
        Self::all()
    }
}

impl MemOptions {
    /// Switch names for `RISC0_G16_METAL_G10_OFF`.
    pub const NAMES: &'static [&'static str] = &["compact", "mmap", "alias"];

    pub fn all() -> Self {
        Self {
            compact: true,
            mmap: true,
            alias: true,
        }
    }

    /// The G9 layout.
    pub fn none() -> Self {
        Self {
            compact: false,
            mmap: false,
            alias: false,
        }
    }

    /// Turn the switch `name` on or off; false for an unknown name.
    pub fn set(&mut self, name: &str, on: bool) -> bool {
        match name {
            "compact" => self.compact = on,
            "mmap" => self.mmap = on,
            "alias" => self.alias = on,
            _ => return false,
        }
        true
    }

    /// Like [Tuning::parse]: "base" (= none), "all", "-name", "+name", comma separated.
    pub fn parse(spec: &str) -> Option<Self> {
        let mut t = Self::all();
        for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match item {
                "base" => t = Self::none(),
                "all" => t = Self::all(),
                _ => {
                    let (on, name) = match item.as_bytes()[0] {
                        b'-' => (false, &item[1..]),
                        b'+' => (true, &item[1..]),
                        _ => (true, item),
                    };
                    if !t.set(name, on) {
                        return None;
                    }
                }
            }
        }
        Some(t)
    }

    /// The switches from the environment (see [MemOptions]).
    pub fn from_env() -> Self {
        if std::env::var("RISC0_G16_METAL_G10").as_deref() == Ok("0") {
            return Self::none();
        }
        let mut t = Self::all();
        if let Ok(off) = std::env::var("RISC0_G16_METAL_G10_OFF") {
            for name in off.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                if !t.set(name, false) {
                    eprintln!("[groth16-metal] RISC0_G16_METAL_G10_OFF: unknown switch {name}");
                }
            }
        }
        t
    }
}

/// The coefficients in the form [MemOptions::compact] selects.
pub enum ProverCoeffs {
    Full(Coeffs),
    Compact(CompactCoeffs),
}

/// A borrowed [ProverCoeffs].
#[derive(Clone, Copy)]
pub enum CoeffsRef<'a> {
    Full(&'a Coeffs),
    Compact(&'a CompactCoeffs),
}

impl ProverCoeffs {
    pub fn as_ref(&self) -> CoeffsRef<'_> {
        match self {
            Self::Full(c) => CoeffsRef::Full(c),
            Self::Compact(c) => CoeffsRef::Compact(c),
        }
    }

    /// (coefficients of A, of B, index entries of A, of B).
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        match self {
            Self::Full(c) => (c.count_a, c.count_b, c.ind_a, c.ind_b),
            Self::Compact(c) => (c.count_a, c.count_b, c.ind_a, c.ind_b),
        }
    }

    /// Payload bytes in GPU buffers.
    pub fn resident_bytes(&self) -> u64 {
        match self {
            Self::Full(c) => c.resident_bytes(),
            Self::Compact(c) => c.payload_bytes(),
        }
    }
}

/// A [SharedBuf] as an MSM point argument (at its offset: mapped sections start inside a page).
fn pts<T: srs::Pod>(b: &SharedBuf<T>) -> MsmBuf<'_> {
    MsmBuf::at(b.as_buf(), b.offset())
}

/// A [SharedBuf] (plus `extra` bytes) as a kernel argument.
fn slice_of<T: srs::Pod>(b: &SharedBuf<T>, extra: u64) -> BufSlice<'_> {
    BufSlice::at(b.as_buf(), b.offset() + extra)
}

/// What building a [MetalProver] cost.
#[derive(Clone, Debug, Default)]
pub struct BuildStats {
    /// Pipelines of [kernels] (zero if already built in this process).
    pub kernels: Duration,
    pub srs_load: Duration,
    pub coeffs_load: Duration,
    pub coeffs_validate: Duration,
    /// NTT plan (its pipelines and twiddle tables) + fuzz generation + scratch allocation.
    pub gpu_setup: Duration,
    pub total: Duration,
    /// Device `currentAllocatedSize` before loading and once built (process-wide).
    pub allocated_before: u64,
    pub allocated_after: u64,
    /// The footprint checked against the working-set budget before loading.
    pub planned_bytes: u64,
    /// The memory switches the prover was built with (item G10).
    pub mem: MemOptions,
    /// Process physical footprint (`proc_pid_rusage`) before loading and once built.
    pub footprint_before: u64,
    pub footprint_after: u64,
}

/// What one proof cost.
#[derive(Clone, Debug, Default)]
pub struct ProveStats {
    /// (stage names, GPU milliseconds) per command buffer, in order.
    pub gpu_ms: Vec<(String, f64)>,
    /// Copying the witness into the GPU buffer.
    pub upload_ms: f64,
    /// Encoding and committing every command buffer.
    pub encode_ms: f64,
    /// Waiting for the GPU after the last commit.
    pub wait_ms: f64,
    /// Window combination, fuzz removal and blinding on the host.
    pub host_ms: f64,
    /// The whole [MetalProver::prove].
    pub total_ms: f64,
    /// Device `currentAllocatedSize` after the proof (nothing is allocated during one).
    pub allocated: u64,
}

impl ProveStats {
    /// Sum of the command buffers' GPU times.
    pub fn gpu_total_ms(&self) -> f64 {
        self.gpu_ms.iter().map(|(_, t)| t).sum()
    }
}

/// The proof's MSM results with the fuzz removed and before blinding: A' = MSM(A, w),
/// B1' = MSM(B1, w), B2' = MSM(B2, w), C' = MSM(C, w[wn - |C|..]), H = MSM(H, h). They depend only
/// on the proving key and the witness, not on the blinding scalars or the fuzz, so they are the
/// values to compare with another backend (CUDA, Intel) proving the same seal when a proof does
/// not verify. Printed with `RISC0_VERBOSE` ([PreBlinding::dump]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreBlinding {
    pub a: G1Affine,
    pub b1: G1Affine,
    pub b2: G2Affine,
    pub c: G1Affine,
    pub h: G1Affine,
}

impl PreBlinding {
    /// One line per point, coordinates as canonical `0x` hex (the CUDA prover's `operator<<`
    /// form); `infinity` for the identity.
    pub fn dump(&self) -> String {
        fn hex<F: ark_ff::PrimeField>(v: F) -> String {
            let b = v.into_bigint();
            let l = b.as_ref();
            format!(
                "0x{}",
                l.iter()
                    .rev()
                    .map(|w| format!("{w:016x}"))
                    .collect::<String>()
            )
        }
        let g1 = |name: &str, p: &G1Affine| match p.xy() {
            Some((x, y)) => format!("{name}=({}, {})", hex(x), hex(y)),
            None => format!("{name}=infinity"),
        };
        let b2 = match self.b2.xy() {
            Some((x, y)) => format!(
                "B2'=(({}, {}), ({}, {}))",
                hex(x.c0),
                hex(x.c1),
                hex(y.c0),
                hex(y.c1)
            ),
            None => "B2'=infinity".into(),
        };
        [
            g1("A'", &self.a),
            g1("B1'", &self.b1),
            b2,
            g1("C'", &self.c),
            g1("H", &self.h),
        ]
        .join("\n")
    }
}

/// A Groth16 proof and its public inputs.
#[derive(Clone, Debug)]
pub struct Proof {
    pub a: G1Affine,
    pub b: G2Affine,
    pub c: G1Affine,
    /// The witness' public signals (`witness[1..=num_public]`).
    pub public: Vec<Fr>,
    /// The MSM results before blinding (diagnostics).
    pub pre_blinding: PreBlinding,
    pub stats: ProveStats,
}

impl Proof {
    /// `proof.json` in the CUDA/Intel prover's format (snarkjs-style decimal coordinates; pi_b's
    /// pairs are (c0, c1)).
    pub fn proof_json(&self) -> String {
        let (a, b, c) = (&self.a, &self.b, &self.c);
        format!(
            "{{ \"pi_a\":[\"{}\",\"{}\",\"1\"],  \"pi_b\": [[\"{}\",\"{}\"],[\"{}\",\"{}\"], \
             [\"1\",\"0\"]],  \"pi_c\": [\"{}\",\"{}\",\"1\"],  \"protocol\":\"groth16\" }}",
            dec(a.x),
            dec(a.y),
            dec(b.x.c0),
            dec(b.x.c1),
            dec(b.y.c0),
            dec(b.y.c1),
            dec(c.x),
            dec(c.y),
        )
    }

    /// `public.json`: the public signals as a JSON array of decimal strings (as the CUDA
    /// prover's `write_public_file`).
    pub fn public_json(&self) -> String {
        let vals: Vec<String> = self.public.iter().map(|v| format!("\"{v}\"")).collect();
        format!("[{}]", vals.join(","))
    }

    /// Write [Self::proof_json] and [Self::public_json].
    pub fn write_files(&self, proof_path: &Path, public_path: &Path) -> Result<()> {
        std::fs::write(proof_path, self.proof_json())
            .with_context(|| format!("failed to write the proof to {}", proof_path.display()))?;
        std::fs::write(public_path, self.public_json()).with_context(|| {
            format!(
                "failed to write the public inputs to {}",
                public_path.display()
            )
        })
    }
}

/// Decimal of a field element's canonical value (ark-ff prints `Fp` that way; zero is "0").
fn dec<F: std::fmt::Display>(v: F) -> String {
    v.to_string()
}

/// The Groth16 prover for one proving key: key, coefficients, fuzz, NTT plan, scratch, all on
/// the GPU. Built by [MetalProver::new]; [MetalProver::prove] proves.
pub struct MetalProver {
    paths: [PathBuf; 3],
    srs: Srs,
    coeffs: ProverCoeffs,
    fuzzed: FuzzedResultsArk,
    vk: VerifyingKeyArk,
    ntt: NttPlan,
    /// The ChaCha fuzz, `2 * fuzz_blocks(wn)` Fr (the first `wn` are used); `None` with
    /// [MemOptions::alias] (regenerated per proof at the start of `abc`).
    fuzz: Option<Buffer>,
    /// a, b, c over the domain (3 n Fr), or a, b (2 n Fr) with c in the MSM scratch
    /// ([MemOptions::alias]).
    abc: Buffer,
    /// c lives in the MSM scratch's `sorted` buffer (else at `2 n` in `abc`).
    c_in_scratch: bool,
    /// Blocks of the ChaCha fuzz (`fuzz_blocks(wn)`).
    fuzz_blocks: u64,
    mem: MemOptions,
    /// The witness, then witness + fuzz (wn Fr; shared: the host copies the witness in).
    witness: Buffer,
    /// witness_into_poly's fault word.
    fault: Buffer,
    /// The MSMs' window sums: H, A, B1, C (G1), then B2 (G2).
    wsums: Buffer,
    scratch: MsmScratch,
    wn: usize,
    n: usize,
    c_len: usize,
    num_public: usize,
    pub build_stats: BuildStats,
}

/// [PreBlinding] before the affine conversion.
struct PreBlindingProjective {
    a: G1Projective,
    b1: G1Projective,
    b2: G2Projective,
    c: G1Projective,
    h: G1Projective,
}

/// Window-sum slots of [MetalProver::wsums].
#[derive(Clone, Copy)]
enum Slot {
    H,
    A,
    B1,
    C,
    B2,
}

impl MetalProver {
    /// Load the key files of `setup` and build everything a proof needs, with the memory
    /// switches from the environment ([MemOptions::from_env]). Takes no GPU lock (see `super`
    /// docs); fails cleanly, before loading, if the whole footprint would exceed the working-set
    /// budget.
    pub fn new(setup: &SetupParams) -> Result<Self> {
        Self::new_with(setup, MemOptions::from_env())
    }

    /// [Self::new] with given memory switches.
    pub fn new_with(setup: &SetupParams, mem: MemOptions) -> Result<Self> {
        let t0 = Instant::now();
        let kernels_built = KERNELS_BUILT.get().is_none();
        let k = kernels()?;
        let _ = KERNELS_BUILT.set(());
        let device: &DeviceRef = &k.device;
        let mut stats = BuildStats {
            kernels: if kernels_built {
                k.build_time
            } else {
                Duration::ZERO
            },
            allocated_before: device.current_allocated_size(),
            footprint_before: super::filemap::process_memory().phys_footprint,
            mem,
            ..Default::default()
        };
        let srs_path = setup.srs_path.as_path();
        let coeffs_path = setup.pcoeffs_path.as_path();
        let fres_path = setup.fres_path.as_path();

        // Sizes first (a few KB of reads), so that an over-budget machine fails before loading.
        let Layout {
            wn,
            n,
            lg,
            cap,
            cfg,
            fuzz_blocks,
            fuzz_len,
            wsum_bytes,
            ntt_bytes,
            c_in_scratch,
            fuzz_late,
            polys,
            pieces_lent,
            planned_bytes,
        } = layout(srs_path, coeffs_path, mem)?;
        stats.planned_bytes = planned_bytes;
        check_working_set(
            device,
            "the Groth16 prover (proving key, coefficients and scratch)",
            stats.planned_bytes,
        )?;

        let t = Instant::now();
        let srs = Srs::load_with(device, srs_path, mem.mmap)?;
        stats.srs_load = t.elapsed();
        ensure!(
            srs.a.len() == wn
                && srs.b_g1.len() == wn
                && srs.b_g2.len() == wn
                && srs.c.len() <= wn
                && srs.h.len() == n,
            "groth16-metal: the proving key's A, B1, B2, C or H does not match the witness \
             length {wn} / domain {n}"
        );
        let t = Instant::now();
        let coeffs = if mem.compact {
            // Validated inside (the original in full on a build, the compact form on a cache
            // hit).
            let c = CompactCoeffs::load(device, coeffs_path, wn, n, mem.mmap)?;
            stats.coeffs_load = t.elapsed();
            ProverCoeffs::Compact(c)
        } else {
            let c = Coeffs::load(device, coeffs_path)?;
            stats.coeffs_load = t.elapsed();
            let t = Instant::now();
            c.validate(wn, n)?;
            stats.coeffs_validate = t.elapsed();
            ProverCoeffs::Full(c)
        };
        let fuzzed = FuzzedResults::load(fres_path)?.to_ark()?;
        let vk = srs.header.vk.to_ark()?;

        let t = Instant::now();
        let alloc = |bytes: u64, private: bool, what: &str| -> Result<Buffer> {
            alloc_buffer(device, bytes, private, what)
        };
        let fuzz = if fuzz_late {
            None
        } else {
            Some(alloc(fuzz_len * FR_BYTES, true, "the fuzz")?)
        };
        let abc = alloc(polys * n as u64 * FR_BYTES, true, "a, b, c")?;
        let witness = alloc(wn as u64 * FR_BYTES, false, "the witness")?;
        let fault = alloc(16, false, "the fault word")?;
        let wsums = alloc(wsum_bytes, false, "the MSM window sums")?;
        check_working_set(device, "the NTT tables", ntt_bytes)?;
        let ntt = k.ntt_plan(lg)?;
        let mut scratch = MsmScratch::new_with(device, cap, cfg, pieces_lent.then_some(&*abc))?;
        ensure!(
            scratch.pieces_lent() == pieces_lent,
            "groth16-metal: MSM piece buffer layout"
        );
        // Every MSM scalar is a canonical Fr (< r < 2^254): the witness + fuzz (fr_add), h
        // (from_mont) and the fuzz itself (< 2^253, fuzz.metal).
        scratch.set_scalar_bits(254);
        if let Some(fuzz) = &fuzz {
            let state = fuzz_state();
            run_one(&k.queue, |enc| {
                k.small
                    .chacha_fuzz(enc, BufSlice::new(fuzz), &state, fuzz_blocks)
            })
            .context("generating the fuzz")?;
        }
        stats.gpu_setup = t.elapsed();
        stats.allocated_after = device.current_allocated_size();
        stats.footprint_after = super::filemap::process_memory().phys_footprint;
        stats.total = t0.elapsed();

        let c_len = srs.c.len();
        let num_public = srs.header.num_public;
        Ok(Self {
            paths: [
                srs_path.to_path_buf(),
                coeffs_path.to_path_buf(),
                fres_path.to_path_buf(),
            ],
            srs,
            coeffs,
            fuzzed,
            vk,
            ntt,
            fuzz,
            abc,
            c_in_scratch,
            fuzz_blocks,
            mem,
            witness,
            fault,
            wsums,
            scratch,
            wn,
            n,
            c_len,
            num_public,
            build_stats: stats,
        })
    }

    /// The memory switches this prover was built with.
    pub fn mem_options(&self) -> MemOptions {
        self.mem
    }

    /// The coefficients (full or compact).
    pub fn coeffs(&self) -> &ProverCoeffs {
        &self.coeffs
    }

    /// Whether this prover was built from `setup`'s files.
    pub fn is_for(&self, setup: &SetupParams) -> bool {
        self.paths[0] == setup.srs_path.as_path()
            && self.paths[1] == setup.pcoeffs_path.as_path()
            && self.paths[2] == setup.fres_path.as_path()
    }

    /// The witness length (`num_vars` of the key): [Self::prove] takes `32 *` this many bytes.
    pub fn witness_len(&self) -> usize {
        self.wn
    }

    /// The domain size.
    pub fn domain_size(&self) -> usize {
        self.n
    }

    /// Prove: `witness` is the circuit's full witness, [Self::witness_len] canonical
    /// little-endian 32-byte elements. The blinding scalars come from the OS.
    pub fn prove(&mut self, witness: &[u8], opts: &ProveOptions) -> Result<Proof> {
        let (r, s) = (random_fr()?, random_fr()?);
        self.prove_with_blinding(witness, opts, r, s)
    }

    /// [Self::prove] with given blinding scalars r, s (tests).
    pub fn prove_with_blinding(
        &mut self,
        witness: &[u8],
        opts: &ProveOptions,
        r: Fr,
        s: Fr,
    ) -> Result<Proof> {
        self.prove_from(Some(witness), opts, r, s)
    }

    /// The prover's shared witness buffer ([Self::witness_len] elements, each four
    /// little-endian u64 limbs of a canonical Fr), for the host to write the witness into
    /// before [Self::prove_in_place] (item G13: no host copy of the 180 MB witness, no upload).
    /// Not in use by the GPU between proofs (every proof waits for its command buffers).
    pub fn witness_buffer(&mut self) -> &mut [[u64; 4]] {
        // Safety: a shared buffer of wn * 32 bytes (alloc'd in new_with), page-aligned, owned
        // by self and borrowed mutably with it; the GPU does not use it outside a proof.
        unsafe { std::slice::from_raw_parts_mut(self.witness.contents() as *mut [u64; 4], self.wn) }
    }

    /// [Self::prove] of the witness the host wrote into [Self::witness_buffer].
    pub fn prove_in_place(&mut self, opts: &ProveOptions) -> Result<Proof> {
        let (r, s) = (random_fr()?, random_fr()?);
        self.prove_from(None, opts, r, s)
    }

    /// Prove the witness `witness` (copied into the shared buffer), or with `None` the one
    /// already in it.
    fn prove_from(
        &mut self,
        witness: Option<&[u8]>,
        opts: &ProveOptions,
        r: Fr,
        s: Fr,
    ) -> Result<Proof> {
        let t0 = Instant::now();
        let k = kernels()?;
        let (wn, n) = (self.wn, self.n);
        let copy = witness.is_some();
        let witness: &[u8] = match witness {
            Some(w) => w,
            // Safety: the shared witness buffer (wn * 32 bytes), idle between proofs; only read
            // here before the GPU work is encoded.
            None => unsafe {
                std::slice::from_raw_parts(
                    self.witness.contents() as *const u8,
                    wn * FR_BYTES as usize,
                )
            },
        };
        ensure!(
            witness.len() == wn * FR_BYTES as usize,
            "groth16-metal: witness of {} bytes, expected {wn} elements ({} bytes)",
            witness.len(),
            wn * FR_BYTES as usize
        );
        let public = (1..=self.num_public)
            .map(|i| {
                fr_from_canonical(&witness[32 * i..32 * (i + 1)])
                    .ok_or_else(|| anyhow!("groth16-metal: public signal {i} is not canonical"))
            })
            .collect::<Result<Vec<_>>>()?;

        // Every command buffer of the previous proof has completed (prove waits for all), so the
        // shared buffers are free for the host.
        let t = Instant::now();
        // Safety: shared buffers of at least these sizes, not in use by the GPU.
        unsafe {
            if copy {
                std::ptr::copy_nonoverlapping(
                    witness.as_ptr(),
                    self.witness.contents() as *mut u8,
                    witness.len(),
                );
            }
            std::ptr::write_bytes(self.fault.contents() as *mut u8, 0, 16);
        }
        let upload_ms = ms(t);

        let t = Instant::now();
        self.ntt.set_wide(opts.tune.wide_ntt);
        let mut run = Run::new(&k.queue, opts.profile);
        let small = &k.small;
        let msm = &k.msm;
        let nn = n as u64;
        let fr = |i: u64| i * FR_BYTES;
        let (abc, w, fault) = (&self.abc, &self.witness, &self.fault);
        // c: at 2 n in abc, or (item G10 alias) the MSM scratch's `sorted` buffer, which no sort
        // of this proof has written yet (sort_h is the first, after the NTT stage).
        let (c_buf, c_off): (&BufferRef, u64) = if self.c_in_scratch {
            (self.scratch.sorted_buffer(), 0)
        } else {
            (abc, 2 * nn * FR_BYTES)
        };

        run.fill_new(abc, 2 * nn * FR_BYTES);
        let coeffs = &self.coeffs;
        run.stage("witness_poly", 0, |enc| {
            encode_witness_poly(small, enc, coeffs.as_ref(), abc, nn, w, wn as u64, fault);
            small.fr_mul(
                enc,
                BufSlice::at(c_buf, c_off),
                BufSlice::fr_at(abc, 0),
                BufSlice::fr_at(abc, nn),
                nn,
            );
        });
        let ntt = &self.ntt;
        let fuzz_now = self.fuzz.as_ref();
        run.stage("ntt", 0, |enc| {
            ntt.encode_to_coset(enc, abc, 0);
            ntt.encode_to_coset(enc, abc, fr(nn));
            ntt.encode_to_coset(enc, c_buf, c_off);
            small.fr_mul_sub_from_mont(
                enc,
                BufSlice::fr_at(abc, 0),
                BufSlice::fr_at(abc, nn),
                BufSlice::at(c_buf, c_off),
                nn,
            );
            if let Some(fuzz) = fuzz_now {
                small.fr_add(enc, BufSlice::new(w), BufSlice::new(fuzz), wn as u64);
            }
        });

        let cfg = self.scratch.config();
        let scratch = &mut self.scratch;
        scratch.set_tune(opts.tune.msm());
        let srs = &self.srs;
        let wsums = &self.wsums;
        let out = |slot: Slot| MsmBuf::at(wsums, slot_offset(cfg, slot));
        run.stage("sort_h", 1, |enc| {
            msm.encode_sort(enc, scratch, MsmBuf::new(abc), n as u32)
        });
        let deep =
            opts.profile && std::env::var("RISC0_G16_METAL_PROFILE_MSM").as_deref() == Ok("1");
        let h_pts = pts(&srs.h);
        msm_stage::<G1>(
            &mut run,
            msm,
            scratch,
            "msm_h",
            1,
            h_pts,
            0,
            out(Slot::H),
            deep,
        );
        if fuzz_now.is_none() {
            // Item G10 alias: sort_h has consumed h, so a, b are free: regenerate the fuzz there
            // (same kernel and state as the build-time fuzz) and add it to the witness.
            let blocks = self.fuzz_blocks;
            run.stage("fuzz", 2, |enc| {
                small.chacha_fuzz(enc, BufSlice::new(abc), &fuzz_state(), blocks);
                small.fr_add(enc, BufSlice::new(w), BufSlice::new(abc), wn as u64);
            });
        }
        run.stage("sort_w", 2, |enc| {
            msm.encode_sort(enc, scratch, MsmBuf::new(w), wn as u32)
        });
        let a_pts = pts(&srs.a);
        msm_stage::<G1>(
            &mut run,
            msm,
            scratch,
            "msm_a",
            2,
            a_pts,
            0,
            out(Slot::A),
            deep,
        );
        let b1_pts = pts(&srs.b_g1);
        msm_stage::<G1>(
            &mut run,
            msm,
            scratch,
            "msm_b1",
            2,
            b1_pts,
            0,
            out(Slot::B1),
            deep,
        );
        let first = (wn - self.c_len) as u32;
        let c_pts = pts(&srs.c);
        msm_stage::<G1>(
            &mut run,
            msm,
            scratch,
            "msm_c",
            2,
            c_pts,
            first,
            out(Slot::C),
            deep,
        );
        let b2_pts = pts(&srs.b_g2);
        msm_stage::<G2>(
            &mut run,
            msm,
            scratch,
            "msm_b2",
            3,
            b2_pts,
            0,
            out(Slot::B2),
            deep,
        );
        let encode_ms = ms(t);
        let t = Instant::now();
        let gpu_ms = run.finish()?;
        let wait_ms = ms(t);

        // Safety: the command buffers completed; shared buffer.
        let fault_bits = unsafe { *(self.fault.contents() as *const u32) };
        ensure!(
            fault_bits == 0,
            "groth16-metal: witness_into_poly reported bad coefficient or witness data \
             (fault bits {fault_bits:#x})"
        );

        let t = Instant::now();
        let h = self.window_sum::<G1>(Slot::H);
        let msm_a = self.window_sum::<G1>(Slot::A);
        let msm_b1 = self.window_sum::<G1>(Slot::B1);
        let msm_c = self.window_sum::<G1>(Slot::C);
        let msm_b2 = self.window_sum::<G2>(Slot::B2);
        let pre = self.remove_fuzz(h, msm_a, msm_b1, msm_b2, msm_c);
        let (a, b, c) = self.finish_proof(&pre, r, s);
        let host_ms = ms(t);
        let pre_blinding = PreBlinding {
            a: pre.a.into_affine(),
            b1: pre.b1.into_affine(),
            b2: pre.b2.into_affine(),
            c: pre.c.into_affine(),
            h: pre.h.into_affine(),
        };
        if opts.verbose {
            eprintln!(
                "[groth16-metal] pre-blinding MSM results (fuzz removed):\n{}",
                pre_blinding.dump()
            );
        }

        let stats = ProveStats {
            gpu_ms,
            upload_ms,
            encode_ms,
            wait_ms,
            host_ms,
            total_ms: ms(t0),
            allocated: k.device.current_allocated_size(),
        };
        if opts.verbose {
            let stages: Vec<String> = stats
                .gpu_ms
                .iter()
                .map(|(name, t)| format!("{name}={t:.1}"))
                .collect();
            eprintln!(
                "[groth16-metal] gpu ms: {} (sum {:.1}); upload={:.1} encode={:.1} wait={:.1} \
                 host={:.1} total={:.1} ms; allocated={:.1} MiB",
                stages.join(" "),
                stats.gpu_total_ms(),
                stats.upload_ms,
                stats.encode_ms,
                stats.wait_ms,
                stats.host_ms,
                stats.total_ms,
                stats.allocated as f64 / (1u64 << 20) as f64
            );
        }
        Ok(Proof {
            a,
            b,
            c,
            public,
            pre_blinding,
            stats,
        })
    }

    /// Remove the fuzz: the fuzzed results file holds `-MSM(P, f)` for A, B1, B2 and C (H has no
    /// fuzz).
    fn remove_fuzz(
        &self,
        h: G1Projective,
        msm_a: G1Projective,
        msm_b1: G1Projective,
        msm_b2: G2Projective,
        msm_c: G1Projective,
    ) -> PreBlindingProjective {
        let f = &self.fuzzed;
        PreBlindingProjective {
            a: msm_a + f.a,
            b1: msm_b1 + f.b_g1,
            b2: msm_b2 + f.b_g2,
            c: msm_c + f.c,
            h,
        }
    }

    /// The Intel prover's last step: blind with r, s:
    /// A = A' + alpha + r delta, B = B2' + beta + s delta, B1 = B1' + beta + s delta (G1),
    /// C = C' + H + s A + r B1 - r s delta.
    fn finish_proof(
        &self,
        pre: &PreBlindingProjective,
        r: Fr,
        s: Fr,
    ) -> (G1Affine, G2Affine, G1Affine) {
        let vk = &self.vk;
        let delta1 = vk.delta_g1.into_group();
        let ra = pre.a + vk.alpha_g1 + delta1 * r;
        let rb2 = pre.b2 + vk.beta_g2 + vk.delta_g2.into_group() * s;
        let rb1 = pre.b1 + vk.beta_g1 + delta1 * s;
        let rc = pre.c + pre.h + ra * s + rb1 * r - delta1 * (r * s);
        (ra.into_affine(), rb2.into_affine(), rc.into_affine())
    }

    fn window_sum<C: MsmCurve>(&self, slot: Slot) -> C::Projective {
        let cfg = self.scratch.config();
        // Every sort of a proof has the same layout (same switches, canonical scalars).
        let (nwins, wbits) = self
            .scratch
            .sorted_layout()
            .expect("window_sum before a sort");
        let words = nwins as usize * 4 * C::COORD_WORDS;
        let off = slot_offset(cfg, slot) as usize;
        // Safety: the command buffers that wrote it completed; the slot lies inside the buffer
        // (wsum_bytes) and is 4-byte aligned.
        let w = unsafe {
            std::slice::from_raw_parts(
                (self.wsums.contents() as *const u8).add(off) as *const u32,
                words,
            )
        };
        C::combine_windows(w, wbits)
    }

    /// `MSM(P, f) + fuzzed` for P = A, B1, C (G1) and B2 (G2), f the fuzz (C from witness index
    /// `wn - c.len`): the fuzzed results file holds `-MSM(P, f)`, so all four are the identity
    /// when the fuzz, the MSM and the key agree. Returns (A, B1, C, B2).
    pub fn fuzz_check(
        &mut self,
    ) -> Result<(G1Projective, G1Projective, G1Projective, G2Projective)> {
        let k = kernels()?;
        let msm = &k.msm;
        let cfg = self.scratch.config();
        let (srs, wsums, scratch) = (&self.srs, &self.wsums, &mut self.scratch);
        let out = |slot: Slot| MsmBuf::at(wsums, slot_offset(cfg, slot));
        let first = (self.wn - self.c_len) as u32;
        let wn = self.wn as u32;
        // Without a persistent fuzz (item G10 alias) it is generated into a, b first, as a proof
        // does.
        let fuzz: &BufferRef = self.fuzz.as_deref().unwrap_or(&self.abc);
        let blocks = self.fuzz_blocks;
        let late = self.fuzz.is_none();
        run_one(&k.queue, |enc| {
            if late {
                k.small
                    .chacha_fuzz(enc, BufSlice::new(fuzz), &fuzz_state(), blocks);
            }
            msm.encode_sort(enc, scratch, MsmBuf::new(fuzz), wn);
            msm.encode_msm::<G1>(enc, scratch, pts(&srs.a), 0, out(Slot::A));
            msm.encode_msm::<G1>(enc, scratch, pts(&srs.b_g1), 0, out(Slot::B1));
            msm.encode_msm::<G1>(enc, scratch, pts(&srs.c), first, out(Slot::C));
            msm.encode_msm::<G2>(enc, scratch, pts(&srs.b_g2), 0, out(Slot::B2));
        })?;
        let f = &self.fuzzed;
        Ok((
            self.window_sum::<G1>(Slot::A) + f.a,
            self.window_sum::<G1>(Slot::B1) + f.b_g1,
            self.window_sum::<G1>(Slot::C) + f.c,
            self.window_sum::<G2>(Slot::B2) + f.b_g2,
        ))
    }

    /// The verifying key parsed from the zkey.
    pub fn verifying_key(&self) -> &VerifyingKeyArk {
        &self.vk
    }
}

/// The ChaCha state of the CUDA prover's fuzz (all-zero key and nonce).
fn fuzz_state() -> [u32; 16] {
    small::chacha_state(&[0; 32], &[0; 12])
}

/// The prover's buffer layout and device footprint for a key, from the files' headers (a few KB
/// of reads; nothing is loaded).
struct Layout {
    wn: usize,
    n: usize,
    lg: u32,
    cap: u32,
    cfg: MsmConfig,
    fuzz_blocks: u64,
    fuzz_len: u64,
    wsum_bytes: u64,
    ntt_bytes: u64,
    c_in_scratch: bool,
    fuzz_late: bool,
    polys: u64,
    pieces_lent: bool,
    /// The whole footprint, checked against the working-set budget before loading.
    planned_bytes: u64,
}

fn layout(srs_path: &Path, coeffs_path: &Path, mem: MemOptions) -> Result<Layout> {
    // Sizes first (a few KB of reads), so that an over-budget machine fails before loading.
    let header = srs::read_zkey_header(srs_path)?;
    let wn = header.witness_len();
    let n = header.domain_size;
    ensure!(
        n.is_power_of_two() && n >= 2,
        "groth16-metal: domain size {n} is not a power of two >= 2"
    );
    let lg = n.trailing_zeros();
    ensure!(
        wn > header.num_public && wn <= u32::MAX as usize && n <= u32::MAX as usize,
        "groth16-metal: witness length {wn} / domain {n} out of range"
    );
    let coeffs_bytes = planned_coeff_bytes(coeffs_path, mem.compact)?;
    let cap = wn.max(n) as u32;
    let cfg = msm_config_from_env();
    let fuzz_blocks = small::fuzz_blocks(wn as u64);
    let fuzz_len = 2 * fuzz_blocks;
    let wsum_bytes = wsum_bytes(cfg);
    let ntt_bytes = ((n as u64 / 2).max(1) + 4096 + 2 * (n as u64 >> 12).max(1)) * FR_BYTES;
    let scratch_sizes = MsmScratch::sizes_for(cap, cfg);
    // Item G10 aliasing: c in the scratch's `sorted` buffer (index 4 of the sizes) and the
    // fuzz regenerated per proof at the start of a, b, when they fit.
    let c_in_scratch = mem.alias && scratch_sizes[4] >= n as u64 * FR_BYTES;
    let fuzz_late = mem.alias && fuzz_len <= 2 * n as u64;
    let polys = if c_in_scratch { 2 } else { 3 };
    // ... and the MSM piece buffer in a, b (free from MSM H's sort to the end of the proof).
    let pieces_lent = mem.alias && scratch_sizes[5] <= polys * n as u64 * FR_BYTES;
    let own = (polys * n as u64 + wn as u64 + if fuzz_late { 0 } else { fuzz_len }) * FR_BYTES
        + wsum_bytes
        + 16;
    let planned_bytes =
        header.points_bytes() + coeffs_bytes + own + ntt_bytes + scratch_sizes.iter().sum::<u64>()
            - if pieces_lent { scratch_sizes[5] } else { 0 }
            + PLAN_SLACK;
    Ok(Layout {
        wn,
        n,
        lg,
        cap,
        cfg,
        fuzz_blocks,
        fuzz_len,
        wsum_bytes,
        ntt_bytes,
        c_in_scratch,
        fuzz_late,
        polys,
        pieces_lent,
        planned_bytes,
    })
}

/// The device footprint a [MetalProver] for `setup` would allocate with the memory switches
/// `mem` (what [MetalProver::new_with] checks against the working-set budget), without loading
/// anything. Item G13: the early key load decides with it whether the key fits beside the work
/// still to come.
pub fn planned_bytes(setup: &SetupParams, mem: MemOptions) -> Result<u64> {
    Ok(layout(setup.srs_path.as_path(), setup.pcoeffs_path.as_path(), mem)?.planned_bytes)
}

/// Bytes the coefficients will take on the GPU, from the file header (before loading): the
/// whole file, or the compact form's arrays (cs, rowc, indices, a table of at most 1 MiB, the
/// rzup key's is 0.5 MiB, and page rounding).
fn planned_coeff_bytes(path: &Path, compact: bool) -> Result<u64> {
    let file = File::open(path).with_context(|| format!("open(\"{}\")", path.display()))?;
    let size = file
        .metadata()
        .with_context(|| format!("stat(\"{}\")", path.display()))?
        .len();
    if !compact {
        return Ok(size);
    }
    let mut hdr = [0u8; 32];
    std::os::unix::fs::FileExt::read_exact_at(&file, &mut hdr, 0)
        .with_context(|| format!("reading \"{}\"", path.display()))?;
    let [ca, cb, ua, ub] = srs::limbs_le(&hdr);
    Ok((ca.saturating_add(cb)).saturating_mul(8)
        + (ua.saturating_add(ub)).saturating_mul(8)
        + (1 << 20)
        + 4 * 16384)
}

/// a and b (`abc` at 0 and n) from the witness and the coefficients (full or compact).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_witness_poly(
    small: &SmallKernels,
    enc: &ComputeCommandEncoderRef,
    coeffs: CoeffsRef,
    abc: &BufferRef,
    nn: u64,
    w: &BufferRef,
    wn: u64,
    fault: &BufferRef,
) {
    match coeffs {
        CoeffsRef::Full(coeffs) => {
            let (count_a, ind_a) = (coeffs.count_a as u64, coeffs.ind_a as u64);
            let cbuf = coeffs.coeffs.as_buf();
            let ibuf = coeffs.indices.as_buf();
            small.witness_into_poly(
                enc,
                BufSlice::fr_at(abc, 0),
                nn,
                BufSlice::new(cbuf),
                count_a,
                BufSlice::new(ibuf),
                ind_a - 1,
                BufSlice::new(w),
                wn,
                BufSlice::new(fault),
            );
            small.witness_into_poly(
                enc,
                BufSlice::fr_at(abc, nn),
                nn,
                BufSlice::at(cbuf, count_a * small::COEFF_BYTES),
                coeffs.count_b as u64,
                BufSlice::at(ibuf, ind_a * 4),
                coeffs.ind_b as u64 - 1,
                BufSlice::new(w),
                wn,
                BufSlice::new(fault),
            );
        }
        CoeffsRef::Compact(c) => {
            let (count_a, ind_a) = (c.count_a as u64, c.ind_a as u64);
            let ntab = c.ntab as u64;
            for (out, cs_off, count, ind_off, rows) in [
                (0, 0, count_a, 0, ind_a - 1),
                (
                    nn,
                    count_a * 8,
                    c.count_b as u64,
                    ind_a * 4,
                    c.ind_b as u64 - 1,
                ),
            ] {
                small.witness_into_poly_compact(
                    enc,
                    BufSlice::fr_at(abc, out),
                    nn,
                    slice_of(&c.cs, cs_off),
                    count,
                    slice_of(&c.indices, ind_off),
                    rows,
                    BufSlice::new(w),
                    wn,
                    BufSlice::new(fault),
                    slice_of(&c.table, 0),
                    ntab,
                    slice_of(&c.rowc, ind_off),
                );
            }
        }
    }
}

/// Marks that [kernels] was built by an earlier [MetalProver::new] (its build time is then not
/// part of a later prover's [BuildStats]).
static KERNELS_BUILT: OnceLock<()> = OnceLock::new();

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Bytes of the window-sum buffer: 4 G1 MSMs and one G2 MSM.
fn wsum_bytes(cfg: MsmConfig) -> u64 {
    cfg.nwins() as u64 * (4 * G1::XYZZ_BYTES + G2::XYZZ_BYTES)
}

fn slot_offset(cfg: MsmConfig, slot: Slot) -> u64 {
    let g1 = cfg.nwins() as u64 * G1::XYZZ_BYTES;
    match slot {
        Slot::H => 0,
        Slot::A => g1,
        Slot::B1 => 2 * g1,
        Slot::C => 3 * g1,
        Slot::B2 => 4 * g1,
    }
}

/// A new buffer, after the working-set budget and buffer-size checks (nil-checked).
fn alloc_buffer(device: &DeviceRef, bytes: u64, private: bool, what: &str) -> Result<Buffer> {
    super::new_buffer_checked(device, bytes, private, what)
}

/// An Fr from canonical little-endian bytes; `None` if not below the modulus.
fn fr_from_canonical(b: &[u8]) -> Option<Fr> {
    let mut l = [0u64; 4];
    for (i, limb) in l.iter_mut().enumerate() {
        *limb = u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap());
    }
    Fr::from_bigint(BigInt::new(l))
}

/// A uniformly random nonzero Fr from the OS (64 bytes reduced mod r).
fn random_fr() -> Result<Fr> {
    let mut f = File::open("/dev/urandom").context("open(/dev/urandom)")?;
    loop {
        let mut b = [0u8; 64];
        f.read_exact(&mut b).context("reading /dev/urandom")?;
        let v = Fr::from_le_bytes_mod_order(&b);
        if !v.is_zero() {
            return Ok(v);
        }
    }
}

/// The prover's MSM configuration: [MsmConfig::default] (c = 16, 256 MiB piece groups), or the
/// measurement knobs of item G9's sweeps: `RISC0_G16_METAL_MSM_WBITS` (1..=16) and
/// `RISC0_G16_METAL_MSM_PIECE_MIB` (the per-group piece-buffer cap), and item G12's
/// `RISC0_G16_METAL_MSM_SEG` (pieces per segment_reduce thread, default 64). Results are the same for
/// every value (the window layout is the scratch's); only speed and the scratch size change.
fn msm_config_from_env() -> MsmConfig {
    let mut cfg = MsmConfig::default();
    let num = |name: &str| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
    };
    if let Some(w) = num("RISC0_G16_METAL_MSM_WBITS") {
        if (1..=16).contains(&w) {
            cfg.wbits = w as u32;
        } else {
            eprintln!("[groth16-metal] RISC0_G16_METAL_MSM_WBITS={w} out of 1..=16, ignored");
        }
    }
    if let Some(m) = num("RISC0_G16_METAL_MSM_PIECE_MIB") {
        cfg.max_piece_bytes = m.max(1) << 20;
    }
    // Item G12's sweep: pieces per msm_segment_reduce thread.
    if let Some(sg) = num("RISC0_G16_METAL_MSM_SEG") {
        if (1..=4096).contains(&sg) {
            cfg.seg = sg as u32;
        } else {
            eprintln!("[groth16-metal] RISC0_G16_METAL_MSM_SEG={sg} out of 1..=4096, ignored");
        }
    }
    cfg
}

/// One MSM as a [Run] stage; with `deep` (profiling), every dispatch of it is its own stage
/// (`label.buckets0`, `label.segment_reduce0`, ..., `label.window_reduce`).
#[allow(clippy::too_many_arguments)]
fn msm_stage<C: MsmCurve>(
    run: &mut Run,
    msm: &MsmKernels,
    scratch: &MsmScratch,
    label: &str,
    group: u32,
    points: MsmBuf,
    first: u32,
    out: MsmBuf,
    deep: bool,
) {
    if !deep {
        run.stage(label, group, |enc| {
            msm.encode_msm::<C>(enc, scratch, points, first, out)
        });
        return;
    }
    let (mut nb, mut ns) = (0, 0);
    for stage in msm.msm_stages::<C>(scratch, points, first, out) {
        let name = match stage {
            MsmStage::Buckets(_) => {
                nb += 1;
                format!("{label}.buckets{}", nb - 1)
            }
            MsmStage::SegmentReduce(_) => {
                ns += 1;
                format!("{label}.segment_reduce{}", ns - 1)
            }
            MsmStage::WindowReduce(_) => format!("{label}.window_reduce"),
        };
        run.stage(&name, group, |enc| {
            msm.encode_msm_stage::<C>(enc, scratch, points, out, stage)
        });
    }
}

/// Encode `f` into one compute encoder of a new command buffer, commit and wait; errors if the
/// command buffer failed. Returns its GPU milliseconds.
fn run_one(queue: &CommandQueueRef, f: impl FnOnce(&ComputeCommandEncoderRef)) -> Result<f64> {
    let mut run = Run::new(queue, false);
    run.stage("one", 0, f);
    Ok(run.finish()?[0].1)
}

/// Command buffers of one proof: stages with the same group share a command buffer (one compute
/// encoder each); a new group (or every stage with `profile`) starts a new one. Each is committed
/// when the next starts, [Run::finish] waits for all of them.
struct Run<'q> {
    queue: &'q CommandQueueRef,
    profile: bool,
    cur: Option<(CommandBuffer, ComputeCommandEncoder, u32)>,
    labels: Vec<String>,
    done: Vec<CommandBuffer>,
}

impl<'q> Run<'q> {
    fn new(queue: &'q CommandQueueRef, profile: bool) -> Self {
        Self {
            queue,
            profile,
            cur: None,
            labels: Vec::new(),
            done: Vec::new(),
        }
    }

    fn close(&mut self) {
        if let Some((cb, enc, _)) = self.cur.take() {
            enc.end_encoding();
            cb.commit();
            self.done.push(cb);
        }
    }

    fn open(&mut self, label: &str, group: u32, fill: Option<(&BufferRef, u64)>) {
        self.close();
        let cb = self.queue.new_command_buffer().to_owned();
        if let Some((buf, len)) = fill {
            let blit = cb.new_blit_command_encoder();
            blit.fill_buffer(buf, NSRange::new(0, len), 0);
            blit.end_encoding();
        }
        let enc = cb.new_compute_command_encoder().to_owned();
        self.labels.push(label.to_string());
        self.cur = Some((cb, enc, group));
    }

    /// Start a new command buffer (group 0) whose first command zero-fills `len` bytes of `buf`.
    fn fill_new(&mut self, buf: &BufferRef, len: u64) {
        self.open("zero", 0, Some((buf, len)));
    }

    fn stage(&mut self, label: &str, group: u32, f: impl FnOnce(&ComputeCommandEncoderRef)) {
        match &self.cur {
            Some((_, _, g)) if *g == group && !self.profile => {
                let last = self.labels.last_mut().unwrap();
                if last == "zero" {
                    *last = label.to_string();
                } else {
                    last.push('+');
                    last.push_str(label);
                }
            }
            Some((_, _, g))
                if *g == group && self.profile && self.labels.last().unwrap() == "zero" =>
            {
                *self.labels.last_mut().unwrap() = format!("zero+{label}");
            }
            _ => self.open(label, group, None),
        }
        f(&self.cur.as_ref().unwrap().1);
    }

    /// Commit the open command buffer, wait for all, check their status; GPU ms per buffer.
    fn finish(mut self) -> Result<Vec<(String, f64)>> {
        self.close();
        self.wait_all();
        let mut out = Vec::with_capacity(self.done.len());
        for (cb, label) in self.done.iter().zip(&self.labels) {
            if cb.status() != MTLCommandBufferStatus::Completed {
                bail!(
                    "groth16-metal: command buffer `{label}` failed ({:?})",
                    cb.status()
                );
            }
            out.push((label.clone(), gpu_seconds(cb) * 1e3));
        }
        Ok(out)
    }
}

impl Run<'_> {
    fn wait_all(&self) {
        for cb in &self.done {
            cb.wait_until_completed();
        }
    }
}

/// A [Run] dropped without [Run::finish] (a stage panicked, e.g. one of the encoders' `assert!`s
/// on a programming error): end the open encoder (Metal aborts the process when a compute
/// encoder is released without `endEncoding`), never commit it, and wait for the command buffers
/// already committed, so that no GPU work still writes the prover's buffers once the panic has
/// released the prover cache (whose next user rebuilds it, see `lock_discarding_poisoned`).
impl Drop for Run<'_> {
    fn drop(&mut self) {
        if let Some((_cb, enc, _)) = self.cur.take() {
            enc.end_encoding();
        }
        self.wait_all();
    }
}

/// GPUEndTime - GPUStartTime of a completed command buffer (not wrapped by metal-rs 0.29).
// objc's `sel_impl!` checks `feature = "cargo-clippy"`, unknown to this crate's check-cfg.
#[allow(unexpected_cfgs)]
fn gpu_seconds(cb: &CommandBufferRef) -> f64 {
    use metal::objc::{msg_send, sel, sel_impl};
    let (start, end): (f64, f64) =
        unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
    end - start
}

/// The witness-polynomial encoder for other modules' tests.
#[cfg(test)]
pub(crate) mod tests_support {
    pub(crate) use super::{encode_witness_poly, CoeffsRef};
}

#[cfg(test)]
#[path = "prover_tests.rs"]
mod tests;
