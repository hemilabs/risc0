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

//! Multi-scalar multiplication over BN254 G1 and G2 on the GPU (item G6, docs/metal-port/PLAN.md
//! 4.5): `kernels/metal/msm_sort.metal` + `kernels/metal/msm_reduce.metal`.
//!
//! Pippenger with c-bit windows (c = 16 by default), the Intel prover's scheme
//! (`kernels/intel/groth16_intel.hpp` `msm_sort`/`msm_sorted`):
//!
//! | step | kernel(s) | Intel / CUDA counterpart |
//! |---|---|---|
//! | count digits per (window, bucket) | `msm_clear`, `msm_hist_count` | sort step 1 / CUDA `breakdown` + `sort` histogram |
//! | per-window prefix sums, bucket pieces | `msm_window_scan` | sort step 2 / CUDA grid-synced scan |
//! | counting-sort scatter of scalar indices | `msm_scatter` | sort step 3 / CUDA `sort` scatter |
//! | piece sums (affine adds into XYZZ) | `msm_buckets_{g1,g2}` | msm_sorted step 4 / CUDA `accumulate` |
//! | running-sum reduction per segment of pieces | `msm_segment_reduce_{g1,g2}` | msm_sorted step 5 / CUDA `integrate` |
//! | per-window tree sum | `msm_window_reduce_{g1,g2}` | msm_sorted step 6 (host) / CUDA `integrate_split` + `integrate_combine` |
//! | sum_w 2^(c w) S_w | host, [MsmCurve::combine_windows] | msm_sorted step 6 (host) / CUDA host collect |
//!
//! A sort depends on the scalars only, so MSMs over the same scalars share it (A, B1, B2 and C
//! of a proof): [MsmKernels::encode_sort] once, then [MsmKernels::encode_msm] per point set.
//! `first` lets an MSM use the sorted scalars from index `first` on (C's points start at
//! witness index `wn - c.len`, the Intel prover's `msm_sorted(..., first)`).
//!
//! Item G9 variants ([MsmTune], selected per sort with [MsmScratch::set_tune], default all on):
//! `wide_mul` runs the point kernels over bn254_fast.h's `FpF`/`Fp2L` (`_g1w`/`_g2w`: 64-bit
//! multiply-accumulate Montgomery products, ~1.6x faster bucket sums, same values bit for bit);
//! `booth` sorts signed digits of c + 1 bits into 2^c + 1 buckets, so that scalars below 2^254
//! ([MsmScratch::set_scalar_bits], the prover's canonical Fr) need 15 windows instead of 16 at
//! c = 16. The window sums then have the sort's layout ([MsmScratch::sorted_layout]).
//!
//! Load balance: every bucket is cut into pieces of at most `piece` points; one thread sums one
//! piece, and the segment reduction runs over pieces weighted by their bucket. A bucket holding a
//! large share of the scalars (digit 1 of window 0 for a witness of bits) therefore costs no more
//! than the same number of points spread over many buckets.
//!
//! Memory (scratch for capacity `n`, c = 16, 16 windows; bytes):
//! `hist`/`offset`/`piece_off` 3 x 4 MiB, `sorted` 64 n, `pieces` <= `max_piece_bytes` (window
//! groups are processed one after another when all windows do not fit; default 256 MiB),
//! `piece_bucket` 64 max_pieces, `partial` 16 x 256 x max_segs. See [MsmScratch::new].
//!
//! The wrappers only **encode** into a caller-provided compute encoder (default serial dispatch
//! type, tracked `StorageModeShared` buffers: consecutive dispatches see each other's writes);
//! the caller decides command-buffer boundaries. [MsmGpu] is a self-contained convenience
//! (own queue, one command buffer per call) used by the tests.
//!
//! Formats: points are affine, coordinates in Montgomery form (ark-ff's in-memory limbs, R =
//! 2^256), infinity = all-zero coordinates; scalars are canonical (NOT Montgomery) little-endian
//! 256-bit integers, any value accepted (the result is s * P for the integer s); the GPU result
//! is one `bn254::Xyzz` (X, Y, ZZ, ZZZ; ZZ = 0 for infinity) per window, combined on the host by
//! [MsmCurve::combine_windows].

use std::ffi::c_void;

use anyhow::{bail, ensure, Result};
use ark_bn254::{Fq, Fq2, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{AffineRepr, PrimeGroup};
use ark_ff::{AdditiveGroup, BigInt, Field, PrimeField};
use ark_std::Zero;
use metal::{
    Buffer, BufferRef, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device,
    DeviceRef, LibraryRef, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

/// Every MSM kernel entry point (all in `METAL_KERNEL_NAMES`).
pub const MSM_KERNEL_NAMES: &[&str] = &[
    "msm_clear",
    "msm_hist_count",
    "msm_window_scan",
    "msm_scatter",
    "msm_buckets_g1",
    "msm_buckets_g2",
    "msm_segment_reduce_g1",
    "msm_segment_reduce_g2",
    "msm_window_reduce_g1",
    "msm_window_reduce_g2",
    "msm_buckets_g1w",
    "msm_buckets_g2w",
    "msm_segment_reduce_g1w",
    "msm_segment_reduce_g2w",
    "msm_window_reduce_g1w",
    "msm_window_reduce_g2w",
];

/// Kernel-variant switches of an MSM (item G9). [MsmScratch::set_tune] sets them before a sort;
/// the MSMs over that sort use the same. The default is every optimization on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MsmTune {
    /// The `_g1w`/`_g2w` point kernels (bn254_fast.h: 64-bit-accumulate Montgomery
    /// multiplication) instead of `_g1`/`_g2` (bn254.h). Same results, bit for bit.
    pub wide_mul: bool,
    /// Signed (Booth) digits of c + 1 bits for the configured c ([MsmConfig::layout]):
    /// 2^c + 1 buckets per window and, for scalars below 2^254 ([MsmScratch::set_scalar_bits]),
    /// ceil(255 / (c + 1)) windows (15 instead of 16 at c = 16).
    pub booth: bool,
}

impl Default for MsmTune {
    fn default() -> Self {
        Self::all()
    }
}

impl MsmTune {
    /// Every optimization on.
    pub fn all() -> Self {
        Self {
            wide_mul: true,
            booth: true,
        }
    }

    /// The G6 kernels.
    pub fn none() -> Self {
        Self {
            wide_mul: false,
            booth: false,
        }
    }
}

/// Buffer indices of the MSM kernels (the tables at the top of the .metal files).
pub mod buf {
    pub mod clear {
        pub const HIST: u64 = 0;
        pub const PARAMS: u64 = 1;
    }
    pub mod hist_count {
        pub const SCALARS: u64 = 0;
        pub const HIST: u64 = 1;
        pub const PARAMS: u64 = 2;
    }
    pub mod window_scan {
        pub const HIST: u64 = 0;
        pub const OFFSET: u64 = 1;
        pub const PIECE_OFF: u64 = 2;
        pub const TOTALS: u64 = 3;
        pub const PARAMS: u64 = 4;
    }
    pub mod scatter {
        pub const SCALARS: u64 = 0;
        pub const HIST: u64 = 1;
        pub const SORTED: u64 = 2;
        pub const PARAMS: u64 = 3;
    }
    pub mod buckets {
        pub const POINTS: u64 = 0;
        pub const SORTED: u64 = 1;
        pub const OFFSET: u64 = 2;
        pub const HIST: u64 = 3;
        pub const PIECE_OFF: u64 = 4;
        pub const TOTALS: u64 = 5;
        pub const PIECES: u64 = 6;
        pub const PIECE_BUCKET: u64 = 7;
        pub const PARAMS: u64 = 8;
    }
    pub mod segment_reduce {
        pub const PIECES: u64 = 0;
        pub const PIECE_BUCKET: u64 = 1;
        pub const TOTALS: u64 = 2;
        pub const PARTIAL: u64 = 3;
        pub const PARAMS: u64 = 4;
    }
    pub mod window_reduce {
        pub const PARTIAL: u64 = 0;
        pub const TOTALS: u64 = 1;
        pub const WSUM: u64 = 2;
        pub const PARAMS: u64 = 3;
    }
}

/// `MsmParams` of `kernels/metal/msm_common.h` (field order and size must match).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MsmParams {
    pub n: u32,
    pub wbits: u32,
    pub nwins: u32,
    pub nbuckets: u32,
    pub piece: u32,
    pub max_pieces: u32,
    pub seg: u32,
    pub max_segs: u32,
    pub first: u32,
    pub npoints: u32,
    pub w0: u32,
    pub wcount: u32,
    pub booth: u32,
    pub _pad: [u32; 3],
}
const _: () = assert!(std::mem::size_of::<MsmParams>() == 64);

/// Threadgroup memory limit per threadgroup (plan 2.1; larger allocations fail silently).
const MAX_TG_BYTES: u64 = 32768;
/// Threadgroup width of the elementwise kernels.
const TG: u64 = 256;
/// Upper bound on the threadgroup width of msm_window_reduce.
const WINDOW_REDUCE_TG: u64 = 64;

/// Bytes of the largest bucket element (G2 Xyzz: 4 x Fp2).
const MAX_XYZZ_BYTES: u64 = 256;

/// A region of a Metal buffer starting `offset` bytes in.
#[derive(Clone, Copy)]
pub struct MsmBuf<'a> {
    pub buf: &'a BufferRef,
    pub offset: u64,
}

impl<'a> MsmBuf<'a> {
    pub fn new(buf: &'a BufferRef) -> Self {
        Self { buf, offset: 0 }
    }

    pub fn at(buf: &'a BufferRef, offset: u64) -> Self {
        Self { buf, offset }
    }

    fn check(&self, bytes: u64, what: &str) {
        let end = self.offset.checked_add(bytes);
        assert!(
            end.is_some_and(|e| e <= self.buf.length()),
            "{what}: {bytes} bytes at offset {} exceed the buffer ({} bytes)",
            self.offset,
            self.buf.length()
        );
    }
}

/// Tunables of the MSM. The defaults are the Intel prover's c = 16 with 64-piece segments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MsmConfig {
    /// Window width c, 1..=16.
    pub wbits: u32,
    /// Pieces per msm_segment_reduce thread.
    pub seg: u32,
    /// Cap on the `pieces` scratch buffer; the windows are processed in groups that fit.
    pub max_piece_bytes: u64,
}

impl Default for MsmConfig {
    fn default() -> Self {
        Self {
            wbits: 16,
            seg: 64,
            max_piece_bytes: 256 << 20,
        }
    }
}

impl MsmConfig {
    /// A configuration for MSMs of about `n` terms: the default (c = 16) from 2^16 terms on;
    /// below, c = max(4, floor(log2 n)) and shorter segments, so that sparse windows do not
    /// leave long serial chains (empty-bucket gaps) on a few GPU threads.
    pub fn for_size(n: u32) -> Self {
        let lg = 31 - n.max(1).leading_zeros();
        if lg >= 16 {
            return Self::default();
        }
        Self {
            wbits: lg.max(4),
            seg: 16,
            ..Self::default()
        }
    }

    pub fn nwins(&self) -> u32 {
        256u32.div_ceil(self.wbits)
    }

    pub fn nbuckets(&self) -> u32 {
        1 << self.wbits
    }

    /// Points per piece for `n` scalars: about twice the mean bucket occupancy (at least 16), so
    /// that pieces per window stay below ~1.5 x the bucket count.
    pub fn piece(&self, n: u32) -> u32 {
        ((2 * n as u64).div_ceil(self.nbuckets() as u64) as u32).max(16)
    }

    /// Bound on the pieces of one window for any MSM of at most `n` scalars, and
    /// non-decreasing in `n` (so a scratch sized for `cap_n` fits every `n <= cap_n`).
    ///
    /// The pieces of a window number at most ceil(n / piece(n)) + min(n, 2^c - 1) (full pieces
    /// plus one partial piece per non-empty bucket). ceil(n / piece(n)) itself is *not*
    /// monotone: piece(n) steps up by one at multiples of 2^(c-1), where it drops. It is bounded
    /// by ceil(min(n, 8 * 2^c) / 16), which is: for n <= 8 * 2^c, piece(n) = 16; above,
    /// piece(n) >= 2n / 2^c, so n / piece(n) <= 2^(c-1) = (8 * 2^c) / 16.
    pub fn max_pieces(&self, n: u32) -> u32 {
        let nb = self.nbuckets();
        let full = (n as u64).min(8 * nb as u64).div_ceil(16) as u32;
        full + n.min(nb - 1)
    }

    /// Bound on the segments of one window for any MSM of at most `n` scalars (non-decreasing).
    pub fn max_segs(&self, n: u32) -> u32 {
        self.max_pieces(n).div_ceil(self.seg)
    }

    /// The sort layout (window width c, windows, buckets per window) of a sort with the digit
    /// mode `booth`, for scalars below 2^`scalar_bits` (<= 256). Unsigned: c = wbits,
    /// ceil(256 / c) windows (whatever `scalar_bits`, as G6), 2^c buckets. Booth (item G9):
    /// c = wbits + 1, ceil((scalar_bits + 1) / c) windows (Booth needs s < 2^(W c - 1)),
    /// 2^wbits + 1 buckets (|d| <= 2^wbits; bucket 0 unused). Neither exceeds
    /// [Self::nwins] windows or [Self::nbuckets] + 1 buckets.
    pub fn layout(&self, booth: bool, scalar_bits: u32) -> (u32, u32, u32) {
        if booth {
            let c = self.wbits + 1;
            (c, (scalar_bits + 1).div_ceil(c), (1 << self.wbits) + 1)
        } else {
            (self.wbits, self.nwins(), self.nbuckets())
        }
    }

    /// [Self::piece] for `nb` buckets per window.
    pub fn piece_nb(n: u32, nb: u32) -> u32 {
        ((2 * n as u64).div_ceil(nb as u64) as u32).max(16)
    }

    /// [Self::max_pieces] for `nb` buckets per window (same bound: for n <= 8 nb the piece is
    /// 16; above, piece >= 2n / nb, so n / piece <= nb / 2). Non-decreasing in n and nb.
    pub fn max_pieces_nb(n: u32, nb: u32) -> u32 {
        let full = (n as u64).min(8 * nb as u64).div_ceil(16) as u32;
        full + n.min(nb - 1)
    }

    /// [Self::max_segs] for `nb` buckets per window.
    pub fn max_segs_nb(&self, n: u32, nb: u32) -> u32 {
        Self::max_pieces_nb(n, nb).div_ceil(self.seg)
    }

    /// The largest bucket count of either digit mode ([Self::layout]): the scratch's sizing.
    fn nbuckets_any(&self) -> u32 {
        self.nbuckets() + 1
    }

    fn validate(&self) {
        assert!(
            (1..=16).contains(&self.wbits),
            "MSM window width {} out of 1..=16",
            self.wbits
        );
        assert!(self.seg >= 1, "MSM segment length must be positive");
    }
}

/// A BN254 group the MSM kernels support.
pub trait MsmCurve: 'static {
    /// Kernel name suffix ("g1" / "g2").
    const NAME: &'static str;
    /// u32 words of one base-field coordinate (8 for Fp, 16 for Fp2).
    const COORD_WORDS: usize;
    /// Bytes of one affine point on the GPU.
    const AFFINE_BYTES: u64 = 2 * Self::COORD_WORDS as u64 * 4;
    /// Bytes of one Xyzz point on the GPU.
    const XYZZ_BYTES: u64 = 4 * Self::COORD_WORDS as u64 * 4;

    type Affine: Copy;
    type Projective: PrimeGroup;

    /// Appends the GPU form of `p` (Montgomery limbs; infinity = zeros).
    fn write_affine(p: &Self::Affine, out: &mut Vec<u32>);
    /// The point of a GPU Xyzz (`4 * COORD_WORDS` words).
    fn xyzz_to_projective(w: &[u32]) -> Self::Projective;

    /// sum_w 2^(wbits w) S_w for the window sums `w` written by [MsmKernels::encode_msm]
    /// (Horner from the top window).
    fn combine_windows(w: &[u32], wbits: u32) -> Self::Projective {
        let xyzz = 4 * Self::COORD_WORDS;
        let mut r = Self::Projective::zero();
        for win in w.chunks_exact(xyzz).rev() {
            for _ in 0..wbits {
                r.double_in_place();
            }
            r += Self::xyzz_to_projective(win);
        }
        r
    }
}

fn fq_words(v: &Fq, out: &mut Vec<u32>) {
    for l in v.0 .0 {
        out.push(l as u32);
        out.push((l >> 32) as u32);
    }
}

fn fq_from_words(w: &[u32]) -> Fq {
    let mut l = [0u64; 4];
    for (i, limb) in l.iter_mut().enumerate() {
        *limb = w[2 * i] as u64 | ((w[2 * i + 1] as u64) << 32);
    }
    Fq::new_unchecked(BigInt::new(l))
}

fn fq2_from_words(w: &[u32]) -> Fq2 {
    Fq2::new(fq_from_words(&w[..8]), fq_from_words(&w[8..16]))
}

/// BN254 G1 (coordinates in Fp).
pub struct G1;
/// BN254 G2 (coordinates in Fp2 = c0 + c1 u).
pub struct G2;

impl MsmCurve for G1 {
    const NAME: &'static str = "g1";
    const COORD_WORDS: usize = 8;
    type Affine = G1Affine;
    type Projective = G1Projective;

    fn write_affine(p: &G1Affine, out: &mut Vec<u32>) {
        if p.infinity {
            out.extend_from_slice(&[0; 16]);
        } else {
            fq_words(&p.x, out);
            fq_words(&p.y, out);
        }
    }

    fn xyzz_to_projective(w: &[u32]) -> G1Projective {
        let (x, y, zz, zzz) = (
            fq_from_words(&w[0..8]),
            fq_from_words(&w[8..16]),
            fq_from_words(&w[16..24]),
            fq_from_words(&w[24..32]),
        );
        let (Some(izz), Some(izzz)) = (zz.inverse(), zzz.inverse()) else {
            return G1Projective::default();
        };
        G1Affine::new_unchecked(x * izz, y * izzz).into_group()
    }
}

impl MsmCurve for G2 {
    const NAME: &'static str = "g2";
    const COORD_WORDS: usize = 16;
    type Affine = G2Affine;
    type Projective = G2Projective;

    fn write_affine(p: &G2Affine, out: &mut Vec<u32>) {
        if p.infinity {
            out.extend_from_slice(&[0; 32]);
        } else {
            for c in [&p.x.c0, &p.x.c1, &p.y.c0, &p.y.c1] {
                fq_words(c, out);
            }
        }
    }

    fn xyzz_to_projective(w: &[u32]) -> G2Projective {
        let (x, y, zz, zzz) = (
            fq2_from_words(&w[0..16]),
            fq2_from_words(&w[16..32]),
            fq2_from_words(&w[32..48]),
            fq2_from_words(&w[48..64]),
        );
        let (Some(izz), Some(izzz)) = (zz.inverse(), zzz.inverse()) else {
            return G2Projective::default();
        };
        G2Affine::new_unchecked(x * izz, y * izzz).into_group()
    }
}

/// The canonical little-endian words of an Fr, the MSM scalar format.
pub fn scalar_words(s: &Fr) -> [u32; 8] {
    let b = s.into_bigint();
    let mut w = [0u32; 8];
    for (i, l) in b.0.iter().enumerate() {
        w[2 * i] = *l as u32;
        w[2 * i + 1] = (*l >> 32) as u32;
    }
    w
}

struct CurvePsos {
    buckets: ComputePipelineState,
    segment_reduce: ComputePipelineState,
    window_reduce: ComputePipelineState,
}

/// The MSM pipeline states, built once from the groth16 metallib.
pub struct MsmKernels {
    clear: ComputePipelineState,
    hist_count: ComputePipelineState,
    window_scan: ComputePipelineState,
    scatter: ComputePipelineState,
    /// The G7/G8 point kernels (`RISC0_G16_METAL_G9_OFF=wide_mul`, the revert path): built on
    /// first use (item G10), so the default path does not pay their cold pipeline compilation.
    g1: std::sync::OnceLock<CurvePsos>,
    g2: std::sync::OnceLock<CurvePsos>,
    g1w: CurvePsos,
    g2w: CurvePsos,
    device: metal::Device,
    lib: metal::Library,
}

fn curve_psos(device: &DeviceRef, lib: &LibraryRef, g: &str) -> Result<CurvePsos> {
    Ok(CurvePsos {
        buckets: pso(device, lib, &format!("msm_buckets_{g}"))?,
        segment_reduce: pso(device, lib, &format!("msm_segment_reduce_{g}"))?,
        window_reduce: pso(device, lib, &format!("msm_window_reduce_{g}"))?,
    })
}

fn pso(device: &DeviceRef, lib: &LibraryRef, name: &str) -> Result<ComputePipelineState> {
    let f = lib
        .get_function(name, None)
        .map_err(|e| anyhow::anyhow!("MSM kernel {name} missing from the library: {e}"))?;
    device
        .new_compute_pipeline_state_with_function(&f)
        .map_err(|e| anyhow::anyhow!("MSM pipeline {name}: {e}"))
}

impl MsmKernels {
    pub fn new(device: &DeviceRef, lib: &LibraryRef) -> Result<Self> {
        let k = Self {
            clear: pso(device, lib, "msm_clear")?,
            hist_count: pso(device, lib, "msm_hist_count")?,
            window_scan: pso(device, lib, "msm_window_scan")?,
            scatter: pso(device, lib, "msm_scatter")?,
            g1: Default::default(),
            g2: Default::default(),
            g1w: curve_psos(device, lib, "g1w")?,
            g2w: curve_psos(device, lib, "g2w")?,
            device: device.to_owned(),
            lib: lib.to_owned(),
        };
        // `RISC0_G16_METAL_EAGER_PSOS=1`: build the revert-path pipelines now too (the G9
        // behaviour; also what a warm-up that must cover every switch wants).
        if std::env::var("RISC0_G16_METAL_EAGER_PSOS").as_deref() == Ok("1") {
            k.lazy("g1")?;
            k.lazy("g2")?;
        }
        Ok(k)
    }

    /// The lazily built pipelines of `g` ("g1" / "g2").
    fn lazy(&self, g: &str) -> Result<&CurvePsos> {
        let cell = if g == "g1" { &self.g1 } else { &self.g2 };
        if let Some(c) = cell.get() {
            return Ok(c);
        }
        let built = curve_psos(&self.device, &self.lib, g)?;
        Ok(cell.get_or_init(|| built))
    }

    fn curve<C: MsmCurve>(&self, tune: MsmTune) -> &CurvePsos {
        match (C::NAME, tune.wide_mul) {
            (g @ ("g1" | "g2"), false) => self
                .lazy(g)
                .unwrap_or_else(|e| panic!("building the {g} MSM pipelines: {e:#}")),
            ("g1", true) => &self.g1w,
            ("g2", true) => &self.g2w,
            (other, _) => unreachable!("unknown MSM curve {other}"),
        }
    }

    /// Sort the digits of `n` scalars (8 u32 words each, at `scalars`) into `sc`. Encodes only.
    pub fn encode_sort(
        &self,
        enc: &ComputeCommandEncoderRef,
        sc: &mut MsmScratch,
        scalars: MsmBuf,
        n: u32,
    ) {
        assert!(
            n <= sc.cap_n,
            "MSM of {n} scalars exceeds the scratch capacity {}",
            sc.cap_n
        );
        scalars.check(n as u64 * 32, "MSM scalars");
        let cfg = sc.cfg;
        let booth = sc.tune.booth;
        // Booth sorts keep the digit sign in the index's top bit.
        assert!(
            !booth || n < 1 << 31,
            "MSM of {n} scalars too large for Booth digits"
        );
        let (wbits, nwins, nbuckets) = cfg.layout(booth, sc.scalar_bits);
        let p = MsmParams {
            n,
            wbits,
            nwins,
            nbuckets,
            piece: MsmConfig::piece_nb(n, nbuckets),
            // Strides of the piece and partial arrays: the scratch's, whatever n (the dispatch
            // grids use the bound at n, which is no larger).
            max_pieces: sc.max_pieces,
            seg: cfg.seg,
            max_segs: sc.max_segs,
            booth: booth as u32,
            ..Default::default()
        };
        assert!(
            n.div_ceil(p.piece) + n.min(p.nbuckets - 1) <= MsmConfig::max_pieces_nb(n, nbuckets)
                && MsmConfig::max_pieces_nb(n, nbuckets) <= sc.max_pieces
                && cfg.max_segs_nb(n, nbuckets) <= sc.max_segs
                && nwins <= cfg.nwins()
                && nbuckets <= cfg.nbuckets_any(),
            "MSM piece bound of {n} scalars exceeds the scratch (sized for {})",
            sc.cap_n
        );
        let table = p.nwins as u64 * p.nbuckets as u64;

        enc.set_compute_pipeline_state(&self.clear);
        enc.set_buffer(buf::clear::HIST, Some(&sc.hist), 0);
        set_params(enc, buf::clear::PARAMS, &p);
        dispatch_1d(enc, &self.clear, table);

        if n > 0 {
            enc.set_compute_pipeline_state(&self.hist_count);
            enc.set_buffer(buf::hist_count::SCALARS, Some(scalars.buf), scalars.offset);
            enc.set_buffer(buf::hist_count::HIST, Some(&sc.hist), 0);
            set_params(enc, buf::hist_count::PARAMS, &p);
            dispatch_1d(enc, &self.hist_count, n as u64);
        }

        let ts = self.window_scan.max_total_threads_per_threadgroup().min(TG);
        let tg_bytes = 2 * ts * 4;
        assert!(tg_bytes <= MAX_TG_BYTES);
        enc.set_compute_pipeline_state(&self.window_scan);
        enc.set_buffer(buf::window_scan::HIST, Some(&sc.hist), 0);
        enc.set_buffer(buf::window_scan::OFFSET, Some(&sc.offset), 0);
        enc.set_buffer(buf::window_scan::PIECE_OFF, Some(&sc.piece_off), 0);
        enc.set_buffer(buf::window_scan::TOTALS, Some(&sc.totals), 0);
        set_params(enc, buf::window_scan::PARAMS, &p);
        enc.set_threadgroup_memory_length(0, tg_bytes);
        enc.dispatch_thread_groups(MTLSize::new(p.nwins as u64, 1, 1), MTLSize::new(ts, 1, 1));

        if n > 0 {
            enc.set_compute_pipeline_state(&self.scatter);
            enc.set_buffer(buf::scatter::SCALARS, Some(scalars.buf), scalars.offset);
            enc.set_buffer(buf::scatter::HIST, Some(&sc.hist), 0);
            enc.set_buffer(buf::scatter::SORTED, Some(&sc.sorted), 0);
            set_params(enc, buf::scatter::PARAMS, &p);
            dispatch_1d(enc, &self.scatter, n as u64);
        }
        sc.sorted_params = Some(p);
        sc.sorted_tune = sc.tune;
    }

    /// The MSM over the scalars of the last [Self::encode_sort] on `sc`, by window:
    /// `out[w]` (one `C` Xyzz per window, [MsmScratch::windows] of them) = sum over i in
    /// [first, n) of `digit_w(scalar[i]) * points[i - first]`. [MsmCurve::combine_windows] turns
    /// them into `sum_w 2^(c w) out[w]` on the host (the Intel prover's step 6; a serial chain of
    /// 256 doublings is ~10x slower on one GPU thread than in ark on the CPU).
    ///
    /// `points` must hold at least `n - first` affine points. Encodes only; `sc` must not be
    /// re-sorted before this command buffer ran.
    pub fn encode_msm<C: MsmCurve>(
        &self,
        enc: &ComputeCommandEncoderRef,
        sc: &MsmScratch,
        points: MsmBuf,
        first: u32,
        out: MsmBuf,
    ) {
        for stage in self.msm_stages::<C>(sc, points, first, out) {
            self.encode_msm_stage::<C>(enc, sc, points, out, stage);
        }
    }

    /// The dispatches of one MSM, in order (window groups of the pieces buffer, then the
    /// window reduction); [Self::encode_msm_stage] encodes one.
    pub fn msm_stages<C: MsmCurve>(
        &self,
        sc: &MsmScratch,
        points: MsmBuf,
        first: u32,
        out: MsmBuf,
    ) -> Vec<MsmStage> {
        let sp = sc
            .sorted_params
            .expect("encode_msm before encode_sort on this scratch");
        assert!(
            first <= sp.n,
            "MSM first index {first} beyond the {} sorted scalars",
            sp.n
        );
        let npoints = sp.n - first;
        points.check(npoints as u64 * C::AFFINE_BYTES, "MSM points");
        out.check(sp.nwins as u64 * C::XYZZ_BYTES, "MSM window sums");
        let mut p = MsmParams {
            first,
            npoints,
            ..sp
        };
        let mut stages = Vec::new();

        // Window groups that fit the pieces buffer.
        let group_bytes = p.max_pieces as u64 * C::XYZZ_BYTES;
        if p.max_pieces > 0 {
            let per_group = ((sc.pieces_bytes / group_bytes) as u32).clamp(1, p.nwins);
            assert!(
                per_group as u64 * group_bytes <= sc.pieces_bytes
                    && per_group as u64 * p.max_pieces as u64 * 4 <= sc.piece_bucket.length(),
                "MSM pieces scratch too small"
            );
            assert!(
                sp.nwins as u64 * p.max_segs as u64 * C::XYZZ_BYTES <= sc.partial.length(),
                "MSM partial scratch too small"
            );
            let mut w0 = 0;
            while w0 < p.nwins {
                p.w0 = w0;
                p.wcount = per_group.min(p.nwins - w0);
                stages.push(MsmStage::Buckets(p));
                stages.push(MsmStage::SegmentReduce(p));
                w0 += p.wcount;
            }
        }
        p.w0 = 0;
        p.wcount = p.nwins;
        stages.push(MsmStage::WindowReduce(p));
        stages
    }

    /// Encode one stage of [Self::msm_stages] (same `sc`, `points`, `out`).
    pub fn encode_msm_stage<C: MsmCurve>(
        &self,
        enc: &ComputeCommandEncoderRef,
        sc: &MsmScratch,
        points: MsmBuf,
        out: MsmBuf,
        stage: MsmStage,
    ) {
        let ps = self.curve::<C>(sc.sorted_tune);
        match stage {
            MsmStage::Buckets(p) => {
                enc.set_compute_pipeline_state(&ps.buckets);
                if p.npoints > 0 {
                    enc.set_buffer(buf::buckets::POINTS, Some(points.buf), points.offset);
                } else {
                    // No point is read (every index is >= npoints); the caller's buffer may be
                    // shorter than one point, which the validation layer rejects. Bind any
                    // buffer of at least one point instead.
                    enc.set_buffer(buf::buckets::POINTS, Some(&sc.pieces), 0);
                }
                enc.set_buffer(buf::buckets::SORTED, Some(&sc.sorted), 0);
                enc.set_buffer(buf::buckets::OFFSET, Some(&sc.offset), 0);
                enc.set_buffer(buf::buckets::HIST, Some(&sc.hist), 0);
                enc.set_buffer(buf::buckets::PIECE_OFF, Some(&sc.piece_off), 0);
                enc.set_buffer(buf::buckets::TOTALS, Some(&sc.totals), 0);
                enc.set_buffer(buf::buckets::PIECES, Some(&sc.pieces), 0);
                enc.set_buffer(buf::buckets::PIECE_BUCKET, Some(&sc.piece_bucket), 0);
                set_params(enc, buf::buckets::PARAMS, &p);
                let grid = MsmConfig::max_pieces_nb(p.n, p.nbuckets).min(p.max_pieces);
                dispatch_2d(enc, &ps.buckets, grid as u64, p.wcount as u64);
            }
            MsmStage::SegmentReduce(p) => {
                enc.set_compute_pipeline_state(&ps.segment_reduce);
                enc.set_buffer(buf::segment_reduce::PIECES, Some(&sc.pieces), 0);
                enc.set_buffer(buf::segment_reduce::PIECE_BUCKET, Some(&sc.piece_bucket), 0);
                enc.set_buffer(buf::segment_reduce::TOTALS, Some(&sc.totals), 0);
                enc.set_buffer(buf::segment_reduce::PARTIAL, Some(&sc.partial), 0);
                set_params(enc, buf::segment_reduce::PARAMS, &p);
                let grid = sc.cfg.max_segs_nb(p.n, p.nbuckets).min(p.max_segs);
                dispatch_2d(enc, &ps.segment_reduce, grid as u64, p.wcount as u64);
            }
            MsmStage::WindowReduce(p) => {
                let tw = window_reduce_width(&ps.window_reduce, C::XYZZ_BYTES);
                enc.set_compute_pipeline_state(&ps.window_reduce);
                enc.set_buffer(buf::window_reduce::PARTIAL, Some(&sc.partial), 0);
                enc.set_buffer(buf::window_reduce::TOTALS, Some(&sc.totals), 0);
                enc.set_buffer(buf::window_reduce::WSUM, Some(out.buf), out.offset);
                set_params(enc, buf::window_reduce::PARAMS, &p);
                enc.set_threadgroup_memory_length(0, tw * C::XYZZ_BYTES);
                enc.dispatch_thread_groups(
                    MTLSize::new(p.nwins as u64, 1, 1),
                    MTLSize::new(tw, 1, 1),
                );
            }
        }
    }
}

/// One dispatch of an MSM (see [MsmKernels::msm_stages]).
#[derive(Clone, Copy, Debug)]
pub enum MsmStage {
    /// msm_buckets for windows [w0, w0 + wcount).
    Buckets(MsmParams),
    /// msm_segment_reduce for windows [w0, w0 + wcount).
    SegmentReduce(MsmParams),
    /// msm_window_reduce over all windows, into the caller's window sums.
    WindowReduce(MsmParams),
}

/// The msm_window_reduce threadgroup width: a power of two, at most WINDOW_REDUCE_TG, the PSO
/// limit and what fits 32 KB of threadgroup memory.
fn window_reduce_width(pso: &ComputePipelineState, xyzz_bytes: u64) -> u64 {
    let cap = WINDOW_REDUCE_TG
        .min(pso.max_total_threads_per_threadgroup())
        .min(MAX_TG_BYTES / xyzz_bytes);
    let tw = 1u64 << (63 - cap.leading_zeros());
    assert!(tw >= 1 && tw * xyzz_bytes <= MAX_TG_BYTES);
    tw
}

fn set_params(enc: &ComputeCommandEncoderRef, index: u64, p: &MsmParams) {
    enc.set_bytes(
        index,
        std::mem::size_of::<MsmParams>() as u64,
        p as *const MsmParams as *const c_void,
    );
}

fn dispatch_1d(enc: &ComputeCommandEncoderRef, pso: &ComputePipelineState, n: u64) {
    if n == 0 {
        return;
    }
    let tg = pso.max_total_threads_per_threadgroup().min(TG);
    enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
}

fn dispatch_2d(enc: &ComputeCommandEncoderRef, pso: &ComputePipelineState, x: u64, y: u64) {
    if x == 0 || y == 0 {
        return;
    }
    let tg = pso.max_total_threads_per_threadgroup().min(TG);
    enc.dispatch_threads(MTLSize::new(x, y, 1), MTLSize::new(tg, 1, 1));
}

/// GPU scratch of the MSM, sized for at most `cap_n` scalars; reused across MSMs and sorts.
pub struct MsmScratch {
    cfg: MsmConfig,
    cap_n: u32,
    /// cfg.max_pieces(cap_n) / cfg.max_segs(cap_n): the per-window strides of `pieces`,
    /// `piece_bucket` and `partial` for every sort on this scratch.
    max_pieces: u32,
    max_segs: u32,
    hist: Buffer,
    offset: Buffer,
    piece_off: Buffer,
    totals: Buffer,
    sorted: Buffer,
    pieces: Buffer,
    /// Bytes of `pieces` the MSM uses (its configured size; `pieces` may be a larger buffer
    /// lent by the prover, item G10).
    pieces_bytes: u64,
    /// `pieces` is the prover's buffer, not allocated here.
    pieces_lent: bool,
    piece_bucket: Buffer,
    partial: Buffer,
    sorted_params: Option<MsmParams>,
    /// The switches of the next sort ([Self::set_tune]) and of the last one.
    tune: MsmTune,
    sorted_tune: MsmTune,
    /// The sorts' scalars are below 2^scalar_bits (Booth window count; default 256).
    scalar_bits: u32,
}

impl MsmScratch {
    /// The buffer sizes of a scratch for `cap_n` scalars, in allocation order: hist, offset,
    /// piece_off, totals, sorted, pieces, piece_bucket, partial.
    fn sizes(cap_n: u32, cfg: MsmConfig) -> [u64; 8] {
        // Sized for either digit mode ([MsmConfig::layout]): nwins() windows, nbuckets() + 1
        // buckets.
        let nwins = cfg.nwins() as u64;
        let nb = cfg.nbuckets_any();
        let table = nwins * nb as u64 * 4;
        let max_pieces = MsmConfig::max_pieces_nb(cap_n, nb) as u64;
        let max_segs = cfg.max_segs_nb(cap_n, nb) as u64;
        let group = max_pieces * MAX_XYZZ_BYTES;
        let groups = cfg
            .max_piece_bytes
            .checked_div(group)
            .map_or(1, |g| g.clamp(1, nwins));
        // At least one element of every kernel argument type (the validation layer checks
        // bound lengths even when no thread reads them).
        [
            table,
            table,
            table,
            nwins * 8,
            nwins * cap_n as u64 * 4,
            groups * group,
            // Sized for every window: G1 groups hold twice the windows of G2 groups.
            nwins * max_pieces * 4,
            nwins * max_segs * MAX_XYZZ_BYTES,
        ]
        .map(|b| b.max(MAX_XYZZ_BYTES))
    }

    /// The buffer sizes [MsmScratch::new] allocates, in allocation order: hist, offset,
    /// piece_off, totals, sorted, pieces, piece_bucket, partial.
    pub fn sizes_for(cap_n: u32, cfg: MsmConfig) -> [u64; 8] {
        cfg.validate();
        Self::sizes(cap_n, cfg)
    }

    /// Bytes [MsmScratch::new] allocates for `cap_n` scalars (for budgeting before allocating).
    pub fn bytes_for(cap_n: u32, cfg: MsmConfig) -> u64 {
        cfg.validate();
        Self::sizes(cap_n, cfg).iter().sum()
    }

    /// Allocate (StorageModePrivate: the host never reads these) for up to `cap_n` scalars.
    ///
    /// Honours the working-set budget `RISC0_METAL_MAX_WORKING_SET_BYTES`
    /// ([super::check_working_set]): fails, allocating nothing, if the whole scratch would take
    /// the device's `currentAllocatedSize` above it.
    pub fn new(device: &DeviceRef, cap_n: u32, cfg: MsmConfig) -> Result<Self> {
        Self::new_with(device, cap_n, cfg, None)
    }

    /// [Self::new], with `pieces` (item G10) a buffer of the caller's to use as the piece buffer
    /// instead of allocating one, if it holds at least the configured size (else one is
    /// allocated). The caller guarantees nothing else uses its first [Self::sizes_for]`[5]`
    /// bytes while an MSM on this scratch runs (the prover lends a, b once MSM H's sort
    /// consumed h).
    pub fn new_with(
        device: &DeviceRef,
        cap_n: u32,
        cfg: MsmConfig,
        pieces: Option<&BufferRef>,
    ) -> Result<Self> {
        cfg.validate();
        let sizes = Self::sizes(cap_n, cfg);
        let lent = pieces.filter(|b| b.length() >= sizes[5]);
        let own: u64 = sizes.iter().sum::<u64>() - if lent.is_some() { sizes[5] } else { 0 };
        super::check_working_set(device, "the MSM scratch", own)?;
        // Nil-checked (G8): a refused allocation is an error, not a null `Buffer`.
        // `RISC0_G16_METAL_SCRATCH_SHARED=1`: StorageModeShared instead (item G9's Shared vs
        // Private measurement; no difference found, Private stays the default).
        let private = std::env::var("RISC0_G16_METAL_SCRATCH_SHARED").as_deref() != Ok("1");
        let alloc = |bytes: u64, what: &str| -> Result<Buffer> {
            super::new_buffer_checked(device, bytes, private, &format!("the MSM {what}"))
        };
        let [hist, offset, piece_off, totals, sorted, pieces_size, piece_bucket, partial] = sizes;
        Ok(Self {
            cfg,
            cap_n,
            max_pieces: MsmConfig::max_pieces_nb(cap_n, cfg.nbuckets_any()),
            max_segs: cfg.max_segs_nb(cap_n, cfg.nbuckets_any()),
            hist: alloc(hist, "hist")?,
            offset: alloc(offset, "offset")?,
            piece_off: alloc(piece_off, "piece_off")?,
            totals: alloc(totals, "totals")?,
            sorted: alloc(sorted, "sorted")?,
            pieces: match lent {
                Some(b) => b.to_owned(),
                None => alloc(pieces_size, "pieces")?,
            },
            pieces_bytes: pieces_size,
            pieces_lent: lent.is_some(),
            piece_bucket: alloc(piece_bucket, "piece_bucket")?,
            partial: alloc(partial, "partial")?,
            sorted_params: None,
            tune: MsmTune::default(),
            sorted_tune: MsmTune::default(),
            scalar_bits: 256,
        })
    }

    /// Whether the piece buffer is the caller's ([Self::new_with]).
    pub fn pieces_lent(&self) -> bool {
        self.pieces_lent
    }

    /// The kernel variants of the following sorts and the MSMs over them (item G9).
    pub fn set_tune(&mut self, tune: MsmTune) {
        self.tune = tune;
    }

    pub fn tune(&self) -> MsmTune {
        self.tune
    }

    /// Declare that the following sorts' scalars are below 2^bits (1..=256), e.g. 254 for
    /// canonical Fr: Booth sorts then use fewer windows ([MsmConfig::layout]). Larger scalars
    /// would give wrong MSMs (the caller guarantees the bound).
    pub fn set_scalar_bits(&mut self, bits: u32) {
        assert!(
            (1..=256).contains(&bits),
            "scalar bits {bits} out of 1..=256"
        );
        self.scalar_bits = bits;
    }

    /// (windows, window width) of the last sort: the layout of [MsmKernels::encode_msm]'s
    /// window sums, for [MsmCurve::combine_windows].
    pub fn sorted_layout(&self) -> Option<(u32, u32)> {
        self.sorted_params.map(|p| (p.nwins, p.wbits))
    }

    pub fn config(&self) -> MsmConfig {
        self.cfg
    }

    /// The `sorted` buffer (digits/indices of the last sort). Item G10: the prover borrows it
    /// as the third polynomial (c) during the NTT phase, before any sort of the proof writes it.
    pub(crate) fn sorted_buffer(&self) -> &BufferRef {
        &self.sorted
    }

    pub fn capacity(&self) -> u32 {
        self.cap_n
    }

    /// Windows of the configuration: the number of Xyzz [MsmKernels::encode_msm] writes.
    pub fn windows(&self) -> u32 {
        self.cfg.nwins()
    }

    /// Total bytes of the scratch buffers this scratch allocated (a lent piece buffer is not
    /// counted).
    pub fn bytes(&self) -> u64 {
        let pieces = if self.pieces_lent {
            0
        } else {
            self.pieces.length()
        };
        [
            &self.hist,
            &self.offset,
            &self.piece_off,
            &self.totals,
            &self.sorted,
            &self.piece_bucket,
            &self.partial,
        ]
        .iter()
        .map(|b| b.length())
        .sum::<u64>()
            + pieces
    }
}

/// GPU time of one [MsmGpu] call, seconds (command-buffer GPUEndTime - GPUStartTime).
#[derive(Clone, Copy, Debug, Default)]
pub struct MsmTiming {
    pub sort: f64,
    pub msm: f64,
}

/// A self-contained MSM runner (own queue; one command buffer for the sort and one per MSM,
/// waited on). Used by the tests; the prover (G7) encodes into its own command buffers.
pub struct MsmGpu {
    pub device: Device,
    pub queue: CommandQueue,
    pub kernels: MsmKernels,
}

impl MsmGpu {
    /// The system default device and the embedded groth16 library; `None` without a device.
    pub fn new() -> Result<Option<Self>> {
        let Some(device) = Device::system_default() else {
            return Ok(None);
        };
        ensure!(!crate::METAL_LIB.is_empty(), "groth16 metallib is empty");
        let lib = device
            .new_library_with_data(crate::METAL_LIB)
            .map_err(|e| anyhow::anyhow!("loading the groth16 metallib: {e}"))?;
        let kernels = MsmKernels::new(&device, &lib)?;
        let queue = device.new_command_queue();
        Ok(Some(Self {
            device,
            queue,
            kernels,
        }))
    }

    /// A shared buffer holding `words` (at least 16 bytes).
    pub fn buffer(&self, words: &[u32]) -> Buffer {
        let bytes = std::mem::size_of_val(words) as u64;
        let b = self
            .device
            .new_buffer(bytes.max(16), MTLResourceOptions::StorageModeShared);
        if bytes > 0 {
            // Safety: the buffer has at least `bytes` bytes and is not in use by the GPU.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    words.as_ptr() as *const u8,
                    b.contents() as *mut u8,
                    bytes as usize,
                )
            };
        }
        b
    }

    fn run(&self, f: impl FnOnce(&ComputeCommandEncoderRef)) -> Result<f64> {
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        f(enc);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if cb.status() != MTLCommandBufferStatus::Completed {
            bail!("MSM command buffer failed: {:?}", cb.status());
        }
        Ok(gpu_seconds(cb))
    }

    /// Sort the scalars in `scalars` (from byte `offset`) into `sc`, waiting for the GPU.
    pub fn sort(&self, sc: &mut MsmScratch, scalars: MsmBuf, n: u32) -> Result<f64> {
        self.run(|enc| self.kernels.encode_sort(enc, sc, scalars, n))
    }

    /// One MSM over the scalars sorted last into `sc`, waiting for the GPU. With `profile`, each
    /// stage runs in its own command buffer and the returned time is their sum (per-stage times
    /// go to stderr).
    pub fn msm_sorted<C: MsmCurve>(
        &self,
        sc: &MsmScratch,
        points: MsmBuf,
        first: u32,
    ) -> Result<(C::Projective, f64)> {
        self.msm_sorted_impl::<C>(sc, points, first, false)
    }

    /// [Self::msm_sorted] with one command buffer per stage, per-stage GPU times on stderr.
    pub fn msm_sorted_profiled<C: MsmCurve>(
        &self,
        sc: &MsmScratch,
        points: MsmBuf,
        first: u32,
    ) -> Result<(C::Projective, f64)> {
        self.msm_sorted_impl::<C>(sc, points, first, true)
    }

    fn msm_sorted_impl<C: MsmCurve>(
        &self,
        sc: &MsmScratch,
        points: MsmBuf,
        first: u32,
        profile: bool,
    ) -> Result<(C::Projective, f64)> {
        let nw = sc.windows() as usize;
        let out = self.device.new_buffer(
            nw as u64 * C::XYZZ_BYTES,
            MTLResourceOptions::StorageModeShared,
        );
        let ob = MsmBuf::new(&out);
        let t = if profile {
            let mut total = 0.0;
            let mut line = String::new();
            for stage in self.kernels.msm_stages::<C>(sc, points, first, ob) {
                let t = self.run(|enc| {
                    self.kernels
                        .encode_msm_stage::<C>(enc, sc, points, ob, stage)
                })?;
                let name = match stage {
                    MsmStage::Buckets(_) => "buckets",
                    MsmStage::SegmentReduce(_) => "segment_reduce",
                    MsmStage::WindowReduce(_) => "window_reduce",
                };
                line += &format!(" {name} {:.2}", t * 1e3);
                total += t;
            }
            eprintln!("  msm {} stages (ms):{line}", C::NAME);
            total
        } else {
            self.run(|enc| self.kernels.encode_msm::<C>(enc, sc, points, first, ob))?
        };
        let (snw, wbits) = sc.sorted_layout().expect("sorted");
        // Safety: the command buffer completed; the buffer holds snw <= nw Xyzz.
        let w = unsafe {
            std::slice::from_raw_parts(
                out.contents() as *const u32,
                snw as usize * 4 * C::COORD_WORDS,
            )
        };
        Ok((C::combine_windows(w, wbits), t))
    }

    /// `sum_i scalars[i] * points[i]` (raw 256-bit scalars), allocating scratch for this call.
    pub fn msm_raw<C: MsmCurve>(
        &self,
        points: &[C::Affine],
        scalars: &[[u32; 8]],
        cfg: MsmConfig,
    ) -> Result<(C::Projective, MsmTiming)> {
        ensure!(
            points.len() == scalars.len(),
            "points and scalars differ in length"
        );
        let n = u32::try_from(scalars.len())?;
        let mut pw = Vec::with_capacity(points.len() * 4 * C::COORD_WORDS);
        for p in points {
            C::write_affine(p, &mut pw);
        }
        let sw: Vec<u32> = scalars.iter().flatten().copied().collect();
        let pb = self.buffer(&pw);
        let sb = self.buffer(&sw);
        let mut sc = MsmScratch::new(&self.device, n, cfg)?;
        let sort = self.sort(&mut sc, MsmBuf::new(&sb), n)?;
        let (r, msm) = self.msm_sorted::<C>(&sc, MsmBuf::new(&pb), 0)?;
        Ok((r, MsmTiming { sort, msm }))
    }

    /// `sum_i scalars[i] * points[i]` with [MsmConfig::for_size].
    pub fn msm<C: MsmCurve>(&self, points: &[C::Affine], scalars: &[Fr]) -> Result<C::Projective> {
        let sw: Vec<[u32; 8]> = scalars.iter().map(scalar_words).collect();
        Ok(self
            .msm_raw::<C>(points, &sw, MsmConfig::for_size(sw.len() as u32))?
            .0)
    }
}

/// GPUEndTime - GPUStartTime of a completed command buffer (not wrapped by metal-rs 0.29).
// objc's `sel_impl!` checks `feature = "cargo-clippy"`, unknown to this crate's check-cfg.
#[allow(unexpected_cfgs)]
fn gpu_seconds(cb: &metal::CommandBufferRef) -> f64 {
    use metal::objc::{msg_send, sel, sel_impl};
    let (start, end): (f64, f64) =
        unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
    end - start
}

#[cfg(test)]
#[path = "msm_tests.rs"]
mod tests;
