// Copyright 2025 RISC Zero, Inc.
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

//! Z4: host dispatch for `batch_evaluate_any` (eval_u: eu2 trio + Horner fallback) and
//! `mix_poly_coeffs` (v2 / v1-with-table / table-less), ported from
//! `sys/kernels/zkp/cuda/ffi.cu` (`risc0_zkp_cuda_batch_evaluate_any`,
//! `risc0_zkp_cuda_mix_poly_coeffs`). Kernels: `sys/kernels/zkp/metal/{eval,mix}.metal`.
//!
//! Binding tables (index → argument), mirrored from the `.metal` headers:
//! * `batch_evaluate_any`: 0 out, 1 coeffs, 2 which, 3 xs, 4 [EvalAnyParams]; tg(0) = 16 B × 256.
//! * `eu2_pows`: 0 pw, 1 ps, 2 xs, 3 [Eu2Params].
//! * `eu2_main`: 0 partial, 1 coeffs, 2 which, 3 xs, 4 pw, 5 ps, 6 [Eu2Params].
//! * `eu2_reduce`: 0 out, 1 partial, 2 [Eu2Params].
//! * `mix_poly_coeffs{,_v1,_v2,_v2_dcids}`: 0 out, 1 in, 2 combos, 3 [MixParams]; tg(0) = table.

use metal::MTLSize;
use risc0_core::field::baby_bear::{BabyBearElem, BabyBearExtElem};

use super::{as_bytes, BufferImpl, Grid, KernelArg, MetalHal, MetalHash};
use crate::hal::Buffer as _;

/// Coefficients per thread per step in `eu2_main` (CUDA `EU_J`).
pub(crate) const EU_J: u32 = 16;
/// Threads per `eu2_main` / `eu2_pows` threadgroup (CUDA `EU_THREADS`).
pub(crate) const EU_THREADS: u32 = 256;
/// Smallest eu2 chunk: `deg` must be a multiple of this (CUDA `minChunk`).
pub(crate) const EU_MIN_CHUNK: u32 = EU_J * EU_THREADS;
/// CUDA's (non-HIP) `targetBlocks` for the S selection.
pub(crate) const EU2_TARGET_BLOCKS: u64 = 1024;
/// Threads per Horner threadgroup: CUDA's `getCachedSimpleConfig` block (maxThreads / 4).
pub(crate) const HORNER_THREADS: u32 = 256;

/// Threads per mix threadgroup (CUDA `getCachedSimpleConfig` block / `__launch_bounds__(256)`).
pub(crate) const MIX_THREADS: u64 = 256;
/// Largest `inputSize` whose v2 table (16 B mix power + 4 B combo id per input) fits 32 KB.
/// CUDA uses v2 up to 2048 (40 KB of shared memory); Metal has 32 KB.
pub(crate) const MIX_V2_MAX_INPUTS: usize = 32768 / 20;
/// Largest `inputSize` whose mix-power-only table (16 B per input) fits 32 KB.
pub(crate) const MIX_TABLE_MAX_INPUTS: usize = 32768 / 16;
/// Metal's threadgroup memory limit on every Apple GPU this port targets (D0: 32768).
const TG_LIMIT: u64 = 32768;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct EvalAnyParams {
    deg: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct Eu2Params {
    deg: u32,
    chunk: u32,
    s: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct MixParams {
    mix_start: [u32; 4],
    mix: [u32; 4],
    input_size: u32,
    count: u32,
    _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<EvalAnyParams>() == 16);
const _: () = assert!(std::mem::size_of::<Eu2Params>() == 16);
const _: () = assert!(std::mem::size_of::<MixParams>() == 48);
const _: () = assert!(MIX_V2_MAX_INPUTS == 1638 && MIX_TABLE_MAX_INPUTS == 2048);

fn ext_words(x: &BabyBearExtElem) -> [u32; 4] {
    bytemuck::pod_read_unaligned(as_bytes(std::slice::from_ref(x)))
}

/// Which eval_u implementation runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EvalPath {
    /// `batch_evaluate_any`: one threadgroup of 256 per evaluation point.
    Horner,
    /// `eu2_pows` + `eu2_main` over (S, nEvals) + `eu2_reduce`, `deg = s * chunk`.
    Eu2 { s: u32, chunk: u32 },
}

/// The ffi.cu gate and S selection: Horner for `nEvals == 0`, `deg < 4096`, `deg % 4096 != 0`
/// or coefficients not 16-byte aligned; else the smallest power-of-two S (≤ deg / 4096) with
/// `S * nEvals >= 1024`.
pub(crate) fn eval_plan(deg: u32, n_evals: u32, coeffs_addr: u64) -> EvalPath {
    if n_evals == 0
        || deg < EU_MIN_CHUNK
        || !deg.is_multiple_of(EU_MIN_CHUNK)
        || coeffs_addr & 15 != 0
    {
        return EvalPath::Horner;
    }
    let max_s = deg / EU_MIN_CHUNK;
    let mut s = 1u32;
    while s < max_s && (s as u64) * (n_evals as u64) < EU2_TARGET_BLOCKS {
        s <<= 1;
    }
    EvalPath::Eu2 { s, chunk: deg / s }
}

/// `Hal::batch_evaluate_any` body; `force` overrides [eval_plan] (tests, benchmarks).
pub(crate) fn batch_evaluate_any<MH: MetalHash>(
    hal: &MetalHal<MH>,
    coeffs: &BufferImpl<BabyBearElem>,
    poly_count: usize,
    which: &BufferImpl<u32>,
    xs: &BufferImpl<BabyBearExtElem>,
    out: &BufferImpl<BabyBearExtElem>,
    force: Option<EvalPath>,
) -> EvalPath {
    let po2 = crate::core::log2_ceil(coeffs.size() / poly_count);
    let deg = 1usize << po2;
    assert_eq!(poly_count * deg, coeffs.size());
    let n_evals = which.size();
    assert_eq!(xs.size(), n_evals);
    assert_eq!(out.size(), n_evals);
    assert!(deg <= u32::MAX as usize && n_evals <= u32::MAX as usize);
    let (deg, n_evals) = (deg as u32, n_evals as u32);

    let path = force.unwrap_or_else(|| eval_plan(deg, n_evals, coeffs.as_device_ptr()));
    if n_evals == 0 {
        return path;
    }
    match path {
        EvalPath::Horner => {
            let params = [EvalAnyParams { deg, _pad: [0; 3] }];
            let args = &[
                out.as_arg(),
                coeffs.as_arg(),
                which.as_arg(),
                xs.as_arg(),
                KernelArg::Bytes(as_bytes(&params)),
            ];
            let tg = HORNER_THREADS as u64 * 16;
            assert!(tg <= TG_LIMIT);
            let grid = Grid::Groups {
                groups: MTLSize::new(n_evals as u64, 1, 1),
                threads: MTLSize::new(HORNER_THREADS as u64, 1, 1),
            };
            hal.dispatch(hal.pipeline("batch_evaluate_any"), args, grid, &[tg]);
        }
        EvalPath::Eu2 { s, chunk } => {
            assert!(
                s >= 1
                    && chunk.is_multiple_of(EU_MIN_CHUNK)
                    && s as u64 * chunk as u64 == deg as u64
                    && coeffs.as_device_ptr() & 15 == 0,
                "eu2 needs deg = S * chunk, chunk % {EU_MIN_CHUNK} == 0, 16-B aligned coeffs \
                 (deg {deg}, S {s}, chunk {chunk})"
            );
            let n = n_evals as usize;
            let pw: BufferImpl<BabyBearExtElem> =
                hal.alloc_buffer("eu2_pw", n * EU_THREADS as usize);
            let ps: BufferImpl<BabyBearExtElem> = hal.alloc_buffer("eu2_ps", n * s as usize);
            let part: BufferImpl<BabyBearExtElem> = hal.alloc_buffer("eu2_part", n * s as usize);
            let params = [Eu2Params {
                deg,
                chunk,
                s,
                _pad: 0,
            }];
            let p = || KernelArg::Bytes(as_bytes(&params));
            let threads = MTLSize::new(EU_THREADS as u64, 1, 1);
            for name in ["eu2_main", "eu2_reduce"] {
                super::kernels::assert_simd_width(hal.pipeline(name), name);
            }
            hal.dispatch(
                hal.pipeline("eu2_pows"),
                &[pw.as_arg(), ps.as_arg(), xs.as_arg(), p()],
                Grid::Groups {
                    groups: MTLSize::new(n_evals as u64, 1, 1),
                    threads,
                },
                &[],
            );
            hal.dispatch(
                hal.pipeline("eu2_main"),
                &[
                    part.as_arg(),
                    coeffs.as_arg(),
                    which.as_arg(),
                    xs.as_arg(),
                    pw.as_arg(),
                    ps.as_arg(),
                    p(),
                ],
                Grid::Groups {
                    groups: MTLSize::new(s as u64, n_evals as u64, 1),
                    threads,
                },
                &[],
            );
            hal.dispatch(
                hal.pipeline("eu2_reduce"),
                &[out.as_arg(), part.as_arg(), p()],
                Grid::Groups {
                    groups: MTLSize::new(n_evals as u64, 1, 1),
                    threads: MTLSize::new(32, 1, 1),
                },
                &[],
            );
            // pw/ps/part drop here; the pool reuses them only after this batch completes.
        }
    }
    path
}

/// Which mix_poly_coeffs kernel runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MixKernel {
    /// `mix_poly_coeffs_v2`: powers + combo ids in threadgroup memory (inputSize ≤ 1638).
    V2,
    /// `mix_poly_coeffs_v2_dcids`: powers in threadgroup memory, combo ids read from device.
    V2DeviceCids,
    /// `mix_poly_coeffs_v1`: CUDA's original kernel, thread 0 fills the power table. Not
    /// selected by [mix_plan] (V2DeviceCids measured faster); kept for tests and benchmarks.
    #[allow(dead_code)]
    V1Table,
    /// `mix_poly_coeffs`: no table; each thread carries the running power.
    NoTable,
}

impl MixKernel {
    fn name(self) -> &'static str {
        match self {
            MixKernel::V2 => "mix_poly_coeffs_v2",
            MixKernel::V2DeviceCids => "mix_poly_coeffs_v2_dcids",
            MixKernel::V1Table => "mix_poly_coeffs_v1",
            MixKernel::NoTable => "mix_poly_coeffs",
        }
    }

    /// Threadgroup bytes for `input_size` inputs (0 for the table-less kernel).
    pub(crate) fn tg_bytes(self, input_size: usize) -> u64 {
        match self {
            MixKernel::V2 => input_size as u64 * 20,
            MixKernel::V2DeviceCids | MixKernel::V1Table => input_size as u64 * 16,
            MixKernel::NoTable => 0,
        }
    }
}

/// The kernel used for 1638 < inputSize ≤ 2048 (plan open question 4). Measured on the M5 Max at
/// count 2^18 (docs/metal-port/notes/Z4.md): v2 with combo ids read from device is ~1.6x faster
/// than v1-with-table (8.8 vs 14.4 ms at inputSize 2048), so it replaces v1 here.
pub(crate) const MIX_MID_KERNEL: MixKernel = MixKernel::V2DeviceCids;

/// ffi.cu's selection under Metal's 32 KB threadgroup limit.
pub(crate) fn mix_plan(input_size: usize) -> MixKernel {
    if input_size <= MIX_V2_MAX_INPUTS {
        MixKernel::V2
    } else if input_size <= MIX_TABLE_MAX_INPUTS {
        MIX_MID_KERNEL
    } else {
        MixKernel::NoTable
    }
}

/// `Hal::mix_poly_coeffs` body; `force` overrides [mix_plan] (tests, benchmarks).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mix_poly_coeffs<MH: MetalHash>(
    hal: &MetalHal<MH>,
    output: &BufferImpl<BabyBearExtElem>,
    mix_start: &BabyBearExtElem,
    mix: &BabyBearExtElem,
    input: &BufferImpl<BabyBearElem>,
    combos: &BufferImpl<u32>,
    input_size: usize,
    count: usize,
    force: Option<MixKernel>,
) -> MixKernel {
    let kernel = force.unwrap_or_else(|| mix_plan(input_size));
    assert!(input.size() >= input_size * count);
    assert!(combos.size() >= input_size);
    assert!(count <= u32::MAX as usize && input_size <= u32::MAX as usize);
    if count == 0 || input_size == 0 {
        return kernel;
    }
    let params = [MixParams {
        mix_start: ext_words(mix_start),
        mix: ext_words(mix),
        input_size: input_size as u32,
        count: count as u32,
        _pad: [0; 2],
    }];
    let args = &[
        output.as_arg(),
        input.as_arg(),
        combos.as_arg(),
        KernelArg::Bytes(as_bytes(&params)),
    ];
    if kernel == MixKernel::NoTable {
        hal.launch(kernel.name(), count as u64, args);
        return kernel;
    }
    let tg = kernel.tg_bytes(input_size);
    let limit = TG_LIMIT.min(hal.device.max_threadgroup_memory_length());
    assert!(
        tg <= limit,
        "{}: {tg} bytes of threadgroup memory for inputSize {input_size} exceed {limit}",
        kernel.name()
    );
    let grid = Grid::Groups {
        groups: MTLSize::new((count as u64).div_ceil(MIX_THREADS), 1, 1),
        threads: MTLSize::new(MIX_THREADS, 1, 1),
    };
    hal.dispatch(hal.pipeline(kernel.name()), args, grid, &[tg]);
    kernel
}

#[cfg(test)]
mod tests_z8a {
    //! Z8a review tests for the eval_u kernels.
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use risc0_core::field::{
        baby_bear::{BabyBearElem, BabyBearExtElem},
        Elem as _,
    };

    use super::*;
    use crate::hal::{metal::MetalHalSha256, Hal};

    /// Every kernel with cross-lane code reports 32-wide SIMD groups (the assumption checked by
    /// `kernels::assert_simd_width` at dispatch).
    #[test]
    fn simd_width_is_32() {
        let hal = MetalHalSha256::new();
        for name in [
            "eu2_main",
            "eu2_reduce",
            "ct_x_1",
            "ct_x_8",
            "ct_x_8_coalesced",
            "gs_x_1",
            "gs_x_8",
            "gs_x_8_coalesced",
            "mdiv_agg0_e8",
            "mdiv_agg0_e16",
            "mdiv_apply_e8",
            "mdiv_apply_e16",
            "mdiv_scan_32",
            "mdiv_scan_1024",
        ] {
            super::super::kernels::assert_simd_width(hal.pipeline(name), name);
        }
    }

    /// `eu2_main` alone, `Z8A_REPS` times (default 20) on the testutil shape (deg 2^16, 223 polys,
    /// 865 evals, S = 2), each partial checked against the CPU. Deterministic and exact in normal
    /// runs; with `MTL_SHADER_VALIDATION=1` (Metal GPU validation) most reps return a few wrong
    /// partials, see docs/metal-port/notes/Z8a.md. Ignored: a reproducer, not a gate.
    #[test]
    #[ignore]
    fn eu2_main_repeat() {
        let reps: usize = std::env::var("Z8A_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);
        let (deg, polys, n_evals, s) = (1usize << 16, 223usize, 865usize, 2u32);
        let chunk = deg as u32 / s;
        let mut rng = SmallRng::seed_from_u64(32);
        let coeffs: Vec<BabyBearElem> = (0..deg * polys)
            .map(|_| BabyBearElem::random(&mut rng))
            .collect();
        let which: Vec<u32> = (0..n_evals)
            .map(|_| rng.random_range(0..polys as u32))
            .collect();
        let xs: Vec<BabyBearExtElem> = (0..n_evals)
            .map(|_| BabyBearExtElem::random(&mut rng))
            .collect();
        let mut want = vec![BabyBearExtElem::ZERO; n_evals * s as usize];
        for e in 0..n_evals {
            let poly = &coeffs[which[e] as usize * deg..];
            for si in 0..s as usize {
                let base = si * chunk as usize;
                let mut acc = BabyBearExtElem::ZERO;
                for i in (0..chunk as usize).rev() {
                    acc = acc * xs[e] + poly[base + i];
                }
                want[e * s as usize + si] = acc * xs[e].pow(base);
            }
        }
        let hal = MetalHalSha256::new();
        let m_coeffs = hal.copy_from_elem("coeffs", &coeffs);
        let m_which = hal.copy_from_u32("which", &which);
        let m_xs = hal.copy_from_extelem("xs", &xs);
        let pw: BufferImpl<BabyBearExtElem> = hal.alloc_buffer("pw", n_evals * 256);
        let ps: BufferImpl<BabyBearExtElem> = hal.alloc_buffer("ps", n_evals * s as usize);
        let params = [Eu2Params {
            deg: deg as u32,
            chunk,
            s,
            _pad: 0,
        }];
        let p = || KernelArg::Bytes(as_bytes(&params));
        let threads = MTLSize::new(EU_THREADS as u64, 1, 1);
        hal.dispatch(
            hal.pipeline("eu2_pows"),
            &[pw.as_arg(), ps.as_arg(), m_xs.as_arg(), p()],
            Grid::Groups {
                groups: MTLSize::new(n_evals as u64, 1, 1),
                threads,
            },
            &[],
        );
        let mut bad_reps = 0;
        for rep in 0..reps {
            let part: BufferImpl<BabyBearExtElem> = hal.alloc_buffer("part", n_evals * s as usize);
            hal.dispatch(
                hal.pipeline("eu2_main"),
                &[
                    part.as_arg(),
                    m_coeffs.as_arg(),
                    m_which.as_arg(),
                    m_xs.as_arg(),
                    pw.as_arg(),
                    ps.as_arg(),
                    p(),
                ],
                Grid::Groups {
                    groups: MTLSize::new(s as u64, n_evals as u64, 1),
                    threads,
                },
                &[],
            );
            let got = part.to_vec();
            let bad = (0..got.len()).filter(|&i| got[i] != want[i]).count();
            if bad > 0 {
                bad_reps += 1;
                println!("rep {rep}: {bad} wrong partials");
            }
        }
        println!("eu2_main: {bad_reps}/{reps} reps with wrong partials");
        assert_eq!(bad_reps, 0);
    }
}
