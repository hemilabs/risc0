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

//! Poseidon2 (BabyBear) hashing on Metal with the `pv.metal` kernels (plan item Z3): ports of
//! `supra/pv.cuh` (thread per permutation, HIP C++ branch) and `supra/pv_small.cuh` (`pvs::w`,
//! one SIMD group per permutation; `pvs::q`, four lanes per permutation). The host logic follows
//! `supra/api.cu` (`sppark_poseidon2_{rows,fold,fold_tree}`, `pvs::{rows,fold}`, HIP
//! `hip_pv_fold_tail`) with crossovers measured on the M5 Max (see `docs/metal-port/notes/Z3.md`).
//!
//! The classic `poseidon2.metal` kernels (the pre-port oracle) were retired in Z7.
//!
//! Environment (read once per HAL):
//! * `RISC0_METAL_POSEIDON2` = `z9` (default; Z9's reduction in the thread-per-hash kernels,
//!   `sredc64h` in pv.metal, 1.16-1.18x) | `pv` (pv.cuh's HIP int64 arithmetic, the pre-Z9
//!   default and Z9's revert knob) | `m32` (the products split into 32-bit words; A/B only).
//! * `RISC0_METAL_PV_SMALL=0`: never use the `pvs` latency kernels.
//! * `RISC0_METAL_PV_TAIL=0`: no single-dispatch fold-tree tail.
//! * `RISC0_METAL_PVS_{W,Q}_MAX_{FOLD,ROWS}`: override a crossover (largest `n` for that kernel).
//! * `RISC0_METAL_COMMIT_FUSE=0`: no Z14 fusion: a commit's coefficient bit reversal runs as its
//!   own `bit_rev_z_batched_64` pass instead of inside the Merkle row hash (`pv_rows_side_k_z9`),
//!   and `prefill_elem` sets nothing aside (`RISC0_METAL_PREFILL=0` turns off only the latter).

use std::sync::Mutex;

use metal::MTLSize;
use risc0_core::field::baby_bear::{BabyBear, BabyBearElem};

use super::{as_bytes, BufferImpl, Grid, KernelArg, MetalHal, MetalHash};
use crate::{
    core::{
        digest::Digest,
        hash::{poseidon2::Poseidon2HashSuite, HashSuite},
    },
    hal::Buffer as _,
};

/// Threads per threadgroup of every `pv.metal` kernel (`__launch_bounds__(128)` / `PV_BLOCK`).
const PV_BLOCK: u64 = 128;
/// `fold_tail_k`'s threadgroup size limit (api.cu `PV_TAIL`).
const PV_TAIL: u32 = 128;
/// The kernels that need a SIMD width of 32.
const PVS_KERNELS: [&str; 8] = [
    "pvs_w_fold_k",
    "pvs_w_rows_k",
    "pvs_q_fold_k",
    "pvs_q_rows_k",
    "pvs_w_fold_k_m32",
    "pvs_w_rows_k_m32",
    "pvs_q_fold_k_m32",
    "pvs_q_rows_k_m32",
];

/// Which Poseidon2 kernels hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Poseidon2Kernels {
    /// `pv.metal`, pv.cuh's HIP C++ (int64) arithmetic (default).
    Pv,
    /// `pv.metal` `_m32` instantiations: the same values from 32-bit mul/mulhi (A/B only;
    /// measured slower on the M5 Max).
    PvM32,
    /// Z9: `pv.metal` `_z9` instantiations of the thread-per-hash kernels (32x32 -> 64 product,
    /// mulhi reduction; the default); the `pvs` latency kernels use their `Pv` form.
    PvZ9,
}

/// Kernel selection and crossovers. A size `n` (outputs of a fold, rows of `hash_rows`) uses
/// `pvs_w` when `n <= w_max_*`, else `pvs_q` when `n <= q_max_*`, else the thread-per-hash `pv`
/// kernel (the same rule as `pvs::fold` / `pvs::rows` in pv_small.cuh).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poseidon2Config {
    pub kernels: Poseidon2Kernels,
    /// Use the `pvs` latency kernels at all (needs a SIMD width of 32).
    pub small: bool,
    pub w_max_fold: u32,
    pub q_max_fold: u32,
    pub w_max_rows: u32,
    pub q_max_rows: u32,
    /// `hash_fold_tree` folds every layer of at most this many outputs in one `pv_fold_tail_k`
    /// dispatch (0: off; at most 128).
    pub tail_max: u32,
    /// Z14: fold a commit's coefficient bit reversal and `prefill_elem` fills into the row hash
    /// (`pv_rows_side_k_z9`).
    pub commit_fuse: bool,
}

/// Measured crossovers (M5 Max, `tests_z3::bench_{fold,rows}_kernels`, notes/Z3.md): `pvs_w` is
/// 1-5 us (cols = 16) to ~80 us (cols = 211) faster than `pv` up to 2^8 permutations and slower
/// from 2^9 on; `pvs_q` is ~2-50 us faster than `pv` from 2^9 to 2^11 and slower from 2^12 on. The
/// same for folds and rows. CUDA's RTX 5090 values are W_MAX = 2048, Q_MAX = 8192.
pub const DEFAULT_W_MAX_FOLD: u32 = 256;
pub const DEFAULT_Q_MAX_FOLD: u32 = 2048;
pub const DEFAULT_W_MAX_ROWS: u32 = 256;
pub const DEFAULT_Q_MAX_ROWS: u32 = 2048;
/// Off: a 7-layer tree took 123.7 us with one `pv_fold_tail_k` dispatch for every layer (HIP's
/// default) against 118.3 us with one `pvs_w_fold_k` dispatch per layer (thread-per-hash latency
/// is ~22 us per layer, `pvs_w` ~17 us, a dispatch ~1-2 us; `tests_z3::bench_fold_tree`).
pub const DEFAULT_TAIL_MAX: u32 = 0;

impl Default for Poseidon2Config {
    fn default() -> Self {
        Self {
            kernels: Poseidon2Kernels::PvZ9,
            small: true,
            w_max_fold: DEFAULT_W_MAX_FOLD,
            q_max_fold: DEFAULT_Q_MAX_FOLD,
            w_max_rows: DEFAULT_W_MAX_ROWS,
            q_max_rows: DEFAULT_Q_MAX_ROWS,
            tail_max: DEFAULT_TAIL_MAX,
            commit_fuse: true,
        }
    }
}

impl Poseidon2Config {
    /// The defaults with the environment overrides applied.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        let var = |name: &str| {
            std::env::var(name)
                .ok()
                .map(|v| v.trim().to_ascii_lowercase())
        };
        match var("RISC0_METAL_POSEIDON2").as_deref() {
            Some("m32") => cfg.kernels = Poseidon2Kernels::PvM32,
            Some("z9") => cfg.kernels = Poseidon2Kernels::PvZ9,
            Some("pv") => cfg.kernels = Poseidon2Kernels::Pv,
            Some(other) => tracing::warn!("RISC0_METAL_POSEIDON2={other}: unknown, using default"),
            None => {}
        }
        if var("RISC0_METAL_PV_SMALL").as_deref() == Some("0") {
            cfg.small = false;
        }
        if var("RISC0_METAL_PV_TAIL").as_deref() == Some("0") {
            cfg.tail_max = 0;
        }
        if var("RISC0_METAL_COMMIT_FUSE").as_deref() == Some("0") {
            cfg.commit_fuse = false;
        }
        let num = |name: &str, dst: &mut u32| {
            if let Some(v) = var(name) {
                match v.parse() {
                    Ok(n) => *dst = n,
                    Err(_) => tracing::warn!("{name}={v}: not a number, ignored"),
                }
            }
        };
        num("RISC0_METAL_PVS_W_MAX_FOLD", &mut cfg.w_max_fold);
        num("RISC0_METAL_PVS_Q_MAX_FOLD", &mut cfg.q_max_fold);
        num("RISC0_METAL_PVS_W_MAX_ROWS", &mut cfg.w_max_rows);
        num("RISC0_METAL_PVS_Q_MAX_ROWS", &mut cfg.q_max_rows);
        num("RISC0_METAL_PV_TAIL_MAX", &mut cfg.tail_max);
        cfg.tail_max = cfg.tail_max.min(PV_TAIL);
        cfg
    }
}

/// The kernel a single `pv.metal` dispatch uses (for tests and measurements).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PvKernel {
    /// pv.cuh: one thread per permutation.
    Pv,
    /// pv_small.cuh `pvs::w`: one SIMD group per permutation.
    PvsW,
    /// pv_small.cuh `pvs::q`: four lanes per permutation.
    PvsQ,
}

impl PvKernel {
    fn name(self, op: &str, form: Poseidon2Kernels) -> String {
        let base = match self {
            PvKernel::Pv => "pv",
            PvKernel::PvsW => "pvs_w",
            PvKernel::PvsQ => "pvs_q",
        };
        let suffix = match (form, self) {
            (Poseidon2Kernels::PvM32, _) => "_m32",
            (Poseidon2Kernels::PvZ9, PvKernel::Pv) => "_z9",
            _ => "",
        };
        format!("{base}_{op}_k{suffix}")
    }

    /// Threads per permutation.
    fn lanes(self) -> u64 {
        match self {
            PvKernel::Pv => 1,
            PvKernel::PvsW => 32,
            PvKernel::PvsQ => 4,
        }
    }
}

pub struct MetalHashPoseidon2 {
    suite: HashSuite<BabyBear>,
    cfg: Mutex<Poseidon2Config>,
}

impl MetalHashPoseidon2 {
    /// The current kernel selection.
    pub fn config(&self) -> Poseidon2Config {
        *self.cfg.lock().unwrap()
    }

    /// Replace the kernel selection (A/B runs, tests).
    pub fn set_config(&self, cfg: Poseidon2Config) {
        assert!(
            cfg.tail_max <= PV_TAIL,
            "tail_max {} > {PV_TAIL}",
            cfg.tail_max
        );
        *self.cfg.lock().unwrap() = cfg;
    }

    fn fold_kernel(cfg: &Poseidon2Config, n: u32) -> PvKernel {
        if cfg.small && n <= cfg.w_max_fold {
            PvKernel::PvsW
        } else if cfg.small && n <= cfg.q_max_fold {
            PvKernel::PvsQ
        } else {
            PvKernel::Pv
        }
    }

    fn rows_kernel(cfg: &Poseidon2Config, rows: u32, cols: u32) -> PvKernel {
        // The pvs kernels index the matrix with 32-bit offsets.
        let small = cfg.small && (rows as u64) * (cols as u64) < 1 << 32;
        if small && rows <= cfg.w_max_rows {
            PvKernel::PvsW
        } else if small && rows <= cfg.q_max_rows {
            PvKernel::PvsQ
        } else {
            PvKernel::Pv
        }
    }

    /// Encode `out[i] = H(in[2i], in[2i + 1])`, `i < n`, with `kernel`; `out`/`input` are digest
    /// offsets into `io`.
    #[cfg(test)]
    pub(crate) fn encode_fold(
        hal: &MetalHal<Self>,
        kernel: PvKernel,
        m32: bool,
        io: &BufferImpl<Digest>,
        out: usize,
        input: usize,
        n: u32,
    ) {
        Self::encode_fold_form(hal, kernel, form_of(m32), io, out, input, n)
    }

    pub(crate) fn encode_fold_form(
        hal: &MetalHal<Self>,
        kernel: PvKernel,
        form: Poseidon2Kernels,
        io: &BufferImpl<Digest>,
        out: usize,
        input: usize,
        n: u32,
    ) {
        if n == 0 {
            return;
        }
        let args = &[
            io.as_arg_with_offset(out),
            io.as_arg_with_offset(input),
            KernelArg::Integer(n),
        ];
        let name = kernel.name("fold", form);
        hal.dispatch(hal.pipeline(&name), args, groups(kernel, n), &[]);
    }

    /// Encode `hash_rows` over a column-major `rows x cols` matrix with `kernel`.
    #[cfg(test)]
    pub(crate) fn encode_rows(
        hal: &MetalHal<Self>,
        kernel: PvKernel,
        m32: bool,
        output: &BufferImpl<Digest>,
        matrix: &BufferImpl<BabyBearElem>,
        rows: u32,
        cols: u32,
    ) {
        Self::encode_rows_form(hal, kernel, form_of(m32), output, matrix, rows, cols)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_rows_form(
        hal: &MetalHal<Self>,
        kernel: PvKernel,
        form: Poseidon2Kernels,
        output: &BufferImpl<Digest>,
        matrix: &BufferImpl<BabyBearElem>,
        rows: u32,
        cols: u32,
    ) {
        if rows == 0 {
            return;
        }
        let args = &[
            output.as_arg(),
            matrix.as_arg(),
            KernelArg::Integer(rows),
            KernelArg::Integer(cols),
        ];
        let name = kernel.name("rows", form);
        hal.dispatch(hal.pipeline(&name), args, groups(kernel, rows), &[]);
    }

    /// Encode the fold-tree layers `n0, n0 / 2, ..., 1` (`n0 <= 128`) as one dispatch.
    #[cfg(test)]
    pub(crate) fn encode_fold_tail(
        hal: &MetalHal<Self>,
        m32: bool,
        io: &BufferImpl<Digest>,
        n0: u32,
    ) {
        Self::encode_fold_tail_form(hal, form_of(m32), io, n0)
    }

    pub(crate) fn encode_fold_tail_form(
        hal: &MetalHal<Self>,
        form: Poseidon2Kernels,
        io: &BufferImpl<Digest>,
        n0: u32,
    ) {
        assert!((1..=PV_TAIL).contains(&n0) && n0.is_power_of_two());
        let name = match form {
            Poseidon2Kernels::PvM32 => "pv_fold_tail_k_m32",
            Poseidon2Kernels::Pv => "pv_fold_tail_k",
            Poseidon2Kernels::PvZ9 => "pv_fold_tail_k_z9",
        };
        let args = &[io.as_arg(), KernelArg::Integer(n0)];
        let grid = Grid::Groups {
            groups: MTLSize::new(1, 1, 1),
            threads: MTLSize::new(n0 as u64, 1, 1),
        };
        hal.dispatch(hal.pipeline(name), args, grid, &[]);
    }

    fn fold(&self, hal: &MetalHal<Self>, io: &BufferImpl<Digest>, n: usize) {
        let cfg = self.config();
        let n32 = u32::try_from(n).expect("hash_fold: too many outputs");
        Self::encode_fold_form(
            hal,
            Self::fold_kernel(&cfg, n32),
            cfg.kernels,
            io,
            n,
            2 * n,
            n32,
        );
    }
}

/// `_m32` (true) or pv.cuh's HIP form (false), the forms the Z3 test entry points take.
#[cfg(test)]
fn form_of(m32: bool) -> Poseidon2Kernels {
    if m32 {
        Poseidon2Kernels::PvM32
    } else {
        Poseidon2Kernels::Pv
    }
}

/// The launch shape of a `pv.metal` kernel over `n` permutations: the CUDA/HIP shapes
/// (`pv`: blocks of min(n, 128) threads; `pvs`: whole 128-thread blocks).
fn groups(kernel: PvKernel, n: u32) -> Grid {
    let threads = n as u64 * kernel.lanes();
    let bs = match kernel {
        PvKernel::Pv => threads.min(PV_BLOCK),
        PvKernel::PvsW | PvKernel::PvsQ => PV_BLOCK,
    };
    Grid::Groups {
        groups: MTLSize::new(threads.div_ceil(bs), 1, 1),
        threads: MTLSize::new(bs, 1, 1),
    }
}

impl MetalHash for MetalHashPoseidon2 {
    fn new(hal: &MetalHal<Self>) -> Self {
        let mut cfg = Poseidon2Config::from_env();
        // The pvs kernels shuffle across 32 lanes.
        if PVS_KERNELS
            .iter()
            .any(|name| hal.pipeline(name).thread_execution_width() != 32)
        {
            cfg.small = false;
        }
        MetalHashPoseidon2 {
            suite: Poseidon2HashSuite::new_suite(),
            cfg: Mutex::new(cfg),
        }
    }

    fn hash_fold(&self, hal: &MetalHal<Self>, io: &BufferImpl<Digest>, output_size: usize) {
        self.fold(hal, io, output_size);
    }

    fn hash_rows(
        &self,
        hal: &MetalHal<Self>,
        output: &BufferImpl<Digest>,
        matrix: &BufferImpl<BabyBearElem>,
    ) {
        let row_size = output.size();
        if row_size == 0 {
            return;
        }
        let col_size = matrix.size() / row_size;
        assert_eq!(matrix.size(), col_size * row_size);
        let cfg = self.config();
        // Z14: fills set aside by `prefill_elem` ride on the first row hash that can carry them.
        if Self::fuse_shape(&cfg, row_size, col_size) {
            let fills = hal.rt.prefill_unfilled(2);
            if !fills.is_empty() {
                Self::encode_rows_side(hal, output, matrix, None, &fills, PV_SIDE_TG_BYTES);
                return;
            }
        }
        let rows = u32::try_from(row_size).expect("hash_rows: too many rows");
        let cols = u32::try_from(col_size).expect("hash_rows: too many columns");
        Self::encode_rows_form(
            hal,
            Self::rows_kernel(&cfg, rows, cols),
            cfg.kernels,
            output,
            matrix,
            rows,
            cols,
        );
    }

    /// Every layer in one batch (api.cu `sppark_poseidon2_fold_tree`): one dispatch per layer,
    /// and the layers of at most `tail_max` outputs in one `pv_fold_tail_k` dispatch.
    fn hash_fold_tree(&self, hal: &MetalHal<Self>, io: &BufferImpl<Digest>, layers: usize) {
        let cfg = self.config();
        assert!(
            io.size() >= 2 << layers,
            "hash_fold_tree: {} digests for {layers} layers",
            io.size()
        );
        for i in (0..layers).rev() {
            let n = 1usize << i;
            if n <= cfg.tail_max as usize {
                Self::encode_fold_tail_form(hal, cfg.kernels, io, n as u32);
                break;
            }
            self.fold(hal, io, n);
        }
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }

    fn carries_fills(&self) -> bool {
        let cfg = self.config();
        cfg.commit_fuse && cfg.kernels == Poseidon2Kernels::PvZ9
    }

    fn fuses_bit_reverse(&self, rows: usize, cols: usize, lg: usize) -> bool {
        Self::fuse_shape(&self.config(), rows, cols) && (10..32).contains(&lg)
    }

    fn hash_rows_bit_reverse(
        &self,
        hal: &MetalHal<Self>,
        output: &BufferImpl<Digest>,
        matrix: &BufferImpl<BabyBearElem>,
        coeffs: &BufferImpl<BabyBearElem>,
        count: usize,
    ) -> bool {
        let rows = output.size();
        if rows == 0 || count == 0 {
            return false;
        }
        let cols = matrix.size() / rows;
        assert_eq!(matrix.size(), cols * rows);
        let n = coeffs.size() / count;
        assert_eq!(coeffs.size(), n * count);
        if !n.is_power_of_two() || !self.fuses_bit_reverse(rows, cols, n.trailing_zeros() as usize)
        {
            return false;
        }
        let fills = hal.rt.prefill_unfilled(2);
        Self::encode_rows_side(
            hal,
            output,
            matrix,
            Some((coeffs, n.trailing_zeros() as usize, count)),
            &fills,
            PV_SIDE_TG_BYTES,
        );
        true
    }
}

/// Mirrored by `PvSideParams` in pv.metal.
#[derive(Clone, Copy, Debug, Default, bytemuck::NoUninit)]
#[repr(C)]
struct PvSideParams {
    lg: u32,
    cols: u32,
    stride: u32,
    _pad: u32,
    fill_n0: u32,
    fill_v0: u32,
    fill_n1: u32,
    fill_v1: u32,
}

/// `pv_rows_side`'s threadgroup memory: two 32 x 32 tiles with rows padded to 33 words.
pub(crate) const PV_SIDE_TG_BYTES: u64 = 2 * 32 * 33 * 4;

impl MetalHashPoseidon2 {
    /// Z14: `pv_rows_side_k_z9` covers this row-hash shape: the default (Z9) thread-per-hash row
    /// kernel in whole 128-thread groups (bit reversals need columns of at least 2^10, one 32 x
    /// 32 tile pair).
    fn fuse_shape(cfg: &Poseidon2Config, rows: usize, cols: usize) -> bool {
        cfg.commit_fuse
            && cfg.kernels == Poseidon2Kernels::PvZ9
            && cols > 0
            && rows.is_multiple_of(PV_BLOCK as usize)
            && u32::try_from(rows).is_ok()
            && u32::try_from(cols).is_ok()
            && Self::rows_kernel(cfg, rows as u32, cols as u32) == PvKernel::Pv
    }

    /// Encode `pv_rows_side_k_z9`: `hash_rows(output, matrix)`, plus the in-place bit reversal of
    /// `bit_reverse` = (coeffs, lg, count: columns of 2^lg) and up to two `fills` (buffer, bytes,
    /// word; bytes a multiple of 16).
    pub(crate) fn encode_rows_side(
        hal: &MetalHal<Self>,
        output: &BufferImpl<Digest>,
        matrix: &BufferImpl<BabyBearElem>,
        bit_reverse: Option<(&BufferImpl<BabyBearElem>, usize, usize)>,
        fills: &[(metal::Buffer, u64, u32)],
        tg_bytes: u64,
    ) {
        assert!(fills.len() <= 2);
        let rows = output.size();
        let cols = matrix.size() / rows;
        let mut params = PvSideParams::default();
        if let Some((_, lg, count)) = bit_reverse {
            assert!((10..32).contains(&lg));
            params.lg = lg as u32;
            params.cols = u32::try_from(count).expect("bit reversal: too many columns");
            params.stride = 1u32 << lg;
        }
        let fill_n = |k: usize| {
            fills.get(k).map_or(0, |f| {
                assert_eq!(f.1 % 16, 0);
                u32::try_from(f.1 / 16).expect("fill: too large")
            })
        };
        params.fill_n0 = fill_n(0);
        params.fill_n1 = fill_n(1);
        params.fill_v0 = fills.first().map_or(0, |f| f.2);
        params.fill_v1 = fills.get(1).map_or(0, |f| f.2);
        let params = [params];
        // Metal API validation requires every argument the kernel declares to be bound, even when
        // a zero count (`cols` / `fill_n*`) keeps the kernel from touching it, so an absent io or
        // fill gets `matrix` as a dummy (never read or written through that binding).
        let fill_arg = |k: usize| {
            fills.get(k).map_or(matrix.as_arg(), |f| KernelArg::Buffer {
                buffer: &f.0,
                offset: 0,
            })
        };
        let args = &[
            output.as_arg(),
            matrix.as_arg(),
            KernelArg::Integer(rows as u32),
            KernelArg::Integer(cols as u32),
            bit_reverse.map_or(matrix.as_arg(), |b| b.0.as_arg()),
            KernelArg::Bytes(as_bytes(&params)),
            fill_arg(0),
            fill_arg(1),
        ];
        hal.dispatch(
            hal.pipeline("pv_rows_side_k_z9"),
            args,
            groups(PvKernel::Pv, rows as u32),
            &[tg_bytes],
        );
    }
}
