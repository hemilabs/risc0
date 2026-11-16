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

//! The NTT of the Metal HAL (plan 4.1 row Z2): host logic of `supra/ntt_x.cuh` and the
//! `sppark_*` entry points of `supra/ntt.cu`, ported function for function onto the kernels of
//! `sys/kernels/zkp/metal/ntt_x.metal`.
//!
//! * [NttTables]: sppark's `NTTParameters` (partial, radix-6..11 and plus-one twiddles, forward and
//!   inverse) built on the CPU with the same generator kernels' formulas, from sppark's own
//!   `forward/inverse_roots_of_unity` constants, which are checked against risc0's
//!   `ROU_FWD`/`ROU_REV` (and `domain_size_inverse` against `2^-lg`). The first use in a process
//!   also runs GPU transforms against the CPU reference (`core::ntt`) and panics on a mismatch.
//! * `ct_steps`/`gs_steps`/`x_step` (`X_launcher::step`)/`ct_x`/`gs_x`/`bitrev_batched`/
//!   `nttx_batch_zk_shift`/`nttx_auto_chunk`/`nttx_commit` mirror the CUDA functions of the same
//!   names, including launch shapes (block `2^(radix-1)`, grid `(nblocks[/8], cols)`, threadgroup
//!   memory `[8 *] 4 << (radix-1)` bytes, at most 16 KB).
//! * The pre-port `multi_ntt_*`/`batch_expand`/`multi_bit_reverse`/`zk_shift` kernels and their
//!   `RISC0_METAL_LEGACY=1` switch were retired in Z7.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, OnceLock,
    },
};

use metal::MTLSize;
use risc0_core::{
    field::{baby_bear::BabyBearElem, Elem, RootsOfUnity},
    scope,
};

use super::{as_bytes, BufferImpl, Grid, KernelArg, MetalHal, MetalHash};
use crate::{
    core::{log2_ceil, ntt as cpu_ntt},
    hal::Buffer as _,
};

type Fp = BabyBearElem;

/// The Metal kernels of `ntt_x.metal` (appended to `KERNEL_NAMES`; checked by tests_z2).
#[cfg(test)]
pub const NTT_X_KERNELS: &[&str] = &[
    "bit_rev_permutation_batched",
    "bit_rev_z_batched_64",
    "ct_x_1",
    "ct_x_8",
    "ct_x_8_coalesced",
    "gs_x_1",
    "gs_x_8",
    "gs_x_8_coalesced",
    "nttx_zk_shift_batched",
];

// sppark ntt/parameters.cuh with FEATURE_BABY_BEAR.
const MAX_LG_DOMAIN_SIZE: usize = 27;
const LG_WINDOW_SIZE: usize = MAX_LG_DOMAIN_SIZE.div_ceil(5);
const WINDOW_SIZE: usize = 1 << LG_WINDOW_SIZE;
const WINDOW_NUM: usize = MAX_LG_DOMAIN_SIZE.div_ceil(LG_WINDOW_SIZE);
const PLUS_ONE_ROWS: usize = MAX_LG_DOMAIN_SIZE - 15;
/// `X_launcher`'s `Z_COUNT = 256/8/sizeof(fr_t)`.
const Z_COUNT: u64 = 256 / 8 / 4;

/// Layout of one direction's table buffer: partial twiddles, the radix-11..6 blob (sppark
/// `twiddles[5]` first), plus-one twiddles.
const PARTIAL_OFF: usize = 0;
const BLOB_OFF: usize = PARTIAL_OFF + WINDOW_NUM * WINDOW_SIZE;
const BLOB_LEN: usize = 1024 + 512 + 256 + 128 + 64 + 32;
const PLUS_ONE_OFF: usize = BLOB_OFF + BLOB_LEN;
const TABLE_LEN: usize = PLUS_ONE_OFF + PLUS_ONE_ROWS * 1024;

/// Offset in the blob of sppark `twiddles[k]` (radix `k + 6`).
const TWIDDLES_OFF: [usize; 6] = [
    1024 + 512 + 256 + 128 + 64,
    1024 + 512 + 256 + 128,
    1024 + 512 + 256,
    1024 + 512,
    1024,
    0,
];

/// sppark ntt/parameters/baby_bear.h (no BABY_BEAR_CANONICAL), Montgomery words.
const SPPARK_FORWARD_ROOTS: [u32; 28] = [
    0x0ffffffe, 0x68000003, 0x5bc72af0, 0x02ec07f3, 0x67e027ca, 0x19e5f901, 0x3b27e54a, 0x20d1773e,
    0x771ea53a, 0x0fb182ad, 0x146d1455, 0x3e7d65f0, 0x327884f2, 0x53fc8703, 0x20742dd1, 0x31062eda,
    0x642b70ab, 0x1ccd534b, 0x03cc9bf7, 0x6686182f, 0x2e2516d3, 0x5701b5c8, 0x193a6352, 0x112fc5b9,
    0x63ec6b91, 0x5b34b3ff, 0x3fff6398, 0x1ffffedc,
];
const SPPARK_INVERSE_ROOTS: [u32; 28] = [
    0x0ffffffe, 0x68000003, 0x1c38d511, 0x3d85298f, 0x5f06e481, 0x38a3c615, 0x4ed6e525, 0x55372b64,
    0x4d88ae94, 0x5806fd5e, 0x2ced6d6a, 0x1851eacd, 0x2fa36b4d, 0x0a556a3b, 0x18ae7209, 0x742ba568,
    0x3f462cba, 0x50b5c3b2, 0x0dfdfca6, 0x3821b546, 0x45e4cd80, 0x3e6793bd, 0x5bdeafa3, 0x2e01d37a,
    0x2da9f4f0, 0x1db7e183, 0x167ca34b, 0x50b3630a,
];
const SPPARK_DOMAIN_SIZE_INVERSE: [u32; 28] = [
    0x0ffffffe, 0x07ffffff, 0x40000000, 0x20000000, 0x10000000, 0x08000000, 0x04000000, 0x02000000,
    0x01000000, 0x00800000, 0x00400000, 0x00200000, 0x00100000, 0x00080000, 0x00040000, 0x00020000,
    0x00010000, 0x00008000, 0x00004000, 0x00002000, 0x00001000, 0x00000800, 0x00000400, 0x00000200,
    0x00000100, 0x00000080, 0x00000040, 0x00000020,
];
/// sppark `group_gen = fr_t(3)` (the `int` constructor encodes), the zk-shift generator.
const GROUP_GEN: u32 = 3;

/// Mirrors `NttXParams` in ntt_x.metal.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::NoUninit)]
struct NttXParams {
    radix: u32,
    lg: u32,
    stage: u32,
    iterations: u32,
    in_stride: u32,
    out_stride: u32,
    lg_expand: u32,
    skip_expand: u32,
    is_intt: u32,
    dsi: u32,
    post: u32,
    _pad: u32,
}
const _: () = assert!(std::mem::size_of::<NttXParams>() == 48);

/// Mirrors `BitRevParams` in ntt_x.metal.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::NoUninit)]
struct BitRevParams {
    lg: u32,
    cols: u32,
    stride: u32,
    _pad: u32,
}
const _: () = assert!(std::mem::size_of::<BitRevParams>() == 16);

fn bit_rev(i: usize, nbits: usize) -> usize {
    if nbits == 0 {
        0
    } else {
        (i as u32).reverse_bits() as usize >> (32 - nbits)
    }
}

/// sppark `NTTParameters` for both directions, built on the CPU. `dirs[0]` forward, `dirs[1]`
/// inverse, each `TABLE_LEN` elements laid out as described at `PARTIAL_OFF`.
pub(crate) struct NttTables {
    pub(crate) dirs: [Vec<Fp>; 2],
}

impl NttTables {
    /// The process-wide tables (built once).
    pub(crate) fn get() -> &'static NttTables {
        static TABLES: OnceLock<NttTables> = OnceLock::new();
        TABLES.get_or_init(NttTables::build)
    }

    fn build() -> Self {
        check_sppark_constants();
        let dirs = [false, true].map(|inverse| {
            let roots = if inverse {
                &SPPARK_INVERSE_ROOTS
            } else {
                &SPPARK_FORWARD_ROOTS
            };
            let mut t = vec![Fp::ZERO; TABLE_LEN];
            generate_all_twiddles(&mut t[BLOB_OFF..PLUS_ONE_OFF], Fp::new_raw(roots[11]));
            generate_plus_one_twiddles(
                &mut t[PLUS_ONE_OFF..],
                Fp::new_raw(roots[MAX_LG_DOMAIN_SIZE]),
            );
            generate_partial_twiddles(
                &mut t[PARTIAL_OFF..BLOB_OFF],
                Fp::new_raw(roots[MAX_LG_DOMAIN_SIZE]),
            );
            t
        });
        Self { dirs }
    }
}

/// sppark's root/inverse tables must be risc0's: same field elements as `ROU_FWD`/`ROU_REV`, and
/// `domain_size_inverse[lg] == 2^-lg`.
fn check_sppark_constants() {
    for k in 0..=MAX_LG_DOMAIN_SIZE {
        assert_eq!(
            Fp::new_raw(SPPARK_FORWARD_ROOTS[k]),
            Fp::ROU_FWD[k],
            "sppark forward root {k}"
        );
        assert_eq!(
            Fp::new_raw(SPPARK_INVERSE_ROOTS[k]),
            Fp::ROU_REV[k],
            "sppark inverse root {k}"
        );
        assert_eq!(
            Fp::new_raw(SPPARK_DOMAIN_SIZE_INVERSE[k]),
            Fp::new(1u32 << k).inv(),
            "sppark domain_size_inverse {k}"
        );
    }
}

/// sppark `generate_partial_twiddles`: `roots[off][tid] = root^(tid << (off * LG_WINDOW_SIZE))`.
fn generate_partial_twiddles(out: &mut [Fp], root: Fp) {
    for tid in 0..WINDOW_SIZE {
        let mut r = root.pow(tid);
        out[tid] = r;
        for off in 1..WINDOW_NUM {
            for _ in 0..LG_WINDOW_SIZE {
                r = r * r;
            }
            out[off * WINDOW_SIZE + tid] = r;
        }
    }
}

/// sppark `generate_all_twiddles`: radix-11 (1024) .. radix-6 (32) tables of `root11^t`.
fn generate_all_twiddles(blob: &mut [Fp], root11: Fp) {
    let mut r = Fp::ONE;
    for t in 0..1024 {
        blob[t] = r;
        let mut base = 0;
        let mut len = 1024;
        for step in 1..6 {
            base += len;
            len /= 2;
            if t % (1 << step) == 0 {
                blob[base + (t >> step)] = r;
            }
        }
        r *= root11;
    }
}

/// sppark `generate_plus_one_twiddles`: `[0][t] = root^bitrev(t, 10)`, `[off] = [off-1]^2`.
fn generate_plus_one_twiddles(out: &mut [Fp], root: Fp) {
    for tid in 0..1024 {
        let mut r = root.pow(bit_rev(tid, 10));
        out[tid] = r;
        for off in 1..PLUS_ONE_ROWS {
            r = r * r;
            out[off * 1024 + tid] = r;
        }
    }
}

/// `nttx_tables`: `lo[t] = g^t` (t < min(n, 1024), else 1), `hi[u] = g^(1024 u)`
/// (u < 2^(lg-10), or a single 1 for lg <= 10), stored as `lo ++ hi`. sppark derives these from
/// `LDE_powers` on an all-ones vector (`pw[i] = g^bitrev(i)`); these are the same field elements.
fn zk_table(lg: usize) -> Vec<Fp> {
    let n = 1usize << lg;
    let nhi = if lg > 10 { 1usize << (lg - 10) } else { 1 };
    let g = Fp::new(GROUP_GEN);
    let mut out = Vec::with_capacity(1024 + nhi);
    let mut cur = Fp::ONE;
    for t in 0..1024 {
        out.push(if t < n { cur } else { Fp::ONE });
        cur *= g;
    }
    let g1024 = g.pow(1024);
    let mut cur = Fp::ONE;
    for _ in 0..nhi {
        out.push(if lg > 10 { cur } else { Fp::ONE });
        cur *= g1024;
    }
    out
}

/// Per-HAL NTT state: the uploaded twiddle tables and zk tables per lg.
pub(crate) struct NttState {
    tables: OnceLock<[BufferImpl<Fp>; 2]>,
    zk: Mutex<HashMap<usize, BufferImpl<Fp>>>,
    /// Z9: per-HAL step-split overrides `(ct, lg) -> iterations per pass` (measurements, tests).
    split: Mutex<HashMap<(bool, usize), Vec<usize>>>,
}

impl NttState {
    pub(crate) fn new() -> Self {
        Self {
            tables: OnceLock::new(),
            zk: Mutex::new(HashMap::new()),
            split: Mutex::new(HashMap::new()),
        }
    }

    /// The passes of a forward (`ct`) or inverse NTT of `2^lg`: an override if set, else sppark's
    /// split.
    fn steps(&self, ct: bool, lg: usize) -> Vec<usize> {
        if let Some(v) = self.split.lock().unwrap().get(&(ct, lg)) {
            return v.clone();
        }
        if ct {
            ct_steps(lg)
        } else {
            gs_steps(lg)
        }
    }
}

/// Set once the GPU-vs-CPU self-check of the tables has started in this process.
static SELF_CHECKED: AtomicBool = AtomicBool::new(false);

/// One pass's buffers (`XArgs`): input/output buffer + element offset + column stride.
struct XArgs<'a> {
    input: (&'a BufferImpl<Fp>, usize),
    in_stride: usize,
    output: (&'a BufferImpl<Fp>, usize),
    out_stride: usize,
    lg_expand: usize,
    skip_expand: bool,
    /// GS last pass only: the zk table (`lo ++ hi`) to post-multiply with.
    post: Option<&'a BufferImpl<Fp>>,
}

/// sppark `CT_NTT` step split (lg <= 30).
fn ct_steps(lg: usize) -> Vec<usize> {
    if lg <= 10 {
        vec![lg]
    } else if lg <= 18 {
        let s = lg / 2;
        vec![s + lg % 2, s]
    } else {
        let s = lg / 3;
        let rem = lg % 3;
        let l29 = (lg == 29) as usize;
        vec![s, s + l29, s + if lg == 29 { 1 } else { rem }]
    }
}

/// sppark `GS_NTT` step split (lg <= 30).
fn gs_steps(lg: usize) -> Vec<usize> {
    if lg <= 10 {
        vec![lg]
    } else if lg <= 18 {
        let s = lg / 2;
        vec![s, s + lg % 2]
    } else {
        let s = lg / 3;
        let rem = lg % 3;
        let l29 = (lg == 29) as usize;
        vec![s + if lg == 29 { 1 } else { rem }, s + l29, s]
    }
}

/// `nttx_auto_chunk`: columns per chunk so a chunk touches ~96 MB (L2-sized on CUDA).
fn nttx_auto_chunk(lg: usize, lg_blowup: usize) -> usize {
    let per_col = (4usize << lg) + (4usize << (lg + lg_blowup));
    let c = (96usize << 20) / per_col;
    let mut k = 1;
    while k * 2 <= c && k < 64 {
        k *= 2;
    }
    k
}

impl<MH: MetalHash> MetalHal<MH> {
    /// This HAL's copy of the twiddle tables (`[forward, inverse]`); the first call in the
    /// process also runs the GPU-vs-CPU self-check.
    fn ntt_tables(&self) -> &[BufferImpl<Fp>; 2] {
        let tables = self.ntt.tables.get_or_init(|| {
            let t = NttTables::get();
            [
                self.copy_from_slice("ntt_twiddles_fwd", &t.dirs[0]),
                self.copy_from_slice("ntt_twiddles_inv", &t.dirs[1]),
            ]
        });
        if !SELF_CHECKED.swap(true, Ordering::AcqRel) {
            self.ntt_self_check();
        }
        tables
    }

    /// The zk-shift table for columns of `2^lg` rows (`nttx_tables`).
    fn ntt_zk_table(&self, lg: usize) -> BufferImpl<Fp> {
        let mut zk = self.ntt.zk.lock().unwrap();
        zk.entry(lg)
            .or_insert_with(|| self.copy_from_slice("ntt_zk_table", &zk_table(lg)))
            .clone()
    }

    /// `X_launcher::step<CT>`.
    fn x_step(&self, ct: bool, cols: usize, lg: usize, stage: usize, iterations: usize, a: &XArgs) {
        let tables = &self.ntt_tables()[!ct as usize];
        let radix = iterations.max(6);
        let num_threads = 1u64 << (lg - 1);
        let block_size = num_threads.min(1u64 << (radix - 1));
        let num_blocks = num_threads.div_ceil(block_size);
        let shared_sz = 4u64 << (radix - 1);
        let (name, groups, tg_bytes) = if num_blocks < Z_COUNT {
            (if ct { "ct_x_1" } else { "gs_x_1" }, num_blocks, shared_sz)
        } else {
            let plain = if ct {
                stage == 0 || lg < 12
            } else {
                stage == iterations || lg < 12
            };
            let name = match (ct, plain) {
                (true, true) => "ct_x_8",
                (true, false) => "ct_x_8_coalesced",
                (false, true) => "gs_x_8",
                (false, false) => "gs_x_8_coalesced",
            };
            (name, num_blocks / Z_COUNT, Z_COUNT * shared_sz)
        };
        // Column strides are 32-bit in the kernel (offsets within a column are < 2^27).
        let in_stride = u32::try_from(a.in_stride).expect("NTT input column stride exceeds u32");
        let out_stride = u32::try_from(a.out_stride).expect("NTT output column stride exceeds u32");
        let params = [NttXParams {
            radix: radix as u32,
            lg: lg as u32,
            stage: stage as u32,
            iterations: iterations as u32,
            in_stride,
            out_stride,
            lg_expand: a.lg_expand as u32,
            skip_expand: a.skip_expand as u32,
            is_intt: !ct as u32,
            dsi: SPPARK_DOMAIN_SIZE_INVERSE[lg],
            post: a.post.is_some() as u32,
            _pad: 0,
        }];
        // Bindings a variant's code never reads are left unbound (the validation layer warns
        // about them): plus-one twiddles in the Z = 1 variants, lo/hi in the coalesced GS one.
        let z1 = name.ends_with("_1");
        let mut args = vec![
            a.input.0.as_arg_with_offset(a.input.1),
            a.output.0.as_arg_with_offset(a.output.1),
            tables.as_arg_with_offset(PARTIAL_OFF),
            if z1 {
                KernelArg::Null
            } else {
                tables.as_arg_with_offset(PLUS_ONE_OFF)
            },
            tables.as_arg_with_offset(BLOB_OFF + TWIDDLES_OFF[0]),
            tables.as_arg_with_offset(BLOB_OFF + TWIDDLES_OFF[radix - 6]),
            KernelArg::Bytes(as_bytes(&params)),
        ];
        if !ct {
            // lo, hi: only the last pass (never coalesced) post-multiplies. The plain variants
            // read them behind the runtime `post` flag, so they need a binding (the twiddle
            // table, never read); the coalesced variant has no such code.
            match (a.post, name.ends_with("_coalesced")) {
                (Some(zk), _) => {
                    args.push(zk.as_arg_with_offset(0));
                    args.push(zk.as_arg_with_offset(1024));
                }
                (None, false) => {
                    args.push(tables.as_arg_with_offset(0));
                    args.push(tables.as_arg_with_offset(0));
                }
                (None, true) => {
                    args.push(KernelArg::Null);
                    args.push(KernelArg::Null);
                }
            }
        }
        let grid = Grid::Groups {
            groups: MTLSize::new(groups, cols as u64, 1),
            threads: MTLSize::new(block_size, 1, 1),
        };
        let pso = self.pipeline(name);
        super::kernels::assert_simd_width(pso, name);
        self.dispatch(pso, &args, grid, &[tg_bytes]);
    }

    /// `ct_x`: forward CT (RN), first pass out of place from `input` (optionally synthesising
    /// the zero-padded LDE input, `lg_expand`, and skipping its trivial levels, `skip`).
    #[allow(clippy::too_many_arguments)]
    fn ct_x(
        &self,
        cols: usize,
        input: (&BufferImpl<Fp>, usize),
        in_stride: usize,
        lg_expand: usize,
        output: (&BufferImpl<Fp>, usize),
        out_stride: usize,
        lg: usize,
        skip: bool,
    ) {
        let st = self.ntt.steps(true, lg);
        let mut stage = 0;
        for (i, &iterations) in st.iter().enumerate() {
            let a = XArgs {
                input: if i == 0 { input } else { output },
                in_stride: if i == 0 { in_stride } else { out_stride },
                output,
                out_stride,
                lg_expand: if i == 0 { lg_expand } else { 0 },
                skip_expand: skip && i == 0 && lg_expand > 0 && lg_expand < 6 && st[0] > lg_expand,
                post: None,
            };
            self.x_step(true, cols, lg, stage, iterations, &a);
            stage += iterations;
        }
    }

    /// `gs_x`: inverse GS (NR) including the `n^-1` scaling, first pass out of place from
    /// `input`, optional zk-shift post-multiply on the last pass.
    #[allow(clippy::too_many_arguments)]
    fn gs_x(
        &self,
        cols: usize,
        input: (&BufferImpl<Fp>, usize),
        in_stride: usize,
        output: (&BufferImpl<Fp>, usize),
        out_stride: usize,
        lg: usize,
        zk: Option<&BufferImpl<Fp>>,
    ) {
        let st = self.ntt.steps(false, lg);
        let n = st.len();
        let mut stage = lg;
        for (i, &iterations) in st.iter().enumerate() {
            let a = XArgs {
                input: if i == 0 { input } else { output },
                in_stride: if i == 0 { in_stride } else { out_stride },
                output,
                out_stride,
                lg_expand: 0,
                skip_expand: false,
                post: if i == n - 1 { zk } else { None },
            };
            self.x_step(false, cols, lg, stage, iterations, &a);
            stage -= iterations;
        }
    }

    /// `bitrev_batched`: in-place bit reversal of `cols` columns of `2^lg` (stride `2^lg`) at
    /// `io[off..]`. n < 4096: sppark `bit_rev_permutation` (swap form), one 2-D dispatch over all
    /// columns; otherwise `bit_rev_z_batched<64>`, grid (n/4096, cols) x 64, 16 KB.
    fn bitrev_batched(&self, io: (&BufferImpl<Fp>, usize), lg: usize, cols: usize) {
        const Z: u64 = 64;
        let n = 1u64 << lg;
        let params = [BitRevParams {
            lg: lg as u32,
            cols: cols as u32,
            stride: n as u32,
            _pad: 0,
        }];
        let args = [
            io.0.as_arg_with_offset(io.1),
            KernelArg::Bytes(as_bytes(&params)),
        ];
        if n < Z * Z {
            let pso = self.pipeline("bit_rev_permutation_batched");
            self.dispatch(pso, &args, Grid::Threads2D(n, cols as u64), &[]);
        } else {
            let grid = Grid::Groups {
                groups: MTLSize::new(n / Z / Z, cols as u64, 1),
                threads: MTLSize::new(Z, 1, 1),
            };
            let pso = self.pipeline("bit_rev_z_batched_64");
            self.dispatch(pso, &args, grid, &[Z * Z * 4]);
        }
    }

    /// `nttx_batch_zk_shift`: zk shift of `cols` bit-reversed columns of `2^lg`.
    fn nttx_batch_zk_shift(&self, io: &BufferImpl<Fp>, lg: usize, cols: usize) {
        let zk = self.ntt_zk_table(lg);
        let n = 1u64 << lg;
        let params = [BitRevParams {
            lg: lg as u32,
            cols: cols as u32,
            stride: n as u32,
            _pad: 0,
        }];
        let args = [
            io.as_arg(),
            zk.as_arg_with_offset(0),
            zk.as_arg_with_offset(1024),
            KernelArg::Bytes(as_bytes(&params)),
        ];
        let pso = self.pipeline("nttx_zk_shift_batched");
        self.dispatch(pso, &args, Grid::Threads2D(n, cols as u64), &[]);
    }

    /// `nttx_commit` (the nttq branch is not ported: its threadgroup memory exceeds 32 KB).
    #[allow(clippy::too_many_arguments)]
    fn nttx_commit(
        &self,
        coeffs: &BufferImpl<Fp>,
        evaluated: &BufferImpl<Fp>,
        witness: &BufferImpl<Fp>,
        lg: usize,
        lg_blowup: usize,
        cols: usize,
        chunk: usize,
        natural: bool,
    ) {
        let zk = self.ntt_zk_table(lg);
        let chunk = if chunk == 0 {
            nttx_auto_chunk(lg, lg_blowup)
        } else {
            chunk
        };
        let n = 1usize << lg;
        let big_n = n << lg_blowup;
        let mut c0 = 0;
        while c0 < cols {
            let k = chunk.min(cols - c0);
            self.gs_x(k, (witness, c0 * n), n, (coeffs, c0 * n), n, lg, Some(&zk));
            self.ct_x(
                k,
                (coeffs, c0 * n),
                n,
                lg_blowup,
                (evaluated, c0 * big_n),
                big_n,
                lg + lg_blowup,
                false,
            );
            if natural {
                self.bitrev_batched((coeffs, c0 * n), lg, k);
            }
            c0 += chunk;
        }
    }

    // ---- Hal entry points (`sppark_*` of supra/ntt.cu) ----

    /// `batch_expand_into_evaluate_ntt`: the expand folded into the first CT pass (zero-padded
    /// input synthesised on load, trivial levels skipped), all columns per launch.
    pub(crate) fn ntt_batch_expand_into_evaluate(
        &self,
        output: &BufferImpl<Fp>,
        input: &BufferImpl<Fp>,
        count: usize,
    ) {
        scope!("batch_expand_into_evaluate_ntt");
        let out_size = output.size() / count;
        let in_size = input.size() / count;
        let expand_bits = log2_ceil(out_size / in_size);
        assert_eq!(output.size(), out_size * count);
        assert_eq!(input.size(), in_size * count);
        assert_eq!(out_size, in_size * (1 << expand_bits));
        let in_bits = log2_ceil(in_size);
        assert_eq!(in_size, 1 << in_bits);
        let lg = in_bits + expand_bits;
        assert!(lg < Fp::MAX_ROU_PO2);
        if lg == 0 {
            self.launch(
                "eltwise_copy_fp",
                output.size() as u64,
                &[output.as_arg(), input.as_arg()],
            );
        } else {
            self.ct_x(
                count,
                (input, 0),
                in_size,
                expand_bits,
                (output, 0),
                out_size,
                lg,
                true,
            );
        }
        self.flush();
    }

    /// `sppark_batch_iNTT`: in place, NR, scaled by `n^-1`.
    pub(crate) fn ntt_batch_interpolate(&self, io: &BufferImpl<Fp>, count: usize) {
        scope!("batch_interpolate_ntt");
        let lg = self.ntt_row_bits(io, count);
        if lg > 0 {
            self.gs_x(count, (io, 0), 1 << lg, (io, 0), 1 << lg, lg, None);
        }
        self.flush();
    }

    /// `sppark_batch_iNTT_zk_shift_fused`, with the zk shift fused into the last GS pass.
    pub(crate) fn ntt_batch_interpolate_zk_shift(&self, io: &BufferImpl<Fp>, count: usize) {
        scope!("batch_interpolate_ntt_zk_shift");
        let lg = self.ntt_row_bits(io, count);
        if lg > 0 {
            let zk = self.ntt_zk_table(lg);
            self.gs_x(count, (io, 0), 1 << lg, (io, 0), 1 << lg, lg, Some(&zk));
        }
        self.flush();
    }

    /// `dst = interpolate_zk_shift(src)`: the first GS pass reads `src`, `src` is unchanged.
    pub(crate) fn ntt_batch_interpolate_zk_shift_from(
        &self,
        dst: &BufferImpl<Fp>,
        src: &BufferImpl<Fp>,
        count: usize,
    ) {
        scope!("batch_interpolate_ntt_zk_shift_from");
        assert_eq!(dst.size(), src.size());
        let lg = self.ntt_row_bits(src, count);
        if lg == 0 {
            self.launch(
                "eltwise_copy_fp",
                dst.size() as u64,
                &[dst.as_arg(), src.as_arg()],
            );
        } else {
            let zk = self.ntt_zk_table(lg);
            self.gs_x(count, (src, 0), 1 << lg, (dst, 0), 1 << lg, lg, Some(&zk));
        }
        self.flush();
    }

    /// `sppark_batch_zk_shift` (`nttx_batch_zk_shift`).
    pub(crate) fn ntt_zk_shift(&self, io: &BufferImpl<Fp>, count: usize) {
        let lg = self.ntt_row_bits(io, count);
        if lg > 0 {
            self.nttx_batch_zk_shift(io, lg, count);
        }
        self.flush();
    }

    /// `sppark_batch_bit_reverse_tiled` (`bitrev_batched`).
    pub(crate) fn ntt_batch_bit_reverse(&self, io: &BufferImpl<Fp>, count: usize) {
        let lg = self.ntt_row_bits(io, count);
        if lg > 0 && count > 0 {
            self.bitrev_batched((io, 0), lg, count);
        }
        self.flush();
    }

    /// `has_commit_coeffs_lde`: the range `sppark_commit_coeffs_lde` accepts (10..=23).
    pub(crate) fn ntt_has_commit_coeffs_lde(&self, po2: usize) -> bool {
        (10..=23).contains(&po2)
    }

    /// `sppark_commit_coeffs_lde` (`nttx_commit`, chunk 0 = auto).
    pub(crate) fn ntt_commit_coeffs_lde(
        &self,
        coeffs: &BufferImpl<Fp>,
        evaluated: &BufferImpl<Fp>,
        witness: &BufferImpl<Fp>,
        count: usize,
    ) {
        self.ntt_commit_coeffs_lde_impl(coeffs, evaluated, witness, count, true);
    }

    /// `ntt_commit_coeffs_lde`; `natural`: bit-reverse the coefficients to natural order (else
    /// they stay in the inverse NTT's bit-reversed order: Z14's `commit_coeffs_lde_bit_reversed`).
    pub(crate) fn ntt_commit_coeffs_lde_impl(
        &self,
        coeffs: &BufferImpl<Fp>,
        evaluated: &BufferImpl<Fp>,
        witness: &BufferImpl<Fp>,
        count: usize,
        natural: bool,
    ) {
        scope!("commit_coeffs_lde");
        let row_size = witness.size() / count;
        assert_eq!(row_size * count, witness.size());
        assert_eq!(coeffs.size(), witness.size());
        let lg = log2_ceil(row_size);
        assert_eq!(row_size, 1 << lg);
        let lg_blowup = log2_ceil(crate::INV_RATE);
        assert_eq!(evaluated.size(), witness.size() << lg_blowup);
        assert!(lg + lg_blowup < Fp::MAX_ROU_PO2);
        assert!(
            (10..=23).contains(&lg),
            "commit_coeffs_lde: lg_domain_size {lg} out of range"
        );
        self.nttx_commit(coeffs, evaluated, witness, lg, lg_blowup, count, 0, natural);
        self.flush();
    }

    fn ntt_row_bits(&self, io: &BufferImpl<Fp>, count: usize) -> usize {
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let lg = log2_ceil(row_size);
        assert_eq!(row_size, 1 << lg);
        assert!(lg < Fp::MAX_ROU_PO2);
        lg
    }

    /// Cross-check of the CPU-built tables: forward (expand + CT, 1..3 passes, all kernel
    /// variants) and inverse + zk shift (GS) against `core::ntt` on deterministic data.
    fn ntt_self_check(&self) {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            Fp::new((state % (risc0_core::field::baby_bear::P as u64)) as u32)
        };
        for lg in [3usize, 12, 19] {
            let n = 1usize << (lg - 2);
            let input: Vec<Fp> = (0..n).map(|_| next()).collect();
            let gin = self.copy_from_slice("ntt_check_in", &input);
            let gout = self.alloc_buffer::<Fp>("ntt_check_out", n << 2);
            self.ct_x(1, (&gin, 0), n, 2, (&gout, 0), n << 2, lg, true);
            let mut want = vec![Fp::ZERO; n << 2];
            cpu_ntt::expand(&mut want, &input, 2);
            cpu_ntt::evaluate_ntt::<Fp, Fp>(&mut want, 2);
            assert!(
                gout.to_vec() == want,
                "Metal NTT self-check (forward, lg {lg}) failed"
            );

            let ginv = self.alloc_buffer::<Fp>("ntt_check_inv", n << 2);
            let zk = self.ntt_zk_table(lg);
            self.gs_x(1, (&gout, 0), n << 2, (&ginv, 0), n << 2, lg, Some(&zk));
            let mut want_inv = want.clone();
            cpu_ntt::interpolate_ntt::<Fp, Fp>(&mut want_inv);
            for (i, x) in want_inv.iter_mut().enumerate() {
                *x *= Fp::new(GROUP_GEN).pow(bit_rev(i, lg));
            }
            assert!(
                ginv.to_vec() == want_inv,
                "Metal NTT self-check (inverse + zk shift, lg {lg}) failed"
            );
        }
        tracing::debug!("Metal NTT tables self-check passed");
    }
}

#[cfg(test)]
pub(crate) mod test_hooks {
    //! Internals exposed to `tests_z2.rs`.
    use super::*;

    pub(crate) fn ct_steps(lg: usize) -> Vec<usize> {
        super::ct_steps(lg)
    }

    pub(crate) fn gs_steps(lg: usize) -> Vec<usize> {
        super::gs_steps(lg)
    }

    /// Override (Some) or reset (None) the pass split of `hal`'s forward (`ct`) or inverse NTTs
    /// of `2^lg`. Every pass must have 1..=10 iterations and they must sum to `lg`.
    pub(crate) fn set_split<MH: MetalHash>(
        hal: &MetalHal<MH>,
        ct: bool,
        lg: usize,
        split: Option<Vec<usize>>,
    ) {
        let mut m = hal.ntt.split.lock().unwrap();
        match split {
            Some(v) => {
                assert!(v.iter().all(|&i| (1..=10).contains(&i)) && v.iter().sum::<usize>() == lg);
                m.insert((ct, lg), v);
            }
            None => {
                m.remove(&(ct, lg));
            }
        }
    }

    /// Z13: `ct_x` of `cols` columns of 2^(lg - 2) from `input` expanded x4 into `output` (column
    /// strides 2^(lg - 2) and 2^lg), zeros synthesised on load (the commit's form).
    pub(crate) fn ct_expand4<MH: MetalHash>(
        hal: &MetalHal<MH>,
        cols: usize,
        input: &BufferImpl<Fp>,
        output: &BufferImpl<Fp>,
        lg: usize,
    ) {
        let n = 1usize << (lg - 2);
        hal.ct_x(cols, (input, 0), n, 2, (output, 0), 4 * n, lg, false);
    }

    /// Z13: `gs_x` (no zk shift) of `cols` columns of 2^lg from `input` into `output`.
    pub(crate) fn gs<MH: MetalHash>(
        hal: &MetalHal<MH>,
        cols: usize,
        input: &BufferImpl<Fp>,
        output: &BufferImpl<Fp>,
        lg: usize,
    ) {
        let n = 1usize << lg;
        hal.gs_x(cols, (input, 0), n, (output, 0), n, lg, None);
    }

    pub(crate) fn zk_table(lg: usize) -> Vec<Fp> {
        super::zk_table(lg)
    }

    pub(crate) fn nttx_auto_chunk(lg: usize, lg_blowup: usize) -> usize {
        super::nttx_auto_chunk(lg, lg_blowup)
    }

    /// `nttx_commit` with an explicit chunk (0 = auto).
    pub(crate) fn commit_chunked<MH: MetalHash>(
        hal: &MetalHal<MH>,
        coeffs: &BufferImpl<Fp>,
        evaluated: &BufferImpl<Fp>,
        witness: &BufferImpl<Fp>,
        lg: usize,
        cols: usize,
        chunk: usize,
    ) {
        let lg_blowup = log2_ceil(crate::INV_RATE);
        hal.nttx_commit(coeffs, evaluated, witness, lg, lg_blowup, cols, chunk, true);
        hal.flush();
    }

    /// One `x_step` (for per-kernel timings).
    pub(crate) fn step<MH: MetalHash>(
        hal: &MetalHal<MH>,
        ct: bool,
        io: &BufferImpl<Fp>,
        lg: usize,
        cols: usize,
        stage: usize,
        iterations: usize,
    ) {
        let n = 1usize << lg;
        let a = XArgs {
            input: (io, 0),
            in_stride: n,
            output: (io, 0),
            out_stride: n,
            lg_expand: 0,
            skip_expand: false,
            post: None,
        };
        hal.x_step(ct, cols, lg, stage, iterations, &a);
    }

    pub(crate) fn bitrev<MH: MetalHash>(
        hal: &MetalHal<MH>,
        io: &BufferImpl<Fp>,
        lg: usize,
        cols: usize,
    ) {
        hal.bitrev_batched((io, 0), lg, cols);
    }
}
