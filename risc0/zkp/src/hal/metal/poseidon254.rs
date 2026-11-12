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

//! Poseidon254 (BN254 Fr) Merkle hashing on Metal with the `poseidon254.metal` kernels (plan
//! item Z10): ports of `supra/poseidon254.cuh` (`_poseidon254_rows`, `_poseidon254_fold`, one
//! thread per hash) with the host logic of `supra/api.cu` (`sppark_poseidon254_{rows,fold,
//! fold_tree}`). The recursion prover's `poseidon_254` suite (identity_p254, the last STARK
//! before the Groth16 wrap) runs on it. Bit-identical to the CPU suite
//! (`core::hash::poseidon_254`); see `docs/metal-port/notes/Z10.md`.
//!
//! Environment (read once per HAL):
//! * `RISC0_METAL_P254_KERNELS` = `w64` (default: CIOS with 64-bit multiply-accumulate, lazy
//!   range, fused dot products, SOS squaring) | `w64c` (squaring by multiplication) | `ref`
//!   (bn254.h's fully reduced 32-bit CIOS `Fr`, the CUDA kernel's arithmetic).
//! * `RISC0_METAL_P254_TG`: threads per threadgroup (default: the PSO's limit, at most 256).
//! * `RISC0_METAL_P254_CPU_TAIL`: fold layers (`hash_fold`, the top of `hash_fold_tree`) of at
//!   most this many outputs run on the CPU (after a sync; 0 = never). One GPU permutation takes
//!   ~0.85 ms of latency, so the top of a tree is latency-bound on the GPU.
//! * `RISC0_METAL_P254_CPU_ROWS`: `hash_rows` with at most this many rows runs on the CPU (0 =
//!   never).

use std::sync::Mutex;

use metal::MTLSize;
use rayon::prelude::*;
use risc0_core::field::baby_bear::{BabyBear, BabyBearElem};

use super::{BufferImpl, Grid, KernelArg, MetalHal, MetalHash};
use crate::{
    core::{
        digest::Digest,
        hash::{poseidon_254::Poseidon254HashSuite, HashSuite},
    },
    hal::Buffer as _,
};

/// Which Poseidon254 kernels hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Poseidon254Kernels {
    /// `p254_*_w64`: CIOS with 64-bit multiply-accumulate, lazy range [0, 2r), fused 3-term dot
    /// products, SOS squaring (default).
    W64,
    /// `p254_*_w64c`: `W64` with squaring by CIOS multiplication (A/B baseline).
    W64Cios,
    /// `p254_*_ref`: bn254.h `Fr` (32-bit carry chains, fully reduced), one reduction per
    /// product: the CUDA kernel's arithmetic (revert path, cross-check).
    Ref,
}

impl Poseidon254Kernels {
    fn suffix(self) -> &'static str {
        match self {
            Poseidon254Kernels::W64 => "w64",
            Poseidon254Kernels::W64Cios => "w64c",
            Poseidon254Kernels::Ref => "ref",
        }
    }
}

/// Kernel selection and launch shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poseidon254Config {
    pub kernels: Poseidon254Kernels,
    /// Threads per threadgroup; 0 = the PSO's limit (at most 256).
    pub tg: u32,
    /// Fold layers of at most this many outputs run on the CPU (0: never).
    pub cpu_tail: u32,
    /// `hash_rows` of at most this many rows runs on the CPU (0: never).
    pub cpu_rows: u32,
}

/// CPU crossovers, off by default (M5 Max, notes/Z10.md). One permutation on the GPU is ~0.85 ms
/// of latency, so the top of a fold tree is latency-bound: timed alone (synced before and after,
/// `tests_z10::bench_kernels`) a 2^20 tree takes 66.0 ms all on the GPU and 59.9 ms with the
/// layers of <= 1024 outputs on the CPU, the FRI trees of 2^12 / 2^8 leaves 10.4 -> 4.2 ms and
/// 6.9 -> 0.9 ms, and 256 x 64 rows 3.8 -> 1.2 ms. A whole po2-18 p254 proof, A/B interleaved (7
/// pairs x 5 proofs, and 7 x 3), did not change (0.896-0.904 s either way), so the CPU stays out
/// of the default path; the knobs remain for other GPUs.
pub const DEFAULT_CPU_TAIL: u32 = 0;
pub const DEFAULT_CPU_ROWS: u32 = 0;

impl Default for Poseidon254Config {
    fn default() -> Self {
        Self {
            kernels: Poseidon254Kernels::W64,
            tg: 0,
            cpu_tail: DEFAULT_CPU_TAIL,
            cpu_rows: DEFAULT_CPU_ROWS,
        }
    }
}

impl Poseidon254Config {
    /// The defaults with the environment overrides applied.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        let var = |name: &str| {
            std::env::var(name)
                .ok()
                .map(|v| v.trim().to_ascii_lowercase())
        };
        match var("RISC0_METAL_P254_KERNELS").as_deref() {
            Some("w64") => cfg.kernels = Poseidon254Kernels::W64,
            Some("w64c") => cfg.kernels = Poseidon254Kernels::W64Cios,
            Some("ref") => cfg.kernels = Poseidon254Kernels::Ref,
            Some(other) => {
                tracing::warn!("RISC0_METAL_P254_KERNELS={other}: unknown, using default")
            }
            None => {}
        }
        let num = |name: &str, dst: &mut u32| {
            if let Some(v) = var(name) {
                match v.parse() {
                    Ok(n) => *dst = n,
                    Err(_) => tracing::warn!("{name}={v}: not a number, ignored"),
                }
            }
        };
        num("RISC0_METAL_P254_TG", &mut cfg.tg);
        num("RISC0_METAL_P254_CPU_TAIL", &mut cfg.cpu_tail);
        num("RISC0_METAL_P254_CPU_ROWS", &mut cfg.cpu_rows);
        cfg
    }
}

pub struct MetalHashPoseidon254 {
    suite: HashSuite<BabyBear>,
    cfg: Mutex<Poseidon254Config>,
}

impl MetalHashPoseidon254 {
    /// The current kernel selection.
    pub fn config(&self) -> Poseidon254Config {
        *self.cfg.lock().unwrap()
    }

    /// Replace the kernel selection (A/B runs, tests).
    pub fn set_config(&self, cfg: Poseidon254Config) {
        *self.cfg.lock().unwrap() = cfg;
    }

    fn grid(hal: &MetalHal<Self>, name: &str, cfg: &Poseidon254Config, n: u64) -> Grid {
        if cfg.tg == 0 {
            return Grid::Threads(n);
        }
        let max = hal.pipeline(name).max_total_threads_per_threadgroup();
        let bs = (cfg.tg as u64).clamp(1, max).min(n);
        Grid::Groups {
            groups: MTLSize::new(n.div_ceil(bs), 1, 1),
            threads: MTLSize::new(bs, 1, 1),
        }
    }

    /// Encode `io[out + i] = H(io[input + 2i], io[input + 2i + 1])`, `i < n`.
    pub(crate) fn encode_fold(
        hal: &MetalHal<Self>,
        cfg: &Poseidon254Config,
        io: &BufferImpl<Digest>,
        out: usize,
        input: usize,
        n: usize,
    ) {
        if n == 0 {
            return;
        }
        let n32 = u32::try_from(n).expect("hash_fold: too many outputs");
        let args = &[
            io.as_arg_with_offset(out),
            io.as_arg_with_offset(input),
            KernelArg::Integer(n32),
        ];
        let name = format!("p254_fold_{}", cfg.kernels.suffix());
        let grid = Self::grid(hal, &name, cfg, n as u64);
        hal.dispatch(hal.pipeline(&name), args, grid, &[]);
    }

    /// Encode `hash_rows` over a column-major `rows x cols` matrix.
    pub(crate) fn encode_rows(
        hal: &MetalHal<Self>,
        cfg: &Poseidon254Config,
        output: &BufferImpl<Digest>,
        matrix: &BufferImpl<BabyBearElem>,
        rows: usize,
        cols: usize,
    ) {
        if rows == 0 {
            return;
        }
        let rows32 = u32::try_from(rows).expect("hash_rows: too many rows");
        let cols32 = u32::try_from(cols).expect("hash_rows: too many columns");
        let args = &[
            output.as_arg(),
            matrix.as_arg(),
            KernelArg::Integer(rows32),
            KernelArg::Integer(cols32),
        ];
        let name = format!("p254_rows_{}", cfg.kernels.suffix());
        let grid = Self::grid(hal, &name, cfg, rows as u64);
        hal.dispatch(hal.pipeline(&name), args, grid, &[]);
    }
}

impl MetalHashPoseidon254 {
    /// Fold layers `n, n / 2, ..` down to the first one of at most `last` outputs (or 1) on the
    /// CPU: the layer of `m` outputs writes `io[m..2m]` from `io[2m..4m]`. The host view syncs
    /// the HAL first.
    fn cpu_fold_layers(&self, io: &BufferImpl<Digest>, mut n: usize, last: usize) {
        let hashfn = self.suite.hashfn.clone();
        io.view_mut(|v| loop {
            let (lo, hi) = v.split_at_mut(2 * n);
            let out = &mut lo[n..2 * n];
            let input = &hi[..2 * n];
            out.par_iter_mut().enumerate().for_each(|(i, d)| {
                *d = *hashfn.hash_pair(&input[2 * i], &input[2 * i + 1]);
            });
            if n <= last || n == 1 {
                break;
            }
            n /= 2;
        });
    }

    /// `hash_rows` on the CPU (the CPU HAL's loop over the host views).
    fn cpu_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        let rows = output.size();
        let cols = matrix.size() / rows;
        let hashfn = self.suite.hashfn.clone();
        matrix.view(|m| {
            output.view_mut(|o| {
                o.par_iter_mut().enumerate().for_each(|(r, d)| {
                    let col: Vec<BabyBearElem> = (0..cols).map(|c| m[c * rows + r]).collect();
                    *d = *hashfn.hash_elem_slice(&col);
                });
            })
        });
    }
}

impl MetalHash for MetalHashPoseidon254 {
    fn new(_hal: &MetalHal<Self>) -> Self {
        MetalHashPoseidon254 {
            suite: Poseidon254HashSuite::new_suite(),
            cfg: Mutex::new(Poseidon254Config::from_env()),
        }
    }

    fn hash_fold(&self, hal: &MetalHal<Self>, io: &BufferImpl<Digest>, output_size: usize) {
        let cfg = self.config();
        if output_size > 0 && output_size <= cfg.cpu_tail as usize {
            self.cpu_fold_layers(io, output_size, output_size);
            return;
        }
        Self::encode_fold(hal, &cfg, io, output_size, 2 * output_size, output_size);
    }

    /// One dispatch per layer (api.cu `sppark_poseidon254_fold_tree`) while a layer has more
    /// than `cpu_tail` outputs, then the rest of the tree on the CPU.
    fn hash_fold_tree(&self, hal: &MetalHal<Self>, io: &BufferImpl<Digest>, layers: usize) {
        let cfg = self.config();
        for i in (0..layers).rev() {
            let n = 1usize << i;
            if n <= cfg.cpu_tail as usize {
                self.cpu_fold_layers(io, n, 1);
                return;
            }
            Self::encode_fold(hal, &cfg, io, n, 2 * n, n);
        }
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
        if row_size <= cfg.cpu_rows as usize {
            self.cpu_rows(output, matrix);
            return;
        }
        Self::encode_rows(hal, &cfg, output, matrix, row_size, col_size);
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}
