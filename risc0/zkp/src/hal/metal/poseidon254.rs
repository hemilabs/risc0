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
//!   range, fused dot products, SOS squaring; compact since item C16) | `w64u` (the same fully
//!   unrolled: the pre-C16 `w64`) | `w64c` (squaring by multiplication) | `ref` (bn254.h's fully
//!   reduced 32-bit CIOS `Fr`, the CUDA kernel's arithmetic).
//! * `RISC0_METAL_P254_TG`: threads per threadgroup (default: the PSO's limit, at most 256).
//! * `RISC0_METAL_P254_CPU_TAIL`: fold layers (`hash_fold`, the top of `hash_fold_tree`) of at
//!   most this many outputs run on the CPU (after a sync; 0 = never). One GPU permutation takes
//!   ~0.85 ms of latency, so the top of a tree is latency-bound on the GPU.
//! * `RISC0_METAL_P254_CPU_ROWS`: `hash_rows` with at most this many rows runs on the CPU (0 =
//!   never).
//! * `RISC0_METAL_P254_LANES`: fold layers and `hash_rows` calls of at most this many hashes run
//!   4 SIMD lanes per hash (`p254_fold4_w64`, `p254_rows4_w64`: latency, not throughput, bounds
//!   those; item C16). Default [default_lanes] (128 per GPU core: 4096 on a 32-core M5 Max);
//!   0 = never (the pre-C16 kernels).

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
    /// products, SOS squaring, compact (loops over one copy of the S-box, matrix-row and word
    /// code; item C16) (default).
    W64,
    /// `p254_*_w64u`: `W64` fully unrolled (the pre-C16 `W64`; ~6x the native code).
    W64Unrolled,
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
            Poseidon254Kernels::W64Unrolled => "w64u",
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
    /// `hash_fold` layers / `hash_rows` calls of at most this many hashes use the 4-lanes-per-hash
    /// kernels `p254_fold4_w64` / `p254_rows4_w64` (0: never; not with `Ref`).
    pub lanes: u32,
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
/// Layers of at most this many hashes per GPU core run 4 lanes per hash (`p254_fold4_w64`,
/// `p254_rows4_w64`; item C16, notes/C16.md). One hash per thread leaves a small layer latency-
/// bound (~0.87 ms per permutation on an M5 Max however few hashes there are), and the 4-lane
/// kernels cut that latency by ~38 %, but they do ~2.7x the arithmetic, so they lose once 4 lanes
/// per hash fill the GPU. Measured on the 32-core M5 Max: 4096 and 8192 hashes are equally good
/// (fold trees -4 to -5 ms per 2^20 tree, FRI trees -35 to -38 %), 16384 is worse; 128 per core
/// is the low end of that range, so a smaller GPU switches at a proportionally smaller layer.
pub const LANES_PER_CORE: u32 = 128;
/// The 4-lane threshold when the GPU's core count is unknown (that of an 8-core GPU).
pub const LANES_FALLBACK: u32 = 1024;

/// The 4-lane threshold for this machine: [LANES_PER_CORE] x the GPU's cores.
pub fn default_lanes() -> u32 {
    static LANES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LANES.get_or_init(|| {
        iokit::gpu_core_count().map_or(LANES_FALLBACK, |c| c.saturating_mul(LANES_PER_CORE))
    })
}

/// The GPU's core count from the IORegistry (`AGXAccelerator`'s `gpu-core-count`, the number
/// System Information reports), or None.
mod iokit {
    use std::ffi::{c_char, c_void};

    type CFTypeRef = *const c_void;

    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IOServiceMatching(name: *const c_char) -> *mut c_void;
        fn IOServiceGetMatchingService(main_port: u32, matching: *mut c_void) -> u32;
        fn IORegistryEntryCreateCFProperty(
            entry: u32,
            key: CFTypeRef,
            allocator: CFTypeRef,
            options: u32,
        ) -> CFTypeRef;
        fn IOObjectRelease(object: u32) -> i32;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFStringCreateWithCString(
            alloc: CFTypeRef,
            s: *const c_char,
            encoding: u32,
        ) -> CFTypeRef;
        fn CFGetTypeID(cf: CFTypeRef) -> usize;
        fn CFNumberGetTypeID() -> usize;
        fn CFNumberGetValue(number: CFTypeRef, the_type: isize, value: *mut c_void) -> u8;
        fn CFRelease(cf: CFTypeRef);
    }

    const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
    const K_CF_NUMBER_SINT64_TYPE: isize = 4;

    pub(super) fn gpu_core_count() -> Option<u32> {
        // SAFETY: plain IOKit / CoreFoundation calls; every object created here is released
        // (IOServiceGetMatchingService consumes `matching`), and the property's type is checked
        // before it is read.
        unsafe {
            let matching = IOServiceMatching(c"AGXAccelerator".as_ptr());
            if matching.is_null() {
                return None;
            }
            let service = IOServiceGetMatchingService(0, matching);
            if service == 0 {
                return None;
            }
            let key = CFStringCreateWithCString(
                std::ptr::null(),
                c"gpu-core-count".as_ptr(),
                K_CF_STRING_ENCODING_UTF8,
            );
            let value = if key.is_null() {
                std::ptr::null()
            } else {
                let v = IORegistryEntryCreateCFProperty(service, key, std::ptr::null(), 0);
                CFRelease(key);
                v
            };
            IOObjectRelease(service);
            if value.is_null() {
                return None;
            }
            let mut n: i64 = 0;
            let ok = CFGetTypeID(value) == CFNumberGetTypeID()
                && CFNumberGetValue(value, K_CF_NUMBER_SINT64_TYPE, (&mut n as *mut i64).cast())
                    != 0;
            CFRelease(value);
            (ok && n > 0 && n <= u32::MAX as i64).then_some(n as u32)
        }
    }
}

impl Default for Poseidon254Config {
    fn default() -> Self {
        Self {
            kernels: Poseidon254Kernels::W64,
            tg: 0,
            cpu_tail: DEFAULT_CPU_TAIL,
            cpu_rows: DEFAULT_CPU_ROWS,
            lanes: default_lanes(),
        }
    }
}

impl Poseidon254Config {
    /// The launch shape of the 4-lane kernels: whole hash groups per threadgroup, i.e. the PSO's
    /// limit (a multiple of 32) or `tg` rounded down to a multiple of 4.
    fn lanes_tg(&self) -> Self {
        Self {
            tg: if self.tg == 0 {
                0
            } else {
                (self.tg & !3).max(4)
            },
            ..*self
        }
    }

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
            Some("w64u") => cfg.kernels = Poseidon254Kernels::W64Unrolled,
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
        num("RISC0_METAL_P254_LANES", &mut cfg.lanes);
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
        let (name, threads) = if cfg.kernels != Poseidon254Kernels::Ref && n <= cfg.lanes as usize {
            ("p254_fold4_w64".to_string(), 4 * n as u64)
        } else {
            (format!("p254_fold_{}", cfg.kernels.suffix()), n as u64)
        };
        let grid = if threads == n as u64 {
            Self::grid(hal, &name, cfg, threads)
        } else {
            Self::grid(hal, &name, &cfg.lanes_tg(), threads)
        };
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
        let (name, grid) = if cfg.kernels != Poseidon254Kernels::Ref && rows <= cfg.lanes as usize {
            let name = "p254_rows4_w64".to_string();
            let grid = Self::grid(hal, &name, &cfg.lanes_tg(), 4 * rows as u64);
            (name, grid)
        } else {
            let name = format!("p254_rows_{}", cfg.kernels.suffix());
            let grid = Self::grid(hal, &name, cfg, rows as u64);
            (name, grid)
        };
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
