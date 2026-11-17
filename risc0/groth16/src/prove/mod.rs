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

//! # Groth16 Prover

#[cfg(any(feature = "rocm", feature = "intel", feature = "metal"))]
mod component;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(not(any(feature = "cuda", feature = "rocm")))]
mod docker;
#[cfg(all(
    any(feature = "intel", feature = "metal"),
    not(any(feature = "cuda", feature = "rocm"))
))]
mod fast;
#[cfg(feature = "rocm")]
mod hip;
#[cfg(all(feature = "intel", not(any(feature = "cuda", feature = "rocm"))))]
mod intel;
#[cfg(all(
    feature = "metal",
    not(any(feature = "cuda", feature = "rocm", feature = "intel"))
))]
mod metal;
#[cfg(all(
    feature = "metal",
    not(any(feature = "cuda", feature = "rocm", feature = "intel"))
))]
mod metal_witness;
mod seal_format;
mod seal_to_json;
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    all(feature = "metal", not(feature = "intel"))
))]
mod witness_fast;
#[cfg(any(
    feature = "cuda",
    all(feature = "metal", not(any(feature = "rocm", feature = "intel")))
))]
mod witness_par;

use anyhow::Result;

use crate::Seal;

pub use self::seal_to_json::to_json;

/// Preload the circom graph for Groth16 witness calculation.
/// Call on a background thread to overlap the ~728ms graph read+parse
/// with other work (e.g. composite_to_succinct recursion proving).
pub fn preload_graph() -> Result<()> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::cuda::preload_graph()
        } else if #[cfg(feature = "rocm")] {
            self::hip::preload_graph()
        } else if #[cfg(feature = "intel")] {
            self::fast::preload()
        } else if #[cfg(feature = "metal")] {
            if self::metal::enabled() {
                self::metal::preload_graph()
            } else if self::docker::fast_enabled() {
                self::fast::preload()
            } else {
                Ok(())
            }
        } else {
            Ok(()) // docker mode doesn't use local graph
        }
    }
}

/// Start [preload_graph] on a background thread (CUDA with the cold-path optimizations on, see
/// `RISC0_CUDA_G16_COLD`; no-op otherwise), e.g. while the guest executes.
pub fn preload_graph_background() {
    #[cfg(feature = "cuda")]
    if self::cuda::cold_path_enabled() {
        std::thread::spawn(|| {
            let _ = self::cuda::preload_graph();
        });
    }
    #[cfg(all(
        feature = "metal",
        not(any(feature = "cuda", feature = "rocm", feature = "intel"))
    ))]
    if self::metal::enabled() {
        // Item P31-G1: on the caller's thread (the one that will prestart and wrap): drops a
        // reservation its previous proof's prestart left without a wrap.
        self::metal::proof_start();
        std::thread::spawn(|| {
            let _ = self::metal::preload_graph();
        });
    }
}

/// Item W7-G1: drop the cached Groth16 witness graph (~0.9 GB live) if no wrap is using it and no
/// build is running; the next wrap's [prestart_prover] rebuilds it (0.7-0.9 s measured with its
/// file digest known, beside identity_p254). Never waits, never takes the GPU lock: meant for a
/// memory-pressure drain (`risc0_zkp::hal::metal::register_idle_releaser`).
///
/// "In use" covers a whole wrap, from its start (so also its `pre_gpu`, key load and GPU-lock
/// wait, before it touches the graph), and the graph [prestart_prover] built for the wrap that
/// follows it (until that wrap starts, the same thread's next proof start or prestart, or 30 s,
/// `RISC0_G16_GRAPH_RESERVE_MS`, whichever comes first; item P31-G1): a drain there would
/// only move the rebuild onto the wrap's critical path, not lower the wrap's peak.
///
/// It drops the graph only where that lowers the footprint: when the process was **spawned** with
/// `MallocLargeCache=0` (libmalloc reads it at process start; a later `set_var` does not count),
/// or with `RISC0_G16_GRAPH_RELEASE=1`; `RISC0_G16_GRAPH_RELEASE=0` turns it off. Otherwise (the
/// macOS default) it is a no-op: with libmalloc's large cache on, a drop frees nothing and each
/// rebuild leaves ~580 MiB more behind, so a drain followed by a wrap would grow the process.
/// Returns the bytes freed (0 when nothing was idle, the drop is not allowed, or off Metal).
pub fn release_graph() -> u64 {
    #[cfg(all(
        feature = "metal",
        not(any(feature = "cuda", feature = "rocm", feature = "intel"))
    ))]
    if self::metal::enabled() {
        return self::metal::release_graph();
    }
    0
}

/// Start the Groth16 prover ahead of [shrink_wrap] so that it loads its
/// proving key while the caller finishes `identity_p254` (intel and metal builds
/// with the rzup `risc0-groth16` component installed; otherwise a no-op).
/// Item W7-G1 (Metal, graph release on): also where the witness graph is built for the wrap.
pub fn prestart_prover() -> Result<()> {
    cfg_if::cfg_if! {
        if #[cfg(all(feature = "intel", not(any(feature = "cuda", feature = "rocm"))))] {
            if self::intel::enabled() {
                // Uploads the proving key to the GPU in the background.
                return self::intel::prestart();
            }
            // Only check that the graph exists here: building the witness
            // program (~1 s) stays on the preload thread, off this path.
            if self::docker::fast_enabled()
                && self::fast::graph_path().map(|p| p.exists()).unwrap_or(false)
            {
                self::fast::prestart()?;
            }
            Ok(())
        } else if #[cfg(feature = "metal")] {
            if self::metal::enabled() {
                return self::metal::prestart();
            }
            if self::docker::fast_enabled()
                && self::fast::graph_path().map(|p| p.exists()).unwrap_or(false)
            {
                self::fast::prestart()?;
            }
            Ok(())
        } else {
            Ok(())
        }
    }
}

/// Start loading the Groth16 proving key in the background (CUDA; no-op elsewhere, on cards
/// with less than 30 GB, or with `RISC0_CUDA_G16_COLD=0`). The loader waits for
/// [prestart_key_go] (or 2 s).
pub fn prestart_key() {
    #[cfg(feature = "cuda")]
    self::cuda::prestart_key();
    #[cfg(all(
        feature = "metal",
        not(any(feature = "cuda", feature = "rocm", feature = "intel"))
    ))]
    if self::metal::enabled() {
        self::metal::prestart_key();
    }
}

/// Release a pending [prestart_key] loader (no-op if none is waiting).
pub fn prestart_key_go() {
    #[cfg(feature = "cuda")]
    self::cuda::prestart_key_go();
    #[cfg(all(
        feature = "metal",
        not(any(feature = "cuda", feature = "rocm", feature = "intel"))
    ))]
    if self::metal::enabled() {
        self::metal::prestart_key_go();
    }
}

/// Preload the SRS and Groth16 prover into C++ static cache.
/// Call after STARK proving is complete and GPU is idle.
pub fn preload_srs() -> Result<()> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::cuda::preload_srs()
        } else if #[cfg(feature = "rocm")] {
            Ok(()) // hip.rs handles SRS loading internally
        } else if #[cfg(feature = "intel")] {
            if self::intel::enabled() {
                self::intel::preload_srs()
            } else {
                Ok(())
            }
        } else if #[cfg(feature = "metal")] {
            if self::metal::enabled() {
                self::metal::preload_srs()
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }
}

/// Free the Groth16 prover and SRS that `shrink_wrap` caches on the GPU, returning that memory to
/// the device. The next `shrink_wrap` rebuilds them; measured, that adds at most ~0.1 s per proof
/// (2.08-2.12 s wraps on an RTX 5080 releasing every time, against 2.01-2.05 s on an RTX 4090 keeping
/// the cache).
///
/// The cache otherwise lives as long as the process, so every later STARK proof on that device has to
/// fit beside it. On a card without the room — measured on a 16 GB RTX 5080, a 46M-cycle STARK at po2
/// 20 peaked at 15.8 of 16.3 GB with the cache resident — call this after each proof. On a card with
/// room, keeping the cache is faster. Deciding which is the caller's business, since only the caller
/// knows what it will prove next.
pub fn release_srs() -> Result<()> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::cuda::release_srs()
        } else if #[cfg(feature = "rocm")] {
            self::hip::release_srs()
        } else if #[cfg(feature = "intel")] {
            if self::intel::enabled() {
                self::intel::release_srs()
            } else {
                Ok(())
            }
        } else if #[cfg(feature = "metal")] {
            if self::metal::enabled() {
                self::metal::release_srs()
            } else {
                Ok(())
            }
        } else {
            // Docker mode holds nothing.
            Ok(())
        }
    }
}

/// Discard the Groth16 prover's derived-data caches: the files the CUDA library keeps in its cache
/// directory (compact coefficients and split-MSM corrections, normally validated by format, key
/// and checksum) and the prover cached in memory. For a caller whose Groth16 seal failed
/// verification: the next [shrink_wrap] rebuilds them from the proving key. No-op without CUDA.
pub fn invalidate_caches() {
    #[cfg(feature = "cuda")]
    self::cuda::invalidate_caches();
    #[cfg(all(
        feature = "metal",
        not(any(feature = "cuda", feature = "rocm", feature = "intel"))
    ))]
    if self::metal::enabled() {
        self::metal::invalidate_caches();
    }
}

/// Like [shrink_wrap], but runs `pre_gpu` on the calling thread while the witness is computed on
/// a helper thread (CUDA and the Metal GPU prover; elsewhere `pre_gpu` runs first). `pre_gpu` has returned before the GPU
/// part of the wrap starts. Used to overlap releasing the STARK HAL's buffer pool (thread-local,
/// ~35 ms of frees) with the CPU-only witness generation.
pub fn shrink_wrap_with(identity_p254_seal_bytes: &[u8], pre_gpu: impl FnOnce()) -> Result<Seal> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::cuda::shrink_wrap_with(identity_p254_seal_bytes, pre_gpu)
        } else if #[cfg(all(feature = "metal", not(any(feature = "rocm", feature = "intel"))))] {
            if self::metal::enabled() {
                self::metal::shrink_wrap_with(identity_p254_seal_bytes, pre_gpu)
            } else {
                pre_gpu();
                shrink_wrap(identity_p254_seal_bytes)
            }
        } else {
            pre_gpu();
            shrink_wrap(identity_p254_seal_bytes)
        }
    }
}

/// Produce a Groth16 proof from an `identity_p254` seal.
pub fn shrink_wrap(identity_p254_seal_bytes: &[u8]) -> Result<Seal> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::cuda::shrink_wrap(identity_p254_seal_bytes)
        } else if #[cfg(feature = "rocm")] {
            self::hip::shrink_wrap(identity_p254_seal_bytes)
        } else if #[cfg(feature = "intel")] {
            if self::intel::enabled() {
                self::intel::shrink_wrap(identity_p254_seal_bytes)
            } else {
                self::docker::shrink_wrap(identity_p254_seal_bytes)
            }
        } else if #[cfg(feature = "metal")] {
            if self::metal::enabled() {
                self::metal::shrink_wrap(identity_p254_seal_bytes)
            } else {
                self::docker::shrink_wrap(identity_p254_seal_bytes)
            }
        } else {
            self::docker::shrink_wrap(identity_p254_seal_bytes)
        }
    }
}
