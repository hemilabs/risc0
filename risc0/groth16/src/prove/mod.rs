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

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "rocm")]
mod hip;
#[cfg(not(any(feature = "cuda", feature = "rocm")))]
mod docker;
mod seal_format;
mod seal_to_json;

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
        } else {
            Ok(()) // docker mode doesn't use local graph
        }
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
        } else {
            // ROCm caches through hip.rs's own path, untested here; docker mode holds nothing.
            Ok(())
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
        } else {
            self::docker::shrink_wrap(identity_p254_seal_bytes)
        }
    }
}
