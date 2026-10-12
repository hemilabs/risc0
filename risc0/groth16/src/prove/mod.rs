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
#[cfg(not(any(feature = "cuda", feature = "rocm")))]
mod docker;
#[cfg(all(feature = "intel", not(any(feature = "cuda", feature = "rocm"))))]
mod fast;
#[cfg(feature = "rocm")]
mod hip;
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
        } else if #[cfg(feature = "intel")] {
            self::fast::preload()
        } else {
            Ok(()) // docker mode doesn't use local graph
        }
    }
}

/// Start the Groth16 prover ahead of [shrink_wrap] so that it loads its
/// proving key while the caller finishes `identity_p254` (intel builds with
/// the rzup `risc0-groth16` component installed; otherwise a no-op).
pub fn prestart_prover() -> Result<()> {
    cfg_if::cfg_if! {
        if #[cfg(all(feature = "intel", not(any(feature = "cuda", feature = "rocm"))))] {
            // Only check that the graph exists here: building the witness
            // program (~1 s) stays on the preload thread, off this path.
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
