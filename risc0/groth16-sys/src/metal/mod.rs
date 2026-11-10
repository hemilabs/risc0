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

//! The Groth16 prover on the Apple GPU (`metal` feature; docs/metal-port/PLAN.md 4.5).
//!
//! The entry points mirror the Intel prover's (`intel_preload`/`intel_prove`/`intel_release`):
//! same parameter structs, same files (`proof.json`/`public.json` under the prover parameters'
//! directory), same caching contract (the key stays loaded until [metal_release]).
//!
//! Not implemented yet (item G1 is the plumbing; the kernels and host orchestration are G2..G7):
//! [metal_supported], [metal_preload] and [metal_prove] return a [MetalNotImplemented] error,
//! which `risc0-groth16` recognises (`err.is::<MetalNotImplemented>()`) to fall back to the Docker
//! prover. [metal_release] has nothing to free and succeeds.
//!
//! Callers in `risc0-groth16` hold `risc0_zkp::hal::metal::exclusive()` around [metal_prove] and
//! [metal_release]. [metal_preload] may run on a background thread without that lock (as
//! `prestart` does, like the CUDA prestart), so its key cache needs its own lock.

use std::fmt;

use anyhow::Result;

use crate::{ProverParams, SetupParams};

pub mod msm;

/// The error every entry point returns until the Metal Groth16 prover exists. Callers fall back
/// to another prover when they see it (`anyhow::Error::is::<MetalNotImplemented>()`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalNotImplemented;

impl fmt::Display for MetalNotImplemented {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            "the Metal Groth16 prover is not implemented yet (risc0-groth16-sys metal module); \
             use the Docker prover",
        )
    }
}

impl std::error::Error for MetalNotImplemented {}

/// Whether this build can prove on the GPU: `Ok` once the Metal prover exists and a Metal device
/// is present, else an error ([MetalNotImplemented] today). Cheap: needs no proving key, so
/// callers check it before computing a witness they could not use.
pub fn metal_supported() -> Result<()> {
    Err(MetalNotImplemented.into())
}

/// Load the proving key and coefficients onto the GPU, where they stay cached until
/// [metal_release]. [metal_prove] does this itself if needed.
pub fn metal_preload(setup_params: &SetupParams) -> Result<()> {
    let _ = setup_params;
    Err(MetalNotImplemented.into())
}

/// Prove with the Metal GPU prover; writes the proof to `prover_params.proof_path` (and the
/// public inputs to `prover_params.public_path`) in the CUDA prover's JSON format.
pub fn metal_prove(prover_params: &ProverParams, setup_params: &SetupParams) -> Result<()> {
    let _ = (prover_params, setup_params);
    Err(MetalNotImplemented.into())
}

/// Free the GPU prover's cached proving key, coefficients and scratch. Nothing is cached yet, so
/// this succeeds (a caller releasing memory between proofs must not fail).
pub fn metal_release() -> Result<()> {
    Ok(())
}

#[cfg(test)]
#[path = "tests.rs"]
mod bn254_tests;

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn entry_points_report_not_implemented() {
        let dir = Path::new("/nonexistent-risc0-groth16");
        let setup = SetupParams::new(dir).unwrap();
        let params = ProverParams::new(dir, std::ptr::null()).unwrap();

        for err in [
            metal_supported().unwrap_err(),
            metal_preload(&setup).unwrap_err(),
            metal_prove(&params, &setup).unwrap_err(),
        ] {
            assert!(err.is::<MetalNotImplemented>(), "{err:#}");
            assert!(err.to_string().contains("not implemented"), "{err:#}");
        }
        metal_release().unwrap();
    }

    #[test]
    fn kernel_names_match_library() {
        // Until the first kernel lands the library and the name list are both empty; afterwards
        // the library is a real metallib (magic "MTLB") and names are unique.
        // (RISC0_SKIP_BUILD_KERNELS embeds an empty library whatever the sources.)
        if crate::METAL_KERNEL_NAMES.is_empty() {
            assert!(crate::METAL_LIB.is_empty());
        }
        if !crate::METAL_LIB.is_empty() {
            assert_eq!(&crate::METAL_LIB[..4], b"MTLB");
        }
        let mut names = crate::METAL_KERNEL_NAMES.to_vec();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), crate::METAL_KERNEL_NAMES.len());

        // Every listed name is a function of the library (what `MetalHal::load_library` checks).
        if crate::METAL_LIB.is_empty() {
            return;
        }
        let Some(device) = ::metal::Device::system_default() else {
            return;
        };
        let lib = device.new_library_with_data(crate::METAL_LIB).unwrap();
        let functions = lib.function_names();
        for name in crate::METAL_KERNEL_NAMES {
            assert!(
                functions.iter().any(|f| f == name),
                "{name} not in {functions:?}"
            );
        }
    }
}
