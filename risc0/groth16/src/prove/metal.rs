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

//! Groth16 on the Apple GPU (`metal` feature).
//!
//! Stub (docs/metal-port/PLAN.md, item F2): [AVAILABLE] is false, so [enabled] is false and every
//! entry point in [super] takes its Docker/no-op path. The Metal Groth16 items (G1..G7) fill in
//! this module and turn [AVAILABLE] on; the kernels are in `risc0_groth16_sys::METAL_LIB`.
//! `RISC0_GROTH16_METAL_PROVER=docker`, `RISC0_METAL=0` or `RISC0_METAL_SKIP=groth16` select
//! the Docker prover.

use anyhow::{bail, Result};

use crate::Seal;

/// Whether this build has a Metal Groth16 prover (false until the Metal Groth16 items land).
const AVAILABLE: bool = false;

/// Whether to prove on the GPU rather than with the Docker prover.
pub(crate) fn enabled() -> bool {
    AVAILABLE
        && risc0_zkp::hal::metal::enabled("groth16")
        && std::env::var("RISC0_GROTH16_METAL_PROVER").as_deref() != Ok("docker")
}

/// Load the witness graph (in-process witness generation).
pub(crate) fn preload_graph() -> Result<()> {
    super::fast::preload()
}

/// Start loading the proving key onto the GPU in the background.
pub(crate) fn prestart() -> Result<()> {
    Ok(())
}

/// Start loading the proving key early (see `super::prestart_key`).
pub(crate) fn prestart_key() {}

/// Release a pending [prestart_key] loader.
pub(crate) fn prestart_key_go() {}

/// Load the proving key onto the GPU now.
pub(crate) fn preload_srs() -> Result<()> {
    Ok(())
}

/// Free the proving key and scratch cached on the GPU.
pub(crate) fn release_srs() -> Result<()> {
    Ok(())
}

/// Discard the prover's derived-data caches.
pub(crate) fn invalidate_caches() {}

/// Like [shrink_wrap], running `pre_gpu` before the GPU part of the wrap.
pub(crate) fn shrink_wrap_with(
    identity_p254_seal_bytes: &[u8],
    pre_gpu: impl FnOnce(),
) -> Result<Seal> {
    pre_gpu();
    shrink_wrap(identity_p254_seal_bytes)
}

/// The in-process witness (as the intel prover computes it) that the Metal prover will consume.
#[allow(dead_code)] // used once the Metal prover lands
pub(crate) fn witness(identity_p254_seal_bytes: &[u8]) -> Result<Vec<wtns_file::FieldElement<32>>> {
    super::fast::witness(identity_p254_seal_bytes)
}

pub(crate) fn shrink_wrap(_identity_p254_seal_bytes: &[u8]) -> Result<Seal> {
    bail!("the Metal Groth16 prover is not implemented yet")
}
