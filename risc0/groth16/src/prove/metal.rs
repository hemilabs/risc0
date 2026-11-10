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

//! Groth16 on the Apple GPU (`metal` feature): the Metal prover in `risc0-groth16-sys`
//! (`src/metal`, `kernels/metal`), with the witness computed in-process (see [super::fast]).
//! Adapted from [intel.rs](super) (docs/metal-port/PLAN.md 4.5, item G1).
//!
//! It uses the rzup `risc0-groth16` component's proving key, preprocessed coefficients and
//! fuzzed MSM results, as the CUDA and Intel provers do, and produces the same kind of proof.
//!
//! Docker fallback:
//! - `RISC0_GROTH16_METAL_PROVER=docker`, `RISC0_METAL=0` or `RISC0_METAL_SKIP=groth16` select the
//!   Docker prover ([enabled] is false and every entry point in [super] takes its Docker path).
//! - While the Metal prover is not implemented, `risc0_groth16_sys::metal_supported()` fails, so
//!   [enabled] is false too. If a GPU entry point still reports
//!   [MetalNotImplemented](risc0_groth16_sys::MetalNotImplemented), [shrink_wrap] falls back to the
//!   Docker prover rather than failing.
//!
//! Locking: the GPU part of a wrap (waiting for the key upload, the prove, reading the proof) and
//! [release_srs] run under `risc0_zkp::hal::metal::exclusive()`, as CUDA runs its prove under the
//! CUDA singleton. The witness is computed before the lock is taken, so STARK proofs on other
//! threads are not blocked by CPU work. The key loader thread does not take the lock (it would
//! deadlock against the waiting holder, see `exclusive()`); `metal_preload` must therefore not use
//! a `MetalHal`.

use std::path::Path;

use anyhow::{anyhow, Context as _, Result};
use risc0_groth16_sys::{MetalNotImplemented, ProverParams, SetupParams};
use tempfile::tempdir;

use super::component::component_dir;
use crate::{ProofJson, Seal};

/// Whether to prove on the GPU rather than with the Docker prover.
pub(crate) fn enabled() -> bool {
    select_gpu(
        risc0_zkp::hal::metal::enabled("groth16"),
        std::env::var("RISC0_GROTH16_METAL_PROVER").ok().as_deref(),
        risc0_groth16_sys::metal_supported().is_ok(),
    )
}

/// [enabled] from its inputs: the Metal backend allowed for groth16 (`RISC0_METAL`,
/// `RISC0_METAL_SKIP`), `RISC0_GROTH16_METAL_PROVER`, and whether this build has a Metal prover.
fn select_gpu(metal_enabled: bool, prover: Option<&str>, supported: bool) -> bool {
    metal_enabled && prover != Some("docker") && supported
}

/// Whether `err` (or any error it wraps) says the Metal prover is not implemented.
fn is_not_implemented(err: &anyhow::Error) -> bool {
    err.chain().any(|e| e.is::<MetalNotImplemented>())
}

fn preload(dir: &Path) -> Result<()> {
    let setup = SetupParams::new(dir).context("failed to create groth16 setup params")?;
    risc0_groth16_sys::metal_preload(&setup).context("failed to load the groth16 proving key")
}

/// Load the witness graph (in-process witness generation).
pub(crate) fn preload_graph() -> Result<()> {
    super::fast::preload()
}

/// Start loading the proving key onto the GPU in the background.
pub(crate) fn prestart() -> Result<()> {
    let dir = component_dir()?;
    std::thread::spawn(move || match preload(&dir) {
        Ok(()) => {}
        Err(e) if is_not_implemented(&e) => tracing::debug!("groth16 metal preload: {e:#}"),
        Err(e) => tracing::warn!("groth16 proving key preload failed: {e:#}"),
    });
    Ok(())
}

/// Start loading the proving key early. CUDA-only gating (its loader waits for
/// [prestart_key_go]); on Metal, [prestart] already loads in the background, so a no-op.
pub(crate) fn prestart_key() {}

/// Release a pending [prestart_key] loader (no-op on Metal).
pub(crate) fn prestart_key_go() {}

/// Load the proving key onto the GPU now.
pub(crate) fn preload_srs() -> Result<()> {
    let _gpu = risc0_zkp::hal::metal::exclusive();
    match preload(&component_dir()?) {
        // Nothing to load: shrink_wrap will use the Docker prover.
        Err(e) if is_not_implemented(&e) => Ok(()),
        r => r,
    }
}

/// Free the proving key, coefficients and scratch cached on the GPU.
pub(crate) fn release_srs() -> Result<()> {
    let _gpu = risc0_zkp::hal::metal::exclusive();
    risc0_groth16_sys::metal_release().context("failed to release the groth16 prover")
}

/// Discard the prover's caches. The Metal prover keeps no derived data on disk, so this frees
/// the in-memory prover; the next [shrink_wrap] reloads it.
pub(crate) fn invalidate_caches() {
    if let Err(e) = release_srs() {
        tracing::warn!("{e:#}");
    }
}

/// Produce a Groth16 proof from an `identity_p254` seal.
pub(crate) fn shrink_wrap(identity_p254_seal_bytes: &[u8]) -> Result<Seal> {
    shrink_wrap_with(identity_p254_seal_bytes, || {})
}

/// [shrink_wrap], running `pre_gpu` on the calling thread while the witness is computed on a
/// scoped helper thread. `pre_gpu` runs exactly once, and has returned before the GPU part of the
/// wrap starts (or before the Docker prover, on fallback).
pub(crate) fn shrink_wrap_with(
    identity_p254_seal_bytes: &[u8],
    pre_gpu: impl FnOnce(),
) -> Result<Seal> {
    let mut pre_gpu = Some(pre_gpu);
    match gpu_shrink_wrap(identity_p254_seal_bytes, &mut pre_gpu) {
        Err(e) if is_not_implemented(&e) => {
            tracing::info!("{e:#}; using the Docker Groth16 prover");
            if let Some(f) = pre_gpu.take() {
                f();
            }
            super::docker::shrink_wrap(identity_p254_seal_bytes)
        }
        r => r,
    }
}

fn gpu_shrink_wrap(
    identity_p254_seal_bytes: &[u8],
    pre_gpu: &mut Option<impl FnOnce()>,
) -> Result<Seal> {
    // Before any work the fallback would waste (and before needing the component).
    risc0_groth16_sys::metal_supported()?;

    let dir = component_dir()?;
    let setup = SetupParams::new(&dir).context("failed to create groth16 setup params")?;

    let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
    let t0 = std::time::Instant::now();
    let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1000.0;

    // The proving key's upload (first use only) overlaps the witness computation.
    let loader = {
        let dir = dir.clone();
        std::thread::spawn(move || preload(&dir))
    };
    let witness = std::thread::scope(|sc| {
        let h = sc.spawn(|| super::fast::witness(identity_p254_seal_bytes));
        if let Some(f) = pre_gpu.take() {
            f();
        }
        h.join()
            .unwrap_or_else(|_| Err(anyhow!("groth16 witness thread panicked")))
    })
    .context("failed to calculate the groth16 witness")?;
    let witness_ms = ms(t0);

    let _gpu = risc0_zkp::hal::metal::exclusive();
    loader
        .join()
        .map_err(|_| anyhow!("the groth16 proving key loader panicked"))??;
    let loaded_ms = ms(t0);

    let work_dir = tempdir().context("failed to create a temporary directory")?;
    let params = ProverParams::new(work_dir.path(), witness.as_ptr() as *const u8)
        .context("failed to create groth16 prover parameters")?;
    risc0_groth16_sys::metal_prove(&params, &setup).context("groth16 prove failed")?;
    if verbose {
        eprintln!(
            "[groth16-metal] witness={witness_ms:.1}ms key loaded={loaded_ms:.1}ms proved={:.1}ms",
            ms(t0)
        );
    }

    let contents = std::fs::read_to_string(params.proof_path.as_path())
        .context("failed to read the groth16 proof")?;
    let proof_json: ProofJson =
        serde_json::from_str(&contents).context("failed to decode the groth16 proof JSON")?;
    proof_json.try_into()
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn gpu_selection() {
        assert!(select_gpu(true, None, true));
        assert!(select_gpu(true, Some("metal"), true));
        assert!(!select_gpu(true, Some("docker"), true));
        assert!(!select_gpu(false, None, true));
        assert!(!select_gpu(true, None, false));
        // Today's build has no Metal prover, so the GPU path is never selected.
        assert!(!enabled());
    }

    #[test]
    fn not_implemented_is_recognised_through_context() {
        let err = anyhow::Error::from(MetalNotImplemented).context("groth16 prove failed");
        assert!(is_not_implemented(&err));
        assert!(!is_not_implemented(&anyhow!("other")));
    }

    /// The Metal path falls back to the Docker prover cleanly: the same result as the Docker
    /// (CPU-build) path, not a panic, and `pre_gpu` runs exactly once. Without Docker installed
    /// both fail with "Please install docker first.".
    #[test]
    fn falls_back_to_docker() {
        let seal: &[u8] = &[];
        let docker = super::super::docker::shrink_wrap(seal).map(|_| ());

        let calls = Cell::new(0);
        let metal = shrink_wrap_with(seal, || calls.set(calls.get() + 1)).map(|_| ());
        assert_eq!(calls.get(), 1);
        let public = crate::prove::shrink_wrap(seal).map(|_| ());
        let calls = Cell::new(0);
        let public_with =
            crate::prove::shrink_wrap_with(seal, || calls.set(calls.get() + 1)).map(|_| ());
        assert_eq!(calls.get(), 1);

        let docker = format!("{:#}", docker.unwrap_err());
        for r in [metal, public, public_with] {
            assert_eq!(format!("{:#}", r.unwrap_err()), docker);
        }
        if !std::process::Command::new("docker")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            assert_eq!(docker, "Please install docker first.");
        }
    }

    #[test]
    fn cache_entry_points_succeed_without_prover() {
        // The component may be absent here; preload_srs then fails on it, which is the Intel
        // behaviour. Release and invalidate never fail.
        if component_dir().is_ok() {
            preload_srs().unwrap();
        }
        release_srs().unwrap();
        invalidate_caches();
        crate::prove::release_srs().unwrap();
        crate::prove::preload_srs().unwrap();
        crate::prove::prestart_prover().unwrap();
        crate::prove::preload_graph().unwrap();
    }
}
