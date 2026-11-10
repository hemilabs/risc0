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
//! Adapted from [intel.rs](super) (docs/metal-port/PLAN.md 4.5, items G1 and G7).
//!
//! It uses the rzup `risc0-groth16` component's proving key, preprocessed coefficients and
//! fuzzed MSM results, as the CUDA and Intel provers do, and produces the same kind of proof.
//!
//! Docker fallback:
//! - `RISC0_GROTH16_METAL_PROVER=docker`, `RISC0_METAL=0` or `RISC0_METAL_SKIP=groth16` select the
//!   Docker prover ([enabled] is false and every entry point in [super] takes its Docker path).
//! - Without a usable Metal device (`risc0_groth16_sys::metal_supported()` fails with
//!   [MetalUnavailable]) [enabled] is false too; if a GPU entry point still reports
//!   [MetalUnavailable], [shrink_wrap] falls back to the Docker prover rather than failing. Any
//!   other error (a missing component, an exceeded `RISC0_METAL_MAX_WORKING_SET_BYTES` budget, a
//!   GPU failure) is returned as is.
//!
//! Locking: the GPU part of a wrap (waiting for the key upload, the prove, reading the proof) and
//! [release_srs] run under `risc0_zkp::hal::metal::exclusive()`, as CUDA runs its prove under the
//! CUDA singleton. The witness is computed before the lock is taken, so STARK proofs on other
//! threads are not blocked by CPU work. The key loader thread does not take the lock (it would
//! deadlock against the waiting holder, see `exclusive()`); `metal_preload` therefore uses no
//! `MetalHal` (its cache has its own mutex).

use std::path::Path;

use anyhow::{anyhow, Context as _, Result};
use risc0_groth16_sys::{MetalUnavailable, ProverParams, SetupParams};
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
/// `RISC0_METAL_SKIP`), `RISC0_GROTH16_METAL_PROVER`, and whether this machine can run the Metal
/// prover.
fn select_gpu(metal_enabled: bool, prover: Option<&str>, supported: bool) -> bool {
    metal_enabled && prover != Some("docker") && supported
}

/// Whether `err` (or any error it wraps) says the Metal prover cannot run here.
fn is_unavailable(err: &anyhow::Error) -> bool {
    err.chain().any(|e| e.is::<MetalUnavailable>())
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
        Err(e) if is_unavailable(&e) => tracing::debug!("groth16 metal preload: {e:#}"),
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
        Err(e) if is_unavailable(&e) => Ok(()),
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
    shrink_wrap_or_docker(
        identity_p254_seal_bytes,
        pre_gpu,
        risc0_groth16_sys::metal_supported,
    )
}

/// [shrink_wrap_with] with the capability check as a parameter (tests force the fallback).
fn shrink_wrap_or_docker(
    identity_p254_seal_bytes: &[u8],
    pre_gpu: impl FnOnce(),
    supported: impl Fn() -> Result<()>,
) -> Result<Seal> {
    let mut pre_gpu = Some(pre_gpu);
    let work_dir = tempdir().context("failed to create a temporary directory")?;
    let r = supported()
        .and_then(|()| gpu_prove(identity_p254_seal_bytes, &mut pre_gpu, work_dir.path()));
    match r {
        Err(e) if is_unavailable(&e) => {
            tracing::info!("{e:#}; using the Docker Groth16 prover");
            if let Some(f) = pre_gpu.take() {
                f();
            }
            super::docker::shrink_wrap(identity_p254_seal_bytes)
        }
        r => r,
    }
}

/// The GPU wrap: witness (helper thread, beside `pre_gpu`) and key load (loader thread), then
/// the prove under the GPU lock. Writes `proof.json` and `public.json` into `work_dir`.
fn gpu_prove(
    identity_p254_seal_bytes: &[u8],
    pre_gpu: &mut Option<impl FnOnce()>,
    work_dir: &Path,
) -> Result<Seal> {
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
    .context("failed to calculate the groth16 witness");
    let witness_ms = ms(t0);

    let _gpu = risc0_zkp::hal::metal::exclusive();
    // Join the loader before returning a witness error (it may hold the prover cache).
    let loaded = loader
        .join()
        .map_err(|_| anyhow!("the groth16 proving key loader panicked"))?;
    let witness = witness?;
    loaded?;
    let loaded_ms = ms(t0);

    const _: () = assert!(std::mem::size_of::<wtns_file::FieldElement<32>>() == 32);
    // Safety: FieldElement<32> is a plain [u8; 32] newtype (size asserted above, alignment 1).
    let witness_bytes =
        unsafe { std::slice::from_raw_parts(witness.as_ptr() as *const u8, witness.len() * 32) };
    let params = ProverParams::new(work_dir, witness.as_ptr() as *const u8)
        .context("failed to create groth16 prover parameters")?;
    risc0_groth16_sys::metal_prove_witness(witness_bytes, &params, &setup)
        .context("groth16 prove failed")?;
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
        // Every Mac with a Metal device selects the GPU prover unless told otherwise.
        let told_otherwise = std::env::var("RISC0_GROTH16_METAL_PROVER").as_deref() == Ok("docker")
            || !risc0_zkp::hal::metal::enabled("groth16");
        if risc0_groth16_sys::metal_supported().is_ok() && !told_otherwise {
            assert!(enabled());
        }
    }

    #[test]
    fn unavailable_is_recognised_through_context() {
        let err = anyhow::Error::from(MetalUnavailable("no Metal device".into()))
            .context("groth16 prove failed");
        assert!(is_unavailable(&err));
        assert!(!is_unavailable(&anyhow!("other")));
    }

    /// Without a usable Metal prover the wrap falls back to the Docker prover cleanly: the same
    /// result as the Docker (CPU-build) path, not a panic, and `pre_gpu` runs exactly once.
    /// Without Docker installed both fail with "Please install docker first.".
    #[test]
    fn falls_back_to_docker() {
        let seal: &[u8] = &[];
        let docker = super::super::docker::shrink_wrap(seal).map(|_| ());

        let calls = Cell::new(0);
        let metal = shrink_wrap_or_docker(
            seal,
            || calls.set(calls.get() + 1),
            || Err(MetalUnavailable("forced by the test".into()).into()),
        )
        .map(|_| ());
        assert_eq!(calls.get(), 1);

        let docker = format!("{:#}", docker.unwrap_err());
        assert_eq!(format!("{:#}", metal.unwrap_err()), docker);
        if !std::process::Command::new("docker")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            assert_eq!(docker, "Please install docker first.");
        }
    }

    /// A bad seal on the GPU path is an error (not a Docker fallback, not a panic), and
    /// `pre_gpu` runs exactly once.
    #[test]
    fn bad_seal_is_an_error() {
        if !enabled() || component_dir().is_err() {
            eprintln!("skipped: no Metal prover or no risc0-groth16 component");
            return;
        }
        let calls = Cell::new(0);
        let err = shrink_wrap_with(&[0u8; 12], || calls.set(calls.get() + 1)).unwrap_err();
        assert_eq!(calls.get(), 1);
        assert!(format!("{err:#}").contains("seal has 12 bytes"), "{err:#}");
    }

    #[test]
    fn cache_entry_points_succeed() {
        // The component may be absent here; preload_srs then fails on it, which is the Intel
        // behaviour. Release and invalidate never fail.
        if component_dir().is_ok() {
            match preload_srs() {
                // Budget runs: the clean budget error, before loading anything.
                Err(e)
                    if risc0_groth16_sys::metal::working_set_budget().is_some()
                        && format!("{e:#}").contains("working-set budget exceeded") => {}
                r => r.unwrap(),
            }
        }
        release_srs().unwrap();
        invalidate_caches();
        crate::prove::release_srs().unwrap();
        crate::prove::prestart_prover().unwrap();
        crate::prove::preload_graph().unwrap();
        crate::prove::release_srs().unwrap();
    }
}

#[cfg(test)]
#[path = "metal_tests.rs"]
mod metal_tests;
