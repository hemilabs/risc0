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
//! threads are not blocked by CPU work. The key load does not take the lock either:
//! `metal_preload` uses no `MetalHal` (its cache has its own mutex).
//!
//! Memory (G8): the key is loaded only after `pre_gpu` (the zkvm's `release_hal_buffers`) has
//! returned, never beside the STARK prover's pools; [prestart] does not load it.

use std::{
    path::Path,
    sync::atomic::{AtomicU8, Ordering},
};

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
    check_key(&setup)?;
    risc0_groth16_sys::metal_preload(&setup).context("failed to load the groth16 proving key")
}

/// Item P31-12: the proving key in the resolved component ([component_dir]: rzup's
/// `risc0-groth16` 0.2.0, or `RISC0_GROTH16_DIR`) must be the one [crate::verifying_key] verifies
/// with (the v0.2.0 ceremony), so that a directory holding another ceremony's key (such as
/// v0.1.0's, which the 3.1 verifier rejects) is a clear error before anything is loaded, not a
/// wrap that fails verification. Reads the zkey header (a few KB) once per key file.
fn check_key(setup: &SetupParams) -> Result<()> {
    static CHECKED: std::sync::Mutex<Option<std::path::PathBuf>> = std::sync::Mutex::new(None);
    let mut checked = CHECKED.lock().unwrap_or_else(|e| e.into_inner());
    if checked.as_deref() == Some(setup.srs_path.as_path()) {
        return Ok(());
    }
    let header = risc0_groth16_sys::metal::srs::read_zkey_header(setup.srs_path.as_path())
        .context("failed to read the groth16 proving key header")?;
    let vk = header.vk.to_ark()?;
    let want = crate::verifying_key().0;
    if vk.alpha_g1 != want.alpha_g1
        || vk.beta_g2 != want.beta_g2
        || vk.gamma_g2 != want.gamma_g2
        || vk.delta_g2 != want.delta_g2
        || vk.ic != want.gamma_abc_g1
    {
        return Err(anyhow!(
            "the groth16 proving key {} does not match this verifier's key: install the \
             risc0-groth16 0.2.0 component (rzup install risc0-groth16 0.2.0), or point \
             RISC0_GROTH16_DIR at it",
            setup.srs_path.as_path().display()
        ));
    }
    *checked = Some(setup.srs_path.as_path().to_path_buf());
    Ok(())
}

/// Load the witness graph (in-process witness generation) and compile it for this process's
/// evaluator ([super::metal_witness]).
pub(crate) fn preload_graph() -> Result<()> {
    super::metal_witness::preload()
}

/// Prepare a wrap in the background: build the witness graph (CPU), and (item G13) start loading
/// the proving key if it fits beside what is still to come ([early_key_load]).
///
/// G8: the zkvm calls this before identity_p254, while the STARK prover's HAL pools may still be
/// allocated, and on unified memory a key loaded then sits beside them (5.7 GB on top of 2-11 GB
/// of pools: 16.6-17.6 GB peaks at po2 20 without a budget, and a refused load under an 8 GiB
/// one; zk-miner M1). So the early load waits until those pools are gone (the zkvm releases them
/// right after this call, Z10) and loads only if the key and identity_p254's own working set then
/// fit the working-set limit; otherwise [shrink_wrap_with] loads the key after `pre_gpu`, as
/// before. Checks that the component is installed (the error the Intel prestart returns).
pub(crate) fn prestart() -> Result<()> {
    let dir = component_dir()?;
    std::thread::spawn(|| {
        if let Err(e) = preload_graph() {
            tracing::warn!("groth16 witness graph preload failed: {e:#}");
        }
    });
    if early_key_enabled()
        && EARLY_KEY
            .compare_exchange(
                EARLY_IDLE,
                EARLY_WAITING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    {
        std::thread::spawn(move || {
            if let Err(e) = early_key_load(&dir) {
                tracing::warn!("groth16 early key load failed (the wrap loads it): {e:#}");
            }
            EARLY_KEY.store(EARLY_IDLE, Ordering::Release);
        });
    }
    Ok(())
}

/// Item G13 (c): whether [prestart] may load the proving key early (`RISC0_G16_METAL_EARLY_KEY=0`
/// turns it off: the key is then always loaded after `pre_gpu`, the G8 order).
fn early_key_enabled() -> bool {
    std::env::var("RISC0_G16_METAL_EARLY_KEY").as_deref() != Ok("0")
}

/// What identity_p254 (its own recursion HAL, run after [prestart]) may still allocate beside an
/// early-loaded key: its measured peak is ~1.4-1.7 GiB (BASELINES 5b: 1731 MiB for a whole
/// fibonacci recursion phase), plus margin.
const P254_RESERVE_BYTES: u64 = 2 << 30;

/// The longest [early_key_load] waits for the STARK pools to be released and the key to fit; the
/// wrap also stops the wait when it starts ([stop_early_key_wait]).
const EARLY_KEY_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// [early_key_load]'s state: no loader, a loader waiting for room, a loader loading (it holds
/// the prover cache's lock), or a waiting loader told to give up by the wrap.
static EARLY_KEY: AtomicU8 = AtomicU8::new(EARLY_IDLE);
const EARLY_IDLE: u8 = 0;
const EARLY_WAITING: u8 = 1;
const EARLY_LOADING: u8 = 2;
const EARLY_STOP: u8 = 3;

/// Called when the wrap starts: a loader still waiting for room gives up (the wrap loads the key
/// after `pre_gpu` instead); one already loading finishes (the wrap waits for it on the cache).
fn stop_early_key_wait() {
    let _ = EARLY_KEY.compare_exchange(
        EARLY_WAITING,
        EARLY_STOP,
        Ordering::AcqRel,
        Ordering::Acquire,
    );
}

/// Whether an early key load fits: nothing beyond identity_p254's share is allocated (no STARK
/// pools; G8), and the key plus identity_p254's reserve fit the working-set limit on top of what
/// is.
fn early_key_fits(allocated: u64, planned: u64, limit: u64) -> bool {
    allocated <= P254_RESERVE_BYTES
        && allocated
            .saturating_add(planned)
            .saturating_add(P254_RESERVE_BYTES)
            <= limit
}

/// Load the proving key during identity_p254 if it fits ([early_key_fits]), polling the device's
/// allocation while the caller releases the STARK pools (freeing 5-11 GiB of pools can take
/// longer than identity_p254's first allocations), until it fits, the wrap starts
/// ([stop_early_key_wait]) or [EARLY_KEY_WAIT] passes. A no-op if the key is cached.
fn early_key_load(dir: &Path) -> Result<()> {
    let t0 = std::time::Instant::now();
    let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
    let setup = SetupParams::new(dir).context("failed to create groth16 setup params")?;
    check_key(&setup)?;
    let planned = risc0_groth16_sys::metal_planned_bytes(&setup)?;
    let limit = risc0_groth16_sys::effective_working_set_limit()
        .ok_or_else(|| anyhow!("no Metal device"))?;
    let mut allocated;
    let go = loop {
        allocated = risc0_groth16_sys::metal::allocated_bytes();
        if early_key_fits(allocated, planned, limit) {
            // Claim the load unless the wrap has started meanwhile.
            break EARLY_KEY
                .compare_exchange(
                    EARLY_WAITING,
                    EARLY_LOADING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok();
        }
        if EARLY_KEY.load(Ordering::Acquire) != EARLY_WAITING || t0.elapsed() >= EARLY_KEY_WAIT {
            break false;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    };
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    if !go {
        let msg = format!(
            "groth16-metal: no early key load after {:.1} ms ({:.0} MiB allocated, key {:.0} MiB, \
             limit {:.0} MiB); the wrap loads it after pre_gpu",
            t0.elapsed().as_secs_f64() * 1000.0,
            mib(allocated),
            mib(planned),
            mib(limit)
        );
        tracing::debug!("{msg}");
        if verbose {
            eprintln!("[{msg}]");
        }
        return Ok(());
    }
    let waited = t0.elapsed();
    let r = risc0_groth16_sys::metal_preload(&setup);
    if verbose {
        eprintln!(
            "[groth16-metal] early key load: waited {:.1} ms ({:.0} MiB allocated, key {:.0} MiB, \
             limit {:.0} MiB), loaded in {:.1} ms{}",
            waited.as_secs_f64() * 1000.0,
            mib(allocated),
            mib(planned),
            mib(limit),
            (t0.elapsed() - waited).as_secs_f64() * 1000.0,
            if r.is_err() { " (failed)" } else { "" }
        );
    }
    match r {
        Err(e) if is_unavailable(&e) => Ok(()),
        r => r,
    }
}

/// Start loading the proving key early. CUDA-only gating (its loader waits for
/// [prestart_key_go]); on Metal, [prestart] loads it early when it fits, so a no-op.
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
    // An early load still waiting for room must not load the key after this release.
    stop_early_key_wait();
    let _gpu = risc0_zkp::hal::metal::exclusive();
    risc0_groth16_sys::metal_release().context("failed to release the groth16 prover")
}

/// Discard the prover's caches: the in-memory prover and the compact-coefficients disk cache
/// (`coeffs_compact_metal_*.bin` in the Groth16 cache directory, item G10); the next
/// [shrink_wrap] rebuilds both from the proving key.
pub(crate) fn invalidate_caches() {
    if let Err(e) = release_srs() {
        tracing::warn!("{e:#}");
    }
    let removed = risc0_groth16_sys::metal::compact::invalidate_cache();
    tracing::debug!("groth16-metal: removed {removed} compact coefficient cache files");
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
    // `fast::witness` and `to_json` read the seal as `&[u32]` (bytemuck::cast_slice, which panics
    // on unaligned data): give them an aligned copy when the caller's slice is not.
    let copy = aligned_copy(identity_p254_seal_bytes);
    let identity_p254_seal_bytes: &[u8] = match &copy {
        Some(words) => bytemuck::cast_slice(words),
        None => identity_p254_seal_bytes,
    };
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

/// An aligned copy of `seal` as u32 words if it is not 4-byte aligned, else `None` (also for a
/// ragged length, passed through for `fast::witness` to report). Native-endian, as
/// `bytemuck::cast_slice` reads it (bytemuck's `pod_collect_to_vec` needs its
/// `extern_crate_alloc` feature, which this crate does not enable).
fn aligned_copy(seal: &[u8]) -> Option<Vec<u32>> {
    if (seal.as_ptr() as usize).is_multiple_of(4) || !seal.len().is_multiple_of(4) {
        return None;
    }
    Some(
        seal.chunks_exact(4)
            .map(|c| u32::from_ne_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

/// Item G13 (b): whether the `par` evaluator writes the witness straight into the prover's shared
/// witness buffer (`RISC0_G16_METAL_DIRECT=0` computes it into a host vector beside the key load
/// and copies it, as the other evaluators do).
fn direct_enabled() -> bool {
    std::env::var("RISC0_G16_METAL_DIRECT").as_deref() != Ok("0")
}

/// A witness computed into host memory.
enum HostWitness {
    /// [super::metal_witness] (`par` without the direct write, or `fast`).
    Limbs(Vec<[u64; 4]>),
    /// [super::fast] (`ref`).
    Elements(Vec<wtns_file::FieldElement<32>>),
}

impl HostWitness {
    fn compute(seal: &[u8]) -> Result<Self> {
        if super::metal_witness::witness_len().is_some() {
            super::metal_witness::witness(seal).map(Self::Limbs)
        } else {
            super::fast::witness(seal).map(Self::Elements)
        }
    }

    fn as_bytes(&self) -> &[u8] {
        const _: () = assert!(std::mem::size_of::<wtns_file::FieldElement<32>>() == 32);
        match self {
            // Little-endian limbs of little-endian u64s: the canonical little-endian bytes.
            Self::Limbs(w) => bytemuck::cast_slice(w),
            // Safety: FieldElement<32> is a plain [u8; 32] newtype (size asserted above,
            // alignment 1).
            Self::Elements(w) => unsafe {
                std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 32)
            },
        }
    }
}

/// The GPU wrap. Writes `proof.json` and `public.json` into `work_dir`.
///
/// - `par` evaluator with the key cached or loaded early by [prestart] (item G13): `pre_gpu` (the
///   zkvm releases the STARK HAL pools there), then under the GPU lock the witness straight into
///   the prover's shared buffer (~25-35 ms) and the prove.
/// - Otherwise the witness is computed on a helper thread while the calling thread runs `pre_gpu`
///   and **then** loads the proving key if it is not cached, so the key never coexists with the
///   pools it replaces (G8); then the prove, under the GPU lock.
fn gpu_prove(
    identity_p254_seal_bytes: &[u8],
    pre_gpu: &mut Option<impl FnOnce()>,
    work_dir: &Path,
) -> Result<Seal> {
    // Aligned for `fast::witness` (see shrink_wrap_or_docker; a no-op when called from there).
    let copy = aligned_copy(identity_p254_seal_bytes);
    let identity_p254_seal_bytes: &[u8] = match &copy {
        Some(words) => bytemuck::cast_slice(words),
        None => identity_p254_seal_bytes,
    };
    let dir = component_dir()?;
    let setup = SetupParams::new(&dir).context("failed to create groth16 setup params")?;

    let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
    let t0 = std::time::Instant::now();
    let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1000.0;

    // A loader still waiting for room gives up now: the key is loaded after pre_gpu below.
    stop_early_key_wait();
    // The direct write needs the prover first, so it is taken only when the key is already
    // cached or being loaded early (metal_is_loaded waits for that load): otherwise the witness
    // runs beside the key load below rather than after it.
    if direct_enabled()
        && super::metal_witness::is_par()
        && risc0_groth16_sys::metal::metal_is_loaded()
    {
        if let Some(f) = pre_gpu.take() {
            f();
        }
        let pre_gpu_ms = ms(t0);
        // Check the seal before the key load (a bad seal is an error, not a load).
        super::fast::seal_inputs(identity_p254_seal_bytes)?;
        preload(&dir)?;
        let loaded_ms = ms(t0);
        let _gpu = risc0_zkp::hal::metal::exclusive();
        let locked_ms = ms(t0);
        let mut witness_ms = 0.0;
        let params = ProverParams::new(work_dir, std::ptr::null())
            .context("failed to create groth16 prover parameters")?;
        risc0_groth16_sys::metal_prove_in_place(&params, &setup, |buf| {
            let t = std::time::Instant::now();
            let r = super::metal_witness::witness_into(identity_p254_seal_bytes, buf)
                .context("failed to calculate the groth16 witness");
            witness_ms = ms(t);
            r
        })
        .context("groth16 prove failed")?;
        if verbose {
            eprintln!(
                "[groth16-metal] pre_gpu done={pre_gpu_ms:.1}ms key loaded={loaded_ms:.1}ms gpu \
                 lock={locked_ms:.1}ms witness (par, in place)={witness_ms:.1}ms proved={:.1}ms \
                 (since the wrap started)",
                ms(t0)
            );
        }
        return read_proof(&params);
    }

    let mut pre_gpu_ms = 0.0;
    let mut loaded_ms = 0.0;
    // Item G15 (opt-in `par_transient`): the witness is computed before the key load rather than
    // beside it, so that its value array (580 MB while in use) never coexists with the key's
    // allocations. (When the key was loaded early, the in-place path above is taken instead, and
    // the array is resident at the wrap's peak: the same peak as `par`; see notes/G15.md.)
    let (witness, witness_ms, loaded) = if super::metal_witness::is_par_transient() {
        if let Some(f) = pre_gpu.take() {
            f();
        }
        pre_gpu_ms = ms(t0);
        let w = HostWitness::compute(identity_p254_seal_bytes);
        let w_ms = ms(t0);
        let loaded = preload(&dir);
        loaded_ms = ms(t0);
        (w, w_ms, loaded)
    } else {
        std::thread::scope(|sc| {
            let h = sc.spawn(|| {
                let w = HostWitness::compute(identity_p254_seal_bytes);
                (w, ms(t0))
            });
            if let Some(f) = pre_gpu.take() {
                f();
            }
            pre_gpu_ms = ms(t0);
            // The proving key (first use, or after release_srs), now that pre_gpu has freed the
            // STARK pools; overlaps the witness.
            let loaded = preload(&dir);
            loaded_ms = ms(t0);
            let (w, w_ms) = h
                .join()
                .unwrap_or_else(|_| (Err(anyhow!("groth16 witness thread panicked")), ms(t0)));
            (w, w_ms, loaded)
        })
    };
    let witness = witness.context("failed to calculate the groth16 witness")?;
    loaded?;

    let _gpu = risc0_zkp::hal::metal::exclusive();
    let locked_ms = ms(t0);

    let params = ProverParams::new(work_dir, witness.as_bytes().as_ptr())
        .context("failed to create groth16 prover parameters")?;
    risc0_groth16_sys::metal_prove_witness(witness.as_bytes(), &params, &setup)
        .context("groth16 prove failed")?;
    if verbose {
        eprintln!(
            "[groth16-metal] pre_gpu done={pre_gpu_ms:.1}ms key loaded={loaded_ms:.1}ms \
             witness={witness_ms:.1}ms gpu lock={locked_ms:.1}ms proved={:.1}ms (since the wrap \
             started)",
            ms(t0)
        );
    }
    read_proof(&params)
}

/// The proof the prover wrote to `params.proof_path`.
fn read_proof(params: &ProverParams) -> Result<Seal> {
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
        let _lock = super::metal_tests::gpu_test_lock();
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

    /// Item G13 (c): the early key load's gate.
    #[test]
    fn early_key_gate() {
        const GIB: u64 = 1 << 30;
        let key = 3928 << 20; // the rzup key with the default memory switches
                              // 36 GB Mac (28 GiB limit), after the zkvm released the STARK pools: load.
        assert!(early_key_fits(1476 << 20, key, 28 * GIB));
        // 8 GiB budget: the key, identity_p254's reserve and what is allocated fit.
        assert!(early_key_fits(1476 << 20, key, 8 * GIB));
        // 4 GiB budget: never (the key alone is ~3.84 GiB).
        assert!(!early_key_fits(0, key, 4 * GIB));
        // 5 GiB: the key fits but not with identity_p254's reserve.
        assert!(!early_key_fits(0, key, 5 * GIB));
        // STARK pools still allocated (G8): not beside them, whatever the limit.
        assert!(!early_key_fits(4602757120, key, 28 * GIB));
        assert!(!early_key_fits(u64::MAX, key, u64::MAX));
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
        let _lock = super::metal_tests::gpu_test_lock();
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
        let _lock = super::metal_tests::gpu_test_lock();
        if !enabled() || component_dir().is_err() {
            eprintln!("skipped: no Metal prover or no risc0-groth16 component");
            return;
        }
        let calls = Cell::new(0);
        let err = shrink_wrap_with(&[0u8; 12], || calls.set(calls.get() + 1)).unwrap_err();
        assert_eq!(calls.get(), 1);
        assert!(format!("{err:#}").contains("seal has 12 bytes"), "{err:#}");
    }

    /// Item P31-12: the resolved component's key is the 3.1 verifier's (v0.2.0); another
    /// ceremony's key (v0.1.0, 3.0's, when it is installed beside it) is a clear error.
    #[test]
    fn key_of_another_ceremony_is_a_clear_error() {
        let Ok(dir) = component_dir() else {
            eprintln!("skipped: no risc0-groth16 component");
            return;
        };
        check_key(&SetupParams::new(&dir).unwrap()).unwrap();
        let old = dir.with_file_name("v0.1.0-risc0-groth16");
        if !old.join("stark_verify_final.zkey").is_file() {
            eprintln!("skipped the old-key half: nothing at {}", old.display());
            return;
        }
        let err = check_key(&SetupParams::new(&old).unwrap()).unwrap_err();
        assert!(
            format!("{err:#}").contains("does not match this verifier's key"),
            "{err:#}"
        );
        // The check is per key file: the right one still passes.
        check_key(&SetupParams::new(&dir).unwrap()).unwrap();
    }

    #[test]
    fn cache_entry_points_succeed() {
        let _lock = super::metal_tests::gpu_test_lock();
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
