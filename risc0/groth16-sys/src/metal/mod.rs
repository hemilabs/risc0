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
//! directory, in the CUDA/Intel JSON format), same caching contract (the proving key, the
//! pipelines and the GPU scratch stay loaded until [metal_release]). The orchestration is
//! [prover::MetalProver]; the kernels and their host wrappers are [small], [ntt] and [msm], the
//! key loaders [srs] and [coeffs].
//!
//! [metal_supported] fails with [MetalUnavailable] when this machine has no Metal device (or the
//! build embeds no kernels, `RISC0_SKIP_BUILD_KERNELS`); `risc0-groth16` then falls back to the
//! Docker prover (`err.is::<MetalUnavailable>()`).
//!
//! Locking: callers in `risc0-groth16` hold `risc0_zkp::hal::metal::exclusive()` around
//! [metal_prove] and [metal_release]. [metal_preload] may run on a background thread without that
//! lock (as `prestart` does, like the CUDA prestart): the prover cache has its own mutex, and the
//! preload only allocates, reads the key files, builds pipelines and runs two small command
//! buffers (fuzz and NTT tables) on its own queue; it never uses a `MetalHal`.
//!
//! Memory: the cached prover holds 4142 MiB of device memory for the rzup v0.2.0 component (item
//! G10: points mapped from the zkey without a copy, compact coefficients from a disk cache,
//! a/b/c, fuzz and MSM pieces aliased; 6004 MiB with `RISC0_G16_METAL_G10=0`); where that does
//! not fit, the smaller layouts of items P31-12 (split MSM H, 4012 MiB) and B9-4G (the NTT
//! twiddles lent to the MSM scratch, 3882 MiB: the 4 GiB tier), see [prover::MemOptions]. `RISC0_METAL_MAX_WORKING_SET_BYTES` (the zkp HAL's budget, PLAN 9.1) is
//! honoured: [check_working_set] fails before any allocation that would take the device's
//! `currentAllocatedSize` above it, and the prover checks its whole footprint before loading.

use std::{
    fmt,
    sync::{Mutex, MutexGuard},
};

use anyhow::{anyhow, Result};
use metal::{
    foreign_types::{ForeignType as _, ForeignTypeRef as _},
    objc::{msg_send, runtime::Object, sel, sel_impl},
    Buffer, DeviceRef, MTLBuffer, MTLResourceOptions,
};

use crate::{ProverParams, SetupParams};

pub mod coeffs;
pub mod compact;
pub mod filemap;
pub mod msm;
pub mod ntt;
pub mod pipelines;
pub mod prover;
pub mod small;
pub mod srs;

/// The error [metal_supported] (and the other entry points) return when this machine cannot run
/// the Metal prover (no Metal device, or a build without kernels). Callers fall back to another
/// prover when they see it (`anyhow::Error::is::<MetalUnavailable>()`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalUnavailable(pub String);

impl fmt::Display for MetalUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the Metal Groth16 prover is unavailable: {}", self.0)
    }
}

impl std::error::Error for MetalUnavailable {}

/// The working-set budget variable of the zkp Metal HAL (`risc0_zkp::hal::metal::
/// MAX_WORKING_SET_ENV`); this crate does not depend on risc0-zkp, so it reads it itself.
pub const MAX_WORKING_SET_ENV: &str = "RISC0_METAL_MAX_WORKING_SET_BYTES";

/// Parse a byte count with an optional binary suffix (`K`, `M`, `G`, `T`, optionally followed by
/// `iB` or `B`); `None` if malformed or zero. Same grammar as the zkp HAL's parser.
pub fn parse_bytes(text: &str) -> Option<u64> {
    let t = text.trim();
    let t = t
        .strip_suffix("iB")
        .or_else(|| t.strip_suffix("ib"))
        .or_else(|| t.strip_suffix('B'))
        .or_else(|| t.strip_suffix('b'))
        .unwrap_or(t);
    let (digits, shift) = match t.chars().last()? {
        'k' | 'K' => (&t[..t.len() - 1], 10),
        'm' | 'M' => (&t[..t.len() - 1], 20),
        'g' | 'G' => (&t[..t.len() - 1], 30),
        't' | 'T' => (&t[..t.len() - 1], 40),
        _ => (t, 0),
    };
    let n: u64 = digits.trim().parse().ok()?;
    n.checked_mul(1u64 << shift).filter(|&b| b > 0)
}

/// The explicit working-set budget ([MAX_WORKING_SET_ENV]), if set and valid. Read at every call
/// (tests change it between runs of one process only through separate processes).
pub fn working_set_budget() -> Option<u64> {
    let text = std::env::var(MAX_WORKING_SET_ENV).ok()?;
    parse_bytes(&text)
}

/// The working-set limit with an explicit budget: `min(budget, recommendedMaxWorkingSetSize)`;
/// `None` without a budget (then nothing is enforced, as in the zkp HAL).
pub fn working_set_limit(device: &DeviceRef) -> Option<u64> {
    working_set_budget().map(|b| b.min(device.recommended_max_working_set_size()))
}

/// With an explicit budget: fail if allocating `bytes` more would take the device's
/// `currentAllocatedSize` above the working-set limit. The message matches the zkp HAL's.
pub fn check_working_set(device: &DeviceRef, what: &str, bytes: u64) -> Result<()> {
    let Some(limit) = working_set_limit(device) else {
        return Ok(());
    };
    let current = device.current_allocated_size();
    if current.saturating_add(bytes) <= limit {
        return Ok(());
    }
    Err(anyhow!(
        "Metal working-set budget exceeded: allocating {bytes} bytes for {what} with {current} \
         bytes already allocated would exceed the limit of {limit} bytes ({MAX_WORKING_SET_ENV}=\
         {}, recommendedMaxWorkingSetSize={})",
        working_set_budget().unwrap_or(0),
        device.recommended_max_working_set_size()
    ))
}

/// The device's `currentAllocatedSize` (process-wide), 0 without a Metal device. Diagnostics:
/// callers that report peak GPU memory per phase (PLAN 9.1) sample it.
pub fn allocated_bytes() -> u64 {
    match prover::kernels() {
        Ok(k) => k.device.current_allocated_size(),
        Err(_) => metal::Device::system_default()
            .map(|d| d.current_allocated_size())
            .unwrap_or(0),
    }
}

/// A new buffer of `bytes` (at least 16) bytes, `StorageModePrivate` if `private` else
/// `StorageModeShared`, after the `max_buffer_length` and working-set budget checks.
///
/// Every buffer of this module comes from here or from `srs`'s loader: metal-rs 0.29's
/// `new_buffer` wraps the result of `newBufferWithLength:options:` in a non-null pointer type, so a
/// nil result (allocation refused by the OS) is undefined behaviour there and crashed later in
/// `contents()`; here it is an error (`"Metal allocation failed ..."`, the zkp HAL's wording, which
/// zk-miner's worker classifies as out-of-memory). Same approach as zkp's
/// `Runtime::new_buffer_raw` (PLAN 2.9), without a pool to drain.
pub fn new_buffer_checked(
    device: &DeviceRef,
    bytes: u64,
    private: bool,
    what: &str,
) -> Result<Buffer> {
    let bytes = bytes.max(16);
    anyhow::ensure!(
        bytes <= device.max_buffer_length(),
        "groth16-metal: {what}: {bytes} bytes exceed this GPU's max_buffer_length {}",
        device.max_buffer_length()
    );
    check_working_set(device, what, bytes)?;
    let options = if private {
        MTLResourceOptions::StorageModePrivate
    } else {
        MTLResourceOptions::StorageModeShared
    };
    new_buffer_raw(device, bytes, options).ok_or_else(|| {
        anyhow!(
            "Metal allocation failed on {what}: {bytes} bytes (currentAllocatedSize {})",
            device.current_allocated_size()
        )
    })
}

#[cfg(test)]
thread_local! {
    /// Test hook: the next `n` [new_buffer_raw] calls on this thread return nil.
    static FAIL_ALLOCS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Test hook: the next `n` allocations of [new_buffer_checked] on this thread fail as if
/// `newBufferWithLength:` returned nil.
#[cfg(test)]
pub(crate) fn inject_alloc_failures(n: usize) {
    FAIL_ALLOCS.with(|c| c.set(n));
}

/// `newBufferWithLength:options:` with a nil check on the raw pointer.
// objc's `msg_send!` expands a `cfg(feature = "cargo-clippy")` this crate does not declare.
#[allow(unexpected_cfgs)]
fn new_buffer_raw(device: &DeviceRef, bytes: u64, options: MTLResourceOptions) -> Option<Buffer> {
    #[cfg(test)]
    if FAIL_ALLOCS.with(|c| {
        let n = c.get();
        c.set(n.saturating_sub(1));
        n > 0
    }) {
        return None;
    }
    // SAFETY: `newBufferWithLength:options:` returns a +1 (owned) reference or nil; only a
    // non-nil result is wrapped, and `from_ptr` takes over that reference.
    let ptr: *mut MTLBuffer = unsafe {
        msg_send![device.as_ptr() as *mut Object, newBufferWithLength: bytes options: options]
    };
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { Buffer::from_ptr(ptr) })
    }
}

/// Whether this build can prove on the GPU: `Ok` if a Metal device is present and the build
/// embeds the groth16 kernels, else [MetalUnavailable]. Cheap: needs no proving key, so callers
/// check it before computing a witness they could not use.
pub fn metal_supported() -> Result<()> {
    if crate::METAL_LIB.is_empty() {
        return Err(MetalUnavailable(
            "this build embeds no groth16 Metal kernels (RISC0_SKIP_BUILD_KERNELS?)".into(),
        )
        .into());
    }
    if metal::Device::system_default().is_none() {
        return Err(MetalUnavailable("no Metal device".into()).into());
    }
    Ok(())
}

/// The cached prover (key, pipelines, scratch); `None` until the first preload or prove, and
/// after [metal_release]. Its own mutex: [metal_preload] runs without the GPU lock.
///
/// The mutex is held for a whole build (key load, ~1 s) or proof (~3 s): a second caller (a
/// prestart preload, [metal_release], [metal_is_loaded]) waits that long; callers in
/// `risc0-groth16` serialize proofs with the GPU lock anyway, so nothing is lost by it.
static PROVER: Mutex<Option<prover::MetalProver>> = Mutex::new(None);

fn cache() -> MutexGuard<'static, Option<prover::MetalProver>> {
    lock_discarding_poisoned(&PROVER)
}

/// Lock a cache slot; if a holder panicked (the mutex is poisoned), discard what it cached and
/// clear the poison. A panic inside a build or proof can leave the cached prover half-built or
/// with its scratch in an unknown state, so it is rebuilt by the next user instead of reused (a
/// rebuild costs one key load, ~1 s).
fn lock_discarding_poisoned<T>(m: &Mutex<Option<T>>) -> MutexGuard<'_, Option<T>> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            let mut g = poisoned.into_inner();
            if g.take().is_some() {
                eprintln!(
                    "groth16-metal: a previous build or proof panicked; discarding the cached \
                     prover"
                );
            }
            m.clear_poison();
            g
        }
    }
}

/// The cached prover for `setup`, built (key loaded, pipelines and scratch created) if absent or
/// built for other files.
fn cached_prover<'a>(
    slot: &'a mut Option<prover::MetalProver>,
    setup_params: &SetupParams,
) -> Result<&'a mut prover::MetalProver> {
    metal_supported()?;
    if slot.as_ref().is_some_and(|p| !p.is_for(setup_params)) {
        // A different key: free the old one first (the budget check counts what is allocated).
        *slot = None;
    }
    if slot.is_none() {
        *slot = Some(prover::MetalProver::new(setup_params)?);
    }
    Ok(slot.as_mut().unwrap())
}

/// Load the proving key and coefficients onto the GPU and build the prover (pipelines, fuzz,
/// NTT tables, scratch), where they stay cached until [metal_release]. [metal_prove] does this
/// itself if needed. Takes no GPU lock (see the module docs).
pub fn metal_preload(setup_params: &SetupParams) -> Result<()> {
    let mut slot = cache();
    cached_prover(&mut slot, setup_params).map(|_| ())
}

/// Prove with the Metal GPU prover; writes the proof to `prover_params.proof_path` and the public
/// inputs to `prover_params.public_path` in the CUDA prover's JSON format.
///
/// `prover_params.witness` must point to the circuit's full witness: `num_vars` (the proving
/// key's witness length) canonical little-endian 32-byte field elements. Prefer
/// [metal_prove_witness], which checks the length.
pub fn metal_prove(prover_params: &ProverParams, setup_params: &SetupParams) -> Result<()> {
    anyhow::ensure!(
        !prover_params.witness.is_null(),
        "groth16-metal: null witness pointer"
    );
    let mut slot = cache();
    let prover = cached_prover(&mut slot, setup_params)?;
    // Safety: the caller's contract above (the Intel/CUDA entry points' contract).
    let witness =
        unsafe { std::slice::from_raw_parts(prover_params.witness, prover.witness_len() * 32) };
    prove_into(prover, witness, prover_params)
}

/// [metal_prove] with the witness as a byte slice (`32 * num_vars` bytes, checked).
/// `prover_params.witness` is ignored.
pub fn metal_prove_witness(
    witness: &[u8],
    prover_params: &ProverParams,
    setup_params: &SetupParams,
) -> Result<()> {
    let mut slot = cache();
    let prover = cached_prover(&mut slot, setup_params)?;
    prove_into(prover, witness, prover_params)
}

/// [metal_prove] with the witness written straight into the prover's shared witness buffer by
/// `fill` (item G13: no 180 MB host vector, no copy). `fill` gets [prover::MetalProver::
/// witness_buffer] (`num_vars` canonical elements as little-endian u64 limbs) and must write every
/// element. Builds the prover (loads the key) first if it is not cached.
pub fn metal_prove_in_place(
    prover_params: &ProverParams,
    setup_params: &SetupParams,
    fill: impl FnOnce(&mut [[u64; 4]]) -> Result<()>,
) -> Result<()> {
    let mut slot = cache();
    let prover = cached_prover(&mut slot, setup_params)?;
    fill(prover.witness_buffer())?;
    let proof = prover.prove_in_place(&prover::ProveOptions::from_env())?;
    proof.write_files(
        prover_params.proof_path.as_path(),
        prover_params.public_path.as_path(),
    )
}

/// The working-set limit that applies to this process's GPU allocations: the explicit budget's
/// ([working_set_limit]), else the device's `recommendedMaxWorkingSetSize`; `None` without a
/// Metal device.
pub fn effective_working_set_limit() -> Option<u64> {
    let device = metal::Device::system_default()?;
    Some(working_set_limit(&device).unwrap_or_else(|| device.recommended_max_working_set_size()))
}

/// The device footprint of the prover for `setup_params` with the environment's memory switches
/// ([prover::planned_bytes]); reads only the files' headers.
pub fn metal_planned_bytes(setup_params: &SetupParams) -> Result<u64> {
    prover::planned_bytes(setup_params, prover::MemOptions::from_env())
}

fn prove_into(
    prover: &mut prover::MetalProver,
    witness: &[u8],
    prover_params: &ProverParams,
) -> Result<()> {
    let proof = prover.prove(witness, &prover::ProveOptions::from_env())?;
    proof.write_files(
        prover_params.proof_path.as_path(),
        prover_params.public_path.as_path(),
    )
}

/// Free the GPU prover's cached proving key, coefficients, pipelines and scratch (several GB).
/// Succeeds when nothing is cached.
pub fn metal_release() -> Result<()> {
    let released = cache().take();
    drop(released);
    Ok(())
}

/// Build the groth16 pipelines now (process-wide, kept across [metal_release]) without loading a
/// key; returns how long the build took. Cold pipeline creation of this library took minutes on
/// the dev machine the first time (docs/metal-port/notes/G7.md), so a long-lived worker should
/// call this at start, outside any job deadline. Takes no GPU lock. The pipelines come through
/// the binary archive ([pipelines], item G11): ~0.02 s when it holds them, whatever the state of
/// the OS shader cache; ~25-30 s cold on the M5 Max the first time a metallib is seen.
pub fn metal_warmup() -> Result<std::time::Duration> {
    Ok(prover::kernels()?.build_time)
}

/// Whether a prover is cached (tests and diagnostics).
pub fn metal_is_loaded() -> bool {
    cache().is_some()
}

/// Test isolation (PLAN 8.7): the tests that build the prover or load the real key (GBs of
/// device memory) run alone ([test_lock::exclusive]); the other GPU tests of real size share
/// ([test_lock::shared]). Under a `RISC0_METAL_MAX_WORKING_SET_BYTES` budget a kernel test would
/// otherwise be refused because a concurrent prover build holds most of the budget. Each test
/// takes it once (no reentrancy). Poison-tolerant.
#[cfg(test)]
pub(crate) mod test_lock {
    use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

    static LOCK: RwLock<()> = RwLock::new(());

    pub(crate) fn shared() -> RwLockReadGuard<'static, ()> {
        LOCK.read().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn exclusive() -> RwLockWriteGuard<'static, ()> {
        LOCK.write().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod bn254_tests;

#[cfg(test)]
mod tests_tierz;

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn supported_on_this_machine() {
        // Every Mac this crate builds for has a Metal device; the build embeds the kernels.
        if ::metal::Device::system_default().is_some() && !crate::METAL_LIB.is_empty() {
            metal_supported().unwrap();
        } else {
            assert!(metal_supported().unwrap_err().is::<MetalUnavailable>());
        }
        metal_release().unwrap();
    }

    #[test]
    fn missing_key_is_a_clean_error() {
        let dir = Path::new("/nonexistent-risc0-groth16");
        let setup = SetupParams::new(dir).unwrap();
        let err = metal_preload(&setup).unwrap_err();
        assert!(!err.is::<MetalUnavailable>(), "{err:#}");
        assert!(
            format!("{err:#}").contains("nonexistent-risc0-groth16"),
            "{err:#}"
        );
    }

    /// A panic while the cache is locked discards the cached value and clears the poison (G8:
    /// before, the next user reused whatever the panicking build or proof had left).
    #[test]
    fn poisoned_cache_is_discarded() {
        static SLOT: Mutex<Option<u32>> = Mutex::new(None);
        *lock_discarding_poisoned(&SLOT) = Some(7);
        let r = std::thread::spawn(|| {
            let mut g = lock_discarding_poisoned(&SLOT);
            *g = Some(8);
            panic!("a build or proof panicked");
        })
        .join();
        assert!(r.is_err());
        assert!(SLOT.is_poisoned());
        assert_eq!(*lock_discarding_poisoned(&SLOT), None);
        assert!(!SLOT.is_poisoned());
        *lock_discarding_poisoned(&SLOT) = Some(9);
        assert_eq!(*lock_discarding_poisoned(&SLOT), Some(9));
    }

    /// A nil `newBufferWithLength:` is an error, not undefined behaviour (G8), exercised with the
    /// injected failure; a length above `max_buffer_length` is refused before reaching Metal.
    #[test]
    fn allocation_failure_is_an_error() {
        let _lock = test_lock::shared();
        let Some(device) = ::metal::Device::system_default() else {
            return;
        };
        inject_alloc_failures(1);
        let err = new_buffer_checked(&device, 1 << 20, true, "a test buffer").unwrap_err();
        assert!(
            format!("{err:#}").contains("Metal allocation failed on a test buffer"),
            "{err:#}"
        );
        assert!(new_buffer_checked(&device, 1 << 20, true, "a test buffer").is_ok());
        let too_big = device.max_buffer_length() + (1 << 20);
        assert!(new_buffer_checked(&device, too_big, true, "too big").is_err());
        // The MSM scratch and the NTT tables allocate through it.
        inject_alloc_failures(1);
        let err = msm::MsmScratch::new(&device, 1 << 10, msm::MsmConfig::default())
            .err()
            .expect("MsmScratch::new ignored a failed allocation");
        assert!(
            format!("{err:#}").contains("Metal allocation failed"),
            "{err:#}"
        );
        let Ok(k) = prover::kernels() else {
            return;
        };
        inject_alloc_failures(1);
        let err = ntt::NttPlan::new(&k.device, &k.queue, &k.library, 10)
            .err()
            .expect("NttPlan::new ignored a failed allocation");
        assert!(
            format!("{err:#}").contains("Metal allocation failed"),
            "{err:#}"
        );
        inject_alloc_failures(0);
    }

    #[test]
    fn byte_counts_parse_like_the_zkp_hal() {
        assert_eq!(parse_bytes("4294967296"), Some(4 << 30));
        assert_eq!(parse_bytes("4G"), Some(4 << 30));
        assert_eq!(parse_bytes("8GiB"), Some(8 << 30));
        assert_eq!(parse_bytes("512M"), Some(512 << 20));
        assert_eq!(parse_bytes(" 1k "), Some(1024));
        assert_eq!(parse_bytes("0"), None);
        assert_eq!(parse_bytes("lots"), None);
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
