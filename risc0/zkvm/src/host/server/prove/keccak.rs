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

use std::collections::VecDeque;

use anyhow::{ensure, Context, Result};
use risc0_binfmt::read_sha_halfs;
use risc0_circuit_keccak::{
    get_control_id,
    prove::{keccak_prover, verify_seal, zkr::get_keccak_zkr_shared, HostWitness, KeccakProver},
    KECCAK_CONTROL_IDS, KECCAK_PO2_RANGE,
};
use risc0_core::field::baby_bear::BabyBearElem;
use risc0_zkp::core::digest::{Digest, DIGEST_SHORTS};

use crate::{
    host::client::env::ProveKeccakRequest, host::recursion::prove::prove_shared_zkr,
    receipt::SuccinctReceipt, Unknown,
};

/// Generate a keccak proof that has been lifted.
pub(crate) fn prove_keccak(request: &ProveKeccakRequest) -> Result<SuccinctReceipt<Unknown>> {
    prove_keccak_with(request, None)
}

/// `RISC0_KECCAK_REUSE_HAL=0` restores the pre-K10 session loop: a fresh keccak prover (and Metal
/// HAL, +95-107 ms each) per request, each proof lifted before the next one is proved, and its
/// seal verified inline. Unset (or anything else): [prove_and_lift_keccaks].
pub(crate) const KECCAK_REUSE_HAL_ENV: &str = "RISC0_KECCAK_REUSE_HAL";

/// `RISC0_KECCAK_VERIFY_OVERLAP=1` verifies each keccak seal on a thread beside the next proof in
/// [prove_and_lift_keccaks]; unset (or anything else), inline on the proving thread. Off by default:
/// on a 3-request session it saved 30 ms of ~6.06 s, inside the A-vs-A spread (notes/K10.md).
pub(crate) const KECCAK_VERIFY_OVERLAP_ENV: &str = "RISC0_KECCAK_VERIFY_OVERLAP";

/// Item W7-V2: the device's Metal allocation (`currentAllocatedSize`, every HAL, pool, pipeline
/// and library of the process) at a point of the keccak phase, printed with `RISC0_VERBOSE` and
/// traced at debug level. Returns `(working-set limit, allocated)`; `None` without a Metal device,
/// on other backends, and when neither `RISC0_VERBOSE` nor debug tracing is on (then it asks
/// nothing).
pub(crate) fn device_memory_at(stage: &str) -> Option<(u64, u64)> {
    #[cfg(feature = "metal")]
    {
        // Only when someone reads it: each query makes an `MTLDevice` call.
        if std::env::var_os("RISC0_VERBOSE").is_none()
            && !tracing::enabled!(tracing::Level::DEBUG)
        {
            return None;
        }
        let memory = risc0_zkp::hal::metal::device_memory();
        if let Some((limit, allocated)) = memory {
            tracing::debug!(
                "[keccak-mem] {stage}: {} MiB allocated, limit {} MiB",
                allocated >> 20,
                limit >> 20
            );
            if std::env::var_os("RISC0_VERBOSE").is_some() {
                eprintln!(
                    "[keccak-mem] {stage}: {} MiB allocated, limit {} MiB",
                    allocated >> 20,
                    limit >> 20
                );
            }
        }
        memory
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = stage;
        None
    }
}

/// `RISC0_KECCAK_SMALL_BUDGET_RELEASE=0` turns off item W7-V2's release before a keccak proof
/// outside a session ([prove_keccak], the coprocessor/API path, Metal): this thread's cached
/// segment and recursion provers are dropped when they leave no room for the proof
/// ([needs_room]). Unset (or anything else): on. At the default budget it does not fire.
pub(crate) const KECCAK_SMALL_BUDGET_RELEASE_ENV: &str = "RISC0_KECCAK_SMALL_BUDGET_RELEASE";

/// `RISC0_KECCAK_PO2_HEADROOM_CAP=0` turns off item W7-V2's keccak max po2 cap
/// ([headroom_capped_max_po2]): the session's keccak max po2 stays what the caller (or the
/// upstream default, 17) set even when its proof cannot fit the Metal working-set limit.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) const KECCAK_PO2_HEADROOM_CAP_ENV: &str = "RISC0_KECCAK_PO2_HEADROOM_CAP";

/// Peak device allocation of a session's keccak phase with one full request of po2 14..=18 (the
/// proof, then its lift; the lift's ~1.5 GiB recursion prover dominates at 14 and 15), in MiB, at
/// an unconstrained budget, from a 29 MiB start (pipelines and libraries). Item W7-V2,
/// `keccak_phase_peaks` (M5 Max, 2026-11-14). What [needs_room] adds to the allocation now: the
/// eval chunk shrinks under a smaller budget, so these are upper bounds there.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
const KECCAK_PEAK_MIB: [u64; 5] = [1501, 1506, 2773, 4484, 7906];

/// The smallest working-set budget each po2 of 14..=18 is known to prove under, in MiB: the
/// keccak phase's peak there (`keccak_phase_peaks`). po2 17 proved at 4 GiB with a 3968 MiB peak
/// (the eval chunk shrinks), po2 18 at 7 GiB with 7084 MiB; at 6 GiB po2 18 fails (its 4.4 GiB
/// `evaluated` buffer beside 2.2 GiB of the proof's other buffers). 14..=16 were not run below
/// 4 GiB: their unconstrained peaks. [headroom_max_po2] caps by these.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
const KECCAK_MIN_PEAK_MIB: [u64; 5] = [1501, 1506, 2773, 3968, 7084];

/// Whether a keccak proof of `po2` (and its lift) would take the device's allocation (`allocated`
/// bytes) above `limit` if what is allocated now stays.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) fn needs_room(po2: usize, allocated: u64, limit: u64) -> bool {
    let i =
        po2.clamp(*KECCAK_PO2_RANGE.start(), *KECCAK_PO2_RANGE.end()) - KECCAK_PO2_RANGE.start();
    allocated.saturating_add(KECCAK_PEAK_MIB[i] << 20) > limit
}

/// The largest keccak po2 up to `requested` whose proof fits a `limit`-byte working set
/// ([KECCAK_MIN_PEAK_MIB]); 14 if none does (the proof then fails as before).
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) fn headroom_max_po2(requested: usize, limit: u64) -> usize {
    let start = *KECCAK_PO2_RANGE.start();
    (start..=requested.min(*KECCAK_PO2_RANGE.end()))
        .rev()
        .find(|&po2| KECCAK_MIN_PEAK_MIB[po2 - start] << 20 <= limit)
        .unwrap_or(start)
}

/// Item W7-V2: the keccak max po2 for a session proved here, when the one asked for (`requested`,
/// or the upstream default 17) cannot fit the Metal working-set limit: `Some(lower po2)`, else
/// `None` (keep it). Only po2 18 is capped at the budgets the port targets (to 17 below ~7 GiB:
/// a 4 GiB or 6 GiB budget, an 8 GB Mac); 17 fits 4 GiB. The guest then gets the batch-full
/// signal at the smaller capacity and asks for more, smaller proofs; the receipts verify the same
/// way (every keccak control ID is under `KECCAK_CONTROL_ROOT`). Only for the Metal keccak prover,
/// and only when the keccak proofs are proved by this process (not for a coprocessor callback,
/// which may prove elsewhere). [KECCAK_PO2_HEADROOM_CAP_ENV]`=0` turns it off.
#[cfg(feature = "metal")]
pub(crate) fn headroom_capped_max_po2(requested: Option<u32>) -> Option<u32> {
    if env_disabled(KECCAK_PO2_HEADROOM_CAP_ENV) || !risc0_zkp::hal::metal::enabled("keccak") {
        return None;
    }
    let want = requested.map_or(risc0_circuit_keccak::KECCAK_DEFAULT_PO2, |p| p as usize);
    // A po2 outside 14..=18 is not capped here: the session's own range check rejects it.
    if !KECCAK_PO2_RANGE.contains(&want) {
        return None;
    }
    let (limit, _) = risc0_zkp::hal::metal::device_memory()?;
    let cap = headroom_max_po2(want, limit);
    (cap < want).then(|| {
        let msg = format!(
            "keccak max po2 {want} capped to {cap}: a po2-{want} keccak proof needs ~{} MiB, \
             the Metal working-set limit is {} MiB ({KECCAK_PO2_HEADROOM_CAP_ENV}=0 keeps {want})",
            KECCAK_MIN_PEAK_MIB[want - KECCAK_PO2_RANGE.start()],
            limit >> 20
        );
        if requested.is_some() {
            tracing::warn!("{msg}");
        } else {
            tracing::debug!("{msg}");
        }
        if std::env::var_os("RISC0_VERBOSE").is_some() {
            eprintln!("[keccak] {msg}");
        }
        cap as u32
    })
}

/// `RISC0_KECCAK_PO2_REBUILD`: item W7-V2's keccak prover rebuild on a po2 change in
/// [prove_and_lift_keccaks] ([rebuild_for_po2_change]). `0` never, `1` whenever the pool leaves
/// no room for the next proof; unset (or anything else, "auto"): the same, but only when no
/// explicit working-set budget (`RISC0_METAL_MAX_WORKING_SET_BYTES`) is set. Under an explicit
/// budget the allocator already drains every pool before it would exceed the budget, and the
/// rebuild measured slower than those drains (4 GiB, `keccak_phase_peaks` with `W7V2_AB=8`, see
/// notes/W7-V2.md); without one (a small Mac's real `recommendedMaxWorkingSetSize`) nothing
/// drains and both sizes' buffers stay pooled. Auto keys on the environment variable: a budget
/// set only through `MetalHal::set_working_set_limit` counts as "no budget" here (the runtime's
/// explicit budget is private to risc0-zkp).
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) const KECCAK_PO2_REBUILD_ENV: &str = "RISC0_KECCAK_PO2_REBUILD";

/// Whether [KECCAK_SMALL_BUDGET_RELEASE_ENV] leaves the small-budget release on.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
fn small_budget_release() -> bool {
    !env_disabled(KECCAK_SMALL_BUDGET_RELEASE_ENV)
}

/// Item W7-V2: before a keccak proof outside a session (the coprocessor/API path), release this
/// thread's cached segment and recursion provers when the allocation they leave plus the proof's
/// peak exceeds the working-set limit (Metal). After one in-process `prove_keccak` the lift leaves
/// its recursion prover cached (~1.5 GiB, +~200 MiB per further keccak po2), and at 4 GiB a
/// second po2-17 request then failed (`budget exceeded`). The session path releases them
/// unconditionally (`release_hal_buffers` at the start of the keccak phase). The next lift
/// rebuilds the recursion prover; the PSOs are process-wide and stay. With
/// `RISC0_KECCAK_REUSE_HAL=0` the session's per-request loop also comes here, so at a small
/// budget that revert path now drops the recursion cache between requests too. [needs_room] uses
/// the unconstrained peaks, so at 4 GiB this fires before every po2-17 proof (releasing an empty
/// cache is cheap). Only this thread's caches are reached: provers other live threads cached stay,
/// and at 4 GiB three threads each proving po2 17 in turn exhaust the budget (a panic in the
/// Metal HAL's budget check), so the small-budget guarantee is per thread.
fn release_before_keccak(po2: usize) {
    #[cfg(feature = "metal")]
    {
        if !small_budget_release() {
            return;
        }
        let Some((limit, allocated)) = risc0_zkp::hal::metal::device_memory() else {
            return;
        };
        if needs_room(po2, allocated, limit) {
            super::prover_impl::release_cached_provers();
            let after = device_memory_at("released cached provers before a keccak proof");
            tracing::debug!(
                "keccak po2 {po2}: released this thread's cached provers ({} MiB -> {} MiB \
                 allocated, limit {} MiB)",
                allocated >> 20,
                after.map_or(0, |m| m.1 >> 20),
                limit >> 20
            );
        }
    }
    #[cfg(not(feature = "metal"))]
    let _ = po2;
}

/// Item W7-V2: in [prove_and_lift_keccaks], whether to drop the keccak prover (its HAL and pool)
/// before a request whose po2 differs from the previous one's: when the pool the previous size
/// left, plus the next proof's peak, exceeds the working-set limit (see [KECCAK_PO2_REBUILD_ENV]
/// for when it applies). A rebuild measured 2.9-9.9 ms at 4 GiB (the PSOs are process-wide; the
/// time is the new HAL and the old pool's release) and the next proof allocates fresh buffers.
/// Without it, a session whose requests change po2 keeps both sizes' buffers pooled (K10's
/// mixed-po2 issue: 14,17,14,17 peaks 528 MiB above a lone 17 at the default budget).
fn rebuild_for_po2_change(previous: Option<usize>, po2: usize) -> bool {
    if previous.is_none_or(|p| p == po2) {
        return false;
    }
    #[cfg(feature = "metal")]
    {
        let knob = std::env::var(KECCAK_PO2_REBUILD_ENV).ok();
        let explicit_budget =
            std::env::var_os(risc0_zkp::hal::metal::MAX_WORKING_SET_ENV).is_some();
        rebuild_allowed(knob.as_deref(), explicit_budget)
            && risc0_zkp::hal::metal::device_memory()
                .is_some_and(|(limit, allocated)| needs_room(po2, allocated, limit))
    }
    #[cfg(not(feature = "metal"))]
    false
}

/// [KECCAK_PO2_REBUILD_ENV]'s value (`knob`): whether a po2-change rebuild may fire (it then
/// still needs [needs_room]). `explicit_budget`: an explicit working-set budget is set.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
fn rebuild_allowed(knob: Option<&str>, explicit_budget: bool) -> bool {
    match knob.map(str::trim) {
        Some("0") => false,
        Some("1") => true,
        _ => !explicit_budget,
    }
}

/// How many times [prove_and_lift_keccaks] rebuilt its keccak prover for a po2 change in this
/// process (read by the A/B harness to check that the arm it runs is the one it names).
pub(crate) static KECCAK_PO2_REBUILDS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// The arms of the in-process rebuild A/B (`keccak_phase_peaks`, `W7V2_AB`): the knob each arm
/// sets and its value. A = never rebuild, B = rebuild whenever there is no room.
#[cfg(test)]
pub(crate) const REBUILD_AB_ARMS: [(&str, &str, &str); 2] = [
    ("A", KECCAK_PO2_REBUILD_ENV, "0"),
    ("B", KECCAK_PO2_REBUILD_ENV, "1"),
];

fn env_disabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v.trim() == "0")
}

/// Whether the session loop takes [prove_and_lift_keccaks] (see [KECCAK_REUSE_HAL_ENV]).
pub(crate) fn reuse_keccak_prover() -> bool {
    !env_disabled(KECCAK_REUSE_HAL_ENV)
}

/// [prove_keccak], from the request's witness if `risc0_circuit_keccak::prove::host_witness` has
/// made it already.
pub(crate) fn prove_keccak_with(
    request: &ProveKeccakRequest,
    witness: Option<HostWitness>,
) -> Result<SuccinctReceipt<Unknown>> {
    let _gpu = super::exclusive_gpu();
    release_before_keccak(request.po2);
    let t_prove = std::time::Instant::now();
    let seal = {
        let prover = keccak_prover()?;
        let seal = prove_seal(prover.as_ref(), request, witness)?;
        // Make sure we have a valid seal so we can fail early if anything went wrong
        verify_seal(&seal)?;
        seal
    };
    let prove_ms = t_prove.elapsed().as_secs_f64() * 1000.0;
    lift_seal(request, &seal, prove_ms)
}

/// Prove every request of a session (K10, LEADS #53 and #57): one keccak prover (one HAL, whose
/// buffer pool the proofs share) proves all the seals; it is dropped before the first lift, so no
/// lift runs beside its pool (before K10 each request built a fresh HAL, and the recursion prover
/// the previous lift cached stayed alive beside the next keccak proof). Each seal is verified on
/// the CPU while the GPU proves the next one if [KECCAK_VERIFY_OVERLAP_ENV] is `1`; at most
/// one verifier thread runs at a time (the previous one is joined before the next is started), and
/// every seal is verified before any lift. `on_receipt` sees each lifted receipt in request order
/// (with its index), e.g. to insert it into the union tree.
pub(crate) fn prove_and_lift_keccaks(
    requests: &[ProveKeccakRequest],
    mut on_receipt: impl FnMut(usize, SuccinctReceipt<Unknown>) -> Result<()>,
) -> Result<()> {
    if requests.is_empty() {
        return Ok(());
    }
    let overlap = std::env::var(KECCAK_VERIFY_OVERLAP_ENV).is_ok_and(|v| v.trim() == "1");
    let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
    // A detached (not scoped) verifier: if a proof fails while one runs, the error is returned and
    // the verifier finishes on its own instead of turning the error into a panic at scope exit.
    type Pending = (usize, std::thread::JoinHandle<Result<()>>);
    let join = |(i, handle): Pending| -> Result<()> {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("keccak request {i}: seal verification panicked"))?
            .with_context(|| format!("keccak request {i}"))
    };
    let mut seals = Vec::with_capacity(requests.len());
    {
        let _gpu = super::exclusive_gpu();
        let t_hal = std::time::Instant::now();
        let mut prover = Some(keccak_prover()?);
        device_memory_at("keccak prover built");
        if verbose {
            eprintln!(
                "[keccak] prover for {} requests in {:.1}ms",
                requests.len(),
                t_hal.elapsed().as_secs_f64() * 1000.0
            );
        }
        let mut pending: Option<Pending> = None;
        let mut previous_po2 = None;
        for (i, request) in requests.iter().enumerate() {
            if rebuild_for_po2_change(previous_po2, request.po2) {
                let t_rebuild = std::time::Instant::now();
                drop(prover.take());
                prover = Some(keccak_prover()?);
                KECCAK_PO2_REBUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                device_memory_at("keccak prover rebuilt for a po2 change");
                if verbose {
                    eprintln!(
                        "[keccak] {i}: po2 {:?} -> {}: prover rebuilt in {:.1}ms",
                        previous_po2,
                        request.po2,
                        t_rebuild.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
            previous_po2 = Some(request.po2);
            let prover = prover.as_deref().expect("keccak prover");
            let t_prove = std::time::Instant::now();
            let seal =
                prove_seal(prover, request, None).with_context(|| format!("keccak request {i}"))?;
            let prove_ms = t_prove.elapsed().as_secs_f64() * 1000.0;
            if verbose {
                eprintln!("[keccak] {i}: po2={} prove={prove_ms:.1}ms", request.po2);
            }
            if let Some(previous) = pending.take() {
                join(previous)?;
            }
            let seal = std::sync::Arc::new(seal);
            if overlap {
                let seal = seal.clone();
                pending = Some((i, std::thread::spawn(move || verify_seal(&seal))));
            } else {
                verify_seal(&seal).with_context(|| format!("keccak request {i}"))?;
            }
            seals.push((seal, prove_ms));
        }
        if let Some(last) = pending.take() {
            join(last)?;
        }
        device_memory_at("keccak proofs done, prover live");
        // The prover (its HAL and pool) drops here, before any lift.
    }
    device_memory_at("keccak prover dropped");
    for (i, (request, (seal, prove_ms))) in requests.iter().zip(seals).enumerate() {
        let receipt = lift_seal(request, &seal, prove_ms)?;
        on_receipt(i, receipt)?;
    }
    device_memory_at("keccak lifts done");
    Ok(())
}

/// The keccak seal of `request` (from `witness` if the prover takes a host witness), with its
/// claim digest checked against the request's.
fn prove_seal(
    prover: &dyn KeccakProver,
    request: &ProveKeccakRequest,
    witness: Option<HostWitness>,
) -> Result<Vec<u32>> {
    let seal = match witness.and_then(|witness| prover.prove_host_witness(witness)) {
        Some(seal) => seal?,
        None => prover.prove(&request.input, request.po2)?,
    };

    let claim_digest: Digest = read_sha_halfs(&mut VecDeque::from_iter(
        bytemuck::checked::cast_slice::<_, BabyBearElem>(&seal[0..DIGEST_SHORTS])
            .iter()
            .copied()
            .map(u32::from),
    ))?;

    ensure!(
        request.claim_digest == claim_digest,
        "keccak claim digest mismatch, expected: {:?}, actual: {claim_digest:?}",
        request.claim_digest
    );
    Ok(seal)
}

/// Lift a verified keccak seal of `request` with the keccak zkr of its po2.
fn lift_seal(
    request: &ProveKeccakRequest,
    seal: &[u32],
    prove_ms: f64,
) -> Result<SuccinctReceipt<Unknown>> {
    let claim_sha_input = request
        .claim_digest
        .as_words()
        .iter()
        .copied()
        .flat_map(|x| [x & 0xffff, x >> 16])
        .map(BabyBearElem::new)
        .collect::<Vec<_>>();

    let mut zkr_input: Vec<u32> = Vec::new();
    zkr_input.extend(request.control_root.as_words());
    zkr_input.extend(seal);
    zkr_input.extend(bytemuck::cast_slice(claim_sha_input.as_slice()));
    zkr_input.push(1 << request.po2);

    let t_lift = std::time::Instant::now();
    let receipt = prove_shared_zkr(
        get_keccak_zkr_shared(request.po2)?,
        get_control_id(request.po2),
        KECCAK_CONTROL_IDS.to_vec(),
        bytemuck::cast_slice(zkr_input.as_slice()),
    );
    if std::env::var_os("RISC0_VERBOSE").is_some() {
        eprintln!(
            "[keccak] prove={prove_ms:.1}ms lift={:.1}ms",
            t_lift.elapsed().as_secs_f64() * 1000.0
        );
    }
    receipt
}

#[cfg(test)]
mod policy_tests {
    use super::{
        headroom_max_po2, needs_room, rebuild_allowed, KECCAK_PO2_REBUILD_ENV, REBUILD_AB_ARMS,
    };

    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;

    #[test]
    fn headroom_cap_by_budget() {
        // po2 17 fits 4 GiB (3968 MiB measured), 18 needs 7084 MiB.
        assert_eq!(headroom_max_po2(17, 4 * GIB), 17);
        assert_eq!(headroom_max_po2(18, 4 * GIB), 17);
        assert_eq!(headroom_max_po2(18, 6 * GIB), 17);
        assert_eq!(headroom_max_po2(18, 7084 * MIB), 18);
        assert_eq!(headroom_max_po2(18, 8 * GIB), 18);
        assert_eq!(headroom_max_po2(18, 28 * GIB), 18);
        // Never above the request, never below 14.
        assert_eq!(headroom_max_po2(15, 28 * GIB), 15);
        assert_eq!(headroom_max_po2(17, 3 * GIB), 16);
        assert_eq!(headroom_max_po2(18, GIB), 14);
    }

    /// Review finding (W7-V2 fix round 1): the rebuild A/B harness toggled
    /// `RISC0_KECCAK_SMALL_BUDGET_RELEASE`, which `prove_and_lift_keccaks` never reads, so both
    /// arms ran the same code. Each arm must set the rebuild knob, A must never rebuild and B must
    /// be allowed to under the explicit budget the A/B runs at.
    #[test]
    fn rebuild_ab_arms_toggle_the_rebuild() {
        let [(a, a_env, a_val), (b, b_env, b_val)] = REBUILD_AB_ARMS;
        assert_eq!((a, b), ("A", "B"));
        assert_eq!(a_env, KECCAK_PO2_REBUILD_ENV);
        assert_eq!(b_env, KECCAK_PO2_REBUILD_ENV);
        for explicit_budget in [false, true] {
            assert!(!rebuild_allowed(Some(a_val), explicit_budget));
            assert!(rebuild_allowed(Some(b_val), explicit_budget));
        }
        // Auto: only without an explicit budget.
        assert!(rebuild_allowed(None, false));
        assert!(!rebuild_allowed(None, true));
        assert!(!rebuild_allowed(Some("auto"), true));
    }

    #[test]
    fn room_for_next_keccak_proof() {
        // A session's keccak phase starts from ~29 MiB: room for any po2 at the default budget.
        for po2 in 14..=18 {
            assert!(!needs_room(po2, 29 * MIB, 28 * GIB), "{po2}");
        }
        // At 4 GiB, a po2-14 pool (~0.5 GiB) beside a po2-17 proof does not fit; the po2-17 pool
        // (~1.3 GiB) beside a po2-14 proof does.
        assert!(needs_room(17, 500 * MIB, 4 * GIB));
        assert!(!needs_room(14, 1367 * MIB, 4 * GIB));
        // The lift's cached recursion prover (~1.5 GiB) beside a po2-17 proof at 4 GiB does not.
        assert!(needs_room(17, 1514 * MIB, 4 * GIB));
        assert!(!needs_room(17, 1514 * MIB, 8 * GIB));
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use risc0_circuit_keccak::{
        compute_keccak_digest, max_keccak_inputs, KeccakState, KECCAK_CONTROL_IDS,
        KECCAK_CONTROL_ROOT, KECCAK_PO2_RANGE,
    };

    use super::*;
    use crate::{SuccinctReceiptVerifierParameters, VerifierContext};

    /// A full batch of `po2` (the keccak guest's inputs), with its claim digest.
    pub(crate) fn full_request(po2: usize) -> ProveKeccakRequest {
        let mut state = KeccakState::default();
        let mut pows = 987654321_u64;
        for part in state.as_mut_slice() {
            *part = pows;
            pows = pows.wrapping_mul(123456789);
        }
        let input = vec![state; max_keccak_inputs(po2)];
        ProveKeccakRequest {
            claim_digest: compute_keccak_digest(bytemuck::cast_slice(&input)),
            po2,
            control_root: KECCAK_CONTROL_ROOT,
            input,
        }
    }

    /// Item W7-V2: the device's peak Metal allocation over [prove_and_lift_keccaks] (one keccak
    /// prover per call, as a session's keccak phase) for each sequence of request po2s in
    /// `W7V2_PO2S` (sequences separated by `;`, po2s by `,`; default `14;15;16;17`), each
    /// receipt verified with the keccak verifier parameters and its lift control ID checked.
    /// Prints `W7V2PEAK seq=.. ms=.. peak_mib=.. start_mib=..`. Ignored: GPU, seconds per
    /// sequence, up to ~9 GiB at po2 18; run alone under the machine's GPU lock (the peak counter
    /// is process-wide).
    #[test]
    #[ignore]
    fn keccak_phase_peaks() {
        let spec = std::env::var("W7V2_PO2S").unwrap_or_else(|_| "14;15;16;17".into());
        let ctx = VerifierContext::default()
            .with_succinct_verifier_parameters(SuccinctReceiptVerifierParameters::for_keccak());
        let hal = risc0_zkp::hal::metal::MetalHalPoseidon2::new();
        // `W7V2_AB=<n>`: run the sequences n times, alternating the arms of [REBUILD_AB_ARMS]
        // (RISC0_KECCAK_PO2_REBUILD=0 (A), =1 (B)) per round, in one process. This sets process
        // env without a lock: the test is ignored and must run alone.
        let ab_rounds: usize = std::env::var("W7V2_AB").map_or(0, |v| v.parse().unwrap());
        let rounds: Vec<Option<(&str, &str, &str)>> = if ab_rounds == 0 {
            vec![None]
        } else {
            (0..ab_rounds)
                .flat_map(|_| REBUILD_AB_ARMS.map(Some))
                .collect()
        };
        let small_limit =
            risc0_zkp::hal::metal::device_memory().is_some_and(|(limit, _)| limit < 5 * (1 << 30));
        for arm in rounds {
            if let Some((_, env, value)) = arm {
                std::env::set_var(env, value);
            }
            for seq in spec.split(';').filter(|s| !s.trim().is_empty()) {
                let po2s: Vec<usize> = seq.split(',').map(|s| s.trim().parse().unwrap()).collect();
                assert!(po2s.iter().all(|p| KECCAK_PO2_RANGE.contains(p)), "{seq}");
                let requests: Vec<_> = po2s.iter().map(|&p| full_request(p)).collect();
                // As a session's keccak phase starts (`release_hal_buffers` in
                // `prove_session_tail`): without this, the recursion provers the previous sequence's
                // lifts cached on this thread (~1.5 GiB, +~200 MiB per keccak zkr) sit beside it.
                crate::host::server::prove::release_hal_buffers();
                let start = hal.allocated_size();
                hal.reset_peak_allocated_size();
                let rebuilds0 = KECCAK_PO2_REBUILDS.load(std::sync::atomic::Ordering::Relaxed);
                let t = std::time::Instant::now();
                prove_and_lift_keccaks(&requests, |i, receipt| {
                    receipt.verify_integrity_with_context(&ctx)?;
                    ensure!(
                        receipt.control_id
                            == KECCAK_CONTROL_IDS[po2s[i] - KECCAK_PO2_RANGE.start()],
                        "request {i}: lift control ID"
                    );
                    Ok(())
                })
                .unwrap();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let rebuilds =
                    KECCAK_PO2_REBUILDS.load(std::sync::atomic::Ordering::Relaxed) - rebuilds0;
                let changes_at_17 = po2s
                    .windows(2)
                    .filter(|w| w[0] != w[1] && w.contains(&17))
                    .count();
                match arm {
                    Some(("A", ..)) => assert_eq!(rebuilds, 0, "arm A rebuilt: {seq}"),
                    // Below 5 GiB a po2 change into or out of 17 leaves no room for the next
                    // proof (any pool beside a po2-17 proof's 4484 MiB, or a po2-17 pool beside a
                    // po2-16 proof), so arm B must rebuild at least once.
                    Some(_) if small_limit && changes_at_17 > 0 => {
                        assert!(rebuilds > 0, "arm B did not rebuild: {seq}")
                    }
                    _ => {}
                }
                println!(
                    "W7V2PEAK arm={} seq={seq} ms={ms:.1} rebuilds={rebuilds} peak_mib={} \
                     start_mib={} end_mib={}",
                    arm.map_or("-", |a| a.0),
                    hal.peak_allocated_size() >> 20,
                    start >> 20,
                    hal.allocated_size() >> 20
                );
            }
        }
    }

    /// Item W7-V2: the coprocessor/API path in one process: [prove_keccak] (one prover and one
    /// lift per call, as `ProverServer::prove_keccak`) for each po2 of `W7V2_PO2S` (`,`-separated,
    /// default `17,17`) on this thread, each receipt verified. Prints `W7V2STANDALONE po2=..
    /// ms=.. peak_mib=.. start_mib=..`. At 4 GiB the second po2-17 call failed before W7-V2
    /// (the first lift's recursion prover stays cached); `RISC0_KECCAK_SMALL_BUDGET_RELEASE=0`
    /// reproduces that. Ignored: GPU; run alone under the machine's GPU lock.
    #[test]
    #[ignore]
    fn keccak_standalone_repeat() {
        let spec = std::env::var("W7V2_PO2S").unwrap_or_else(|_| "17,17".into());
        let ctx = VerifierContext::default()
            .with_succinct_verifier_parameters(SuccinctReceiptVerifierParameters::for_keccak());
        let hal = risc0_zkp::hal::metal::MetalHalPoseidon2::new();
        crate::host::server::prove::release_hal_buffers();
        for po2 in spec.split(',').map(|s| s.trim().parse::<usize>().unwrap()) {
            let request = full_request(po2);
            let start = hal.allocated_size();
            hal.reset_peak_allocated_size();
            let t = std::time::Instant::now();
            let receipt = prove_keccak(&request).unwrap();
            receipt.verify_integrity_with_context(&ctx).unwrap();
            assert_eq!(
                receipt.control_id,
                KECCAK_CONTROL_IDS[po2 - KECCAK_PO2_RANGE.start()]
            );
            println!(
                "W7V2STANDALONE po2={po2} ms={:.1} peak_mib={} start_mib={}",
                t.elapsed().as_secs_f64() * 1e3,
                hal.peak_allocated_size() >> 20,
                start >> 20
            );
        }
    }
}
