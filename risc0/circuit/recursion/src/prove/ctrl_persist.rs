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

//! Committed ctrl groups kept on the host across provers (item C13, Metal).
//!
//! A recursion prover caches each program's committed ctrl group (`cached_ctrl_group`), but a
//! long-lived worker drops its provers, and with them their HALs, between jobs
//! (`release_hal_buffers`, and on Metal the release before identity_p254). The next job then
//! recommitted every program's ctrl: for identity_p254 that is 23 columns x 2^20 rows of
//! Poseidon254 (~180 ms of the ~1.1 s proof), for a poseidon2 lift/join ~12 ms.
//!
//! This module keeps a host copy of such a group (coefficients and the whole Merkle tree, no
//! evaluated domain: [PolyGroupHost]) process-wide, keyed by hash suite and program, and the new
//! prover rebuilds the group from it ([PolyGroup::from_host]: uploads, one expand NTT, no hashing,
//! ~6.5 ms on an M5 Max). The copies are host memory, so they add nothing to the Metal working
//! set; the device buffers of a rebuilt group are the ones a commit would have allocated.
//!
//! What is kept (`RISC0_METAL_REC_CTRL_PERSIST`, read once):
//! - default (`1`): the Poseidon254 groups only, i.e. identity_p254's ctrl (~103 MB of host
//!   memory, code copy included). A poseidon2 ctrl rebuilds in about the time it commits in
//!   (~6.5 vs ~12 ms, a saving no A/B resolves) for ~100 MB each, so those are not kept;
//! - `all`: every hash suite's groups (as many as fit, see below);
//! - `0`: nothing (the pre-C13 behavior).
//!
//! The copies may hold at most [cap_bytes] (a 32nd of the GPU working-set budget: 128 MiB at a
//! 4 GiB budget, enough for identity_p254's group; `RISC0_METAL_REC_CTRL_PERSIST_BYTES`
//! overrides). When they do not all fit, the order is deterministic ([Priority]): Poseidon254
//! groups are kept first, then larger groups, then by key; the lowest go first. Measured commit
//! times are only logged (on a shared GPU they are too noisy to rank by).
//!
//! Why a 32nd: it is a budget-relative bound on host memory (not Metal working set) that holds
//! identity_p254's ~98 MiB group from a 4 GiB budget up, and stays a small share of host RAM on
//! the minimum 16 GB Mac (128 MiB at 4 GiB, 256 MiB at 8 GiB).

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard, OnceLock, Weak},
};

use risc0_zkp::{
    field::baby_bear::BabyBearElem,
    prove::poly_group::{PolyGroup, PolyGroupHost},
};

use super::{CtrlKey, Program};

/// Environment variable (read once): `0` disables the persisted ctrl groups.
pub(crate) const PERSIST_ENV: &str = "RISC0_METAL_REC_CTRL_PERSIST";
/// Environment variable (read once): the host bytes the persisted groups may hold (default: a
/// 32nd of the GPU working-set budget). `0` keeps nothing.
pub(crate) const PERSIST_BYTES_ENV: &str = "RISC0_METAL_REC_CTRL_PERSIST_BYTES";

/// What [PERSIST_ENV] keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Nothing (the pre-C13 behavior).
    Off,
    /// The Poseidon254 groups (identity_p254's ctrl): the default.
    P254,
    /// Every hash suite's groups.
    All,
}

/// `None` for an unparsable value.
pub(crate) fn parse_persist(v: &str) -> Option<Mode> {
    match v.trim().to_ascii_lowercase().as_str() {
        "" | "1" | "on" | "true" | "p254" => Some(Mode::P254),
        "all" => Some(Mode::All),
        "0" | "off" | "false" => Some(Mode::Off),
        _ => None,
    }
}

/// Test hook (in-process A/B): 0 = follow [PERSIST_ENV], else 1 + the forced [Mode] index.
#[cfg(test)]
pub(crate) static TEST_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Force a [Mode] in this process (tests); `None` follows [PERSIST_ENV] again.
#[cfg(test)]
pub(crate) fn set_test_mode(m: Option<Mode>) {
    let v = match m {
        None => 0,
        Some(Mode::Off) => 1,
        Some(Mode::P254) => 2,
        Some(Mode::All) => 3,
    };
    TEST_MODE.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// The process-wide [Mode] (Metal builds; [Mode::Off] elsewhere).
pub(crate) fn mode() -> Mode {
    #[cfg(test)]
    match TEST_MODE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return Mode::Off,
        2 => return Mode::P254,
        3 => return Mode::All,
        _ => {}
    }
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        if !cfg!(feature = "metal") {
            return Mode::Off;
        }
        match std::env::var(PERSIST_ENV) {
            Ok(v) => parse_persist(&v).unwrap_or_else(|| {
                tracing::error!("{PERSIST_ENV}={v:?} is not 0/1/all; keeping p254 ctrl groups");
                Mode::P254
            }),
            Err(_) => Mode::P254,
        }
    })
}

/// Whether persisted groups are looked up at all.
pub(crate) fn enabled() -> bool {
    mode() != Mode::Off
}

/// Whether `mode` keeps the groups of hash suite `hashfn` (a `HashSuite::name`).
pub(crate) fn mode_wants(mode: Mode, hashfn: &str) -> bool {
    match mode {
        Mode::Off => false,
        Mode::P254 => hashfn == p254_suite_name(),
        Mode::All => true,
    }
}

/// Whether this process keeps the groups of hash suite `hashfn`.
pub(crate) fn wants(hashfn: &str) -> bool {
    mode_wants(mode(), hashfn)
}

fn p254_suite_name() -> &'static str {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| {
        risc0_zkp::core::hash::poseidon_254::Poseidon254HashSuite::new_suite()
            .name
            .clone()
    })
}

/// The host bytes the persisted groups may hold, for a GPU working-set budget of `budget` bytes.
pub(crate) fn cap_bytes(budget: usize) -> usize {
    static CAP: OnceLock<Option<usize>> = OnceLock::new();
    let over = *CAP.get_or_init(|| {
        std::env::var(PERSIST_BYTES_ENV)
            .ok()
            .and_then(|v| v.trim().parse().ok())
    });
    over.unwrap_or(budget / 32)
}

struct Entry {
    code: Arc<Vec<BabyBearElem>>,
    /// The shared program last matched to `code` (pointer equality skips the compare).
    program: Weak<Program>,
    group: Arc<PolyGroupHost<BabyBearElem>>,
    /// Wall time of the commit a rebuild replaces (logged only: on a shared GPU it is too noisy
    /// to rank by).
    commit_ms: f64,
    bytes: usize,
}

type Key = (String, CtrlKey);

/// Keep / eviction order, highest kept first. Deterministic (no measured times): Poseidon254
/// groups first (their commit is the expensive one: ~180 ms against ~12 ms for poseidon2),
/// then larger groups (larger programs take longer to commit), then the key for a total order.
type Priority = (u8, usize, Key);

fn priority(hashfn: &str, key: CtrlKey, bytes: usize) -> Priority {
    let suite = u8::from(hashfn == p254_suite_name());
    (suite, bytes, (hashfn.to_string(), key))
}

#[derive(Default)]
struct State {
    entries: HashMap<Key, Entry>,
    restores: usize,
    snapshots: usize,
}

fn state() -> MutexGuard<'static, State> {
    static S: OnceLock<Mutex<State>> = OnceLock::new();
    S.get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn total_bytes(s: &State) -> usize {
    s.entries.values().map(|e| e.bytes).sum()
}

/// The code of `program` as a persisted group under `key` (any hash suite) holds it, so a new
/// prover's code cache shares it instead of copying the program's code again. `None` if no
/// entry is for exactly this code.
pub(crate) fn code_for(
    key: CtrlKey,
    program: &Program,
    shared: Option<&Arc<Program>>,
) -> Option<Arc<Vec<BabyBearElem>>> {
    if !enabled() {
        return None;
    }
    let mut s = state();
    let mut code: Option<Arc<Vec<BabyBearElem>>> = None;
    for ((_, k), e) in s.entries.iter_mut() {
        if *k != key {
            continue;
        }
        if let Some(c) = &code {
            if Arc::ptr_eq(c, &e.code) {
                continue;
            }
        }
        let same_program = shared.is_some_and(|a| {
            e.program
                .upgrade()
                .is_some_and(|seen| Arc::ptr_eq(a, &seen))
        });
        if same_program || super::same_code(&e.code, &program.code) {
            if let Some(a) = shared {
                e.program = Arc::downgrade(a);
            }
            code = Some(e.code.clone());
        }
    }
    code
}

/// The persisted group of `hashfn`'s prover for `key`, if it was built from `code` (the same
/// `Arc` that [code_for] handed out: the caller's code cache already holds exactly this
/// program's code).
pub(crate) fn lookup(
    hashfn: &str,
    key: CtrlKey,
    code: &Arc<Vec<BabyBearElem>>,
) -> Option<Arc<PolyGroupHost<BabyBearElem>>> {
    if !enabled() {
        return None;
    }
    let s = state();
    let e = s.entries.get(&(hashfn.to_string(), key))?;
    Arc::ptr_eq(&e.code, code).then(|| e.group.clone())
}

/// Count a rebuild of `hashfn`/`key` that took `ms` (logged; tests and the bench count them).
pub(crate) fn record_restore(hashfn: &str, key: CtrlKey, ms: f64) {
    let mut s = state();
    s.restores += 1;
    if let Some(e) = s.entries.get(&(hashfn.to_string(), key)) {
        tracing::debug!(
            "C13 ctrl persist: {hashfn} rebuild {ms:.1} ms (commit was {:.1} ms)",
            e.commit_ms
        );
    }
}

/// Whether a group of `bytes` would be kept under `cap`: not kept yet, and it fits together
/// with the kept entries of higher [Priority] (lower ones would be evicted). (The caller checks
/// [wants] first.)
pub(crate) fn would_keep(hashfn: &str, key: CtrlKey, bytes: usize, cap: usize) -> bool {
    if bytes > cap {
        return false;
    }
    let s = state();
    if s.entries.contains_key(&(hashfn.to_string(), key)) {
        return false;
    }
    let mine = priority(hashfn, key, bytes);
    let above: usize = s
        .entries
        .iter()
        .filter(|(k, e)| priority(&k.0, k.1, e.bytes) > mine)
        .map(|(_, e)| e.bytes)
        .sum();
    above + bytes <= cap
}

/// Keep `group` (just committed by `hashfn`'s prover in `commit_ms`) for `key` / `code`, then
/// evict the lowest [Priority] until the copies fit `cap`.
pub(crate) fn store<H>(
    hashfn: &str,
    key: CtrlKey,
    code: Arc<Vec<BabyBearElem>>,
    shared: Option<&Arc<Program>>,
    group: &PolyGroup<H>,
    commit_ms: f64,
    cap: usize,
) where
    H: risc0_zkp::hal::Hal<Elem = BabyBearElem>,
{
    let host = group.to_host();
    let bytes = host.bytes() + code.len() * std::mem::size_of::<BabyBearElem>();
    let mut s = state();
    s.snapshots += 1;
    s.entries.insert(
        (hashfn.to_string(), key),
        Entry {
            code,
            program: shared.map(Arc::downgrade).unwrap_or_default(),
            group: Arc::new(host),
            commit_ms,
            bytes,
        },
    );
    while total_bytes(&s) > cap {
        let Some(victim) = s
            .entries
            .iter()
            .min_by_key(|(k, e)| priority(&k.0, k.1, e.bytes))
            .map(|(k, _)| k.clone())
        else {
            break;
        };
        s.entries.remove(&victim);
    }
    tracing::debug!(
        "C13 ctrl persist: kept {hashfn} group ({bytes} B, commit {commit_ms:.1} ms); {} groups, {} B",
        s.entries.len(),
        total_bytes(&s)
    );
}

/// Environment variable (read per call): `0` makes [prewarm_p254_ctrl] a no-op (item C16 (b)).
pub(crate) const PREWARM_ENV: &str = "RISC0_METAL_REC_P254_PREWARM";

/// The device headroom [prewarm_p254_ctrl] wants free under the working-set limit: the commit of
/// identity_p254's ctrl (23 columns x 2^20 rows of Poseidon254) on a fresh p254 HAL adds a peak of
/// ~217 MB of device memory (coefficients, the evaluated domain, the Merkle nodes and the NTT
/// scratch; measured in notes/C16.md), rounded up with a margin.
#[cfg(feature = "metal")]
const PREWARM_DEVICE_BYTES: u64 = 512 << 20;

/// What [prewarm_p254_ctrl] did.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum P254Prewarm {
    /// The group was committed on a temporary p254 prover and kept: the first identity_p254 of
    /// this process rebuilds it instead of committing it.
    Kept {
        /// Wall time of the prewarm (prover creation, ctrl commit, host snapshot, release).
        ms: f64,
        /// Host bytes the kept group holds.
        bytes: usize,
    },
    /// A group for this program is already kept (a previous prewarm or identity_p254).
    AlreadyKept,
    /// Nothing done; the reason (persistence off, not a Metal build, the knob, the budget).
    Skipped(String),
}

/// Item C16 (b): commit identity_p254's ctrl group once, ahead of the first Groth16 wrap of the
/// process, and keep its host copy (C13), so that the first wrap rebuilds it in ~6 ms instead of
/// committing it (~150-170 ms of Poseidon254 Merkle hashing plus the ~88 MB snapshot copy). Later
/// wraps are unchanged: they already rebuild it.
///
/// `po2` is the recursion programs' po2 (risc0-zkvm's `RECURSION_PO2`, 18). The commit runs on a
/// temporary p254 prover that is dropped before returning (no HAL stays alive; the kept copy is
/// host memory, ~103 MB, counted against C13's cap). It holds the process-wide GPU lock for its
/// duration (~0.22-0.26 s on an M5 Max with warm pipelines), so call it when the GPU is idle, e.g.
/// at worker start; it is safe from any thread.
///
/// Budget-gated: it does nothing when C13 persistence does not keep Poseidon254 groups
/// (`RISC0_METAL_REC_CTRL_PERSIST`), when the group would not fit C13's host cap, or when the
/// device has less than ~512 MiB free under its working-set limit (so a caller never pushes a
/// small Mac's budget, e.g. while a segment prover holds its pool). `RISC0_METAL_REC_P254_PREWARM=0`
/// turns it off.
///
/// The headroom and cap gates are checked before the GPU lock and again once it is held
/// (W7-C1, LEADS2 S5): a Prove holds the lock for its whole call, so a prewarm started on a
/// background thread during a Prove returns `Skipped("no headroom under the GPU lock: ..")`
/// instead of committing beside that Prove's pools. `RISC0_METAL_P254_PREWARM_GATE=early`
/// restores the pre-W7-C1 order (before the lock only).
pub fn prewarm_p254_ctrl(po2: usize) -> anyhow::Result<P254Prewarm> {
    if std::env::var(PREWARM_ENV).is_ok_and(|v| v.trim() == "0") {
        return Ok(P254Prewarm::Skipped(format!("{PREWARM_ENV}=0")));
    }
    prewarm_p254_ctrl_impl(po2)
}

#[cfg(not(feature = "metal"))]
fn prewarm_p254_ctrl_impl(_po2: usize) -> anyhow::Result<P254Prewarm> {
    Ok(P254Prewarm::Skipped("not a Metal build".into()))
}

/// Environment variable (read per call): `early` restores the pre-W7-C1 gate order of
/// [prewarm_p254_ctrl], which checks the device headroom and C13's cap only before taking
/// `risc0_zkp::hal::metal::exclusive()`. Default (unset or anything else): check before the lock
/// and again once it is held (LEADS2 S5).
pub(crate) const PREWARM_GATE_ENV: &str = "RISC0_METAL_P254_PREWARM_GATE";

/// Whether [PREWARM_GATE_ENV] asks for the pre-W7-C1 check order (the gates before the lock only).
#[cfg(feature = "metal")]
pub(crate) fn gate_early_only() -> bool {
    std::env::var(PREWARM_GATE_ENV).is_ok_and(|v| v.trim().eq_ignore_ascii_case("early"))
}

/// Where [prewarm_p254_ctrl] reads the working-set limit and the device's allocation total:
/// [risc0_zkp::hal::metal::device_memory] in production, a fake in the tests (W7-C1).
#[cfg(feature = "metal")]
pub(crate) trait Headroom {
    /// `(limit, allocated)` in bytes, or `None` without a Metal device.
    fn device_memory(&self) -> Option<(u64, u64)>;
}

#[cfg(feature = "metal")]
pub(crate) struct MetalDeviceHeadroom;

#[cfg(feature = "metal")]
impl Headroom for MetalDeviceHeadroom {
    fn device_memory(&self) -> Option<(u64, u64)> {
        risc0_zkp::hal::metal::device_memory()
    }
}

/// The gates of [prewarm_p254_ctrl] that depend on what other threads do: the group is not kept
/// yet, the device has [PREWARM_DEVICE_BYTES] free under its working-set limit, and `bytes` fit
/// C13's cap for that limit. `Ok(())` to go on, else the outcome to return.
#[cfg(feature = "metal")]
fn prewarm_gate(
    src: &dyn Headroom,
    hashfn: &str,
    key: CtrlKey,
    bytes: usize,
    when: &str,
) -> Result<(), P254Prewarm> {
    if state().entries.contains_key(&(hashfn.to_string(), key)) {
        return Err(P254Prewarm::AlreadyKept);
    }
    let Some((limit, allocated)) = src.device_memory() else {
        return Err(P254Prewarm::Skipped("no Metal device".into()));
    };
    if allocated.saturating_add(PREWARM_DEVICE_BYTES) > limit {
        return Err(P254Prewarm::Skipped(format!(
            "no headroom {when}: {} MiB allocated + {} MiB > {} MiB working-set limit",
            allocated >> 20,
            PREWARM_DEVICE_BYTES >> 20,
            limit >> 20
        )));
    }
    let cap = cap_bytes(limit as usize);
    if !would_keep(hashfn, key, bytes, cap) {
        return Err(P254Prewarm::Skipped(format!(
            "~{bytes} B would not fit the {cap} B ctrl-persist cap {when}"
        )));
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn prewarm_p254_ctrl_impl(po2: usize) -> anyhow::Result<P254Prewarm> {
    prewarm_p254_ctrl_with(po2, &MetalDeviceHeadroom, gate_early_only(), |program| {
        let prover = super::recursion_prover("poseidon_254")?;
        prover.prewarm(program)
        // `prover` (and its HAL) is dropped here: only the host copy stays.
    })
}

/// [prewarm_p254_ctrl] with its headroom source, gate order and commit injected (tests).
///
/// The gates run before `exclusive()` is taken (cheap: no lock wait when there is clearly no
/// room) and, unless `early_only`, again once it is held. A Prove holds `exclusive()` for its
/// whole call (`risc0_zkvm`'s `exclusive_gpu`), so a prewarm that passed the first check while
/// a proof was running would otherwise wake up next to that proof's segment pool (LEADS2 S5:
/// a failed allocation at a 4 GiB budget, in the prewarm or in the next Prove). Under the lock
/// no HAL on this process allocates, so the second check holds until the commit is done, with
/// one exception: `prepare()` threads allocate the recursion staging buffers
/// (`hal/metal/staging.rs`, on by default) without the lock. Those are 24-45 MB each (at most
/// three pooled plus one per prepared proof), and each is admitted only if it leaves 512 MiB
/// (`STAGE_MARGIN_BYTES`) under the limit, so they stay inside this gate's own 512 MiB margin.
#[cfg(feature = "metal")]
pub(crate) fn prewarm_p254_ctrl_with(
    po2: usize,
    src: &dyn Headroom,
    early_only: bool,
    commit: impl FnOnce(&Arc<Program>) -> anyhow::Result<()>,
) -> anyhow::Result<P254Prewarm> {
    let hashfn = p254_suite_name();
    if !wants(hashfn) {
        return Ok(P254Prewarm::Skipped(format!(
            "{PERSIST_ENV} does not keep Poseidon254 groups"
        )));
    }
    if !risc0_zkp::hal::metal::enabled("recursion") {
        return Ok(P254Prewarm::Skipped("Metal recursion disabled".into()));
    }
    let program = super::zkr::get_zkr("identity.zkr", po2)?;
    let key = super::ctrl_cache_key(&program);
    // The bytes the kept copy will hold (coefficients, every Merkle node, the code), checked
    // against the cap before any GPU work.
    let size = 1usize << po2;
    let domain = size * risc0_zkp::INV_RATE;
    let elem = std::mem::size_of::<BabyBearElem>();
    let bytes = super::RECURSION_CODE_SIZE * size * elem
        + 2 * domain * std::mem::size_of::<risc0_zkp::core::digest::Digest>()
        + program.code.len() * elem;
    if let Err(outcome) = prewarm_gate(src, hashfn, key, bytes, "before the GPU lock") {
        return Ok(outcome);
    }
    #[cfg(test)]
    test_hooks::EARLY_GATE_PASSED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let t0 = std::time::Instant::now();
    {
        let _gpu = risc0_zkp::hal::metal::exclusive();
        if !early_only {
            if let Err(outcome) = prewarm_gate(src, hashfn, key, bytes, "under the GPU lock") {
                tracing::debug!("C16 p254 ctrl prewarm: {outcome:?} (re-checked under the lock)");
                return Ok(outcome);
            }
        }
        commit(&program)?;
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let s = state();
    Ok(match s.entries.get(&(hashfn.to_string(), key)) {
        Some(e) => {
            tracing::debug!("C16 p254 ctrl prewarm: kept {} B in {ms:.1} ms", e.bytes);
            P254Prewarm::Kept { ms, bytes: e.bytes }
        }
        None => P254Prewarm::Skipped("the group was not kept".into()),
    })
}

/// Test hooks of [prewarm_p254_ctrl_with] (W7-C1).
#[cfg(all(test, feature = "metal"))]
pub(crate) mod test_hooks {
    /// Calls that passed the gate before the lock and are about to take `exclusive()`.
    pub(crate) static EARLY_GATE_PASSED: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
}

/// Host bytes, number of groups, rebuilds and snapshots so far (tests and the bench).
#[cfg(test)]
pub(crate) fn stats() -> (usize, usize, usize, usize) {
    let s = state();
    (total_bytes(&s), s.entries.len(), s.restores, s.snapshots)
}

/// Drop every persisted group (tests).
#[cfg(test)]
pub(crate) fn clear() {
    state().entries.clear();
}
