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

//! Per-phase wall-time profile of the recursion Metal HAL (C11, measurement aid).
//!
//! `RISC0_METAL_REC_PROFILE=1` makes the circuit HAL sync at every phase boundary it sees and
//! record the wall time of the GPU (and host) work since the previous boundary:
//!
//! | label | what it covers |
//! |---|---|
//! | `witgen.*` | the witgen sub-steps (fill, host copies/chains, exec, sort+scan, inject+verify) |
//! | `pre_accum` | after the witgen: data noise + zeroize, ctrl/data commit (NTT + Merkle), mix |
//! | `accum` | the accum batch |
//! | `pre_eval` | accum noise/zeroize, accum commit, check-buffer setup |
//! | `eval_check` | the eval_check dispatches |
//!
//! The rest of a proof (FRI, queries) is the proof's wall time minus these. The syncs serialize
//! what the lazy batch model would overlap, so profile runs are for attribution, not for A/B.
//! Off by default: nothing changes without the variable.

use std::{cell::RefCell, sync::OnceLock, time::Instant};

/// The knob (read once per process).
pub(crate) const PROFILE_ENV: &str = "RISC0_METAL_REC_PROFILE";

/// Whether [PROFILE_ENV] is on.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var(PROFILE_ENV).is_ok_and(|v| !v.is_empty() && v != "0"))
}

#[derive(Default)]
struct State {
    last: Option<Instant>,
    entries: Vec<(&'static str, f64)>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// Start a new proof's profile (at the witgen's entry, after a sync).
pub(crate) fn start() {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.entries.clear();
        s.last = Some(Instant::now());
    });
}

/// Record the time since the previous mark as `label` (no-op before [start]).
pub(crate) fn mark(label: &'static str) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if let Some(last) = s.last {
            let now = Instant::now();
            s.entries.push((label, (now - last).as_secs_f64() * 1e3));
            s.last = Some(now);
        }
    });
}

/// Record an already measured duration (ms) without moving the mark (witgen sub-steps).
pub(crate) fn record(label: &'static str, ms: f64) {
    if enabled() {
        STATE.with(|s| s.borrow_mut().entries.push((label, ms)));
    }
}

/// The calling thread's entries since the last [start].
pub(crate) fn entries() -> Vec<(&'static str, f64)> {
    STATE.with(|s| s.borrow().entries.clone())
}

/// Print the calling thread's entries as one line (at the end of eval_check).
pub(crate) fn report() {
    let e = entries();
    let line: Vec<String> = e.iter().map(|(l, ms)| format!("{l}={ms:.2}")).collect();
    eprintln!("[rec_metal_profile] {}", line.join(" "));
}
