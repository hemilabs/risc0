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

//! The Metal wrap's witness evaluator (item G13): the CUDA prover's compiled evaluators
//! ([super::witness_par], [super::witness_fast]) over the witness graph, instead of
//! [super::fast]'s generic one.
//!
//! - `par`: the parallel dataflow evaluator, ~10x faster than `fast.rs`, but it keeps an 18.1M
//!   x 32 B (580 MB) value array resident for the life of the process. Only where there is room
//!   for it (see [PAR_MIN_LIMIT_BYTES]).
//! - `par_transient` (item G15): `par` whose value array is handed back to the kernel between
//!   wraps (reusable pages, reclaimed under memory pressure; see [super::witness_par]). Its
//!   witness takes ~40-100 ms instead of `fast`'s ~165-280 ms on the critical path at an 8 GiB
//!   budget. With a kept graph it raises the process peak (G15: its program lands in the STARK
//!   phase), so it is the default below [PAR_MIN_LIMIT_BYTES] only when the graph is released
//!   after each wrap (item W7-G1, [release_enabled]), and every build is gated on headroom
//!   ([par_fits]; a 4 GiB budget builds `fast`). See notes/G15.md and notes/W7-G1.md.
//! - `fast`: the single-threaded slot evaluator: no per-node array (~30K recycled slots), so it
//!   also removes `fast.rs`' 580 MB calloc per wrap. The default below [PAR_MIN_LIMIT_BYTES]
//!   with a kept graph.
//!
//! Item W7-G1: the compiled graph is releasable ([release_after_wrap], [release_idle]); see
//! [release_enabled] for when it is dropped after each wrap, and why that needs libmalloc's
//! large cache off.
//! - `ref`: [super::fast] (the pre-G13 evaluator; also the fallback if the graph does not
//!   qualify for the compiled evaluators).
//!
//! `RISC0_G16_METAL_WITNESS=par|par_transient|fast|ref` overrides the choice. All produce the
//! same witness (`metal_tests::witness_evaluators_bit_equal`).

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, OnceLock,
};

use anyhow::{anyhow, Result};
use circom_witnesscalc::{
    field::U254,
    graph::{Nodes, NodesInterface, VecNodes},
};

use super::{witness_fast, witness_par};

/// One canonical little-endian witness element, as four little-endian u64 limbs.
pub(crate) type Limbs = [u64; 4];

/// The working-set limit (an explicit `RISC0_METAL_MAX_WORKING_SET_BYTES` budget, else
/// `recommendedMaxWorkingSetSize`) from which `par` is the default: its 580 MB array is
/// resident host memory, so only on Macs with RAM to spare (a 32 GB Mac's limit is ~21 GiB; a
/// 16-24 GB Mac's, or a 4-8 GiB budget run, is below it). Also the limit below which the graph
/// is released after each wrap by default (item W7-G1, [release_enabled]).
pub(crate) const PAR_MIN_LIMIT_BYTES: u64 = 20 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Par,
    ParTransient,
    Fast,
    Ref,
}

/// The evaluator from `RISC0_G16_METAL_WITNESS` (`v`), the working-set limit and whether the
/// graph is released after each wrap ([release_enabled]). Default: `par` from
/// [PAR_MIN_LIMIT_BYTES]; below it `par_transient` when the graph is released (item W7-G1: its
/// program and array then live only from the wrap's prestart to its witness, and each build is
/// gated on headroom, [build_kind]: `fast` at a 4 GiB budget), else `fast` (G13/G15: a kept
/// `par_transient` program raises the STARK-phase peak).
fn parse_kind(v: Option<&str>, limit: Option<u64>, release: bool) -> Kind {
    match v {
        Some("par") => Kind::Par,
        Some("par_transient") => Kind::ParTransient,
        Some("fast") => Kind::Fast,
        Some("ref") => Kind::Ref,
        _ => match limit {
            Some(l) if l >= PAR_MIN_LIMIT_BYTES => Kind::Par,
            _ if release => Kind::ParTransient,
            _ => Kind::Fast,
        },
    }
}

/// The evaluator this process uses (decided once).
pub(crate) fn kind() -> Kind {
    static KIND: OnceLock<Kind> = OnceLock::new();
    *KIND.get_or_init(|| {
        parse_kind(
            std::env::var("RISC0_G16_METAL_WITNESS").ok().as_deref(),
            risc0_groth16_sys::effective_working_set_limit(),
            release_enabled(),
        )
    })
}

/// Item W7-G1: `RISC0_G16_GRAPH_RELEASE=0|1|auto` (default `auto`): whether the compiled graph
/// is dropped after each wrap's witness (and its build deferred from proof start to the wrap's
/// prestart, so that it is not in the STARK phase's footprint).
///
/// `auto` releases only below [PAR_MIN_LIMIT_BYTES] of working-set limit (the HAL's limit: the
/// explicit budget, else `recommendedMaxWorkingSetSize`; 16-24 GB Macs and simulated 4-8 GiB
/// budgets) **and** when the process runs with libmalloc's large-allocation cache off
/// (`MallocLargeCache=0` in its environment, `large_cache_off`). With the cache on (the macOS
/// default), freeing the graph does not lower the process footprint (the freed blocks stay dirty
/// in the cache; `malloc_zone_pressure_relief` returns nothing), and every rebuild leaves another
/// ~580 MiB behind (the node array's doubling reallocations), measured in notes/W7-G1.md: there
/// a release would make the footprint grow by ~0.58 GB per proof.
fn parse_release(v: Option<&str>, limit: Option<u64>, large_cache_off: bool) -> bool {
    match v.map(str::trim) {
        Some("0" | "off" | "false") => false,
        Some("1" | "on" | "true") => true,
        _ => large_cache_off && limit.is_some_and(|l| l < PAR_MIN_LIMIT_BYTES),
    }
}

/// Whether libmalloc's large-allocation cache is off in this process: `MallocLargeCache` is
/// exactly `0`. libmalloc reads the variable once, at process start, so it must be in the
/// environment the process is spawned with (a launcher's `Command::env`); a value set later with
/// `set_var` is seen here but not by libmalloc, and would turn the release on with the cache
/// still on (the footprint then grows ~580 MiB per wrap). Other spellings (`0x0`, `false`) are
/// not trusted.
fn large_cache_off() -> bool {
    std::env::var("MallocLargeCache").is_ok_and(|v| v.trim() == "0")
}

/// Item W7-G1 (fix round 1): whether the pressure-drain hook ([release_idle]) may drop the graph:
/// only where the drop can lower the footprint. `RISC0_G16_GRAPH_RELEASE=1` forces it, `=0`
/// forbids it; otherwise (auto) only with libmalloc's large cache off (any budget: a drop then
/// returns the graph's ~0.9 GB). With the cache on, a drop frees nothing and the next rebuild
/// leaves ~580 MiB more behind (notes/W7-G1.md), so the hook is a no-op there.
fn parse_idle_release(v: Option<&str>, large_cache_off: bool) -> bool {
    match v.map(str::trim) {
        Some("0" | "off" | "false") => false,
        Some("1" | "on" | "true") => true,
        _ => large_cache_off,
    }
}

/// [parse_idle_release] for this process (decided once).
pub(crate) fn idle_release_allowed() -> bool {
    static ALLOWED: OnceLock<bool> = OnceLock::new();
    *ALLOWED.get_or_init(|| {
        parse_idle_release(
            std::env::var("RISC0_G16_GRAPH_RELEASE").ok().as_deref(),
            large_cache_off(),
        )
    })
}

/// [parse_release] for this process (decided once).
pub(crate) fn release_enabled() -> bool {
    static RELEASE: OnceLock<bool> = OnceLock::new();
    *RELEASE.get_or_init(|| {
        parse_release(
            std::env::var("RISC0_G16_GRAPH_RELEASE").ok().as_deref(),
            risc0_groth16_sys::effective_working_set_limit(),
            large_cache_off(),
        )
    })
}

/// Item W7-G1: what `par_transient`'s evaluation adds to the process beside the wrap's GPU
/// allocation: its 580 MB value array plus the per-node schedule over `fast`'s (measured in G15
/// as +0.74 GB of wrap-phase peak at 8 GiB), rounded up.
pub(crate) const PAR_TRANSIENT_BYTES: u64 = 768 << 20;

/// The headroom kept beyond [PAR_TRANSIENT_BYTES] under the working-set limit.
pub(crate) const PAR_MARGIN_BYTES: u64 = 512 << 20;

/// Item W7-G1 (G15 section 5): whether `par_transient` may evaluate: the system is not under
/// memory pressure and its transient plus the margin fit under the working-set limit beside
/// `allocated` (what the GPU holds, or will hold, while the witness is computed).
pub(crate) fn par_fits(limit: u64, allocated: u64, pressure_normal: bool) -> bool {
    pressure_normal
        && limit.saturating_sub(allocated) >= PAR_TRANSIENT_BYTES.saturating_add(PAR_MARGIN_BYTES)
}

/// Whether the kernel reports no memory pressure (`kern.memorystatus_vm_pressure_level` is
/// normal, 1; also true if it cannot be read).
pub(crate) fn pressure_normal() -> bool {
    #[cfg(target_os = "macos")]
    {
        use std::ffi::{c_char, c_int, c_void};
        extern "C" {
            fn sysctlbyname(
                name: *const c_char,
                oldp: *mut c_void,
                oldlenp: *mut usize,
                newp: *mut c_void,
                newlen: usize,
            ) -> c_int;
        }
        let mut level: c_int = 0;
        let mut len = std::mem::size_of::<c_int>();
        // SAFETY: a NUL-terminated name and an int-sized output buffer.
        let rc = unsafe {
            sysctlbyname(
                c"kern.memorystatus_vm_pressure_level".as_ptr(),
                (&mut level as *mut c_int).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        rc != 0 || level <= 1
    }
    #[cfg(not(target_os = "macos"))]
    true
}

/// `par`'s evaluation threads: `RISC0_G16_WITNESS_THREADS`, else the host's parallelism.
fn par_threads() -> usize {
    std::env::var("RISC0_G16_WITNESS_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
        })
}

enum Eval {
    Par(witness_par::ParProgram),
    Fast(witness_fast::Program),
}

pub(crate) struct Graph {
    nodes: Box<dyn NodesInterface>,
    n_signals: usize,
    input_offset: usize,
    input_len: usize,
    eval: Eval,
    /// Item W7-G1: the `fast` program compiled from these nodes when a `par_transient` graph may
    /// not evaluate at the wrap ([par_fits]), and the signal list it compiles from (kept only for
    /// `par_transient` graphs: 8 bytes per witness element).
    fallback: OnceLock<Option<witness_fast::Program>>,
    signals: Option<Vec<usize>>,
    /// Built for the process's cache ([graph_at]), not a test's throw-away evaluator: only these
    /// drops are counted ([DROPS]).
    cached: bool,
}

/// Cached graphs ([Graph::cached]) dropped so far in this process (diagnostics and tests).
/// Throw-away graphs (`witness_with`) are not counted, so that a test asserting on this counter
/// is not moved by tests compiling their own evaluators in parallel (PLAN 8.7).
static DROPS: AtomicUsize = AtomicUsize::new(0);

impl Drop for Graph {
    fn drop(&mut self) {
        if self.cached {
            DROPS.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Cached graphs dropped so far in this process.
#[cfg(test)]
pub(crate) fn drops() -> usize {
    DROPS.load(Ordering::Relaxed)
}

// Safety: the parsed graph (Nodes<U254, VecNodes>: Vecs, the field, a HashMap) and the compiled
// programs are plain data, written once and only read afterwards (as cuda.rs's CachedGraph).
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Graph {
    fn nodes(&self) -> &Nodes<U254, VecNodes> {
        self.nodes
            .as_any()
            .downcast_ref::<Nodes<U254, VecNodes>>()
            .expect("checked at build")
    }

    /// Whether the compiled program is `par`'s (resident or transient).
    pub(crate) fn is_par(&self) -> bool {
        matches!(self.eval, Eval::Par(_))
    }

    /// Whether the compiled program is `par_transient`'s (the one the headroom gate applies to).
    pub(crate) fn is_par_transient(&self) -> bool {
        self.signals.is_some()
    }

    /// Evaluate the witness of an `identity_p254` seal into `out` (exactly [Self::len] elements,
    /// canonical little-endian). `allow_par = false` evaluates a `par` graph with its `fast`
    /// fallback (compiled on first use).
    pub(crate) fn witness_into(
        &self,
        seal_bytes: &[u8],
        out: &mut [Limbs],
        allow_par: bool,
    ) -> Result<()> {
        anyhow::ensure!(
            out.len() == self.n_signals,
            "groth16 witness buffer of {} elements, expected {}",
            out.len(),
            self.n_signals
        );
        let seal = super::fast::seal_inputs(seal_bytes)?;
        debug_assert_eq!(seal.len(), self.input_len);
        // Input 0 is the constant 1; the seal occupies [input_offset, input_offset + input_len);
        // any other input is 0 (as fast.rs and the reference generator).
        let mut inputs = vec![U254::ZERO; self.input_offset + self.input_len];
        inputs[0] = U254::from(1u64);
        inputs[self.input_offset..].copy_from_slice(&seal);
        match &self.eval {
            Eval::Par(p) if allow_par => witness_par::eval(p, self.nodes(), &inputs, out),
            Eval::Par(_) => {
                let t = std::time::Instant::now();
                let fb = self
                    .fallback
                    .get_or_init(|| witness_fast::compile(self.nodes(), self.signals.as_deref()?));
                let fb = fb
                    .as_ref()
                    .ok_or_else(|| anyhow!("the witness graph does not qualify for `fast`"))?;
                let compiled_ms = t.elapsed().as_secs_f64() * 1000.0;
                witness_fast::eval(fb, self.nodes(), &inputs, out);
                tracing::info!(
                    "groth16 witness: par_transient has no headroom here; evaluated with the fast \
                     fallback (ready in {compiled_ms:.1} ms)"
                );
            }
            Eval::Fast(p) => witness_fast::eval(p, self.nodes(), &inputs, out),
        }
        Ok(())
    }

    /// [Self::witness_into] a new vector.
    pub(crate) fn witness(&self, seal_bytes: &[u8], allow_par: bool) -> Result<Vec<Limbs>> {
        let mut out = vec![[0u64; 4]; self.n_signals];
        self.witness_into(seal_bytes, &mut out, allow_par)?;
        Ok(out)
    }
}

/// The compiled graph (item W7-G1: releasable; G13/G15 kept it in a `OnceLock` for the life of
/// the process). [BUILD] serialises builds, so a caller arriving during a build waits for it
/// instead of building a second copy. A failed build is remembered ([FAILED]), as before.
static SLOT: Mutex<Option<Arc<Graph>>> = Mutex::new(None);
static BUILD: Mutex<()> = Mutex::new(());
static FAILED: OnceLock<String> = OnceLock::new();
/// Builds so far in this process (diagnostics and tests).
static BUILDS: AtomicUsize = AtomicUsize::new(0);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Item W7-G1 (fix round 2): wraps in progress ([WrapGuard]). While one runs, the graph is in use
/// even when the wrap holds no handle yet (`pre_gpu`, the key load and the GPU-lock wait come
/// before its witness), so [release_idle_now] leaves the cache alone and [release_after_wrap]
/// leaves it to the last wrap out. Changed only under the [SLOT] lock, so that a drain and a
/// wrap's start are ordered: either the drain sees the wrap, or the wrap starts after the drop.
static WRAPS: AtomicUsize = AtomicUsize::new(0);

/// Item W7-G1 (fix round 2): [reserve_for_wrap] calls not yet taken by a wrap ([WrapGuard]), and
/// when the latest was made. Under the [SLOT] lock.
static RESERVED: Mutex<(usize, Option<std::time::Instant>)> = Mutex::new((0, None));

/// How long a [reserve_for_wrap] reservation holds without a wrap taking it
/// (`RISC0_G16_GRAPH_RESERVE_MS`, default 30000; `0` turns reservations off). It only bounds a
/// reservation whose wrap never came (identity_p254 failed, or the caller proved no wrap): a
/// wrap normally takes it within identity_p254's 0.4-2 s.
fn reserve_ttl() -> std::time::Duration {
    static TTL: OnceLock<std::time::Duration> = OnceLock::new();
    *TTL.get_or_init(|| {
        std::time::Duration::from_millis(
            std::env::var("RISC0_G16_GRAPH_RESERVE_MS")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(30_000),
        )
    })
}

/// Whether a reservation is live (caller holds the [SLOT] lock).
fn reserved_locked() -> bool {
    let r = lock(&RESERVED);
    r.0 > 0 && r.1.is_some_and(|t| t.elapsed() < reserve_ttl())
}

/// Item W7-G1 (fix round 2): the wrap's prestart ([super::metal::prestart], called before
/// identity_p254) reserves the graph it builds for the wrap that follows: a pressure drain during
/// identity_p254 would only move the graph's allocation to the wrap's critical path (a
/// synchronous 0.7-0.9 s rebuild), not off the wrap's peak, which holds the graph anyway. The
/// next [WrapGuard] takes the reservation; one that no wrap takes lapses after [reserve_ttl].
pub(crate) fn reserve_for_wrap() {
    if reserve_ttl().is_zero() {
        return;
    }
    let _slot = lock(&SLOT);
    let mut r = lock(&RESERVED);
    r.0 += 1;
    r.1 = Some(std::time::Instant::now());
}

/// Tests: drop every reservation (a test that called prestart without a wrap).
#[cfg(test)]
pub(crate) fn clear_reservations() {
    let _slot = lock(&SLOT);
    *lock(&RESERVED) = (0, None);
}

/// Item W7-G1 (fix round 2): one wrap in progress, from the top of [super::metal]'s `gpu_prove`
/// until it returns. Takes one [reserve_for_wrap] reservation.
pub(crate) struct WrapGuard(());

impl WrapGuard {
    pub(crate) fn begin() -> Self {
        let _slot = lock(&SLOT);
        WRAPS.fetch_add(1, Ordering::Relaxed);
        let mut r = lock(&RESERVED);
        r.0 = r.0.saturating_sub(1);
        Self(())
    }
}

impl Drop for WrapGuard {
    fn drop(&mut self) {
        let _slot = lock(&SLOT);
        WRAPS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Wraps in progress now (tests).
#[cfg(test)]
pub(crate) fn wraps_in_progress() -> usize {
    WRAPS.load(Ordering::Relaxed)
}

/// `RISC0_G16_PAR_GATE=0` turns the `par_transient` headroom gate off ([build_kind] and the
/// wrap's check in [super::metal]; G15's behaviour).
pub(crate) fn par_gate_enabled() -> bool {
    std::env::var("RISC0_G16_PAR_GATE").as_deref() != Ok("0")
}

/// Whether a `par_transient` build fits beside the wrap: [par_fits] against the prover's planned
/// bytes (what is allocated while the witness is computed: identity_p254's pool is gone by then),
/// and the limit and planned bytes it used.
fn par_transient_fits() -> (bool, u64, u64) {
    let Some(limit) = risc0_groth16_sys::effective_working_set_limit() else {
        return (true, 0, 0);
    };
    let planned = super::component::component_dir()
        .ok()
        .and_then(|d| risc0_groth16_sys::SetupParams::new(&d).ok())
        .and_then(|s| risc0_groth16_sys::metal_planned_bytes(&s).ok())
        .unwrap_or(0);
    (
        !par_gate_enabled() || par_fits(limit, planned, pressure_normal()),
        limit,
        planned,
    )
}

/// The evaluator of the next build: [kind], except that `par_transient` builds `fast` when it
/// would not fit beside the wrap ([par_transient_fits]).
fn build_kind() -> Kind {
    let k = kind();
    if k != Kind::ParTransient {
        return k;
    }
    let (fits, limit, planned) = par_transient_fits();
    if fits {
        k
    } else {
        tracing::info!(
            "groth16 witness: par_transient does not fit beside the wrap ({} MiB planned, limit {} \
             MiB, pressure normal: {}); building the fast evaluator",
            planned >> 20,
            limit >> 20,
            pressure_normal()
        );
        Kind::Fast
    }
}

fn build(kind: Kind) -> Result<Graph> {
    let t0 = std::time::Instant::now();
    let (nodes, signals, input_mapping, [t_read, t_parse]) =
        super::fast::read_graph(super::fast::memo_enabled())?;
    let vn = nodes
        .as_any()
        .downcast_ref::<Nodes<U254, VecNodes>>()
        .ok_or_else(|| anyhow!("unexpected witness graph node type"))?;
    let (input_offset, input_len) = super::fast::seal_input(&input_mapping)?;
    let t1 = std::time::Instant::now();
    let eval = match kind {
        Kind::Par => witness_par::compile_with(vn, &signals, par_threads(), true).map(Eval::Par),
        Kind::ParTransient => {
            witness_par::compile_with(vn, &signals, par_threads(), false).map(Eval::Par)
        }
        Kind::Fast => witness_fast::compile(vn, &signals).map(Eval::Fast),
        Kind::Ref => None,
    }
    .ok_or_else(|| anyhow!("the witness graph does not qualify for the {kind:?} evaluator"))?;
    let msg = format!(
        "groth16 witness graph ({kind:?} evaluator) ready in {:.1} ms (read {:.1}, digest + \
         parse {:.1}, compile {:.1}; build {}, release after wraps {})",
        t0.elapsed().as_secs_f64() * 1000.0,
        t_read.as_secs_f64() * 1000.0,
        t_parse.as_secs_f64() * 1000.0,
        t1.elapsed().as_secs_f64() * 1000.0,
        BUILDS.load(Ordering::Relaxed) + 1,
        release_enabled(),
    );
    tracing::debug!("{msg}");
    if std::env::var_os("RISC0_VERBOSE").is_some() {
        eprintln!("[groth16-metal] {msg}");
    }
    Ok(Graph {
        n_signals: signals.len(),
        nodes,
        input_offset,
        input_len,
        eval,
        fallback: OnceLock::new(),
        signals: (kind == Kind::ParTransient).then_some(signals),
        cached: true,
    })
}

/// The compiled graph, built now if it is not cached (waiting for a build in progress), or
/// `None` when this process uses `ref` (or the graph did not qualify: then [super::fast] is used,
/// as before G13). `at_wrap`: the caller is a wrap; a graph that is not ready is logged with the
/// time it cost (item W7-G1: the wrap never depends on a background build having run).
fn graph_at(at_wrap: bool) -> Option<Arc<Graph>> {
    if kind() == Kind::Ref {
        return None;
    }
    let failed = |e: &str| {
        static WARNED: OnceLock<()> = OnceLock::new();
        WARNED.get_or_init(|| {
            tracing::warn!("{e}; using the reference witness evaluator");
        });
        None
    };
    if let Some(e) = FAILED.get() {
        return failed(e);
    }
    if let Some(g) = lock(&SLOT).clone() {
        return Some(g);
    }
    let t = std::time::Instant::now();
    let _build = lock(&BUILD);
    if let Some(g) = lock(&SLOT).clone() {
        if at_wrap {
            let msg = format!(
                "groth16 witness graph not ready at the wrap: waited {:.1} ms for the build in \
                 progress",
                t.elapsed().as_secs_f64() * 1000.0
            );
            tracing::info!("{msg}");
            if std::env::var_os("RISC0_VERBOSE").is_some() {
                eprintln!("[groth16-metal] {msg}");
            }
        }
        return Some(g);
    }
    if let Some(e) = FAILED.get() {
        return failed(e);
    }
    match build(build_kind()) {
        Ok(g) => {
            BUILDS.fetch_add(1, Ordering::Relaxed);
            let g = Arc::new(g);
            *lock(&SLOT) = Some(g.clone());
            if at_wrap {
                let msg = format!(
                    "groth16 witness graph not ready at the wrap: built it synchronously in {:.1} \
                     ms",
                    t.elapsed().as_secs_f64() * 1000.0
                );
                tracing::info!("{msg}");
                if std::env::var_os("RISC0_VERBOSE").is_some() {
                    eprintln!("[groth16-metal] {msg}");
                }
            }
            Some(g)
        }
        Err(e) if BUILDS.load(Ordering::Relaxed) == 0 => {
            let e = FAILED.get_or_init(|| format!("{e:#}"));
            failed(e)
        }
        Err(e) => {
            // Item W7-G1 (fix round 1): a rebuild after a release failed (a transient I/O error,
            // or the component replaced mid-process): this wrap uses the reference evaluator, and
            // the next one tries again, instead of latching the failure for the process.
            tracing::warn!(
                "groth16 witness graph rebuild failed ({e:#}); this wrap uses the reference \
                 witness evaluator"
            );
            None
        }
    }
}

fn graph() -> Option<Arc<Graph>> {
    graph_at(false)
}

/// The graph for a wrap (built now if needed; see [graph_at]).
pub(crate) fn wrap_graph() -> Option<Arc<Graph>> {
    graph_at(true)
}

/// Read and compile the witness graph (call early, on a background thread). Item W7-G1: with
/// [release_enabled], a no-op: the zkvm and the worker call this at proof start, which would put
/// the graph back into the STARK phase; the wrap's [preload_for_wrap] builds it instead.
pub(crate) fn preload() -> Result<()> {
    if release_enabled() {
        tracing::debug!("groth16 witness graph build deferred to the wrap (graph release on)");
        return Ok(());
    }
    preload_for_wrap()
}

/// Build the witness graph now (the wrap's prestart: beside identity_p254).
pub(crate) fn preload_for_wrap() -> Result<()> {
    match graph() {
        Some(_) => Ok(()),
        None => super::fast::preload(),
    }
}

/// Item W7-G1: drop the cached graph (and [super::fast]'s program). The memory is freed when the
/// last user (a wrap computing its witness) lets go. Returns whether anything was cached.
#[cfg(test)]
pub(crate) fn release() -> bool {
    let g = lock(&SLOT).take();
    let had = g.is_some();
    drop(g);
    super::fast::release() || had
}

/// [release] after a wrap's witness, if [release_enabled]: `held` (the wrap's handle) and the
/// cached graph are dropped on a helper thread, off the wrap's critical path. Item W7-G1 (fix
/// round 1): with `held`, the cache is emptied only if it still holds that same graph, so a graph
/// that another wrap's prestart built meanwhile (overlapped or pipelined wraps) is kept. Fix round
/// 2: called inside the wrap's [WrapGuard]; the cache is emptied only by the last wrap out, and
/// not while a prestart reservation is live ([release_slot]), so neither an overlapped wrap
/// still before its witness nor the next job's prestart graph is discarded (with or without
/// `held`: a wrap that failed before its witness passes none).
pub(crate) fn release_after_wrap(held: Option<Arc<Graph>>) {
    if !release_enabled() {
        return;
    }
    let g = release_slot(held.as_ref(), 1);
    let p = super::fast::release();
    if g.is_none() && held.is_none() && !p {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("g16-graph-release".into())
        .spawn(move || {
            let t = std::time::Instant::now();
            drop(held);
            drop(g);
            tracing::debug!(
                "groth16 witness graph released after the wrap in {:.1} ms",
                t.elapsed().as_secs_f64() * 1000.0
            );
        });
    // (If no thread can be started, the closure, and with it both handles, is dropped here.)
    drop(spawned);
}

/// Empty the cache: with `held`, only if it holds that graph ([release_after_wrap]); never while
/// more than `own` wraps (the caller's) are in progress or a prestart reservation is live.
pub(crate) fn release_slot(held: Option<&Arc<Graph>>, own: usize) -> Option<Arc<Graph>> {
    let mut slot = lock(&SLOT);
    if WRAPS.load(Ordering::Relaxed) > own || reserved_locked() {
        return None;
    }
    match (held, slot.as_ref()) {
        (Some(h), Some(g)) if !Arc::ptr_eq(h, g) => None,
        _ => slot.take(),
    }
}

/// Item W7-G1: the hook for a memory-pressure drain (`risc0_zkp::hal::metal::
/// register_idle_releaser`, item W7-A1): drop the cached graph if no wrap is using it and no
/// build is running. Never waits (the build lock is only tried). Returns the bytes freed (the
/// process footprint's drop, 0 if nothing was idle).
///
/// Fix round 2: "in use" is a wrap in progress ([WrapGuard], from the top of the GPU wrap, so
/// also its `pre_gpu`, key load and GPU-lock wait, when it holds no handle yet) or a live
/// prestart reservation ([reserve_for_wrap]), not only a handle: a drain there would make the
/// wrap rebuild the graph synchronously.
///
/// Fix round 1: a no-op (returns 0, drops nothing) unless [idle_release_allowed]: with
/// libmalloc's large cache on, a drop frees nothing and the next rebuild leaves ~580 MiB more
/// behind, so a drain followed by a wrap would grow the footprint.
pub(crate) fn release_idle() -> u64 {
    if !idle_release_allowed() {
        return 0;
    }
    release_idle_now()
}

/// [release_idle] without [idle_release_allowed] (tests).
pub(crate) fn release_idle_now() -> u64 {
    let Ok(_build) = BUILD.try_lock() else {
        return 0;
    };
    let g = {
        let mut slot = lock(&SLOT);
        if WRAPS.load(Ordering::Relaxed) > 0 || reserved_locked() {
            return 0;
        }
        match slot.as_ref() {
            Some(g) if Arc::strong_count(g) == 1 => slot.take(),
            _ => None,
        }
    };
    let fast_idle = super::fast::idle_cached();
    if g.is_none() && !fast_idle {
        return 0;
    }
    let before = risc0_groth16_sys::metal::filemap::process_memory().phys_footprint;
    drop(g);
    if fast_idle {
        super::fast::release();
    }
    let after = risc0_groth16_sys::metal::filemap::process_memory().phys_footprint;
    before.saturating_sub(after)
}

/// Builds of the graph so far in this process.
#[cfg(test)]
pub(crate) fn builds() -> usize {
    BUILDS.load(Ordering::Relaxed)
}

/// Whether a compiled graph is cached now.
#[cfg(test)]
pub(crate) fn is_cached() -> bool {
    lock(&SLOT).is_some()
}

/// Whether this wrap evaluates with `par_transient` (decided without waiting for the graph
/// build, so that the build still overlaps `pre_gpu`'s start in [super::metal]): the cached
/// graph's program if there is one, else whether the next build would be `par_transient`
/// ([build_kind]; item W7-G1: a gated-off build is `fast` and takes `fast`'s overlapped path).
pub(crate) fn is_par_transient() -> bool {
    if kind() != Kind::ParTransient {
        return false;
    }
    match lock(&SLOT).clone() {
        Some(g) => g.is_par(),
        None => par_transient_fits().0,
    }
}

/// The wrap's graph if it compiled the `par` evaluator (resident or transient; built now if it
/// is not ready, which the wrap logs), for the wrap to hold from its path decision to its witness
/// (item W7-G1 fix round 2; it was a `bool`, `is_par`, and the witness took the graph again).
/// Decided on [kind] first, so that a `fast` process never waits here for a build.
pub(crate) fn par_graph() -> Option<Arc<Graph>> {
    if !matches!(kind(), Kind::Par | Kind::ParTransient) {
        return None;
    }
    wrap_graph().filter(|g| g.is_par())
}

/// Tests: the witness of `seal_bytes` from the process's cached graph (built if needed).
#[cfg(test)]
pub(crate) fn witness(seal_bytes: &[u8]) -> Result<Vec<Limbs>> {
    let g = graph().ok_or_else(|| anyhow!("no compiled witness evaluator"))?;
    g.witness(seal_bytes, true)
}

/// Tests: the witness of `seal` from a freshly compiled `kind` evaluator (not the process's).
#[cfg(test)]
pub(crate) fn witness_with(kind: Kind, threads: usize, seal_bytes: &[u8]) -> Result<Vec<Limbs>> {
    let (nodes, signals, input_mapping, _) = super::fast::read_graph(true)?;
    let vn = nodes
        .as_any()
        .downcast_ref::<Nodes<U254, VecNodes>>()
        .ok_or_else(|| anyhow!("unexpected witness graph node type"))?;
    let (input_offset, input_len) = super::fast::seal_input(&input_mapping)?;
    let eval = match kind {
        Kind::Par => witness_par::compile_with(vn, &signals, threads, true).map(Eval::Par),
        Kind::ParTransient => {
            witness_par::compile_with(vn, &signals, threads, false).map(Eval::Par)
        }
        Kind::Fast => witness_fast::compile(vn, &signals).map(Eval::Fast),
        Kind::Ref => None,
    }
    .ok_or_else(|| anyhow!("{kind:?} does not compile"))?;
    let g = Graph {
        n_signals: signals.len(),
        nodes,
        input_offset,
        input_len,
        eval,
        fallback: OnceLock::new(),
        signals: (kind == Kind::ParTransient).then(|| signals.clone()),
        cached: false,
    };
    let seal = super::fast::seal_inputs(seal_bytes)?;
    let mut inputs = vec![U254::ZERO; g.input_offset + g.input_len];
    inputs[0] = U254::from(1u64);
    inputs[g.input_offset..].copy_from_slice(&seal);
    let mut out = vec![[0u64; 4]; g.n_signals];
    match &g.eval {
        Eval::Par(p) => {
            assert_eq!(witness_par::is_resident(p), kind == Kind::Par);
            witness_par::eval(p, g.nodes(), &inputs, &mut out);
            if kind == Kind::ParTransient {
                // Again over the handed-back array (its pages reclaimed or not).
                let mut again = vec![[0u64; 4]; g.n_signals];
                witness_par::eval(p, g.nodes(), &inputs, &mut again);
                assert!(again == out, "par_transient: second witness differs");
                // Item W7-G1: the `fast` fallback of a graph without headroom.
                let fb = g.witness(seal_bytes, false)?;
                assert!(
                    fb == out,
                    "par_transient: the fast fallback's witness differs"
                );
            }
        }
        Eval::Fast(p) => witness_fast::eval(p, g.nodes(), &inputs, &mut out),
    }
    Ok(out)
}

/// Tests: footprint and evaluation times of a freshly compiled `kind` program (not `ref`).
#[cfg(test)]
pub(crate) fn footprint_probe(kind: Kind, seal_bytes: &[u8], evals: usize) -> Result<()> {
    use risc0_groth16_sys::metal::filemap::process_memory;
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    let (nodes, signals, input_mapping, _) = super::fast::read_graph(true)?;
    let vn = nodes
        .as_any()
        .downcast_ref::<Nodes<U254, VecNodes>>()
        .ok_or_else(|| anyhow!("unexpected witness graph node type"))?;
    let (input_offset, input_len) = super::fast::seal_input(&input_mapping)?;
    let f0 = process_memory().phys_footprint;
    let t = std::time::Instant::now();
    let eval = match kind {
        Kind::Par => witness_par::compile_with(vn, &signals, par_threads(), true).map(Eval::Par),
        Kind::ParTransient => {
            witness_par::compile_with(vn, &signals, par_threads(), false).map(Eval::Par)
        }
        Kind::Fast => witness_fast::compile(vn, &signals).map(Eval::Fast),
        Kind::Ref => None,
    }
    .ok_or_else(|| anyhow!("{kind:?} does not compile"))?;
    let f1 = process_memory().phys_footprint;
    eprintln!(
        "G15PAR {kind:?}: compile {:.1} ms, footprint +{:.1} MiB",
        t.elapsed().as_secs_f64() * 1e3,
        mib(f1.saturating_sub(f0))
    );
    let seal = super::fast::seal_inputs(seal_bytes)?;
    let mut inputs = vec![U254::ZERO; input_offset + input_len];
    inputs[0] = U254::from(1u64);
    inputs[input_offset..].copy_from_slice(&seal);
    let mut out = vec![[0u64; 4]; signals.len()];
    for i in 0..evals {
        let t = std::time::Instant::now();
        match &eval {
            Eval::Par(p) => witness_par::eval(p, vn, &inputs, &mut out),
            Eval::Fast(p) => witness_fast::eval(p, vn, &inputs, &mut out),
        }
        let f = process_memory().phys_footprint;
        eprintln!(
            "G15PAR {kind:?}: eval {i} {:.1} ms, footprint after +{:.1} MiB",
            t.elapsed().as_secs_f64() * 1e3,
            mib(f.saturating_sub(f0))
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Item W7-G1: `RISC0_G16_GRAPH_RELEASE`.
    #[test]
    fn release_selection() {
        const GIB: u64 = 1 << 30;
        // auto: below 20 GiB, and only with libmalloc's large cache off.
        assert!(parse_release(None, Some(8 * GIB), true));
        assert!(parse_release(Some("auto"), Some(4 * GIB), true));
        assert!(!parse_release(None, Some(8 * GIB), false));
        assert!(!parse_release(Some("auto"), Some(4 * GIB), false));
        assert!(!parse_release(None, Some(PAR_MIN_LIMIT_BYTES), true));
        assert!(!parse_release(None, Some(28 * GIB), true));
        assert!(!parse_release(None, None, true));
        // Explicit.
        assert!(!parse_release(Some("0"), Some(4 * GIB), true));
        assert!(parse_release(Some("1"), Some(28 * GIB), false));
        assert!(parse_release(Some(" on "), None, false));
        assert!(!parse_release(Some("off"), Some(GIB), true));
    }

    /// Item W7-G1 (fix round 1): when the pressure-drain hook may drop the graph.
    #[test]
    fn idle_release_selection() {
        // auto: only with libmalloc's large cache off (a drop with it on frees nothing).
        assert!(!parse_idle_release(None, false));
        assert!(!parse_idle_release(Some("auto"), false));
        assert!(parse_idle_release(None, true));
        // Explicit.
        assert!(parse_idle_release(Some("1"), false));
        assert!(!parse_idle_release(Some(" 0 "), true));
        assert!(!parse_idle_release(Some("off"), true));
    }

    /// Item W7-G1: the `par_transient` headroom gate.
    #[test]
    fn par_gate() {
        const MIB: u64 = 1 << 20;
        let key = 3928 * MIB;
        // 8 GiB, key and working set allocated (the in-place witness): fits.
        assert!(par_fits(8192 * MIB, key, true));
        // 8 GiB with identity_p254's pool still beside the key (5377 MiB): fits.
        assert!(par_fits(8192 * MIB, 5377 * MIB, true));
        // 4 GiB: never beside the key.
        assert!(!par_fits(4096 * MIB, key, true));
        // Exactly the transient plus the margin: fits; one byte less: not.
        let need = PAR_TRANSIENT_BYTES + PAR_MARGIN_BYTES;
        assert!(par_fits(key + need, key, true));
        assert!(!par_fits(key + need - 1, key, true));
        // Memory pressure: never.
        assert!(!par_fits(64 << 30, 0, false));
        // Over-allocated: no underflow.
        assert!(!par_fits(MIB, u64::MAX, true));
    }

    #[test]
    fn kind_selection() {
        for release in [false, true] {
            assert_eq!(parse_kind(Some("par"), Some(1), release), Kind::Par);
            assert_eq!(
                parse_kind(Some("fast"), Some(64 << 30), release),
                Kind::Fast
            );
            assert_eq!(parse_kind(Some("ref"), None, release), Kind::Ref);
            assert_eq!(
                parse_kind(None, Some(PAR_MIN_LIMIT_BYTES), release),
                Kind::Par
            );
            assert_eq!(
                parse_kind(Some("par_transient"), Some(64 << 30), release),
                Kind::ParTransient
            );
        }
        // Item G15: below the limit, a kept graph stays `fast` (`par_transient` opt-in).
        assert_eq!(parse_kind(None, Some(8 << 30), false), Kind::Fast);
        assert_eq!(parse_kind(Some("auto"), Some(4 << 30), false), Kind::Fast);
        assert_eq!(parse_kind(None, None, false), Kind::Fast);
        // Item W7-G1: a released graph is `par_transient` below the limit (gated per build).
        assert_eq!(parse_kind(None, Some(8 << 30), true), Kind::ParTransient);
        assert_eq!(
            parse_kind(Some("auto"), Some(4 << 30), true),
            Kind::ParTransient
        );
        assert_eq!(parse_kind(None, None, true), Kind::ParTransient);
    }
}
