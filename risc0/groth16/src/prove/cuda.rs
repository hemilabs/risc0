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

use std::{collections::HashMap, path::Path, sync::OnceLock};

use anyhow::{Context as _, Result};
use circom_witnesscalc::deserialize_inputs2;
use circom_witnesscalc::field::U254;
use circom_witnesscalc::graph::{evaluate, Node, Nodes, NodesInterface, NodesStorage, Operation, VecNodes};
use circom_witnesscalc::storage::proto_deserializer::deserialize_witnesscalc_graph_from_bytes;
use circom_witnesscalc::InputSignalsInfo;
use risc0_groth16_sys::{ProverParams, SetupParams, WitnessParams};
use rzup::{Component, Rzup, Version};
use tempfile::tempdir;

use crate::{ProofJson, Seal};

use super::seal_to_json::to_json;

/// Cached rzup root directory path.
static ROOT_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();

fn get_root_dir() -> Result<&'static Path> {
    let dir = ROOT_DIR.get_or_init(|| {
        // `RISC0_GROTH16_DIR` names the component's directory outright, bypassing rzup:
        // rzup does not run on Windows (`Platform::detect` knows only Linux and macOS),
        // so there the files are installed by hand and located through this.
        if let Some(dir) = std::env::var_os("RISC0_GROTH16_DIR") {
            return dir.into();
        }
        let version = Version::new(0, 1, 0);
        Rzup::new()
            .and_then(|r| r.get_version_dir(&Component::Risc0Groth16, &version))
            .ok()
            .or_else(|| rzup_layout_dir(&version))
            .expect(
                "Missing required `risc0-groth16` rzup component. \
                To install it, ensure that your `rzup` version is >= 0.5.0, \
                and then run `rzup install risc0-groth16`. \
                Where rzup does not run (Windows), copy the component's files to \
                ~/.risc0/extensions/v0.1.0-risc0-groth16, or set RISC0_GROTH16_DIR \
                to a directory holding them.",
            )
    });
    Ok(dir.as_path())
}

/// Where rzup installs the component, when it is there: `$RISC0_HOME` (else
/// `~/.risc0`) `/extensions/v<version>-risc0-groth16`. For hosts where `Rzup::new`
/// fails outright, such as Windows, which its `Platform::detect` does not know, so
/// that a copy installed by hand in the same place is found the same way.
fn rzup_layout_dir(version: &Version) -> Option<std::path::PathBuf> {
    let risc0_home = std::env::var_os("RISC0_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(|home| std::path::PathBuf::from(home).join(".risc0"))
        })?;
    let dir = risc0_home
        .join("extensions")
        .join(format!("v{version}-risc0-groth16"));
    dir.is_dir().then_some(dir)
}

/// Cached deserialized circom graph to avoid re-parsing ~241MB on each call (~554ms).
struct CachedGraph {
    nodes: Box<dyn NodesInterface>,
    signals: Vec<usize>,
    input_mapping: InputSignalsInfo,
    /// The graph compiled for `witness_fast` (None if it doesn't qualify).
    fast: Option<super::witness_fast::Program>,
    /// C2: band-parallel evaluator (default; RISC0_G16_PAR_WITNESS=0 disables).
    par: Option<super::witness_par::ParProgram>,
}

// Safety: The underlying Nodes<U254, VecNodes> contains Vec<Node>, Field<U254>, HashMap —
// all Send+Sync. The graph is written once (OnceLock init) and read immutably after.
unsafe impl Send for CachedGraph {}
unsafe impl Sync for CachedGraph {}

static CACHED_GRAPH: OnceLock<CachedGraph> = OnceLock::new();

/// C2-R2: witness indices whose signal is `x & 1` (a Band node with the constant 1), i.e.
/// always 0 or 1: the bit set S of the split MSMs. Derived from the graph itself (no index
/// file); the prover still checks every S scalar at prove time and falls back to the full MSMs
/// if one is not 0/1, so S only affects speed, never the proof.
static BIT_SIGNALS: OnceLock<Vec<u32>> = OnceLock::new();

fn bit_signals(nodes: &Nodes<U254, VecNodes>, signals: &[usize]) -> Vec<u32> {
    let one = U254::from(1u64);
    let mut out = Vec::new();
    for (wi, &si) in signals.iter().enumerate() {
        if let Some(Node::Op(Operation::Band, _a, b)) = nodes.nodes.get(si) {
            if let Some(Node::Constant(ci)) = nodes.nodes.get(b) {
                if nodes.constants[ci] == one {
                    out.push(wi as u32);
                }
            }
        }
    }
    out
}

fn get_cached_graph(graph_path: &Path) -> &'static CachedGraph {
    CACHED_GRAPH.get_or_init(|| {
        let graph_bytes = std::fs::read(graph_path).expect("failed to read circom graph file");
        let (nodes, signals, input_mapping) =
            deserialize_witnesscalc_graph_from_bytes(&graph_bytes)
                .expect("failed to deserialize circom graph");
        if let Some(vn) = nodes.as_any().downcast_ref::<Nodes<U254, VecNodes>>() {
            let _ = BIT_SIGNALS.set(bit_signals(vn, &signals));
        } else {
            let _ = BIT_SIGNALS.set(Vec::new());
        }
        let fast = nodes
            .as_any()
            .downcast_ref::<Nodes<U254, VecNodes>>()
            .and_then(|nodes| super::witness_fast::compile(nodes, &signals));
        if fast.is_none() {
            tracing::warn!("groth16: the fast witness evaluator does not support this graph");
        }
        let par = if std::env::var("RISC0_G16_PAR_WITNESS").as_deref() != Ok("0") {
            let t = std::time::Instant::now();
            let par = nodes
                .as_any()
                .downcast_ref::<Nodes<U254, VecNodes>>()
                .and_then(|nodes| super::witness_par::compile(nodes, &signals));
            if std::env::var_os("RISC0_G16_TIMING").is_some() {
                eprintln!(
                    "[g16_timing] witness_par compile {:.1}ms ok={}",
                    t.elapsed().as_secs_f64() * 1e3,
                    par.is_some()
                );
            }
            par
        } else {
            None
        };
        CachedGraph {
            nodes,
            signals,
            input_mapping,
            fast,
            par,
        }
    })
}

/// Preload and cache the deserialized circom graph for witness calculation.
/// Call early (e.g. on a background thread) to overlap the ~554ms graph
/// deserialization with other work (e.g. STARK proving).
pub(crate) fn preload_graph() -> Result<()> {
    let root_dir = get_root_dir()?;
    let witness_params = WitnessParams::new(root_dir);
    let _ = get_cached_graph(&witness_params.graph_path);
    Ok(())
}

/// Preload SRS and prover into C++ static cache.
/// Must be called when GPU is idle (after STARK proving, before shrink_wrap).
pub(crate) fn preload_srs() -> Result<()> {
    let root_dir = get_root_dir()?;
    let setup_params =
        SetupParams::new(root_dir).context("failed to create groth16 setup params for preload")?;
    risc0_groth16_sys::preload(&setup_params).context("failed to preload groth16 SRS")?;
    Ok(())
}

/// C2-R2: start loading the proving key on a background thread (pinned staging, own streams),
/// then pre-allocate the pinned witness buffer, so that neither is on the shrink_wrap path.
/// Meant to be called when a Groth16 proof is about to be produced (e.g. at prover construction
/// or at the start of the STARK). Skipped on cards with < RISC0_G16_PRESTART_MIN_GB (28) GB.
pub(crate) fn prestart_key() {
    static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if STARTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    // The bit set S (split MSMs) comes from the parsed graph: make sure a parse is running.
    std::thread::spawn(|| {
        let _ = preload_graph();
    });
    std::thread::spawn(|| {
        // Let the first STARK segment's cold allocations and uploads go first: wait for
        // `prestart_key_go` (segment 0 queued), at most 2 s; RISC0_G16_PRESTART_DELAY_MS=N
        // waits a fixed N ms instead.
        if let Some(ms) = std::env::var("RISC0_G16_PRESTART_DELAY_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
        {
            std::thread::sleep(std::time::Duration::from_millis(ms));
        } else {
            let (m, cv) = &PRESTART_GO;
            let g = m.lock().unwrap();
            let _ = cv
                .wait_timeout_while(g, std::time::Duration::from_secs(2), |go| !*go)
                .unwrap();
        }
        let t = std::time::Instant::now();
        // S must be set before the prover is built (its constructor permutes the SRS).
        while BIT_SIGNALS.get().is_none() && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        if let Some(bits) = BIT_SIGNALS.get() {
            risc0_groth16_sys::set_bit_signals(bits);
        }
        let t_bits = t.elapsed();
        let timing = std::env::var_os("RISC0_G16_TIMING").is_some();
        let min_gb: usize = std::env::var("RISC0_G16_PRESTART_MIN_GB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(28);
        let r = (|| -> Result<Option<usize>> {
            let root_dir = get_root_dir()?;
            let setup = SetupParams::new(root_dir)?;
            let len = risc0_groth16_sys::prestart(&setup, min_gb << 30)?;
            if let Some(len) = len {
                if std::env::var_os("RISC0_G16_NO_PINPRE").is_none() {
                    drop(risc0_groth16_sys::host_buffer(len * 32)?);
                }
            }
            Ok(len)
        })();
        if timing {
            eprintln!(
                "[g16_timing] prestart_key done in {:.1}ms (bit set wait {:.1}ms): {:?}",
                t.elapsed().as_secs_f64() * 1e3,
                t_bits.as_secs_f64() * 1e3,
                r.as_ref().map_err(|e| format!("{e:#}"))
            );
        }
        if let Err(e) = r {
            tracing::warn!("groth16 key prestart failed (non-fatal): {e:#}");
        }
    });
}

static PRESTART_GO: (std::sync::Mutex<bool>, std::sync::Condvar) =
    (std::sync::Mutex::new(false), std::sync::Condvar::new());

/// Release the `prestart_key` loader (idempotent).
pub(crate) fn prestart_key_go() {
    let (m, cv) = &PRESTART_GO;
    let mut g = m.lock().unwrap();
    if !*g {
        *g = true;
        cv.notify_all();
    }
}

/// Start `preload_srs` on its own thread (C2 prototype; `shrink_wrap`'s own preload then
/// waits for it on the C++ cache mutex).
pub(crate) fn prestart() {
    std::thread::spawn(|| {
        let t = std::time::Instant::now();
        let r = preload_srs();
        if std::env::var_os("RISC0_G16_TIMING").is_some() {
            eprintln!(
                "[g16_timing] prestart preload done in {:.1}ms ok={}",
                t.elapsed().as_secs_f64() * 1e3,
                r.is_ok()
            );
        }
    });
}

/// Drop the prover and SRS that `preload_srs` and `shrink_wrap` cached on the GPU. See
/// `prove::release_srs`.
pub(crate) fn release_srs() -> Result<()> {
    risc0_groth16_sys::release().context("failed to release the cached groth16 SRS")
}

pub(crate) fn shrink_wrap(seal_bytes: &[u8]) -> Result<Seal> {
    tracing::info!("shrink_wrap: {} seal bytes", seal_bytes.len());

    let root_dir = get_root_dir()?;
    let tmp_dir = tempdir().context("failed to create temporary directory")?;
    let work_dir = std::env::var("RISC0_WORK_DIR");
    let work_dir = work_dir.as_ref().map(Path::new).unwrap_or(tmp_dir.path());
    let setup_params =
        SetupParams::new(root_dir).context("failed to create groth16 work directories")?;

    let t_all = std::time::Instant::now();
    let timing = std::env::var_os("RISC0_G16_TIMING").is_some();
    let inputs = to_json(seal_bytes)?;
    let t_json = t_all.elapsed();

    // Start SRS preload on background thread while witness calc runs on CPU.
    // SRS loading (disk I/O + GPU upload) overlaps with witness computation.
    let srs_preload = {
        let setup_clone = SetupParams::new(root_dir)
            .context("failed to create groth16 setup params for preload")?;
        std::thread::spawn(move || {
            let _ = risc0_groth16_sys::preload(&setup_clone);
        })
    };

    let witness_params = WitnessParams::new(root_dir);
    let graph = get_cached_graph(&witness_params.graph_path);
    let t_w0 = std::time::Instant::now();
    let witness = calc_witness(graph, &inputs).context("failed to calculate groth16 witness")?;
    let t_wit = t_w0.elapsed();

    // Wait for SRS preload to complete before taking the GPU lock.
    let t_j0 = std::time::Instant::now();
    let _ = srs_preload.join();
    let t_join = t_j0.elapsed();
    let t_p0 = std::time::Instant::now();
    let _timing_guard = TimingPrint(timing, t_all, t_json, t_wit, t_join, t_p0);

    {
        let _lock = risc0_zkp::hal::cuda::singleton().lock();

        let prover_params = ProverParams::new(work_dir, witness.as_ptr())
            .context("failed to create groth16 prover parameters")?;
        risc0_groth16_sys::prove(&prover_params, &setup_params)
            .context("failed to run groth16 prove operation")?;

        let contents = std::fs::read_to_string(prover_params.proof_path.as_path())
            .context("failed to read groth16 prove output file")?;
        if std::env::var_os("RISC0_G16_DUMP").is_some() {
            eprintln!("[g16_proof] {}", contents.replace('\n', " "));
        }
        let proof_json: ProofJson = serde_json::from_str(&contents)
            .context("failed to decode groth16 prove output file JSON")?;
        proof_json.try_into()
    }
}

struct TimingPrint(
    bool,
    std::time::Instant,
    std::time::Duration,
    std::time::Duration,
    std::time::Duration,
    std::time::Instant,
);
impl Drop for TimingPrint {
    fn drop(&mut self) {
        if self.0 {
            eprintln!(
                "[g16_timing] to_json={:.1}ms witness={:.1}ms srs_join_wait={:.1}ms prove+io={:.1}ms total={:.1}ms",
                self.2.as_secs_f64() * 1e3,
                self.3.as_secs_f64() * 1e3,
                self.4.as_secs_f64() * 1e3,
                self.5.elapsed().as_secs_f64() * 1e3,
                self.1.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}

enum CalcWitness {
    Vec(Vec<wtns_file::FieldElement<32>>),
    /// Canonical little-endian values in the pinned buffer, held until the prove has read it.
    Pinned(risc0_groth16_sys::HostBuffer),
}

impl CalcWitness {
    fn as_ptr(&self) -> *const u8 {
        match self {
            CalcWitness::Vec(w) => w.as_ptr() as *const u8,
            CalcWitness::Pinned(buf) => buf.as_ptr(),
        }
    }
}

fn calc_witness(graph: &CachedGraph, inputs: &str) -> Result<CalcWitness> {
    let nodes = graph
        .nodes
        .as_any()
        .downcast_ref::<Nodes<U254, VecNodes>>()
        .ok_or_else(|| anyhow::anyhow!("unexpected graph node type"))?;

    // Parse inputs into field elements.
    let input_list: HashMap<String, Vec<U254>> = deserialize_inputs2(inputs.as_bytes(), &nodes.ff)
        .map_err(|e| anyhow::anyhow!("failed to deserialize inputs: {e}"))?;

    // Map parsed inputs to the input index array.
    let mut max_idx: usize = 0;
    for (offset, len) in graph.input_mapping.values() {
        let idx = offset + len;
        if idx > max_idx {
            max_idx = idx;
        }
    }
    let mut input_vec = vec![U254::ZERO; max_idx + 1];
    input_vec[0] = U254::from(1u64);
    for (key, value) in &input_list {
        let (offset, len) = graph.input_mapping[key];
        if len != value.len() {
            anyhow::bail!("Invalid input length for {}", key);
        }
        for (i, v) in value.iter().enumerate() {
            input_vec[offset + i] = *v;
        }
    }

    if let Some(program) = graph.par.as_ref() {
        let mut buf = risc0_groth16_sys::host_buffer(graph.signals.len() * 32)
            .context("failed to allocate the pinned witness buffer")?;
        super::witness_par::eval(program, nodes, &input_vec, buf.as_limbs_mut());
        if std::env::var_os("RISC0_G16_PAR_CHECK").is_some() {
            if let Some(fast) = graph.fast.as_ref() {
                let mut chk = vec![[0u64; 4]; graph.signals.len()];
                super::witness_fast::eval(fast, nodes, &input_vec, &mut chk);
                let bad = chk.iter().zip(buf.as_limbs_mut().iter()).filter(|(a, b)| a != b).count();
                eprintln!("[g16_par_check] witness_par vs witness_fast mismatches={bad}/{}", chk.len());
            }
        }
        return Ok(CalcWitness::Pinned(buf));
    }
    if let Some(program) = graph.fast.as_ref() {
        let mut buf = risc0_groth16_sys::host_buffer(graph.signals.len() * 32)
            .context("failed to allocate the pinned witness buffer")?;
        super::witness_fast::eval(program, nodes, &input_vec, buf.as_limbs_mut());
        return Ok(CalcWitness::Pinned(buf));
    }

    // Evaluate the witness graph directly (skip WTNS encode/decode roundtrip).
    // Temporarily redirect stdout to suppress circom_witnesscalc's println! calls
    // which add ~100-500ms of terminal I/O overhead.
    let result: Vec<U254> = {
        #[cfg(unix)]
        let _guard = suppress_stdout();
        evaluate(
            &nodes.ff,
            &nodes.nodes,
            &input_vec,
            &graph.signals,
            &nodes.constants,
        )
    };

    // Convert U254 values to FieldElement<32> (both are 32-byte little-endian).
    let witness: Vec<wtns_file::FieldElement<32>> = result
        .iter()
        .map(|a| {
            TryInto::<[u8; 32]>::try_into(a.as_le_slice())
                .unwrap()
                .into()
        })
        .collect();

    Ok(CalcWitness::Vec(witness))
}

/// RAII guard that redirects stdout to /dev/null and restores on drop.
/// Suppresses println! calls from third-party libraries during critical sections.
#[cfg(unix)]
struct StdoutGuard {
    saved_fd: std::os::unix::io::RawFd,
}

#[cfg(unix)]
impl Drop for StdoutGuard {
    fn drop(&mut self) {
        unsafe {
            extern "C" {
                fn dup2(oldfd: i32, newfd: i32) -> i32;
                fn close(fd: i32) -> i32;
            }
            dup2(self.saved_fd, 1);
            close(self.saved_fd);
        }
    }
}

#[cfg(unix)]
fn suppress_stdout() -> StdoutGuard {
    unsafe {
        extern "C" {
            fn dup(fd: i32) -> i32;
            fn dup2(oldfd: i32, newfd: i32) -> i32;
        }
        let saved = dup(1);
        if let Ok(devnull) = std::fs::File::open("/dev/null") {
            use std::os::unix::io::AsRawFd;
            dup2(devnull.as_raw_fd(), 1);
        }
        StdoutGuard { saved_fd: saved }
    }
}
