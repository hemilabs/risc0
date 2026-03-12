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
use circom_witnesscalc::graph::{evaluate, Nodes, NodesInterface, VecNodes};
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
        Rzup::new()
            .and_then(|r| r.get_version_dir(&Component::Risc0Groth16, &Version::new(0, 1, 0)))
            .expect(
                "Missing required `risc0-groth16` rzup component. \
                To install it, ensure that your `rzup` version is >= 0.5.0, \
                and then run `rzup install risc0-groth16`.",
            )
    });
    Ok(dir.as_path())
}

/// Cached deserialized circom graph to avoid re-parsing ~241MB on each call (~554ms).
struct CachedGraph {
    nodes: Box<dyn NodesInterface>,
    signals: Vec<usize>,
    input_mapping: InputSignalsInfo,
}

// Safety: The underlying Nodes<U254, VecNodes> contains Vec<Node>, Field<U254>, HashMap —
// all Send+Sync. The graph is written once (OnceLock init) and read immutably after.
unsafe impl Send for CachedGraph {}
unsafe impl Sync for CachedGraph {}

static CACHED_GRAPH: OnceLock<CachedGraph> = OnceLock::new();

fn get_cached_graph(graph_path: &Path) -> &'static CachedGraph {
    CACHED_GRAPH.get_or_init(|| {
        let graph_bytes =
            std::fs::read(graph_path).expect("failed to read circom graph file");
        let (nodes, signals, input_mapping) =
            deserialize_witnesscalc_graph_from_bytes(&graph_bytes)
                .expect("failed to deserialize circom graph");
        CachedGraph {
            nodes,
            signals,
            input_mapping,
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
    risc0_groth16_sys::preload(&setup_params)
        .context("failed to preload groth16 SRS")?;
    Ok(())
}

pub(crate) fn shrink_wrap(seal_bytes: &[u8]) -> Result<Seal> {
    tracing::info!("shrink_wrap: {} seal bytes", seal_bytes.len());

    let root_dir = get_root_dir()?;
    let tmp_dir = tempdir().context("failed to create temporary directory")?;
    let work_dir = std::env::var("RISC0_WORK_DIR");
    let work_dir = work_dir.as_ref().map(Path::new).unwrap_or(tmp_dir.path());
    let setup_params =
        SetupParams::new(root_dir).context("failed to create groth16 work directories")?;

    let inputs = to_json(seal_bytes)?;

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
    let witness = calc_witness(graph, &inputs)
        .context("failed to calculate groth16 witness")?;

    // Wait for SRS preload to complete before taking the GPU lock.
    let _ = srs_preload.join();

    {
        let _lock = risc0_zkp::hal::cuda::singleton().lock();

        let prover_params = ProverParams::new(work_dir, witness.as_ptr())
            .context("failed to create groth16 prover parameters")?;
        risc0_groth16_sys::prove(&prover_params, &setup_params)
            .context("failed to run groth16 prove operation")?;

        let contents = std::fs::read_to_string(prover_params.proof_path.as_path())
            .context("failed to read groth16 prove output file")?;
        let proof_json: ProofJson = serde_json::from_str(&contents)
            .context("failed to decode groth16 prove output file JSON")?;
        proof_json.try_into()
    }
}

struct CalcWitness {
    witness: Vec<wtns_file::FieldElement<32>>,
}

impl CalcWitness {
    fn as_ptr(&self) -> *const u8 {
        self.witness.as_ptr() as *const u8
    }
}

fn calc_witness(graph: &CachedGraph, inputs: &str) -> Result<CalcWitness> {
    let nodes = graph
        .nodes
        .as_any()
        .downcast_ref::<Nodes<U254, VecNodes>>()
        .ok_or_else(|| anyhow::anyhow!("unexpected graph node type"))?;

    // Parse inputs into field elements.
    let input_list: HashMap<String, Vec<U254>> =
        deserialize_inputs2(inputs.as_bytes(), &nodes.ff)
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

    Ok(CalcWitness { witness })
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
