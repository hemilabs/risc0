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

use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{anyhow, Context as _, Result};
use circom_witnesscalc::{
    field::U254,
    graph::{evaluate, Nodes, NodesInterface, VecNodes},
    storage::proto_deserializer::deserialize_witnesscalc_graph_from_bytes,
    InputSignalsInfo,
};
use risc0_core::field::baby_bear::BabyBearElem;
use risc0_groth16_sys::{ProverParams, SetupParams, WitnessParams};
use risc0_zkp::core::digest::{Digest, DIGEST_WORDS};
use rzup::{Component, Rzup, Version};
use tempfile::tempdir;

use crate::Seal;

use super::seal_format::{IopType, K_SEAL_ELEMS, K_SEAL_TYPES, K_SEAL_WORDS};

static ROOT_DIR_CACHE: OnceLock<PathBuf> = OnceLock::new();
static PARSED_GRAPH_CACHE: OnceLock<ParsedGraph> = OnceLock::new();

/// Cached parsed circom graph for witness calculation.
/// Avoids re-parsing the ~40MB protobuf graph file on every call (~380ms saved).
struct ParsedGraph {
    nodes: Nodes<U254, VecNodes>,
    signals: Vec<usize>,
    input_mapping: InputSignalsInfo,
}

// SAFETY: All fields of Nodes<U254, VecNodes> are plain data types that are
// inherently Send+Sync: Field<U254> (3 Copy values), VecNodes (Vec<Node>),
// Vec<U254>, HashMap<U254, usize>, String. The NodesInterface trait simply
// doesn't declare Send+Sync bounds, but the concrete type is safe to share.
unsafe impl Send for ParsedGraph {}
unsafe impl Sync for ParsedGraph {}

fn resolve_root_dir() -> Result<PathBuf> {
    Rzup::new()
        .context("failed to initialize rzup")?
        .get_version_dir(&Component::Risc0Groth16, &Version::new(0, 1, 0))
        .context(
            "Missing required `risc0-groth16` rzup component. \
            To install it, ensure that your `rzup` version is >= 0.5.0, \
            and then run `rzup install risc0-groth16`.",
        )
}

fn get_cached_root_dir() -> Result<&'static Path> {
    if let Some(dir) = ROOT_DIR_CACHE.get() {
        return Ok(dir.as_path());
    }
    let dir = resolve_root_dir()?;
    Ok(ROOT_DIR_CACHE.get_or_init(|| dir).as_path())
}

fn parse_graph(graph_bytes: &[u8]) -> ParsedGraph {
    let (nodes_box, signals, input_mapping) =
        deserialize_witnesscalc_graph_from_bytes(graph_bytes)
            .expect("failed to parse witnesscalc graph");

    // Verify the concrete type before downcasting.
    assert!(
        nodes_box.as_any().is::<Nodes<U254, VecNodes>>(),
        "expected BN254 graph type"
    );

    // Extract concrete type from the trait object.
    // SAFETY: Type verified above. Fat-to-thin pointer cast extracts the
    // data pointer (well-defined per Rust reference). We consumed ownership
    // via Box::into_raw and reconstruct it via Box::from_raw.
    let nodes: Nodes<U254, VecNodes> = unsafe {
        let raw: *mut dyn NodesInterface = Box::into_raw(nodes_box);
        *Box::from_raw(raw as *mut Nodes<U254, VecNodes>)
    };

    ParsedGraph {
        nodes,
        signals,
        input_mapping,
    }
}

fn get_cached_parsed_graph(root_dir: &Path) -> &'static ParsedGraph {
    PARSED_GRAPH_CACHE.get_or_init(|| {
        let wp = WitnessParams::new(root_dir);
        let graph_bytes =
            std::fs::read(&wp.graph_path).expect("failed to read groth16 graph file");
        parse_graph(&graph_bytes)
    })
}

/// Pre-load resources needed for Groth16 proving in the background.
/// Call before identity_p254 to overlap I/O with GPU work.
pub(crate) fn prepare() {
    if PARSED_GRAPH_CACHE.get().is_some() && ROOT_DIR_CACHE.get().is_some() {
        return;
    }
    std::thread::spawn(|| {
        if let Ok(root_dir) = resolve_root_dir() {
            let _ = get_cached_parsed_graph(&root_dir);

            // Pre-warm SRS-related files into OS page cache so the SRS
            // constructor's mmap accesses hit RAM instead of disk.
            let setup = SetupParams::new(&root_dir).ok();
            if let Some(setup) = &setup {
                let _ = std::fs::read(setup.srs_path.as_path());
                let _ = std::fs::read(setup.pcoeffs_path.as_path());
                let _ = std::fs::read(setup.fres_path.as_path());
            }

            ROOT_DIR_CACHE.get_or_init(|| root_dir);
        }
    });
}

pub(crate) fn shrink_wrap(seal_bytes: &[u8]) -> Result<Seal> {
    tracing::info!("shrink_wrap: {} seal bytes", seal_bytes.len());

    let root_dir = get_cached_root_dir()?;
    let setup_params =
        SetupParams::new(root_dir).context("failed to create groth16 work directories")?;

    // Overlap SRS init (GPU, main thread) with witness calc (CPU, scoped thread).
    // CudaHal holds a reentrant lock, so only the main thread can acquire it.
    // calc_witness is pure CPU work and doesn't need GPU access.
    let witness = std::thread::scope(|s| -> Result<CalcWitness> {
        let witness_handle = s.spawn(|| -> Result<CalcWitness> {
            let parsed = get_cached_parsed_graph(root_dir);
            calc_witness(parsed, seal_bytes).context("failed to calculate groth16 witness")
        });

        // SRS+prover init on main thread (reentrant lock from cached CudaHal).
        // C++ statics cache the SRS and prover so subsequent calls are free.
        {
            let _lock = risc0_zkp::hal::cuda::singleton().lock();
            risc0_groth16_sys::init(&setup_params)
                .context("failed to initialize groth16 SRS+prover")?;
        }

        witness_handle
            .join()
            .map_err(|_| anyhow!("witness calc thread panicked"))?
    })?;

    // GPU prove using cached SRS+prover — raw bytes path avoids proof file I/O
    {
        let _lock = risc0_zkp::hal::cuda::singleton().lock();

        let tmp_dir = tempdir().context("failed to create temporary directory")?;
        let work_dir = std::env::var("RISC0_WORK_DIR");
        let work_dir = work_dir.as_ref().map(Path::new).unwrap_or(tmp_dir.path());

        let prover_params = ProverParams::new(work_dir, witness.as_ptr())
            .context("failed to create groth16 prover parameters")?;
        let raw = risc0_groth16_sys::prove_raw(&prover_params, &setup_params)
            .context("failed to run groth16 prove operation")?;

        Ok(raw_proof_to_seal(&raw))
    }
}

/// Convert raw proof bytes (256 bytes, 8 × 32-byte LE field elements) to a Seal.
/// Raw layout: [a.x, a.y, c.x, c.y, b0, b1, b2, b3] — each 32 bytes little-endian.
/// Seal expects big-endian, and b coordinates are swapped within each pair.
fn raw_proof_to_seal(raw: &risc0_groth16_sys::RawProofOutput) -> Seal {
    let d = &raw.data;
    let rev = |start: usize| -> Vec<u8> {
        let mut v = d[start..start + 32].to_vec();
        v.reverse();
        v
    };
    Seal {
        a: vec![rev(0), rev(32)],
        b: vec![
            vec![rev(160), rev(128)], // b[1] before b[0]
            vec![rev(224), rev(192)], // b[3] before b[2]
        ],
        c: vec![rev(64), rev(96)],
    }
}

struct CalcWitness {
    witness: Vec<U254>,
}

impl CalcWitness {
    fn as_ptr(&self) -> *const u8 {
        // U254 = Uint<254, 4> = [u64; 4] = 32 bytes LE on x86.
        // C++ fr_t is also 32-byte LE. No conversion needed.
        self.witness.as_ptr() as *const u8
    }
}

/// Build the circom input vector directly from seal bytes, bypassing JSON serialization.
/// The seal contains K_SEAL_ELEMS entries (Fp or Digest types) which map to the "iop" signal.
/// Direct byte conversion avoids: JSON creation, BigUint decimal formatting, JSON parsing,
/// and decimal string → U254 conversion (~200ms saved for 4274 digest conversions).
fn seal_to_input_vec(seal_bytes: &[u8], parsed: &ParsedGraph) -> Result<Vec<U254>> {
    let iop: &[u32] = bytemuck::cast_slice(seal_bytes);
    assert_eq!(iop.len(), K_SEAL_WORDS, "seal word count mismatch");

    // Convert seal elements directly to U254 values
    let mut values = Vec::with_capacity(K_SEAL_ELEMS);
    let mut pos = 0;
    for seal_type in K_SEAL_TYPES.iter().take(K_SEAL_ELEMS) {
        match seal_type {
            IopType::Fp => {
                let value = BabyBearElem::new_raw(iop[pos]).as_u32();
                pos += 1;
                values.push(U254::from(value as u64));
            }
            IopType::Digest => {
                let digest = Digest::try_from(&iop[pos..pos + DIGEST_WORDS])?;
                pos += DIGEST_WORDS;
                // Digest bytes are the 32-byte LE representation of a BN254 field element.
                // U254::try_from_le_slice interprets these bytes directly — same as
                // digest_to_fr() followed by decimal string conversion, but without
                // the BigUint/string round-trip.
                let u = U254::try_from_le_slice(digest.as_bytes())
                    .ok_or_else(|| anyhow!("failed to convert digest to U254"))?;
                values.push(u);
            }
        }
    }

    // Look up the "iop" signal mapping
    let &(offset, len) = parsed
        .input_mapping
        .get("iop")
        .ok_or_else(|| anyhow!("missing 'iop' signal in input mapping"))?;
    if len != values.len() {
        return Err(anyhow!(
            "seal element count mismatch: expected {len}, got {}",
            values.len()
        ));
    }

    // Build the full input vector
    let mut max_idx: usize = 0;
    for &(off, l) in parsed.input_mapping.values() {
        let idx = off + l;
        if idx > max_idx {
            max_idx = idx;
        }
    }
    let mut input_vec = vec![U254::ZERO; max_idx + 1];
    input_vec[0] = U254::from(1u64);
    for (i, v) in values.into_iter().enumerate() {
        input_vec[offset + i] = v;
    }

    Ok(input_vec)
}

fn calc_witness(parsed: &ParsedGraph, seal_bytes: &[u8]) -> Result<CalcWitness> {
    let input_vec = seal_to_input_vec(seal_bytes, parsed)?;

    // Evaluate the circuit graph directly (skips ~380ms graph re-parsing).
    // Returns Vec<U254> which has same 32-byte LE layout as C++ fr_t — no conversion needed.
    let result = evaluate(
        &parsed.nodes.ff,
        &parsed.nodes.nodes,
        &input_vec,
        &parsed.signals,
        &parsed.nodes.constants,
    );

    Ok(CalcWitness { witness: result })
}
