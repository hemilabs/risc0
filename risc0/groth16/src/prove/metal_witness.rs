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
//! - `fast`: the single-threaded slot evaluator: no per-node array (~30K recycled slots), so it
//!   also removes `fast.rs`' 580 MB calloc per wrap. The default below that limit.
//! - `ref`: [super::fast] (the pre-G13 evaluator; also the fallback if the graph does not
//!   qualify for the compiled evaluators).
//!
//! `RISC0_G16_METAL_WITNESS=par|fast|ref` overrides the choice. All three produce the same
//! witness (`metal_tests::witness_evaluators_bit_equal`).

use std::sync::OnceLock;

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
/// 16-24 GB Mac's, or a 4-8 GiB budget run, is below it).
pub(crate) const PAR_MIN_LIMIT_BYTES: u64 = 20 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Par,
    Fast,
    Ref,
}

fn parse_kind(v: Option<&str>, limit: Option<u64>) -> Kind {
    match v {
        Some("par") => Kind::Par,
        Some("fast") => Kind::Fast,
        Some("ref") => Kind::Ref,
        _ => match limit {
            Some(l) if l >= PAR_MIN_LIMIT_BYTES => Kind::Par,
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
        )
    })
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

struct Graph {
    nodes: Box<dyn NodesInterface>,
    n_signals: usize,
    input_offset: usize,
    input_len: usize,
    eval: Eval,
}

// Safety: the parsed graph (Nodes<U254, VecNodes>: Vecs, the field, a HashMap) and the compiled
// programs are plain data, written once (OnceLock) and only read afterwards (as cuda.rs's
// CachedGraph).
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Graph {
    fn nodes(&self) -> &Nodes<U254, VecNodes> {
        self.nodes
            .as_any()
            .downcast_ref::<Nodes<U254, VecNodes>>()
            .expect("checked at build")
    }
}

static GRAPH: OnceLock<std::result::Result<Graph, String>> = OnceLock::new();

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
        Kind::Par => witness_par::compile_with(vn, &signals, par_threads()).map(Eval::Par),
        Kind::Fast => witness_fast::compile(vn, &signals).map(Eval::Fast),
        Kind::Ref => None,
    }
    .ok_or_else(|| anyhow!("the witness graph does not qualify for the {kind:?} evaluator"))?;
    let msg = format!(
        "groth16 witness graph ({kind:?} evaluator) ready in {:.1} ms (read {:.1}, digest + \
         parse {:.1}, compile {:.1})",
        t0.elapsed().as_secs_f64() * 1000.0,
        t_read.as_secs_f64() * 1000.0,
        t_parse.as_secs_f64() * 1000.0,
        t1.elapsed().as_secs_f64() * 1000.0,
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
    })
}

/// The compiled graph, or `None` when this process uses `ref` (or the graph did not qualify:
/// then [super::fast] is used, as before G13).
fn graph() -> Option<&'static Graph> {
    if kind() == Kind::Ref {
        return None;
    }
    match GRAPH.get_or_init(|| build(kind()).map_err(|e| format!("{e:#}"))) {
        Ok(g) => Some(g),
        Err(e) => {
            static WARNED: OnceLock<()> = OnceLock::new();
            WARNED.get_or_init(|| {
                tracing::warn!("{e}; using the reference witness evaluator");
            });
            None
        }
    }
}

/// Read and compile the witness graph (call early, on a background thread).
pub(crate) fn preload() -> Result<()> {
    match graph() {
        Some(_) => Ok(()),
        None => super::fast::preload(),
    }
}

/// Whether the compiled `par` evaluator is in use (and built).
pub(crate) fn is_par() -> bool {
    matches!(
        graph(),
        Some(Graph {
            eval: Eval::Par(_),
            ..
        })
    )
}

/// The number of witness elements, or `None` when the reference evaluator is in use.
pub(crate) fn witness_len() -> Option<usize> {
    graph().map(|g| g.n_signals)
}

/// Evaluate the witness of an `identity_p254` seal into `out` (exactly [witness_len] elements,
/// canonical little-endian). Errors when the reference evaluator is in use.
pub(crate) fn witness_into(seal_bytes: &[u8], out: &mut [Limbs]) -> Result<()> {
    let g = graph().ok_or_else(|| anyhow!("no compiled witness evaluator"))?;
    anyhow::ensure!(
        out.len() == g.n_signals,
        "groth16 witness buffer of {} elements, expected {}",
        out.len(),
        g.n_signals
    );
    let seal = super::fast::seal_inputs(seal_bytes)?;
    debug_assert_eq!(seal.len(), g.input_len);
    // Input 0 is the constant 1; the seal occupies [input_offset, input_offset + input_len);
    // any other input is 0 (as fast.rs and the reference generator).
    let mut inputs = vec![U254::ZERO; g.input_offset + g.input_len];
    inputs[0] = U254::from(1u64);
    inputs[g.input_offset..].copy_from_slice(&seal);
    match &g.eval {
        Eval::Par(p) => witness_par::eval(p, g.nodes(), &inputs, out),
        Eval::Fast(p) => witness_fast::eval(p, g.nodes(), &inputs, out),
    }
    Ok(())
}

/// [witness_into] a new vector.
pub(crate) fn witness(seal_bytes: &[u8]) -> Result<Vec<Limbs>> {
    let n = witness_len().ok_or_else(|| anyhow!("no compiled witness evaluator"))?;
    let mut out = vec![[0u64; 4]; n];
    witness_into(seal_bytes, &mut out)?;
    Ok(out)
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
        Kind::Par => witness_par::compile_with(vn, &signals, threads).map(Eval::Par),
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
    };
    let seal = super::fast::seal_inputs(seal_bytes)?;
    let mut inputs = vec![U254::ZERO; g.input_offset + g.input_len];
    inputs[0] = U254::from(1u64);
    inputs[g.input_offset..].copy_from_slice(&seal);
    let mut out = vec![[0u64; 4]; g.n_signals];
    match &g.eval {
        Eval::Par(p) => witness_par::eval(p, g.nodes(), &inputs, &mut out),
        Eval::Fast(p) => witness_fast::eval(p, g.nodes(), &inputs, &mut out),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_selection() {
        assert_eq!(parse_kind(Some("par"), Some(1)), Kind::Par);
        assert_eq!(parse_kind(Some("fast"), Some(64 << 30)), Kind::Fast);
        assert_eq!(parse_kind(Some("ref"), None), Kind::Ref);
        assert_eq!(parse_kind(None, Some(PAR_MIN_LIMIT_BYTES)), Kind::Par);
        assert_eq!(parse_kind(None, Some(8 << 30)), Kind::Fast);
        assert_eq!(parse_kind(Some("auto"), Some(4 << 30)), Kind::Fast);
        assert_eq!(parse_kind(None, None), Kind::Fast);
    }
}
