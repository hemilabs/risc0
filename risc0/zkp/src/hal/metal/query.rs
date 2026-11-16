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

//! Z11: Merkle query openings read on the host in parallel.
//!
//! Metal buffers live in unified memory, so the query gathers need no kernel: the host reads
//! `matrix[row + i * row_size]` and the sibling digests directly. The generic unified-memory
//! path of `MerkleTreeProver::batch_prove` does that element by element on one thread, one tree
//! after another (~12-15 ms per po2 20 segment, GPU idle). Nearly all of that is first-touch
//! page faults: each strided column read lands on its own page of a GPU-written buffer, and a
//! second pass over the same openings takes ~0.1 ms. Here every buffer is synced once and the
//! (tree, query) reads of all trees are spread over the rayon pool, which overlaps the faults
//! (~3.5 ms). The values and their order are exactly those of the serial path, so proofs are
//! bit-identical. (A GPU `gather_sample` variant was slower on a shared GPU: Z11 note.)
//!
//! `RISC0_METAL_QUERY_PAR=0` reverts to the generic serial path (`gather_multi` returns `None`,
//! `batch_get_digest_at` reads one `get_at` per index).

use std::slice;

use rayon::prelude::*;

use super::buffer::BufferImpl;
use crate::{core::digest::Digest, hal::Buffer as _};

/// Sibling digests read per rayon task.
const DIGEST_CHUNK: usize = 32;

/// Below this many digests `batch_get_digest_at` reads on the calling thread (still one sync).
const DIGEST_PAR_MIN: usize = 2 * DIGEST_CHUNK;

/// Whether the parallel host query reads are on (default) or reverted by
/// `RISC0_METAL_QUERY_PAR=0`. Read per call so a process can A/B both paths.
pub(super) fn query_par_enabled() -> bool {
    !std::env::var("RISC0_METAL_QUERY_PAR").is_ok_and(|v| v.trim() == "0")
}

/// The whole of `buf` as a host slice, after waiting for the GPU work that writes it.
///
/// The slice stays valid for as long as `buf`'s allocation does (the borrow ties it to `buf`);
/// the caller must not encode GPU writes to `buf` while it is in use. Every caller here only
/// reads buffers whose producing work has completed, and encodes nothing meanwhile.
fn host_slice<T: Clone>(buf: &BufferImpl<T>) -> &[T] {
    let mut out: (*const T, usize) = (core::ptr::null(), 0);
    buf.view(|view| out = (view.as_ptr(), view.len()));
    if out.1 == 0 {
        return &[];
    }
    // SAFETY: `view` synced the buffer for host access and handed out exactly this range of
    // its (shared, never moved) allocation, which lives at least as long as `buf`.
    unsafe { slice::from_raw_parts(out.0, out.1) }
}

/// The work items of [gather_multi], each writing a disjoint part of the output.
enum Task<'a, E> {
    /// One query's column: `out[i] = matrix[row + i * row_size]`.
    Column {
        out: &'a mut [E],
        matrix: &'a [E],
        row: usize,
        row_size: usize,
    },
    /// A run of sibling digests: `out[i] = nodes[idx[i]]`.
    Digests {
        out: &'a mut [Digest],
        nodes: &'a [Digest],
        idx: &'a [usize],
    },
}

/// One tree's gather: (matrix, col_size, row_size, query rows, nodes, sibling node indices).
pub(super) struct HostGather<'a, E> {
    pub matrix: &'a BufferImpl<E>,
    pub col_size: usize,
    pub row_size: usize,
    pub rows: &'a [usize],
    pub nodes: &'a BufferImpl<Digest>,
    pub node_indices: &'a [usize],
}

/// [crate::hal::Hal::gather_multi] on the host: per job, the `rows.len() * col_size` column
/// elements (query-major) and the digests at `node_indices`, read in parallel.
pub(super) fn gather_multi<E: Copy + Default + Send + Sync>(
    jobs: &[HostGather<'_, E>],
) -> Vec<(Vec<E>, Vec<Digest>)> {
    // Sync every buffer once up front (on this thread: rayon workers must not touch the HAL).
    let views: Vec<(&[E], &[Digest])> = jobs
        .iter()
        .map(|j| (host_slice(j.matrix), host_slice(j.nodes)))
        .collect();
    let mut out: Vec<(Vec<E>, Vec<Digest>)> = jobs
        .iter()
        .map(|j| {
            (
                vec![E::default(); j.rows.len() * j.col_size],
                vec![Digest::ZERO; j.node_indices.len()],
            )
        })
        .collect();
    let mut tasks: Vec<Task<'_, E>> = Vec::new();
    for ((job, &(matrix, nodes)), (cols, digests)) in jobs.iter().zip(&views).zip(out.iter_mut()) {
        if job.col_size > 0 {
            for (chunk, &row) in cols.chunks_mut(job.col_size).zip(job.rows) {
                assert!(row < job.row_size, "query row {row} >= {}", job.row_size);
                tasks.push(Task::Column {
                    out: chunk,
                    matrix,
                    row,
                    row_size: job.row_size,
                });
            }
        }
        for (chunk, idx) in digests
            .chunks_mut(DIGEST_CHUNK)
            .zip(job.node_indices.chunks(DIGEST_CHUNK))
        {
            tasks.push(Task::Digests {
                out: chunk,
                nodes,
                idx,
            });
        }
    }
    // Columns and digest runs together, so the pool stays busy across trees.
    tasks.into_par_iter().for_each(|task| match task {
        Task::Column {
            out,
            matrix,
            row,
            row_size,
        } => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = matrix[row + i * row_size];
            }
        }
        Task::Digests { out, nodes, idx } => {
            for (o, &i) in out.iter_mut().zip(idx) {
                *o = nodes[i];
            }
        }
    });
    out
}

/// [crate::hal::Hal::batch_get_digest_at]: one sync, then the reads (in parallel when there are
/// enough of them).
pub(super) fn batch_get_digest_at(buf: &BufferImpl<Digest>, indices: &[usize]) -> Vec<Digest> {
    if !query_par_enabled() {
        return indices.iter().map(|&idx| buf.get_at(idx)).collect();
    }
    if indices.is_empty() {
        return Vec::new();
    }
    let nodes = host_slice(buf);
    if indices.len() < DIGEST_PAR_MIN {
        return indices.iter().map(|&i| nodes[i]).collect();
    }
    let mut out = vec![Digest::ZERO; indices.len()];
    out.par_chunks_mut(DIGEST_CHUNK)
        .zip(indices.par_chunks(DIGEST_CHUNK))
        .for_each(|(out, idx)| {
            for (o, &i) in out.iter_mut().zip(idx) {
                *o = nodes[i];
            }
        });
    out
}
