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

use alloc::{sync::Arc, vec::Vec};

use parking_lot::Mutex;

use risc0_core::scope;

use crate::{
    core::digest::Digest,
    hal::{Buffer, Hal},
    merkle::MerkleTreeParams,
    prove::write_iop::WriteIOP,
};

pub struct MerkleTreeProver<H: Hal> {
    params: MerkleTreeParams,

    // The retained matrix of values (Option to allow releasing for VRAM savings)
    pub(crate) matrix: Option<H::Buffer<H::Elem>>,

    // A heap style array where node N has children 2*N and 2*N+1.  The size of
    // this buffer is (1 << (layers + 1)) and begins at offset 1 (zero is unused
    // to make indexing nicer).
    nodes: H::Buffer<Digest>,

    // The root value
    root: Digest,

    // Pre-allocated buffer for gather_sample in prove(), avoids per-query alloc/free
    sample_buf: Option<H::Buffer<H::Elem>>,

    // Cached top nodes for commit() — shared across clones via Arc.
    // Populated on first commit(), reused on subsequent commits to avoid GPU
    // D2H. Arc+spin::Mutex (instead of Rc+RefCell) so MerkleTreeProver is
    // Send — required for the finalize-overlap path that moves the prover
    // state into a background thread.
    cached_top_nodes: Arc<Mutex<Option<Vec<Digest>>>>,
}

impl<H: Hal> Clone for MerkleTreeProver<H> {
    fn clone(&self) -> Self {
        Self {
            params: self.params,
            matrix: self.matrix.clone(),
            nodes: self.nodes.clone(),
            root: self.root,
            sample_buf: self.sample_buf.clone(),
            cached_top_nodes: self.cached_top_nodes.clone(),
        }
    }
}

impl<H: Hal> MerkleTreeProver<H> {
    /// Release the matrix buffer to free GPU memory.
    /// After this, batch_prove will panic unless set_matrix() is called first.
    pub fn release_matrix(&mut self) {
        self.matrix = None;
    }

    /// Set the matrix buffer (e.g., after reconstructing evaluated from coefficients).
    pub fn set_matrix(&mut self, matrix: H::Buffer<H::Elem>) {
        self.matrix = Some(matrix);
    }

    /// Check if the matrix is present.
    pub fn matrix_is_some(&self) -> bool {
        self.matrix.is_some()
    }
}

impl<H: Hal> MerkleTreeProver<H> {
    /// Generate a merkle tree from a matrix of values.
    ///
    /// The proofs will prove a single 'column' of values in the tree at a
    /// certain row. Layout is presumed to be packed row-major.
    /// The number of queries represents the expected # of queries and
    /// determines the size of the 'top' layer. It is important that the
    /// verifier is constructed with identical size parameters, including # of
    /// queries, or verification may fail.
    ///
    /// matrix: `rows * cols`
    /// rows: `domain = steps * INV_RATE`, `steps` is always a power of 2.
    /// cols: `count = circuit_cols`
    pub fn new(
        hal: &H,
        matrix: &H::Buffer<H::Elem>,
        rows: usize,
        cols: usize,
        queries: usize,
    ) -> Self {
        assert_eq!(matrix.size(), rows * cols);
        let params = MerkleTreeParams::new(rows, cols, queries);
        // Allocate nodes
        let nodes = hal.alloc_digest("nodes", rows * 2);
        // hash each column
        hal.hash_rows(&nodes.slice(rows, rows), matrix);
        // For each layer, hash up the layer below
        scope!("hash_fold", {
            hal.hash_fold_tree(&nodes, params.layers);
        });
        // ROCm: the root and the top layer (written by `commit`) in one download: nodes[1..2 *
        // top_size) holds both (root at 1, top layer at top_size..2 * top_size).
        let (root, top_nodes) = if cfg!(feature = "rocm") {
            let top_size = params.top_size;
            let mut root = Digest::ZERO;
            let mut top_nodes = Vec::new();
            nodes.slice(1, 2 * top_size - 1).view(|view| {
                root = view[0];
                top_nodes = view[top_size - 1..].to_vec();
            });
            (root, Some(top_nodes))
        } else {
            (nodes.get_at(1), None)
        };
        let sample_buf = if !hal.has_unified_memory() && params.col_size > 0 {
            Some(hal.alloc_elem("sample_reuse", params.col_size))
        } else {
            None
        };
        MerkleTreeProver {
            params,
            matrix: Some(matrix.clone()),
            nodes,
            root,
            sample_buf,
            cached_top_nodes: Arc::new(Mutex::new(top_nodes)),
        }
    }

    /// Write the 'top' of the merkle tree and commit to the root.
    pub fn commit(&self, iop: &mut WriteIOP<H::Field>) {
        scope!("commit");
        let top_size = self.params.top_size;
        let cached = self.cached_top_nodes.lock();
        if let Some(ref top_nodes) = *cached {
            // Use cached CPU data — no GPU D2H needed.
            iop.write_pod_slice(top_nodes);
        } else {
            drop(cached);
            let slice = self.nodes.slice(top_size, top_size);
            slice.view(|view| {
                *self.cached_top_nodes.lock() = Some(view.to_vec());
                iop.write_pod_slice(view);
            });
        }
        iop.commit(self.root());
    }

    /// Get the root digest of the tree.
    pub fn root(&self) -> &Digest {
        &self.root
    }

    /// Generate a proof at a given index, and return the values at that column.
    ///
    /// The format of the proof is always:
    /// 1) The column itself
    /// 2) The 'other' digests up to the top.
    ///
    /// It is presumed the verifier is given the index of the row from other
    /// parts of the protocol, and verification will of course fail if the
    /// wrong row is specified.
    pub fn prove(&self, hal: &H, iop: &mut WriteIOP<H::Field>, idx: usize) -> Vec<H::Elem> {
        assert!(idx < self.params.row_size);
        let matrix = self
            .matrix
            .as_ref()
            .expect("matrix released; call set_matrix() first");
        let mut out = Vec::with_capacity(self.params.col_size);
        if hal.has_unified_memory() {
            matrix.view(|view| {
                for i in 0..self.params.col_size {
                    out.push(view[idx + i * self.params.row_size]);
                }
            });
        } else if let Some(ref sample) = self.sample_buf {
            hal.gather_sample(
                sample,
                matrix,
                idx,
                self.params.col_size,
                self.params.row_size,
            );
            sample.view(|view| {
                out.extend_from_slice(view);
            });
        }
        iop.write_field_elem_slice::<H::Elem>(out.as_slice());
        // Collect all sibling indices for the proof path
        let mut node_indices = Vec::new();
        let mut cur_idx = idx + self.params.row_size;
        while cur_idx >= 2 * self.params.top_size {
            let low_bit = cur_idx % 2;
            cur_idx /= 2;
            node_indices.push(2 * cur_idx + (1 - low_bit));
        }
        // Batch-read all sibling digests (single D2H on CUDA)
        let siblings = hal.batch_get_digest_at(&self.nodes, &node_indices);
        for digest in &siblings {
            iop.write_pod_slice(&[*digest]);
        }
        out
    }

    /// Sibling node indices of the Merkle paths of `indices`, and how many each path has.
    fn sibling_indices(&self, indices: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let mut all = Vec::new();
        let mut counts = Vec::with_capacity(indices.len());
        for &idx in indices {
            let mut count = 0;
            let mut cur_idx = idx + self.params.row_size;
            while cur_idx >= 2 * self.params.top_size {
                let low_bit = cur_idx % 2;
                cur_idx /= 2;
                all.push(2 * cur_idx + (1 - low_bit));
                count += 1;
            }
            counts.push(count);
        }
        (all, counts)
    }

    /// [Self::batch_prove] for several trees (each with its own indices), with the gathers of
    /// all of them in one host round trip where the HAL supports it (`Hal::gather_multi`).
    #[allow(clippy::type_complexity)]
    pub fn batch_prove_many(
        hal: &H,
        jobs: &[(&Self, &[usize])],
    ) -> Vec<Vec<(Vec<H::Elem>, Vec<Digest>)>> {
        if let Some(out) = Self::try_gather_multi(hal, jobs) {
            return out;
        }
        jobs.iter()
            .map(|(tree, indices)| tree.batch_prove(hal, indices))
            .collect()
    }

    /// The query openings of `jobs` through `Hal::gather_multi`, or `None` where the HAL has no
    /// such path (then the caller uses the per-tree [Self::batch_prove]). Unified-memory HALs
    /// other than Metal return `None` from `gather_multi`, so they keep the per-tree path; Metal
    /// reads the openings on the host in parallel (`RISC0_METAL_QUERY_PAR=0` reverts it).
    #[allow(clippy::type_complexity)]
    fn try_gather_multi(
        hal: &H,
        jobs: &[(&Self, &[usize])],
    ) -> Option<Vec<Vec<(Vec<H::Elem>, Vec<Digest>)>>> {
        if jobs
            .iter()
            .any(|(t, _)| t.matrix.is_none() || t.params.col_size == 0)
        {
            return None;
        }
        for (tree, indices) in jobs {
            for &idx in indices.iter() {
                assert!(idx < tree.params.row_size);
            }
        }
        let paths: Vec<(Vec<usize>, Vec<usize>)> = jobs
            .iter()
            .map(|(t, indices)| t.sibling_indices(indices))
            .collect();
        let gjobs: Vec<crate::hal::GatherJob<'_, H>> = jobs
            .iter()
            .zip(&paths)
            .map(|((t, indices), (nodes, _))| crate::hal::GatherJob {
                matrix: t.matrix.as_ref().unwrap(),
                col_size: t.params.col_size,
                row_size: t.params.row_size,
                rows: indices,
                nodes: &t.nodes,
                node_indices: nodes,
            })
            .collect();
        let gathered = hal.gather_multi(&gjobs)?;
        Some(
            gathered
                .into_iter()
                .zip(jobs.iter().zip(&paths))
                .map(|((cols, digests), ((t, indices), (_, counts)))| {
                    let col_size = t.params.col_size;
                    let mut off = 0;
                    (0..indices.len())
                        .map(|q| {
                            let c = cols[q * col_size..(q + 1) * col_size].to_vec();
                            let d = digests[off..off + counts[q]].to_vec();
                            off += counts[q];
                            (c, d)
                        })
                        .collect()
                })
                .collect(),
        )
    }

    /// Batch-prove multiple indices at once, returning data without writing to IOP.
    /// GPU work is batched: N gather_sample kernels + 1 D2H, 1 batch digest read + 1 D2H.
    /// The caller is responsible for writing to IOP in the correct order.
    pub fn batch_prove(&self, hal: &H, indices: &[usize]) -> Vec<(Vec<H::Elem>, Vec<Digest>)> {
        let n = indices.len();
        if n == 0 {
            return Vec::new();
        }
        let matrix = self
            .matrix
            .as_ref()
            .expect("matrix released; call set_matrix() first");
        let col_size = self.params.col_size;
        // Unified memory: a HAL with a parallel host gather (Metal) reads all of it at once.
        if hal.has_unified_memory() {
            if let Some(mut out) = Self::try_gather_multi(hal, &[(self, indices)]) {
                return out.pop().unwrap();
            }
        }

        // Collect column data for all queries
        let mut all_col_data: Vec<Vec<H::Elem>> = Vec::with_capacity(n);
        if hal.has_unified_memory() {
            matrix.view(|view| {
                for &idx in indices {
                    assert!(idx < self.params.row_size);
                    let mut out = Vec::with_capacity(col_size);
                    for i in 0..col_size {
                        out.push(view[idx + i * self.params.row_size]);
                    }
                    all_col_data.push(out);
                }
            });
        } else if col_size > 0 {
            // GPU path: batch all gather_sample kernels (async), single D2H via view()
            let batch_buf = hal.alloc_elem("batch_samples", n * col_size);
            for &idx in indices.iter() {
                assert!(idx < self.params.row_size);
            }
            hal.gather_samples(&batch_buf, matrix, indices, col_size, self.params.row_size);
            batch_buf.view(|view| {
                for i in 0..n {
                    all_col_data.push(view[i * col_size..(i + 1) * col_size].to_vec());
                }
            });
        } else {
            all_col_data.resize_with(n, Vec::new);
        }

        // Compute all sibling node indices across all queries
        let mut all_node_indices = Vec::new();
        let mut per_query_digest_count = Vec::new();
        for &idx in indices {
            let mut count = 0;
            let mut cur_idx = idx + self.params.row_size;
            while cur_idx >= 2 * self.params.top_size {
                let low_bit = cur_idx % 2;
                cur_idx /= 2;
                all_node_indices.push(2 * cur_idx + (1 - low_bit));
                count += 1;
            }
            per_query_digest_count.push(count);
        }

        // Single batch digest read (1 H2D + 1 kernel + 1 D2H on CUDA)
        let all_digests = hal.batch_get_digest_at(&self.nodes, &all_node_indices);

        // Assemble per-query results
        let mut results = Vec::with_capacity(n);
        let mut digest_offset = 0;
        for (i, &count) in per_query_digest_count.iter().enumerate() {
            let digests = all_digests[digest_offset..digest_offset + count].to_vec();
            digest_offset += count;
            results.push((core::mem::take(&mut all_col_data[i]), digests));
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use risc0_core::field::{
        baby_bear::{BabyBear, BabyBearElem},
        Elem,
    };

    use super::*;
    use crate::{
        core::{
            hash::{poseidon2::Poseidon2HashSuite, sha::Sha256HashSuite, HashSuite},
            log2_ceil,
        },
        hal::cpu::CpuHal,
        verify::{MerkleTreeVerifier, ReadIOP, VerificationError},
    };

    fn init_prover<H: Hal>(
        hal: &H,
        rows: usize,
        cols: usize,
        queries: usize,
    ) -> MerkleTreeProver<H> {
        // Initialize a prover with leaves 0..size
        let size: u32 = (rows * cols) as u32;
        let mut data: Vec<H::Elem> = Vec::new();
        for val in 0..size {
            data.push(H::Elem::from_u64((u32::MAX / 2) as u64 - val as u64));
        }
        let matrix = hal.copy_from_elem("matrix", data.as_slice());

        MerkleTreeProver::new(hal, &matrix, rows, cols, queries)
    }

    fn bad_row_access(suite: HashSuite<BabyBear>, rows: usize, cols: usize, queries: usize) {
        let hal = CpuHal::new(suite);
        let prover = init_prover(&hal, rows, cols, queries);
        let mut iop = WriteIOP::new(hal.get_hash_suite().rng.as_ref());
        prover.prove(&hal, &mut iop, rows);
    }

    fn bad_row_access_all(rows: usize, cols: usize, queries: usize) {
        bad_row_access(Sha256HashSuite::new_suite(), rows, cols, queries);
        bad_row_access(Poseidon2HashSuite::new_suite(), rows, cols, queries);
    }

    fn possibly_bad_verify(
        suite: HashSuite<BabyBear>,
        rows: usize,
        cols: usize,
        queries: usize,
        bad_query: usize,
        manipulate_proof: bool,
    ) {
        let hal = CpuHal::new(suite);
        let hashfn = hal.get_hash_suite().hashfn.as_ref();
        let rng = hal.get_hash_suite().rng.as_ref();
        let prover = init_prover(&hal, rows, cols, queries);

        let mut iop = WriteIOP::new(rng);
        prover.commit(&mut iop);
        for _query in 0..queries {
            let r_idx = iop.rng.random_bits(log2_ceil(rows)) as usize;
            let col = prover.prove(&hal, &mut iop, r_idx);
            for (c_idx, col) in col.iter().enumerate() {
                assert_eq!(
                    *col,
                    BabyBearElem::from_u64((u32::MAX / 2) as u64 - ((r_idx + c_idx * rows) as u64))
                );
            }
        }
        if manipulate_proof {
            let mut rng = rand::rng();
            let manip_idx = rng.random_range(0..iop.proof.len());
            iop.proof[manip_idx] ^= 1;
        }
        let mut r_iop = ReadIOP::new(&iop.proof, rng);
        let verifier = MerkleTreeVerifier::new(&mut r_iop, hashfn, rows, cols, queries).unwrap();
        assert_eq!(verifier.root(), prover.root());
        let mut err = false;
        for query in 0..queries {
            let r_idx = r_iop.random_bits(log2_ceil(rows)) as usize;
            if query == bad_query {
                assert_ne!(
                    rows, 1,
                    "Cannot test for bad query if there is only one row"
                );
                let r_idx = (r_idx + 1) % rows;
                let verification = verifier.verify(&mut r_iop, hashfn, r_idx);
                match verification {
                    Ok(_) => {
                        panic!("Merkle tree wrongly passed verify when tested on the wrong row")
                    }
                    Err(VerificationError::InvalidProof) => {}
                    Err(_) => panic!("Merkle tree failed validation for an unexpected reason"),
                }
                err = true;
                break;
            }
            let col = verifier.verify(&mut r_iop, hashfn, r_idx).unwrap();
            for (c_idx, cell) in col.iter().enumerate().take(cols) {
                assert_eq!(
                    *cell,
                    BabyBearElem::from((u32::MAX / 2) - ((r_idx + c_idx * rows) as u32))
                );
            }
        }
        if !err {
            r_iop.verify_complete().unwrap();
        }
    }

    fn possibly_bad_verify_all(
        rows: usize,
        cols: usize,
        queries: usize,
        bad_query: usize,
        manipulate_proof: bool,
    ) {
        possibly_bad_verify(
            Sha256HashSuite::new_suite(),
            rows,
            cols,
            queries,
            bad_query,
            manipulate_proof,
        );
        possibly_bad_verify(
            Poseidon2HashSuite::new_suite(),
            rows,
            cols,
            queries,
            bad_query,
            manipulate_proof,
        );
    }

    fn randomize_sizes() -> (usize, usize, usize) {
        // Chooses random values of `rows`, `cols`, and `queries` such that:
        // `rows` is a power of 2
        // `cols` & `queries` have a wide distribution but tend to take small values
        let mut rng = rand::rng();
        let rows = 1 << (rng.random_range(0..10));
        let cols_po2 = rng.random_range(0..10);
        let cols = (rng.random_range(0..(1 << cols_po2))) + 1;
        let queries_po2 = rng.random_range(0..10);
        let queries = (rng.random_range(0..(1 << queries_po2))) + 1;
        (rows, cols, queries)
    }

    #[test]
    #[should_panic(expected = "assertion failed: idx < self.params.row_size")]
    fn merkle_cpu_1_1_1_bad_row_access() {
        bad_row_access_all(1, 1, 1);
    }

    #[test]
    #[should_panic(expected = "assertion failed: idx < self.params.row_size")]
    fn merkle_cpu_4_4_2_bad_row_access() {
        bad_row_access_all(4, 4, 2);
    }

    #[test]
    #[should_panic(expected = "assertion failed: idx < self.params.row_size")]
    fn merkle_cpu_randomized_bad_row_access() {
        let (rows, cols, queries) = randomize_sizes();
        bad_row_access_all(rows, cols, queries);
    }

    #[test]
    fn merkle_cpu_1_1_1_verify() {
        // Test a complete verification with no bad queries (by setting bad_query out of
        // range)
        possibly_bad_verify_all(1, 1, 1, 4, false);
    }

    #[test]
    fn merkle_cpu_4_4_2_verify() {
        // Test a complete verification with no bad queries (by setting bad_query out of
        // range)
        possibly_bad_verify_all(4, 4, 2, 4, false);
    }

    #[test]
    fn merkle_cpu_randomized_verify() {
        for _rep in 0..100 {
            let (rows, cols, queries) = randomize_sizes();
            // Test a complete verification with no bad queries (by setting bad_query out of
            // range)
            possibly_bad_verify_all(rows, cols, queries, queries + 1, false);
        }
    }

    #[test]
    fn merkle_cpu_2_1_1_bad_query() {
        // n.b. since we test bad queries by incrementing the row, we can't test for a
        // bad query with rows == 1
        possibly_bad_verify_all(2, 1, 1, 0, false);
    }

    #[test]
    fn merkle_cpu_4_4_2_bad_query() {
        let mut rng = rand::rng();
        let queries = 2;
        // Test a complete verification with a bad query
        let bad_query = rng.random_range(0..queries);
        possibly_bad_verify_all(4, 4, queries, bad_query, false);
    }

    #[test]
    fn merkle_cpu_randomized_bad_query() {
        let mut rng = rand::rng();
        let (rows, cols, queries) = randomize_sizes();
        // At least two rows are required to test querying an incorrect row
        let rows = if rows == 1 { 2 } else { rows };
        // Test a complete verification with a bad query
        let bad_query = rng.random_range(0..queries);
        possibly_bad_verify_all(rows, cols, queries, bad_query, false);
    }

    #[test]
    #[should_panic]
    fn merkle_cpu_1_1_1_verify_manipulated() {
        for _rep in 0..50 {
            // Test a verification with a manipulated proof but no bad queries (by setting
            // bad_query out of range) Do this multiple times as the
            // manipulation location is random
            possibly_bad_verify_all(1, 1, 1, 2, true);
        }
    }

    #[test]
    #[should_panic]
    fn merkle_cpu_4_4_2_verify_manipulated() {
        for _rep in 0..50 {
            // Test a verification with a manipulated proof but no bad queries (by setting
            // bad_query out of range) Do this multiple times as the
            // manipulation location is random
            possibly_bad_verify_all(4, 4, 2, 4, true);
        }
    }

    #[test]
    #[should_panic]
    fn merkle_cpu_randomized_verify_manipulated() {
        for _rep in 0..50 {
            let (rows, cols, queries) = randomize_sizes();
            // Test a verification with a manipulated proof but no bad queries (by setting
            // bad_query out of range) Do this multiple times as the
            // manipulation location is random
            possibly_bad_verify_all(rows, cols, queries, queries + 1, true);
        }
    }

    /// Z11: on Metal the query openings go through `Hal::gather_multi` /
    /// `batch_get_digest_at` (parallel host reads). They must equal the CPU HAL's exactly, and
    /// so must the reverted path (`RISC0_METAL_QUERY_PAR=0`).
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    mod metal_z11 {
        use rand::{rngs::SmallRng, Rng, SeedableRng};
        use risc0_core::field::{baby_bear::BabyBearElem, Elem};

        use super::super::MerkleTreeProver;
        use crate::{
            core::hash::poseidon2::Poseidon2HashSuite,
            hal::{
                cpu::CpuHal,
                metal::{gpu_test_lock, MetalHalPoseidon2},
                Hal,
            },
            prove::write_iop::WriteIOP,
        };

        type Opening = (Vec<BabyBearElem>, Vec<crate::core::digest::Digest>);
        type Cpu = CpuHal<risc0_core::field::baby_bear::BabyBear>;

        /// (rows, cols, queries): the po2-20 tree widths at a small row count, a single column,
        /// tiny trees whose paths end at the top layer, and a FRI-round-like shape.
        const SHAPES: &[(usize, usize, usize)] = &[
            (1 << 12, 211, 50),
            (1 << 12, 103, 50),
            (1 << 12, 16, 50),
            (1 << 10, 1, 50),
            (1 << 8, 64, 50),
            (4, 3, 2),
            (1, 5, 1),
            (1 << 6, 7, 64),
        ];

        fn matrix(rng: &mut SmallRng, n: usize) -> Vec<BabyBearElem> {
            (0..n).map(|_| BabyBearElem::random(&mut *rng)).collect()
        }

        /// Query rows with the edges and a duplicate.
        fn rows(rng: &mut SmallRng, row_size: usize, n: usize) -> Vec<usize> {
            let mut out: Vec<usize> = (0..n).map(|_| rng.random_range(0..row_size)).collect();
            out[0] = 0;
            if n > 2 {
                out[1] = row_size - 1;
                out[2] = out[n - 1];
            }
            out
        }

        struct Trees {
            cpu: Vec<MerkleTreeProver<Cpu>>,
            metal: Vec<MerkleTreeProver<MetalHalPoseidon2>>,
            rows: Vec<Vec<usize>>,
        }

        fn build(cpu: &Cpu, metal: &MetalHalPoseidon2, seed: u64) -> Trees {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut t = Trees {
                cpu: Vec::new(),
                metal: Vec::new(),
                rows: Vec::new(),
            };
            for &(rows, cols, queries) in SHAPES {
                let data = matrix(&mut rng, rows * cols);
                let cm = cpu.copy_from_elem("matrix", &data);
                let mm = metal.copy_from_elem("matrix", &data);
                t.cpu
                    .push(MerkleTreeProver::new(cpu, &cm, rows, cols, queries));
                t.metal
                    .push(MerkleTreeProver::new(metal, &mm, rows, cols, queries));
                t.rows.push(self::rows(&mut rng, rows, queries));
            }
            t
        }

        fn many<H>(hal: &H, trees: &[MerkleTreeProver<H>], rows: &[Vec<usize>]) -> Vec<Vec<Opening>>
        where
            H: Hal<Elem = BabyBearElem>,
        {
            let jobs: Vec<(&MerkleTreeProver<H>, &[usize])> = trees
                .iter()
                .zip(rows)
                .map(|(t, r)| (t, r.as_slice()))
                .collect();
            MerkleTreeProver::batch_prove_many(hal, &jobs)
        }

        fn single<H>(
            hal: &H,
            trees: &[MerkleTreeProver<H>],
            rows: &[Vec<usize>],
        ) -> Vec<Vec<Opening>>
        where
            H: Hal<Elem = BabyBearElem>,
        {
            trees
                .iter()
                .zip(rows)
                .map(|(t, r)| t.batch_prove(hal, r))
                .collect()
        }

        /// The IOP words of `prove` for every query of every tree.
        fn proofs<H>(hal: &H, trees: &[MerkleTreeProver<H>], rows: &[Vec<usize>]) -> Vec<u32>
        where
            H: Hal<Elem = BabyBearElem, Field = risc0_core::field::baby_bear::BabyBear>,
        {
            let suite = Poseidon2HashSuite::new_suite();
            let mut iop = WriteIOP::new(suite.rng.as_ref());
            for (t, r) in trees.iter().zip(rows) {
                for &row in r {
                    t.prove(hal, &mut iop, row);
                }
            }
            iop.proof
        }

        fn check(cpu: &Cpu, metal: &MetalHalPoseidon2, t: &Trees, what: &str) {
            let want_many = many(cpu, &t.cpu, &t.rows);
            assert_eq!(
                many(metal, &t.metal, &t.rows),
                want_many,
                "{what}: batch_prove_many"
            );
            assert_eq!(single(cpu, &t.cpu, &t.rows), want_many, "cpu batch_prove");
            assert_eq!(
                single(metal, &t.metal, &t.rows),
                want_many,
                "{what}: batch_prove"
            );
            assert_eq!(
                proofs(metal, &t.metal, &t.rows),
                proofs(cpu, &t.cpu, &t.rows),
                "{what}: prove"
            );
        }

        #[test]
        fn query_openings_match_cpu() {
            // Not for the GPU: keeps `query_openings_match_cpu_knob_off`'s env change out.
            let _lock = gpu_test_lock();
            let cpu = CpuHal::new(Poseidon2HashSuite::new_suite());
            let metal = MetalHalPoseidon2::new();
            for seed in 0..3 {
                let t = build(&cpu, &metal, 0x5a11 + seed);
                check(&cpu, &metal, &t, "query_par");
            }
        }

        /// `RISC0_METAL_QUERY_PAR=0` (the revert knob) gives the same openings. Takes the GPU test
        /// lock: it changes process-wide state (plan 8.7).
        #[test]
        fn query_openings_match_cpu_knob_off() {
            let _lock = gpu_test_lock();
            let cpu = CpuHal::new(Poseidon2HashSuite::new_suite());
            let metal = MetalHalPoseidon2::new();
            let t = build(&cpu, &metal, 0x5a21);
            std::env::set_var("RISC0_METAL_QUERY_PAR", "0");
            let off = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check(&cpu, &metal, &t, "knob off")
            }));
            std::env::remove_var("RISC0_METAL_QUERY_PAR");
            if let Err(e) = off {
                std::panic::resume_unwind(e);
            }
            check(&cpu, &metal, &t, "knob back on");
        }

        /// The digest gather around its serial / parallel switch and chunk boundaries, on a
        /// slice of a buffer (offset views), against `get_at`.
        #[test]
        fn batch_get_digest_at_matches_get_at() {
            use crate::hal::Buffer;
            let metal = MetalHalPoseidon2::new();
            let mut rng = SmallRng::seed_from_u64(0x5a31);
            let n = 1000;
            let words: Vec<u32> = (0..n * 8).map(|_| rng.random()).collect();
            let digests: Vec<crate::core::digest::Digest> = words
                .chunks(8)
                .map(|w| crate::core::digest::Digest::new(w.try_into().unwrap()))
                .collect();
            let buf = metal.copy_from_digest("digests", &digests);
            let view = buf.slice(7, n - 9);
            for len in [0, 1, 31, 32, 63, 64, 65, 97, 500, 2048] {
                let idx: Vec<usize> = (0..len).map(|_| rng.random_range(0..view.size())).collect();
                let want: Vec<_> = idx.iter().map(|&i| digests[7 + i]).collect();
                assert_eq!(metal.batch_get_digest_at(&view, &idx), want, "len {len}");
            }
        }

        #[test]
        #[should_panic]
        fn batch_get_digest_at_out_of_range_panics() {
            let metal = MetalHalPoseidon2::new();
            let buf = metal.copy_from_digest("digests", &[crate::core::digest::Digest::ZERO; 100]);
            let idx: Vec<usize> = (0..99).chain([100]).collect();
            metal.batch_get_digest_at(&buf, &idx);
        }
    }
}
