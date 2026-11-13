// Copyright 2024 RISC Zero, Inc.
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

use std::rc::Rc;

use risc0_core::scope_with;

use crate::{
    core::log2_ceil,
    hal::{Buffer, Hal},
    prove::merkle::MerkleTreeProver,
    INV_RATE, QUERIES,
};

/// A PolyGroup represents a group of polynomials, all of the same maximum
/// degree, as well as the evaluation of those polynomials over some domain that
/// is larger than that degree by some invRate. Additionally, it includes a
/// dense Merkle tree, where each entry is a single point of the domain, and the
/// leaf hash is a simple linear hash of all of the values at that point.  That
/// is, if we have 100 polynomials evaluated on 2^16 points, the merkle tree has
/// 2^16 entries, each being a hash of 100 values.  The size of the domain is
/// always a power of 2 so that we can use NTTs.
///
/// The primary purpose of the PolyGroup is for use in the DEEP-ALI protocol,
/// which basically needs 4 methods during proof generation, specifically we
/// need to: 1) Resolve queries (i.e. make MerkleColProofs)
/// 2) Do evaluation of the polynomials at 'randomly' chosen points
/// 3) Mix the polynomials via a random set of linear coefficients
/// 4) Access the raw values in the evaluation domain to 'evaluate' the
///    constraint polynomial
///
/// The poly group holds 3 buffers:
/// 1) The per-polynomial coefficients, used for evaluation + mixing
/// 2) The points evaluated on the domain in question (for the 'col' part of
///    merkle proofs)
/// 3) The Merkle tree itself.
///
/// PolyGroups are constructed from two basic sources: steps of a computations,
/// and a single higher degree polynomial that has been split into lower degree
/// parts.  In the case of computations, the resulting steps must be padded
/// (possibly with randomized data), which is presumed to be done by the caller.
/// The constructor additionally 'shifts' the polynomial so that f(x) -> f(3*x),
/// which means that the normal NTT evaluation domain does not reveal anything
/// about the original datapoints (i.e. is zero knowledge) so long as there is
/// sufficient randomized padding.
pub struct PolyGroup<H: Hal> {
    /// Coefficients of the polynomials. At po2>=22 on tight-VRAM GPUs, this
    /// buffer may be temporarily spilled to host memory between commit time
    /// and the finalize_complete eval_u phase (see `release_coeffs` /
    /// `restore_coeffs`). In that case the device buffer is freed and the
    /// data lives in `coeffs_host`; callers MUST invoke `restore_coeffs`
    /// before touching this field.
    pub coeffs: Option<H::Buffer<H::Elem>>,
    pub coeffs_host: Option<Rc<Vec<H::Elem>>>,
    pub coeffs_name: &'static str,
    pub count: usize,
    pub evaluated: Option<H::Buffer<H::Elem>>,
    /// Host-side mirror of `evaluated`, populated by `release_evaluated_to_host`.
    /// Restored to device via `restore_evaluated_from_host`.
    pub evaluated_host: Option<Rc<Vec<H::Elem>>>,
    pub merkle: MerkleTreeProver<H>,
}

impl<H: Hal> Clone for PolyGroup<H> {
    fn clone(&self) -> Self {
        Self {
            coeffs: self.coeffs.clone(),
            coeffs_host: self.coeffs_host.clone(),
            coeffs_name: self.coeffs_name,
            count: self.count,
            evaluated: self.evaluated.clone(),
            evaluated_host: self.evaluated_host.clone(),
            merkle: self.merkle.clone(),
        }
    }
}

impl<H: Hal> PolyGroup<H> {
    /// Release evaluated buffers (both PolyGroup.evaluated and merkle.matrix)
    /// to free GPU memory. On 16GB GPUs at po2=21, this frees ~10.5 GB.
    /// Call restore_evaluated() before FRI batch_prove to reconstruct.
    pub fn release_evaluated(&mut self) {
        self.evaluated = None;
        self.merkle.release_matrix();
    }

    /// Copy evaluated to host then release the device buffers. Heavier than
    /// `release_evaluated` (3.4-13 GB D2H) but avoids re-doing expand_NTT in
    /// `restore_evaluated`. Use when coeffs is ALSO spilled (at po2>=22) —
    /// in that case host→device of coeffs + NTT restore is slower than
    /// host→device of evaluated directly. Pair with `restore_evaluated_from_host`.
    pub fn release_evaluated_to_host(&mut self, hal: &H) {
        if self.evaluated_host.is_some() {
            // Already spilled; just release device side if present.
            self.evaluated = None;
            self.merkle.release_matrix();
            return;
        }
        // Copy to host first (while ev is still live), then drop ALL device refs
        // in quick succession so the RawBuffer Rc count hits 0 and the VMM /
        // pool path can actually reclaim physical VRAM.
        let host: Vec<H::Elem>;
        {
            let ev = match self.evaluated.as_ref() {
                Some(e) => e,
                None => return,
            };
            host = ev.to_vec();
            ev.set_bypass_pool();
        }
        // Now drop both device references (self.evaluated + merkle.matrix).
        self.evaluated = None;
        self.merkle.release_matrix();
        self.evaluated_host = Some(Rc::new(host));
        hal.trim_device_memory();
    }

    /// Restore evaluated from host memory. Also stores it as `merkle.matrix`
    /// so FRI batch_prove can read it for query samples.
    pub fn restore_evaluated_from_host(&mut self, hal: &H) {
        if self.merkle.matrix_is_some() && self.evaluated.is_some() {
            return;
        }
        let host = self
            .evaluated_host
            .take()
            .expect("evaluated_host empty; nothing to restore");
        // Maximize contiguous VRAM before this large alloc: trim the driver
        // mempool (sppark scratch) and drain our own reuse pool. The restored
        // buffer can be multi-GB and contention matters at po2>=22.
        hal.trim_device_memory();
        let dev = hal.copy_from_elem("evaluated", host.as_slice());
        self.merkle.set_matrix(dev.clone());
        self.evaluated = Some(dev);
    }

    /// Spill coeffs from GPU to host memory. Saves up to 3.4 GB per group at
    /// po2=22 during merkle construction + eval_check. Safe to call repeatedly.
    pub fn release_coeffs(&mut self) {
        if self.coeffs_host.is_some() {
            return;
        }
        let coeffs = self
            .coeffs
            .take()
            .expect("coeffs already released");
        let host: Vec<H::Elem> = coeffs.to_vec();
        // coeffs goes out of scope here -> device buffer freed (no other Rc refs).
        self.coeffs_host = Some(Rc::new(host));
        drop(coeffs);
    }

    /// Restore coeffs from host back to GPU. Safe to call if already on device.
    pub fn restore_coeffs(&mut self, hal: &H) {
        if self.coeffs.is_some() {
            return;
        }
        let host = self
            .coeffs_host
            .take()
            .expect("coeffs_host is empty; nothing to restore");
        let host_slice: &[H::Elem] = host.as_slice();
        let dev = hal.copy_from_elem(self.coeffs_name, host_slice);
        self.coeffs = Some(dev);
    }

    /// Reconstruct the evaluated buffer from coefficients and restore it
    /// into merkle.matrix for batch_prove. Costs 2× batch_bit_reverse +
    /// 1× batch_expand_into_evaluate_ntt.
    pub fn restore_evaluated(&mut self, hal: &H) {
        if self.merkle.matrix_is_some() {
            return; // Already restored or never released
        }
        self.restore_coeffs(hal);
        let coeffs = self.coeffs.as_ref().expect("coeffs must be present");
        let size = coeffs.size() / self.count;
        let domain = size * INV_RATE;
        let evaluated = hal.alloc_elem("evaluated", self.count * domain);
        // Coefficients are bit-reversed; undo before expand+NTT
        hal.batch_bit_reverse(coeffs, self.count);
        hal.batch_expand_into_evaluate_ntt(&evaluated, coeffs, self.count, log2_ceil(INV_RATE));
        // Re-bit-reverse coefficients (other code expects them bit-reversed)
        hal.batch_bit_reverse(coeffs, self.count);
        self.merkle.set_matrix(evaluated);
    }
}

/// Host copy of a committed [PolyGroup] (C13): the coefficients and the whole Merkle tree.
/// [PolyGroup::from_host] rebuilds the group on any HAL of the same type without hashing (the
/// evaluated domain is re-expanded from the coefficients, as [PolyGroup::restore_evaluated]
/// does). Used to keep a recursion program's committed ctrl group across the HALs that a
/// long-lived prover drops between jobs.
#[derive(Clone)]
pub struct PolyGroupHost<E> {
    coeffs: Vec<E>,
    count: usize,
    name: &'static str,
    params: crate::merkle::MerkleTreeParams,
    nodes: Vec<crate::core::digest::Digest>,
    root: crate::core::digest::Digest,
    top_nodes: Vec<crate::core::digest::Digest>,
}

impl<E> PolyGroupHost<E> {
    /// Host bytes held.
    pub fn bytes(&self) -> usize {
        self.coeffs.len() * std::mem::size_of::<E>()
            + (self.nodes.len() + self.top_nodes.len())
                * std::mem::size_of::<crate::core::digest::Digest>()
    }

    /// The Merkle root of the group.
    pub fn root(&self) -> &crate::core::digest::Digest {
        &self.root
    }

    /// Whether two copies hold the same group (coefficients, every tree node, root, top layer).
    pub fn same_as(&self, other: &Self) -> bool
    where
        E: PartialEq,
    {
        self.count == other.count
            && self.params.row_size == other.params.row_size
            && self.params.col_size == other.params.col_size
            && self.params.top_size == other.params.top_size
            && self.root == other.root
            && self.coeffs == other.coeffs
            && self.nodes == other.nodes
            && self.top_nodes == other.top_nodes
    }
}

impl<H: Hal> PolyGroup<H> {
    /// Host bytes [Self::to_host] would copy (its [PolyGroupHost::bytes]), without copying.
    pub fn host_bytes(&self) -> usize {
        let coeffs = match (&self.coeffs, &self.coeffs_host) {
            (Some(c), _) => c.size(),
            (None, Some(h)) => h.len(),
            (None, None) => 0,
        };
        coeffs * std::mem::size_of::<H::Elem>() + self.merkle.host_bytes()
    }

    /// Copy the group's coefficients and Merkle tree to the host ([PolyGroupHost]).
    pub fn to_host(&self) -> PolyGroupHost<H::Elem> {
        let coeffs = match (&self.coeffs, &self.coeffs_host) {
            (Some(c), _) => c.to_vec(),
            (None, Some(h)) => h.as_ref().clone(),
            (None, None) => panic!("PolyGroup::to_host: no coefficients"),
        };
        let (params, nodes, root, top_nodes) = self.merkle.to_host();
        PolyGroupHost {
            coeffs,
            count: self.count,
            name: self.coeffs_name,
            params,
            nodes,
            root,
            top_nodes,
        }
    }

    /// Rebuild a group [Self::to_host] copied, on `hal`: upload the coefficients and the tree,
    /// and re-expand the evaluated domain from the coefficients (bit-reverse, expand + NTT,
    /// bit-reverse back: what [Self::new] computed from them). No Merkle hashing.
    pub fn from_host(hal: &H, host: &PolyGroupHost<H::Elem>) -> Self {
        scope_with!("poly_group_from_host({})", host.name);
        let count = host.count;
        let coeffs = hal.copy_from_elem(host.name, &host.coeffs);
        let size = host.coeffs.len() / count;
        let domain = size * INV_RATE;
        assert_eq!(host.params.row_size, domain);
        assert_eq!(host.params.col_size, count);
        let evaluated = hal.alloc_elem("evaluated", count * domain);
        hal.batch_bit_reverse(&coeffs, count);
        hal.batch_expand_into_evaluate_ntt(&evaluated, &coeffs, count, log2_ceil(INV_RATE));
        hal.batch_bit_reverse(&coeffs, count);
        let merkle = MerkleTreeProver::from_host(
            hal,
            &evaluated,
            host.params,
            &host.nodes,
            host.root,
            &host.top_nodes,
        );
        PolyGroup {
            coeffs: Some(coeffs),
            coeffs_host: None,
            coeffs_name: host.name,
            count,
            evaluated: Some(evaluated),
            evaluated_host: None,
            merkle,
        }
    }
}

/// A GPU HAL whose buffers live in host-shared memory (Apple Metal): host spills of device buffers
/// save nothing there. The CPU HAL also reports unified memory, but no GPU memory size.
fn unified_memory_gpu<H: Hal>(hal: &H) -> bool {
    hal.has_unified_memory() && hal.gpu_total_memory() != usize::MAX
}

impl<H: Hal> PolyGroup<H> {
    pub fn new(
        hal: &H,
        coeffs: H::Buffer<H::Elem>,
        count: usize,
        size: usize,
        name: &'static str,
    ) -> Self {
        Self::new_impl(hal, coeffs, None, false, count, size, name)
    }

    /// Build from coefficients (natural order) and their already-computed LDE, as
    /// [`Hal::commit_coeffs_lde`] produces them.
    pub fn new_with_evaluated(
        hal: &H,
        coeffs: H::Buffer<H::Elem>,
        evaluated: H::Buffer<H::Elem>,
        count: usize,
        size: usize,
        name: &'static str,
    ) -> Self {
        Self::new_impl(hal, coeffs, Some(evaluated), false, count, size, name)
    }

    /// [Self::new_with_evaluated] for coefficients still in bit-reversed order, as
    /// [`Hal::commit_coeffs_lde_bit_reversed`] leaves them (Z14): the Merkle hash bit-reverses
    /// them.
    pub fn new_with_evaluated_bit_reversed(
        hal: &H,
        coeffs: H::Buffer<H::Elem>,
        evaluated: H::Buffer<H::Elem>,
        count: usize,
        size: usize,
        name: &'static str,
    ) -> Self {
        Self::new_impl(hal, coeffs, Some(evaluated), true, count, size, name)
    }

    /// `coeffs_bit_reversed`: `coeffs` (with `pre_evaluated`) is still bit-reversed. Without
    /// `pre_evaluated` it always is (the expand NTT reads that order).
    fn new_impl(
        hal: &H,
        coeffs: H::Buffer<H::Elem>,
        pre_evaluated: Option<H::Buffer<H::Elem>>,
        coeffs_bit_reversed: bool,
        count: usize,
        size: usize,
        name: &'static str,
    ) -> Self {
        scope_with!("poly_group({})", name);
        assert_eq!(coeffs.size(), count * size);
        let domain = size * INV_RATE;
        let t0 = std::time::Instant::now();
        // See the coeffs spill below.
        let spill_enabled = size >= (1 << 22) && !unified_memory_gpu(hal);
        let coeffs_bytes = coeffs.size() * std::mem::size_of::<H::Elem>();
        let spill_threshold: usize = 512 << 20;
        let spill = spill_enabled && coeffs_bytes >= spill_threshold;
        // Z14: bit-reverse the coefficients inside the Merkle row hash when the HAL fuses the two
        // (not when they go to the host before it).
        let defer_bit_reverse =
            !spill && hal.fuses_bit_reverse_into_hash(domain, count, log2_ceil(size));
        let (evaluated, bit_reverse_pending) = match pre_evaluated {
            Some(e) => {
                assert_eq!(e.size(), count * domain);
                (e, coeffs_bit_reversed)
            }
            None => {
                let evaluated = hal.alloc_elem("evaluated", count * domain);
                hal.batch_expand_into_evaluate_ntt(&evaluated, &coeffs, count, log2_ceil(INV_RATE));
                (evaluated, true)
            }
        };
        let t_ntt = t0.elapsed();
        let merkle_bit_reverse = bit_reverse_pending && defer_bit_reverse;
        if bit_reverse_pending && !merkle_bit_reverse {
            hal.batch_bit_reverse(&coeffs, count);
        }
        let t_bitrev = t0.elapsed();

        // Spill coeffs to host BEFORE merkle alloc if large enough. At po2=22
        // this frees 3.4 GB (DATA) / 1.65 GB (ACCUM) needed to fit the merkle
        // `nodes` allocation (1 GB at po2=22) in tight-VRAM GPUs. The device
        // buffer is dropped here; caller must invoke restore_coeffs() before
        // re-using coeffs (prover::finalize_complete handles this).
        // Gate coeffs spill on cycles>=4M (po2>=22). At po2<=21 the committed
        // PolyGroups plus witgen fit comfortably in 24 GB; spilling adds
        // H2D/D2H overhead and perturbs the driver mempool state in ways that
        // regress performance (and can OOM later stages via mempool churn).
        // Unified-memory GPU HALs (Metal) skip it: the "host" copy would sit in the same physical
        // memory as the device buffer and save nothing (same test as prover.rs's
        // `unified_memory_gpu`, which gates the evaluated spill).
        let (coeffs_dev, coeffs_host) = if spill {
            let host: Vec<H::Elem> = coeffs.to_vec();
            // Mark the underlying device buffer to bypass the reuse pool on
            // drop — we want the VRAM returned to the driver immediately so
            // the next large allocation (e.g. this group's merkle nodes, or
            // the next group's coeffs) can physically succeed.
            coeffs.set_bypass_pool();
            drop(coeffs);
            // Trim the driver's stream-ordered mempool in case sppark's
            // expand_NTT scratch (also multi-GB at po2>=22) is still held.
            hal.trim_device_memory();
            (None, Some(Rc::new(host)))
        } else {
            (Some(coeffs), None)
        };

        let merkle = MerkleTreeProver::new_bit_reverse(
            hal,
            &evaluated,
            domain,
            count,
            QUERIES,
            coeffs_dev
                .as_ref()
                .filter(|_| merkle_bit_reverse)
                .map(|c| (c, count)),
        );
        let t_merkle = t0.elapsed();
        if std::env::var_os("RISC0_VERBOSE").is_some() {
            eprintln!(
                "[poly_group({name})] count={count} size={size} domain={domain} ntt={:.1}ms bitrev={:.1}ms merkle={:.1}ms total={:.1}ms spill={}",
                t_ntt.as_secs_f64() * 1000.0,
                (t_bitrev - t_ntt).as_secs_f64() * 1000.0,
                (t_merkle - t_bitrev).as_secs_f64() * 1000.0,
                t_merkle.as_secs_f64() * 1000.0,
                coeffs_host.is_some(),
            );
        }
        PolyGroup {
            coeffs: coeffs_dev,
            coeffs_host,
            coeffs_name: name,
            count,
            evaluated: Some(evaluated),
            evaluated_host: None,
            merkle,
        }
    }
}
