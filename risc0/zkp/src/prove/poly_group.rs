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
    pub coeffs: H::Buffer<H::Elem>,
    pub count: usize,
    pub evaluated: Option<H::Buffer<H::Elem>>,
    pub merkle: MerkleTreeProver<H>,
}

impl<H: Hal> Clone for PolyGroup<H> {
    fn clone(&self) -> Self {
        Self {
            coeffs: self.coeffs.clone(),
            count: self.count,
            evaluated: self.evaluated.clone(),
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

    /// Reconstruct the evaluated buffer from coefficients and restore it
    /// into merkle.matrix for batch_prove. Costs 2× batch_bit_reverse +
    /// 1× batch_expand_into_evaluate_ntt.
    pub fn restore_evaluated(&mut self, hal: &H) {
        if self.merkle.matrix_is_some() {
            return; // Already restored or never released
        }
        let size = self.coeffs.size() / self.count;
        let domain = size * INV_RATE;
        let evaluated = hal.alloc_elem("evaluated", self.count * domain);
        // Coefficients are bit-reversed; undo before expand+NTT
        hal.batch_bit_reverse(&self.coeffs, self.count);
        hal.batch_expand_into_evaluate_ntt(&evaluated, &self.coeffs, self.count, log2_ceil(INV_RATE));
        // Re-bit-reverse coefficients (other code expects them bit-reversed)
        hal.batch_bit_reverse(&self.coeffs, self.count);
        self.merkle.set_matrix(evaluated);
    }
}

impl<H: Hal> PolyGroup<H> {
    pub fn new(
        hal: &H,
        coeffs: H::Buffer<H::Elem>,
        count: usize,
        size: usize,
        name: &'static str,
    ) -> Self {
        scope_with!("poly_group({})", name);
        assert_eq!(coeffs.size(), count * size);
        let domain = size * INV_RATE;
        let evaluated = hal.alloc_elem("evaluated", count * domain);
        let t0 = std::time::Instant::now();
        hal.batch_expand_into_evaluate_ntt(&evaluated, &coeffs, count, log2_ceil(INV_RATE));
        let t_ntt = t0.elapsed();
        hal.batch_bit_reverse(&coeffs, count);
        let t_bitrev = t0.elapsed();
        let merkle = MerkleTreeProver::new(hal, &evaluated, domain, count, QUERIES);
        let t_merkle = t0.elapsed();
        if std::env::var_os("RISC0_VERBOSE").is_some() {
            eprintln!(
                "[poly_group({name})] count={count} size={size} domain={domain} ntt={:.1}ms bitrev={:.1}ms merkle={:.1}ms total={:.1}ms",
                t_ntt.as_secs_f64() * 1000.0,
                (t_bitrev - t_ntt).as_secs_f64() * 1000.0,
                (t_merkle - t_bitrev).as_secs_f64() * 1000.0,
                t_merkle.as_secs_f64() * 1000.0,
            );
        }
        PolyGroup {
            coeffs,
            count,
            evaluated: Some(evaluated),
            merkle,
        }
    }
}
