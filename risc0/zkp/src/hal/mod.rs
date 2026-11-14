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

//! Hardware Abstraction Layer (HAL) for accelerating the ZKP system.

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod dual;
#[cfg(feature = "rocm")]
pub mod hip;
#[cfg(feature = "intel")]
pub mod intel;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod metal;

#[cfg(all(feature = "metal", not(any(target_os = "macos", target_os = "ios"))))]
compile_error!("Feature 'metal' is only supported on macOS and iOS targets.");

use std::{
    fmt::Debug,
    sync::{Mutex, OnceLock},
};

use risc0_core::{
    field::{Elem, ExtElem, Field, RootsOfUnity},
    scope,
};

use crate::{
    core::{digest::Digest, hash::HashSuite, poly::poly_divide},
    INV_RATE,
};

pub trait Buffer<T>: Clone {
    fn name(&self) -> &'static str;

    fn size(&self) -> usize;

    fn slice(&self, offset: usize, size: usize) -> Self;

    fn get_at(&self, idx: usize) -> T;

    fn view<F: FnOnce(&[T])>(&self, f: F);

    fn view_mut<F: FnOnce(&mut [T])>(&self, f: F);

    fn to_vec(&self) -> Vec<T>;

    /// Opt-in hint: when all clones of this Buffer are dropped, release the
    /// underlying device memory directly to the driver instead of caching it
    /// in a reuse pool. HALs without a reuse pool can leave this as the
    /// default no-op.
    fn set_bypass_pool(&self) {}
}

/// One Merkle tree's query gathers for [Hal::gather_multi].
pub struct GatherJob<'a, H: Hal> {
    pub matrix: &'a H::Buffer<H::Elem>,
    pub col_size: usize,
    pub row_size: usize,
    pub rows: &'a [usize],
    pub nodes: &'a H::Buffer<Digest>,
    pub node_indices: &'a [usize],
}

pub trait Hal {
    type Field: Field<Elem = Self::Elem, ExtElem = Self::ExtElem>;
    type Elem: Elem + RootsOfUnity;
    type ExtElem: ExtElem<SubElem = Self::Elem>;
    type Buffer<T: Clone + Debug + PartialEq>: Buffer<T>;

    const CHECK_SIZE: usize = INV_RATE * Self::ExtElem::EXT_SIZE;

    fn has_unified_memory(&self) -> bool;

    /// Ask the HAL to return any cached / stream-ordered mempool memory back
    /// to the driver. Called at GPU-memory-pressure points (e.g. after a host
    /// spill at po2>=22) so the next large allocation can succeed.
    /// Default no-op for HALs without a reuse cache.
    fn trim_device_memory(&self) {}

    /// Returns the amount of free GPU memory in bytes.
    /// Returns `usize::MAX` for CPU HALs or when the information is unavailable.
    fn gpu_free_memory(&self) -> usize {
        usize::MAX
    }

    /// Returns the total GPU memory in bytes.
    /// Returns `usize::MAX` for CPU HALs or when the information is unavailable.
    fn gpu_total_memory(&self) -> usize {
        usize::MAX
    }

    fn get_hash_suite(&self) -> &HashSuite<Self::Field>;

    fn alloc_digest(&self, name: &'static str, size: usize) -> Self::Buffer<Digest>;
    fn alloc_elem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::Elem>;
    fn alloc_extelem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem>;
    fn alloc_u32(&self, name: &'static str, size: usize) -> Self::Buffer<u32>;

    fn alloc_elem_init(
        &self,
        name: &'static str,
        size: usize,
        value: Self::Elem,
    ) -> Self::Buffer<Self::Elem> {
        let buffer = self.alloc_elem(name, size);
        buffer.view_mut(|slice| {
            slice.fill(value);
        });
        buffer
    }

    /// Hint that `alloc_elem_init` will soon be called with this `size` and `value` (e.g. by the
    /// next segment): a HAL may fill a buffer it holds in the background now and hand it out
    /// then. The default does nothing.
    fn prefill_elem(&self, _size: usize, _value: Self::Elem) {}

    /// [Self::prefill_elem] from a caller that knows its next phase will still allocate about
    /// `need` bytes before the buffer is taken (W7-R2): a HAL that sets aside memory may keep
    /// it only within the headroom left after that phase. The default ignores `need`.
    fn prefill_elem_within(&self, size: usize, value: Self::Elem, _need: usize) {
        self.prefill_elem(size, value)
    }

    fn alloc_extelem_zeroed(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem> {
        let buffer = self.alloc_extelem(name, size);
        buffer.view_mut(|slice| {
            slice.fill(Self::ExtElem::ZERO);
        });
        buffer
    }

    fn copy_from_digest(&self, name: &'static str, slice: &[Digest]) -> Self::Buffer<Digest>;
    fn copy_from_elem(&self, name: &'static str, slice: &[Self::Elem]) -> Self::Buffer<Self::Elem>;
    fn copy_from_extelem(
        &self,
        name: &'static str,
        slice: &[Self::ExtElem],
    ) -> Self::Buffer<Self::ExtElem>;
    fn copy_from_u32(&self, name: &'static str, slice: &[u32]) -> Self::Buffer<u32>;

    fn batch_expand_into_evaluate_ntt(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
        count: usize,
        expand_bits: usize,
    );

    fn batch_interpolate_ntt(&self, io: &Self::Buffer<Self::Elem>, count: usize);

    fn batch_bit_reverse(&self, io: &Self::Buffer<Self::Elem>, count: usize);

    fn batch_evaluate_any(
        &self,
        coeffs: &Self::Buffer<Self::Elem>,
        poly_count: usize,
        which: &Self::Buffer<u32>,
        xs: &Self::Buffer<Self::ExtElem>,
        out: &Self::Buffer<Self::ExtElem>,
    );

    fn zk_shift(&self, io: &Self::Buffer<Self::Elem>, count: usize);

    fn batch_interpolate_ntt_zk_shift(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        self.batch_interpolate_ntt(io, count);
        self.zk_shift(io, count);
    }

    /// True if `commit_coeffs_lde` is implemented for columns of `2^po2` rows.
    fn has_commit_coeffs_lde(&self, _po2: usize) -> bool {
        false
    }

    /// Fused commit: `coeffs` = the natural-order zk-shifted coefficients of `witness`
    /// (what `make_coeffs` + `PolyGroup::new`'s bit-reverse produce) and `evaluated` = the
    /// INV_RATE LDE (what `batch_expand_into_evaluate_ntt` produces). `witness` is unchanged.
    fn commit_coeffs_lde(
        &self,
        _coeffs: &Self::Buffer<Self::Elem>,
        _evaluated: &Self::Buffer<Self::Elem>,
        _witness: &Self::Buffer<Self::Elem>,
        _count: usize,
    ) {
        unimplemented!()
    }

    /// Z14: true if [Self::hash_rows_bit_reverse] over a `rows`-row matrix bit-reverses `cols`
    /// columns of `2^lg` inside the hash kernel (no separate pass). Committing code then leaves the
    /// coefficients bit-reversed until the Merkle hash ([Self::commit_coeffs_lde_bit_reversed]).
    fn fuses_bit_reverse_into_hash(&self, _rows: usize, _cols: usize, _lg: usize) -> bool {
        false
    }

    /// Z14: [Self::hash_rows] of `matrix` into `output` and [Self::batch_bit_reverse] of the
    /// `count` columns of `coeffs` (a buffer the hash does not read), in either order.
    fn hash_rows_bit_reverse(
        &self,
        output: &Self::Buffer<Digest>,
        matrix: &Self::Buffer<Self::Elem>,
        coeffs: &Self::Buffer<Self::Elem>,
        count: usize,
    ) {
        self.hash_rows(output, matrix);
        self.batch_bit_reverse(coeffs, count);
    }

    /// Z14: [Self::commit_coeffs_lde], but `coeffs` is left bit-reversed (as the inverse NTT
    /// produces it); the caller bit-reverses it, e.g. with [Self::hash_rows_bit_reverse]. Only
    /// worth calling where [Self::fuses_bit_reverse_into_hash] holds.
    fn commit_coeffs_lde_bit_reversed(
        &self,
        coeffs: &Self::Buffer<Self::Elem>,
        evaluated: &Self::Buffer<Self::Elem>,
        witness: &Self::Buffer<Self::Elem>,
        count: usize,
    ) {
        self.commit_coeffs_lde(coeffs, evaluated, witness, count);
        self.batch_bit_reverse(coeffs, count);
    }

    /// W7-R2: the caller owns `witness` and will not read it again after its next
    /// `Prover::commit_group`, which may then commit it in place (the witness becomes the group's
    /// coefficient buffer; see [Self::commit_coeffs_lde_in_place]). Only the buffer donated last
    /// is remembered. The default ignores it (every commit allocates its own coefficients).
    fn donate_to_commit(&self, _witness: &Self::Buffer<Self::Elem>) {}

    /// W7-R2: whether `witness` is the buffer donated by [Self::donate_to_commit]; a match
    /// consumes the donation. The default: never.
    fn take_commit_donation(&self, _witness: &Self::Buffer<Self::Elem>) -> bool {
        false
    }

    /// W7-R2: commit `io` in place: `io` holds the witness on entry and its zk-shifted
    /// coefficients on exit, `evaluated` gets the INV_RATE LDE. With `bit_reversed` the
    /// coefficients stay in bit-reversed order (as [Self::commit_coeffs_lde_bit_reversed]
    /// leaves them), else natural order (as [Self::commit_coeffs_lde]). Saves the separate
    /// coefficient buffer (the witness's size). The default runs the unfused steps
    /// (`make_coeffs` + `PolyGroup::new`): inverse NTT with zk shift in place, expand + NTT,
    /// bit reversal; every HAL computes the same values either way.
    fn commit_coeffs_lde_in_place(
        &self,
        io: &Self::Buffer<Self::Elem>,
        evaluated: &Self::Buffer<Self::Elem>,
        count: usize,
        bit_reversed: bool,
    ) {
        self.batch_interpolate_ntt_zk_shift(io, count);
        self.batch_expand_into_evaluate_ntt(
            evaluated,
            io,
            count,
            crate::core::log2_ceil(crate::INV_RATE),
        );
        if !bit_reversed {
            self.batch_bit_reverse(io, count);
        }
    }

    /// `dst = interpolate_zk_shift(src)`, leaving `src` unchanged.
    fn batch_interpolate_ntt_zk_shift_from(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        count: usize,
    ) {
        self.eltwise_copy_elem(dst, src);
        self.batch_interpolate_ntt_zk_shift(dst, count);
    }

    #[allow(clippy::too_many_arguments)]
    fn mix_poly_coeffs(
        &self,
        out: &Self::Buffer<Self::ExtElem>,
        mix_start: &Self::ExtElem,
        mix: &Self::ExtElem,
        input: &Self::Buffer<Self::Elem>,
        combos: &Self::Buffer<u32>,
        input_size: usize,
        count: usize,
    );

    /// True if [Hal::mix_poly_coeffs_fused] is implemented (Metal, Z15).
    fn has_mix_poly_coeffs_fused(&self) -> bool {
        false
    }

    /// Fused FRI mixing (Z15): `out` (`count * n` extension elements, any initial contents)
    /// becomes what zeroing it and then calling, for each `(input, combo_ids)` of `inputs` in
    /// order, `mix_poly_coeffs(out, mix^k, mix, input, combo_ids, combo_ids.len(), count)`
    /// produces, where `k` is the number of inputs of the earlier groups.
    #[allow(clippy::type_complexity)]
    fn mix_poly_coeffs_fused(
        &self,
        _out: &Self::Buffer<Self::ExtElem>,
        _mix: &Self::ExtElem,
        _inputs: &[(&Self::Buffer<Self::Elem>, &[u32])],
        _count: usize,
    ) {
        unimplemented!()
    }

    fn eltwise_add_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input1: &Self::Buffer<Self::Elem>,
        input2: &Self::Buffer<Self::Elem>,
    );

    fn eltwise_sum_extelem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::ExtElem>,
    );

    fn eltwise_copy_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
    );

    #[allow(clippy::too_many_arguments)]
    fn eltwise_copy_elem_slice(
        &self,
        into: &Self::Buffer<Self::Elem>,
        from: &[Self::Elem],
        from_rows: usize,
        from_cols: usize,
        from_offset: usize,
        from_stride: usize,
        into_offset: usize,
        into_stride: usize,
    );

    fn eltwise_zeroize_elem(&self, elems: &Self::Buffer<Self::Elem>);

    fn fri_fold(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
        mix: &Self::ExtElem,
    );

    fn hash_rows(&self, output: &Self::Buffer<Digest>, matrix: &Self::Buffer<Self::Elem>);

    fn hash_fold(&self, io: &Self::Buffer<Digest>, input_size: usize, output_size: usize);

    fn hash_fold_tree(&self, io: &Self::Buffer<Digest>, layers: usize) {
        for i in (0..layers).rev() {
            let layer_size = 1 << i;
            self.hash_fold(io, layer_size * 2, layer_size);
        }
    }

    fn gather_sample(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        idx: usize,
        size: usize,
        stride: usize,
    );

    /// Gather `size` strided samples for each index in `indices` into consecutive slices of
    /// `dst` (`dst[q * size + i] = src[idxs[q] + i * stride]`). Default: one gather_sample per
    /// index. CUDA overrides this with a single launch.
    fn gather_samples(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        indices: &[usize],
        size: usize,
        stride: usize,
    ) {
        for (q, &idx) in indices.iter().enumerate() {
            self.gather_sample(&dst.slice(q * size, size), src, idx, size, stride);
        }
    }

    /// Batch-read digests at scattered indices. Default impl uses individual get_at() calls.
    /// CUDA overrides this with a gather kernel + single D2H transfer.
    fn batch_get_digest_at(&self, buf: &Self::Buffer<Digest>, indices: &[usize]) -> Vec<Digest> {
        indices.iter().map(|&idx| buf.get_at(idx)).collect()
    }

    /// The Merkle query gathers of several trees in one host round trip, where the HAL supports
    /// it (`None` otherwise): for each job, the `rows.len() * col_size` elements
    /// `matrix[row + i * row_size]` (query-major, as `gather_samples`) and the digests at
    /// `node_indices` (as `batch_get_digest_at`).
    #[allow(clippy::type_complexity)]
    fn gather_multi(
        &self,
        _jobs: &[GatherJob<'_, Self>],
    ) -> Option<Vec<(Vec<Self::Elem>, Vec<Digest>)>>
    where
        Self: Sized,
    {
        None
    }

    fn scatter(
        &self,
        into: &Self::Buffer<Self::Elem>,
        index: &[u32],
        offsets: &[u32],
        values: &[Self::Elem],
    );

    /// Scatter u32 values as 32-bit decompositions into the data buffer.
    /// `bit_data` contains packed (row, base_col, value) triples. Each triple
    /// writes 32 field elements: data[(base_col+i)*cycles + row] = (value>>i)&1.
    fn scatter_bits(&self, into: &Self::Buffer<Self::Elem>, bit_data: &[u32], cycles: u32) {
        let n = bit_data.len() / 3;
        if n == 0 {
            return;
        }
        let mut index = Vec::with_capacity(n + 1);
        let mut offsets = Vec::with_capacity(n * 32);
        let mut vals: Vec<Self::Elem> = Vec::with_capacity(n * 32);
        for i in 0..n {
            index.push((i * 32) as u32);
            let row = bit_data[i * 3];
            let base = bit_data[i * 3 + 1];
            let value = bit_data[i * 3 + 2];
            for bit in 0..32u32 {
                offsets.push((base + bit) * cycles + row);
                vals.push(Self::Elem::from_u64(((value >> bit) & 1) as u64));
            }
        }
        index.push((n * 32) as u32);
        self.scatter(into, &index, &offsets, &vals);
    }

    fn prefix_products(&self, io: &Self::Buffer<Self::ExtElem>);

    #[allow(clippy::too_many_arguments)]
    fn combos_prepare(
        &self,
        combos: &Self::Buffer<Self::ExtElem>,
        coeff_u: &[Self::ExtElem],
        combo_count: usize,
        cycles: usize,
        reg_sizes: &[u32],
        reg_combo_ids: &[u32],
        mix: &Self::ExtElem,
    ) {
        combos.view_mut(|combos| {
            scope!("combos_prepare", {
                let mut cur_pos = 0;
                let mut cur = Self::ExtElem::ONE;
                // Subtract the U coeffs from the combos
                for (reg_size, reg_combo_id) in reg_sizes.iter().zip(reg_combo_ids) {
                    let reg_size = *reg_size as usize;
                    let reg_combo_id = *reg_combo_id as usize;
                    for i in 0..reg_size {
                        combos[cycles * reg_combo_id + i] -= cur * coeff_u[cur_pos + i];
                    }
                    cur *= *mix;
                    cur_pos += reg_size;
                }
                // Subtract the final 'check' coefficients
                for _ in 0..Self::CHECK_SIZE {
                    combos[cycles * combo_count] -= cur * coeff_u[cur_pos];
                    cur_pos += 1;
                    cur *= *mix;
                }
            });
        });
    }

    fn combos_divide(
        &self,
        combos: &Self::Buffer<Self::ExtElem>,
        chunks: Vec<(usize, Vec<Self::ExtElem>)>,
        cycles: usize,
    ) {
        use rayon::prelude::*;

        scope!("combos_divide");

        combos.view_mut(|combos| {
            combos
                .par_chunks_exact_mut(cycles)
                .zip(chunks)
                .for_each(|(combo_slice, (i, pows))| {
                    for pow in pows {
                        let remainder = poly_divide(combo_slice, pow);
                        assert_eq!(remainder, Self::ExtElem::ZERO, "i: {i}");
                    }
                });
        });
    }
}

#[derive(Clone, Default)]
pub struct AccumPreflight {
    pub is_par_safe: Vec<u8>,
}

pub trait CircuitHal<H: Hal> {
    #[allow(clippy::too_many_arguments)]
    fn accumulate(
        &self,
        preflight: &AccumPreflight,
        ctrl: &H::Buffer<H::Elem>,
        io: &H::Buffer<H::Elem>,
        data: &H::Buffer<H::Elem>,
        mix: &H::Buffer<H::Elem>,
        accum: &H::Buffer<H::Elem>,
        steps: usize,
    );

    /// Compute check polynomial.
    fn eval_check(
        &self,
        check: &H::Buffer<H::Elem>,
        // Register groups, e.g. accum, code, data.  These should have one row for each cycle.
        groups: &[&H::Buffer<H::Elem>],
        // Globals.  These should have one row total.
        globals: &[&H::Buffer<H::Elem>],
        poly_mix: H::ExtElem,
        po2: usize,
        steps: usize,
    );

    /// Insert a GPU-side dependency so that subsequent operations on the main
    /// stream wait for eval_check (running on a separate stream) to complete.
    /// Default is no-op (for CPU HAL or when eval_check uses the same stream).
    fn eval_check_dep(&self) {}
}

pub fn tracker() -> &'static Mutex<MemoryTracker> {
    static ONCE: OnceLock<Mutex<MemoryTracker>> = OnceLock::new();
    ONCE.get_or_init(|| Mutex::new(MemoryTracker::default()))
}

#[derive(Debug, Default)]
pub struct MemoryTracker {
    pub total: isize,
    pub peak: isize,
}

impl MemoryTracker {
    pub fn reset(&mut self) {
        self.total = 0;
        self.peak = 0;
    }

    pub fn alloc(&mut self, size: usize) {
        self.total += size as isize;
        self.peak = self.peak.max(self.total);
    }

    pub fn free(&mut self, size: usize) {
        self.total -= size as isize;
    }
}

/// Reports every combo whose division by `(X - z)` left a nonzero remainder, for the GPU HALs'
/// batched `combos_divide`. `remainders` holds one entry per root, in `chunks` order.
///
/// The CPU HAL, and upstream's GPU HALs, assert that each remainder is zero: a nonzero one means the
/// combo polynomial does not vanish at the query point, so something upstream of this stage (the
/// witness, the coefficients, the evaluations, the check polynomial) is inconsistent and the seal
/// will not verify. Logged rather than asserted, so a bad proof still fails where callers already
/// handle it (verifying the segment) instead of panicking the prover; but it is named here, at the
/// stage it first shows.
#[cfg(any(feature = "cuda", feature = "rocm", feature = "metal"))]
#[cfg_attr(not(any(feature = "cuda", feature = "rocm")), allow(dead_code))]
pub(crate) fn report_nonzero_remainders<E: ExtElem>(chunks: &[(usize, Vec<E>)], remainders: &[E]) {
    let mut remainders = remainders.iter();
    for (combo, pows) in chunks {
        for pow in 0..pows.len() {
            if remainders.next().is_some_and(|r| *r != E::ZERO) {
                tracing::error!(
                    "combos_divide: combo {combo} has a nonzero remainder at root {pow}; this \
                     segment's proof will not verify"
                );
            }
        }
    }
}

#[cfg(test)]
#[allow(unused)]
mod testutil {
    use std::rc::Rc;

    use rand::RngCore;
    use risc0_core::field::{
        baby_bear::{BabyBearElem, BabyBearExtElem},
        Elem, ExtElem,
    };

    use super::{dual::DualHal, Hal};
    use crate::{
        core::digest::Digest,
        hal::{cpu::CpuHal, Buffer},
        FRI_FOLD, INV_RATE,
    };

    const COUNTS: [usize; 7] = [1, 9, 12, 1001, 1024, 1025, 1024 * 1024];
    const DATA_SIZE: usize = 224;

    fn generate_elem<H: Hal, R: RngCore>(hal: &H, rng: &mut R, size: usize) -> H::Buffer<H::Elem> {
        let values: Vec<H::Elem> = (0..size).map(|_| H::Elem::random(rng)).collect();
        hal.copy_from_elem("values", &values)
    }

    fn generate_extelem<H: Hal, R: RngCore>(
        hal: &H,
        rng: &mut R,
        size: usize,
    ) -> H::Buffer<H::ExtElem> {
        let values: Vec<H::ExtElem> = (0..size).map(|_| H::ExtElem::random(rng)).collect();
        hal.copy_from_extelem("values", &values)
    }

    pub(crate) fn batch_bit_reverse<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let steps = 1 << 12;
        let count = DATA_SIZE;
        let domain = steps * INV_RATE;
        let io_size = count * domain;

        let io = generate_elem(&hal, &mut rng, io_size);
        hal.batch_bit_reverse(&io, count);
    }

    pub(crate) fn batch_evaluate_any<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let eval_size = 865;
        let poly_count = 223;
        let steps = 1 << 16;
        let coeffs_size = steps * poly_count;

        let z = H::ExtElem::random(&mut rng);
        let z_pow = z.pow(H::ExtElem::EXT_SIZE);

        let coeffs = generate_elem(&hal, &mut rng, coeffs_size);
        let which = hal.copy_from_u32("which", &vec![0; eval_size]);
        let xs = hal.copy_from_extelem("xs", &vec![z_pow; eval_size]);
        let out = hal.alloc_extelem("out", eval_size);

        hal.batch_evaluate_any(&coeffs, poly_count, &which, &xs, &out);
    }

    pub(crate) fn batch_expand_into_evaluate_ntt<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let count = DATA_SIZE;
        let expand_bits = 2;
        let steps = 1 << 16;
        let domain = steps * INV_RATE;
        let input_size = count * steps;
        let output_size = count * domain;

        let input = generate_elem(&hal, &mut rng, input_size);
        let output = hal.alloc_elem("output", output_size);
        hal.batch_expand_into_evaluate_ntt(&output, &input, count, expand_bits);
    }

    /// Expands by 2 and by 1 instead of INV_RATE (4), as the Intel rv32im eval_check recovers the
    /// parts of the check polynomial that it evaluates on half or three quarters of the coset.
    pub(crate) fn batch_expand_into_evaluate_ntt_by_2<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let count = 4;
        let steps = 1 << 17;
        for expand_bits in [1, 0] {
            let domain = steps << expand_bits;
            let input = generate_elem(&hal, &mut rng, count * steps);
            let output = hal.alloc_elem("output", count * domain);
            hal.batch_expand_into_evaluate_ntt(&output, &input, count, expand_bits);
        }
    }

    pub(crate) fn batch_expand_into_evaluate_ntt_large<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));
        for (po2, count) in [(16, 5), (17, 3), (18, 3), (19, 2), (20, 3), (21, 2)] {
            let steps = 1 << po2;
            let input = generate_elem(&hal, &mut rng, count * steps);
            let output = hal.alloc_elem("output", count * steps * INV_RATE);
            hal.batch_expand_into_evaluate_ntt(&output, &input, count, 2);
            // a second call checks that the input was left unchanged
            hal.batch_expand_into_evaluate_ntt(&output, &input, count, 2);
        }
    }

    pub(crate) fn batch_interpolate_ntt<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let count = DATA_SIZE;
        let steps = 1 << 16;
        let domain = steps * INV_RATE;
        let io_size = count * domain;

        let io = generate_elem(&hal, &mut rng, io_size);
        hal.batch_interpolate_ntt(&io, count);
    }

    pub(crate) fn gather_sample<H: Hal>(hal: H) {
        let mut rng = rand::rng();
        let rows = 1000;
        let cols = 900;
        let idx = 400;
        let src_size = rows * cols;
        let src = hal.alloc_elem("src", src_size);
        let dst = hal.alloc_elem("dst", rows);
        src.view_mut(|buf| {
            for x in 0..cols {
                for y in 0..rows {
                    let value = H::Elem::random(&mut rng);
                    buf[y * cols + x] = value;
                }
            }
        });
        hal.gather_sample(&dst, &src, idx, rows, cols);
        src.view(|src| {
            dst.view(|dst| {
                for y in 0..rows {
                    assert_eq!(src[y * cols + idx], dst[y]);
                }
            });
        });
    }

    pub(crate) fn check_req<H: Hal>(hal: H) {
        let a = hal.alloc_elem("a", 10);
        let b = hal.alloc_elem("b", 20);
        hal.eltwise_add_elem(&a, &b, &b);
    }

    pub(crate) fn eltwise_add_elem<H: Hal>(hal_gpu: H) {
        for (x, count) in COUNTS.iter().enumerate() {
            let a = hal_gpu.alloc_elem("a", *count);
            let b = hal_gpu.alloc_elem("b", *count);
            let o = hal_gpu.alloc_elem("o", *count);
            let mut golden = Vec::with_capacity(*count);

            let mut rng = rand::rng();
            a.view_mut(|a| {
                b.view_mut(|b| {
                    assert_eq!(a.len(), b.len());
                    for i in 0..a.len() {
                        a[i] = H::Elem::random(&mut rng);
                        b[i] = H::Elem::random(&mut rng);
                    }
                    for i in 0..a.len() {
                        golden.push(a[i] + b[i]);
                    }
                });
            });

            hal_gpu.eltwise_add_elem(&o, &a, &b);

            o.view(|o| {
                for i in 0..o.len() {
                    assert_eq!(o[i], golden[i], "x: {x}, count: {count}, i: {i}");
                }
            });
        }
    }

    pub(crate) fn eltwise_copy_elem<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        for count in COUNTS {
            let input = generate_elem(&hal_gpu, &mut rng, count);
            let output = hal_gpu.alloc_elem("output", count);
            hal_gpu.eltwise_copy_elem(&output, &input);
            output.view(|output| {
                input.view(|input| assert_eq!(output, input));
            });
        }
    }

    pub(crate) fn eltwise_sum_extelem<H: Hal>(hal_gpu: H) {
        const COUNT: usize = 1024 * 1024;

        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let input = generate_extelem(&hal, &mut rng, COUNT);
        let output = hal.alloc_elem("output", COUNT);
        hal.eltwise_sum_extelem(&output, &input);
    }

    pub(crate) fn fri_fold<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));
        for count in COUNTS {
            let output_size = count * H::ExtElem::EXT_SIZE;
            let input_size = output_size * FRI_FOLD;

            let output = hal.alloc_elem("output", output_size);
            let mix = H::ExtElem::random(&mut rng);
            let input = generate_elem(&hal, &mut rng, input_size);
            hal.fri_fold(&output, &input, &mix);
        }
    }

    pub(crate) fn mix_poly_coeffs<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let combo_count = 100;
        let steps = 1 << 12;
        let domain = steps * INV_RATE;
        let input_size = H::CHECK_SIZE * domain;
        let output_size = steps * (combo_count + 1);
        let combos = vec![0; H::CHECK_SIZE];
        let mix_start = H::ExtElem::random(&mut rng);
        let mix = H::ExtElem::random(&mut rng);

        // Zeroed, as the prover allocates it: mix_poly_coeffs adds into it, and a GPU HAL's
        // buffer pool can hand back a previous test's buffer.
        let output = hal.alloc_extelem_zeroed("output", output_size);
        let combos = hal.copy_from_u32("combos", &combos);
        let input = generate_elem(&hal, &mut rng, input_size);

        hal.mix_poly_coeffs(
            &output,
            &mix_start,
            &mix,
            &input,
            &combos,
            H::CHECK_SIZE,
            steps,
        );
    }

    pub(crate) fn hash_fold<H: Hal>(hal_gpu: H) {
        const INPUTS: usize = 1024;
        const OUTPUTS: usize = INPUTS / 2;
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));
        let io = hal.alloc_digest("io", INPUTS * 2);
        io.view_mut(|g| {
            for i in 0..INPUTS {
                g[i + INPUTS] = Digest::from([
                    rng.next_u32() / 3,
                    rng.next_u32() / 3,
                    rng.next_u32() / 3,
                    rng.next_u32() / 3,
                    rng.next_u32() / 3,
                    rng.next_u32() / 3,
                    rng.next_u32() / 3,
                    rng.next_u32() / 3,
                ]);
            }
        });
        hal.hash_fold(&io, INPUTS, OUTPUTS);
    }

    /// `hash_fold_tree` (which GPU HALs implement as one call) against the CPU HAL.
    pub(crate) fn hash_fold_tree<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        // 13 and 14 reach the quad-per-permutation small-level kernels (pv_small.cuh, 2048 < n <= 8192).
        // Every layer count 1..=14: each crossover of the Poseidon2 fold kernels (pvs_w / pvs_q /
        // pv, plan 4.1 Z3) and of the single-layer fold paths.
        for layers in 1..=14 {
            let leaves = 1 << layers;
            let digests: Vec<Digest> = (0..leaves * 2)
                .map(|_| Digest::from(std::array::from_fn(|_| rng.next_u32() / 3)))
                .collect();
            let gpu_io = hal_gpu.copy_from_digest("io", &digests);
            let cpu_io = hal_cpu.copy_from_digest("io", &digests);
            hal_gpu.hash_fold_tree(&gpu_io, layers);
            hal_cpu.hash_fold_tree(&cpu_io, layers);
            assert!(
                gpu_io.to_vec()[1..] == cpu_io.to_vec()[1..],
                "layers={layers}"
            );
        }
    }

    pub(crate) fn hash_rows<H: Hal<Elem = BabyBearElem>>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));
        // Includes every column count the provers hash (1, 12, 16, 23, 64, 103, 128, 211) and
        // rows past one CUDA block.
        // 2049..8193 cover the warp / quad / thread-per-hash boundaries of pv_small.cuh.
        let rows = [1, 2, 3, 4, 10, 129, 1000, 2048, 2049, 8192, 8193];
        let cols = [1, 12, 16, 23, 32, 64, 103, 128, 211];
        for row_count in rows {
            for col_count in cols {
                let matrix_size = row_count * col_count;
                let matrix = generate_elem(&hal, &mut rng, matrix_size);
                let output = hal.alloc_digest("output", row_count);
                hal.hash_rows(&output, &matrix);
            }
        }
    }

    pub(crate) fn slice<H: Hal<Elem = BabyBearElem>>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));

        let rows = 4096;
        let cols = 256;
        let matrix_size = rows * cols;

        let nodes = hal.alloc_digest("nodes", rows * 2);
        let matrix = generate_elem(&hal, &mut rng, matrix_size);
        hal.hash_rows(&nodes.slice(rows, rows), &matrix);
    }

    /// `commit_coeffs_lde` against the unfused path (`make_coeffs` + `PolyGroup::new`) on the
    /// CPU HAL.
    pub(crate) fn commit_coeffs_lde<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let lg_blowup = crate::core::log2_ceil(INV_RATE);
        for (po2, count) in [
            (10, 1),
            (10, 7),
            (11, 64),
            (13, 33),
            (16, 211),
            (18, 5),
            (21, 2),
            (20, 9),
            (21, 3),
            (22, 1),
            (17, 7),
            (19, 3),
            (18, 33),
        ] {
            assert!(hal_gpu.has_commit_coeffs_lde(po2));
            let size = (1 << po2) * count;
            let values: Vec<H::Elem> = (0..size).map(|_| H::Elem::random(&mut rng)).collect();
            let witness = hal_gpu.copy_from_elem("witness", &values);
            let coeffs = hal_gpu.alloc_elem("coeffs", size);
            let evaluated = hal_gpu.alloc_elem("evaluated", size * INV_RATE);
            hal_gpu.commit_coeffs_lde(&coeffs, &evaluated, &witness, count);

            let cpu_witness = hal_cpu.copy_from_elem("witness", &values);
            let cpu_coeffs = hal_cpu.alloc_elem("coeffs", size);
            hal_cpu.batch_interpolate_ntt_zk_shift_from(&cpu_coeffs, &cpu_witness, count);
            let cpu_evaluated = hal_cpu.alloc_elem("evaluated", size * INV_RATE);
            hal_cpu.batch_expand_into_evaluate_ntt(&cpu_evaluated, &cpu_coeffs, count, lg_blowup);
            hal_cpu.batch_bit_reverse(&cpu_coeffs, count);

            assert!(
                witness.to_vec() == values,
                "po2={po2} count={count}: witness changed"
            );
            assert!(
                coeffs.to_vec() == cpu_coeffs.to_vec(),
                "po2={po2} count={count}: coeffs differ"
            );
            assert!(
                evaluated.to_vec() == cpu_evaluated.to_vec(),
                "po2={po2} count={count}: evaluated differs"
            );
        }
    }

    pub(crate) fn zk_shift<H: Hal>(hal_gpu: H) {
        let mut rng = rand::rng();
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        let hal = DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu));
        let counts = [(1000, (1 << 8)), (900, (1 << 12))];
        for (poly_count, steps) in counts {
            let count = poly_count * steps;
            let io = generate_elem(&hal, &mut rng, count);
            hal.zk_shift(&io, poly_count);
        }
    }

    // ---- Z6: CPU-vs-GPU conformance for every Hal method (plan 4.1 row Z6, 6.1) ----
    //
    // Methods `DualHal` forwards go through it (it compares both sides after each call). For the
    // ones it does not forward (`scatter_bits`, `combos_prepare`, `combos_divide`,
    // `alloc_extelem_zeroed`, the fused NTT entry points) the CPU HAL and the GPU HAL run
    // separately on identical inputs and the buffers are compared.

    use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
    use risc0_core::field::{Field, RootsOfUnity};

    use crate::{
        adapter::{
            CircuitCoreDef, CircuitInfo, MixState, PolyExt, ProtocolInfo, TapsProvider,
            PROOF_SYSTEM_INFO,
        },
        core::{log2_ceil, poly::poly_divide},
        hal::{AccumPreflight, CircuitHal},
        prove::Prover,
        taps::{TapData, TapSet},
    };

    fn dual<H: Hal>(hal_gpu: H) -> DualHal<H::Field, CpuHal<H::Field>, H> {
        let hal_cpu = CpuHal::new(hal_gpu.get_hash_suite().clone());
        DualHal::new(Rc::new(hal_cpu), Rc::new(hal_gpu))
    }

    fn cpu_of<H: Hal>(hal_gpu: &H) -> CpuHal<H::Field> {
        CpuHal::new(hal_gpu.get_hash_suite().clone())
    }

    fn rand_elems<E: Elem>(rng: &mut SmallRng, n: usize) -> Vec<E> {
        (0..n).map(|_| E::random(rng)).collect()
    }

    fn rand_exts<E: ExtElem>(rng: &mut SmallRng, n: usize) -> Vec<E> {
        (0..n).map(|_| E::random(rng)).collect()
    }

    /// `scatter` through DualHal: cycle counts 1..2^16, 0..=5 entries per cycle (empty ranges
    /// included), unique offsets in random order, and the empty-index no-op.
    pub(crate) fn scatter<H: Hal>(hal_gpu: H) {
        let hal = dual(hal_gpu);
        let mut rng = SmallRng::seed_from_u64(0x5ca7);
        for cycles in [1usize, 9, 1025, 1 << 16] {
            let mut index = vec![0u32];
            for _ in 0..cycles {
                let last = *index.last().unwrap();
                index.push(last + rng.random_range(0..=5));
            }
            let entries = *index.last().unwrap() as usize;
            let size = entries * 2 + 7;
            let mut perm: Vec<u32> = (0..size as u32).collect();
            perm.shuffle(&mut rng);
            let offsets = &perm[..entries];
            let values: Vec<H::Elem> = rand_elems(&mut rng, entries);
            let into = hal.copy_from_elem("into", &rand_elems::<H::Elem>(&mut rng, size));
            hal.scatter(&into, &index, offsets, &values);
        }
        let into = hal.copy_from_elem("into", &rand_elems::<H::Elem>(&mut rng, 4));
        hal.scatter(&into, &[], &[], &[]);
    }

    /// `scatter_bits` (not forwarded by DualHal): unique (row, base) cells, triples up to 2^13,
    /// plus a trailing partial triple (ignored by every HAL).
    pub(crate) fn scatter_bits<H: Hal>(hal_gpu: H) {
        let hal_cpu = cpu_of(&hal_gpu);
        let mut rng = SmallRng::seed_from_u64(0xb175);
        let cycles = 4096u32;
        let bases = 4u32;
        for n in [0usize, 1, 7, 1000, 8192] {
            let mut cells: Vec<(u32, u32)> = (0..cycles)
                .flat_map(|r| (0..bases).map(move |b| (r, b * 32)))
                .collect();
            cells.shuffle(&mut rng);
            let mut bit_data = Vec::with_capacity(n * 3 + 2);
            for &(row, base) in &cells[..n] {
                bit_data.extend([row, base, rng.random::<u32>()]);
            }
            bit_data.extend([1, 2]);
            let size = (bases * 32 * cycles) as usize;
            let init: Vec<H::Elem> = rand_elems(&mut rng, size);
            let gpu = hal_gpu.copy_from_elem("into", &init);
            let cpu = hal_cpu.copy_from_elem("into", &init);
            hal_gpu.scatter_bits(&gpu, &bit_data, cycles);
            hal_cpu.scatter_bits(&cpu, &bit_data, cycles);
            assert!(gpu.to_vec() == cpu.to_vec(), "scatter_bits n={n}");
        }
    }

    /// `eltwise_zeroize_elem` through DualHal: ~25% INVALID cells, every COUNTS size, and a
    /// sub-slice at an offset.
    pub(crate) fn eltwise_zeroize_elem<H: Hal>(hal_gpu: H) {
        let hal = dual(hal_gpu);
        let mut rng = SmallRng::seed_from_u64(0x2e70);
        let mut make = |rng: &mut SmallRng, n: usize| -> Vec<H::Elem> {
            (0..n)
                .map(|_| {
                    if rng.random_range(0..4) == 0 {
                        H::Elem::INVALID
                    } else {
                        H::Elem::random(rng)
                    }
                })
                .collect()
        };
        for count in COUNTS {
            let buf = hal.copy_from_elem("elems", &make(&mut rng, count));
            hal.eltwise_zeroize_elem(&buf);
        }
        let buf = hal.copy_from_elem("elems", &make(&mut rng, 5000));
        hal.eltwise_zeroize_elem(&buf.slice(13, 4000));
        assert!(
            buf.to_vec()[..13]
                .iter()
                .chain(&buf.to_vec()[4013..])
                .any(|e| !e.is_valid()),
            "cells outside the slice must be left alone"
        );
    }

    /// `eltwise_copy_elem_slice` through DualHal: offsets, strides != cols, empty shapes.
    pub(crate) fn eltwise_copy_elem_slice<H: Hal>(hal_gpu: H) {
        let hal = dual(hal_gpu);
        let mut rng = SmallRng::seed_from_u64(0xc0b1);
        // (rows, cols, from_offset, from_stride, into_offset, into_stride)
        for (rows, cols, fo, fs, io, is) in [
            (1usize, 1usize, 0usize, 1usize, 0usize, 1usize),
            (5, 3, 2, 4, 1, 7),
            (1000, 24, 13, 30, 5, 1024),
            (65, 211, 0, 211, 0, 256),
            (4096, 16, 0, 16, 3, 16),
            (0, 5, 0, 5, 0, 5),
            (7, 0, 0, 3, 0, 3),
        ] {
            let from: Vec<H::Elem> = rand_elems(&mut rng, fo + rows * fs + cols + 1);
            let into_size = io + rows * is + cols + 1;
            let into = hal.copy_from_elem("into", &rand_elems::<H::Elem>(&mut rng, into_size));
            hal.eltwise_copy_elem_slice(&into, &from, rows, cols, fo, fs, io, is);
        }
    }

    /// `alloc_elem_init` (DualHal) and `alloc_extelem_zeroed` (direct; the pool is dirtied at
    /// each size first, so a fill that is skipped or lands late shows up).
    pub(crate) fn alloc_init<H: Hal>(hal_gpu: H) {
        let mut rng = SmallRng::seed_from_u64(0xa110);
        for size in [1usize, 9, 1000, 1 << 20] {
            let junk = hal_gpu.copy_from_extelem("junk", &rand_exts::<H::ExtElem>(&mut rng, size));
            drop(junk);
            let zeroed = hal_gpu.alloc_extelem_zeroed("zeroed", size);
            assert!(
                zeroed
                    .to_vec()
                    .iter()
                    .all(|e| e.to_u32_words() == H::ExtElem::ZERO.to_u32_words()),
                "alloc_extelem_zeroed size={size}"
            );
            let junk = hal_gpu.copy_from_elem("junk", &rand_elems::<H::Elem>(&mut rng, size));
            drop(junk);
            for value in [
                H::Elem::ZERO,
                H::Elem::ONE,
                H::Elem::INVALID,
                H::Elem::random(&mut rng),
            ] {
                let buf = hal_gpu.alloc_elem_init("init", size, value);
                // Raw words: `==` asserts validity, and INVALID is one of the fill values.
                let want = value.to_u32_words();
                assert!(
                    buf.to_vec().iter().all(|e| e.to_u32_words() == want),
                    "alloc_elem_init size={size}"
                );
            }
        }
        let hal = dual(hal_gpu);
        let buf = hal.alloc_elem_init("init", 1025, H::Elem::random(&mut rng));
        hal.eltwise_zeroize_elem(&buf);
    }

    /// `combos_prepare` (not forwarded by DualHal): 300 registers of 1..=4 taps over 5 combos,
    /// repeated combo ids, cycles 16..2^14.
    pub(crate) fn combos_prepare<H: Hal>(hal_gpu: H) {
        let hal_cpu = cpu_of(&hal_gpu);
        let mut rng = SmallRng::seed_from_u64(0xc0b0);
        let combo_count = 5;
        for cycles in [16usize, 1024, 1 << 14] {
            let regs = 300;
            let reg_sizes: Vec<u32> = (0..regs).map(|_| rng.random_range(1..=4)).collect();
            let reg_combo_ids: Vec<u32> = (0..regs)
                .map(|_| rng.random_range(0..combo_count as u32))
                .collect();
            let taps: usize = reg_sizes.iter().map(|&s| s as usize).sum();
            let coeff_u: Vec<H::ExtElem> = rand_exts(&mut rng, taps + H::CHECK_SIZE);
            let mix = H::ExtElem::random(&mut rng);
            let init: Vec<H::ExtElem> = rand_exts(&mut rng, cycles * (combo_count + 1));
            let gpu = hal_gpu.copy_from_extelem("combos", &init);
            let cpu = hal_cpu.copy_from_extelem("combos", &init);
            hal_gpu.combos_prepare(
                &gpu,
                &coeff_u,
                combo_count,
                cycles,
                &reg_sizes,
                &reg_combo_ids,
                &mix,
            );
            hal_cpu.combos_prepare(
                &cpu,
                &coeff_u,
                combo_count,
                cycles,
                &reg_sizes,
                &reg_combo_ids,
                &mix,
            );
            assert!(
                gpu.to_vec() == cpu.to_vec(),
                "combos_prepare cycles={cycles}"
            );
        }
    }

    /// Polynomial of `n` coefficients: random, or (`divisible`) a random polynomial times
    /// prod (X - r) over `roots`, so each division leaves a zero remainder.
    fn divisible_poly<E: ExtElem>(
        rng: &mut SmallRng,
        n: usize,
        roots: &[E],
        divisible: bool,
    ) -> Vec<E> {
        if !divisible {
            return rand_exts(rng, n);
        }
        let m = roots.len();
        let mut p: Vec<E> = rand_exts(rng, n - m);
        p.resize(n, E::ZERO);
        for (deg, &z) in (n - m..).zip(roots) {
            // p * (X - z): q[i] = p[i - 1] - z * p[i]
            for i in (0..=deg).rev() {
                let lo = if i > 0 { p[i - 1] } else { E::ZERO };
                let hi = if i < deg { p[i] } else { E::ZERO };
                p[i] = lo - z * hi;
            }
        }
        p
    }

    /// The `tracing` output of `f` on this thread (for `report_nonzero_remainders`).
    fn capture_logs(f: impl FnOnce()) -> Vec<String> {
        use std::{
            io::Write,
            sync::{Arc, Mutex},
        };
        #[derive(Clone)]
        struct W(Arc<Mutex<Vec<u8>>>);
        impl Write for W {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let buf = Arc::new(Mutex::new(vec![]));
        let w = W(buf.clone());
        let sub = tracing_subscriber::fmt()
            .with_writer(move || w.clone())
            .with_ansi(false)
            .without_time()
            .with_max_level(tracing::Level::ERROR)
            .finish();
        {
            let _guard = tracing_subscriber::util::SubscriberInitExt::set_default(sub);
            f();
        }
        let text = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        text.lines()
            .filter(|l| l.contains("nonzero remainder"))
            .map(str::to_owned)
            .collect()
    }

    /// `combos_divide` at `2^po2` cycles (not forwarded by DualHal).
    ///
    /// Zero remainders: 7 slices, 6 in-order chunks with 1/2/3/1/2/1 roots (the prover's shape:
    /// the last chunk is the check combo), slice 6 untouched; the CPU HAL (which asserts zero
    /// remainders) is the oracle.
    ///
    /// Nonzero remainders (GPU HALs only; the CPU HAL panics): chunks 1 and 4 hold random data. The
    /// oracle is the CUDA/sppark `div_by_x_minus_z<rotate=true>` semantics the GPU HALs share:
    /// `poly_divide`, then the remainder lands in `a[n - 1]`; one `report_nonzero_remainders`
    /// line per nonzero remainder.
    pub(crate) fn combos_divide<H: Hal>(hal_gpu: H, po2: usize) {
        let hal_cpu = cpu_of(&hal_gpu);
        let n = 1usize << po2;
        let roots = [1usize, 2, 3, 1, 2, 1];
        for bad in [&[][..], &[1usize, 4][..]] {
            let mut rng = SmallRng::seed_from_u64(0xd1u64 << 8 | po2 as u64);
            let mut data: Vec<H::ExtElem> = rand_exts(&mut rng, 7 * n);
            let mut chunks = vec![];
            for (c, &r) in roots.iter().enumerate() {
                let pows: Vec<H::ExtElem> = rand_exts(&mut rng, r);
                let slice = divisible_poly(&mut rng, n, &pows, !bad.contains(&c));
                data[c * n..(c + 1) * n].copy_from_slice(&slice);
                chunks.push((c, pows));
            }
            let gpu = hal_gpu.copy_from_extelem("combos", &data);
            let logs = capture_logs(|| hal_gpu.combos_divide(&gpu, chunks.clone(), n));
            let gpu = gpu.to_vec();
            if bad.is_empty() {
                let cpu = hal_cpu.copy_from_extelem("combos", &data);
                hal_cpu.combos_divide(&cpu, chunks, n);
                assert!(logs.is_empty(), "po2={po2}: unexpected {logs:?}");
                assert!(gpu == cpu.to_vec(), "po2={po2}: combos differ");
            } else {
                let mut expect = data.clone();
                let mut nonzero = 0;
                for (c, pows) in &chunks {
                    let slice = &mut expect[c * n..(c + 1) * n];
                    for &pow in pows {
                        let rem = poly_divide(slice, pow);
                        slice[n - 1] = rem;
                        nonzero += usize::from(rem != H::ExtElem::ZERO);
                    }
                }
                assert!(nonzero >= 2, "po2={po2}: the bad chunks must not divide");
                assert_eq!(logs.len(), nonzero, "po2={po2}: {logs:?}");
                assert!(
                    gpu == expect,
                    "po2={po2}: combos differ (nonzero remainders)"
                );
            }
        }
    }

    /// The zk-shift family at many shapes: `zk_shift` and `batch_interpolate_ntt` through
    /// DualHal; the fused `batch_interpolate_ntt_zk_shift` / `_from` (which DualHal does not
    /// forward) directly, with `src` checked unchanged.
    pub(crate) fn zk_shift_variants<H: Hal>(hal_gpu: H) {
        let hal_cpu = cpu_of(&hal_gpu);
        let mut rng = SmallRng::seed_from_u64(0x2c5f);
        for lg in [1usize, 2, 5, 10, 11, 12, 13, 16, 20] {
            for cols in [1usize, 3, 211] {
                if lg + log2_ceil(cols) > 22 {
                    continue;
                }
                let size = cols << lg;
                let values: Vec<H::Elem> = rand_elems(&mut rng, size);
                let what = format!("lg={lg} cols={cols}");

                let gpu = hal_gpu.copy_from_elem("io", &values);
                let cpu = hal_cpu.copy_from_elem("io", &values);
                hal_gpu.zk_shift(&gpu, cols);
                hal_cpu.zk_shift(&cpu, cols);
                assert!(gpu.to_vec() == cpu.to_vec(), "zk_shift {what}");

                let gpu = hal_gpu.copy_from_elem("io", &values);
                let cpu = hal_cpu.copy_from_elem("io", &values);
                hal_gpu.batch_interpolate_ntt_zk_shift(&gpu, cols);
                hal_cpu.batch_interpolate_ntt_zk_shift(&cpu, cols);
                let fused = cpu.to_vec();
                assert!(gpu.to_vec() == fused, "interpolate_zk_shift {what}");

                let src = hal_gpu.copy_from_elem("src", &values);
                let dst = hal_gpu.alloc_elem("dst", size);
                hal_gpu.batch_interpolate_ntt_zk_shift_from(&dst, &src, cols);
                assert!(dst.to_vec() == fused, "interpolate_zk_shift_from {what}");
                assert!(
                    src.to_vec() == values,
                    "interpolate_zk_shift_from {what}: src changed"
                );
            }
        }
    }

    /// `batch_evaluate_any` through DualHal at every dispatcher branch (Metal: eu2 S = 1, S capped
    /// by deg / 4096, S stopped by S * nEvals >= 1024, nEvals >= 1024, Horner for deg < 4096,
    /// deg % 4096 != 0 and a coeffs slice that is not 16-byte aligned, eu2 again at a 16-byte
    /// aligned slice offset, and nEvals = 0).
    pub(crate) fn eval_any_shapes<H: Hal>(hal_gpu: H) {
        let hal = dual(hal_gpu);
        let mut rng = SmallRng::seed_from_u64(0xe0a1);
        // (deg, polys, evals, coeffs element offset); degrees are powers of two (the CPU HAL's
        // contract), so the Horner gate is reached through deg < 4096 and misalignment.
        for (deg, polys, evals, off) in [
            (4096usize, 1usize, 1usize, 0usize),
            (1 << 16, 4, 1, 0),
            (1 << 16, 223, 865, 0),
            (1 << 14, 8, 1100, 0),
            (1 << 14, 300, 300, 0),
            (1 << 20, 2, 4, 0),
            (2048, 3, 9, 0),
            (1, 3, 5, 0),
            (8192, 3, 5, 1),
            (8192, 3, 5, 3),
            (8192, 3, 5, 4),
            (4096, 1, 0, 0),
        ] {
            let all = hal.copy_from_elem(
                "coeffs",
                &rand_elems::<H::Elem>(&mut rng, off + deg * polys),
            );
            let coeffs = all.slice(off, deg * polys);
            let which: Vec<u32> = (0..evals)
                .map(|_| rng.random_range(0..polys as u32))
                .collect();
            let which = hal.copy_from_u32("which", &which);
            let xs = hal.copy_from_extelem("xs", &rand_exts::<H::ExtElem>(&mut rng, evals));
            let out = hal.alloc_extelem("out", evals);
            hal.batch_evaluate_any(&coeffs, polys, &which, &xs, &out);
        }
    }

    /// `mix_poly_coeffs` through DualHal at every kernel gate (Metal: v2 <= 1638 inputs,
    /// v2 with device combo ids 1639..=2048, table-less above) and counts with count % 256 != 0.
    /// Combo ids come in runs of 1..=5 (v2's run-end read-modify-write) and `out` starts random.
    pub(crate) fn mix_poly_coeffs_shapes<H: Hal>(hal_gpu: H) {
        let hal = dual(hal_gpu);
        let mut rng = SmallRng::seed_from_u64(0x312);
        for input_size in [1usize, 2, 7, 100, 1637, 1638, 1639, 2047, 2048, 2049, 2500] {
            for count in [1usize, 255, 257, 1000, 4097] {
                let mut combos = Vec::with_capacity(input_size);
                let mut id = 0u32;
                while combos.len() < input_size {
                    let run = rng.random_range(1..=5);
                    for _ in 0..run {
                        if combos.len() < input_size {
                            combos.push(id);
                        }
                    }
                    id = (id + rng.random_range(1..=3)) % 8;
                }
                let ncombos = *combos.iter().max().unwrap() as usize + 1;
                let out = hal
                    .copy_from_extelem("out", &rand_exts::<H::ExtElem>(&mut rng, count * ncombos));
                let input = hal.copy_from_elem(
                    "input",
                    &rand_elems::<H::Elem>(&mut rng, count * input_size),
                );
                let combos = hal.copy_from_u32("combos", &combos);
                let mix_start = H::ExtElem::random(&mut rng);
                let mix = H::ExtElem::random(&mut rng);
                hal.mix_poly_coeffs(&out, &mix_start, &mix, &input, &combos, input_size, count);
            }
        }
    }

    // ---- a toy circuit for a full prove / verify round trip ----
    //
    // Groups (alphabetical, as the codegen orders them): accum (0) = [acc], code (1) = [k],
    // data (2) = [a, c1, c2, c3]; `a` is also tapped one cycle back. Globals: out (1), mix (1).
    // Constraints, cyclic over the 2^po2 rows:
    //   c1 = a + a[-1] + k        c3 = c2 * a        acc = a * mix + out
    // so the trace has degree-1 and degree-2 constraints, a two-tap combo (backs 0 and 1) and a
    // one-tap combo, three committed groups and the accum mix drawn after code/data.

    const fn tap(group: usize, offset: u16, back: u16, combo: u8, skip: u8) -> TapData {
        TapData {
            offset,
            back,
            group,
            combo,
            skip,
        }
    }

    static TOY_TAPS: TapSet<'static> = TapSet {
        taps: &[
            tap(0, 0, 0, 0, 1),
            tap(1, 0, 0, 0, 1),
            tap(2, 0, 0, 1, 2),
            tap(2, 0, 1, 1, 2),
            tap(2, 1, 0, 0, 1),
            tap(2, 2, 0, 0, 1),
            tap(2, 3, 0, 0, 1),
        ],
        combo_taps: &[0, 0, 1],
        combo_begin: &[0, 1, 3],
        group_begin: &[0, 1, 2, 7],
        combos_count: 2,
        reg_count: 6,
        tot_combo_backs: 3,
        group_names: &["accum", "code", "data"],
    };

    pub(crate) struct ToyCircuit;

    impl CircuitInfo for ToyCircuit {
        const CIRCUIT_INFO: ProtocolInfo = ProtocolInfo(*b"Z6_TOY_CIRCUIT__");
        const OUTPUT_SIZE: usize = 1;
        const MIX_SIZE: usize = 1;
    }

    impl TapsProvider for ToyCircuit {
        fn get_taps(&self) -> &'static TapSet<'static> {
            &TOY_TAPS
        }
    }

    /// The mixed constraint value: `u` = [acc, k, a, a[-1], c1, c2, c3] (taps order).
    fn toy_constraints<E: ExtElem>(poly_mix: E, u: [E; 7], out: E, mix: E) -> E {
        let [acc, k, a, a_back, c1, c2, c3] = u;
        let c_sum = c1 - a - a_back - k;
        let c_mul = c3 - c2 * a;
        let c_acc = acc - a * mix - out;
        c_sum + poly_mix * c_mul + poly_mix * poly_mix * c_acc
    }

    impl<F: Field> PolyExt<F> for ToyCircuit {
        fn poly_ext(
            &self,
            mix: &F::ExtElem,
            u: &[F::ExtElem],
            args: &[&[F::Elem]],
        ) -> MixState<F::ExtElem> {
            let lift = |e: &F::Elem| F::ExtElem::from_subfield(e);
            let tot = toy_constraints(
                *mix,
                std::array::from_fn(|i| u[i]),
                lift(&args[0][0]),
                lift(&args[1][0]),
            );
            MixState {
                tot,
                mul: F::ExtElem::ONE,
            }
        }
    }

    impl<F: Field> CircuitCoreDef<F> for ToyCircuit {}

    /// Host eval_check (any HAL, through `to_vec`/`view_mut`): the same layout and point
    /// convention as the circuits' CPU eval_check (`check[k * domain + i]` = component k of
    /// `C(3 x_i) / ((3 x_i)^n - 1)`, `x_i = rou^i`, back b = row `i - b * INV_RATE`).
    impl<H: Hal> CircuitHal<H> for ToyCircuit {
        fn accumulate(
            &self,
            _preflight: &AccumPreflight,
            _ctrl: &H::Buffer<H::Elem>,
            _io: &H::Buffer<H::Elem>,
            _data: &H::Buffer<H::Elem>,
            _mix: &H::Buffer<H::Elem>,
            _accum: &H::Buffer<H::Elem>,
            _steps: usize,
        ) {
            unimplemented!("the toy circuit accumulates on the host")
        }

        fn eval_check(
            &self,
            check: &H::Buffer<H::Elem>,
            groups: &[&H::Buffer<H::Elem>],
            globals: &[&H::Buffer<H::Elem>],
            poly_mix: H::ExtElem,
            po2: usize,
            steps: usize,
        ) {
            let domain = steps * INV_RATE;
            let acc = groups[0].to_vec();
            let code = groups[1].to_vec();
            let data = groups[2].to_vec();
            let lift = |e: H::Elem| H::ExtElem::from_subfield(&e);
            let out = lift(globals[0].to_vec()[0]);
            let mix = lift(globals[1].to_vec()[0]);
            let rou = H::Elem::ROU_FWD[po2 + log2_ceil(INV_RATE)];
            let three = H::Elem::from_u64(3);
            let mut res = vec![H::Elem::ZERO; H::ExtElem::EXT_SIZE * domain];
            let mut x = H::Elem::ONE;
            for i in 0..domain {
                let back = (i + domain - INV_RATE) % domain;
                let u = [
                    acc[i],
                    code[i],
                    data[i],
                    data[back],
                    data[domain + i],
                    data[2 * domain + i],
                    data[3 * domain + i],
                ]
                .map(lift);
                let tot = toy_constraints(poly_mix, u, out, mix);
                let y = (three * x).pow(steps) - H::Elem::ONE;
                let ret = tot * lift(y.inv());
                for (k, e) in ret.subelems().iter().enumerate() {
                    res[k * domain + i] = *e;
                }
                x *= rou;
            }
            check.view_mut(|c| c.copy_from_slice(&res));
        }
    }

    /// Prove the toy circuit at `2^po2` cycles on `hal`; the witness depends only on `seed`.
    pub(crate) fn toy_prove<H: Hal + 'static>(hal: &H, po2: usize, seed: u64) -> Vec<u32> {
        let n = 1usize << po2;
        let mut rng = SmallRng::seed_from_u64(seed);
        let k: Vec<H::Elem> = rand_elems(&mut rng, n);
        let a: Vec<H::Elem> = rand_elems(&mut rng, n);
        let c2: Vec<H::Elem> = rand_elems(&mut rng, n);
        let out = H::Elem::random(&mut rng);
        let c1: Vec<H::Elem> = (0..n).map(|i| a[i] + a[(i + n - 1) % n] + k[i]).collect();
        let c3: Vec<H::Elem> = (0..n).map(|i| c2[i] * a[i]).collect();
        let data = [a.as_slice(), &c1, &c2, &c3].concat();

        let mut prover = Prover::new(hal, &TOY_TAPS);
        let hashfn = hal.get_hash_suite().hashfn.clone();
        prover
            .iop()
            .commit(&hashfn.hash_elem_slice(&PROOF_SYSTEM_INFO.encode()));
        prover
            .iop()
            .commit(&hashfn.hash_elem_slice(&ToyCircuit::CIRCUIT_INFO.encode()));
        let header = [out, H::Elem::from_u32_words(&[po2 as u32])];
        prover.iop().commit(&hashfn.hash_elem_slice(&header));
        prover.iop().write_field_elem_slice(&header);
        prover.set_po2(po2);

        let code_buf = hal.copy_from_elem("code", &k);
        prover.commit_group(1, &code_buf);
        let data_buf = hal.copy_from_elem("data", &data);
        prover.commit_group(2, &data_buf);
        let mix = prover.iop().random_elem();
        let acc: Vec<H::Elem> = a.iter().map(|&a| a * mix + out).collect();
        let accum_buf = hal.copy_from_elem("accum", &acc);
        prover.commit_group(0, &accum_buf);

        let out_buf = hal.copy_from_elem("out", &[out]);
        let mix_buf = hal.copy_from_elem("mix", &[mix]);
        prover.finalize(&[&out_buf, &mix_buf], &ToyCircuit)
    }

    /// Verify a toy seal with the CPU verifier.
    pub(crate) fn toy_verify<F: Field>(
        suite: &crate::core::hash::HashSuite<F>,
        seal: &[u32],
    ) -> Result<(), crate::verify::VerificationError> {
        crate::verify::verify(&ToyCircuit, suite, seal, |_, _| Ok(()))
    }

    /// Full prove / verify round trip at each `po2`: the toy circuit proved on the CPU HAL, on the
    /// GPU HAL alone (fused NTT entry points, GPU combos_divide etc.) and on DualHal(CPU, GPU)
    /// (every forwarded method compared after each call). All three seals must be identical
    /// (the prover is deterministic and every HAL is bit-exact) and verify; a corrupted seal must
    /// not.
    pub(crate) fn prove_verify_round_trip<H: Hal + 'static>(make: impl Fn() -> H, po2s: &[usize]) {
        for &po2 in po2s {
            let seed = 0x70e0 + po2 as u64;
            let hal_gpu = make();
            let suite = hal_gpu.get_hash_suite().clone();
            let hal_cpu = cpu_of(&hal_gpu);
            let seal_cpu = toy_prove(&hal_cpu, po2, seed);
            toy_verify(&suite, &seal_cpu).unwrap_or_else(|e| panic!("po2={po2}: CPU seal: {e}"));
            let seal_gpu = toy_prove(&hal_gpu, po2, seed);
            toy_verify(&suite, &seal_gpu).unwrap_or_else(|e| panic!("po2={po2}: GPU seal: {e}"));
            assert!(
                seal_gpu == seal_cpu,
                "po2={po2}: GPU seal differs from the CPU seal"
            );
            let hal_dual = dual(make());
            let seal_dual = toy_prove(&hal_dual, po2, seed);
            assert!(
                seal_dual == seal_cpu,
                "po2={po2}: DualHal seal differs from the CPU seal"
            );
            let mut bad = seal_gpu.clone();
            let i = bad.len() / 2;
            bad[i] ^= 1;
            assert!(
                toy_verify(&suite, &bad).is_err(),
                "po2={po2}: a corrupted seal verified"
            );
        }
    }
}
