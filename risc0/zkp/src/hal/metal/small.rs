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

//! Small kernels of the Metal HAL (plan item Z1): fills, zeroize, region copy, scatter,
//! scatter_bits and combos_prepare/combos_sub. Each method mirrors the CUDA HAL
//! (`zkp/src/hal/cuda.rs`) and its launcher in `sys/kernels/zkp/cuda/ffi.cu`: same kernel, same
//! 1-D launch over the same count. The kernels are in `sys/kernels/zkp/metal/{eltwise,misc}.metal`.
//!
//! Everything here only encodes; host inputs are copied into fresh pooled shared buffers (the
//! unified-memory equivalent of CUDA's H2D copies), which needs no sync (see
//! [BufferImpl::copy_from]). Indices that come from the caller are validated on the host and
//! panic where the CPU HAL would panic on an out-of-bounds index, so a bad trace can never make a
//! kernel write outside its buffer.

use std::collections::BTreeMap;

use risc0_core::field::{
    baby_bear::{BabyBearElem, BabyBearExtElem},
    Elem as _, ExtElem as _,
};

use super::{as_bytes, BufferImpl, KernelArg, MetalHal, MetalHash};
use crate::{hal::Buffer as _, INV_RATE};

/// `misc.metal` `CopyRegionParams`, bound with `set_bytes` at index 2.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::NoUninit)]
struct CopyRegionParams {
    from_rows: u32,
    from_cols: u32,
    from_offset: u32,
    from_stride: u32,
    into_offset: u32,
    into_stride: u32,
    _pad0: u32,
    _pad1: u32,
}
const _: () = assert!(std::mem::size_of::<CopyRegionParams>() == 32);

/// `Hal::CHECK_SIZE` for BabyBear (`INV_RATE * EXT_SIZE`).
const CHECK_SIZE: usize = INV_RATE * BabyBearExtElem::EXT_SIZE;

fn to_u32(value: usize, what: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{what} = {value} does not fit in u32"))
}

impl<MH: MetalHash> MetalHal<MH> {
    /// Fill `buf` with the 32-bit word `word`: a blit fill when all four bytes are equal (zero,
    /// `Fp::INVALID`), else the `fill_u32` kernel (plan 2.5). Encode only.
    pub(crate) fn fill_u32<T: Clone + std::fmt::Debug + PartialEq>(
        &self,
        buf: &BufferImpl<T>,
        word: u32,
    ) {
        assert_eq!(std::mem::size_of::<T>() % 4, 0);
        let words = buf.size() * std::mem::size_of::<T>() / 4;
        if words == 0 {
            return;
        }
        let bytes = word.to_le_bytes();
        if bytes.iter().all(|&b| b == bytes[0]) {
            let offset = (buf.offset() * std::mem::size_of::<T>()) as u64;
            self.blit_fill(buf.as_buf(), offset, (words * 4) as u64, bytes[0]);
        } else {
            let count = to_u32(words, "fill_u32 words");
            let args = &[
                buf.as_arg(),
                KernelArg::Integer(word),
                KernelArg::Integer(count),
            ];
            self.launch("fill_u32", count as u64, args);
        }
    }

    /// `Hal::alloc_elem_init` (CUDA: `set_u32(value.as_u32_montgomery())`, a memset).
    pub(crate) fn small_alloc_elem_init(
        &self,
        name: &'static str,
        size: usize,
        value: BabyBearElem,
    ) -> BufferImpl<BabyBearElem> {
        let buf = self.alloc_buffer(name, size);
        self.fill_u32(&buf, value.as_u32_montgomery());
        buf
    }

    /// `Hal::alloc_extelem_zeroed` (CUDA: `set_u32(0)`).
    pub(crate) fn small_alloc_extelem_zeroed(
        &self,
        name: &'static str,
        size: usize,
    ) -> BufferImpl<BabyBearExtElem> {
        let buf = self.alloc_buffer(name, size);
        self.fill_u32(&buf, 0);
        buf
    }

    /// `Hal::eltwise_zeroize_elem`: `eltwise_zeroize_fp` over every element (ffi.cu:233).
    pub(crate) fn small_eltwise_zeroize_elem(&self, elems: &BufferImpl<BabyBearElem>) {
        let count = to_u32(elems.size(), "eltwise_zeroize_elem size");
        let args = &[elems.as_arg(), KernelArg::Integer(count)];
        self.launch("eltwise_zeroize_fp", count as u64, args);
    }

    /// `Hal::eltwise_sum_extelem`: `eltwise_sum_fpext(out, in, to_add, count)` (ffi.cu:226).
    pub(crate) fn small_eltwise_sum_extelem(
        &self,
        output: &BufferImpl<BabyBearElem>,
        input: &BufferImpl<BabyBearExtElem>,
    ) {
        let count = output.size() / BabyBearExtElem::EXT_SIZE;
        let to_add = input.size() / count;
        assert_eq!(output.size(), count * BabyBearExtElem::EXT_SIZE);
        assert_eq!(input.size(), count * to_add);
        // The kernel writes out[gid + 3 * count] in 32 bits (as CUDA); the input index is 64-bit.
        to_u32(output.size(), "eltwise_sum output size");
        let args = &[
            output.as_arg(),
            input.as_arg(),
            KernelArg::Integer(to_u32(to_add, "eltwise_sum to_add")),
            KernelArg::Integer(to_u32(count, "eltwise_sum count")),
        ];
        self.launch("eltwise_sum_fpext", count as u64, args);
    }

    /// `Hal::eltwise_copy_elem_slice` (cuda.rs `eltwise_copy_elem_slice`, ffi.cu:205): upload
    /// `from`, then `eltwise_copy_fp_region` with one thread per row. Only the part of `from` the
    /// region reads is uploaded (CUDA uploads all of `from`; the copied values are identical).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn small_eltwise_copy_elem_slice(
        &self,
        into: &BufferImpl<BabyBearElem>,
        from: &[BabyBearElem],
        from_rows: usize,
        from_cols: usize,
        from_offset: usize,
        from_stride: usize,
        into_offset: usize,
        into_stride: usize,
    ) {
        if from_rows == 0 || from_cols == 0 {
            return;
        }
        // The last element each side touches, +1 (the CPU HAL panics on these bounds).
        let from_end = from_offset + (from_rows - 1) * from_stride + from_cols;
        let into_end = into_offset + (from_rows - 1) * into_stride + from_cols;
        assert!(
            from_end <= from.len(),
            "eltwise_copy_elem_slice: source region ends at {from_end}, slice has {}",
            from.len()
        );
        assert!(
            into_end <= into.size(),
            "eltwise_copy_elem_slice: target region ends at {into_end}, buffer has {}",
            into.size()
        );
        to_u32(into_end, "eltwise_copy_elem_slice target end");
        let src = &from[from_offset..from_end];
        to_u32(src.len(), "eltwise_copy_elem_slice source length");
        let src = self.copy_from_slice("from", src);
        let params = [CopyRegionParams {
            from_rows: to_u32(from_rows, "from_rows"),
            from_cols: to_u32(from_cols, "from_cols"),
            from_offset: 0,
            from_stride: to_u32(from_stride, "from_stride"),
            into_offset: to_u32(into_offset, "into_offset"),
            into_stride: to_u32(into_stride, "into_stride"),
            _pad0: 0,
            _pad1: 0,
        }];
        let args = &[
            into.as_arg(),
            src.as_arg(),
            KernelArg::Bytes(as_bytes(&params)),
        ];
        self.launch("eltwise_copy_fp_region", from_rows as u64, args);
    }

    /// `Hal::scatter` (cuda.rs `scatter`, ffi.cu:384 `risc0_zkp_cuda_scatter_from_host`): upload
    /// `index`/`offsets`/`values`, then one `scatter` thread per cycle.
    pub(crate) fn small_scatter(
        &self,
        into: &BufferImpl<BabyBearElem>,
        index: &[u32],
        offsets: &[u32],
        values: &[BabyBearElem],
    ) {
        if index.len() < 2 {
            return;
        }
        let count = index.len() - 1;
        // Validate exactly what the kernel (and the CPU HAL) reads: for each cycle range
        // [index[c], index[c+1]) with start < end, the range must lie inside offsets/values and
        // every offset in it must be inside `into`. Offsets outside every range are never read,
        // so they are not checked (the CPU HAL and CUDA accept them too).
        let limit = offsets.len().min(values.len());
        let size = into.size();
        let mut max_end = 0usize;
        for cycle in 0..count {
            let (start, end) = (index[cycle] as usize, index[cycle + 1] as usize);
            if start < end {
                assert!(
                    end <= limit,
                    "scatter: index[{}] = {end} beyond offsets/values ({limit})",
                    cycle + 1
                );
                if let Some(bad) = offsets[start..end].iter().find(|&&o| o as usize >= size) {
                    panic!("scatter: offset {bad} out of bounds for a buffer of {size}");
                }
                max_end = max_end.max(end);
            }
        }
        if max_end == 0 {
            return;
        }
        let index = self.copy_from_slice("scatter_index", index);
        let offsets = self.copy_from_slice("scatter_offsets", &offsets[..max_end]);
        let values = self.copy_from_slice("scatter_values", &values[..max_end]);
        let count = to_u32(count, "scatter count");
        let args = &[
            into.as_arg(),
            index.as_arg(),
            offsets.as_arg(),
            values.as_arg(),
            KernelArg::Integer(count),
        ];
        self.launch("scatter", count as u64, args);
    }

    /// `Hal::scatter_bits` (cuda.rs `scatter_bits`, ffi.cu:461): upload the packed
    /// `(row, base_col, value)` triples, then one `scatter_bits` thread per triple.
    pub(crate) fn small_scatter_bits(
        &self,
        into: &BufferImpl<BabyBearElem>,
        bit_data: &[u32],
        cycles: u32,
    ) {
        let n = bit_data.len() / 3;
        if n == 0 {
            return;
        }
        // The kernel indexes in u32; a buffer that fits in u32 plus the u64 bound below rules
        // out wraparound. Only the highest write offset (bit 31) needs checking, since offsets
        // increase with the bit index. No `row < cycles` requirement: the CPU HAL and CUDA
        // accept any in-bounds triple.
        let size = to_u32(into.size(), "scatter_bits into size") as u64;
        for t in bit_data[..n * 3].chunks_exact(3) {
            let (row, base) = (t[0] as u64, t[1] as u64);
            let last = (base + 31) * cycles as u64 + row;
            assert!(
                last < size,
                "scatter_bits: triple (row {row}, base_col {base}) out of bounds for a buffer of \
                 {size} with {cycles} cycles"
            );
        }
        // The kernel reads data[gid * 3 + 2] in 32 bits.
        to_u32(n * 3, "scatter_bits data length");
        let data = self.copy_from_slice("scatter_bits_data", &bit_data[..n * 3]);
        let count = to_u32(n, "scatter_bits count");
        let args = &[
            into.as_arg(),
            data.as_arg(),
            KernelArg::Integer(cycles),
            KernelArg::Integer(count),
        ];
        self.launch("scatter_bits", count as u64, args);
    }

    /// `Hal::combos_prepare` exactly as cuda.rs does it: the per-element deltas are summed on
    /// the host (coeff_u is host data), then `combos_sub` applies `combos[off] -= delta` with one
    /// thread per distinct offset. Field arithmetic is exact, so this equals the CPU HAL's
    /// sequential subtractions bit for bit.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn small_combos_prepare(
        &self,
        combos: &BufferImpl<BabyBearExtElem>,
        coeff_u: &[BabyBearExtElem],
        combo_count: usize,
        cycles: usize,
        reg_sizes: &[u32],
        reg_combo_ids: &[u32],
        mix: &BabyBearExtElem,
    ) {
        let mut deltas: BTreeMap<usize, BabyBearExtElem> = BTreeMap::new();
        let mut cur = BabyBearExtElem::ONE;
        let mut pos = 0usize;
        for (size, combo_id) in reg_sizes.iter().zip(reg_combo_ids.iter()) {
            for j in 0..*size as usize {
                let off = cycles * *combo_id as usize + j;
                *deltas.entry(off).or_insert(BabyBearExtElem::ZERO) += cur * coeff_u[pos + j];
            }
            cur *= *mix;
            pos += *size as usize;
        }
        for _ in 0..CHECK_SIZE {
            let off = cycles * combo_count;
            *deltas.entry(off).or_insert(BabyBearExtElem::ZERO) += cur * coeff_u[pos];
            pos += 1;
            cur *= *mix;
        }
        if deltas.is_empty() {
            return;
        }
        // BTreeMap keys are sorted: the last one is the largest offset.
        let max_off = *deltas.keys().next_back().unwrap();
        assert!(
            max_off < combos.size(),
            "combos_prepare: offset {max_off} out of bounds for combos of {}",
            combos.size()
        );
        to_u32(max_off, "combos_prepare offset");
        let offs: Vec<u32> = deltas.keys().map(|&o| o as u32).collect();
        let vals: Vec<BabyBearExtElem> = deltas.values().copied().collect();
        let offs = self.copy_from_slice("combos_offs", &offs);
        let vals = self.copy_from_slice("combos_deltas", &vals);
        let count = to_u32(offs.size(), "combos_sub count");
        let args = &[
            combos.as_arg(),
            vals.as_arg(),
            offs.as_arg(),
            KernelArg::Integer(count),
        ];
        self.launch("combos_sub", count as u64, args);
    }
}
