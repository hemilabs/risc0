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

//! Z15 (LEADS #22 segtail-mix-wide): `Hal::mix_poly_coeffs_fused` for Metal. One
//! `mix_poly_coeffs_fused` dispatch (sys/kernels/zkp/metal/mix.metal) replaces the 80 MB zero fill
//! of `combos` and the per-group `mix_poly_coeffs` read-modify-writes: every input of every group
//! is read once with 16-byte loads and every combo slot is written once. Shapes the kernel does
//! not take (more than 4 groups, a row count that is not a multiple of 4, unaligned buffers,
//! combo ids out of range) and `RISC0_METAL_MIX_FUSED=0` take the pre-Z15 path (zero fill, then
//! `mix_poly_coeffs` per group), which gives the same bits.

use risc0_core::field::{
    baby_bear::{BabyBearElem, BabyBearExtElem},
    Elem as _,
};

use super::{as_bytes, BufferImpl, KernelArg, MetalHal, MetalHash};
use crate::hal::{Buffer as _, Hal as _};

/// Revert knob: `RISC0_METAL_MIX_FUSED=0` keeps the zero fill + per-group mix (read per call).
pub(crate) const MIX_FUSED_ENV: &str = "RISC0_METAL_MIX_FUSED";

pub(crate) fn mix_fused_enabled() -> bool {
    !matches!(std::env::var(MIX_FUSED_ENV).as_deref(), Ok(v) if v.trim() == "0")
}

/// Largest number of inputs (all groups) the fused kernel takes: the power and entry tables are
/// `constant` buffers read uniformly by every thread.
pub(crate) const MIX_FUSED_MAX_INPUTS: usize = 4096;
/// Poly index bits in an entry's source word (the top 4 bits select the buffer).
const POLY_BITS: u32 = 28;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct MixFusedParams {
    count: u32,
    n_combos: u32,
    _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<MixFusedParams>() == 16);

/// Raw mix powers, (source, power) entries sorted by combo, combo boundaries.
pub(crate) type FusedTables = (Vec<[u32; 4]>, Vec<[u32; 2]>, Vec<u32>);

/// The host tables of one fused mix: raw mix powers, (source, power) entries sorted by combo
/// (stable), and the combo boundaries. `None` if the shape does not fit the kernel.
pub(crate) fn fused_tables(
    mix: &BabyBearExtElem,
    group_sizes: &[usize],
    combo_ids: &[u32],
    n_combos: usize,
) -> Option<FusedTables> {
    let total: usize = group_sizes.iter().sum();
    if group_sizes.len() > 4
        || total != combo_ids.len()
        || total > MIX_FUSED_MAX_INPUTS
        || n_combos == 0
        || n_combos > u32::MAX as usize
        || combo_ids.iter().any(|&c| c as usize >= n_combos)
        || group_sizes.iter().any(|&s| s >= 1 << POLY_BITS)
    {
        return None;
    }
    let mut pows = Vec::with_capacity(total);
    let mut cur = BabyBearExtElem::ONE;
    for _ in 0..total {
        pows.push(ext_words(&cur));
        cur *= *mix;
    }
    let mut begin = vec![0u32; n_combos + 1];
    for &c in combo_ids {
        begin[c as usize + 1] += 1;
    }
    for c in 0..n_combos {
        begin[c + 1] += begin[c];
    }
    let mut next = begin.clone();
    let mut entries = vec![[0u32; 2]; total];
    let mut j = 0usize;
    for (g, &size) in group_sizes.iter().enumerate() {
        for i in 0..size {
            let c = combo_ids[j] as usize;
            entries[next[c] as usize] = [(g as u32) << POLY_BITS | i as u32, j as u32];
            next[c] += 1;
            j += 1;
        }
    }
    Some((pows, entries, begin))
}

fn ext_words(x: &BabyBearExtElem) -> [u32; 4] {
    bytemuck::pod_read_unaligned(as_bytes(std::slice::from_ref(x)))
}

/// `Hal::mix_poly_coeffs_fused` body; `force_old` takes the zero fill + per-group path.
pub(crate) fn mix_poly_coeffs_fused<MH: MetalHash>(
    hal: &MetalHal<MH>,
    out: &BufferImpl<BabyBearExtElem>,
    mix: &BabyBearExtElem,
    inputs: &[(&BufferImpl<BabyBearElem>, &[u32])],
    count: usize,
    force_old: bool,
) -> bool {
    mix_poly_coeffs_fused_with(hal, out, mix, inputs, count, force_old, MIX_FUSED_KERNEL)
}

/// The fused kernel and the rows each thread owns.
pub(crate) const MIX_FUSED_KERNEL: (&str, usize) = ("mix_poly_coeffs_fused", 4);

/// [mix_poly_coeffs_fused] with an explicit kernel (benchmarks).
pub(crate) fn mix_poly_coeffs_fused_with<MH: MetalHash>(
    hal: &MetalHal<MH>,
    out: &BufferImpl<BabyBearExtElem>,
    mix: &BabyBearExtElem,
    inputs: &[(&BufferImpl<BabyBearElem>, &[u32])],
    count: usize,
    force_old: bool,
    kernel: (&str, usize),
) -> bool {
    assert!(count > 0 && out.size().is_multiple_of(count));
    let n_combos = out.size() / count;
    for (input, ids) in inputs {
        assert!(input.size() >= ids.len() * count);
    }
    let sizes: Vec<usize> = inputs.iter().map(|(_, ids)| ids.len()).collect();
    let ids: Vec<u32> = inputs
        .iter()
        .flat_map(|(_, ids)| ids.iter().copied())
        .collect();
    let fits = !force_old
        && count.is_multiple_of(kernel.1)
        && count <= u32::MAX as usize
        && out.as_device_ptr() & 15 == 0
        && inputs.iter().all(|(b, _)| b.as_device_ptr() & 15 == 0);
    let tables = if fits {
        fused_tables(mix, &sizes, &ids, n_combos)
    } else {
        None
    };
    let Some((pows, entries, begin)) = tables else {
        // The pre-Z15 path (prover.rs before Z15): zero fill, then one mix per group.
        hal.fill_u32(out, 0);
        let ids_buf = hal.copy_from_u32("which", &ids);
        let mut cur = BabyBearExtElem::ONE;
        let mut off = 0;
        for (input, gids) in inputs {
            hal.mix_poly_coeffs(
                out,
                &cur,
                mix,
                input,
                &ids_buf.slice(off, gids.len()),
                gids.len(),
                count,
            );
            cur *= mix.pow(gids.len());
            off += gids.len();
        }
        return false;
    };
    if ids.is_empty() {
        hal.fill_u32(out, 0);
        return true;
    }
    let pows = hal.copy_from_slice("mix_fused_pows", &pows);
    let entries = hal.copy_from_slice("mix_fused_entries", &entries);
    let begin = hal.copy_from_slice("mix_fused_begin", &begin);
    let params = [MixFusedParams {
        count: count as u32,
        n_combos: n_combos as u32,
        _pad: [0; 2],
    }];
    let input = |k: usize| inputs.get(k).unwrap_or(&inputs[0]).0.as_arg();
    let args = &[
        out.as_arg(),
        input(0),
        input(1),
        input(2),
        input(3),
        pows.as_arg(),
        entries.as_arg(),
        begin.as_arg(),
        KernelArg::Bytes(as_bytes(&params)),
    ];
    hal.launch(kernel.0, (count / kernel.1) as u64, args);
    true
}
