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

//! `combos_divide` on Metal (plan item Z5): the host side of `sys/kernels/zkp/metal/mdiv.metal`,
//! a port of `mdiv_supported` / `mdiv_run` in `sys/kernels/zkp/cuda/supra/mdiv.cuh` and of the
//! `supra_poly_divide_multi` dispatch in `supra/api.cu`.
//!
//! Semantics (identical to the CUDA HAL, for every input): each combo slice is divided by its
//! roots in order, as sequential sppark `div_by_x_minus_z<rotate = true>` calls would: after a
//! root, `a[k] = S_{k+1}` for `k < n - 1` and `a[n - 1] = S_0`, the remainder. The remainders (one
//! per root, in `chunks` order) are returned for `report_nonzero_remainders`. With zero
//! remainders this equals the CPU HAL, whose `poly_divide` leaves `a[n - 1] = 0`.
//!
//! Shapes:
//! * `n = ntiles * T`, `T = 256 * E`, `ntiles` a power of two, `1 <= ntiles <= 1024 * MD_MAXK`.
//!   CUDA requires `32 <= ntiles <= 1024` (po2 16..=21 at E = 8). Here `mdiv_scan` pads with zero
//!   tiles below 32 (one 32-thread threadgroup per slice: the "single-threadgroup" scan for po2
//!   11..=15) and takes `K = ntiles / 1024` tiles per thread above 1024 (the po2 22/23 "two-level"
//!   scan). The arithmetic is exact field arithmetic, so the result does not depend on the shape.
//! * E = 8 (CUDA's `MD_E`) below po2 17 and E = 16 from po2 17 up (Z9: E = 16 is 1.12-1.21x
//!   faster at po2 17-19, Z5 measured it from po2 20; `RISC0_METAL_MDIV_E16_MIN_PO2` moves the
//!   switch, 20 restores Z5's); `RISC0_METAL_MDIV_E=8|16` forces one. So po2 22 runs at E = 16
//!   with 1024 tiles; at E = 8 it takes the K = 2 scan (tested, selectable).
//! * Z9 CPU/GPU crossover: slices below `2^RISC0_METAL_MDIV_GPU_MIN_PO2` elements (default 16)
//!   use the host loop on the unified-memory buffer, which measured faster there (po2 11: 5.8x,
//!   13: 2.9x, 14: 1.7x, 15: ~1.0x; GPU 1.3x from po2 16; docs/metal-port/notes/Z9.md). `0`
//!   restores the pre-Z9 behaviour (the GPU whenever a shape exists).
//! * Anything else (po2 < 11, non-power-of-two lengths, `RISC0_METAL_MDIV=0`) uses the host
//!   loop with the same rotate semantics, as CUDA falls back to the per-root sppark path.
//!
//! Jobs carry element offsets into the two bound buffers (combos, scratch) instead of device
//! pointers, so no `use_resource` is needed (see the header of `mdiv.metal`).

use metal::MTLSize;
use rayon::prelude::*;
use risc0_core::field::{baby_bear::BabyBearExtElem, Elem as _};

use super::{as_bytes, BufferImpl, Grid, KernelArg, MetalHal, MetalHash};
use crate::{core::poly::poly_divide, hal::Buffer as _};

/// Threads per threadgroup of `mdiv_agg0` / `mdiv_apply` (CUDA `MD_NTH`).
pub(crate) const MD_NTH: usize = 256;
/// Jobs (combo slices) per launch (CUDA `MD_MAXJ`); more combos run as several groups.
pub(crate) const MD_MAXJ: usize = 8;
/// Maximum tiles per `mdiv_scan` thread (`MD_MAXK` in mdiv.metal).
pub(crate) const MD_MAXK: usize = 4;
const MD_NONE: u32 = u32::MAX;

/// One job (`MdJob` in mdiv.metal). Offsets are element indices into buffer 0 (`a`) or 1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MdJob {
    pub a: u32,
    pub agg: u32,
    pub carry: u32,
    pub rem: u32,
    pub prev_rem: u32,
    pub has_next: u32,
    pub _pad: [u32; 2],
    pub z: [u32; 4],
    pub znext: [u32; 4],
}

/// The `set_bytes` argument of every mdiv kernel (`MdJobs` in mdiv.metal).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MdJobs {
    pub j: [MdJob; MD_MAXJ],
    pub n: u32,
    pub ntiles: u32,
    pub k: u32,
    pub t: u32,
}

const _: () = assert!(std::mem::size_of::<MdJob>() == 64);
const _: () = assert!(std::mem::size_of::<MdJobs>() == 528);

/// Launch shape of one `mdiv_run`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MdShape {
    /// Elements per thread in agg0/apply (8 or 16).
    pub e: usize,
    /// Tile size `256 * e`.
    pub tile: usize,
    pub ntiles: usize,
    /// Threads of the `mdiv_scan` threadgroup (32..=1024).
    pub scan_nt: usize,
    /// Tiles per `mdiv_scan` thread.
    pub k: usize,
}

/// The shape for slices of `n` elements, or `None` if the kernels do not handle it (port of
/// `mdiv_supported`, widened as described in the module docs).
pub(crate) fn mdiv_shape(n: usize, e: usize) -> Option<MdShape> {
    if e != 8 && e != 16 {
        return None;
    }
    let tile = MD_NTH * e;
    if n == 0 || !n.is_multiple_of(tile) || n > u32::MAX as usize {
        return None;
    }
    let ntiles = n / tile;
    if !ntiles.is_power_of_two() || ntiles > 1024 * MD_MAXK {
        return None;
    }
    let scan_nt = ntiles.clamp(32, 1024);
    let k = ntiles.div_ceil(scan_nt);
    Some(MdShape {
        e,
        tile,
        ntiles,
        scan_nt,
        k,
    })
}

/// Slices of at least this many elements use E = 16 by default (Z5: 20-30% less GPU time than
/// E = 8 at po2 20, 21 and 22; Z9: 12-21% at po2 17, 18 and 19, ambiguous at 16; M5 Max).
pub(crate) const MD_E16_MIN_N: usize = 1 << 17;

/// Slices below this many elements use the host loop (Z9 crossover, see the module docs).
pub(crate) const MD_GPU_MIN_N: usize = 1 << 16;

fn env_po2(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&p| p < usize::BITS as usize)
}

/// The E for slices of `n` elements: `RISC0_METAL_MDIV_E` if set to 8 or 16, else 16 from
/// [MD_E16_MIN_N] (or `2^RISC0_METAL_MDIV_E16_MIN_PO2`) up and 8 (CUDA's `MD_E`) below.
pub(crate) fn mdiv_default_e(n: usize) -> usize {
    let e16_min = env_po2("RISC0_METAL_MDIV_E16_MIN_PO2").map_or(MD_E16_MIN_N, |p| 1 << p);
    match std::env::var("RISC0_METAL_MDIV_E")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("16") => 16,
        Some("8") => 8,
        _ if n >= e16_min => 16,
        _ => 8,
    }
}

/// Whether slices of `n` elements are large enough for the GPU path ([MD_GPU_MIN_N], or
/// `2^RISC0_METAL_MDIV_GPU_MIN_PO2`; 0 = always).
pub(crate) fn mdiv_gpu_size(n: usize) -> bool {
    n >= env_po2("RISC0_METAL_MDIV_GPU_MIN_PO2").map_or(MD_GPU_MIN_N, |p| 1 << p)
}

/// Whether the GPU path is enabled (`RISC0_METAL_MDIV=0` forces the host loop, like
/// `RISC0_CUDA_MDIV=0`).
fn mdiv_enabled() -> bool {
    !std::env::var("RISC0_METAL_MDIV").is_ok_and(|v| v.trim() == "0")
}

/// One dispatch of a plan.
#[derive(Clone, Debug)]
pub(crate) struct MdStep {
    pub kernel: String,
    pub groups: MTLSize,
    pub threads: MTLSize,
    pub jobs: MdJobs,
    /// `mdiv_final`'s job count (buffer 3).
    pub njobs: Option<u32>,
}

/// Every dispatch of a `combos_divide`, plus the scratch layout.
#[derive(Clone, Debug)]
pub(crate) struct MdPlan {
    pub steps: Vec<MdStep>,
    /// Scratch elements: `2 * min(MD_MAXJ, combos) * ntiles` aggregates/carries, then the
    /// remainders.
    pub scratch_len: usize,
    pub rems_off: usize,
    pub total_pows: usize,
}

fn words(x: BabyBearExtElem) -> [u32; 4] {
    let w = x.to_u32_words();
    [w[0], w[1], w[2], w[3]]
}

/// Build the dispatch list of `mdiv_run` (mdiv.cuh), once per group of up to `MD_MAXJ` combos
/// with at least one root (combos without roots are left untouched and have no remainders, as on
/// CUDA). `combos_len` is the combos buffer length in elements.
pub(crate) fn mdiv_plan(
    shape: MdShape,
    combos_len: usize,
    n: usize,
    chunks: &[(usize, Vec<BabyBearExtElem>)],
) -> MdPlan {
    let ntiles = shape.ntiles;
    // Remainder offsets in chunks order.
    let mut roff = Vec::with_capacity(chunks.len());
    let mut total_pows = 0;
    for (_, pows) in chunks {
        roff.push(total_pows);
        total_pows += pows.len();
    }
    let active: Vec<usize> = (0..chunks.len())
        .filter(|&c| !chunks[c].1.is_empty())
        .collect();
    let slots = active.len().min(MD_MAXJ);
    let rems_off = 2 * slots * ntiles;
    let scratch_len = rems_off + total_pows;
    assert!(
        scratch_len <= u32::MAX as usize,
        "mdiv scratch of {scratch_len} elements"
    );
    // Two jobs on one slice would run concurrently on the same elements (a race, silently wrong
    // results); the host loop asserts the same (Z8a).
    let mut seen = std::collections::HashSet::with_capacity(active.len());
    for &c in &active {
        let idx = chunks[c].0;
        assert!(seen.insert(idx), "combos_divide: combo {idx} listed twice");
        assert!(
            (idx + 1) * n <= combos_len && (idx + 1) * n <= u32::MAX as usize,
            "combos_divide: combo {idx} of {n} elements is outside the {combos_len}-element buffer"
        );
    }

    let job = |slot: usize, c: usize, r: usize| -> MdJob {
        let (idx, pows) = &chunks[c];
        let agg = slot * 2 * ntiles;
        let has_next = r + 1 < pows.len();
        MdJob {
            a: (idx * n) as u32,
            agg: agg as u32,
            carry: (agg + ntiles) as u32,
            rem: (rems_off + roff[c] + r) as u32,
            prev_rem: if r > 0 {
                (rems_off + roff[c] + r - 1) as u32
            } else {
                MD_NONE
            },
            has_next: has_next as u32,
            _pad: [0; 2],
            z: words(pows[r]),
            znext: words(if has_next { pows[r + 1] } else { pows[r] }),
        }
    };
    let empty = MdJobs {
        j: [MdJob::default(); MD_MAXJ],
        n: n as u32,
        ntiles: ntiles as u32,
        k: shape.k as u32,
        t: shape.tile as u32,
    };
    let tg = MTLSize::new(MD_NTH as u64, 1, 1);
    let agg0 = format!("mdiv_agg0_e{}", shape.e);
    let apply = format!("mdiv_apply_e{}", shape.e);
    let scan = format!("mdiv_scan_{}", shape.scan_nt);

    let mut steps = vec![];
    for group in active.chunks(MD_MAXJ) {
        let maxm = group.iter().map(|&c| chunks[c].1.len()).max().unwrap();
        let mut jobs = empty;
        for (slot, &c) in group.iter().enumerate() {
            jobs.j[slot] = job(slot, c, 0);
        }
        steps.push(MdStep {
            kernel: agg0.clone(),
            groups: MTLSize::new(ntiles as u64, group.len() as u64, 1),
            threads: tg,
            jobs,
            njobs: None,
        });
        for r in 0..maxm {
            // Jobs of this root layer; a combo keeps its slot (its agg/carry scratch).
            let mut jobs = empty;
            let mut nj = 0;
            for (slot, &c) in group.iter().enumerate() {
                if chunks[c].1.len() > r {
                    jobs.j[nj] = job(slot, c, r);
                    nj += 1;
                }
            }
            steps.push(MdStep {
                kernel: scan.clone(),
                groups: MTLSize::new(nj as u64, 1, 1),
                threads: MTLSize::new(shape.scan_nt as u64, 1, 1),
                jobs,
                njobs: None,
            });
            steps.push(MdStep {
                kernel: apply.clone(),
                groups: MTLSize::new(ntiles as u64, nj as u64, 1),
                threads: tg,
                jobs,
                njobs: None,
            });
        }
        let mut jobs = empty;
        for (slot, &c) in group.iter().enumerate() {
            let mut j = job(slot, c, 0);
            j.prev_rem = (rems_off + roff[c] + chunks[c].1.len() - 1) as u32;
            jobs.j[slot] = j;
        }
        steps.push(MdStep {
            kernel: "mdiv_final".into(),
            groups: MTLSize::new(1, 1, 1),
            threads: MTLSize::new(32, 1, 1),
            jobs,
            njobs: Some(group.len() as u32),
        });
    }
    MdPlan {
        steps,
        scratch_len,
        rems_off,
        total_pows,
    }
}

/// The host loop with the GPU semantics (rotate: the remainder lands in `a[n - 1]`): the
/// fallback for shapes the kernels do not take, and the reference in tests. Returns the
/// remainders in `chunks` order.
pub(crate) fn host_combos_divide(
    combos: &mut [BabyBearExtElem],
    chunks: &[(usize, Vec<BabyBearExtElem>)],
    n: usize,
) -> Vec<BabyBearExtElem> {
    if n == 0 {
        return chunks
            .iter()
            .flat_map(|(_, pows)| pows.iter().map(|_| BabyBearExtElem::ZERO))
            .collect();
    }
    let mut by_index: Vec<Option<usize>> = vec![None; combos.len() / n];
    for (c, (idx, _)) in chunks.iter().enumerate() {
        assert!(
            *idx < by_index.len(),
            "combos_divide: combo {idx} out of range"
        );
        assert!(
            by_index[*idx].is_none(),
            "combos_divide: combo {idx} listed twice"
        );
        by_index[*idx] = Some(c);
    }
    let mut rems: Vec<Vec<BabyBearExtElem>> = vec![vec![]; chunks.len()];
    let per_slice: Vec<(usize, Vec<BabyBearExtElem>)> = combos
        .par_chunks_exact_mut(n)
        .enumerate()
        .filter_map(|(idx, slice)| {
            let c = by_index[idx]?;
            let r = chunks[c]
                .1
                .iter()
                .map(|&pow| {
                    let rem = poly_divide(slice, pow);
                    slice[n - 1] = rem;
                    rem
                })
                .collect();
            Some((c, r))
        })
        .collect();
    for (c, r) in per_slice {
        rems[c] = r;
    }
    rems.into_iter().flatten().collect()
}

impl<MH: MetalHash> MetalHal<MH> {
    /// `combos_divide` (see the module docs); returns the remainders, one per root in `chunks`
    /// order. Ends synchronised (CUDA ends `supra_poly_divide_multi` with a stream sync).
    pub(crate) fn mdiv_combos_divide(
        &self,
        combos: &BufferImpl<BabyBearExtElem>,
        chunks: &[(usize, Vec<BabyBearExtElem>)],
        n: usize,
    ) -> Vec<BabyBearExtElem> {
        let shape = if mdiv_enabled() && mdiv_gpu_size(n) {
            mdiv_shape(n, mdiv_default_e(n))
        } else {
            None
        };
        self.mdiv_combos_divide_with(combos, chunks, n, shape)
    }

    /// [Self::mdiv_combos_divide] with an explicit shape (`None`: host loop).
    pub(crate) fn mdiv_combos_divide_with(
        &self,
        combos: &BufferImpl<BabyBearExtElem>,
        chunks: &[(usize, Vec<BabyBearExtElem>)],
        n: usize,
        shape: Option<MdShape>,
    ) -> Vec<BabyBearExtElem> {
        let total_pows: usize = chunks.iter().map(|(_, p)| p.len()).sum();
        if total_pows == 0 {
            return vec![];
        }
        let Some(shape) = shape else {
            let mut rems = vec![];
            combos.view_mut(|slice| rems = host_combos_divide(slice, chunks, n));
            return rems;
        };
        let plan = mdiv_plan(shape, combos.size(), n, chunks);
        let scratch = self.alloc_buffer::<BabyBearExtElem>("mdiv_scratch", plan.scratch_len);
        self.mdiv_encode(combos, &scratch, &plan);
        // Host read: commits and waits (the batch closes here, as CUDA syncs its stream).
        scratch.slice(plan.rems_off, plan.total_pows).to_vec()
    }

    /// Encode every step of `plan` into the open batch.
    pub(crate) fn mdiv_encode(
        &self,
        combos: &BufferImpl<BabyBearExtElem>,
        scratch: &BufferImpl<BabyBearExtElem>,
        plan: &MdPlan,
    ) {
        for step in &plan.steps {
            let pso = self.pipeline(&step.kernel);
            super::kernels::assert_simd_width(pso, &step.kernel);
            let jobs = [step.jobs];
            let mut args = vec![
                combos.as_arg(),
                scratch.as_arg(),
                KernelArg::Bytes(as_bytes(&jobs)),
            ];
            if let Some(nj) = step.njobs {
                args.push(KernelArg::Integer(nj));
            }
            self.dispatch(
                pso,
                &args,
                Grid::Groups {
                    groups: step.groups,
                    threads: step.threads,
                },
                &[],
            );
        }
    }
}

#[cfg(test)]
mod tests_z8a {
    //! Z8a review regressions for the mdiv host side.
    use risc0_core::field::{baby_bear::BabyBearExtElem, Elem as _};

    use super::*;
    use crate::hal::{metal::MetalHalSha256, Hal};

    fn root(k: u32) -> BabyBearExtElem {
        BabyBearExtElem::from_u32_words(&[k, 2 * k + 1, 3, 4])
    }

    /// A combo listed twice used to become two concurrent jobs on the same slice (a race with
    /// silently wrong output); the host loop already rejected it. Both paths now panic alike.
    #[test]
    #[should_panic(expected = "combos_divide: combo 1 listed twice")]
    fn plan_rejects_duplicate_combo() {
        let n = 1 << 13;
        let chunks = vec![(0, vec![root(1)]), (1, vec![root(2)]), (1, vec![root(3)])];
        mdiv_plan(mdiv_shape(n, 8).unwrap(), 3 * n, n, &chunks);
    }

    #[test]
    #[should_panic(expected = "combos_divide: combo 0 listed twice")]
    fn gpu_path_rejects_duplicate_combo() {
        let n = 1 << 13;
        let hal = MetalHalSha256::new();
        let combos = hal.alloc_extelem_zeroed("combos", 2 * n);
        let chunks = vec![(0, vec![root(1)]), (0, vec![root(2)])];
        hal.mdiv_combos_divide_with(&combos, &chunks, n, mdiv_shape(n, 8));
    }

    /// The host fallback keeps its own check (same message).
    #[test]
    #[should_panic(expected = "combos_divide: combo 0 listed twice")]
    fn host_path_rejects_duplicate_combo() {
        let n = 1 << 5;
        let mut combos = vec![BabyBearExtElem::ZERO; 2 * n];
        host_combos_divide(&mut combos, &[(0, vec![root(1)]), (0, vec![root(2)])], n);
    }
}
