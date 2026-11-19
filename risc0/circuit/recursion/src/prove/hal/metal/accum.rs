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

//! Recursion accumulation on the Metal HAL (docs/metal-port/PLAN.md 4.3, item C7).
//!
//! Port of `HostAccumContext` (recursion-sys `kernels/cuda/ffi.cu`, the
//! `SPPARK_NO_COOPERATIVE_LAUNCH` path: `fillOneExt` device fill, `computeAccum`,
//! `calcPrefixProducts` with the Chunks/Totals/Fixup trio, `verifyAccum`), encoded into the HAL's
//! open batch with no host wait in between, then one wait and the fault check:
//!
//! ```text
//! [fault := 0] -> fillOneExt(tmp, n) -> parStepComputeAccum -> prefixProductsChunks
//!   (-> prefixProductsTotals -> prefixProductsFixup when n > 2048) -> parStepVerifyAccum
//!   -> sync -> fault word
//! ```
//!
//! `n` is `work_cycles`; `tmp` (CUDA `g_accumCache.accum`, n FpExt) and the chunk totals
//! (ceil(n / 2048) FpExt) are HAL-owned scratch grown with 25 % headroom. The CPU reference is
//! `risc0_circuit_recursion_cpu_accum` (cxx/ffi.cpp `AccumContext`): compute per cycle,
//! `std::inclusive_scan` with `*`, verify per cycle. FpExt multiplication is exact and
//! associative, so the blocked scan gives the sequential scan's products bit for bit.
//!
//! `RISC0_METAL_REC_ACCUM_CHECK=1` (read once) re-runs every accum on the CPU over host copies of
//! the inputs and fails the proof on the first differing element (debug aid; costs the CPU accum
//! plus a copy of the data group per proof).

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    OnceLock,
};

use anyhow::{bail, Result};
use metal::MTLSize;
use risc0_circuit_recursion_sys::{risc0_circuit_recursion_cpu_accum, RawAccumBuffers};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    adapter::{CircuitInfo as _, TapsProvider as _},
    field::baby_bear::{BabyBearElem, BabyBearExtElem},
    hal::{
        metal::{BufferImpl, Grid, KernelArg, MetalHal, MetalHash},
        Buffer as _,
    },
};

use super::{grow, MetalCircuitHal, Phase};
use crate::{prove::hal::CircuitAccumulator, CircuitImpl, CIRCUIT};

/// Threads per `prefixProductsChunks` threadgroup (rec_kernels.metal `kScanThreads`); the kernel
/// must be dispatched with exactly this many.
const SCAN_THREADS: u64 = 256;
/// Elements per scan chunk (`kScanChunk` = 256 threads x 8).
const SCAN_CHUNK: u32 = 2048;

/// `RecAccumParams` (recursion-sys `kernels/metal/context.h`, 16 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct RecAccumParams {
    total_cycles: u32,
    work_cycles: u32,
    _pad0: u32,
    _pad1: u32,
}
const _: () = assert!(std::mem::size_of::<RecAccumParams>() == 16);

/// Environment variable enabling the CPU shadow check of every Metal accum.
pub(crate) const ACCUM_CHECK_ENV: &str = "RISC0_METAL_REC_ACCUM_CHECK";

/// Test override of [ACCUM_CHECK_ENV].
pub(crate) static FORCE_ACCUM_CHECK: AtomicBool = AtomicBool::new(false);
/// Accum shadow checks that matched the CPU (tests and logs).
pub(crate) static ACCUM_CHECKS_OK: AtomicUsize = AtomicUsize::new(0);

fn accum_check_enabled() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    FORCE_ACCUM_CHECK.load(Ordering::Relaxed)
        || *ENV.get_or_init(|| {
            matches!(
                std::env::var(ACCUM_CHECK_ENV).as_deref(),
                Ok(v) if !v.is_empty() && v != "0"
            )
        })
}

/// Number of 2048-element scan chunks of `n` elements.
pub(crate) fn scan_blocks(n: u32) -> u32 {
    n.div_ceil(SCAN_CHUNK)
}

/// Host copies of the accum inputs, for the shadow check.
struct AccumSnapshot {
    ctrl: Vec<BabyBearElem>,
    global: Vec<BabyBearElem>,
    data: Vec<BabyBearElem>,
    mix: Vec<BabyBearElem>,
    accum: Vec<BabyBearElem>,
}

impl AccumSnapshot {
    /// The CPU accum over the copies; returns (accum, global) afterwards.
    fn run_cpu(
        mut self,
        work_cycles: u32,
        total_cycles: u32,
    ) -> Result<(Vec<BabyBearElem>, Vec<BabyBearElem>)> {
        let buffers = RawAccumBuffers {
            ctrl: self.ctrl.as_ptr(),
            global: self.global.as_mut_ptr(),
            data: self.data.as_ptr(),
            mix: self.mix.as_ptr(),
            accum: self.accum.as_mut_ptr(),
        };
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_cpu_accum(&buffers, work_cycles, total_cycles)
        })?;
        Ok((self.accum, self.global))
    }
}

/// First index whose raw (Montgomery) words differ; INVALID compares as a word (the field's `==`
/// asserts validity).
fn first_mismatch(a: &[BabyBearElem], b: &[BabyBearElem]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter()
        .zip(b)
        .position(|(x, y)| x.as_u32_montgomery() != y.as_u32_montgomery())
}

impl<MH: MetalHash> MetalCircuitHal<MH> {
    /// Host-side argument checks (programming errors or a corrupt caller): the step kernels
    /// index `col * total_cycles + ((cycle - d) & (total_cycles - 1))` in every group, so the
    /// shapes must hold before anything is encoded.
    fn check_accum_shapes(
        work_cycles: u32,
        total_cycles: u32,
        sizes: [(&str, usize, usize); 5],
    ) -> Result<()> {
        if !total_cycles.is_power_of_two() {
            bail!("recursion Metal Accum: total_cycles {total_cycles} is not a power of two");
        }
        if work_cycles > u32::MAX - SCAN_CHUNK {
            bail!("recursion Metal Accum: work_cycles {work_cycles} exceeds the 32-bit scan range");
        }
        for (name, have, need) in sizes {
            if have < need {
                bail!(
                    "recursion Metal Accum: {name} has {have} elements, needs {need} for \
                     {total_cycles} cycles"
                );
            }
        }
        Ok(())
    }

    /// Encode fillOneExt -> parStepComputeAccum -> prefix trio -> parStepVerifyAccum into the open
    /// batch (no wait). `fault` must already be zeroed in the same batch.
    #[allow(clippy::too_many_arguments)]
    fn encode_accum(
        &self,
        work_cycles: u32,
        total_cycles: u32,
        ctrl: &BufferImpl<BabyBearElem>,
        global: &BufferImpl<BabyBearElem>,
        data: &BufferImpl<BabyBearElem>,
        mix: &BufferImpl<BabyBearElem>,
        accum: &BufferImpl<BabyBearElem>,
        fault: &BufferImpl<u32>,
    ) {
        let n = work_cycles;
        let nblocks = scan_blocks(n);
        let (tmp, totals) = {
            let mut scratch = self.scratch();
            let tmp: BufferImpl<BabyBearExtElem> = grow(
                self.hal(),
                &mut scratch.accum_tmp,
                "rec_accum_tmp",
                n as usize,
            );
            let totals: BufferImpl<BabyBearExtElem> = grow(
                self.hal(),
                &mut scratch.accum_totals,
                "rec_accum_totals",
                nblocks as usize,
            );
            (tmp, totals)
        };
        self.record(Phase::Accum);

        let hal = self.hal();
        let k = self.kernels();
        let params = RecAccumParams {
            total_cycles,
            work_cycles,
            _pad0: 0,
            _pad1: 0,
        };
        let params = bytemuck::bytes_of(&params);
        let step_args = |hal: &MetalHal<MH>, name: &str| {
            hal.dispatch(
                k.pipeline(name),
                &[
                    ctrl.as_arg(),
                    global.as_arg(),
                    data.as_arg(),
                    mix.as_arg(),
                    accum.as_arg(),
                    tmp.as_arg(),
                    fault.as_arg(),
                    KernelArg::Bytes(params),
                ],
                super::grid_1d(super::TgKind::Step, k.pipeline(name), n as u64),
                &[],
            )
        };

        // ffi.cu HostAccumContext::HostAccumContext (device_fill).
        hal.dispatch(
            k.pipeline("fillOneExt"),
            &[tmp.as_arg(), KernelArg::Integer(n)],
            Grid::Threads(n as u64),
            &[],
        );
        // computeAccum.
        step_args(hal, "parStepComputeAccum");
        // calcPrefixProducts (SPPARK_NO_COOPERATIVE_LAUNCH): exactly 256-thread groups.
        hal.dispatch(
            k.pipeline("prefixProductsChunks"),
            &[tmp.as_arg(), KernelArg::Integer(n), totals.as_arg()],
            Grid::Groups {
                groups: MTLSize::new(nblocks as u64, 1, 1),
                threads: MTLSize::new(SCAN_THREADS, 1, 1),
            },
            &[],
        );
        if nblocks > 1 {
            hal.dispatch(
                k.pipeline("prefixProductsTotals"),
                &[totals.as_arg(), KernelArg::Integer(nblocks)],
                Grid::Threads(1),
                &[],
            );
            hal.dispatch(
                k.pipeline("prefixProductsFixup"),
                &[tmp.as_arg(), KernelArg::Integer(n), totals.as_arg()],
                Grid::Threads((n - SCAN_CHUNK) as u64),
                &[],
            );
        }
        // verifyAccum.
        step_args(hal, "parStepVerifyAccum");
    }
}

impl<MH: MetalHash> CircuitAccumulator<MetalHal<MH>> for MetalCircuitHal<MH> {
    fn accumulate(
        &self,
        work_cycles: u32,
        total_cycles: u32,
        ctrl: &BufferImpl<BabyBearElem>,
        global: &BufferImpl<BabyBearElem>,
        data: &BufferImpl<BabyBearElem>,
        mix: &BufferImpl<BabyBearElem>,
        accum: &BufferImpl<BabyBearElem>,
    ) -> Result<()> {
        scope!("metal_accum");
        let _lock = risc0_zkp::hal::metal::exclusive();
        self.enter(Phase::Accum);
        let total = total_cycles as usize;
        Self::check_accum_shapes(
            work_cycles,
            total_cycles,
            [
                ("ctrl", ctrl.size(), CIRCUIT.ctrl_size() * total),
                ("global", global.size(), CircuitImpl::OUTPUT_SIZE),
                ("data", data.size(), CIRCUIT.data_size() * total),
                ("mix", mix.size(), CircuitImpl::MIX_SIZE),
                ("accum", accum.size(), CIRCUIT.accum_size() * total),
            ],
        )?;
        if work_cycles == 0 {
            // The CPU accum does nothing either.
            return Ok(());
        }

        // Shadow check: host copies taken before anything below is encoded (to_vec syncs).
        let snapshot = accum_check_enabled().then(|| AccumSnapshot {
            ctrl: ctrl.to_vec(),
            global: global.to_vec(),
            data: data.to_vec(),
            mix: mix.to_vec(),
            accum: accum.to_vec(),
        });

        let fault = self.fault_word();
        self.encode_accum(
            work_cycles,
            total_cycles,
            ctrl,
            global,
            data,
            mix,
            accum,
            &fault,
        );
        self.check_fault(&fault, Phase::Accum)?;
        self.prof_mark("accum");

        if let Some(snapshot) = snapshot {
            let t0 = std::time::Instant::now();
            let (cpu_accum, cpu_global) = snapshot.run_cpu(work_cycles, total_cycles)?;
            if let Some(i) = first_mismatch(&accum.to_vec(), &cpu_accum) {
                bail!(
                    "recursion Metal Accum: accum differs from the CPU accum at elem {i} (col {}, \
                     row {}), work_cycles {work_cycles}, total_cycles {total_cycles}",
                    i / total,
                    i % total
                );
            }
            if let Some(i) = first_mismatch(&global.to_vec(), &cpu_global) {
                bail!("recursion Metal Accum: global differs from the CPU accum at elem {i}");
            }
            ACCUM_CHECKS_OK.fetch_add(1, Ordering::Relaxed);
            tracing::info!(
                "recursion Metal accum == CPU accum (work_cycles {work_cycles}, {:.1} ms)",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "test"))]
mod c7_tests {
    use std::{
        collections::{HashMap, VecDeque},
        sync::{Arc, Mutex, OnceLock},
    };

    use rand::SeedableRng as _;
    use risc0_core::field::baby_bear::{BabyBear, BabyBearElem};
    use risc0_zkp::{
        core::hash::poseidon2::Poseidon2HashSuite,
        field::Elem as _,
        hal::{
            cpu::CpuHal,
            metal::{MetalHalPoseidon2, MetalHashPoseidon2},
            Hal,
        },
        ZK_CYCLES,
    };
    use test_log::test;

    use super::*;

    const INVALID: u32 = BabyBearElem::INVALID.as_u32_montgomery();

    fn raw(e: &BabyBearElem) -> u32 {
        e.as_u32_montgomery()
    }

    thread_local! {
        /// Wall time of the last `accumulate` call made by [Inputs::run] (no uploads/readback).
        static LAST_ACCUM_S: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    }
    use crate::prove::{
        hal::cpu::CpuCircuitHal, preflight::Preflight, witgen::WitnessGenerator,
        zkr::get_zkr_shared,
    };

    /// The real witness of test_recursion_circuit.zkr at `po2` (CPU witgen): ctrl, data, global
    /// (with `valid_or_zero` applied, as the prover's header does before the accum) and the
    /// program's code rows. Cached per po2.
    struct Witness {
        code_rows: u32,
        ctrl: Vec<BabyBearElem>,
        data: Vec<BabyBearElem>,
        global: Vec<BabyBearElem>,
    }

    fn witness(po2: usize) -> Arc<Witness> {
        static CACHE: OnceLock<Mutex<HashMap<usize, Arc<Witness>>>> = OnceLock::new();
        let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
        cache
            .entry(po2)
            .or_insert_with(|| {
                let program = get_zkr_shared("test_recursion_circuit.zkr", po2).unwrap();
                let input: VecDeque<u32> = (0..16u32).map(|i| i * 0x101 + 7).collect();
                let mut preflight = Preflight::new(input);
                for (cycle, row) in program.code_by_row().enumerate() {
                    preflight.step(cycle, row).unwrap();
                }
                let hal: CpuHal<BabyBear> = CpuHal::new(Poseidon2HashSuite::new_suite());
                let w = WitnessGenerator::new(&hal, &CpuCircuitHal, &program, &preflight, None)
                    .unwrap();
                let global = w
                    .global
                    .to_vec()
                    .iter()
                    .map(|e| e.valid_or_zero())
                    .collect();
                Arc::new(Witness {
                    code_rows: program.code_rows() as u32,
                    ctrl: w.ctrl.to_vec(),
                    data: w.data.to_vec(),
                    global,
                })
            })
            .clone()
    }

    /// The witness with its live rows repeated cyclically to fill `work` rows of ctrl and data
    /// (the test program has only a few hundred code rows; the accum steps of a row read that
    /// row's ctrl/data, the mix and the running product, so a tiled trace is as valid for the
    /// accum as the original).
    fn tile(w: &Witness, work: u32, total: u32) -> Arc<Witness> {
        let (t, rows) = (total as usize, w.code_rows as usize);
        let rep = |src: &[BabyBearElem], cols: usize| {
            let mut out = src.to_vec();
            for col in 0..cols {
                for row in rows..work as usize {
                    out[col * t + row] = src[col * t + row % rows];
                }
            }
            out
        };
        Arc::new(Witness {
            code_rows: work,
            ctrl: rep(&w.ctrl, CIRCUIT.ctrl_size()),
            data: rep(&w.data, CIRCUIT.data_size()),
            global: w.global.clone(),
        })
    }

    /// Accum inputs over a real witness: random mix, accum INVALID with random noise in the last
    /// ZK_CYCLES rows (as `WitnessGenerator::accum` leaves it), `work` live cycles.
    struct Inputs {
        work: u32,
        total: u32,
        w: Arc<Witness>,
        data: Option<Vec<BabyBearElem>>,
        mix: Vec<BabyBearElem>,
        accum: Vec<BabyBearElem>,
    }

    impl Inputs {
        /// `work` = None: the program's own code rows (what the prover passes).
        fn new(po2: usize, work: Option<u32>, seed: u64) -> Self {
            let w = witness(po2);
            let total = 1u32 << po2;
            let work = work.unwrap_or(w.code_rows);
            assert!(
                work as usize <= total as usize - ZK_CYCLES,
                "work {work} too large"
            );
            let w = if work > w.code_rows {
                tile(&w, work, total)
            } else {
                w
            };
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let t = total as usize;
            let mut accum = vec![BabyBearElem::INVALID; CIRCUIT.accum_size() * t];
            for col in 0..CIRCUIT.accum_size() {
                for row in t - ZK_CYCLES..t {
                    accum[col * t + row] = BabyBearElem::random(&mut rng);
                }
            }
            let mix = (0..CircuitImpl::MIX_SIZE)
                .map(|_| BabyBearElem::random(&mut rng))
                .collect();
            Self {
                work,
                total,
                w,
                data: None,
                mix,
                accum,
            }
        }

        fn data(&self) -> &[BabyBearElem] {
            self.data.as_deref().unwrap_or(&self.w.data)
        }

        /// Run `acc` on `hal`; returns (accum, global, data) after the call.
        #[allow(clippy::type_complexity)]
        fn run<H, C>(
            &self,
            hal: &H,
            acc: &C,
        ) -> Result<(Vec<BabyBearElem>, Vec<BabyBearElem>, Vec<BabyBearElem>)>
        where
            H: Hal<Elem = BabyBearElem>,
            C: CircuitAccumulator<H>,
        {
            let ctrl = hal.copy_from_elem("ctrl", &self.w.ctrl);
            let global = hal.copy_from_elem("global", &self.w.global);
            let data = hal.copy_from_elem("data", self.data());
            let mix = hal.copy_from_elem("mix", &self.mix);
            let accum = hal.copy_from_elem("accum", &self.accum);
            let t0 = std::time::Instant::now();
            acc.accumulate(self.work, self.total, &ctrl, &global, &data, &mix, &accum)?;
            LAST_ACCUM_S.set(t0.elapsed().as_secs_f64());
            Ok((accum.to_vec(), global.to_vec(), data.to_vec()))
        }
    }

    fn metal() -> (Arc<MetalHalPoseidon2>, MetalCircuitHal<MetalHashPoseidon2>) {
        let hal = Arc::new(MetalHalPoseidon2::new());
        let c = MetalCircuitHal::new(hal.clone()).unwrap();
        (hal, c)
    }

    /// Metal first (a fault returns an error), then the CPU (whose asserts abort), compared
    /// element by element.
    fn check_equivalence(po2: usize, work: Option<u32>) {
        let inputs = Inputs::new(
            po2,
            work,
            0xC7 ^ ((po2 as u64) << 40) ^ work.unwrap_or(0) as u64,
        );
        let (work, total) = (inputs.work, inputs.total);
        let (hal, m) = metal();
        hal.sync();
        hal.reset_peak_allocated_size();
        let before = hal.allocated_size();
        let t0 = std::time::Instant::now();
        let (m_accum, m_global, m_data) = inputs.run(hal.as_ref(), &m).unwrap();
        let metal_first_s = LAST_ACCUM_S.get();
        // Again on the same HAL (scratch and PSOs warm, as for every proof after the first).
        let again = inputs.run(hal.as_ref(), &m).unwrap();
        let metal_accum_s = LAST_ACCUM_S.get();
        assert!(
            first_mismatch(&again.0, &m_accum).is_none(),
            "metal accum not deterministic"
        );
        let metal_s = t0.elapsed().as_secs_f64();
        let peak = hal.peak_allocated_size();
        let cpu_hal: CpuHal<BabyBear> = CpuHal::new(Poseidon2HashSuite::new_suite());
        let t0 = std::time::Instant::now();
        let (c_accum, c_global, c_data) = inputs.run(&cpu_hal, &CpuCircuitHal).unwrap();
        let cpu_accum_s = LAST_ACCUM_S.get();
        let cpu_s = t0.elapsed().as_secs_f64();
        eprintln!(
            "C7 accum po2 {po2} work {work}: accumulate metal {:.2} ms (first call {:.2} ms), cpu {:.2} ms (with uploads \
             and readback: metal {:.1} ms, cpu {:.1} ms); device alloc before {before} B, peak \
             {peak} B",
            metal_accum_s * 1e3,
            metal_first_s * 1e3,
            cpu_accum_s * 1e3,
            metal_s * 1e3,
            cpu_s * 1e3,
        );
        let t = total as usize;
        if let Some(i) = first_mismatch(&m_accum, &c_accum) {
            panic!(
                "po2 {po2} work {work}: accum differs at elem {i} (col {}, row {}): metal \
                 0x{:08x} cpu 0x{:08x}",
                i / t,
                i % t,
                raw(&m_accum[i]),
                raw(&c_accum[i])
            );
        }
        assert!(first_mismatch(&m_global, &c_global).is_none(), "global");
        assert!(first_mismatch(&m_data, &c_data).is_none(), "data");
        // The accum wrote something in the live rows (not every cell: the steps leave unused
        // accum columns INVALID, on the CPU too), nothing in the dead rows, and the noise rows
        // are untouched.
        let mut written = 0usize;
        for col in 0..CIRCUIT.accum_size() {
            for row in 0..work as usize {
                written += (raw(&m_accum[col * t + row]) != INVALID) as usize;
            }
            for row in work as usize..t - ZK_CYCLES {
                assert_eq!(raw(&m_accum[col * t + row]), INVALID, "col {col} row {row}");
            }
            for row in t - ZK_CYCLES..t {
                assert_eq!(
                    raw(&m_accum[col * t + row]),
                    raw(&inputs.accum[col * t + row])
                );
            }
        }
        assert!(
            written >= work as usize,
            "only {written} accum cells written"
        );
    }

    /// n = 1, 2047, 2048, 2049 (scan edge cases: one partial chunk, one full chunk, a second
    /// chunk of one element) over the real po2-16 witness (truncated, or tiled past its code
    /// rows).
    #[test]
    fn accum_matches_cpu_scan_edges() {
        let _heavy = crate::prove::hal::metal::heavy_test_lock();
        for n in [1, 2, 2047, 2048, 2049, 4097] {
            check_equivalence(16, Some(n));
        }
        // Many chunks with a partial last one.
        check_equivalence(17, Some(100_001));
    }

    /// The program's own work cycles at po2 16 and 18, and 2^18 live cycles (tiled) at po2 19,
    /// plus the realistic lift/join shape 2^18 - ZK_CYCLES at po2 18.
    #[test]
    fn accum_matches_cpu_full() {
        let _heavy = crate::prove::hal::metal::heavy_test_lock();
        check_equivalence(16, None);
        check_equivalence(18, None);
        check_equivalence(18, Some((1 << 18) - ZK_CYCLES as u32));
        check_equivalence(19, Some(1 << 18));
    }

    #[test]
    fn accum_scan_blocks() {
        assert_eq!(scan_blocks(1), 1);
        assert_eq!(scan_blocks(2047), 1);
        assert_eq!(scan_blocks(2048), 1);
        assert_eq!(scan_blocks(2049), 2);
        assert_eq!(scan_blocks(1 << 18), 128);
    }

    #[test]
    fn accum_work_zero_is_noop() {
        let inputs = Inputs::new(16, Some(0), 1);
        let (hal, m) = metal();
        let (accum, _, _) = inputs.run(hal.as_ref(), &m).unwrap();
        assert!(first_mismatch(&accum, &inputs.accum).is_none());
    }

    /// An INVALID element in a live data row makes the steps' c1_bad fire: the accum returns an
    /// error naming the phase instead of a wrong witness, and the HAL stays usable.
    #[test]
    fn accum_fault_on_invalid_input() {
        let _heavy = crate::prove::hal::metal::heavy_test_lock();
        let (hal, m) = metal();
        let mut inputs = Inputs::new(16, None, 7);
        let t = inputs.total as usize;
        let mut data = inputs.w.data.clone();
        for col in 0..CIRCUIT.data_size() {
            data[col * t + 100] = BabyBearElem::INVALID;
        }
        inputs.data = Some(data);
        let err = inputs.run(hal.as_ref(), &m).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Accum") && msg.contains("kernel fault"),
            "{msg}"
        );
        inputs.data = None;
        inputs.run(hal.as_ref(), &m).unwrap();
    }

    /// The kernels' own guard: work_cycles > total_cycles faults (thread 0) instead of indexing
    /// past the groups. Host shape checks reject non-power-of-two totals and short buffers.
    #[test]
    fn accum_fault_on_bad_shape() {
        let _heavy = crate::prove::hal::metal::heavy_test_lock();
        let (hal, m) = metal();
        let words = |n: usize| hal.copy_from_elem("x", &vec![BabyBearElem::ZERO; n]);
        let (ctrl, data, accum) = (
            words(CIRCUIT.ctrl_size() * 4),
            words(CIRCUIT.data_size() * 4),
            words(CIRCUIT.accum_size() * 4),
        );
        let global = words(CircuitImpl::OUTPUT_SIZE);
        let mix = words(CircuitImpl::MIX_SIZE);
        let err = m
            .accumulate(5, 4, &ctrl, &global, &data, &mix, &accum)
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("kernel fault 0x00000001"),
            "{err:#}"
        );
        let err = m
            .accumulate(3, 3, &ctrl, &global, &data, &mix, &accum)
            .unwrap_err();
        assert!(format!("{err:#}").contains("power of two"), "{err:#}");
        let err = m
            .accumulate(4, 8, &ctrl, &global, &data, &mix, &accum)
            .unwrap_err();
        assert!(format!("{err:#}").contains("needs"), "{err:#}");
    }

    /// Real traces end to end: test_recursion_circuit.zkr proven through the Metal HAL with the
    /// shadow check on, so the proof's accum is compared with the CPU accum over the same inputs
    /// (lift/join: risc0-zkvm host::recursion tests with RISC0_METAL_REC_ACCUM_CHECK=1).
    #[test]
    fn accum_real_trace_shadow_check() {
        let _heavy = crate::prove::hal::metal::heavy_test_lock();
        let _guard = crate::prove::hal::metal::global_state_test_lock();
        for po2 in [16, 18] {
            let program = get_zkr_shared("test_recursion_circuit.zkr", po2).unwrap();
            let input: VecDeque<u32> = (0..16u32).map(|i| i * 0x101 + 7).collect();
            let prover = super::super::recursion_prover("poseidon2").unwrap();
            let probe = MetalHalPoseidon2::new();
            probe.reset_peak_allocated_size();
            let before = ACCUM_CHECKS_OK.load(Ordering::Relaxed);
            FORCE_ACCUM_CHECK.store(true, Ordering::Relaxed);
            let r = prover.prove(&program, input.clone(), None);
            FORCE_ACCUM_CHECK.store(false, Ordering::Relaxed);
            let m = r.unwrap();
            assert!(ACCUM_CHECKS_OK.load(Ordering::Relaxed) > before);
            let c = crate::prove::hal::cpu::recursion_prover("poseidon2")
                .unwrap()
                .prove(&program, input, None)
                .unwrap();
            assert_eq!(m.output, c.output);
            eprintln!(
                "C7 real trace po2 {po2}: shadow check ok; accum phase alloc {} B, proof peak {} B",
                super::super::phase_alloc()[1],
                probe.peak_allocated_size()
            );
        }
    }
}
