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

//! R9 tests: the Metal accum path against the CPU circuit HAL. R13: the GPU prefix scans.

use std::{cell::Cell, rc::Rc, sync::Arc, time::Instant};

use rand::Rng;
use risc0_binfmt::{MemoryImage, Program};
use risc0_core::field::Elem as _;
use risc0_zkp::{
    adapter::CircuitInfo as _,
    core::hash::poseidon2::Poseidon2HashSuite,
    hal::{cpu::CpuHal, metal::MetalHalPoseidon2, Buffer as _},
};
use test_log::test;

use super::*;
use crate::prove::hal::metal::MetalCircuitHalPoseidon2;
use crate::{
    execute::testutil::{self, NullSyscall, DEFAULT_SESSION_LIMIT},
    prove::{
        hal::{cpu::CpuCircuitHal, StepMode},
        witgen::{PreflightResults, WitnessGenerator},
    },
    zirgen::{
        circuit::{ExtVal, Val, REGCOUNT_ACCUM, REGCOUNT_DATA, REGCOUNT_GLOBAL, REGCOUNT_MIX},
        CircuitImpl,
    },
    MAX_INSN_CYCLES,
};

// ---------------------------------------------------------------------------------------------
// perm
// ---------------------------------------------------------------------------------------------

/// The reference order: CUDA's `majorKeys` keys, stably sorted (what cub's stable LSD radix sort
/// over the key bits in use gives).
fn reference_perm(majors: &[u8], win: u32) -> Vec<u32> {
    let key = |i: u32| (if win < 32 { (i >> win) << 8 } else { 0 }) | majors[i as usize] as u32;
    let mut idx: Vec<u32> = (0..majors.len() as u32).collect();
    idx.sort_by_key(|&i| key(i)); // stable
    idx
}

fn cycles_with_majors(majors: &[u8]) -> Vec<RawPreflightCycle> {
    majors
        .iter()
        .map(|&major| RawPreflightCycle {
            state: 0,
            pc: 0,
            major,
            minor: 0,
            machine_mode: 0,
            padding: 0,
            user_cycle: 0,
            txn_idx: 0,
            paging_idx: 0,
            bigint_idx: 0,
            diff_count: [0; 2],
        })
        .collect()
}

fn check_perm(majors: &[u8], win: u32) {
    let cycles = cycles_with_majors(majors);
    let mut perm = vec![u32::MAX; majors.len() + 3];
    build_perm(&cycles, win, &mut perm);
    let n = majors.len();
    assert_eq!(&perm[n..], &[u32::MAX; 3], "build_perm wrote past n");
    assert_eq!(
        &perm[..n],
        reference_perm(majors, win).as_slice(),
        "n {n} win {win}"
    );
}

#[test]
fn perm_equivalence() {
    let mut rng = rand::rng();
    // Random majors (all 256 values, and a realistic handful), several sizes and windows,
    // including partial last windows and a window wider than the trace.
    for &n in &[1usize, 2, 5, 255, 2047, 2048, 2049, (1 << 14) + 3, 100_000] {
        for &win in &[4u32, 11, 12, 13, 14, 24] {
            let all: Vec<u8> = (0..n).map(|_| rng.random()).collect();
            check_perm(&all, win);
            let few: Vec<u8> = (0..n).map(|_| rng.random_range(0..19u8)).collect();
            check_perm(&few, win);
        }
    }
    // Constant majors: the identity.
    let same = vec![7u8; 5000];
    let cycles = cycles_with_majors(&same);
    let mut perm = vec![0; 5000];
    build_perm(&cycles, 11, &mut perm);
    assert!(perm.iter().enumerate().all(|(i, &p)| p == i as u32));

    // A real trace (the majors of an executed segment).
    let seg = segment(testutil::kernel::simple_loop(20_000), 15);
    let trace = seg.preflight(ExtVal::random(&mut rng)).unwrap();
    let majors: Vec<u8> = trace.cycles.iter().map(|c| c.major).collect();
    for win in 11..=14 {
        check_perm(&majors, win);
    }
}

#[test]
fn sort_window_env_values() {
    assert_eq!(parse_sort_win(None), Some(DEFAULT_SORT_WIN));
    assert_eq!(parse_sort_win(Some("")), Some(DEFAULT_SORT_WIN));
    assert_eq!(parse_sort_win(Some("0")), None);
    assert_eq!(parse_sort_win(Some("off")), None);
    assert_eq!(parse_sort_win(Some("11")), Some(11));
    assert_eq!(parse_sort_win(Some("2")), Some(4));
    assert_eq!(parse_sort_win(Some("99")), Some(24));
    assert_eq!(parse_sort_win(Some("x")), Some(DEFAULT_SORT_WIN));
    assert!(super::super::phase_listed("witgen, accum", "accum"));
    assert!(super::super::phase_listed("ACCUM", "accum"));
    assert!(!super::super::phase_listed("witgen,eval", "accum"));
    assert_eq!(user_accum_split(), 23);
}

// ---------------------------------------------------------------------------------------------
// scans
// ---------------------------------------------------------------------------------------------

#[test]
fn fp_scan_matches_field() {
    let mut rng = rand::rng();
    for &n in &[1usize, 255, 256, (1 << 20) + 7] {
        let vals: Vec<Val> = (0..n).map(|_| Val::random(&mut rng)).collect();
        let mut want = Vec::with_capacity(n);
        let mut acc = Val::ZERO;
        for v in &vals {
            acc += *v;
            want.push(acc);
        }
        let mut got: Vec<u32> = Val::as_u32_slice(&vals).to_vec();
        fp_inclusive_scan(&mut got);
        assert_eq!(got.as_slice(), Val::as_u32_slice(&want), "n {n}");
    }
    let mut empty: [u32; 0] = [];
    fp_inclusive_scan(&mut empty);
}

// ---------------------------------------------------------------------------------------------
// accum vs CPU
// ---------------------------------------------------------------------------------------------

fn segment(program: Program, po2: usize) -> crate::execute::segment::Segment {
    let image = MemoryImage::new_kernel(program);
    let session = testutil::execute(
        image,
        po2,
        MAX_INSN_CYCLES,
        DEFAULT_SESSION_LIMIT,
        &NullSyscall,
        None,
    )
    .unwrap();
    session.segments.into_iter().next().unwrap()
}

fn riscv_test(name: &str) -> Program {
    use std::io::Read;

    use flate2::read::GzDecoder;
    use tar::Archive;

    let bytes = include_bytes!("../../../testdata/riscv-tests.tgz");
    let mut tar = Archive::new(GzDecoder::new(&bytes[..]));
    for entry in tar.entries().unwrap() {
        let mut entry = entry.unwrap();
        if entry.header().entry_type().is_file()
            && entry.path().unwrap().file_name().unwrap().to_str().unwrap() == name
        {
            let mut elf = Vec::new();
            entry.read_to_end(&mut elf).unwrap();
            return Program::load_elf(&elf, u32::MAX).unwrap();
        }
    }
    panic!("no riscv test {name}");
}

/// Accum variants of the Metal circuit HAL: the GPU path with a given perm window (`None`: no
/// perm) and GPU scans (R13), the same with R9's CPU scans (`RISC0_METAL_ACCUM_SCAN=cpu`), or
/// the CPU fallback (`RISC0_METAL_RV32IM_CPU_PHASES=accum`).
#[derive(Clone, Copy, Debug)]
enum Variant {
    Gpu(Option<u32>),
    GpuCpuScan(Option<u32>),
    CpuFallback,
}

struct WithVariant<'a>(&'a MetalCircuitHalPoseidon2, Variant);

thread_local! {
    /// Wall time (ms) of the last `WithVariant::step_accum` call, from an idle GPU queue.
    static LAST_STEP_MS: Cell<f64> = const { Cell::new(0.0) };
}

impl CircuitAccumulator<MetalHalPoseidon2> for WithVariant<'_> {
    fn step_accum(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<MetalHalPoseidon2>,
        accum: &MetaBuffer<MetalHalPoseidon2>,
        global: &MetaBuffer<MetalHalPoseidon2>,
        mix: &MetaBuffer<MetalHalPoseidon2>,
    ) -> Result<()> {
        // Start from an idle queue so the time is the step's own (scatter/mix upload excluded).
        self.0.hal.sync();
        marks::take();
        let t = Instant::now();
        let res = match self.1 {
            Variant::Gpu(win) => {
                self.0
                    .step_accum_metal(preflight, data, accum, global, mix, win, ScanMode::Gpu)
            }
            Variant::GpuCpuScan(win) => {
                self.0
                    .step_accum_metal(preflight, data, accum, global, mix, win, ScanMode::Cpu)
            }
            Variant::CpuFallback => self.0.step_accum_cpu(preflight, data, accum, global, mix),
        };
        LAST_STEP_MS.with(|c| c.set(t.elapsed().as_secs_f64() * 1e3));
        res
    }

    fn step_accum_zeroizes(&self) -> bool {
        !matches!(self.1, Variant::CpuFallback)
    }
}

fn words(v: &[Val]) -> &[u32] {
    Val::as_u32_slice_unchecked(v)
}

/// First differing (row, col, cpu, metal) of two column-major `rows`-row buffers.
fn first_diff(cpu: &[u32], metal: &[u32], rows: usize) -> Option<(usize, usize, u32, u32)> {
    assert_eq!(cpu.len(), metal.len());
    cpu.iter()
        .zip(metal)
        .position(|(a, b)| a != b)
        .map(|i| (i % rows, i / rows, cpu[i], metal[i]))
}

fn count_diffs(cpu: &[u32], metal: &[u32]) -> usize {
    cpu.iter().zip(metal).filter(|(a, b)| a != b).count()
}

/// The CPU circuit HAL's witness and accum (the reference) for one segment.
struct Reference {
    preflight: PreflightResults,
    rows: usize,
    cycles: Vec<RawPreflightCycle>,
    data: Vec<Val>,
    global: Vec<Val>,
    mix: Vec<Val>,
    accum: Vec<Val>,
}

fn reference(seg: &crate::execute::segment::Segment) -> Reference {
    let mut rng = rand::rng();
    let rand_z = ExtVal::random(&mut rng);
    let preflight = PreflightResults::new(seg, rand_z).unwrap();
    let hal = Rc::new(CpuHal::new(Poseidon2HashSuite::new_suite()));
    let mut wg = WitnessGenerator::new(
        hal.as_ref(),
        &CpuCircuitHal,
        preflight.clone(),
        StepMode::Parallel,
    )
    .unwrap();
    let mix: Vec<Val> = (0..CircuitImpl::MIX_SIZE)
        .map(|_| Val::random(&mut rng))
        .collect();
    let data = wg.data.as_ref().unwrap().buf.to_vec();
    let global = wg.global.buf.to_vec();
    let rows = wg.cycles();
    let cycles = wg.trace.cycles.clone();
    wg.accum(hal.as_ref(), &CpuCircuitHal, &mix).unwrap();
    let accum = wg.accum.as_ref().unwrap().buf.to_vec();
    Reference {
        preflight,
        rows,
        cycles,
        data,
        global,
        mix,
        accum,
    }
}

/// Runs witgen on Metal (whatever path R8 left it on), replaces data/global with the CPU
/// reference so this test isolates accum, then accumulates with `variant`. Returns the accum
/// buffer, the phase's wall time and the peak GPU allocation during it.
fn metal_accum(
    hal: &Arc<MetalHalPoseidon2>,
    circuit_hal: &MetalCircuitHalPoseidon2,
    r: &Reference,
    variant: Variant,
) -> (Vec<Val>, f64, u64, u64) {
    let mut wg = WitnessGenerator::new(
        hal.as_ref(),
        circuit_hal,
        r.preflight.clone(),
        StepMode::Parallel,
    )
    .unwrap();
    let data = wg.data.as_ref().unwrap();
    let (got_data, got_global) = (data.buf.to_vec(), wg.global.buf.to_vec());
    if words(&got_data) != words(&r.data) || words(&got_global) != words(&r.global) {
        tracing::warn!(
            "Metal witgen differs from CPU (data {} / global {} cells); using the CPU witness \
             to test accum alone",
            count_diffs(words(&got_data), words(&r.data)),
            count_diffs(words(&got_global), words(&r.global)),
        );
    }
    data.buf.view_mut(|v| v.copy_from_slice(&r.data));
    wg.global.buf.view_mut(|v| v.copy_from_slice(&r.global));
    hal.sync();
    let before = hal.allocated_size();
    hal.reset_peak_allocated_size();
    let t = Instant::now();
    wg.accum(hal.as_ref(), &WithVariant(circuit_hal, variant), &r.mix)
        .unwrap();
    hal.sync();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let peak = hal.peak_allocated_size();
    (wg.accum.as_ref().unwrap().buf.to_vec(), ms, before, peak)
}

fn assert_accum_eq(what: &str, rows: usize, cpu: &[Val], metal: &[Val]) {
    assert_eq!(cpu.len(), rows * REGCOUNT_ACCUM);
    assert_eq!(metal.len(), cpu.len());
    if let Some((row, col, a, b)) = first_diff(words(cpu), words(metal), rows) {
        let split = user_accum_split() as usize;
        let kind = if col < split {
            "user"
        } else if col >= REGCOUNT_ACCUM - 4 {
            "scanned"
        } else {
            "finalized machine"
        };
        panic!(
            "{what}: accum differs in {} of {} cells; first at row {row}, col {col} ({kind}): \
             cpu {a:#010x} metal {b:#010x}",
            count_diffs(words(cpu), words(metal)),
            cpu.len()
        );
    }
    assert!(!words(metal).contains(&0xffff_ffff), "{what}: INVALID left");
}

/// Every accum column (user, finalized machine, scanned) of the Metal path equals the CPU circuit
/// HAL's, for several programs and perm variants, repeated `R9_ACCUM_REPEAT` times (default 2)
/// with fresh rand_z/mix each time.
#[test]
fn accum_matches_cpu() {
    let repeat: usize = std::env::var("R9_ACCUM_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let programs: Vec<(&str, Program, usize)> = vec![
        ("basic", testutil::kernel::basic(), 20),
        ("multi_read", testutil::kernel::multi_read(), 20),
        ("riscv rem", riscv_test("rem"), 20),
        ("simple_loop", testutil::kernel::simple_loop(60_000), 16),
        ("riscv mul", riscv_test("mul"), 20),
        ("riscv divu", riscv_test("divu"), 20),
        ("riscv sb", riscv_test("sb"), 20),
        ("riscv lhu", riscv_test("lhu"), 20),
    ];
    let hal = Arc::new(MetalHalPoseidon2::new());
    let circuit_hal = MetalCircuitHalPoseidon2::new(hal.clone()).unwrap();
    for (name, program, po2) in programs {
        let seg = segment(program, po2);
        for rep in 0..repeat {
            let r = reference(&seg);
            let rows = r.rows;
            for variant in [
                Variant::Gpu(Some(DEFAULT_SORT_WIN)),
                Variant::Gpu(Some(11)),
                Variant::Gpu(None),
                Variant::GpuCpuScan(Some(DEFAULT_SORT_WIN)),
                Variant::CpuFallback,
            ] {
                let (metal, ms, _, _) = metal_accum(&hal, &circuit_hal, &r, variant);
                tracing::info!("{name} po2 {} rep {rep} {variant:?}: {ms:.1} ms", seg.po2);
                assert_accum_eq(
                    &format!("{name} (po2 {}, rep {rep}, {variant:?})", seg.po2),
                    rows,
                    &r.accum,
                    &metal,
                );
            }
        }
    }
}

/// A corrupt perm is reported through the fault word, not by reading out of bounds.
#[test]
fn bad_perm_faults() {
    let hal = Arc::new(MetalHalPoseidon2::new());
    let circuit_hal = MetalCircuitHalPoseidon2::new(hal.clone()).unwrap();
    let ctx = circuit_hal.ctx();
    ctx.require(Phase::Accum).unwrap();
    let rows = 1 << 10;
    let mk = |name, rows: usize, cols: usize| MetaBuffer::<MetalHalPoseidon2> {
        buf: hal.alloc_buffer::<Val>(name, rows * cols),
        rows,
        cols,
        checked: false,
    };
    let data = mk("data", rows, REGCOUNT_DATA);
    let accum = mk("accum", rows, REGCOUNT_ACCUM);
    let global = mk("global", 1, REGCOUNT_GLOBAL);
    let mix = mk("mix", 1, REGCOUNT_MIX);
    for m in [&data, &accum, &global, &mix] {
        m.buf.view_mut(|v: &mut [Val]| v.fill(Val::ZERO));
    }
    let perm: Vec<u32> = (0..rows as u32).map(|i| i ^ (rows as u32)).collect();
    let perm_buf = hal.copy_from_slice("bad_perm", &perm);
    let fault = hal.alloc_buffer::<u32>("fault", 1);
    hal.blit_fill(fault.as_buf(), 0, 4, 0);
    let params = AccumParams {
        count: rows as u32,
        has_perm: 1,
        data_rows: rows as u32,
        data_cols: data.cols as u32,
        accum_rows: rows as u32,
        accum_cols: accum.cols as u32,
        global_rows: 1,
        global_cols: global.cols as u32,
        mix_rows: 1,
        mix_cols: mix.cols as u32,
        _pad: [0; 2],
    };
    hal.dispatch(
        ctx.pipeline("stepAccumA"),
        &[
            data.buf.as_arg(),
            accum.buf.as_arg(),
            global.buf.as_arg(),
            mix.buf.as_arg(),
            perm_buf.as_arg(),
            fault.as_arg(),
            KernelArg::Bytes(as_bytes(&[params])),
        ],
        Grid::Threads(rows as u64),
        &[],
    );
    hal.sync();
    assert_eq!(fault.to_vec()[0] & FAULT_ACCUM_PERM, FAULT_ACCUM_PERM);
}

/// Measurement (not an assertion): accum phase time per perm variant and the GPU memory peak, at
/// po2 16/18/20 (`R9_MEASURE_PO2=16,18,20`). Run alone (`--ignored --test-threads=1`): the peak is
/// process-wide.
#[test]
#[ignore]
fn accum_measure() {
    let po2s: Vec<usize> = std::env::var("R9_MEASURE_PO2")
        .unwrap_or_else(|_| "16,18,20".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let hal = Arc::new(MetalHalPoseidon2::new());
    let circuit_hal = MetalCircuitHalPoseidon2::new(hal.clone()).unwrap();
    eprintln!("working-set limit {} MB", hal.working_set_limit() >> 20);
    for po2 in po2s {
        // ~3 cycles per iteration: a full first segment of 2^po2 rows.
        let seg = segment(testutil::kernel::simple_loop(1u32 << po2), po2);
        let r = reference(&seg);
        let rows = r.rows;
        let cycles = &r.cycles;
        for win in 11..=14 {
            let mut perm = vec![0u32; cycles.len()];
            let t = Instant::now();
            build_perm(cycles, win, &mut perm);
            eprintln!(
                "po2 {po2}: build_perm win {win}: {:.2} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        let mut cols: Vec<u32> = words(&r.accum)[(REGCOUNT_ACCUM - 4) * rows..].to_vec();
        let t = Instant::now();
        cols.par_chunks_mut(rows).for_each(fp_inclusive_scan);
        eprintln!(
            "po2 {po2}: 4 CPU scans: {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        let variants = [
            Variant::Gpu(None),
            Variant::Gpu(Some(11)),
            Variant::Gpu(Some(12)),
            Variant::Gpu(Some(13)),
            Variant::Gpu(Some(14)),
            Variant::CpuFallback,
        ];
        for variant in variants {
            for rep in 0..3 {
                let (metal, ms, before, peak) = metal_accum(&hal, &circuit_hal, &r, variant);
                assert_accum_eq(&format!("po2 {po2} {variant:?}"), rows, &r.accum, &metal);
                eprintln!(
                    "po2 {po2} (rows {rows}) {variant:?} rep {rep}: accum phase {ms:.1} ms, \
                     GPU allocated before {} MB, peak during {} MB",
                    before >> 20,
                    peak >> 20
                );
            }
        }
    }
}

/// Measurement: prove one full 2^po2 segment (`R9_PROVE_PO2`, default 18) with the default Metal
/// segment prover (accum on the GPU path), verify the seal, and print the process-wide GPU memory
/// peak. With `RISC0_METAL_MAX_WORKING_SET_BYTES` it shows whether that shape fits the budget.
#[test]
#[ignore]
fn prove_segment_measure() {
    let po2: usize = std::env::var("R9_PROVE_PO2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(18);
    let seg = segment(testutil::kernel::simple_loop(1u32 << po2), po2);
    assert_eq!(seg.po2 as usize, po2);
    let hal = MetalHalPoseidon2::new();
    hal.reset_peak_allocated_size();
    eprintln!("working-set limit {} MB", hal.working_set_limit() >> 20);
    let prover = crate::prove::segment_prover().unwrap();
    let t = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prover.prove(&seg)));
    let ms = t.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "po2 {po2}: prove {ms:.0} ms, GPU peak {} MB",
        hal.peak_allocated_size() >> 20
    );
    match result {
        Ok(Ok(seal)) => {
            crate::verify(&seal).unwrap();
            eprintln!("po2 {po2}: seal verifies");
        }
        Ok(Err(e)) => panic!("po2 {po2}: prove returned an error: {e:#}"),
        Err(_) => panic!("po2 {po2}: prove panicked"),
    }
}

// ---------------------------------------------------------------------------------------------
// R17 audit: repeated-run determinism of the accum kernel on real segments
// ---------------------------------------------------------------------------------------------

/// R17 audit (the Apple GPU private-array store defect found in witgen): `stepAccumA` has the
/// same IR shape (`back_PoseidonState_A`'s 24-element and `back_BigIntState_A`'s 16-element
/// map loops: constructor-initialised array, loop calling the non-inlined `load`, store after
/// the call; tools/metal_audit/ir_scan.py tier A). Accumulates each `R17_AUDIT_SEGMENTS`
/// segment (comma list of postcard files `(u32, Segment)`) `R17_AUDIT_REPS` times (default 10)
/// per perm variant on one witness and compares every accum buffer bit for bit with the first
/// run and with the CPU circuit HAL. Prints `R17AUDIT rv32im accum ...: k/n` lines.
#[test]
#[ignore]
fn r17_accum_segment_stress() {
    let Ok(paths) = std::env::var("R17_AUDIT_SEGMENTS") else {
        eprintln!("R17_AUDIT_SEGMENTS unset: skipped");
        return;
    };
    let reps: usize = std::env::var("R17_AUDIT_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let hal = Arc::new(MetalHalPoseidon2::new());
    let circuit_hal = MetalCircuitHalPoseidon2::new(hal.clone()).unwrap();
    let mut bad = vec![];
    for path in paths.split(',').filter(|p| !p.is_empty()) {
        let bytes = std::fs::read(path).unwrap();
        let ((_idx, seg), _) =
            postcard::take_from_bytes::<(u32, crate::execute::segment::Segment)>(&bytes).unwrap();
        let r = reference(&seg);
        let rows = r.rows;
        let mut wg = WitnessGenerator::new(
            hal.as_ref(),
            &circuit_hal,
            r.preflight.clone(),
            StepMode::Parallel,
        )
        .unwrap();
        {
            let data = wg.data.as_ref().unwrap();
            data.buf.view_mut(|v| v.copy_from_slice(&r.data));
            wg.global.buf.view_mut(|v| v.copy_from_slice(&r.global));
        }
        hal.sync();
        let name = std::path::Path::new(path)
            .iter()
            .rev()
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|s| s.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        for variant in [
            Variant::Gpu(Some(DEFAULT_SORT_WIN)),
            Variant::Gpu(None),
            Variant::GpuCpuScan(Some(DEFAULT_SORT_WIN)),
        ] {
            let (mut vs_first, mut vs_cpu) = (0, 0);
            let mut first: Option<Vec<Val>> = None;
            let mut detail = vec![];
            let t = Instant::now();
            for rep in 0..reps {
                wg.accum(hal.as_ref(), &WithVariant(&circuit_hal, variant), &r.mix)
                    .unwrap();
                hal.sync();
                let got = wg.accum.as_ref().unwrap().buf.to_vec();
                let n = count_diffs(words(&r.accum), words(&got));
                if n > 0 {
                    vs_cpu += 1;
                    let (row, col, a, b) = first_diff(words(&r.accum), words(&got), rows).unwrap();
                    detail.push(format!(
                        "rep {rep}: {n} cells vs CPU, first row {row} col {col} cpu {a:#x} gpu {b:#x}"
                    ));
                }
                match &first {
                    None => first = Some(got),
                    Some(f) => {
                        let n = count_diffs(words(f), words(&got));
                        if n > 0 {
                            vs_first += 1;
                            detail.push(format!("rep {rep}: {n} cells vs rep 0"));
                        }
                    }
                }
            }
            println!(
                "R17AUDIT rv32im accum {name} po2 {} {variant:?}: {vs_first}/{reps} reps differ \
                 from rep 0, {vs_cpu}/{reps} differ from CPU ({:.0} ms/rep) {detail:?}",
                seg.po2,
                t.elapsed().as_secs_f64() * 1e3 / reps as f64
            );
            if vs_first + vs_cpu > 0 {
                bad.push(format!("{name} {variant:?}"));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "R17AUDIT accum nondeterministic or wrong: {bad:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// R13: GPU prefix scans
// ---------------------------------------------------------------------------------------------

#[test]
fn scan_mode_env_values() {
    assert_eq!(parse_scan_mode(None), ScanMode::Gpu);
    assert_eq!(parse_scan_mode(Some("")), ScanMode::Gpu);
    assert_eq!(parse_scan_mode(Some(" GPU ")), ScanMode::Gpu);
    assert_eq!(parse_scan_mode(Some("cpu")), ScanMode::Cpu);
    assert_eq!(parse_scan_mode(Some("Cpu")), ScanMode::Cpu);
    assert_eq!(parse_scan_mode(Some("bogus")), ScanMode::Gpu);
    assert!(check_fault(0).is_ok());
    let e = format!(
        "{:#}",
        check_fault(FAULT_ACCUM_SCAN | FAULT_ASSERT).unwrap_err()
    );
    assert!(e.contains("canonical") && e.contains("assert"), "{e}");
}

/// Runs the GPU scan of columns `col0 .. col0 + ncols`, rows `[0, n)`, of a `rows x cols` buffer
/// filled from `init`, and returns (buffer after, fault word).
#[allow(clippy::too_many_arguments)]
fn gpu_scan(
    hal: &MetalHalPoseidon2,
    circuit_hal: &MetalCircuitHalPoseidon2,
    init: &[u32],
    rows: usize,
    col0: usize,
    ncols: usize,
    n: usize,
    tg: Option<u64>,
) -> (Vec<u32>, u32) {
    let buf = hal.copy_from_slice("scan_test", init);
    let fault = hal.alloc_buffer::<u32>("scan_fault", 1);
    hal.blit_fill(fault.as_buf(), 0, 4, 0);
    circuit_hal
        .encode_fp_scan(&buf, rows, col0, ncols, n, &fault, tg)
        .unwrap();
    hal.sync();
    (buf.to_vec(), fault.to_vec()[0])
}

/// The host reference: [fp_inclusive_scan] of rows `[0, n)` of each scanned column.
fn cpu_scan(init: &[u32], rows: usize, col0: usize, ncols: usize, n: usize) -> Vec<u32> {
    let mut want = init.to_vec();
    for c in col0..col0 + ncols {
        fp_inclusive_scan(&mut want[c * rows..c * rows + n]);
    }
    want
}

/// The GPU scan equals the host scan bit for bit (every cell of the buffer: scanned rows,
/// rows >= n and other columns untouched) at sizes around the block and chunk boundaries,
/// including 2^20 + 7 (more blocks than one fpScanBlockSums round at the default threadgroup), for
/// several threadgroup sizes and value patterns (random, all P - 1, zeros, small).
#[test]
fn gpu_fp_scan_matches_cpu() {
    let hal = Arc::new(MetalHalPoseidon2::new());
    let circuit_hal = MetalCircuitHalPoseidon2::new(hal.clone()).unwrap();
    circuit_hal.ctx().require(Phase::Accum).unwrap();
    let default_tg = circuit_hal.fp_scan_tg(None).unwrap();
    let width = circuit_hal
        .ctx()
        .pipeline("fpScanApply")
        .thread_execution_width();
    let mut tgs = vec![None, Some(width), Some(2 * width)];
    let big = circuit_hal.fp_scan_tg(Some(1024)).unwrap();
    if big != default_tg {
        tgs.push(Some(big));
    }
    tracing::info!("scan tg default {default_tg}, width {width}, tgs {tgs:?}");
    let mut rng = rand::rng();
    let p = P;
    let sizes = [
        1usize,
        2,
        31,
        255,
        256,
        257,
        1023,
        1024,
        1025,
        4097,
        65_537,
        (1 << 20) + 7,
    ];
    for &n in &sizes {
        let rows = n.next_power_of_two().max(2);
        let (cols, col0, ncols) = (6, 2, 4);
        for pattern in 0..4 {
            let init: Vec<u32> = (0..rows * cols)
                .map(|_| match pattern {
                    0 => rng.random_range(0..p),
                    1 => p - 1,
                    2 => 0,
                    _ => rng.random_range(0..4),
                })
                .collect();
            let want = cpu_scan(&init, rows, col0, ncols, n);
            for &tg in &tgs {
                // The big sizes only with the default and the narrowest threadgroup (time).
                if n > 1 << 16 && pattern != 0 && tg.is_some() {
                    continue;
                }
                let (got, fault) = gpu_scan(&hal, &circuit_hal, &init, rows, col0, ncols, n, tg);
                assert_eq!(fault, 0, "n {n} pattern {pattern} tg {tg:?}: fault");
                if let Some((row, col, a, b)) = first_diff(&want, &got, rows) {
                    panic!(
                        "n {n} pattern {pattern} tg {tg:?}: {} cells differ, first row {row} \
                         col {col}: cpu {a:#x} gpu {b:#x}",
                        count_diffs(&want, &got)
                    );
                }
            }
        }
    }
    // One column, and a column range that ends at the last column.
    for (cols, col0, ncols, n) in [(3, 1, 1, 5000), (5, 1, 4, 3000), (1, 0, 1, 1 << 12)] {
        let rows = 1 << 13;
        let init: Vec<u32> = (0..rows * cols).map(|_| rng.random_range(0..p)).collect();
        let want = cpu_scan(&init, rows, col0, ncols, n);
        let (got, fault) = gpu_scan(&hal, &circuit_hal, &init, rows, col0, ncols, n, None);
        assert_eq!(fault, 0);
        assert!(got == want, "cols {cols} col0 {col0} ncols {ncols} n {n}");
    }
    // n = 0 encodes nothing.
    let init = vec![7u32; 64];
    let (got, fault) = gpu_scan(&hal, &circuit_hal, &init, 16, 0, 4, 0, None);
    assert_eq!((got, fault), (init, 0));
}

/// A non-canonical (e.g. INVALID) scanned cell is reported through fault bit 0x100.
#[test]
fn gpu_fp_scan_faults_on_noncanonical() {
    let hal = Arc::new(MetalHalPoseidon2::new());
    let circuit_hal = MetalCircuitHalPoseidon2::new(hal.clone()).unwrap();
    let rows = 1 << 12;
    for (bad_row, bad) in [(0usize, 0xffff_ffffu32), (3000, P)] {
        let mut init = vec![1u32; rows * 4];
        init[2 * rows + bad_row] = bad;
        let (_, fault) = gpu_scan(&hal, &circuit_hal, &init, rows, 0, 4, rows, None);
        assert_eq!(fault, FAULT_ACCUM_SCAN, "row {bad_row} value {bad:#x}");
        // Outside [0, n) it is not read.
        let (_, fault) = gpu_scan(&hal, &circuit_hal, &init, rows, 0, 4, bad_row, None);
        assert_eq!(fault, 0, "row {bad_row} beyond n");
    }
}

/// Repeated GPU scans of the same input give the same words (and the host's): the R17-style
/// determinism check for the scan kernels (`R13_SCAN_REPS`, default 20) at 2^20 rows x 4 columns.
#[test]
fn gpu_fp_scan_repeat() {
    let reps: usize = std::env::var("R13_SCAN_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let hal = Arc::new(MetalHalPoseidon2::new());
    let circuit_hal = MetalCircuitHalPoseidon2::new(hal.clone()).unwrap();
    let mut rng = rand::rng();
    let rows = 1 << 20;
    let n = rows - 3;
    let init: Vec<u32> = (0..rows * 4).map(|_| rng.random_range(0..P)).collect();
    let want = cpu_scan(&init, rows, 0, 4, n);
    let mut bad = 0;
    for _ in 0..reps {
        let (got, fault) = gpu_scan(&hal, &circuit_hal, &init, rows, 0, 4, n, None);
        if fault != 0 || got != want {
            bad += 1;
        }
    }
    println!("R17AUDIT rv32im accum_scan 2^20x4: {bad}/{reps} reps differ from CPU");
    assert_eq!(bad, 0);
}

fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[s.len() / 2], s[0], s[s.len() - 1])
}

/// R13 measurement (not an assertion): interleaved A/B of the accum step, A = R9's CPU scans
/// (`ScanMode::Cpu`), B = GPU scans, `R13_AB_PAIRS` pairs (default 9) at `R13_AB_PO2` (default
/// 20; comma list), full simple_loop segments. Prints per-arm medians/min/max of the step's wall
/// time (from an idle queue) and of its internal marks, and checks every result against the
/// CPU circuit HAL. Run alone: `--ignored --nocapture --test-threads=1`.
#[test]
#[ignore]
fn accum_ab() {
    let pairs: usize = std::env::var("R13_AB_PAIRS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9);
    let po2s: Vec<usize> = std::env::var("R13_AB_PO2")
        .unwrap_or_else(|_| "20".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let hal = Arc::new(MetalHalPoseidon2::new());
    let circuit_hal = MetalCircuitHalPoseidon2::new(hal.clone()).unwrap();
    eprintln!(
        "working-set limit {} MB, scan tg {}",
        hal.working_set_limit() >> 20,
        circuit_hal.fp_scan_tg(None).unwrap()
    );
    let win = sort_win();
    for po2 in po2s {
        let seg = segment(testutil::kernel::simple_loop(1u32 << po2), po2);
        let r = reference(&seg);
        let rows = r.rows;
        // A' repeats A: the A-vs-A spread is the noise floor of the A/B claim.
        let arms = [
            ("A cpu-scan", Variant::GpuCpuScan(win)),
            ("B gpu-scan", Variant::Gpu(win)),
            ("A' cpu-scan", Variant::GpuCpuScan(win)),
        ];
        // Warm-up (PSOs, pools).
        for (_, v) in arms {
            metal_accum(&hal, &circuit_hal, &r, v);
        }
        let mut step: [Vec<f64>; 3] = Default::default();
        let mut phase: [Vec<f64>; 3] = Default::default();
        let mut peak: [u64; 3] = [0; 3];
        let mut segs: [std::collections::BTreeMap<String, Vec<f64>>; 3] = Default::default();
        for pair in 0..pairs {
            for (k, (name, v)) in arms.iter().enumerate() {
                let (metal, ms, _before, pk) = metal_accum(&hal, &circuit_hal, &r, *v);
                assert_accum_eq(&format!("po2 {po2} {name}"), rows, &r.accum, &metal);
                let st = LAST_STEP_MS.with(Cell::get);
                let m = marks::take();
                step[k].push(st);
                phase[k].push(ms);
                peak[k] = peak[k].max(pk);
                for w in m.windows(2) {
                    let d = (w[1].1 - w[0].1).as_secs_f64() * 1e3;
                    segs[k]
                        .entry(format!("{}->{}", w[0].0, w[1].0))
                        .or_default()
                        .push(d);
                }
                eprintln!("po2 {po2} pair {pair} {name}: step {st:.2} ms, phase {ms:.2} ms");
            }
        }
        for (k, (name, _)) in arms.iter().enumerate() {
            let (sm, sl, sh) = stats(&step[k]);
            let (pm, pl, ph) = stats(&phase[k]);
            println!(
                "R13AB po2 {po2} {name}: step median {sm:.2} ms [{sl:.2}, {sh:.2}], whole \
                 accum phase median {pm:.2} ms [{pl:.2}, {ph:.2}], GPU peak {} MB ({pairs} runs)",
                peak[k] >> 20
            );
            for (seg_name, v) in &segs[k] {
                let (m, l, h) = stats(v);
                println!("R13AB po2 {po2} {name}:   {seg_name} median {m:.2} ms [{l:.2}, {h:.2}]");
            }
        }
        // Per-pair differences: B - A and A' - A (noise floor).
        let d = |x: usize| -> Vec<f64> { (0..pairs).map(|i| step[x][i] - step[0][i]).collect() };
        let (bm, bl, bh) = stats(&d(1));
        let (am, al, ah) = stats(&d(2));
        println!(
            "R13AB po2 {po2}: per-pair step B-A median {bm:.2} ms [{bl:.2}, {bh:.2}]; A'-A median \
             {am:.2} ms [{al:.2}, {ah:.2}]"
        );
        // The scans alone, the 4 accum columns' shape (rows x 4, n = rows), from an idle queue:
        // the host scan vs the GPU scan (encode + wait) per threadgroup size.
        let mut rng = rand::rng();
        let init: Vec<u32> = (0..rows * 4).map(|_| rng.random_range(0..P)).collect();
        let mut times: Vec<(String, Vec<f64>)> = vec![("cpu".into(), vec![])];
        let width = circuit_hal
            .ctx()
            .pipeline("fpScanApply")
            .thread_execution_width();
        let tgs: Vec<u64> = [width, 64, 128, 256, 512, 1024]
            .into_iter()
            .filter(|&t| t % width == 0 && circuit_hal.fp_scan_tg(Some(t)).unwrap() == t)
            .collect();
        for &t in &tgs {
            times.push((format!("gpu tg {t}"), vec![]));
        }
        let buf = hal.copy_from_slice("scan_ab", &init);
        let fault = hal.alloc_buffer::<u32>("scan_ab_fault", 1);
        for _ in 0..pairs {
            let mut host = init.clone();
            let t0 = Instant::now();
            host.par_chunks_mut(rows).for_each(fp_inclusive_scan);
            times[0].1.push(t0.elapsed().as_secs_f64() * 1e3);
            for (k, &t) in tgs.iter().enumerate() {
                buf.view_mut(|v: &mut [u32]| v.copy_from_slice(&init));
                hal.sync();
                let t0 = Instant::now();
                circuit_hal
                    .encode_fp_scan(&buf, rows, 0, 4, rows, &fault, Some(t))
                    .unwrap();
                hal.sync();
                times[k + 1].1.push(t0.elapsed().as_secs_f64() * 1e3);
                assert!(buf.to_vec() == host, "tg {t}");
            }
        }
        for (name, v) in &times {
            let (m, l, h) = stats(v);
            println!("R13AB po2 {po2} scans alone {name}: median {m:.3} ms [{l:.3}, {h:.3}]");
        }
    }
}
