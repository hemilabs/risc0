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

//! R8: the Metal witgen against the CPU witgen on the same `PreflightResults` (data and global
//! after `WitnessGenerator::new`, i.e. after witgen and zeroize), over several guests including
//! the 46 riscv-tests cases, in the parallel and both sequential modes, repeated.
//!
//! The GPU path is the production default (since R17, witgen.rs module doc); these tests force it
//! through the [Gpu] wrapper whatever `RISC0_METAL_RV32IM_CPU_PHASES` says. [witgen_default_path]
//! checks the default path, [witgen_sorted_amplifier] is the R17 regression test of the Apple GPU
//! lost-store defect (notes/R17.md), [witgen_segment_file] runs a dumped zkvm segment repeatedly.
//!
//! Knobs: `RISC0_METAL_WITGEN_REPS` (parallel repetitions per segment, default 3),
//! `RISC0_METAL_WITGEN_PO2S` (po2 list of `witgen_matches_cpu_large`, default `16,18`).

use std::{rc::Rc, sync::Arc, time::Instant};

use risc0_binfmt::{MemoryImage, Program};
use risc0_zkp::{
    core::hash::poseidon2::Poseidon2HashSuite,
    field::Elem as _,
    hal::{
        cpu::CpuHal,
        metal::{BudgetScope, MetalHalPoseidon2},
    },
};
use test_log::test;

use super::*;
use crate::{
    execute::{
        testutil::{self, NullSyscall, DEFAULT_SESSION_LIMIT},
        DEFAULT_SEGMENT_LIMIT_PO2,
    },
    prove::{
        hal::{cpu::CpuCircuitHal, metal::MetalCircuitHalPoseidon2},
        witgen::WitnessGenerator,
        PreflightResults,
    },
    zirgen::circuit::{ExtVal, Val, REGCOUNT_DATA, REGCOUNT_GLOBAL},
    MAX_INSN_CYCLES,
};

const RISCV_TESTS: &[&str] = &[
    "add", "addi", "and", "andi", "auipc", "beq", "bge", "bgeu", "blt", "bltu", "bne", "div",
    "divu", "jal", "jalr", "lb", "lbu", "lh", "lhu", "lui", "lw", "mul", "mulh", "mulhsu", "mulhu",
    "or", "ori", "rem", "remu", "sb", "sh", "simple", "sll", "slli", "slt", "slti", "sltiu",
    "sltu", "sra", "srai", "srl", "srli", "sub", "sw", "xor", "xori",
];

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn reps() -> usize {
    env_or("RISC0_METAL_WITGEN_REPS", 3)
}

/// The order `par_stepExecW` runs a launch's cycles in (witgen.rs `test_perm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LaunchOrder {
    /// `start + i` (production).
    Plain,
    /// The same, through the perm binding.
    Identity,
    /// Sorted by (major, minor, cycle): uniform work per SIMD group and dispatch batch. The R17
    /// amplifier of the Apple GPU lost-store defect (notes/R17.md): about 100x more bad rows than
    /// the plain order on a build that has the defect's code shape.
    Sorted,
}

thread_local! {
    /// Set by a test for the calls on its thread; `None`: `RISC0_METAL_WITGEN_PERM`
    /// (`identity` | `sorted`, unset: plain).
    static LAUNCH_ORDER: std::cell::Cell<Option<LaunchOrder>> = const { std::cell::Cell::new(None) };
}

fn launch_order() -> LaunchOrder {
    LAUNCH_ORDER.with(|o| o.get()).unwrap_or_else(|| {
        match std::env::var("RISC0_METAL_WITGEN_PERM").as_deref() {
            Err(_) | Ok("") | Ok("plain") => LaunchOrder::Plain,
            Ok("identity") => LaunchOrder::Identity,
            Ok("sorted") => LaunchOrder::Sorted,
            Ok(other) => panic!("RISC0_METAL_WITGEN_PERM={other}: plain, identity or sorted"),
        }
    })
}

/// witgen.rs `test_perm`.
pub(super) fn launch_perm(preflight: &PreflightTrace, start: u32, count: u32) -> Option<Vec<u32>> {
    let mut cycles: Vec<u32> = (start..start + count).collect();
    match launch_order() {
        LaunchOrder::Plain => return None,
        LaunchOrder::Identity => {}
        LaunchOrder::Sorted => cycles.sort_by_key(|&i| {
            let c = &preflight.cycles[i as usize];
            (c.major, c.minor, i)
        }),
    }
    (count > 0).then_some(cycles)
}

/// The 46 riscv-tests programs (as `prove::tests::riscv`), in [RISCV_TESTS] order.
fn riscv_programs() -> Vec<(String, Program)> {
    use std::io::Read;

    use flate2::read::GzDecoder;
    use tar::Archive;

    let bytes = include_bytes!("../../testdata/riscv-tests.tgz");
    let mut tar = Archive::new(GzDecoder::new(&bytes[..]));
    let mut found = Vec::new();
    for entry in tar.entries().unwrap() {
        let mut entry = entry.unwrap();
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let name = entry
            .path()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        if !RISCV_TESTS.contains(&name.as_str()) {
            continue;
        }
        let mut elf = Vec::new();
        entry.read_to_end(&mut elf).unwrap();
        found.push((name, Program::load_elf(&elf, u32::MAX).unwrap()));
    }
    found.sort_by_key(|(n, _)| RISCV_TESTS.iter().position(|t| t == n));
    assert_eq!(found.len(), RISCV_TESTS.len(), "riscv-tests.tgz cases");
    found
}

fn preflights(program: Program, po2: usize) -> Vec<PreflightResults> {
    let session = testutil::execute(
        MemoryImage::new_kernel(program),
        po2,
        MAX_INSN_CYCLES,
        DEFAULT_SESSION_LIMIT,
        &NullSyscall,
        None,
    )
    .unwrap();
    let rand_z = ExtVal::random(&mut rand::rng());
    session
        .segments
        .iter()
        .map(|s| PreflightResults::new(s, rand_z).unwrap())
        .collect()
}

/// Number of differing cells; logs the first as (row, col) of a column-major `rows`-row matrix.
fn first_diff(label: &str, cpu: &[Val], gpu: &[Val], rows: usize) -> usize {
    assert_eq!(cpu.len(), gpu.len(), "{label}: lengths");
    let mut bad = 0;
    let mut first = None;
    let mut cols = std::collections::BTreeSet::new();
    for (i, (a, b)) in cpu.iter().zip(gpu).enumerate() {
        if a != b {
            bad += 1;
            cols.insert(i / rows);
            first.get_or_insert(i);
        }
    }
    if let Some(i) = first {
        eprintln!(
            "[r8] {label}: {bad} of {} cells differ in {} cols {:?}; first (row {}, col {}): \
             cpu={:?} metal={:?}",
            cpu.len(),
            cols.len(),
            cols.iter().take(16).collect::<Vec<_>>(),
            i % rows,
            i / rows,
            cpu[i],
            gpu[i]
        );
    }
    bad
}

/// The Metal circuit HAL with its witgen forced onto the GPU path (opt-in in production, see the
/// module doc of witgen.rs), binding `pre_data` as the [PreData] says.
struct Gpu<'a>(&'a MetalCircuitHalPoseidon2, PreData);

impl CircuitWitnessGenerator<MetalHalPoseidon2> for Gpu<'_> {
    fn writes_cycle_cols(&self) -> bool {
        true
    }
    fn takes_zeroed_data(&self) -> bool {
        true
    }
    fn generate_witness(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        global: &MetaBuffer<MetalHalPoseidon2>,
        data: &MetaBuffer<MetalHalPoseidon2>,
        pre_data: &MetaBuffer<MetalHalPoseidon2>,
    ) -> anyhow::Result<()> {
        self.0
            .metal_witgen(mode, preflight, global, data, pre_data, self.1)
    }
}

struct Harness {
    cpu_hal: Rc<CpuHal<crate::zirgen::circuit::CircuitField>>,
    hal: Arc<MetalHalPoseidon2>,
    circuit: MetalCircuitHalPoseidon2,
}

impl Harness {
    fn new() -> Self {
        let hal = Arc::new(MetalHalPoseidon2::new());
        let circuit = MetalCircuitHalPoseidon2::new(hal.clone()).unwrap();
        Self {
            cpu_hal: Rc::new(CpuHal::new(Poseidon2HashSuite::new_suite())),
            hal,
            circuit,
        }
    }

    /// CPU sequential witgen: (data, global) after zeroize.
    fn cpu(&self, pf: &PreflightResults) -> (Vec<Val>, Vec<Val>) {
        let w = WitnessGenerator::new(
            self.cpu_hal.as_ref(),
            &CpuCircuitHal,
            pf.clone(),
            StepMode::SeqForward,
        )
        .unwrap();
        (w.data.as_ref().unwrap().to_vec(), w.global.to_vec())
    }

    /// Metal witgen in `mode`: (data, global) after zeroize, and the peak allocation.
    fn metal(&self, pf: &PreflightResults, mode: StepMode) -> (Vec<Val>, Vec<Val>, u64) {
        self.hal.reset_peak_allocated_size();
        let gpu = Gpu(&self.circuit, PreData::Auto);
        let w = WitnessGenerator::new(self.hal.as_ref(), &gpu, pf.clone(), mode).unwrap();
        let peak = self.hal.peak_allocated_size();
        (w.data.as_ref().unwrap().to_vec(), w.global.to_vec(), peak)
    }

    /// The Metal circuit HAL's default witgen (the CPU kernel unless opted in): as [Self::metal].
    fn default_path(&self, pf: &PreflightResults, mode: StepMode) -> (Vec<Val>, Vec<Val>, u64) {
        self.hal.reset_peak_allocated_size();
        let w = WitnessGenerator::new(self.hal.as_ref(), &self.circuit, pf.clone(), mode).unwrap();
        let peak = self.hal.peak_allocated_size();
        (w.data.as_ref().unwrap().to_vec(), w.global.to_vec(), peak)
    }

    /// Compare every mode against the CPU; returns the number of differing cells.
    fn check(&self, label: &str, pf: &PreflightResults, modes: &[(StepMode, usize)]) -> usize {
        let rows = 1usize << pf.po2();
        let (cpu_data, cpu_global) = self.cpu(pf);
        assert_eq!(cpu_data.len(), rows * REGCOUNT_DATA);
        assert_eq!(cpu_global.len(), REGCOUNT_GLOBAL);
        let mut bad = 0;
        for &(mode, reps) in modes {
            let name = match mode {
                StepMode::Parallel => "par",
                StepMode::SeqForward => "fwd",
                StepMode::SeqReverse => "rev",
            };
            for rep in 0..reps {
                let t = Instant::now();
                let (data, global, _) = self.metal(pf, mode);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let l = format!("{label} po2={} {name} rep={rep}", pf.po2());
                let b = first_diff(&format!("{l} data"), &cpu_data, &data, rows)
                    + first_diff(&format!("{l} global"), &cpu_global, &global, 1);
                tracing::info!("[r8] {l}: {b} bad cells, witgen_new {ms:.1} ms");
                bad += b;
            }
        }
        bad
    }
}

fn all_modes(reps: usize) -> [(StepMode, usize); 3] {
    [
        (StepMode::Parallel, reps),
        (StepMode::SeqForward, 1),
        (StepMode::SeqReverse, 1),
    ]
}

#[test]
fn witgen_params_layout() {
    assert_eq!(std::mem::offset_of!(WitgenParams, table_split_cycle), 28);
    assert_eq!(std::mem::offset_of!(WitgenParams, use_perm), 40);
    assert_eq!(std::mem::offset_of!(SeqParams, last_cycle), 12);
    assert_eq!(std::mem::offset_of!(InjectParams, c_mode), 24);
    assert!(describe_fault(0x14).contains("transaction") && describe_fault(0x14).contains("asser"));
    assert!(
        describe_fault(0x80).contains("getDiffCount") && !describe_fault(0x80).contains("unknown")
    );
    assert!(describe_fault(0x40).contains("unknown bits"));
}

/// The Metal circuit HAL's default witgen: the GPU kernel (the CPU kernel with the CPU HAL's flags
/// under the `RISC0_METAL_RV32IM_CPU_PHASES=witgen` kill switch); its witness equals the CPU HAL's.
#[test]
fn witgen_default_path() {
    let h = Harness::new();
    let gpu = !cpu_phase("witgen");
    assert_eq!(cpu_witgen(), !gpu);
    assert_eq!(h.circuit.writes_cycle_cols(), gpu);
    assert_eq!(h.circuit.takes_zeroed_data(), gpu);
    assert_eq!(defers_cycle_cols(), gpu);
    let mut bad = 0;
    for (label, program) in [
        ("basic", testutil::kernel::basic()),
        ("simple_loop(40000)", testutil::kernel::simple_loop(40000)),
    ] {
        for pf in preflights(program, DEFAULT_SEGMENT_LIMIT_PO2) {
            let rows = 1usize << pf.po2();
            let (cpu_data, cpu_global) = h.cpu(&pf);
            let w =
                WitnessGenerator::new(h.hal.as_ref(), &h.circuit, pf.clone(), StepMode::Parallel)
                    .unwrap();
            let (data, global) = (w.data.as_ref().unwrap().to_vec(), w.global.to_vec());
            bad += first_diff(&format!("{label} default data"), &cpu_data, &data, rows)
                + first_diff(&format!("{label} default global"), &cpu_global, &global, 1);
        }
    }
    assert_eq!(
        bad, 0,
        "the default Metal witgen path differs from the CPU HAL"
    );
}

/// basic, multi_read and a two-segment simple_loop (table split > 0 and lookups) in all modes.
#[test]
fn witgen_matches_cpu() {
    let h = Harness::new();
    let mut bad = 0;
    let mut n = 0;
    for (label, program) in [
        ("basic", testutil::kernel::basic()),
        ("multi_read", testutil::kernel::multi_read()),
        ("simple_loop(2000)", testutil::kernel::simple_loop(2000)),
        ("simple_loop(40000)", testutil::kernel::simple_loop(40000)),
    ] {
        for pf in preflights(program, DEFAULT_SEGMENT_LIMIT_PO2) {
            bad += h.check(label, &pf, &all_modes(reps()));
            n += 1;
        }
    }
    eprintln!("[r8] witgen_matches_cpu: {n} segments, {bad} bad cells");
    assert_eq!(bad, 0, "Metal witgen differs from the CPU witgen");
}

/// The traces of the 46 riscv-tests cases (prove::tests::riscv), every mode.
#[test]
fn witgen_matches_cpu_riscv() {
    let h = Harness::new();
    let mut bad = 0;
    let mut n = 0;
    for (name, program) in riscv_programs() {
        for pf in preflights(program, DEFAULT_SEGMENT_LIMIT_PO2) {
            bad += h.check(&name, &pf, &all_modes(reps()));
            n += 1;
        }
    }
    eprintln!("[r8] witgen_matches_cpu_riscv: {n} segments, {bad} bad cells");
    assert_eq!(bad, 0, "Metal witgen differs from the CPU witgen");
}

/// Larger segments (parallel mode only; the sequential kernels run ~40 us per cycle), with the
/// peak Metal allocation of `WitnessGenerator::new` (process-wide: run this test alone for a
/// clean figure). `RISC0_METAL_WITGEN_LARGE_DEFAULT=1` measures the default path instead.
#[test]
fn witgen_matches_cpu_large() {
    let po2s: Vec<usize> = std::env::var("RISC0_METAL_WITGEN_PO2S")
        .unwrap_or_else(|_| "16,18".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    // RISC0_METAL_WITGEN_LARGE_DEFAULT=1: measure the circuit HAL's default path instead.
    let default_path = env_or("RISC0_METAL_WITGEN_LARGE_DEFAULT", 0) == 1;
    let h = Harness::new();
    let mut bad = 0;
    for po2 in po2s {
        // ~ a full first segment: the loop body is a few cycles per iteration.
        let pfs = preflights(testutil::kernel::simple_loop((1 << po2) as u32 / 4), po2);
        let pf = &pfs[0];
        let rows = 1usize << pf.po2();
        let (cpu_data, cpu_global) = h.cpu(pf);
        for rep in 0..reps() {
            h.hal.try_sync().unwrap();
            let before = h.hal.allocated_size();
            let t = Instant::now();
            let (data, global, peak) = if default_path {
                h.default_path(pf, StepMode::Parallel)
            } else {
                h.metal(pf, StepMode::Parallel)
            };
            let ms = t.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "[r8] large po2={} rep={rep}: witgen_new {ms:.1} ms, allocated before {:.1} MB, \
                 peak {:.1} MB (+{:.1} MB), working-set limit {:.0} MB",
                pf.po2(),
                before as f64 / 1e6,
                peak as f64 / 1e6,
                peak.saturating_sub(before) as f64 / 1e6,
                h.hal.working_set_limit() as f64 / 1e6
            );
            bad += first_diff("large data", &cpu_data, &data, rows)
                + first_diff("large global", &cpu_global, &global, 1);
        }
    }
    assert_eq!(bad, 0, "Metal witgen differs from the CPU witgen");
}

/// A kernel-mode loop of `iters` iterations over register shifts (SLL/SRL/SRA by the loop counter,
/// so every shift amount 0..31 recurs), immediate shifts, MUL and DIVU: the Mul0 / Div0 cycles
/// whose DynPo2 (ToBits<5>) lost bit 4 on the Apple GPU before R17 (notes/R17.md).
fn shift_loop(iters: u32) -> Program {
    // RISC-V R-type / I-type shift encodings.
    fn r(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32) -> u32 {
        (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0b0110011
    }
    fn shi(funct7: u32, shamt: u32, rs1: u32, funct3: u32, rd: u32) -> u32 {
        (funct7 << 25) | (shamt << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0b0010011
    }
    const A0: u32 = 10;
    const A1: u32 = 11;
    const A2: u32 = 12;
    const A3: u32 = 13;
    const A4: u32 = 14;
    const A5: u32 = 15;
    const A6: u32 = 16;
    let mut asm = testutil::Assembler::new();
    asm.li(A0 as usize, 0x9e37_79b9);
    asm.addi(A4 as usize, 0, 0);
    asm.li(A5 as usize, iters);
    let body = [
        r(0x00, A4, A0, 1, A1),  // sll  a1, a0, a4
        r(0x00, A4, A0, 5, A2),  // srl  a2, a0, a4
        r(0x20, A4, A0, 5, A3),  // sra  a3, a0, a4
        shi(0x00, 7, A1, 1, A6), // slli a6, a1, 7
        shi(0x20, 3, A6, 5, A6), // srai a6, a6, 3
        r(0x01, A2, A1, 0, A1),  // mul  a1, a1, a2
        r(0x01, A4, A3, 5, A2),  // divu a2, a3, a4
        r(0x00, A1, A0, 0, A0),  // add  a0, a0, a1
        r(0x00, A6, A0, 4, A0),  // xor  a0, a0, a6
    ];
    for insn in body {
        asm.raw(insn);
    }
    asm.addi(A4 as usize, A4 as usize, 1);
    asm.blt(A4 as usize, A5 as usize, -4 * (body.len() as i32 + 1));
    asm.host_terminate(0, 0);
    asm.program()
}

/// R17 regression test of the Apple GPU lost-store defect (notes/R17.md). Parallel witgen of a
/// full po2 20 [shift_loop] segment with the launch sorted by (major, minor) ([LaunchOrder::Sorted]:
/// uniform work per dispatch batch, which made the defect ~100x more frequent), against the CPU.
/// A kernel with the defect's code shape fails this on every call (measured with a
/// `witgen_tu_gen.py --map-lowering loop` build). `RISC0_METAL_WITGEN_AMP_REPS` calls (default 4).
#[test]
fn witgen_sorted_amplifier() {
    let h = Harness::new();
    let pfs = preflights(shift_loop(1 << 17), 20);
    let pf = &pfs[0];
    assert_eq!(pf.po2(), 20);
    let mut bad = 0;
    let mut bad_calls = 0;
    let reps = env_or("RISC0_METAL_WITGEN_AMP_REPS", 4);
    LAUNCH_ORDER.with(|o| o.set(Some(LaunchOrder::Sorted)));
    let (cpu_data, cpu_global) = h.cpu(pf);
    for rep in 0..reps {
        let w = WitnessGenerator::new(
            h.hal.as_ref(),
            &Gpu(&h.circuit, PreData::Auto),
            pf.clone(),
            StepMode::Parallel,
        );
        let b = match w {
            Ok(w) => {
                let (data, global) = (w.data.as_ref().unwrap().to_vec(), w.global.to_vec());
                first_diff(&format!("sorted rep={rep} data"), &cpu_data, &data, 1 << 20)
                    + first_diff(&format!("sorted rep={rep} global"), &cpu_global, &global, 1)
            }
            Err(e) => {
                eprintln!("[r17] sorted rep={rep}: {e:#}");
                1
            }
        };
        bad += b;
        bad_calls += (b != 0) as usize;
    }
    LAUNCH_ORDER.with(|o| o.set(None));
    eprintln!("[r17] witgen_sorted_amplifier: {bad_calls} of {reps} calls bad, {bad} bad cells");
    assert_eq!(
        bad, 0,
        "sorted parallel Metal witgen differs from the CPU witgen"
    );
}

/// An all-zero preflight cycle with `txn_idx`.
fn cycle(txn_idx: u32) -> risc0_circuit_rv32im_sys::RawPreflightCycle {
    risc0_circuit_rv32im_sys::RawPreflightCycle {
        state: 0,
        pc: 0,
        major: 0,
        minor: 0,
        machine_mode: 0,
        padding: 0,
        user_cycle: 0,
        txn_idx,
        paging_idx: 0,
        bigint_idx: 0,
        diff_count: [0, 0],
    }
}

/// A trace whose cycles index past the memory transactions sets the fault word: `Err`, not a
/// silently wrong witness.
#[test]
fn witgen_fault_is_an_error() {
    let h = Harness::new();
    let rows = 1024;
    let mut trace = PreflightTrace {
        cycles: vec![cycle(1 << 30); rows],
        cycle_cols_deferred: true,
        ..Default::default()
    };
    let meta = |name, rows, cols| MetaBuffer {
        buf: h.hal.alloc_elem_init(name, rows * cols, Val::ZERO),
        rows,
        cols,
        checked: false,
    };
    let global = meta("global", 1, REGCOUNT_GLOBAL);
    let data = meta("data", rows, REGCOUNT_DATA);
    for mode in [StepMode::Parallel, StepMode::SeqForward] {
        let err = h
            .circuit
            .metal_witgen(mode, &trace, &global, &data, &data, PreData::Auto)
            .expect_err("a corrupt trace must fail");
        let err = format!("{err:#}");
        assert!(err.contains("kernel fault"), "{err}");
    }
    // A shape the kernels cannot take is refused before encoding anything.
    trace.cycles.truncate(rows - 1);
    let err = h
        .circuit
        .metal_witgen(
            StepMode::Parallel,
            &trace,
            &global,
            &data,
            &data,
            PreData::Auto,
        )
        .unwrap_err();
    assert!(format!("{err:#}").contains("preflight cycles"));
}

/// A cycle column value past the preflight cycles makes getDiffCount (DoCycleTable) set
/// `RV32IM_FAULT_DIFF_COUNT_INDEX` instead of reading out of bounds (R16 finding 1).
#[test]
fn witgen_diff_count_fault() {
    let h = Harness::new();
    let rows = 1024;
    let trace = PreflightTrace {
        cycles: vec![cycle(0); rows],
        ..Default::default()
    };
    let global = MetaBuffer {
        buf: h.hal.alloc_elem_init("global", REGCOUNT_GLOBAL, Val::ZERO),
        rows: 1,
        cols: REGCOUNT_GLOBAL,
        checked: false,
    };
    let data: MetaBuffer<MetalHalPoseidon2> = MetaBuffer {
        buf: h
            .hal
            .alloc_elem_init("data", rows * REGCOUNT_DATA, Val::ZERO),
        rows,
        cols: REGCOUNT_DATA,
        checked: false,
    };
    // Not deferred: the kernel takes the cycle column as it is, here 2^24 + row.
    let col = crate::prove::witgen::CYCLE_COLS[0] as usize;
    data.buf.view_mut(|d: &mut [Val]| {
        for row in 0..rows {
            d[col * rows + row] = Val::new((1 << 24) + row as u32);
        }
    });
    let err = h
        .circuit
        .metal_witgen(
            StepMode::Parallel,
            &trace,
            &global,
            &data,
            &data,
            PreData::Auto,
        )
        .expect_err("a cycle past the trace must fail");
    assert!(format!("{err:#}").contains("getDiffCount"), "{err:#}");
}

/// Under a working-set budget too small for the `pre_data` snapshot, witgen returns an error
/// (the HAL's allocation panic is converted) instead of unwinding through the prover.
#[test]
fn witgen_snapshot_over_budget_is_an_error() {
    let h = Harness::new();
    let rows = 1 << 14;
    let trace = PreflightTrace {
        cycles: vec![cycle(0); rows],
        ..Default::default()
    };
    let global = MetaBuffer {
        buf: h.hal.alloc_elem_init("global", REGCOUNT_GLOBAL, Val::ZERO),
        rows: 1,
        cols: REGCOUNT_GLOBAL,
        checked: false,
    };
    let data = MetaBuffer {
        buf: h
            .hal
            .alloc_elem_init("data", rows * REGCOUNT_DATA, Val::ZERO),
        rows,
        cols: REGCOUNT_DATA,
        checked: false,
    };
    h.hal.try_sync().unwrap();
    // Room for the preflight copy and the tables, not for a second data buffer (13.8 MB). The
    // budget counts only this HAL's bytes (Z8b `BudgetScope::Hal`), so tests allocating and
    // freeing concurrently in the parallel harness cannot move it (R13 fix round 1).
    h.hal.set_budget_scope(BudgetScope::Hal);
    h.hal
        .set_working_set_limit(Some(h.hal.owned_allocated_size() + (2 << 20)));
    let res = h.circuit.metal_witgen(
        StepMode::Parallel,
        &trace,
        &global,
        &data,
        &data,
        PreData::Auto,
    );
    h.hal.set_working_set_limit(None);
    let err = format!("{:#}", res.expect_err("the snapshot cannot fit"));
    // Which allocation hits the budget first may vary; any of witgen's allocations reports it.
    assert!(
        err.contains("rv32im Metal witgen: allocating") && err.contains("budget"),
        "{err}"
    );
}

/// Single-buffer mode (CUDA's default `pre_data == data`) against the CPU: recorded for R15, not
/// the default (R17: 0 of 60 bad calls on two po2 20 segments, notes/R17.md).
#[test]
fn witgen_single_buffer_matches_cpu() {
    let h = Harness::new();
    let mut bad = 0;
    for pf in preflights(
        testutil::kernel::simple_loop(40000),
        DEFAULT_SEGMENT_LIMIT_PO2,
    ) {
        let rows = 1usize << pf.po2();
        let (cpu_data, cpu_global) = h.cpu(&pf);
        for rep in 0..reps() {
            let (data, global) = single_buffer_witgen(&h, &pf);
            bad += first_diff(&format!("single rep={rep} data"), &cpu_data, &data, rows)
                + first_diff(&format!("single rep={rep} global"), &cpu_global, &global, 1);
        }
    }
    assert_eq!(
        bad, 0,
        "single-buffer Metal witgen differs from the CPU witgen"
    );
}

/// [WitnessGenerator::new]'s buffer setup with the single-buffer kernels.
fn single_buffer_witgen(h: &Harness, pf: &PreflightResults) -> (Vec<Val>, Vec<Val>) {
    let w = WitnessGenerator::new(
        h.hal.as_ref(),
        &Gpu(&h.circuit, PreData::Single),
        pf.clone(),
        StepMode::Parallel,
    )
    .unwrap();
    (w.data.as_ref().unwrap().to_vec(), w.global.to_vec())
}

/// What one Metal witgen call saw and left, read back after syncs around it.
struct Captured {
    data_in: Vec<Val>,
    data_out: Vec<Val>,
    global_out: Vec<Val>,
    pf_out: Vec<u8>,
    result: Result<(), String>,
}

/// [WitnessGenerator::new]'s buffer setup, capturing the witgen call's inputs and outputs.
fn captured_witgen(h: &Harness, pf: &PreflightResults, mode: StepMode, pre: PreData) -> Captured {
    struct Cap<'a>(
        &'a MetalCircuitHalPoseidon2,
        PreData,
        std::cell::RefCell<Option<Captured>>,
    );
    impl CircuitWitnessGenerator<MetalHalPoseidon2> for Cap<'_> {
        fn writes_cycle_cols(&self) -> bool {
            true
        }
        fn takes_zeroed_data(&self) -> bool {
            true
        }
        fn generate_witness(
            &self,
            mode: StepMode,
            preflight: &PreflightTrace,
            global: &MetaBuffer<MetalHalPoseidon2>,
            data: &MetaBuffer<MetalHalPoseidon2>,
            pre_data: &MetaBuffer<MetalHalPoseidon2>,
        ) -> anyhow::Result<()> {
            let data_in = data.buf.to_vec();
            let result = self
                .0
                .metal_witgen(mode, preflight, global, data, pre_data, self.1);
            let pf_out = self.0.ctx.scratch().pf.as_ref().unwrap().to_vec();
            *self.2.borrow_mut() = Some(Captured {
                data_in,
                data_out: data.buf.to_vec(),
                global_out: global.buf.to_vec(),
                pf_out,
                result: result.as_ref().map(|_| ()).map_err(|e| format!("{e:#}")),
            });
            result
        }
    }
    let cap = Cap(&h.circuit, pre, Default::default());
    let _ = WitnessGenerator::new(h.hal.as_ref(), &cap, pf.clone(), mode);
    cap.2.into_inner().unwrap()
}

/// Rows (sorted, deduplicated) where two column-major matrices differ.
fn diff_rows(a: &[Val], b: &[Val], rows: usize) -> Vec<usize> {
    let mut out: Vec<usize> = a
        .iter()
        .zip(b)
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, _)| i % rows)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Reproducer of the kernel nondeterminism (notes/R8.md): a segment dumped by a zkvm harness (`postcard` of `risc0_zkvm::Segment`, whose
/// encoding starts with the index and this crate's [crate::execute::segment::Segment]), named by
/// `RISC0_METAL_WITGEN_SEGMENT`, through the CPU witgen once and the Metal witgen
/// `RISC0_METAL_WITGEN_REPS` times per mode of `RISC0_METAL_WITGEN_MODES` (`par,single,fwd,rev`,
/// default `par`), with the input and output of every call compared. Skipped when unset.
#[test]
fn witgen_segment_file() {
    let Ok(path) = std::env::var("RISC0_METAL_WITGEN_SEGMENT") else {
        eprintln!("RISC0_METAL_WITGEN_SEGMENT unset: skipped");
        return;
    };
    let bytes = std::fs::read(&path).unwrap();
    let ((_index, segment), _) =
        postcard::take_from_bytes::<(u32, crate::execute::segment::Segment)>(&bytes).unwrap();
    let pf = PreflightResults::new(&segment, ExtVal::random(&mut rand::rng())).unwrap();
    let rows = 1usize << pf.po2();
    let h = Harness::new();
    let w = WitnessGenerator::new(
        h.cpu_hal.as_ref(),
        &CpuCircuitHal,
        pf.clone(),
        StepMode::SeqForward,
    )
    .unwrap();
    let (cpu_data, cpu_global) = (w.data.as_ref().unwrap().to_vec(), w.global.to_vec());
    let trace = w.trace;
    let modes = std::env::var("RISC0_METAL_WITGEN_MODES").unwrap_or_else(|_| "par".into());
    let mut first_in: Option<Vec<Val>> = None;
    let mut good_pf: Option<Vec<u8>> = None;
    let mut failures = 0;
    for mode_name in modes.split(',') {
        let (mode, pre) = match mode_name.trim() {
            "par" => (StepMode::Parallel, PreData::Auto),
            "single" => (StepMode::Parallel, PreData::Single),
            "fwd" => (StepMode::SeqForward, PreData::Auto),
            "rev" => (StepMode::SeqReverse, PreData::Auto),
            other => panic!("unknown mode {other}"),
        };
        for rep in 0..reps() {
            let t = Instant::now();
            let c = captured_witgen(&h, &pf, mode, pre);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let input_rows = match &first_in {
                None => {
                    first_in = Some(c.data_in.clone());
                    vec![]
                }
                Some(first) => diff_rows(first, &c.data_in, rows),
            };
            let global: Vec<Val> = c.global_out.iter().map(|v| v.valid_or_zero()).collect();
            let out_rows = diff_rows(&cpu_data, &c.data_out, rows);
            let global_bad = first_diff("global", &cpu_global, &global, 1);
            let pf_same = match (&good_pf, c.result.is_ok() && out_rows.is_empty()) {
                (None, true) => {
                    good_pf = Some(c.pf_out.clone());
                    "ref".to_string()
                }
                (Some(g), _) => {
                    let n = g.iter().zip(&c.pf_out).filter(|(a, b)| a != b).count();
                    format!("{n} bytes differ")
                }
                (None, false) => "no ref yet".to_string(),
            };
            let ok = c.result.is_ok() && out_rows.is_empty() && global_bad == 0;
            if !ok {
                failures += 1;
                first_diff(
                    &format!("{mode_name} rep={rep} data"),
                    &cpu_data,
                    &c.data_out,
                    rows,
                );
            }
            eprintln!(
                "[r8] seg po2={} {mode_name} rep={rep} {ms:.0} ms: {:?}; input rows differing \
                 from rep 0: {} {:?}; output rows differing from cpu: {} {:?}; pf_out vs a good \
                 run: {pf_same}",
                pf.po2(),
                c.result,
                input_rows.len(),
                &input_rows[..input_rows.len().min(8)],
                out_rows.len(),
                &out_rows[..out_rows.len().min(8)],
            );
            for (i, (a, b)) in cpu_data.iter().zip(&c.data_out).enumerate() {
                if a != b {
                    eprintln!(
                        "[r8]   cell row {} col {}: cpu {:#x} metal {:#x}",
                        i % rows,
                        i / rows,
                        a.as_u32(),
                        b.as_u32()
                    );
                }
            }
            if !out_rows.is_empty() {
                let r = out_rows[0];
                let cyc = &trace.cycles[r];
                eprintln!(
                    "[r8]   first bad row {r}: cycle major {} minor {} pc {:#x} state {} \
                     txn_idx {}; split {}",
                    cyc.major, cyc.minor, cyc.pc, cyc.state, cyc.txn_idx, trace.table_split_cycle
                );
                // RISC0_METAL_WITGEN_DUMP_ROW=1: the whole first bad row (input, cpu, metal).
                if env_or("RISC0_METAL_WITGEN_DUMP_ROW", 0) == 1 {
                    for col in 0..REGCOUNT_DATA {
                        let i = col * rows + r;
                        eprintln!(
                            "[r8]   row {r} col {col} (raw): in {:#x} cpu {:#x} metal {:#x}",
                            c.data_in[i].as_u32_montgomery(),
                            cpu_data[i].as_u32_montgomery(),
                            c.data_out[i].as_u32_montgomery()
                        );
                    }
                }
            }
        }
    }
    assert_eq!(
        failures, 0,
        "Metal witgen calls that failed or differ from the CPU"
    );
}
