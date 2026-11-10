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
//! Knobs: `RISC0_METAL_WITGEN_REPS` (parallel repetitions per segment, default 3),
//! `RISC0_METAL_WITGEN_PO2S` (po2 list of `witgen_matches_cpu_large`, default `16,18`).

use std::{rc::Rc, sync::Arc, time::Instant};

use risc0_binfmt::{MemoryImage, Program};
use risc0_zkp::{
    core::hash::poseidon2::Poseidon2HashSuite,
    field::Elem as _,
    hal::{cpu::CpuHal, metal::MetalHalPoseidon2},
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
/// clean figure).
#[test]
fn witgen_matches_cpu_large() {
    let po2s: Vec<usize> = std::env::var("RISC0_METAL_WITGEN_PO2S")
        .unwrap_or_else(|_| "16,18".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
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
            let (data, global, peak) = h.metal(pf, StepMode::Parallel);
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
    // Room for the preflight copy and the tables, not for a second data buffer (13.8 MB).
    h.hal
        .set_working_set_limit(Some(h.hal.allocated_size() + (2 << 20)));
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
    // Which allocation hits the budget first depends on what other tests hold on the device
    // (the budget counts the whole process).
    assert!(
        err.contains("rv32im Metal witgen: allocating") && err.contains("budget"),
        "{err}"
    );
}

/// Single-buffer mode (CUDA's default `pre_data == data`) against the CPU: recorded for R15, not
/// the default.
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
    struct Single<'a>(&'a MetalCircuitHalPoseidon2);
    impl CircuitWitnessGenerator<MetalHalPoseidon2> for Single<'_> {
        fn writes_cycle_cols(&self) -> bool {
            self.0.writes_cycle_cols()
        }
        fn takes_zeroed_data(&self) -> bool {
            self.0.takes_zeroed_data()
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
                .metal_witgen(mode, preflight, global, data, pre_data, PreData::Single)
        }
    }
    let w = WitnessGenerator::new(
        h.hal.as_ref(),
        &Single(&h.circuit),
        pf.clone(),
        StepMode::Parallel,
    )
    .unwrap();
    (w.data.as_ref().unwrap().to_vec(), w.global.to_vec())
}
