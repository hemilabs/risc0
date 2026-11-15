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

//! C6 tests: the Metal witgen against the CPU witgen (bit-exact `data` and `global`), with and
//! without host chains and in the sequential modes; the fault path; the prewitgen / shadow check
//! hooks with `upload_host_witness`; the WOM sort against a full sort. Peak device memory of the
//! witgen is printed per po2 (PLAN 9.1).

use std::{
    collections::VecDeque,
    sync::{atomic::Ordering, Arc},
};

use rand::{Rng, SeedableRng};
use risc0_circuit_recursion_sys::{
    risc0_circuit_recursion_cpu_chains, RawPreflightTrace, StepMode,
};
use risc0_core::field::baby_bear::{BabyBear, BabyBearElem};
use risc0_zkp::{
    adapter::{CircuitInfo as _, TapsProvider as _},
    core::hash::sha::Sha256HashSuite,
    field::Elem as _,
    hal::{cpu::CpuHal, metal::MetalHalPoseidon2, Buffer as _, Hal as _},
};
use test_log::test;

use super::*;
use crate::{
    prove::{
        hal::{cpu::CpuCircuitHal, metal::PHASE_ALLOC},
        preflight::Preflight,
        program::Program,
        witgen::{host_chains, host_ctrl_for},
        zkr::get_zkr,
    },
    CircuitImpl, CIRCUIT,
};

fn test_input() -> VecDeque<u32> {
    // Two Poseidon2 digests (8 words each), as zkvm's `test_zkr` adds them.
    (0..16u32).map(|i| i * 0x101 + 7).collect()
}

pub(super) fn preflight_for(po2: usize) -> (Arc<Program>, Preflight) {
    let program = get_zkr("test_recursion_circuit.zkr", po2).unwrap();
    let mut pf = Preflight::new(test_input());
    for (cycle, row) in program.code_by_row().enumerate() {
        pf.step(cycle, row).unwrap();
    }
    (program, pf)
}

pub(super) fn raw_trace(program: &Program, pf: &Preflight) -> RawPreflightTrace {
    RawPreflightTrace {
        wom: pf.trace.wom.as_ptr(),
        cycles: pf.trace.cycles.as_ptr(),
        iops: pf.trace.iops.as_ptr(),
        num_woms: pf.trace.wom.len() as u32,
        num_cycles: program.code_rows() as u32,
        num_iops: pf.trace.iops.len() as u32,
    }
}

fn raw(v: &[BabyBearElem]) -> Vec<u32> {
    v.iter().map(|x| x.as_u32_montgomery()).collect()
}

/// The CPU HAL witness (data, global) as the prover sets it up (ctrl transposed, data and global
/// INVALID).
pub(super) fn cpu_witness(
    program: &Program,
    pf: &Preflight,
    mode: StepMode,
) -> (Vec<u32>, Vec<u32>) {
    let total = 1usize << program.po2;
    let hal: CpuHal<BabyBear> = CpuHal::new(Sha256HashSuite::new_suite());
    let ctrl = hal.copy_from_elem("ctrl", &host_ctrl_for(program));
    let data = hal.alloc_elem_init("data", total * CIRCUIT.data_size(), BabyBearElem::INVALID);
    let global = hal.alloc_elem_init("global", CircuitImpl::OUTPUT_SIZE, BabyBearElem::INVALID);
    // The CPU witgen advances iopIdx in the trace it is given: run it on a copy.
    let cycles = pf.trace.cycles.clone();
    let mut trace = raw_trace(program, pf);
    trace.cycles = cycles.as_ptr();
    CpuCircuitHal
        .generate_witness(mode, total as u32, &trace, &ctrl, &data, &global)
        .unwrap();
    (raw(&data.to_vec()), raw(&global.to_vec()))
}

pub(super) struct MetalRun {
    pub(super) hal: Arc<MetalHalPoseidon2>,
    pub(super) circuit: MetalCircuitHal<risc0_zkp::hal::metal::MetalHashPoseidon2>,
}

impl MetalRun {
    pub(super) fn new() -> Self {
        let hal = Arc::new(MetalHalPoseidon2::new());
        let circuit = MetalCircuitHal::new(hal.clone()).unwrap();
        Self { hal, circuit }
    }

    /// The Metal witness (data, global); returns the peak device allocation during the call.
    pub(super) fn witness(
        &self,
        program: &Program,
        trace: &RawPreflightTrace,
        mode: StepMode,
        chains: Option<&HostChains>,
    ) -> Result<(Vec<u32>, Vec<u32>, u64)> {
        let total = 1usize << program.po2;
        let ctrl = self.hal.copy_from_elem("ctrl", &host_ctrl_for(program));
        let data =
            self.hal
                .alloc_elem_init("data", total * CIRCUIT.data_size(), BabyBearElem::INVALID);
        let global =
            self.hal
                .alloc_elem_init("global", CircuitImpl::OUTPUT_SIZE, BabyBearElem::INVALID);
        self.hal.sync();
        self.hal.reset_peak_allocated_size();
        self.circuit.generate_witness_with_chains(
            mode,
            total as u32,
            trace,
            chains,
            &ctrl,
            &data,
            &global,
        )?;
        let peak = self.hal.peak_allocated_size();
        Ok((raw(&data.to_vec()), raw(&global.to_vec()), peak))
    }
}

fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

pub(super) fn assert_same(
    what: &str,
    total: usize,
    metal: &(Vec<u32>, Vec<u32>),
    cpu: &(Vec<u32>, Vec<u32>),
) {
    if let Some(i) = first_diff(&metal.0, &cpu.0) {
        panic!(
            "{what}: data differs at col {} row {}: metal {:#x} cpu {:#x}",
            i / total,
            i % total,
            metal.0[i],
            cpu.0[i]
        );
    }
    if let Some(i) = first_diff(&metal.1, &cpu.1) {
        panic!("{what}: global differs at {i}");
    }
}

/// Chains of at least `min_len` cycles executed on the host (`witgen::try_host_chains_range` with
/// a configurable threshold; the production threshold is 16).
pub(super) fn chains_with_min_len(
    program: &Program,
    pf: &Preflight,
    min_len: usize,
) -> Option<HostChains> {
    let cycles = &pf.trace.cycles[..program.code_rows()];
    let n = cycles.len();
    let (mut chains, mut heads, mut start, mut num_rows) = (Vec::new(), Vec::new(), 0usize, 0);
    for c in 1..=n {
        if cycles.get(c).is_none_or(|cy| cy.is_par_safe != 0) {
            let len = c - start;
            if len >= min_len {
                chains.extend([start as u32, len as u32]);
                heads.push(start as u32);
                num_rows += len;
            }
            start = c;
        }
    }
    if heads.is_empty() {
        return None;
    }
    let data_cols = CIRCUIT.data_size();
    let trace = raw_trace(program, pf);
    let mut out = HostChains {
        heads,
        cycles: vec![0; num_rows],
        rows: vec![0; num_rows * data_cols],
        wom_idx: vec![0; num_rows],
        wom_rows: vec![0; num_rows * 9 * 5],
        pinned: None,
    };
    risc0_sys::ffi_wrap(|| unsafe {
        risc0_circuit_recursion_cpu_chains(
            program.code.as_ptr(),
            program.code_size as u32,
            &trace,
            1u32 << program.po2,
            data_cols as u32,
            chains.as_ptr(),
            out.heads.len() as u32,
            out.cycles.as_mut_ptr(),
            out.rows.as_mut_ptr(),
            out.wom_idx.as_mut_ptr(),
            out.wom_rows.as_mut_ptr(),
        )
    })
    .ok()?;
    Some(out)
}

pub(super) fn witgen_matches_cpu(po2: usize) {
    let (program, pf) = preflight_for(po2);
    let total = 1usize << po2;
    let trace = raw_trace(&program, &pf);
    let pristine = pf.trace.cycles.clone();
    let cpu = cpu_witness(&program, &pf, StepMode::Parallel);
    let m = MetalRun::new();
    let calls = thread_calls(Call::Parallel);
    let t0 = std::time::Instant::now();
    let (d, g, peak) = m
        .witness(&program, &trace, StepMode::Parallel, None)
        .unwrap();
    let first_s = t0.elapsed().as_secs_f64();
    assert!(thread_calls(Call::Parallel) > calls);
    assert_same("parallel", total, &(d, g), &cpu);
    assert_eq!(pf.trace.cycles, pristine, "caller's trace modified");
    // Again on the same circuit HAL (scratch reused), then with the production host chains and
    // with every chain of >= 2 cycles on the host.
    let t0 = std::time::Instant::now();
    let (d, g, _) = m
        .witness(&program, &trace, StepMode::Parallel, None)
        .unwrap();
    let second_s = t0.elapsed().as_secs_f64();
    assert_same("parallel (2nd)", total, &(d, g), &cpu);
    let prod = host_chains(&program, &pf);
    let mut n_prod = 0;
    if let Some(hc) = prod.as_ref() {
        n_prod = hc.heads.len();
        let (d, g, _) = m
            .witness(&program, &trace, StepMode::Parallel, Some(hc))
            .unwrap();
        assert_same("parallel + host chains (min 16)", total, &(d, g), &cpu);
    }
    let small = chains_with_min_len(&program, &pf, 2).expect("chains of >= 2 cycles");
    let chain_calls = thread_calls(Call::Chain);
    let (d, g, _) = m
        .witness(&program, &trace, StepMode::Parallel, Some(&small))
        .unwrap();
    assert!(thread_calls(Call::Chain) > chain_calls);
    assert_same("parallel + host chains (min 2)", total, &(d, g), &cpu);
    assert_eq!(pf.trace.cycles, pristine, "caller's trace modified");
    eprintln!(
        "C6 po2 {po2}: {} cycles, metal witgen {first_s:.3}s (first) {second_s:.3}s (second); \
         host chains: {n_prod} (min 16), {} heads / {} rows (min 2); peak device allocation \
         during the witgen {peak} B ({:.1} MB)",
        program.code_rows(),
        small.heads.len(),
        small.cycles.len(),
        peak as f64 / 1e6
    );
}

#[test]
fn witgen_parallel_matches_cpu_po2_16() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    witgen_matches_cpu(16);
}

#[test]
fn witgen_parallel_matches_cpu_po2_18() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    witgen_matches_cpu(18);
}

#[test]
fn witgen_seq_modes_match_cpu() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    let (program, pf) = preflight_for(16);
    let total = 1usize << 16;
    let trace = raw_trace(&program, &pf);
    let m = MetalRun::new();
    for mode in [StepMode::SeqForward, StepMode::SeqReverse] {
        let cpu = cpu_witness(&program, &pf, mode);
        let calls = thread_calls(Call::SeqCpu);
        let (d, g, _) = m.witness(&program, &trace, mode, None).unwrap();
        assert!(thread_calls(Call::SeqCpu) > calls);
        assert_same("seq", total, &(d, g), &cpu);
    }
    // The sequential witness equals the parallel one.
    let par = cpu_witness(&program, &pf, StepMode::Parallel);
    let (d, g, _) = m
        .witness(&program, &trace, StepMode::SeqForward, None)
        .unwrap();
    assert_same("seq vs parallel", total, &(d, g), &par);
}

/// A corrupted trace element makes the witgen fail with a fault error, and the HAL stays usable.
#[test]
fn witgen_fault_on_corrupt_trace() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    let (program, pf) = preflight_for(16);
    let trace = raw_trace(&program, &pf);
    let m = MetalRun::new();

    // 1. An IOP index past the IOP table (caught by readIOPBody's clamp in parStepExec).
    let reader = pf.trace.cycles[..program.code_rows()]
        .windows(2)
        .position(|w| w[1].iop_idx != w[0].iop_idx)
        .expect("a cycle that reads IOP bodies");
    let mut cycles = pf.trace.cycles.clone();
    cycles[reader].iop_idx = pf.trace.iops.len() as u32 + 1000;
    let mut bad = raw_trace(&program, &pf);
    bad.cycles = cycles.as_ptr();
    let err = m
        .witness(&program, &bad, StepMode::Parallel, None)
        .unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("Witgen") && msg.contains("fault"), "{msg}");

    // 2. A WOM table entry changed (caught by step_verify_mem's write-once checks in
    //    parStepVerifyWom, or by the exec asserts): try entries until one is detected.
    let mut detected = None;
    for idx in (1..pf.trace.wom.len()).step_by(pf.trace.wom.len().div_ceil(64).max(1)) {
        let mut wom = pf.trace.wom.clone();
        wom[idx] += risc0_core::field::baby_bear::BabyBearExtElem::ONE;
        let mut bad = raw_trace(&program, &pf);
        bad.wom = wom.as_ptr();
        if let Err(err) = m.witness(&program, &bad, StepMode::Parallel, None) {
            let msg = format!("{err:#}");
            assert!(msg.contains("fault"), "{msg}");
            detected = Some((idx, msg));
            break;
        }
    }
    let (idx, msg) = detected.expect("no corrupted WOM entry detected");
    eprintln!("C6 fault: wom[{idx}] corrupted -> {msg}");

    // 3. Host chains pointing outside the trace are refused.
    let mut hc = chains_with_min_len(&program, &pf, 2).unwrap();
    hc.cycles[0] = u32::MAX;
    assert!(m
        .witness(&program, &trace, StepMode::Parallel, Some(&hc))
        .is_err());

    // The HAL recovers: a good trace still matches the CPU.
    let cpu = cpu_witness(&program, &pf, StepMode::Parallel);
    let (d, g, _) = m
        .witness(&program, &trace, StepMode::Parallel, None)
        .unwrap();
    assert_same("after faults", 1 << 16, &(d, g), &cpu);
}

/// A full prove of `program` through the Metal recursion prover with `pf` prepared.
fn prove_metal(program: &Program, pf: Preflight) -> Result<crate::prove::RecursionReceipt> {
    let prover = super::super::recursion_prover("poseidon2")?;
    prover.prove(program, VecDeque::new(), Some(pf))
}

/// RISC0_RECURSION_PREWITGEN_CHECK / RISC0_RECURSION_SHADOW_CHECK: with a host witness in the
/// preflight, the prover regenerates it through the Metal witgen, compares, and uploads the host
/// witness with `upload_host_witness`. A corrupted host witness is caught.
#[test]
fn prewitgen_and_shadow_check_hooks() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    let po2 = 16;
    let cpu_receipt = crate::prove::hal::cpu::recursion_prover("poseidon2")
        .unwrap()
        .prove(
            &get_zkr("test_recursion_circuit.zkr", po2).unwrap(),
            test_input(),
            None,
        )
        .unwrap();
    let globals = CircuitImpl::OUTPUT_SIZE;
    let with_host_witness = |corrupt: bool| {
        let (program, pf) = preflight_for(po2);
        let (data, global) = cpu_witness(&program, &pf, StepMode::Parallel);
        let mut data: Vec<BabyBearElem> = data.into_iter().map(BabyBearElem::new_raw).collect();
        let global: Vec<BabyBearElem> = global.into_iter().map(BabyBearElem::new_raw).collect();
        if corrupt {
            let i = 3 * (1 << po2) + 5;
            data[i] += BabyBearElem::ONE;
        }
        *pf.host_witness.lock().unwrap() = Some((data, global));
        (program, pf)
    };
    // SAFETY (env): only this test sets these variables; the other tests never put a host
    // witness in a preflight, the only case the variables are read.
    for (var, val) in [
        ("RISC0_RECURSION_PREWITGEN_CHECK", "1"),
        ("RISC0_RECURSION_SHADOW_CHECK", "1"),
    ] {
        std::env::remove_var("RISC0_RECURSION_PREWITGEN_CHECK");
        std::env::remove_var("RISC0_RECURSION_SHADOW_CHECK");
        std::env::set_var(var, val);
        let calls = thread_calls(Call::Parallel);
        let (program, pf) = with_host_witness(false);
        let r = prove_metal(&program, pf).unwrap();
        assert!(
            thread_calls(Call::Parallel) > calls,
            "{var}: no Metal witgen"
        );
        assert_eq!(r.output, cpu_receipt.output, "{var}");
        assert_eq!(r.seal[..globals], cpu_receipt.seal[..globals], "{var}");
        let (program, pf) = with_host_witness(true);
        let err = prove_metal(&program, pf).unwrap_err();
        assert!(
            format!("{err:#}").contains("prewitgen witness mismatch in data"),
            "{var}: {err:#}"
        );
    }
    std::env::remove_var("RISC0_RECURSION_PREWITGEN_CHECK");
    std::env::remove_var("RISC0_RECURSION_SHADOW_CHECK");
    // Without a check, the host witness is uploaded as is (upload_host_witness): the proof is
    // the CPU prover's.
    let calls = thread_calls(Call::Parallel);
    let (program, pf) = with_host_witness(false);
    let r = prove_metal(&program, pf).unwrap();
    assert_eq!(thread_calls(Call::Parallel), calls);
    assert_eq!(r.output, cpu_receipt.output);
    assert_eq!(r.seal[..globals], cpu_receipt.seal[..globals]);
}

#[test]
fn upload_host_witness_rejects_wrong_sizes() {
    let m = MetalRun::new();
    let data = m.hal.alloc_elem("data", 64);
    let global = m.hal.alloc_elem("global", 8);
    let v = |n| vec![BabyBearElem::new(7); n];
    assert!(m
        .circuit
        .upload_host_witness(&data, &global, v(63), v(8))
        .is_err());
    m.circuit
        .upload_host_witness(&data, &global, v(64), v(8))
        .unwrap();
    assert!(data.to_vec().iter().all(|x| *x == BabyBearElem::new(7)));
    assert!(global.to_vec().iter().all(|x| *x == BabyBearElem::new(7)));
}

/// The host chain check hook (`set_host_chains_check`) on full Metal proofs: every witness
/// generated with host chains is regenerated without them and must match. Prints the witgen
/// peak allocation per po2.
#[test]
fn prove_with_host_chains_check() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    for po2 in [16, 18] {
        let program = get_zkr("test_recursion_circuit.zkr", po2).unwrap();
        let (_, pf) = preflight_for(po2);
        let expect_chains = host_chains(&program, &pf).is_some();
        let before = crate::prove::set_host_chains_check(true);
        PHASE_ALLOC[0].store(0, Ordering::Relaxed);
        let r = super::super::recursion_prover("poseidon2").unwrap().prove(
            &program,
            test_input(),
            None,
        );
        let after = crate::prove::set_host_chains_check(false);
        r.unwrap();
        if expect_chains {
            assert!(after > before, "host chain check did not run");
        }
        eprintln!(
            "C6 prove po2 {po2}: host chains {expect_chains}, checks {}; device allocation at \
             witgen (max of entry/exit) {} B",
            after - before,
            PHASE_ALLOC[0].load(Ordering::Relaxed)
        );
    }
}

fn reference_sort(rows: &[WomRow], counts: &[u32]) -> (Vec<WomRow>, Vec<u32>) {
    let mut valid = Vec::new();
    for (c, &n) in counts.iter().enumerate() {
        valid.extend_from_slice(&rows[c * 9..c * 9 + n as usize]);
    }
    valid.sort_by_key(sort_key); // stable full sort by operator< (canonical tuple)
    let mut out = vec![[INVALID; 5]; rows.len()];
    out[..valid.len()].copy_from_slice(&valid);
    let mut scan = Vec::with_capacity(counts.len());
    let mut acc = 0;
    for &n in counts {
        scan.push(acc);
        acc += n;
    }
    (out, scan)
}

fn random_rows(rng: &mut impl Rng, cycles: usize, dense: bool) -> (Vec<WomRow>, Vec<u32>) {
    let mut rows = vec![[INVALID; 5]; cycles * 9];
    let mut counts = vec![0u32; cycles];
    let enc = |x: u32| BabyBearElem::new(x).as_u32_montgomery();
    for c in 0..cycles {
        let n = rng.random_range(0..=9u32);
        counts[c] = n;
        for j in 0..n as usize {
            let addr = if dense {
                rng.random_range(0..(cycles as u32).max(1))
            } else {
                rng.random_range(0..0x7000_0000u32)
            };
            // Mostly identical values per address (a consistent WOM), sometimes not.
            let v = if rng.random_bool(0.9) {
                addr.wrapping_mul(2654435761) % 0x7800_0000
            } else {
                rng.random_range(0..0x7800_0000u32)
            };
            rows[c * 9 + j] = [enc(addr), enc(v), enc(v ^ 1), enc(3), enc(c as u32)];
        }
    }
    (rows, counts)
}

#[test]
fn wom_sort_matches_full_sort() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(6);
    for (cycles, dense) in [
        (1, true),
        (2, true),
        (100, true),
        (5000, true),
        (5000, false),
        (70000, true),
    ] {
        let (mut rows, mut counts) = random_rows(&mut rng, cycles, dense);
        let (want_rows, want_scan) = reference_sort(&rows, &counts);
        let sum: usize = counts.iter().map(|&n| n as usize).sum();
        let nv = sort_wom_rows(&mut rows, &mut counts).unwrap();
        assert_eq!(nv, sum);
        assert!(
            rows == want_rows,
            "rows differ ({cycles} cycles, dense {dense})"
        );
        assert_eq!(counts, want_scan);
    }
    // A count above 9 is refused.
    let mut rows = vec![[INVALID; 5]; 9];
    assert!(sort_wom_rows(&mut rows, &mut [10]).is_err());
}
