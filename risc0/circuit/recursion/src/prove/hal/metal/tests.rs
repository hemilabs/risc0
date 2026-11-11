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

//! C9 integration tests of the recursion Metal HAL (PLAN 4.3 row C9): the translators' `--check`
//! round trips, a metallib smoke test, bit-identical seals against the CPU prover (deterministic
//! ZK noise, [with_noise_seed]), identity_p254 through the CPU fallback with the host-chain check
//! on, and the CPU HAL witgen leaving the caller's trace untouched. The per-phase equivalence and
//! fault-path tests live with their phases (`c6_tests`, `c7_tests`, `c8_tests`).

use std::{
    collections::{BTreeSet, VecDeque},
    path::PathBuf,
    process::Command,
    sync::atomic::Ordering,
};

use risc0_circuit_recursion_sys::{RawPreflightTrace, StepMode};
use risc0_core::field::baby_bear::BabyBearElem;
use risc0_zkp::{
    adapter::{CircuitInfo as _, TapsProvider as _},
    field::Elem as _,
    hal::metal::MetalHalPoseidon2,
};
use test_log::test;

use super::*;
use crate::{
    prove::{
        hal::{cpu::CpuCircuitHal, CircuitWitnessGenerator as _},
        preflight::Preflight,
        witgen::host_ctrl_for,
        zkr::get_zkr,
    },
    CircuitImpl, CIRCUIT,
};

fn test_input() -> VecDeque<u32> {
    // Two Poseidon2 digests (8 words each), as zkvm's `test_zkr` adds them.
    (0..16u32).map(|i| i * 0x101 + 7).collect()
}

fn translator_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../recursion-sys/kernels/cuda")
}

/// Run a translator with `--check` (in-memory translation, pinned rewrite counts, residual scan,
/// round trip; nothing written). PLAN 4.3 row C9 / C4 note: the build enforces only the pins.
fn run_translator_check(script: &str, extra: &[&str]) {
    let dir = translator_dir();
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let out = Command::new(&python)
        .arg(dir.join(script))
        .arg("--check")
        .args(extra)
        .current_dir(&dir)
        .output()
        .unwrap_or_else(|e| panic!("running {python} {script}: {e}"));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    eprintln!("{script} --check {extra:?}:\n{stdout}{stderr}");
    assert!(
        out.status.success(),
        "{script} --check {extra:?} failed ({}):\n{stdout}\n{stderr}",
        out.status
    );
}

#[test]
fn translator_steps_check() {
    run_translator_check("cuda2metal_steps.py", &[]);
}

#[test]
fn translator_eval_check_monolith() {
    run_translator_check("cuda2metal_eval.py", &[]);
}

#[test]
fn translator_eval_check_split() {
    run_translator_check("cuda2metal_eval.py", &["--split"]);
}

/// The embedded metallib holds exactly the entry points `METAL_KERNEL_NAMES` lists (no PSO is
/// built: cold eval_check PSOs take tens of seconds), and the base kernels load.
#[test]
fn metallib_smoke() {
    let device = metal::Device::system_default().expect("no Metal device");
    let library = device
        .new_library_with_data(METAL_LIB)
        .expect("METAL_LIB loads");
    let in_lib: BTreeSet<String> = library.function_names().into_iter().collect();
    let listed: BTreeSet<String> = METAL_KERNEL_NAMES.iter().map(|s| s.to_string()).collect();
    assert_eq!(listed.len(), METAL_KERNEL_NAMES.len(), "duplicate names");
    let missing: Vec<_> = listed.difference(&in_lib).collect();
    assert!(
        missing.is_empty(),
        "listed but not in METAL_LIB: {missing:?}"
    );
    let unlisted: Vec<_> = in_lib.difference(&listed).collect();
    assert!(
        unlisted.is_empty(),
        "in METAL_LIB but not listed: {unlisted:?}"
    );
    for name in EVAL_MONOLITH.iter().chain(EVAL_SPLIT) {
        assert!(listed.contains(*name), "{name}");
    }
    let kernels = RecursionKernels::get(&device).unwrap();
    for name in [
        "parStepExec",
        "parStepVerifyWom",
        "parStepComputeAccum",
        "parStepVerifyAccum",
    ] {
        assert!(kernels.base.get(name).is_some(), "{name}");
    }
    eprintln!(
        "C9 metallib: {} bytes, {} entry points",
        METAL_LIB.len(),
        in_lib.len()
    );
}

/// The seal check is not vacuous: without a noise seed two CPU proofs of the same input differ
/// (ZK noise); with the same seed they are identical; with different seeds they differ.
#[test]
fn noise_seed_controls_the_seal() {
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    let program = get_zkr("test_recursion_circuit.zkr", 16).unwrap();
    let cpu = crate::prove::hal::cpu::recursion_prover("poseidon2").unwrap();
    let a = cpu.prove(&program, test_input(), None).unwrap();
    let b = cpu.prove(&program, test_input(), None).unwrap();
    assert_ne!(a.seal, b.seal, "unseeded seals should differ");
    let c = with_noise_seed(7, || cpu.prove(&program, test_input(), None)).unwrap();
    let d = with_noise_seed(7, || cpu.prove(&program, test_input(), None)).unwrap();
    let e = with_noise_seed(8, || cpu.prove(&program, test_input(), None)).unwrap();
    assert_eq!(c.seal, d.seal);
    assert_ne!(c.seal, e.seal);
    // The seed is restored: unseeded again afterwards.
    let f = cpu.prove(&program, test_input(), None).unwrap();
    assert_ne!(f.seal, c.seal);
}

/// Full Metal proofs (all three circuit phases on the GPU) with deterministic noise give the CPU
/// prover's seal bit for bit. Prints wall time and peak device memory per phase (PLAN 9.1).
fn seal_matches_cpu(hashfn: &str, po2: usize) {
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    let program = get_zkr("test_recursion_circuit.zkr", po2).unwrap();
    let metal = metal_recursion_prover(hashfn).unwrap();
    let cpu = crate::prove::hal::cpu::recursion_prover(hashfn).unwrap();
    // Warm (PSOs, ctrl caches, pool) so the timing below is a steady-state proof.
    metal.prove(&program, test_input(), None).unwrap();
    cpu.prove(&program, test_input(), None).unwrap();
    let probe = MetalHalPoseidon2::new();
    probe.reset_peak_allocated_size();
    for a in &PHASE_ALLOC {
        a.store(0, Ordering::Relaxed);
    }
    let calls = phase_calls();
    let ok = SEAL_CHECKS_OK.load(Ordering::Relaxed);
    let r = prove_seal_checked(metal.as_ref(), cpu.as_ref(), &program, test_input(), None).unwrap();
    assert_eq!(SEAL_CHECKS_OK.load(Ordering::Relaxed), ok + 1);
    let after = phase_calls();
    for i in 0..3 {
        assert!(after[i] > calls[i], "phase {i} not served by the Metal HAL");
    }
    let alloc = phase_alloc();
    eprintln!(
        "C9 {hashfn} po2 {po2}: seal {} words == CPU; device alloc at witgen {} B, accum {} B, \
         eval_check {} B, proof peak {} B",
        r.seal.len(),
        alloc[0],
        alloc[1],
        alloc[2],
        probe.peak_allocated_size()
    );
}

#[test]
fn seal_matches_cpu_poseidon2_po2_16() {
    seal_matches_cpu("poseidon2", 16);
}

#[test]
fn seal_matches_cpu_poseidon2_po2_18() {
    seal_matches_cpu("poseidon2", 18); // zkvm RECURSION_PO2: the lift/join shape
}

#[test]
fn seal_matches_cpu_sha256_po2_16() {
    seal_matches_cpu("sha-256", 16);
}

/// A seal mismatch is reported (not vacuous): the "CPU" side proves with another seed.
#[test]
fn seal_check_detects_a_difference() {
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    struct Reseeded(Box<dyn RecursionProver>);
    impl RecursionProver for Reseeded {
        fn prove(
            &self,
            program: &Program,
            input: VecDeque<u32>,
            preflight: Option<Preflight>,
        ) -> Result<RecursionReceipt> {
            with_noise_seed(1, || self.0.prove(program, input, preflight))
        }
    }
    let program = get_zkr("test_recursion_circuit.zkr", 16).unwrap();
    let metal = metal_recursion_prover("poseidon2").unwrap();
    let cpu = Reseeded(crate::prove::hal::cpu::recursion_prover("poseidon2").unwrap());
    let err = prove_seal_checked(metal.as_ref(), &cpu, &program, test_input(), None).unwrap_err();
    assert!(format!("{err:#}").contains("seal differs"), "{err:#}");
}

/// The CPU HAL witgen runs on a copy of the cycles: the caller's trace is unchanged and a second
/// witgen over it gives the same witness (the host-chain / prewitgen checks rerun it).
#[test]
fn cpu_witgen_keeps_trace() {
    let po2 = 16;
    let program = get_zkr("test_recursion_circuit.zkr", po2).unwrap();
    let mut pf = Preflight::new(test_input());
    for (cycle, row) in program.code_by_row().enumerate() {
        pf.step(cycle, row).unwrap();
    }
    let total = 1usize << po2;
    let ctrl = host_ctrl_for(&program);
    let cycles_before = pf.trace.cycles.clone();
    let raw = RawPreflightTrace {
        wom: pf.trace.wom.as_ptr(),
        cycles: pf.trace.cycles.as_ptr(),
        iops: pf.trace.iops.as_ptr(),
        num_woms: pf.trace.wom.len() as u32,
        num_cycles: program.code_rows() as u32,
        num_iops: pf.trace.iops.len() as u32,
    };
    let run = || {
        let data = vec![BabyBearElem::INVALID; total * CIRCUIT.data_size()];
        let global = vec![BabyBearElem::INVALID; CircuitImpl::OUTPUT_SIZE];
        let hal = risc0_zkp::hal::cpu::CpuHal::<risc0_core::field::baby_bear::BabyBear>::new(
            risc0_zkp::core::hash::poseidon2::Poseidon2HashSuite::new_suite(),
        );
        use risc0_zkp::hal::{Buffer as _, Hal as _};
        let (c, d, g) = (
            hal.copy_from_elem("ctrl", &ctrl),
            hal.copy_from_elem("data", &data),
            hal.copy_from_elem("global", &global),
        );
        CpuCircuitHal
            .generate_witness(StepMode::Parallel, total as u32, &raw, &c, &d, &g)
            .unwrap();
        let raw = |v: Vec<BabyBearElem>| -> Vec<u32> {
            v.iter().map(|e| e.as_u32_montgomery()).collect()
        };
        (raw(d.to_vec()), raw(g.to_vec()))
    };
    let a = run();
    assert_eq!(
        pf.trace.cycles, cycles_before,
        "CPU witgen changed the trace"
    );
    let b = run();
    assert!(a == b, "second CPU witgen over the same trace differs");
}

/// identity_p254 runs on the CPU fallback (no Metal Poseidon254 until Z10) through the public
/// [crate::prove::Prover] path (prepare: preflight + host chains), with the host-chain check on.
/// Before C9 the check's second CPU witgen read the iopIdx the first one had advanced and failed
/// ("eqz failed at: zirgen/circuit/recursion/wom.cpp:78"), which kept zkvm's
/// `test_recursion_host_chains_check` ignored on Metal.
#[test]
fn p254_cpu_fallback_with_host_chains_check() {
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    for hashfn in ["poseidon_254", "poseidon2"] {
        let program = get_zkr("test_recursion_circuit.zkr", 16).unwrap();
        let before = crate::prove::set_host_chains_check(true);
        let mut prover = crate::prove::Prover::new(program.clone(), hashfn);
        prover.add_input(test_input().make_contiguous());
        let r = prover.prepare().and_then(|_| prover.run());
        let after = crate::prove::set_host_chains_check(false);
        let r = r.unwrap();
        let cpu = crate::prove::hal::cpu::recursion_prover(hashfn).unwrap();
        let c = cpu.prove(&program, test_input(), None).unwrap();
        assert_eq!(r.output, c.output);
        let globals = CircuitImpl::OUTPUT_SIZE;
        assert_eq!(r.seal[..globals], c.seal[..globals]);
        eprintln!("C9 {hashfn}: host chain checks {}", after - before);
        assert!(after > before, "{hashfn}: no host-chain check ran");
    }
}
