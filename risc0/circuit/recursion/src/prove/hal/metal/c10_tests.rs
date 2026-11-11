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

//! C10 review regression tests (docs/metal-port/notes/C10.md): each one failed before its fix.

use std::{collections::VecDeque, panic::AssertUnwindSafe, sync::Arc};

use risc0_circuit_recursion_sys::{RawPreflightTrace, StepMode};
use risc0_core::field::baby_bear::BabyBearElem;
use risc0_zkp::{
    adapter::{CircuitInfo as _, TapsProvider as _},
    field::Elem as _,
    hal::{metal::MetalHalPoseidon2, Hal as _},
};
use test_log::test;

use super::*;
use crate::{
    prove::{hal::CircuitWitnessGenerator as _, zkr::get_zkr, RecursionProverImpl},
    CircuitImpl, CIRCUIT,
};

fn test_input() -> VecDeque<u32> {
    (0..16u32).map(|i| i * 0x101 + 7).collect()
}

/// M1's finding, reproduced: the zkp HAL's working-set budget panics inside `grow()` while the
/// scratch guard is held, which poisoned the scratch mutex, and every later proof of the same
/// (cached) prover failed with `PoisonError`. After the fix the next proof of the same circuit
/// HAL succeeds and matches the CPU prover.
#[test]
fn scratch_poison_recovers_after_budget_panic() {
    // A full Metal proof: serialized with the tests that assert on process-wide state (PLAN 8.7).
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    let hal = Arc::new(MetalHalPoseidon2::new());
    let circuit_hal = Arc::new(MetalCircuitHal::new(hal.clone()).unwrap());
    let before = SCRATCH_POISON_RECOVERIES.load(Ordering::Relaxed);

    // A budget no allocation fits: the first scratch allocation (the fault word) panics with the
    // zkp HAL's "working-set budget exceeded" while `scratch()` is held.
    hal.set_working_set_limit(Some(1));
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| circuit_hal.fault_word()));
    // Back to what a fresh HAL has: the env budget if one is set (the acceptance runs use plain
    // byte counts), else no explicit budget.
    let env_budget = std::env::var(risc0_zkp::hal::metal::MAX_WORKING_SET_ENV)
        .ok()
        .and_then(|t| t.trim().parse::<u64>().ok());
    hal.set_working_set_limit(env_budget);
    let msg = r
        .err()
        .and_then(|p| {
            p.downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        })
        .expect("the scratch allocation must panic under a 1-byte budget");
    assert!(msg.contains("working-set budget exceeded"), "{msg}");
    assert!(
        circuit_hal.scratch.is_poisoned(),
        "the panic must have poisoned the scratch mutex (else the test exercises nothing)"
    );

    // The same circuit HAL proves again (as the zkvm's thread-local cached prover would).
    let program = get_zkr("test_recursion_circuit.zkr", 16).unwrap();
    let metal = RecursionProverImpl::new(hal.clone(), circuit_hal.clone());
    let m = metal
        .prove(&program, test_input(), None)
        .expect("a proof after the poisoning panic must succeed");
    assert!(!circuit_hal.scratch.is_poisoned());
    assert!(SCRATCH_POISON_RECOVERIES.load(Ordering::Relaxed) > before);

    let cpu = crate::prove::hal::cpu::recursion_prover("poseidon2").unwrap();
    let c = cpu.prove(&program, test_input(), None).unwrap();
    assert_eq!(m.output, c.output);
    let globals = CircuitImpl::OUTPUT_SIZE;
    assert_eq!(m.seal[..globals], c.seal[..globals]);
}

/// An empty trace with buffers too small for the circuit: the step kernels / CPU witgen would
/// write every data column at `col * total_cycles + row`, so a short buffer must be rejected
/// before anything is encoded. Before the fix, the Parallel path accepted any data buffer of at
/// least 5 columns and returned `Ok` for an empty trace (and would write past a short one).
#[test]
fn witgen_rejects_short_buffers() {
    // Runs the Parallel witgen: serialized with the tests that assert on process-wide state.
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    const PO2: u32 = 4;
    let steps = 1usize << PO2;
    let hal = Arc::new(MetalHalPoseidon2::new());
    let circuit_hal = MetalCircuitHal::new(hal.clone()).unwrap();
    let trace = RawPreflightTrace {
        wom: std::ptr::null(),
        cycles: std::ptr::null(),
        iops: std::ptr::null(),
        num_woms: 0,
        num_cycles: 0,
        num_iops: 0,
    };
    let full = |n: usize| hal.alloc_elem_init("c10", n, BabyBearElem::INVALID);
    let ctrl = full(CIRCUIT.ctrl_size() * steps);
    let data = full(CIRCUIT.data_size() * steps);
    let global = full(CircuitImpl::OUTPUT_SIZE);
    // Full-size buffers: accepted (nothing to do for an empty trace).
    circuit_hal
        .generate_witness(
            StepMode::Parallel,
            steps as u32,
            &trace,
            &ctrl,
            &data,
            &global,
        )
        .unwrap();

    let short_data = full(5 * steps);
    let short_ctrl = full(steps);
    let short_global = full(1);
    for mode in [StepMode::Parallel, StepMode::SeqForward] {
        for (what, c, d, g) in [
            ("data", &ctrl, &short_data, &global),
            ("ctrl", &short_ctrl, &data, &global),
            ("global", &ctrl, &data, &short_global),
        ] {
            let err = circuit_hal
                .generate_witness(mode, steps as u32, &trace, c, d, g)
                .expect_err(what);
            assert!(format!("{err:#}").contains(what), "{what}: {err:#}");
        }
    }
    // A row count that is not a power of two is rejected on every path too.
    // A rejected call never reaches the Parallel witgen.
    let parallel = witgen::thread_calls(witgen::Call::Parallel);
    assert!(circuit_hal
        .generate_witness(
            StepMode::Parallel,
            steps as u32,
            &trace,
            &ctrl,
            &short_data,
            &global
        )
        .is_err());
    assert_eq!(witgen::thread_calls(witgen::Call::Parallel), parallel);
    let err = circuit_hal
        .generate_witness(
            StepMode::SeqForward,
            steps as u32 - 1,
            &trace,
            &ctrl,
            &data,
            &global,
        )
        .unwrap_err();
    assert!(format!("{err:#}").contains("rows"), "{err:#}");
}

/// The process-wide host ctrl cache stays usable after a panic poisoned its lock (a poisoned
/// `CTRL_CACHE` used to fail every later proof of the process with `PoisonError`).
#[test]
fn host_ctrl_cache_survives_poison() {
    let program = get_zkr("test_recursion_circuit.zkr", 16).unwrap();
    let a = crate::prove::witgen::host_ctrl_for(&program);
    let _ = std::thread::spawn(|| {
        let _guard = crate::prove::witgen::ctrl_cache();
        panic!("C10: poison the host ctrl cache on purpose");
    })
    .join();
    let b = crate::prove::witgen::host_ctrl_for(&program);
    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(b.iter()).all(|(x, y)| x == y));
}

/// PLAN 8.7: the witgen call counters the tests assert on are per thread, so a witgen on another
/// thread (another test of the parallel harness) does not move them. Before the fix the C6 hook
/// test asserted that the process-wide Parallel count stayed unchanged across one proof, which a
/// concurrent Parallel witgen on any other test thread broke.
#[test]
fn witgen_call_counters_are_per_thread() {
    use witgen::{process_calls, thread_calls, Call};
    let mine = thread_calls(Call::Parallel);
    let total = process_calls(Call::Parallel);
    std::thread::spawn(|| {
        const STEPS: usize = 16;
        let hal = Arc::new(MetalHalPoseidon2::new());
        let circuit_hal = MetalCircuitHal::new(hal.clone()).unwrap();
        let trace = RawPreflightTrace {
            wom: std::ptr::null(),
            cycles: std::ptr::null(),
            iops: std::ptr::null(),
            num_woms: 0,
            num_cycles: 0,
            num_iops: 0,
        };
        let full = |n: usize| hal.alloc_elem_init("c10", n, BabyBearElem::INVALID);
        let before = thread_calls(Call::Parallel);
        circuit_hal
            .generate_witness(
                StepMode::Parallel,
                STEPS as u32,
                &trace,
                &full(CIRCUIT.ctrl_size() * STEPS),
                &full(CIRCUIT.data_size() * STEPS),
                &full(CircuitImpl::OUTPUT_SIZE),
            )
            .unwrap();
        assert_eq!(thread_calls(Call::Parallel), before + 1);
    })
    .join()
    .unwrap();
    assert!(process_calls(Call::Parallel) > total);
    assert_eq!(thread_calls(Call::Parallel), mine);
}

/// The `RISC0_RECURSION_PREWITGEN_CHECK=once` set stays usable after a panic poisoned its lock.
#[test]
fn prewitgen_once_set_survives_poison() {
    let key = (usize::MAX - 7, 0xc10, 0xc10c10);
    assert!(crate::prove::witgen::first_prewitgen_check(key));
    let _ = std::thread::spawn(|| {
        let _guard = crate::prove::witgen::prewitgen_checked();
        panic!("C10: poison the prewitgen-once set on purpose");
    })
    .join();
    assert!(!crate::prove::witgen::first_prewitgen_check(key));
    assert!(crate::prove::witgen::first_prewitgen_check((
        usize::MAX - 8,
        0xc10,
        0xc10c10
    )));
}
