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

//! Z10 tests: the `poseidon_254` recursion prover on Metal (seal bit-exact against the CPU
//! prover with a fixed noise seed, the `RISC0_METAL_REC_P254=cpu` revert knob) and the profile /
//! timing helpers of `docs/metal-port/notes/Z10.md`.

use std::{collections::VecDeque, sync::atomic::Ordering, time::Instant};

use risc0_core::field::baby_bear::{BabyBear, BabyBearElem};
use risc0_zkp::{
    adapter::{CircuitInfo as _, TapsProvider as _},
    core::hash::poseidon_254::Poseidon254HashSuite,
    hal::{cpu::CpuHal, Hal},
    INV_RATE,
};
use test_log::test;

use super::*;
use crate::{
    prove::{witgen::with_noise_seed, zkr::get_zkr},
    CircuitImpl, CIRCUIT, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CODE, REGISTER_GROUP_DATA,
};

fn test_input() -> VecDeque<u32> {
    (0..16u32).map(|i| i * 0x101 + 7).collect()
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// The Merkle shapes (rows, cols) a recursion proof at `po2` hashes: the three register groups,
/// the check group and the FRI rounds (each one tree; `hash_rows` + a fold tree of `rows`).
pub(crate) fn p254_shapes(po2: usize) -> Vec<(&'static str, usize, usize)> {
    let taps = CIRCUIT.get_taps();
    let domain = (1usize << po2) * INV_RATE;
    let mut v = vec![
        ("code", domain, taps.group_size(REGISTER_GROUP_CODE)),
        ("data", domain, taps.group_size(REGISTER_GROUP_DATA)),
        ("accum", domain, taps.group_size(REGISTER_GROUP_ACCUM)),
        ("check", domain, INV_RATE * 4),
    ];
    // FRI: fold by 16 while the degree is above 256.
    let mut degree = 1usize << po2;
    while degree > 256 {
        degree /= risc0_zkp::FRI_FOLD;
        v.push(("fri", degree * INV_RATE, risc0_zkp::FRI_FOLD * 4));
    }
    v
}

fn random_elems(n: usize, seed: u64) -> Vec<BabyBearElem> {
    use rand::{Rng as _, SeedableRng as _};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n).map(|_| BabyBearElem::new(rng.random::<u32>() % 0x7800_0001)).collect()
}

/// Profile (Z10 step 1): where an identity_p254-shaped proof (po2 18) spends its time on the CPU
/// prover, against the Metal poseidon2 prover, plus the CPU cost of the Poseidon254 Merkle work
/// alone at every shape. `Z10_PROFILE_PO2` (18), `Z10_PROFILE_REPS` (2).
#[test]
#[ignore]
fn z10_profile_cpu_p254() {
    let _guard = global_state_test_lock();
    let po2 = env_usize("Z10_PROFILE_PO2", 18);
    let reps = env_usize("Z10_PROFILE_REPS", 2);
    let cpu = CpuHal::<BabyBear>::new(Poseidon254HashSuite::new_suite());
    let mut hash_total = 0.0;
    for (name, rows, cols) in p254_shapes(po2) {
        let m = cpu.copy_from_elem("m", &random_elems(rows * cols, rows as u64 ^ cols as u64));
        let nodes = cpu.alloc_digest("nodes", 2 * rows);
        let t = Instant::now();
        cpu.hash_rows(&nodes.slice(rows, rows), &m);
        let tr = t.elapsed().as_secs_f64();
        let t = Instant::now();
        cpu.hash_fold_tree(&nodes, rows.trailing_zeros() as usize);
        let tf = t.elapsed().as_secs_f64();
        hash_total += tr + tf;
        let perms = rows * cols.div_ceil(16);
        println!(
            "Z10PROF cpu p254 {name}: rows {rows} cols {cols}: hash_rows {:.3} s ({perms} perms), \
             fold tree {:.3} s ({} perms)",
            tr,
            tf,
            rows - 1
        );
    }
    println!("Z10PROF cpu p254 merkle total {hash_total:.3} s");
    let program = get_zkr("test_recursion_circuit.zkr", po2).unwrap();
    for hashfn in ["poseidon_254", "poseidon2"] {
        let p = crate::prove::hal::cpu::recursion_prover(hashfn).unwrap();
        for _ in 0..reps {
            let t = Instant::now();
            p.prove(&program, test_input(), None).unwrap();
            println!(
                "Z10PROF cpu prover {hashfn} po2 {po2}: {:.3} s",
                t.elapsed().as_secs_f64()
            );
        }
    }
    let m = metal_recursion_prover("poseidon2").unwrap();
    for _ in 0..reps + 1 {
        let t = Instant::now();
        m.prove(&program, test_input(), None).unwrap();
        println!(
            "Z10PROF metal prover poseidon2 po2 {po2}: {:.3} s",
            t.elapsed().as_secs_f64()
        );
    }
    let _ = CircuitImpl::OUTPUT_SIZE;
}

#[test]
fn rec_p254_knob_parses() {
    assert_eq!(parse_rec_p254("cpu"), Some(false));
    assert_eq!(parse_rec_p254(" CPU "), Some(false));
    assert_eq!(parse_rec_p254("metal"), Some(true));
    assert_eq!(parse_rec_p254("gpu"), Some(true));
    assert_eq!(parse_rec_p254(""), Some(true));
    assert_eq!(parse_rec_p254("cuda"), None);
}

/// The Metal poseidon_254 prover (all three circuit phases and the Merkle hashing on the GPU)
/// with a fixed noise seed gives the CPU prover's seal bit for bit, through
/// [prove_seal_checked] (C9's validation path, now also used for p254). Prints wall times and
/// the proof's peak device allocation.
fn p254_seal_matches_cpu(po2: usize) {
    let _guard = global_state_test_lock();
    let program = get_zkr("test_recursion_circuit.zkr", po2).unwrap();
    let metal = metal_p254_prover().unwrap();
    let cpu = crate::prove::hal::cpu::recursion_prover("poseidon_254").unwrap();
    // Warm (PSOs, ctrl caches, pool).
    metal.prove(&program, test_input(), None).unwrap();
    let probe = MetalHalPoseidon254::new();
    probe.reset_peak_allocated_size();
    let calls = phase_calls();
    let ok = SEAL_CHECKS_OK.load(Ordering::Relaxed);
    let r = prove_seal_checked(metal.as_ref(), cpu.as_ref(), &program, test_input(), None).unwrap();
    assert_eq!(SEAL_CHECKS_OK.load(Ordering::Relaxed), ok + 1);
    let after = phase_calls();
    for i in 0..3 {
        assert!(after[i] > calls[i], "phase {i} not served by the Metal circuit HAL");
    }
    // A second seeded Metal proof is identical (determinism).
    let a = with_noise_seed(11, || metal.prove(&program, test_input(), None)).unwrap();
    let b = with_noise_seed(11, || metal.prove(&program, test_input(), None)).unwrap();
    assert!(a.seal == b.seal, "two seeded Metal p254 proofs differ");
    eprintln!(
        "Z10 poseidon_254 po2 {po2}: seal {} words == CPU; proof peak {} B",
        r.seal.len(),
        probe.peak_allocated_size()
    );
}

#[test]
fn p254_seal_matches_cpu_po2_16() {
    p254_seal_matches_cpu(16);
}

#[test]
fn p254_seal_matches_cpu_po2_18() {
    p254_seal_matches_cpu(18); // RECURSION_PO2: identity_p254's shape
}

/// The factory's default `poseidon_254` arm is the Metal prover (unless the process runs with
/// `RISC0_METAL_REC_P254=cpu`): a proof through it is served by the Metal circuit HAL.
#[test]
fn factory_p254_uses_metal() {
    let _guard = global_state_test_lock();
    let program = get_zkr("test_recursion_circuit.zkr", 14).unwrap();
    let p = recursion_prover("poseidon_254").unwrap();
    let calls = phase_calls();
    let r = p.prove(&program, test_input(), None).unwrap();
    let after = phase_calls();
    let cpu = crate::prove::hal::cpu::recursion_prover("poseidon_254").unwrap();
    let c = cpu.prove(&program, test_input(), None).unwrap();
    assert_eq!(r.output, c.output);
    let globals = CircuitImpl::OUTPUT_SIZE;
    assert_eq!(r.seal[..globals], c.seal[..globals]);
    if p254_on_metal() {
        assert!(after[0] > calls[0], "poseidon_254 did not use the Metal circuit HAL");
    } else {
        assert_eq!(after[0], calls[0], "RISC0_METAL_REC_P254=cpu but the Metal HAL ran");
    }
}

/// identity_p254-shaped proofs (po2 18), the CPU prover (A, pre-Z10) and the Metal prover (B)
/// interleaved, `Z10_AB_PAIRS` pairs (7): median / min / max wall time of each.
#[test]
#[ignore]
fn z10_ab_p254_prover() {
    let _guard = global_state_test_lock();
    let po2 = env_usize("Z10_PROFILE_PO2", 18);
    let pairs = env_usize("Z10_AB_PAIRS", 7);
    let program = get_zkr("test_recursion_circuit.zkr", po2).unwrap();
    let cpu = crate::prove::hal::cpu::recursion_prover("poseidon_254").unwrap();
    let metal = metal_p254_prover().unwrap();
    metal.prove(&program, test_input(), None).unwrap();
    let (mut a, mut b) = (vec![], vec![]);
    for _ in 0..pairs {
        let t = Instant::now();
        cpu.prove(&program, test_input(), None).unwrap();
        a.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        metal.prove(&program, test_input(), None).unwrap();
        b.push(t.elapsed().as_secs_f64());
    }
    for (name, v) in [("A cpu", &mut a), ("B metal", &mut b)] {
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!(
            "Z10AB p254 prover po2 {po2} {name}: med {:.3} s [{:.3}, {:.3}] ({} runs)",
            v[v.len() / 2],
            v[0],
            v[v.len() - 1],
            v.len()
        );
    }
}

/// Metal p254 prover timing alone (`Z10_REPS`, 5), for the per-phase profile with
/// `RISC0_METAL_P254_KERNELS` / budgets set by the caller.
#[test]
#[ignore]
fn z10_time_metal_p254() {
    let _guard = global_state_test_lock();
    let po2 = env_usize("Z10_PROFILE_PO2", 18);
    let reps = env_usize("Z10_REPS", 5);
    let program = get_zkr("test_recursion_circuit.zkr", po2).unwrap();
    let metal = metal_p254_prover().unwrap();
    metal.prove(&program, test_input(), None).unwrap();
    let probe = MetalHalPoseidon254::new();
    probe.reset_peak_allocated_size();
    let mut v = vec![];
    for _ in 0..reps {
        let t = Instant::now();
        metal.prove(&program, test_input(), None).unwrap();
        v.push(t.elapsed().as_secs_f64());
    }
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!(
        "Z10T metal p254 po2 {po2}: med {:.3} s [{:.3}, {:.3}], peak alloc {} B",
        v[v.len() / 2],
        v[0],
        v[v.len() - 1],
        probe.peak_allocated_size()
    );
}
