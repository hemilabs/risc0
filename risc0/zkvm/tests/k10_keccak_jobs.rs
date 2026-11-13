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

//! K10: keccak jobs end to end (session prover, `pending_keccaks` path).
//!
//! - `k10_rightsized_e2e`: a guest whose keccak batch is a full batch of `g` (K10_GUEST_PO2S,
//!   default 14,15,16), run with the keccak max po2 at K10_MAX_PO2 (default 17). The request must
//!   be proved at `g` (at the max with `RISC0_KECCAK_RIGHTSIZE_PO2=0`): its lifted receipt's
//!   control ID is `KECCAK_CONTROL_IDS[po2 - 14]`, and the composite receipt verifies against the
//!   guest image (which checks the assumption under `KECCAK_CONTROL_ROOT`).
//! - `k10_job_timing`: the A/B driver. K10_JOB = `small:<g>:<max>` (the keccak guest, one request
//!   of a full po2-`g` batch, max po2 `<max>`) or `union:<n>` (the multi-test guest, `n` full
//!   requests at the default po2 17). One warm-up proof, then K10_REPS (default 3) timed proofs,
//!   each verified; prints `K10TIMING job=.. rep=.. ms=..` lines and the process' peak Metal
//!   allocation. Uses only the public API, so it also builds against the unmodified tree (the A
//!   arm).
//!
//! Both are ignored (seconds to minutes, several GiB of GPU memory): run them under the machine's
//! GPU lock.

#![cfg(feature = "prove")]

use risc0_circuit_keccak::{compute_keccak_digest, KeccakState, KECCAK_CONTROL_IDS};
use risc0_circuit_keccak_methods::{KECCAK_ELF, KECCAK_ID};
use risc0_zkp::core::digest::Digest;
use risc0_zkvm::{get_prover_server, ExecutorEnv, InnerAssumptionReceipt, ProverOpts, Receipt};
use risc0_zkvm_methods::{multi_test::MultiTestSpec, MULTI_TEST_ELF, MULTI_TEST_ID};

/// The keccak guest's inputs (`circuit/keccak/methods/guest/src/bin/keccak.rs`).
fn guest_inputs(po2: usize) -> Vec<KeccakState> {
    let mut state = KeccakState::default();
    let mut pows = 987654321_u64;
    for part in state.as_mut_slice() {
        *part = pows;
        pows = pows.wrapping_mul(123456789);
    }
    vec![state; (1 << po2) / 200]
}

fn claim_digest(po2: usize) -> Digest {
    compute_keccak_digest(bytemuck::cast_slice(&guest_inputs(po2)))
}

fn env_list(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect()
}

fn prove_small(guest_po2: usize, max_po2: usize) -> Receipt {
    let env = ExecutorEnv::builder()
        .write(&(claim_digest(guest_po2), guest_po2 as u32))
        .unwrap()
        .keccak_max_po2(max_po2 as u32)
        .unwrap()
        .build()
        .unwrap();
    let receipt = get_prover_server(&ProverOpts::fast())
        .unwrap()
        .prove(env, KECCAK_ELF)
        .unwrap()
        .receipt;
    receipt.verify(KECCAK_ID).unwrap();
    receipt
}

fn prove_union(requests: usize) -> Receipt {
    let env = ExecutorEnv::builder()
        .write(&MultiTestSpec::KeccakUnion(requests))
        .unwrap()
        .build()
        .unwrap();
    let receipt = get_prover_server(&ProverOpts::fast())
        .unwrap()
        .prove(env, MULTI_TEST_ELF)
        .unwrap()
        .receipt;
    receipt.verify(MULTI_TEST_ID).unwrap();
    receipt
}

#[cfg(feature = "metal")]
fn assert_metal() {
    assert!(
        risc0_zkp::hal::metal::enabled("keccak"),
        "the Metal keccak prover is disabled (RISC0_METAL / RISC0_METAL_SKIP): not a Metal run"
    );
}

#[cfg(not(feature = "metal"))]
fn assert_metal() {}

#[test]
#[ignore]
fn k10_rightsized_e2e() {
    assert_metal();
    let max_po2 = env_list("K10_MAX_PO2", "17")[0];
    let rightsize = std::env::var("RISC0_KECCAK_RIGHTSIZE_PO2").map_or(true, |v| v.trim() != "0");
    for guest_po2 in env_list("K10_GUEST_PO2S", "14,15,16") {
        assert!(guest_po2 <= max_po2);
        let t0 = std::time::Instant::now();
        let receipt = prove_small(guest_po2, max_po2);
        let elapsed = t0.elapsed();
        let assumptions = &receipt.inner.composite().unwrap().assumption_receipts;
        assert_eq!(assumptions.len(), 1, "guest po2 {guest_po2}");
        let InnerAssumptionReceipt::Succinct(lifted) = &assumptions[0] else {
            panic!("guest po2 {guest_po2}: the keccak assumption is not a succinct receipt");
        };
        let proved_po2 = if rightsize { guest_po2 } else { max_po2 };
        assert_eq!(
            lifted.control_id,
            KECCAK_CONTROL_IDS[proved_po2 - 14],
            "guest po2 {guest_po2}, max {max_po2}: lift control ID"
        );
        println!(
            "K10 e2e: {} keccaks (full po2 {guest_po2}) under max po2 {max_po2}: proved at po2 \
             {proved_po2}, verified, {:.2} s",
            guest_inputs(guest_po2).len(),
            elapsed.as_secs_f64()
        );
    }
}

#[test]
#[ignore]
fn k10_job_timing() {
    assert_metal();
    let job = std::env::var("K10_JOB").unwrap_or_else(|_| "small:14:17".into());
    let reps = env_list("K10_REPS", "3")[0];
    let parts: Vec<&str> = job.split(':').collect();
    let run = || match parts[0] {
        "small" => drop(prove_small(
            parts[1].parse().unwrap(),
            parts[2].parse().unwrap(),
        )),
        "union" => drop(prove_union(parts[1].parse().unwrap())),
        other => panic!("K10_JOB {other}"),
    };
    run(); // warm-up: PSOs, archives, cached provers
    #[cfg(feature = "metal")]
    let hal = risc0_zkp::hal::metal::MetalHalPoseidon2::new();
    #[cfg(feature = "metal")]
    hal.reset_peak_allocated_size();
    let mut ms = Vec::new();
    for rep in 0..reps {
        let t0 = std::time::Instant::now();
        run();
        let t = t0.elapsed().as_secs_f64() * 1000.0;
        println!("K10TIMING job={job} rep={rep} ms={t:.1}");
        ms.push(t);
    }
    ms.sort_by(f64::total_cmp);
    #[cfg(feature = "metal")]
    let peak = hal.peak_allocated_size() as f64 / (1u64 << 20) as f64;
    #[cfg(not(feature = "metal"))]
    let peak = 0.0;
    println!(
        "K10SUMMARY job={job} min={:.1} median={:.1} max={:.1} peak_mib={peak:.0}",
        ms[0],
        ms[ms.len() / 2],
        ms[ms.len() - 1]
    );
}
