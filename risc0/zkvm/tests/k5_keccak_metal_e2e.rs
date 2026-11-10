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

//! End-to-end keccak proofs at every po2 of `KECCAK_PO2_RANGE` (14..=18), as `tests/keccak.rs`
//! does at po2 16/17, on whatever keccak prover the features select (`--features metal`: the
//! Metal keccak prover of K1-K6, the item K5 integration run). Each proof is lifted by the keccak
//! zkr of its po2, its control ID must be `KECCAK_CONTROL_IDS[po2 - 14]`, and the composite
//! receipt verifies against the guest image (which checks the assumption against
//! `KECCAK_CONTROL_ROOT`). Ignored by default (minutes per po2):
//!
//! `K5_E2E_PO2S=14,15,16,17,18 cargo test -p risc0-zkvm --release --features metal --test k5_keccak_metal_e2e -- --ignored --nocapture`

#![cfg(feature = "prove")]

use risc0_circuit_keccak::{
    compute_keccak_digest, KeccakState, KECCAK_CONTROL_IDS, KECCAK_PO2_RANGE,
};
use risc0_circuit_keccak_methods::{KECCAK_ELF, KECCAK_ID};
use risc0_zkp::{core::digest::Digest, digest};
use risc0_zkvm::{get_prover_server, ExecutorEnv, InnerAssumptionReceipt, ProverOpts};

/// The guest's inputs (`methods/guest/src/bin/keccak.rs` `generate_input`).
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

/// The host-side claim digest reproduces `tests/keccak.rs`'s constants.
#[test]
fn claim_digest_matches_known() {
    assert_eq!(
        claim_digest(16),
        digest!("b83c10da0c23587bf318cbcec2c2ac0260dbd6c0fa6905df639f8f6056f0d56c")
    );
    assert_eq!(
        claim_digest(17),
        digest!("a920de1111b7fa680c29f0dbd8eea64a3b9e2994de1b242e30f53fd8cd727ba2")
    );
}

#[test]
#[ignore]
fn keccak_e2e_po2_range() {
    let po2s: Vec<usize> = std::env::var("K5_E2E_PO2S")
        .unwrap_or_else(|_| "14,15,16,17,18".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    // With `--features metal` the keccak proofs must take the Metal path
    // (`risc0_circuit_keccak::prove::keccak_prover` picks it when `enabled("keccak")`), so a
    // silent fallback to the CPU prover fails here instead of passing as a Metal run.
    #[cfg(feature = "metal")]
    assert!(
        risc0_zkp::hal::metal::enabled("keccak"),
        "the Metal keccak prover is disabled (RISC0_METAL / RISC0_METAL_SKIP): not a Metal run"
    );
    let prover = get_prover_server(&ProverOpts::fast()).unwrap();
    for po2 in po2s {
        assert!(KECCAK_PO2_RANGE.contains(&po2), "po2 {po2}");
        let claim = claim_digest(po2);
        let env = ExecutorEnv::builder()
            .write(&(claim, po2 as u32))
            .unwrap()
            .keccak_max_po2(po2 as u32)
            .unwrap()
            .build()
            .unwrap();
        let t0 = std::time::Instant::now();
        let receipt = prover.prove(env, KECCAK_ELF).unwrap().receipt;
        let elapsed = t0.elapsed();

        let assumptions = &receipt.inner.composite().unwrap().assumption_receipts;
        assert_eq!(assumptions.len(), 1, "po2 {po2}");
        let InnerAssumptionReceipt::Succinct(lifted) = &assumptions[0] else {
            panic!("po2 {po2}: the keccak assumption is not a succinct receipt");
        };
        assert_eq!(
            lifted.control_id,
            KECCAK_CONTROL_IDS[po2 - KECCAK_PO2_RANGE.start()],
            "po2 {po2}: lift control ID"
        );
        receipt.verify(KECCAK_ID).unwrap();
        println!(
            "K5 e2e keccak po2 {po2}: {} keccaks, prove+lift+verify ok in {:.2} s",
            guest_inputs(po2).len(),
            elapsed.as_secs_f64()
        );
    }
}
