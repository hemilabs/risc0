// Copyright 2025 RISC Zero, Inc.
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

use std::collections::VecDeque;

use anyhow::{ensure, Context, Result};
use risc0_binfmt::read_sha_halfs;
use risc0_circuit_keccak::{
    get_control_id,
    prove::{keccak_prover, verify_seal, zkr::get_keccak_zkr_shared, HostWitness, KeccakProver},
    KECCAK_CONTROL_IDS,
};
use risc0_core::field::baby_bear::BabyBearElem;
use risc0_zkp::core::digest::{Digest, DIGEST_SHORTS};

use crate::{
    host::client::env::ProveKeccakRequest, host::recursion::prove::prove_shared_zkr,
    receipt::SuccinctReceipt, Unknown,
};

/// Generate a keccak proof that has been lifted.
pub(crate) fn prove_keccak(request: &ProveKeccakRequest) -> Result<SuccinctReceipt<Unknown>> {
    prove_keccak_with(request, None)
}

/// `RISC0_KECCAK_REUSE_HAL=0` restores the pre-K10 session loop: a fresh keccak prover (and Metal
/// HAL, +95-107 ms each) per request, each proof lifted before the next one is proved, and its
/// seal verified inline. Unset (or anything else): [prove_and_lift_keccaks].
pub(crate) const KECCAK_REUSE_HAL_ENV: &str = "RISC0_KECCAK_REUSE_HAL";

/// `RISC0_KECCAK_VERIFY_OVERLAP=1` verifies each keccak seal on a thread beside the next proof in
/// [prove_and_lift_keccaks]; unset (or anything else), inline on the proving thread. Off by default:
/// on a 3-request session it saved 30 ms of ~6.06 s, inside the A-vs-A spread (notes/K10.md).
pub(crate) const KECCAK_VERIFY_OVERLAP_ENV: &str = "RISC0_KECCAK_VERIFY_OVERLAP";

fn env_disabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v.trim() == "0")
}

/// Whether the session loop takes [prove_and_lift_keccaks] (see [KECCAK_REUSE_HAL_ENV]).
pub(crate) fn reuse_keccak_prover() -> bool {
    !env_disabled(KECCAK_REUSE_HAL_ENV)
}

/// [prove_keccak], from the request's witness if `risc0_circuit_keccak::prove::host_witness` has
/// made it already.
pub(crate) fn prove_keccak_with(
    request: &ProveKeccakRequest,
    witness: Option<HostWitness>,
) -> Result<SuccinctReceipt<Unknown>> {
    let _gpu = super::exclusive_gpu();
    let t_prove = std::time::Instant::now();
    let seal = {
        let prover = keccak_prover()?;
        let seal = prove_seal(prover.as_ref(), request, witness)?;
        // Make sure we have a valid seal so we can fail early if anything went wrong
        verify_seal(&seal)?;
        seal
    };
    let prove_ms = t_prove.elapsed().as_secs_f64() * 1000.0;
    lift_seal(request, &seal, prove_ms)
}

/// Prove every request of a session (K10, LEADS #53 and #57): one keccak prover (one HAL, whose
/// buffer pool the proofs share) proves all the seals; it is dropped before the first lift, so no
/// lift runs beside its pool (before K10 each request built a fresh HAL, and the recursion prover
/// the previous lift cached stayed alive beside the next keccak proof). Each seal is verified on
/// the CPU while the GPU proves the next one if [KECCAK_VERIFY_OVERLAP_ENV] is `1`; at most
/// one verifier thread runs at a time (the previous one is joined before the next is started), and
/// every seal is verified before any lift. `on_receipt` sees each lifted receipt in request order
/// (with its index), e.g. to insert it into the union tree.
pub(crate) fn prove_and_lift_keccaks(
    requests: &[ProveKeccakRequest],
    mut on_receipt: impl FnMut(usize, SuccinctReceipt<Unknown>) -> Result<()>,
) -> Result<()> {
    if requests.is_empty() {
        return Ok(());
    }
    let overlap = std::env::var(KECCAK_VERIFY_OVERLAP_ENV).is_ok_and(|v| v.trim() == "1");
    let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
    // A detached (not scoped) verifier: if a proof fails while one runs, the error is returned and
    // the verifier finishes on its own instead of turning the error into a panic at scope exit.
    type Pending = (usize, std::thread::JoinHandle<Result<()>>);
    let join = |(i, handle): Pending| -> Result<()> {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("keccak request {i}: seal verification panicked"))?
            .with_context(|| format!("keccak request {i}"))
    };
    let mut seals = Vec::with_capacity(requests.len());
    {
        let _gpu = super::exclusive_gpu();
        let t_hal = std::time::Instant::now();
        let prover = keccak_prover()?;
        if verbose {
            eprintln!(
                "[keccak] prover for {} requests in {:.1}ms",
                requests.len(),
                t_hal.elapsed().as_secs_f64() * 1000.0
            );
        }
        let mut pending: Option<Pending> = None;
        for (i, request) in requests.iter().enumerate() {
            let t_prove = std::time::Instant::now();
            let seal = prove_seal(prover.as_ref(), request, None)
                .with_context(|| format!("keccak request {i}"))?;
            let prove_ms = t_prove.elapsed().as_secs_f64() * 1000.0;
            if verbose {
                eprintln!("[keccak] {i}: po2={} prove={prove_ms:.1}ms", request.po2);
            }
            if let Some(previous) = pending.take() {
                join(previous)?;
            }
            let seal = std::sync::Arc::new(seal);
            if overlap {
                let seal = seal.clone();
                pending = Some((i, std::thread::spawn(move || verify_seal(&seal))));
            } else {
                verify_seal(&seal).with_context(|| format!("keccak request {i}"))?;
            }
            seals.push((seal, prove_ms));
        }
        if let Some(last) = pending.take() {
            join(last)?;
        }
        // The prover (its HAL and pool) drops here, before any lift.
    }
    for (i, (request, (seal, prove_ms))) in requests.iter().zip(seals).enumerate() {
        let receipt = lift_seal(request, &seal, prove_ms)?;
        on_receipt(i, receipt)?;
    }
    Ok(())
}

/// The keccak seal of `request` (from `witness` if the prover takes a host witness), with its
/// claim digest checked against the request's.
fn prove_seal(
    prover: &dyn KeccakProver,
    request: &ProveKeccakRequest,
    witness: Option<HostWitness>,
) -> Result<Vec<u32>> {
    let seal = match witness.and_then(|witness| prover.prove_host_witness(witness)) {
        Some(seal) => seal?,
        None => prover.prove(&request.input, request.po2)?,
    };

    let claim_digest: Digest = read_sha_halfs(&mut VecDeque::from_iter(
        bytemuck::checked::cast_slice::<_, BabyBearElem>(&seal[0..DIGEST_SHORTS])
            .iter()
            .copied()
            .map(u32::from),
    ))?;

    ensure!(
        request.claim_digest == claim_digest,
        "keccak claim digest mismatch, expected: {:?}, actual: {claim_digest:?}",
        request.claim_digest
    );
    Ok(seal)
}

/// Lift a verified keccak seal of `request` with the keccak zkr of its po2.
fn lift_seal(
    request: &ProveKeccakRequest,
    seal: &[u32],
    prove_ms: f64,
) -> Result<SuccinctReceipt<Unknown>> {
    let claim_sha_input = request
        .claim_digest
        .as_words()
        .iter()
        .copied()
        .flat_map(|x| [x & 0xffff, x >> 16])
        .map(BabyBearElem::new)
        .collect::<Vec<_>>();

    let mut zkr_input: Vec<u32> = Vec::new();
    zkr_input.extend(request.control_root.as_words());
    zkr_input.extend(seal);
    zkr_input.extend(bytemuck::cast_slice(claim_sha_input.as_slice()));
    zkr_input.push(1 << request.po2);

    let t_lift = std::time::Instant::now();
    let receipt = prove_shared_zkr(
        get_keccak_zkr_shared(request.po2)?,
        get_control_id(request.po2),
        KECCAK_CONTROL_IDS.to_vec(),
        bytemuck::cast_slice(zkr_input.as_slice()),
    );
    if std::env::var_os("RISC0_VERBOSE").is_some() {
        eprintln!(
            "[keccak] prove={prove_ms:.1}ms lift={:.1}ms",
            t_lift.elapsed().as_secs_f64() * 1000.0
        );
    }
    receipt
}
