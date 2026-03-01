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

use std::{cell::RefCell, collections::HashMap};

use anyhow::{anyhow, bail, ensure, Context, Result};

use super::{keccak::prove_keccak, ProverServer};
use risc0_circuit_rv32im::prove::SegmentProver;
use risc0_core::field::{baby_bear::BabyBearExtElem as ExtVal, Elem as _};

thread_local! {
    // ManuallyDrop prevents TLS cleanup from dropping CUDA buffers after the CUDA
    // context is already destroyed (which causes a panic/abort on process exit).
    // GPU memory is reclaimed by CUDA context teardown regardless.
    static CACHED_SEGMENT_PROVER: std::mem::ManuallyDrop<RefCell<Option<Box<dyn SegmentProver>>>> =
        std::mem::ManuallyDrop::new(RefCell::new(None));
}

fn with_segment_prover<R>(f: impl FnOnce(&dyn SegmentProver) -> Result<R>) -> Result<R> {
    CACHED_SEGMENT_PROVER.with(|cell| {
        {
            let mut borrow = cell.borrow_mut();
            if borrow.is_none() {
                *borrow = Some(risc0_circuit_rv32im::prove::segment_prover()?);
            }
        }
        let borrow = cell.borrow();
        f(borrow.as_ref().unwrap().as_ref())
    })
}
use crate::{
    claim::merge::Merge,
    host::{
        client::prove::opts::ReceiptKind,
        prove_info::ProveInfo,
        recursion::{identity_p254, join, lift, resolve},
        server::{exec::executor::ExecutorImpl, prove::union_peak::UnionPeak},
    },
    mmr::MerkleMountainAccumulator,
    receipt::{InnerReceipt, SegmentReceipt, SuccinctReceipt},
    recursion::prove::{
        join_povw, join_unwrap_povw, lift_povw, resolve_povw, resolve_unwrap_povw, union,
        unwrap_povw,
    },
    sha::Digestible,
    Assumption, AssumptionReceipt, CompositeReceipt, ExecutorEnv, InnerAssumptionReceipt,
    MaybePruned, Output, PreflightResults, ProverOpts, Receipt, ReceiptClaim, Segment, Session,
    UnionClaim, Unknown, VerifierContext, WorkClaim,
};

/// An implementation of a Prover that runs locally.
pub struct ProverImpl {
    opts: ProverOpts,
}

impl ProverImpl {
    /// Construct a [ProverImpl].
    pub fn new(opts: ProverOpts) -> Self {
        Self { opts }
    }
}

impl ProverServer for ProverImpl {
    fn prove(&self, env: ExecutorEnv<'_>, elf: &[u8]) -> Result<ProveInfo> {
        let ctx = VerifierContext::default().with_dev_mode(self.opts.dev_mode());
        self.prove_with_ctx(env, &ctx, elf)
    }

    fn prove_with_ctx(
        &self,
        env: ExecutorEnv<'_>,
        ctx: &VerifierContext,
        elf: &[u8],
    ) -> Result<ProveInfo> {
        let session = ExecutorImpl::from_elf(env, elf)?.run()?;
        self.prove_session(ctx, &session)
    }

    fn prove_session(&self, ctx: &VerifierContext, session: &Session) -> Result<ProveInfo> {
        tracing::debug!(
            "prove_session: exit_code = {:?}, journal = {:?}, segments: {}",
            session.exit_code,
            session.journal.as_ref().map(hex::encode),
            session.segments.len()
        );

        ensure!(
            self.opts.hashfn == "poseidon2",
            "provided `ProverOpts` has unsupported `hashfn` value of \"{}\"; \
            supported `hashfn` values are: \"poseidon2\".",
            &self.opts.hashfn
        );

        // Trigger CUDA module loading in background while first segment is prepared.
        // With primary context, this is a one-time cost amortized across all segments.
        #[cfg(feature = "cuda")]
        let mut warmup_handle = Some(std::thread::spawn(|| {
            risc0_circuit_rv32im::prove::cuda_warmup();
        }));
        #[cfg(feature = "rocm")]
        let mut warmup_handle = Some(std::thread::spawn(|| {
            risc0_circuit_rv32im::prove::rocm_warmup();
        }));
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        let mut warmup_handle: Option<std::thread::JoinHandle<()>> = None;

        let skip_verify = std::env::var("RISC0_SKIP_VERIFY").is_ok();
        let max_po2 = self.opts.max_segment_po2;

        // Pipeline: overlap preflight(N+1) CPU work with prove_core(N) GPU work.
        // Preflight is ~275ms of pure CPU emulation; prove_core is ~443ms of GPU work.
        // By overlapping them, we hide preflight behind GPU time (saves ~37% per segment).
        let mut segments = Vec::new();
        let mut pending_preflight: Option<
            std::thread::JoinHandle<Result<(Segment, PreflightResults)>>,
        > = None;

        // Background receipt pipeline: decode(N) + verify(N) overlaps with prove(N+1).
        // After prove_core returns the raw seal, we immediately start prove_core(N+1)
        // while a background thread decodes the seal, builds the receipt, and verifies.
        // This saves ~5ms of decode time from the critical path per segment.
        let verify_params = ctx.segment_verifier_parameters.clone();
        // Precompute the verifier_parameters digest once (avoids per-segment SHA256).
        let seg_verifier_params_digest = ctx
            .segment_verifier_parameters
            .as_ref()
            .ok_or_else(|| anyhow!("segment receipt verifier parameters missing from context"))?
            .digest();
        let mut pending_receipt: Option<
            std::thread::JoinHandle<Result<SegmentReceipt>>,
        > = None;
        // Metadata from the previous iteration, paired with the seal returned by prove_begin.
        let mut prev_seal_meta: Option<(u32, u32, Option<Output>, String)> = None;

        for (seg_idx, segment_ref) in session.segments.iter().enumerate() {
            let t_seg = std::time::Instant::now();

            // Get segment + preflight results: from pipelined thread or compute here.
            let (segment, results) = if let Some(handle) = pending_preflight.take() {
                handle
                    .join()
                    .map_err(|_| anyhow!("preflight thread panicked"))??
            } else {
                // First segment: compute synchronously.
                let segment = segment_ref.resolve()?;
                let results = self.segment_preflight(&segment)?;
                (segment, results)
            };
            let t_preflight = t_seg.elapsed();

            for hook in &session.hooks {
                hook.on_pre_prove_segment(&segment);
            }

            // Join warmup thread before first segment's prove_core.
            if seg_idx == 0 {
                if let Some(handle) = warmup_handle.take() {
                    let _ = handle.join();
                }
            }

            // Start preflight for NEXT segment in background while GPU proves current.
            if seg_idx + 1 < session.segments.len() {
                let next_segment = session.segments[seg_idx + 1].resolve()?;
                pending_preflight = Some(std::thread::spawn(move || {
                    ensure!(
                        next_segment.po2() <= max_po2,
                        "segment po2 exceeds max: {} > {}",
                        next_segment.po2(),
                        max_po2
                    );
                    let rand_z = ExtVal::random(&mut rand::rng());
                    let inner = risc0_circuit_rv32im::prove::PreflightResults::new(
                        &next_segment.inner,
                        rand_z,
                    )?;
                    let pr = PreflightResults {
                        inner,
                        terminate_state: next_segment.inner.claim.terminate_state,
                        output: next_segment.output.clone(),
                        segment_index: next_segment.index,
                    };
                    Ok((next_segment, pr))
                }));
            }

            // Extract metadata before prove_begin consumes the inner preflight.
            let po2 = results.inner.po2();
            let segment_index = results.segment_index;
            let output = results.output;
            let hashfn = self.opts.hashfn.clone();

            // GPU prove_begin: main phase (overlapping with prev eval_check) +
            // complete prev deferred finalize + launch eval_check for this segment.
            // Returns the PREVIOUS segment's seal (None on first call).
            let t_psc = std::time::Instant::now();
            let prev_seal =
                with_segment_prover(|sp| sp.prove_begin(results.inner))?;
            let prove_begin_ms = t_psc.elapsed().as_secs_f64() * 1000.0;

            // If we got a previous segment's seal, process it.
            if let Some(seal) = prev_seal {
                // Collect the background receipt from the segment before that.
                if let Some(handle) = pending_receipt.take() {
                    let receipt = handle
                        .join()
                        .map_err(|_| anyhow!("receipt thread panicked"))??;
                    segments.push(receipt);
                }

                // Start background receipt for the PREVIOUS segment.
                let (prev_po2, prev_seg_idx, prev_output, prev_hashfn) =
                    prev_seal_meta.take().unwrap();
                let params = verify_params.clone();
                let vp_digest = seg_verifier_params_digest;
                let skip_verify_copy = skip_verify;
                pending_receipt = Some(std::thread::spawn(move || {
                    let mut claim =
                        ReceiptClaim::decode_from_seal_v2(&seal, Some(prev_po2))?;
                    claim.output = prev_output.into();
                    let receipt = SegmentReceipt {
                        seal,
                        index: prev_seg_idx,
                        hashfn: prev_hashfn,
                        claim,
                        verifier_parameters: vp_digest,
                    };
                    if !skip_verify_copy {
                        receipt
                            .verify_integrity_with_context(&VerifierContext {
                                segment_verifier_parameters: params,
                                ..VerifierContext::empty()
                            })
                            .context("verify segment")?;
                    }
                    Ok(receipt)
                }));
            }

            // Store current segment's metadata for next iteration.
            prev_seal_meta = Some((po2, segment_index, output, hashfn));

            let t_prove = t_seg.elapsed() - t_preflight;
            let t_total = t_seg.elapsed();
            eprintln!(
                "[prove_session] seg {seg_idx}: preflight={:.1}ms prove_begin={:.1}ms prove={:.1}ms total={:.1}ms",
                t_preflight.as_secs_f64() * 1000.0,
                prove_begin_ms,
                t_prove.as_secs_f64() * 1000.0,
                t_total.as_secs_f64() * 1000.0,
            );

            for hook in &session.hooks {
                hook.on_post_prove_segment(&segment);
            }
        }

        // Complete the last segment's deferred finalize.
        let last_seal = with_segment_prover(|sp| sp.prove_end())?;

        // Collect the second-to-last receipt (if any).
        if let Some(handle) = pending_receipt.take() {
            let receipt = handle
                .join()
                .map_err(|_| anyhow!("receipt thread panicked"))??;
            segments.push(receipt);
        }

        // Process the last segment's seal.
        {
            let (last_po2, last_seg_idx, last_output, last_hashfn) =
                prev_seal_meta.take().unwrap();
            let mut claim =
                ReceiptClaim::decode_from_seal_v2(&last_seal, Some(last_po2))?;
            claim.output = last_output.into();
            let receipt = SegmentReceipt {
                seal: last_seal,
                index: last_seg_idx,
                hashfn: last_hashfn,
                claim,
                verifier_parameters: seg_verifier_params_digest,
            };
            if !skip_verify {
                receipt
                    .verify_integrity_with_context(&VerifierContext {
                        segment_verifier_parameters: verify_params.clone(),
                        ..VerifierContext::empty()
                    })
                    .context("verify segment")?;
            }
            segments.push(receipt);
        }

        let (assumptions, session_assumption_receipts): (Vec<_>, Vec<_>) =
            session.assumptions.iter().cloned().unzip();

        // Merge the output, including journal digest and assumptions, into the last segment.
        segments
            .last_mut()
            .ok_or_else(|| anyhow!("session is empty"))?
            .claim
            .output
            .merge_with(
                &session
                    .journal
                    .as_ref()
                    .map(|journal| Output {
                        journal: MaybePruned::Pruned(journal.digest()),
                        assumptions: assumptions.into(),
                    })
                    .into(),
            )
            .context("failed to merge output into final segment claim")?;

        let verifier_parameters = ctx
            .composite_verifier_parameters()
            .ok_or_else(|| anyhow!("composite receipt verifier parameters missing from context"))?
            .digest();

        let mut zkr_receipts = HashMap::new();
        let mut keccak_receipts: MerkleMountainAccumulator<UnionPeak> =
            MerkleMountainAccumulator::new();
        for proof_request in session.pending_keccaks.iter() {
            let receipt = prove_keccak(proof_request)?;
            tracing::debug!("adding keccak assumption: {}", receipt.claim.digest());
            keccak_receipts.insert(receipt)?;
        }

        // NOTE: Calling keccak_receipts.root() proves the union tree.
        if let Ok(root_receipt) = keccak_receipts.root() {
            let assumption = Assumption {
                claim: root_receipt.claim.digest(),
                control_root: root_receipt.control_root()?,
            };

            tracing::debug!("keccak root assumption: {:?}", assumption);
            zkr_receipts.insert(assumption, root_receipt.clone());
        }

        // TODO: add test case for when a single session refers to the same assumption multiple times
        let inner_assumption_receipts: Vec<_> = session_assumption_receipts
            .into_iter()
            .map(|assumption_receipt| match assumption_receipt {
                AssumptionReceipt::Proven(receipt) => Ok(receipt),
                AssumptionReceipt::Unresolved(assumption) => {
                    let receipt = zkr_receipts.get(&assumption).ok_or_else(|| {
                        anyhow!("no receipt available for unresolved assumption: {assumption:#?}")
                    })?;
                    Ok(InnerAssumptionReceipt::Succinct(receipt.clone()))
                }
            })
            .collect::<Result<_>>()?;

        let composite_receipt = CompositeReceipt {
            segments,
            assumption_receipts: inner_assumption_receipts,
            verifier_parameters,
        };

        let session_claim = session.claim()?;

        // Verify the receipt to catch if something is broken in the proving process.
        // When background verification verified each segment's seal, skip the redundant
        // composite verify (which re-verifies all seals) and just check the claims match.
        if !skip_verify {
            check_claims(
                &session_claim,
                "composite",
                MaybePruned::Value(composite_receipt.claim()?),
            )?;
        }

        if self.opts.receipt_kind == ReceiptKind::Composite {
            let receipt = Receipt::new(
                InnerReceipt::Composite(composite_receipt),
                session.journal.clone().unwrap_or_default().bytes,
            );
            return Ok(ProveInfo {
                receipt,
                work_receipt: None,
                stats: session.stats(),
            });
        }

        let (succinct_receipt, work_receipt) = match session.povw_job_id.is_some() {
            true => {
                let work_receipt = self.composite_to_succinct_povw(&composite_receipt)?;
                let unwrapped = self.unwrap_povw(&work_receipt)?;
                (unwrapped, Some(work_receipt))
            }
            false => (self.composite_to_succinct(&composite_receipt)?, None),
        };

        if self.opts.receipt_kind == ReceiptKind::Succinct {
            let receipt = Receipt::new(
                InnerReceipt::Succinct(succinct_receipt),
                session.journal.clone().unwrap_or_default().bytes,
            );
            return Ok(ProveInfo {
                receipt,
                work_receipt: work_receipt.map(Into::into),
                stats: session.stats(),
            });
        }

        let groth16_receipt = self.succinct_to_groth16(&succinct_receipt)?;

        if self.opts.receipt_kind == ReceiptKind::Groth16 {
            let receipt = Receipt::new(
                InnerReceipt::Groth16(groth16_receipt),
                session.journal.clone().unwrap_or_default().bytes,
            );
            return Ok(ProveInfo {
                receipt,
                work_receipt: work_receipt.map(Into::into),
                stats: session.stats(),
            });
        }

        // As long as the checks above are exhaustive, this code is unreachable. If this statement
        // is reached, this is an implementation error.
        unreachable!(
            "proving not implemented for receipt kind {:?}",
            self.opts.receipt_kind
        );
    }

    fn segment_preflight(&self, segment: &Segment) -> Result<PreflightResults> {
        tracing::debug!("segment_preflight");

        ensure!(
            segment.po2() <= self.opts.max_segment_po2,
            "segment po2 exceeds max on ProverOpts: {} > {}",
            segment.po2(),
            self.opts.max_segment_po2
        );
        let inner = with_segment_prover(|sp| sp.preflight(&segment.inner))?;

        Ok(PreflightResults {
            inner,
            terminate_state: segment.inner.claim.terminate_state,
            output: segment.output.clone(),
            segment_index: segment.index,
        })
    }

    fn prove_segment_core(
        &self,
        ctx: &VerifierContext,
        preflight_results: PreflightResults,
    ) -> Result<SegmentReceipt> {
        tracing::debug!("prove_segment_core");

        ensure!(
            self.opts.hashfn == "poseidon2",
            "provided `ProverOpts` has unsupported `hashfn` value of \"{}\"; \
            supported `hashfn` values are: \"poseidon2\".",
            &self.opts.hashfn
        );

        let po2 = preflight_results.inner.po2();
        let t_psc = std::time::Instant::now();
        let seal =
            with_segment_prover(|sp| sp.prove_core(preflight_results.inner))?;
        eprintln!("[prove_segment_core] prove_core: {:.1}ms", t_psc.elapsed().as_secs_f64() * 1000.0);
        let t_dec = std::time::Instant::now();
        let mut claim = ReceiptClaim::decode_from_seal_v2(&seal, Some(po2))?;
        eprintln!("[prove_segment_core] decode: {:.1}ms (seal len: {})", t_dec.elapsed().as_secs_f64() * 1000.0, seal.len());
        claim.output = preflight_results.output.into();

        let verifier_parameters = ctx
            .segment_verifier_parameters
            .as_ref()
            .ok_or_else(|| anyhow!("segment receipt verifier parameters missing from context"))?
            .digest();
        let receipt = SegmentReceipt {
            seal,
            index: preflight_results.segment_index,
            hashfn: self.opts.hashfn.clone(),
            claim,
            verifier_parameters,
        };

        Ok(receipt)
    }

    fn lift(&self, receipt: &SegmentReceipt) -> Result<SuccinctReceipt<ReceiptClaim>> {
        let receipt = lift(receipt)?;
        receipt.verify_integrity().context("verify lift")?;
        Ok(receipt)
    }

    fn lift_povw(
        &self,
        receipt: &SegmentReceipt,
    ) -> Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>> {
        lift_povw(receipt)
    }

    fn join(
        &self,
        a: &SuccinctReceipt<ReceiptClaim>,
        b: &SuccinctReceipt<ReceiptClaim>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        let receipt = join(a, b)?;
        receipt.verify_integrity().context("verify join")?;
        Ok(receipt)
    }

    fn join_povw(
        &self,
        a: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        b: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
    ) -> Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>> {
        join_povw(a, b)
    }

    fn join_unwrap_povw(
        &self,
        a: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        b: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        join_unwrap_povw(a, b)
    }

    fn resolve(
        &self,
        conditional: &SuccinctReceipt<ReceiptClaim>,
        assumption: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        let receipt = resolve(conditional, assumption)?;
        receipt.verify_integrity().context("verify resolve")?;
        Ok(receipt)
    }

    fn resolve_povw(
        &self,
        conditional: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        assumption: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>> {
        resolve_povw(conditional, assumption)
    }

    fn resolve_unwrap_povw(
        &self,
        conditional: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        assumption: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        resolve_unwrap_povw(conditional, assumption)
    }

    fn identity_p254(
        &self,
        a: &SuccinctReceipt<ReceiptClaim>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        // TODO: figure out how to verify this
        identity_p254(a)
    }

    fn prove_keccak(
        &self,
        request: &crate::ProveKeccakRequest,
    ) -> Result<SuccinctReceipt<Unknown>> {
        // TODO: figure out how to verify this
        prove_keccak(request)
    }

    fn union(
        &self,
        a: &SuccinctReceipt<Unknown>,
        b: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<UnionClaim>> {
        let receipt = union(a, b)?;
        receipt.verify_integrity().context("verify union")?;
        Ok(receipt)
    }

    fn unwrap_povw(
        &self,
        a: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        unwrap_povw(a)
    }
}

fn check_claims(
    session_claim: &ReceiptClaim,
    other_name: &str,
    other_claim: MaybePruned<ReceiptClaim>,
) -> Result<()> {
    let session_claim_digest = session_claim.digest();
    let other_claim_digest = other_claim.digest();
    if session_claim_digest != other_claim_digest {
        tracing::debug!("session claim and {other_name} do not match");
        tracing::debug!("session claim: {session_claim:#?}");
        tracing::debug!("{other_name} claim: {other_claim:#?}");
        bail!(
            "session claim: {} != {other_name} claim: {}",
            hex::encode(session_claim_digest),
            hex::encode(other_claim_digest)
        );
    }
    Ok(())
}
