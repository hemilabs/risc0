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

use std::{cell::RefCell, collections::HashMap, sync::LazyLock, sync::mpsc};

use anyhow::{anyhow, bail, ensure, Context, Result};

static VERBOSE: LazyLock<bool> = LazyLock::new(|| std::env::var("RISC0_VERBOSE").is_ok());

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
        recursion::{identity_p254, join, join_with_opts, lift, lift_with_opts, resolve},
        server::{exec::executor::ExecutorImpl, prove::union_peak::UnionPeak},
    },
    mmr::MerkleMountainAccumulator,
    receipt::{Groth16Receipt, Groth16ReceiptVerifierParameters, InnerReceipt, SegmentReceipt, SuccinctReceipt},
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

    /// Parallel STARK proving: both GPUs prove segments simultaneously, then recursion runs
    /// sequentially. This halves STARK time at the cost of sequential recursion.
    #[cfg(feature = "rocm")]
    fn prove_session_parallel_stark(
        &self,
        ctx: &VerifierContext,
        session: &Session,
    ) -> Result<ProveInfo> {
        let t_total = std::time::Instant::now();
        let max_po2 = self.opts.max_segment_po2;
        let num_segments = session.segments.len();

        // Preload circom graph in background.
        let _graph_preload = if self.opts.receipt_kind == ReceiptKind::Groth16 {
            Some(std::thread::spawn(|| {
                if let Err(e) = risc0_groth16::prove::preload_graph() {
                    eprintln!("[groth16] graph preload failed (non-fatal): {e}");
                }
            }))
        } else {
            None
        };

        // Warmup both GPUs sequentially (sppark init has shared static state).
        let warmup0 = std::thread::spawn(|| {
            risc0_zkp::hal::hip::switch_to_device(0);
            risc0_circuit_rv32im::prove::rocm_warmup();
        });
        let _ = warmup0.join();
        let warmup1 = std::thread::spawn(|| {
            risc0_zkp::hal::hip::switch_to_device(1);
            risc0_circuit_rv32im::prove::rocm_warmup();
        });
        let _ = warmup1.join();

        // Split segments between GPUs.
        // GPU1 (7900 XTX, gfx1100) is ~15% faster per segment than GPU0 (9070 XT, gfx1201)
        // due to higher memory BW and more CUs. Give GPU0 ~46% and GPU1 ~54%.
        let split = (num_segments * 46 + 50) / 100;
        eprintln!(
            "[parallel-stark] {} segments: GPU0={}, GPU1={}",
            num_segments, split, num_segments - split
        );

        let seg_verifier_params_digest = ctx
            .segment_verifier_parameters
            .as_ref()
            .ok_or_else(|| anyhow!("segment receipt verifier parameters missing from context"))?
            .digest();
        let hashfn = self.opts.hashfn.clone();

        // Resolve all segments upfront (needed for Send to threads).
        let all_segments: Vec<Segment> = session
            .segments
            .iter()
            .map(|s| s.resolve())
            .collect::<Result<_>>()?;

        // Thread function: prove a subset of segments on a specific GPU device.
        let prove_subset = |device: i32,
                            segments: Vec<Segment>,
                            vp_digest: crate::sha::Digest,
                            hashfn: String,
                            max_po2: usize|
         -> Result<Vec<SegmentReceipt>> {
            risc0_zkp::hal::hip::switch_to_device(device);

            // Each thread gets its own SegmentProver (thread-local).
            let sp = risc0_circuit_rv32im::prove::segment_prover()?;
            let mut receipts = Vec::with_capacity(segments.len());
            let mut prev_po2_stash: u32 = 0;

            for (local_idx, segment) in segments.iter().enumerate() {
                let t0 = std::time::Instant::now();
                ensure!(
                    segment.po2() <= max_po2,
                    "segment po2 exceeds max: {} > {}",
                    segment.po2(),
                    max_po2
                );
                let rand_z = ExtVal::random(&mut rand::rng());
                let preflight = risc0_circuit_rv32im::prove::PreflightResults::new(
                    &segment.inner,
                    rand_z,
                )?;
                let po2 = preflight.po2();

                // Use prove_begin/prove_end pipeline for eval_check overlap.
                let prev_seal = sp.prove_begin(preflight)?;

                // Process previous segment's seal (if any).
                if let Some(seal) = prev_seal {
                    let prev_seg = &segments[local_idx - 1];
                    let prev_po2 = prev_po2_stash;
                    let mut claim =
                        ReceiptClaim::decode_from_seal_v2(&seal, Some(prev_po2))?;
                    claim.output = prev_seg.output.clone().into();
                    receipts.push(SegmentReceipt {
                        seal,
                        index: prev_seg.index,
                        hashfn: hashfn.clone(),
                        claim,
                        verifier_parameters: vp_digest,
                    });
                }
                prev_po2_stash = po2;

                if *VERBOSE {
                    eprintln!(
                        "[parallel-stark] GPU{device} seg {}: {:.1}ms",
                        segment.index,
                        t0.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }

            // Complete last segment's deferred finalize.
            let last_seal = sp.prove_end()?;
            let last_seg = segments.last().unwrap();
            let mut claim =
                ReceiptClaim::decode_from_seal_v2(&last_seal, Some(prev_po2_stash))?;
            claim.output = last_seg.output.clone().into();
            receipts.push(SegmentReceipt {
                seal: last_seal,
                index: last_seg.index,
                hashfn: hashfn.clone(),
                claim,
                verifier_parameters: vp_digest,
            });

            Ok(receipts)
        };

        // Split segments and spawn two STARK threads.
        let segs0 = all_segments[..split].to_vec();
        let segs1 = all_segments[split..].to_vec();
        let vp0 = seg_verifier_params_digest;
        let vp1 = seg_verifier_params_digest;
        let hf0 = hashfn.clone();
        let hf1 = hashfn.clone();

        let t_stark = std::time::Instant::now();
        let handle0 = std::thread::Builder::new()
            .name("stark-gpu0".into())
            .spawn(move || prove_subset(0, segs0, vp0, hf0, max_po2))?;
        let handle1 = std::thread::Builder::new()
            .name("stark-gpu1".into())
            .spawn(move || prove_subset(1, segs1, vp1, hf1, max_po2))?;

        let receipts0 = handle0
            .join()
            .map_err(|_| anyhow!("STARK GPU0 thread panicked"))??;
        let receipts1 = handle1
            .join()
            .map_err(|_| anyhow!("STARK GPU1 thread panicked"))??;

        let stark_ms = t_stark.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[parallel-stark] STARK done: {:.1}ms ({} + {} receipts)",
            stark_ms,
            receipts0.len(),
            receipts1.len()
        );

        // Merge receipts in segment index order.
        let mut segments: Vec<SegmentReceipt> = Vec::with_capacity(num_segments);
        segments.extend(receipts0);
        segments.extend(receipts1);
        segments.sort_by_key(|r| r.index);

        // Merge output + assumptions into the last segment.
        let (assumptions, _session_assumption_receipts): (Vec<_>, Vec<_>) =
            session.assumptions.iter().cloned().unzip();
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
            )?;

        let verifier_parameters = ctx
            .composite_verifier_parameters()
            .ok_or_else(|| anyhow!("composite receipt verifier parameters missing from context"))?
            .digest();

        let composite_receipt = CompositeReceipt {
            segments,
            assumption_receipts: Vec::new(),
            verifier_parameters,
        };

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

        // Recursion: multi-GPU lift+join pipeline.
        // Lift(N+1) on device 0 overlaps with join(acc, lift(N)) on device 1.
        // Saves ~175ms per step (42 steps = ~7.3s).
        let t_rec = std::time::Instant::now();
        let lift_device = 0;
        let join_device = 1;

        let segments_ref = &composite_receipt.segments;
        let num_segs = segments_ref.len();
        let recursion_opts = if self.opts.max_segment_po2 > crate::receipt::DEFAULT_MAX_PO2 {
            ProverOpts::from_max_po2(self.opts.max_segment_po2)
                .with_receipt_kind(ReceiptKind::Succinct)
        } else {
            ProverOpts::succinct()
        };
        let _skip_verify = std::env::var("RISC0_SKIP_VERIFY").is_ok();

        // Multi-GPU recursion pipeline:
        // - Persistent lift thread on device 0 (preserves thread-local PROVER_CACHE
        //   across all lifts → cached ctrl PolyGroup reuse).
        // - Main thread does joins on device 1.
        // - Pipeline: lift(N+1) overlaps with join(N).
        //
        // IMPORTANT: The main thread must NEVER call lift_with_opts, because the
        // RecursionProverImpl caches its HipHal (which holds a per-device
        // ReentrantMutex lock) in thread-local storage. If the main thread cached
        // a device 0 prover, the persistent lift thread would deadlock trying to
        // acquire device 0's lock.

        // Channel-based persistent lift thread: receives segments, returns lifted receipts.
        let (lift_tx, lift_rx) = std::sync::mpsc::sync_channel::<(SegmentReceipt, ProverOpts)>(1);
        let (result_tx, result_rx) = std::sync::mpsc::sync_channel::<Result<SuccinctReceipt<ReceiptClaim>>>(1);

        let lift_thread = std::thread::Builder::new()
            .name("lift-gpu0".into())
            .spawn(move || {
                risc0_zkp::hal::hip::switch_to_device(lift_device);
                // Process lift requests until channel closes.
                while let Ok((seg, opts)) = lift_rx.recv() {
                    let result = lift_with_opts(&seg, opts);
                    if result_tx.send(result).is_err() {
                        break; // main thread dropped receiver
                    }
                }
            })?;

        // Send lift(0).
        lift_tx.send((segments_ref[0].clone(), recursion_opts.clone()))?;
        let mut accumulator = result_rx.recv()
            .map_err(|_| anyhow!("lift thread died"))??;
        if *VERBOSE { eprintln!("[parallel-recursion] step 0/{num_segs}: lift on device {lift_device}"); }

        // Send lift(1) ahead (will run while main thread is idle or doing join).
        if num_segs > 1 {
            lift_tx.send((segments_ref[1].clone(), recursion_opts.clone()))?;
        }

        // Pipelined loop: join(N) on device 1 overlaps with lift(N+1) on device 0.
        for step_idx in 1..num_segs {
            let t_step = std::time::Instant::now();

            // Wait for lift(step_idx) to complete.
            let lifted = result_rx.recv()
                .map_err(|_| anyhow!("lift thread died"))??;
            let lift_ms = t_step.elapsed().as_secs_f64() * 1000.0;

            // Send lift(step_idx+1) BEFORE running join.
            // This way lift(N+1) runs on device 0 while join(N) runs on device 1.
            if step_idx + 1 < num_segs {
                lift_tx.send((segments_ref[step_idx + 1].clone(), recursion_opts.clone()))?;
            }

            // Run join on device 1 (overlaps with lift(N+1) on device 0).
            let t_join = std::time::Instant::now();
            risc0_zkp::hal::hip::switch_to_device(join_device);
            let joined = join_with_opts(&accumulator, &lifted, recursion_opts.clone())?;
            let join_ms = t_join.elapsed().as_secs_f64() * 1000.0;

            if *VERBOSE {
                eprintln!(
                    "[parallel-recursion] step {step_idx}/{num_segs}: lift_wait={lift_ms:.1}ms join={join_ms:.1}ms total={:.1}ms",
                    t_step.elapsed().as_secs_f64() * 1000.0
                );
            }
            accumulator = joined;
        }

        // Shut down lift thread.
        drop(lift_tx);
        let _ = lift_thread.join();

        let succinct_receipt = accumulator;
        let rec_ms = t_rec.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[parallel-stark] recursion done: {:.1}ms (multi-GPU pipeline)", rec_ms);

        if self.opts.receipt_kind == ReceiptKind::Succinct {
            let receipt = Receipt::new(
                InnerReceipt::Succinct(succinct_receipt),
                session.journal.clone().unwrap_or_default().bytes,
            );
            return Ok(ProveInfo {
                receipt,
                work_receipt: None,
                stats: session.stats(),
            });
        }

        // Groth16 wrapping.
        // Use device 1 (7900 XTX) for Groth16 — full-rate INT32 makes MSM ~2.5x faster.
        risc0_zkp::hal::hip::switch_to_device(1);
        let groth16_receipt = self.succinct_to_groth16(&succinct_receipt)?;
        let receipt = Receipt::new(
            InnerReceipt::Groth16(groth16_receipt),
            session.journal.clone().unwrap_or_default().bytes,
        );

        eprintln!(
            "[parallel-stark] total: {:.1}ms",
            t_total.elapsed().as_secs_f64() * 1000.0
        );

        Ok(ProveInfo {
            receipt,
            work_receipt: None,
            stats: session.stats(),
        })
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

        // Check for parallel STARK mode: both GPUs prove segments simultaneously.
        #[cfg(feature = "rocm")]
        if std::env::var("RISC0_PARALLEL_STARK").is_ok()
            && risc0_zkp::hal::hip::device_count() >= 2
            && self.opts.receipt_kind != ReceiptKind::Composite
            && !session.povw_job_id.is_some()
            && session.assumptions.is_empty()
        {
            return self.prove_session_parallel_stark(ctx, session);
        }

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

        // Preload circom graph on background thread so the ~728ms file read + parse
        // overlaps with segment proving and recursion.  Only needed for Groth16.
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        let _graph_preload = if self.opts.receipt_kind == ReceiptKind::Groth16 {
            Some(std::thread::spawn(|| {
                if let Err(e) = risc0_groth16::prove::preload_graph() {
                    eprintln!("[groth16] graph preload failed (non-fatal): {e}");
                }
            }))
        } else {
            None
        };

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

        // Multi-GPU: overlap recursion on secondary device with STARK proving.
        // Send SegmentReceipts to a background thread that runs lift+join on device 1.
        #[cfg(feature = "rocm")]
        let multi_gpu_active = risc0_zkp::hal::hip::recursion_device().is_some()
            && self.opts.receipt_kind != ReceiptKind::Composite
            && !session.povw_job_id.is_some()
            && session.assumptions.is_empty();
        #[cfg(not(feature = "rocm"))]
        let multi_gpu_active = false;

        let (mut recursion_tx, recursion_worker) = if multi_gpu_active {
            let (tx, rx) = mpsc::channel::<SegmentReceipt>();
            let max_po2 = self.opts.max_segment_po2;
            let want_groth16 = self.opts.receipt_kind == ReceiptKind::Groth16;
            let handle = std::thread::Builder::new()
                .name("recursion-worker".into())
                .spawn(move || -> Result<(SuccinctReceipt<ReceiptClaim>, Option<SuccinctReceipt<ReceiptClaim>>)> {
                    #[cfg(feature = "rocm")]
                    {
                        let device = risc0_zkp::hal::hip::recursion_device().unwrap();
                        risc0_zkp::hal::hip::switch_to_device(device);
                    }

                    let recursion_opts =
                        if max_po2 > crate::receipt::DEFAULT_MAX_PO2 {
                            ProverOpts::from_max_po2(max_po2)
                                .with_receipt_kind(ReceiptKind::Succinct)
                        } else {
                            ProverOpts::succinct()
                        };

                    let mut accumulator: Option<SuccinctReceipt<ReceiptClaim>> = None;

                    for (idx, segment_receipt) in rx.iter().enumerate() {
                        let t0 = std::time::Instant::now();
                        let lifted =
                            lift_with_opts(&segment_receipt, recursion_opts.clone())?;
                        let lift_ms = t0.elapsed().as_secs_f64() * 1000.0;

                        accumulator = match accumulator.take() {
                            Some(left) => {
                                let t1 = std::time::Instant::now();
                                let joined = join_with_opts(
                                    &left,
                                    &lifted,
                                    recursion_opts.clone(),
                                )?;
                                let join_ms = t1.elapsed().as_secs_f64() * 1000.0;
                                if *VERBOSE {
                                    eprintln!(
                                        "[multi-gpu recursion] step {idx}: \
                                         lift={lift_ms:.1}ms join={join_ms:.1}ms"
                                    );
                                }
                                Some(joined)
                            }
                            None => {
                                if *VERBOSE {
                                    eprintln!(
                                        "[multi-gpu recursion] step {idx}: \
                                         lift={lift_ms:.1}ms (first)"
                                    );
                                }
                                Some(lifted)
                            }
                        };
                    }

                    let succinct = accumulator.ok_or_else(|| {
                        anyhow!("no segments received by recursion worker")
                    })?;

                    // Run identity_p254 here to overlap with remaining STARK segments.
                    let ident = if want_groth16 {
                        let t0 = std::time::Instant::now();
                        let r = identity_p254(&succinct)?;
                        if *VERBOSE {
                            eprintln!(
                                "[multi-gpu recursion] identity_p254={:.1}ms",
                                t0.elapsed().as_secs_f64() * 1000.0
                            );
                        }
                        Some(r)
                    } else {
                        None
                    };

                    Ok((succinct, ident))
                })?;
            (Some(tx), Some(handle))
        } else {
            (None, None)
        };
        #[cfg(feature = "rocm")]
        if multi_gpu_active {
            eprintln!(
                "[multi-gpu] recursion worker started on device {}",
                risc0_zkp::hal::hip::recursion_device().unwrap_or(0)
            );
        }

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
                    // Multi-GPU: send receipt to recursion worker for parallel lift+join
                    if let Some(ref tx) = recursion_tx {
                        let _ = tx.send(receipt.clone());
                    }
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
            if *VERBOSE { eprintln!(
                "[prove_session] seg {seg_idx}: preflight={:.1}ms prove_begin={:.1}ms prove={:.1}ms total={:.1}ms",
                t_preflight.as_secs_f64() * 1000.0,
                prove_begin_ms,
                t_prove.as_secs_f64() * 1000.0,
                t_total.as_secs_f64() * 1000.0,
            ); }

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
            // Multi-GPU: send to recursion worker
            if let Some(ref tx) = recursion_tx {
                let _ = tx.send(receipt.clone());
            }
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

        // Multi-GPU: send last receipt (with merged output) to recursion worker.
        if let Some(tx) = recursion_tx.take() {
            let _ = tx.send(segments.last().unwrap().clone());
            // Channel closed on drop → worker finishes after processing this receipt.
        }

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

        let (succinct_receipt, work_receipt, pre_ident) = if let Some(handle) = recursion_worker {
            // Multi-GPU: collect the SuccinctReceipt from the background recursion worker.
            // Assumptions already checked to be empty at multi_gpu_active decision.
            let t_wait = std::time::Instant::now();
            let (succinct, ident) = handle
                .join()
                .map_err(|_| anyhow!("recursion worker thread panicked"))??;
            let wait_ms = t_wait.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[multi-gpu] recursion worker finished (wait={wait_ms:.1}ms)");
            (succinct, None, ident)
        } else {
            match session.povw_job_id.is_some() {
                true => {
                    let work_receipt =
                        self.composite_to_succinct_povw(&composite_receipt)?;
                    let unwrapped = self.unwrap_povw(&work_receipt)?;
                    (unwrapped, Some(work_receipt), None)
                }
                false => (self.composite_to_succinct(&composite_receipt)?, None, None),
            }
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

        let groth16_receipt = if let Some(ident_receipt) = pre_ident {
            // identity_p254 already computed by recursion worker (overlapped with STARK).
            use risc0_groth16::prove::shrink_wrap;
            let t0 = std::time::Instant::now();
            let seal_bytes = ident_receipt.get_seal_bytes();
            let seal = shrink_wrap(&seal_bytes)?.to_vec();
            let wrap_ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[succinct_to_groth16] identity_p254=pre-computed shrink_wrap={wrap_ms:.1}ms \
                 total={wrap_ms:.1}ms"
            );
            Groth16Receipt {
                seal,
                claim: succinct_receipt.claim.clone(),
                verifier_parameters: Groth16ReceiptVerifierParameters::default().digest(),
            }
        } else {
            self.succinct_to_groth16(&succinct_receipt)?
        };

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
        if *VERBOSE { eprintln!("[prove_segment_core] prove_core: {:.1}ms", t_psc.elapsed().as_secs_f64() * 1000.0); }
        let t_dec = std::time::Instant::now();
        let mut claim = ReceiptClaim::decode_from_seal_v2(&seal, Some(po2))?;
        if *VERBOSE { eprintln!("[prove_segment_core] decode: {:.1}ms (seal len: {})", t_dec.elapsed().as_secs_f64() * 1000.0, seal.len()); }
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
        let t0 = std::time::Instant::now();
        let receipt = if self.opts.max_segment_po2 > crate::receipt::DEFAULT_MAX_PO2 {
            let recursion_opts = ProverOpts::from_max_po2(self.opts.max_segment_po2)
                .with_receipt_kind(ReceiptKind::Succinct);
            lift_with_opts(receipt, recursion_opts)?
        } else {
            lift(receipt)?
        };
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if std::env::var("RISC0_SKIP_VERIFY").is_err() {
            let t1 = std::time::Instant::now();
            receipt.verify_integrity().context("verify lift")?;
            if *VERBOSE { eprintln!("[lift] prove={prove_ms:.1}ms verify={:.1}ms", t1.elapsed().as_secs_f64() * 1000.0); }
        } else {
            if *VERBOSE { eprintln!("[lift] prove={prove_ms:.1}ms verify=skipped"); }
        }
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
        let t0 = std::time::Instant::now();
        let receipt = if self.opts.max_segment_po2 > crate::receipt::DEFAULT_MAX_PO2 {
            let recursion_opts = ProverOpts::from_max_po2(self.opts.max_segment_po2)
                .with_receipt_kind(ReceiptKind::Succinct);
            join_with_opts(a, b, recursion_opts)?
        } else {
            join(a, b)?
        };
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if std::env::var("RISC0_SKIP_VERIFY").is_err() {
            let t1 = std::time::Instant::now();
            receipt.verify_integrity().context("verify join")?;
            if *VERBOSE { eprintln!("[join] prove={prove_ms:.1}ms verify={:.1}ms", t1.elapsed().as_secs_f64() * 1000.0); }
        } else {
            if *VERBOSE { eprintln!("[join] prove={prove_ms:.1}ms verify=skipped"); }
        }
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
        let t0 = std::time::Instant::now();
        let receipt = resolve(conditional, assumption)?;
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if std::env::var("RISC0_SKIP_VERIFY").is_err() {
            let t1 = std::time::Instant::now();
            receipt.verify_integrity().context("verify resolve")?;
            if *VERBOSE { eprintln!("[resolve] prove={prove_ms:.1}ms verify={:.1}ms", t1.elapsed().as_secs_f64() * 1000.0); }
        } else {
            if *VERBOSE { eprintln!("[resolve] prove={prove_ms:.1}ms verify=skipped"); }
        }
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
        let t0 = std::time::Instant::now();
        let receipt = union(a, b)?;
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if std::env::var("RISC0_SKIP_VERIFY").is_err() {
            let t1 = std::time::Instant::now();
            receipt.verify_integrity().context("verify union")?;
            if *VERBOSE { eprintln!("[union] prove={prove_ms:.1}ms verify={:.1}ms", t1.elapsed().as_secs_f64() * 1000.0); }
        } else {
            if *VERBOSE { eprintln!("[union] prove={prove_ms:.1}ms verify=skipped"); }
        }
        Ok(receipt)
    }

    fn unwrap_povw(
        &self,
        a: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        unwrap_povw(a)
    }

    /// Tree-based composite_to_succinct: batch-preflights lifts/joins with rayon,
    /// then executes GPU proofs sequentially. Saves ~600ms vs left-fold by eliminating
    /// sequential join preflight overhead.
    fn composite_to_succinct(
        &self,
        composite_receipt: &CompositeReceipt,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        use crate::{
            claim::merge::Merge,
            host::recursion::prove::{
                make_succinct_receipt, Prover as RecursionProverJob,
            },
        };
        use rayon::prelude::*;

        let t_pipeline = std::time::Instant::now();
        let segments = &composite_receipt.segments;
        let num_segments = segments.len();
        eprintln!(
            "[composite_to_succinct] starting: {} segments, {} assumptions",
            num_segments,
            composite_receipt.assumption_receipts.len()
        );

        let recursion_opts = if self.opts.max_segment_po2 > crate::receipt::DEFAULT_MAX_PO2 {
            ProverOpts::from_max_po2(self.opts.max_segment_po2)
                .with_receipt_kind(ReceiptKind::Succinct)
        } else {
            ProverOpts::succinct()
        };

        // Phase 1: Batch prepare all lift preflights in parallel (CPU-only).
        let mut prepared_lifts: Vec<RecursionProverJob> = segments
            .par_iter()
            .map(|seg| -> Result<RecursionProverJob> {
                let mut prover =
                    RecursionProverJob::new_lift(seg, recursion_opts.clone())?;
                prover.prepare()?;
                Ok(prover)
            })
            .collect::<Result<Vec<_>>>()?;

        // Phase 2: Execute lifts sequentially on GPU with pre-computed preflights.
        let mut receipts: Vec<SuccinctReceipt<ReceiptClaim>> =
            Vec::with_capacity(num_segments);
        for (seg, mut prover) in segments.iter().zip(prepared_lifts.drain(..)) {
            let receipt = prover.run()?;
            let claim_decoded = ReceiptClaim::decode(&mut receipt.out_stream())?;
            let claim = claim_decoded.merge(&seg.claim)?;
            receipts.push(make_succinct_receipt(prover, receipt, claim)?);
        }

        // Phase 3: Tree reduction via pairwise joins with batch preflighting.
        let mut level = 0;
        while receipts.len() > 1 {
            let num_pairs = receipts.len() / 2;
            let has_odd = receipts.len() % 2 == 1;

            // Batch prepare join preflights for all pairs (rayon).
            let prepared_joins = {
                let chunks: Vec<_> = receipts
                    .chunks(2)
                    .filter(|c| c.len() == 2)
                    .collect();
                chunks
                    .par_iter()
                    .map(|c| -> Result<RecursionProverJob> {
                        let mut prover = RecursionProverJob::new_join(
                            &c[0],
                            &c[1],
                            recursion_opts.clone(),
                        )?;
                        prover.prepare()?;
                        Ok(prover)
                    })
                    .collect::<Result<Vec<_>>>()?
            };

            // Execute joins sequentially on GPU.
            let mut next_level = Vec::with_capacity(num_pairs + if has_odd { 1 } else { 0 });
            let mut joiner_iter = prepared_joins.into_iter();
            let mut receipt_iter = receipts.into_iter();

            while let Some(left) = receipt_iter.next() {
                if let Some(right) = receipt_iter.next() {
                    let mut prover = joiner_iter.next().unwrap();
                    let receipt = prover.run()?;
                    let claim_decoded =
                        ReceiptClaim::decode(&mut receipt.out_stream())?;
                    let claim = claim_decoded
                        .merge(&left.claim.join(&right.claim)?.value()?)?;
                    next_level
                        .push(make_succinct_receipt(prover, receipt, claim)?);
                } else {
                    // Odd receipt carries at END to preserve segment ordering.
                    next_level.push(left);
                }
            }

            if *VERBOSE {
                eprintln!(
                    "[composite_to_succinct] level {level}: {num_pairs} joins{}",
                    if has_odd { " + 1 carry" } else { "" }
                );
            }
            level += 1;
            receipts = next_level;
        }

        eprintln!(
            "[composite_to_succinct] lift/join done: {:.1}s",
            t_pipeline.elapsed().as_secs_f64()
        );

        let continuation_receipt = receipts.into_iter().next().ok_or_else(|| {
            anyhow!("malformed composite receipt has no continuation segment receipts")
        })?;

        // Compress assumptions and resolve them (same as generic implementation).
        let result = composite_receipt.assumption_receipts.iter().try_fold(
            continuation_receipt,
            |conditional, assumption| match assumption {
                InnerAssumptionReceipt::Succinct(assumption) => {
                    self.resolve(&conditional, assumption)
                }
                InnerAssumptionReceipt::Composite(assumption) => {
                    self.resolve(
                        &conditional,
                        &SuccinctReceipt::<ReceiptClaim>::into_unknown(
                            <Self as super::Compress<_>>::composite_to_succinct(self, assumption)?,
                        ),
                    )
                }
                InnerAssumptionReceipt::Fake(_) => bail!(
                    "compressing composite receipts with fake receipt assumptions is not supported"
                ),
                InnerAssumptionReceipt::Groth16(_) => bail!(
                    "compressing composite receipts with Groth16 receipt assumptions is not supported"
                ),
            },
        )?;

        eprintln!(
            "[composite_to_succinct] total: {:.1}s",
            t_pipeline.elapsed().as_secs_f64()
        );
        Ok(result)
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
