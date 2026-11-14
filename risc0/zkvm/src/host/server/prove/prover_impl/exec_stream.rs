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

//! Item Z17 (docs/metal-port/notes/Z17.md): prove the segments while the guest executes.
//!
//! Without this, `prove_with_ctx` executes the whole guest first (`ExecutorImpl::run`, which
//! writes every segment to a temporary file) and only then proves the segments. The executor is
//! CPU work with the GPU idle: ~0.85 s of the worker benchmark's 189 s (sha256-chain 0.25 s,
//! ecdsa-verify 0.38 s).
//!
//! Here the executor runs on the calling thread as before (`run_with_callback`, so every user
//! callback in the [ExecutorEnv] still runs on the thread that built it), and its segment thread
//! preflights each segment (overlapping the previous segment's proof, as the loop's preflight
//! thread does) and hands it to a proving thread that runs the rest of
//! [ProverImpl::prove_session_impl]'s loop: `prove_begin` (which returns the previous segment's
//! seal), and the previous segment's receipt decode and verification on its own thread.
//! After segment 0 the executor runs while the GPU proves, so its time is hidden. The segments
//! are no longer written to disk and read back, and at most a few of them are in host memory at
//! once (the executor's queue of 4 plus the one being proved).
//!
//! The proving thread uses the calling thread's cached segment prover (moved there and back, see
//! [SendProver]). It takes the GPU lock for each `prove_begin` only, not across its wait for the
//! next segment: the guest runs on the calling thread meanwhile, and a user callback there that
//! uses the GPU (e.g. an io callback that proves) would otherwise wait for a lock held by a thread
//! that waits for the executor (Z17 rev 1). The HAL takes the same lock around each of its
//! calls, so a GPU user that slips in between two segments only interleaves with the stream, as
//! a second session thread does. A caller that already holds the lock on its thread takes the
//! old path (see [eligible]). The calling thread takes the lock back for the last finalize and
//! the session's tail ([ProverImpl::prove_session_tail]: keccak proofs, composite, recursion,
//! wrap), which is shared with the loop, so the receipts are built the same way.
//!
//! [Stream::step] and [Stream::finish] mirror the per-segment body and the tail of the loop in
//! [ProverImpl::prove_session_impl] (prover_impl.rs) on Metal; a change to one must be made to
//! the other. Loop behaviour the stream leaves out, none of which applies to a streamed Metal
//! session: `session.hooks` (prove_with_ctx builds its own Session, which has none),
//! `release_pool_on_po2_growth` (CUDA only), `RISC0_TAIL_VERIFY=0` (the stream always verifies
//! the last two segments in parallel), and the `early_seal`/`rec_pipe` branches (CUDA/ROCm only).
//!
//! Kept on the old path (execute, then prove): `RISC0_METAL_EXEC_STREAM=0`, a CPU rv32im prover,
//! a calling thread that holds the GPU lock (`risc0_zkp::hal::metal::exclusive`),
//! and sessions whose [ExecutorEnv] has a coprocessor callback (keccak requests the callback
//! proves during execution, on the GPU, beside the segment prover), assumptions, trace
//! callbacks or a profiler, a segment path, or a PoVW job id. A session whose guest asks for keccak
//! proofs without a coprocessor streams its segments; its keccak proofs run after them, as before.

use std::sync::mpsc;

use super::*;
use crate::SegmentRef;

/// Revert knob: `0` executes the whole guest before proving (the pre-Z17 path).
pub(crate) const EXEC_STREAM_ENV: &str = "RISC0_METAL_EXEC_STREAM";

#[cfg(test)]
thread_local! {
    /// Test hook: overrides [EXEC_STREAM_ENV] on this thread (the process environment is shared
    /// with the tests running in parallel).
    pub(crate) static EXEC_STREAM_MODE: std::cell::RefCell<Option<&'static str>> =
        const { std::cell::RefCell::new(None) };
    /// Test hook: sessions this thread proved through the stream.
    pub(crate) static STREAMED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn knob() -> String {
    #[cfg(test)]
    if let Some(m) = EXEC_STREAM_MODE.with(|m| *m.borrow()) {
        return m.to_string();
    }
    std::env::var(EXEC_STREAM_ENV).unwrap_or_default()
}

/// Whether this session can stream: `Err` names why not (it then takes the old path).
pub(super) fn eligible(opts: &ProverOpts, env: &ExecutorEnv<'_>) -> Result<(), &'static str> {
    if knob().trim() == "0" {
        return Err("RISC0_METAL_EXEC_STREAM=0");
    }
    if !risc0_zkp::hal::metal::enabled("rv32im") {
        return Err("the rv32im segment prover is not the Metal one");
    }
    if risc0_zkp::hal::metal::singleton().is_owned_by_current_thread() {
        // The proving thread needs the lock for each segment while this thread, which holds
        // it, waits in the executor for that thread: a deadlock. The old path is reentrant.
        return Err("the calling thread holds the GPU lock");
    }
    if opts.hashfn != "poseidon2" {
        // The old path reports the error.
        return Err("unsupported hashfn");
    }
    if env.coprocessor.is_some() {
        return Err("a coprocessor callback");
    }
    if !env.assumptions.borrow().0.is_empty() {
        return Err("assumptions");
    }
    if !env.trace.is_empty() || env.pprof_out.is_some() {
        return Err("a trace callback or profiler");
    }
    if env.segment_path.is_some() {
        return Err("a segment path");
    }
    if env.povw_job_id.is_some() {
        return Err("a PoVW job");
    }
    Ok(())
}

/// What a streamed session keeps in `Session::segments`: the segment was proved as it was
/// executed and is not kept (nothing after the segment loop resolves a segment).
struct StreamedSegmentRef;

impl SegmentRef for StreamedSegmentRef {
    fn resolve(&self) -> Result<Segment> {
        bail!("this segment was proved while the guest executed (exec stream) and is not kept")
    }
}

/// The calling thread's cached segment prover, lent to the proving thread for one stream.
///
/// SAFETY: `Box<dyn SegmentProver>` is not `Send` because the trait has no `Send` bound, not
/// because the provers are tied to a thread. The prover this path uses is the Metal one
/// (`eligible` requires it): its state is `RefCell`s and `Arc`s it owns alone (no clone of them
/// leaves the prover), the `MetalHal`s it holds (Metal device, queue and buffer objects, which
/// Metal allows on any thread; objc objects are made inside `autoreleasepool`s), and no
/// thread-local state outside tests. Exactly one thread uses it at a time: the calling thread
/// moves it into the proving thread before the stream and takes it back after joining that
/// thread (`std::thread::scope`), so every access is ordered by the spawn and the join.
struct SendProver(Option<Box<dyn SegmentProver>>);

unsafe impl Send for SendProver {}

/// Take this thread's cached segment prover out of its slot (None when there is none yet).
fn take_cached_segment_prover() -> Option<Box<dyn SegmentProver>> {
    CACHED_SEGMENT_PROVER.with(|cell| cell.borrow_mut().take())
}

/// Put a prover back in this thread's (empty) slot.
fn restore_cached_segment_prover(prover: Option<Box<dyn SegmentProver>>) {
    if let Some(p) = prover {
        CACHED_SEGMENT_PROVER.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_none() {
                *slot = Some(p);
            }
        });
    }
}

type SealMeta = (u32, u32, Option<Output>, String);

/// A segment after its preflight (CPU only), ready for `prove_begin`.
struct Prepared {
    inner: risc0_circuit_rv32im::prove::PreflightResults,
    po2: u32,
    index: u32,
    output: Option<Output>,
    preflight_ms: f64,
}

/// The preflight, on the executor's segment thread: it overlaps the previous segment's
/// `prove_begin`, as the loop's preflight thread does. The segment is dropped after it (the
/// preflight holds what proving needs).
fn prepare(segment: Segment, max_po2: usize) -> Result<Prepared> {
    let t = std::time::Instant::now();
    ensure!(
        segment.po2() <= max_po2,
        "segment po2 exceeds max on ProverOpts: {} > {}",
        segment.po2(),
        max_po2
    );
    let rand_z = risc0_circuit_rv32im::prove::preflight_rand_z();
    let inner = risc0_circuit_rv32im::prove::PreflightResults::new(&segment.inner, rand_z)?;
    Ok(Prepared {
        po2: inner.po2(),
        inner,
        index: segment.index,
        output: segment.output,
        preflight_ms: t.elapsed().as_secs_f64() * 1000.0,
    })
}

/// The segment pipeline's state, owned by the proving thread during the stream and by the calling
/// thread before and after it.
struct Stream {
    prover: SendProver,
    hashfn: String,
    skip_verify: bool,
    verify_params: Option<crate::receipt::SegmentReceiptVerifierParameters>,
    vp_digest: risc0_zkp::core::digest::Digest,
    /// A `prove_begin` whose segment has not been finalized yet (its seal comes from the next
    /// `prove_begin` or from `prove_end`).
    begun: bool,
    /// Metadata of the segment whose seal the next `prove_begin`/`prove_end` returns.
    prev_meta: Option<SealMeta>,
    /// Decode + verify of the segment before that, on its own thread.
    pending_receipt: Option<std::thread::JoinHandle<Result<SegmentReceipt>>>,
    segments: Vec<SegmentReceipt>,
    t_start: std::time::Instant,
}

impl Stream {
    fn prover(&mut self) -> Result<&dyn SegmentProver> {
        if self.prover.0.is_none() {
            self.prover.0 = Some(risc0_circuit_rv32im::prove::segment_prover()?);
        }
        Ok(self.prover.0.as_deref().unwrap())
    }

    /// The proving thread: every segment the executor sends, in order. Takes the GPU lock per
    /// segment ([Stream::step]), never while it waits for the next one.
    fn run(&mut self, rx: mpsc::Receiver<Prepared>) -> Result<()> {
        for prepared in rx.iter() {
            self.step(prepared)?;
        }
        Ok(())
    }

    /// One iteration of the segment loop (see `prove_session_impl`), after its preflight.
    fn step(&mut self, prepared: Prepared) -> Result<()> {
        let t_seg = std::time::Instant::now();
        let Prepared {
            inner,
            po2,
            index,
            output,
            preflight_ms,
        } = prepared;

        let t_psc = std::time::Instant::now();
        let prev_seal = {
            let _gpu = super::super::exclusive_gpu();
            self.prover()?.prove_begin(inner)?
        };
        let prove_begin_ms = t_psc.elapsed().as_secs_f64() * 1000.0;
        let had_prev = std::mem::replace(&mut self.begun, true);
        let cur_meta = (po2, index, output, self.hashfn.clone());

        match (prev_seal, had_prev) {
            (Some(seal), true) => {
                if let Some(handle) = self.pending_receipt.take() {
                    self.segments.push(
                        handle
                            .join()
                            .map_err(|_| anyhow!("receipt thread panicked"))??,
                    );
                }
                let meta = self
                    .prev_meta
                    .take()
                    .ok_or_else(|| anyhow!("segment pipeline: seal without metadata"))?;
                self.pending_receipt = Some(self.spawn_receipt(seal, meta, false));
            }
            (None, false) => {}
            (Some(_), false) => bail!("segment pipeline: a seal from the first prove_begin"),
            (None, true) => bail!("segment pipeline: prove_begin returned no previous seal"),
        }
        self.prev_meta = Some(cur_meta);

        if *VERBOSE {
            eprintln!(
                "[prove_session] seg {index} (exec stream, at {:.1}ms): preflight={preflight_ms:.1}ms \
                 (on the executor's segment thread) prove_begin={prove_begin_ms:.1}ms total={:.1}ms",
                (t_seg - self.t_start).as_secs_f64() * 1000.0,
                t_seg.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(())
    }

    /// Decode `seal` into its receipt and verify it (unless RISC0_SKIP_VERIFY), on a thread.
    fn spawn_receipt(
        &self,
        seal: Vec<u32>,
        (po2, seg_idx, output, hashfn): SealMeta,
        last: bool,
    ) -> std::thread::JoinHandle<Result<SegmentReceipt>> {
        let params = self.verify_params.clone();
        let vp_digest = self.vp_digest;
        let skip_verify = self.skip_verify;
        std::thread::spawn(move || {
            let mut claim = ReceiptClaim::decode_from_seal_v2(&seal, Some(po2))?;
            claim.output = output.into();
            let receipt = SegmentReceipt {
                seal,
                index: seg_idx,
                hashfn,
                claim,
                verifier_parameters: vp_digest,
            };
            if !skip_verify {
                receipt
                    .verify_integrity_with_context(&VerifierContext {
                        segment_verifier_parameters: params,
                        ..VerifierContext::empty()
                    })
                    .with_context(|| {
                        if last {
                            format!("verify segment (last) {seg_idx}")
                        } else {
                            format!("verify segment {seg_idx}")
                        }
                    })?;
            }
            Ok(receipt)
        })
    }

    /// After a successful execution: finalize the last segment and collect every receipt. The
    /// last segment's verification overlaps the one before it, as in the loop's tail.
    fn finish(&mut self) -> Result<Vec<SegmentReceipt>> {
        ensure!(self.begun, "session is empty");
        let seal = self.prover()?.prove_end()?;
        self.begun = false;
        let meta = self
            .prev_meta
            .take()
            .ok_or_else(|| anyhow!("segment pipeline: seal without metadata"))?;
        let last = self.spawn_receipt(seal, meta, true);
        if let Some(handle) = self.pending_receipt.take() {
            self.segments.push(
                handle
                    .join()
                    .map_err(|_| anyhow!("receipt thread panicked"))??,
            );
        }
        self.segments.push(
            last.join()
                .map_err(|_| anyhow!("receipt thread panicked"))??,
        );
        Ok(std::mem::take(&mut self.segments))
    }

    /// After a failure: complete and discard the pending finalize, so the cached prover is idle
    /// for the next session (see `DrainSegmentProverOnExit`).
    fn drain(&mut self) {
        if std::mem::take(&mut self.begun) {
            if let Some(p) = self.prover.0.as_deref() {
                let _ = p.prove_end();
            }
        }
    }
}

impl ProverImpl {
    /// Execute `elf` and prove its segments as the executor produces them, then prove the rest of
    /// the session as [ProverServer::prove_session] does. Only for sessions [eligible] accepts.
    pub(super) fn prove_streamed(
        &self,
        ctx: &VerifierContext,
        env: ExecutorEnv<'_>,
        elf: &[u8],
    ) -> Result<ProveInfo> {
        #[cfg(test)]
        STREAMED.with(|n| n.set(n.get() + 1));

        // As in prove_session_impl: parse the Groth16 witness graph meanwhile (no GPU work; not
        // joined).
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
        if skip_verify {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    "RISC0_SKIP_VERIFY is set: the prover will not verify the segments, composite \
                     or succinct receipts it produces; only their claims are checked"
                )
            });
        }
        let vp_digest = ctx
            .segment_verifier_parameters
            .as_ref()
            .ok_or_else(|| anyhow!("segment receipt verifier parameters missing from context"))?
            .digest();

        // The cached prover may hold a seal a failed earlier session left in its pipeline.
        let prover = take_cached_segment_prover();
        if let Some(p) = prover.as_deref() {
            let _gpu = super::super::exclusive_gpu();
            p.reset_pipeline();
        }
        let mut stream = Stream {
            prover: SendProver(prover),
            hashfn: self.opts.hashfn.clone(),
            skip_verify,
            verify_params: ctx.segment_verifier_parameters.clone(),
            vp_digest,
            begun: false,
            prev_meta: None,
            pending_receipt: None,
            segments: Vec::new(),
            t_start: std::time::Instant::now(),
        };

        let t_exec = std::time::Instant::now();
        let mut exec = match ExecutorImpl::from_elf(env, elf) {
            Ok(exec) => exec,
            Err(e) => {
                restore_cached_segment_prover(stream.prover.0.take());
                return Err(e);
            }
        };
        // Rendezvous: the executor's segment thread preflights a segment, then hands it over
        // when the proving thread asks for the next one, so what is in host memory is the
        // executor's own queue, one preflight waiting, and the segment being proved (the loop
        // also holds one preflight ahead).
        let (tx, rx) = mpsc::sync_channel::<Prepared>(0);
        let max_po2 = self.opts.max_segment_po2;
        let mut panicked = false;
        let (session, streamed) = std::thread::scope(|scope| {
            let stream_ref = &mut stream;
            let worker = std::thread::Builder::new()
                .name("segment-stream".into())
                .spawn_scoped(scope, move || stream_ref.run(rx));
            let worker = match worker {
                Ok(w) => w,
                Err(e) => {
                    return (
                        Err(anyhow!("segment stream thread not started: {e}")),
                        Ok(()),
                    )
                }
            };
            let session = exec.run_with_callback(move |segment| {
                tx.send(prepare(segment, max_po2)?)
                    .map_err(|_| anyhow!("segment stream: the segment prover stopped"))?;
                Ok(Box::new(StreamedSegmentRef) as Box<dyn SegmentRef>)
            });
            let streamed = worker.join().unwrap_or_else(|_| {
                panicked = true;
                Err(anyhow!("segment stream thread panicked"))
            });
            (session, streamed)
        });
        drop(exec);
        let exec_ms = t_exec.elapsed().as_secs_f64() * 1000.0;

        // The session's GPU lock, for the last finalize and the tail (as prove_session_impl).
        let _gpu = super::super::exclusive_gpu();
        // A prover error first: the executor's error is then only "the segment prover stopped".
        let outcome = match (session, streamed) {
            (_, Err(e)) => Err(e),
            (Err(e), Ok(())) => Err(e),
            (Ok(session), Ok(())) => stream.finish().map(|segments| (session, segments)),
        };
        if panicked {
            // Not reused: a panic may have left it mid-segment.
            stream.prover.0 = None;
        } else if outcome.is_err() {
            stream.drain();
        }
        restore_cached_segment_prover(stream.prover.0.take());
        let (session, segments) = outcome?;
        if *VERBOSE {
            eprintln!(
                "[prove_session] exec stream: {} segments, executor + segments {exec_ms:.1}ms",
                segments.len()
            );
        }
        ensure!(
            segments.len() == session.segments.len(),
            "segment stream: {} receipts for {} segments",
            segments.len(),
            session.segments.len()
        );
        self.prove_session_tail(ctx, &session, segments, skip_verify, SessionTail::default())
    }
}
