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

//! Item Z17 (docs/metal-port/notes/Z17.md): the executor stream (`prover_impl::exec_stream`).
//!
//! Every test picks the path with the thread-local override of `RISC0_METAL_EXEC_STREAM`, so the
//! tests beside it are not affected, and checks which path ran with the thread's stream counter.
//! With the `witgen_debug` feature (`rand_z` = 1, so a segment's seal is a function of the
//! segment), `z17_stream_matches_loop` also requires the two paths' seals to be equal word for
//! word: `cargo test -p risc0-zkvm --features metal,witgen_debug --release --lib -- tests_z17`.

use risc0_zkvm_methods::{multi_test::MultiTestSpec, MULTI_TEST_ELF, MULTI_TEST_ID};

use super::prover_impl::exec_stream::{EXEC_STREAM_MODE, STREAMED};
use crate::{get_prover_server, sha::Digestible, ExecutorEnv, InnerReceipt, ProverOpts, Receipt};

fn streamed() -> u64 {
    STREAMED.with(|n| n.get())
}

fn with_mode<R>(mode: Option<&'static str>, f: impl FnOnce() -> R) -> R {
    EXEC_STREAM_MODE.with(|m| *m.borrow_mut() = mode);
    let r = f();
    EXEC_STREAM_MODE.with(|m| *m.borrow_mut() = None);
    r
}

/// Seeded, so two executions of it are the same (the guest's `sys_rand` reads otherwise differ).
fn busy_env(cycles: u64, po2: u32) -> ExecutorEnv<'static> {
    ExecutorEnv::builder()
        .write(&MultiTestSpec::BusyLoop { cycles })
        .unwrap()
        .random_seed([7; 32])
        .segment_limit_po2(po2)
        .build()
        .unwrap()
}

/// Prove with `mode`; returns the receipt and whether the stream proved it.
fn prove(mode: Option<&'static str>, opts: &ProverOpts, env: ExecutorEnv<'_>) -> (Receipt, bool) {
    let before = streamed();
    let receipt = with_mode(mode, || {
        get_prover_server(opts)
            .unwrap()
            .prove(env, MULTI_TEST_ELF)
            .unwrap()
            .receipt
    });
    (receipt, streamed() > before)
}

/// The stream and the loop give the same composite receipt: the same segments (count, indexes,
/// claims, and under `witgen_debug` the same seals), journal and claim; both verify.
#[test]
fn z17_stream_matches_loop() {
    let opts = ProverOpts::composite();
    let (old, s0) = prove(Some("0"), &opts, busy_env(1 << 18, 16));
    assert!(!s0, "RISC0_METAL_EXEC_STREAM=0 streamed");
    let (new, s1) = prove(None, &opts, busy_env(1 << 18, 16));
    assert!(s1, "the default did not stream");
    old.verify(MULTI_TEST_ID).unwrap();
    new.verify(MULTI_TEST_ID).unwrap();
    let (a, b) = (
        old.inner.composite().unwrap(),
        new.inner.composite().unwrap(),
    );
    assert!(a.segments.len() >= 4, "{} segments", a.segments.len());
    assert_eq!(a.segments.len(), b.segments.len());
    for (x, y) in a.segments.iter().zip(&b.segments) {
        assert_eq!(x.index, y.index);
        assert_eq!(
            x.claim.digest(),
            y.claim.digest(),
            "segment {}: {:#?} vs {:#?}",
            x.index,
            x.claim,
            y.claim
        );
        assert_eq!(x.hashfn, y.hashfn);
        assert_eq!(x.verifier_parameters, y.verifier_parameters);
        assert_eq!(x.seal.len(), y.seal.len());
        #[cfg(feature = "witgen_debug")]
        assert!(x.seal == y.seal, "segment {}: the seals differ", x.index);
    }
    assert_eq!(old.journal.bytes, new.journal.bytes);
    assert_eq!(old.claim().unwrap().digest(), new.claim().unwrap().digest());
    assert_eq!(a.claim().unwrap().digest(), b.claim().unwrap().digest());
}

/// A streamed succinct proof (the recursion runs on the calling thread after the stream, with
/// the segment prover back in its cache) verifies, twice on one thread.
#[test]
fn z17_stream_succinct_twice() {
    let opts = ProverOpts::succinct();
    for _ in 0..2 {
        let (receipt, s) = prove(None, &opts, busy_env(1 << 17, 16));
        assert!(s);
        assert!(matches!(receipt.inner, InnerReceipt::Succinct(_)));
        receipt.verify(MULTI_TEST_ID).unwrap();
    }
}

/// A session that fails mid-stream (the session limit, after segments were proved) returns its
/// error, and the next session on the thread proves (the pending finalize was drained).
#[test]
fn z17_stream_error_then_prove() {
    let opts = ProverOpts::composite();
    let env = ExecutorEnv::builder()
        .write(&MultiTestSpec::BusyLoop { cycles: 1 << 19 })
        .unwrap()
        .segment_limit_po2(16)
        .session_limit(Some(1 << 18))
        .build()
        .unwrap();
    let before = streamed();
    let err = get_prover_server(&opts)
        .unwrap()
        .prove(env, MULTI_TEST_ELF)
        .err()
        .expect("a session over its limit proved");
    assert!(streamed() > before);
    assert!(
        format!("{err:#}").contains("Session limit exceeded"),
        "unexpected error: {err:#}"
    );

    // A segment above the prover's max po2: the stream's preflight refuses it.
    let small = ProverOpts::composite().with_segment_po2_max(15);
    let err = get_prover_server(&small)
        .unwrap()
        .prove(busy_env(1 << 17, 16), MULTI_TEST_ELF)
        .err()
        .expect("a po2 16 segment proved with max po2 15");
    assert!(
        format!("{err:#}").contains("segment po2 exceeds max"),
        "unexpected error: {err:#}"
    );

    let (receipt, s) = prove(None, &opts, busy_env(1 << 17, 16));
    assert!(s);
    receipt.verify(MULTI_TEST_ID).unwrap();
}

/// Run `f` on a new thread; fail (instead of hanging the suite) when it does not return in time.
fn within<R: Send + 'static>(what: &str, f: impl FnOnce() -> R + Send + 'static) -> R {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(120))
        .unwrap_or_else(|_| panic!("{what}: no result in 120 s (deadlock?)"))
}

/// Regression (Z17 rev 1): an io callback that proves on the GPU while the guest executes. The
/// stream used to hold the GPU lock on its proving thread for the whole execution, so the nested
/// proof on the calling thread waited for it forever.
#[test]
fn z17_prove_inside_io_callback() {
    use risc0_zkvm_methods::multi_test::SYS_MULTI_TEST;
    let (receipt, s) = within("prove inside an io callback", || {
        let env = ExecutorEnv::builder()
            .write(&MultiTestSpec::Syscall { count: 1 })
            .unwrap()
            .io_callback(SYS_MULTI_TEST, |_| {
                // The old path, so only the outer session counts as streamed.
                let (inner, _) = prove(Some("0"), &ProverOpts::composite(), busy_env(1 << 16, 16));
                inner.verify(MULTI_TEST_ID).unwrap();
                Ok(bytes::Bytes::new())
            })
            .build()
            .unwrap();
        prove(None, &ProverOpts::composite(), env)
    });
    assert!(s, "the outer session did not stream");
    receipt.verify(MULTI_TEST_ID).unwrap();
}

/// Regression (Z17 rev 1): a caller that holds the (reentrant) GPU lock and proves takes the old
/// path; the stream's proving thread could not take the lock it holds.
#[test]
fn z17_caller_holds_gpu_lock() {
    let (receipt, s) = within("prove under the GPU lock", || {
        let _gpu = risc0_zkp::hal::metal::exclusive();
        prove(None, &ProverOpts::composite(), busy_env(1 << 17, 16))
    });
    assert!(!s, "a caller holding the GPU lock streamed");
    receipt.verify(MULTI_TEST_ID).unwrap();
}

/// Sessions the stream leaves on the old path.
#[test]
fn z17_eligibility() {
    use crate::{Assumption, AssumptionReceipt};
    let opts = ProverOpts::composite();

    // An assumption (the guest does not use it, so the session proves).
    let assumption = Assumption {
        claim: crate::sha::Digest::ZERO,
        control_root: crate::sha::Digest::ZERO,
    };
    let env = ExecutorEnv::builder()
        .write(&MultiTestSpec::DoNothing)
        .unwrap()
        .add_assumption(AssumptionReceipt::Unresolved(assumption))
        .build()
        .unwrap();
    let (receipt, s) = prove(None, &opts, env);
    assert!(!s, "a session with assumptions streamed");
    receipt.verify(MULTI_TEST_ID).unwrap();

    // A segment path.
    let dir = tempfile::tempdir().unwrap();
    let env = ExecutorEnv::builder()
        .write(&MultiTestSpec::DoNothing)
        .unwrap()
        .segment_path(dir.path())
        .build()
        .unwrap();
    let (receipt, s) = prove(None, &opts, env);
    assert!(!s, "a session with a segment path streamed");
    receipt.verify(MULTI_TEST_ID).unwrap();

    // The knob.
    let env = ExecutorEnv::builder()
        .write(&MultiTestSpec::DoNothing)
        .unwrap()
        .build()
        .unwrap();
    let (receipt, s) = prove(Some(" 0 "), &opts, env);
    assert!(!s);
    receipt.verify(MULTI_TEST_ID).unwrap();
}

/// Lead #19 probe (commit-pool-keep): the first segment of a session costs more than the others.
/// Proves `Z17_REPS` (default 4) composite sessions of a BusyLoop at `Z17_PO2` (default 20) on
/// one thread and prints each session's wall time; with `Z17_RELEASE=1` every other session
/// starts after `release_hal_buffers` (what the worker's wrap does after every job), so the
/// excess of a rebuilt segment prover (fresh pool) shows against a kept one. Run with
/// `RISC0_VERBOSE=1` for the per-segment times:
/// `cargo test -p risc0-zkvm --features metal --release --lib -- tests_z17::z17_first_segment_probe
/// --ignored --nocapture`.
#[test]
#[ignore = "po2 20 sessions (~9.4 GiB of GPU memory); run alone"]
fn z17_first_segment_probe() {
    let env_u64 = |k: &str, d: u64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let po2 = env_u64("Z17_PO2", 20) as u32;
    let reps = env_u64("Z17_REPS", 4);
    let release = env_u64("Z17_RELEASE", 1) == 1;
    let opts = ProverOpts::composite();
    for rep in 0..reps {
        let released = release && rep % 2 == 1;
        if released {
            super::release_hal_buffers();
        }
        let t = std::time::Instant::now();
        let (receipt, _) = prove(None, &opts, busy_env(3 << po2, po2));
        let s = t.elapsed().as_secs_f64();
        let n = receipt.inner.composite().unwrap().segments.len();
        eprintln!(
            "[Z17-probe] rep={rep} po2={po2} segments={n} released_before={released} wall={s:.3}s"
        );
    }
}
