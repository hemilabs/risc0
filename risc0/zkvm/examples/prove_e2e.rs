//! End-to-end Intel prover validation.
//!
//! Executes a guest, proves it in the chosen receipt kind, and verifies the
//! output. Default `succinct` exercises the lift + join recursion path, which
//! is the main validation goal on Intel (segment STARK is already covered by
//! prove_busy and prove_and_verify). `groth16` additionally wraps the
//! succinct receipt via the configured groth16 backend (Docker fallback when
//! neither cuda nor rocm features are enabled).
//!
//! Usage:
//!   cargo run --release --features intel --example prove_e2e -- \
//!     [iters] [kind] [po2]
//!
//!   iters:  guest SimpleLoop iterations         (default 2_900_000)
//!   kind:   composite | succinct | groth16      (default succinct)
//!   po2:    segment_limit_po2                   (default 20)
//!
//! Note: run via `run_e2e.sh` (repo root) to set LD_LIBRARY_PATH /
//! IGC_TotalGRFNum / NEO/SYCL cache env vars. Running `cargo run` directly
//! may hit a missing-.so error or a cold-start regression on the first run.

use std::process::ExitCode;
use std::time::Instant;

use risc0_zkvm::{
    get_prover_server, ExecutorEnv, ExecutorImpl, ExitCode as ZkvmExit, ProverOpts, ReceiptKind,
    VerifierContext,
};
use risc0_zkvm_methods::{bench::BenchmarkSpec, BENCH_ELF, BENCH_ID};

fn main() -> ExitCode {
    let iters: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_900_000);
    let kind_arg = std::env::args().nth(2).unwrap_or_else(|| "succinct".into());
    let kind = match kind_arg.as_str() {
        "composite" => ReceiptKind::Composite,
        "succinct" => ReceiptKind::Succinct,
        "groth16" => ReceiptKind::Groth16,
        other => {
            eprintln!(
                "unknown receipt kind '{other}' (expected composite|succinct|groth16)"
            );
            return ExitCode::from(2);
        }
    };
    let po2: u32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    if !(16..=22).contains(&po2) {
        eprintln!("po2 {po2} outside supported range [16..=22]");
        return ExitCode::from(2);
    }

    // Groth16 wrap uses the Docker backend on Intel (no native SYCL backend).
    // Fail fast with a clear message rather than panicking deep inside the
    // groth16 crate on a Command spawn error.
    if matches!(kind, ReceiptKind::Groth16)
        && !cfg!(any(feature = "cuda", feature = "rocm"))
        && std::process::Command::new("docker")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
    {
        eprintln!(
            "groth16 mode on an intel-only build requires `docker` on PATH; \
             install docker or use kind=succinct"
        );
        return ExitCode::from(2);
    }

    let spec = BenchmarkSpec::SimpleLoop { iters };

    eprintln!("Executing guest (SimpleLoop {iters}, po2={po2}) -> {kind:?} receipt");
    let env = ExecutorEnv::builder()
        .write(&spec)
        .unwrap()
        .segment_limit_po2(po2)
        .build()
        .unwrap();
    let session = ExecutorImpl::from_elf(env, BENCH_ELF).unwrap().run().unwrap();

    // Surface non-halt exits — a Fault or SystemSplit would otherwise just
    // propagate as a possibly-valid receipt that proves "the program crashed".
    match session.exit_code {
        ZkvmExit::Halted(0) => {}
        other => {
            eprintln!("guest did not halt(0): {other:?}");
            return ExitCode::from(3);
        }
    }

    let n_seg = session.segments.len();
    eprintln!(
        "  segments: {}, total_cycles: {}, user_cycles: {}",
        n_seg, session.total_cycles, session.user_cycles,
    );

    // For kinds that use recursion, require at least 2 segments so lift+join
    // is actually exercised. A 1-segment Succinct is just a lift; it validates
    // the lift path only, which is not the main thing this example is for.
    if matches!(kind, ReceiptKind::Succinct | ReceiptKind::Groth16) && n_seg < 2 {
        eprintln!(
            "kind={kind:?} but session produced only {n_seg} segment(s). \
             Increase iters so po2={po2} spans >=2 segments (join is not exercised otherwise)."
        );
        return ExitCode::from(4);
    }

    // Use the per-kind constructor so `max_segment_po2` and `control_ids`
    // stay in sync — `with_segment_po2_max()` alone changes the first without
    // regenerating the second, which causes GPU-side crashes (DEVICE_LOST) at
    // multi-segment counts.
    let opts = match kind {
        ReceiptKind::Composite => ProverOpts::composite(),
        ReceiptKind::Succinct => ProverOpts::succinct(),
        ReceiptKind::Groth16 => ProverOpts::groth16(),
        _ => ProverOpts::default().with_receipt_kind(kind),
    }
    .with_hashfn("poseidon2".to_string());
    let prover = get_prover_server(&opts).unwrap();
    let ctx = VerifierContext::default();

    eprintln!("Proving ({kind_arg})...");
    let t_prove = Instant::now();
    let prove_info = match prover.prove_session(&ctx, &session) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("prove_session failed: {e:#}");
            return ExitCode::from(5);
        }
    };
    let prove_elapsed = t_prove.elapsed();

    let receipt = prove_info.receipt;

    eprintln!("Verifying...");
    let t_verify = Instant::now();
    if let Err(e) = receipt.verify(BENCH_ID) {
        eprintln!("receipt verification FAILED: {e:#}");
        return ExitCode::from(6);
    }
    let verify_elapsed = t_verify.elapsed();

    let receipt_bytes = bincode::serialize(&receipt).unwrap().len();
    println!();
    println!("=== prove_e2e PASS ===");
    println!("mode:          {kind_arg}");
    println!("segments:      {}", n_seg);
    println!("prove total:   {:.3}s", prove_elapsed.as_secs_f64());
    if matches!(kind, ReceiptKind::Composite) {
        // ms/seg is only a meaningful number for pure STARK proving. For
        // succinct/groth16, the wall includes lift/join/shrink_wrap which
        // dominate at low segment counts.
        println!(
            "  per-segment: {:.1}ms  (composite STARK only)",
            prove_elapsed.as_secs_f64() * 1000.0 / n_seg as f64,
        );
    }
    println!("verify:        {:.3}s", verify_elapsed.as_secs_f64());
    println!("receipt size:  {receipt_bytes} bytes");
    println!("stats:         {:?}", prove_info.stats);

    ExitCode::SUCCESS
}
