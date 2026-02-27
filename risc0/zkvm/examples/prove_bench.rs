use std::time::Instant;

use risc0_zkvm::{get_prover_server, ExecutorEnv, ExecutorImpl, ProverOpts, VerifierContext};
use risc0_zkvm_methods::{bench::BenchmarkSpec, BENCH_ELF};

fn main() {
    let sha_iters: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let spec = BenchmarkSpec::HashBytesIter {
        buf: vec![0u8; 64],
        iters: sha_iters,
    };

    eprintln!("Executing guest ({sha_iters} SHA256 iterations)...");
    let env = ExecutorEnv::builder()
        .write(&spec)
        .unwrap()
        .build()
        .unwrap();
    let session = ExecutorImpl::from_elf(env, BENCH_ELF).unwrap().run().unwrap();
    eprintln!(
        "  segments: {}, total_cycles: {}, user_cycles: {}",
        session.segments.len(),
        session.total_cycles,
        session.user_cycles,
    );

    let opts = ProverOpts::composite().with_hashfn("poseidon2".to_string());
    let prover = get_prover_server(&opts).unwrap();
    let ctx = VerifierContext::default();

    // Pre-load CUDA modules so the first kernel launch doesn't stall ~100ms.
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    risc0_circuit_rv32im::prove::cuda_warmup();

    eprintln!("Proving {} segments...", session.segments.len());
    let t0 = Instant::now();
    let prove_info = prover.prove_session(&ctx, &session).unwrap();
    let elapsed = t0.elapsed();

    println!();
    println!(
        "prove_session: {:.2}s ({} segments, {:.1}ms/segment)",
        elapsed.as_secs_f64(),
        session.segments.len(),
        elapsed.as_secs_f64() * 1000.0 / session.segments.len() as f64,
    );
    println!("  stats: {:?}", prove_info.stats);
}
