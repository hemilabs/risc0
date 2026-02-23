use std::time::Instant;

use risc0_zkvm::{
    get_prover_server, ExecutorEnv, ExecutorImpl, ProverOpts, SimpleSegmentRef, VerifierContext,
};
#[cfg(feature = "cuda")]
use risc0_circuit_rv32im;
use risc0_zkvm_methods::{bench::BenchmarkSpec, BENCH_ELF, BENCH_ID};

fn main() {
    let iters: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let po2: Option<u32> = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok());

    let spec = BenchmarkSpec::HashBytesIter {
        buf: vec![0u8; 64],
        iters,
    };

    eprintln!("Executing guest ({iters} SHA256 iterations, po2={})...", po2.map_or("default(20)".to_string(), |p| p.to_string()));
    let mut builder = ExecutorEnv::builder();
    builder.write(&spec).unwrap();
    if let Some(po2) = po2 {
        builder.segment_limit_po2(po2);
    }
    let env = builder.build().unwrap();
    let session = ExecutorImpl::from_elf(env, BENCH_ELF)
        .unwrap()
        .run_with_callback(|segment| Ok(Box::new(SimpleSegmentRef::new(segment))))
        .unwrap();
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
    #[cfg(feature = "cuda")]
    risc0_circuit_rv32im::prove::cuda_warmup();

    eprintln!("Proving {} segments...", session.segments.len());
    let t0 = Instant::now();
    let prove_info = prover.prove_session(&ctx, &session).unwrap();
    let elapsed = t0.elapsed();

    eprintln!("Verifying receipt...");
    prove_info
        .receipt
        .verify(BENCH_ID)
        .expect("receipt verification failed");
    eprintln!("  OK");

    let secs = elapsed.as_secs_f64();
    let per_seg = secs * 1000.0 / session.segments.len() as f64;
    let throughput = iters as f64 / secs;

    println!();
    println!("=== SHA256 Benchmark ({iters} iterations) ===");
    println!(
        "prove_session: {secs:.2}s ({} segments, {per_seg:.1}ms/segment)",
        session.segments.len(),
    );
    println!("  throughput:   {throughput:.0} SHA256/s (proving)");
    println!("  total_cycles: {}", session.total_cycles);
    println!("  user_cycles:  {}", session.user_cycles);
    println!("  stats:        {:?}", prove_info.stats);
}
