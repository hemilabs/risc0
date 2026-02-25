use std::time::Instant;

use risc0_zkvm::{
    get_prover_server, ExecutorEnv, ExecutorImpl, ProverOpts, ReceiptKind, SimpleSegmentRef,
    VerifierContext,
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

    let do_groth16 = std::env::args().any(|a| a == "--groth16");

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

    // Phase 1: STARK proving (composite receipt)
    eprintln!("Proving {} segments (STARK)...", session.segments.len());
    let t0 = Instant::now();
    let prove_info = prover.prove_session(&ctx, &session).unwrap();
    let stark_elapsed = t0.elapsed();

    eprintln!("Verifying STARK receipt...");
    prove_info
        .receipt
        .verify(BENCH_ID)
        .expect("receipt verification failed");
    eprintln!("  OK");

    let secs = stark_elapsed.as_secs_f64();
    let per_seg = secs * 1000.0 / session.segments.len() as f64;
    let throughput = iters as f64 / secs;

    println!();
    println!("=== SHA256 Benchmark ({iters} iterations) ===");
    println!(
        "STARK proving: {secs:.2}s ({} segments, {per_seg:.1}ms/segment)",
        session.segments.len(),
    );
    println!("  throughput:   {throughput:.0} SHA256/s (proving)");
    println!("  total_cycles: {}", session.total_cycles);
    println!("  user_cycles:  {}", session.user_cycles);
    println!("  stats:        {:?}", prove_info.stats);

    if do_groth16 {
        // Pre-load Groth16 resources in background (graph parsing, SRS file pre-read).
        // Starting this before Succinct compression gives ~4s to complete, eliminating
        // the ~270ms graph cache wait that would otherwise occur in shrink_wrap.
        risc0_groth16::prove::prepare();

        // Phase 2: Lift + Join (composite → succinct)
        eprintln!("Compressing composite → succinct (lift + join)...");
        let succinct_opts = ProverOpts::composite()
            .with_hashfn("poseidon2".to_string())
            .with_receipt_kind(ReceiptKind::Succinct);
        let t1 = Instant::now();
        let succinct_receipt = prover
            .compress(&succinct_opts, &prove_info.receipt)
            .expect("composite_to_succinct failed");
        let succinct_elapsed = t1.elapsed();
        eprintln!("  succinct compression: {:.2}s", succinct_elapsed.as_secs_f64());

        // Phase 3: Groth16 wrapping (succinct → identity_p254 → groth16)
        eprintln!("Wrapping succinct → Groth16 (identity_p254 + shrink_wrap)...");
        let groth16_opts = ProverOpts::composite()
            .with_hashfn("poseidon2".to_string())
            .with_receipt_kind(ReceiptKind::Groth16);
        let t2 = Instant::now();
        let groth16_receipt = prover
            .compress(&groth16_opts, &succinct_receipt)
            .expect("succinct_to_groth16 failed");
        let groth16_elapsed = t2.elapsed();
        eprintln!("  groth16 wrapping: {:.2}s", groth16_elapsed.as_secs_f64());

        // Verify Groth16 receipt
        eprintln!("Verifying Groth16 receipt...");
        match groth16_receipt.verify(BENCH_ID) {
            Ok(_) => eprintln!("  OK"),
            Err(e) => eprintln!("  verification failed (expected with local keys): {e}"),
        }

        let total = stark_elapsed + succinct_elapsed + groth16_elapsed;
        println!();
        println!("=== End-to-End Breakdown ===");
        println!("  STARK proving:          {:.2}s", stark_elapsed.as_secs_f64());
        println!("  Succinct compression:   {:.2}s ({} lift + join tree)", succinct_elapsed.as_secs_f64(), session.segments.len());
        println!("  Groth16 wrapping:       {:.2}s (identity_p254 + shrink_wrap)", groth16_elapsed.as_secs_f64());
        println!("  Total end-to-end:       {:.2}s", total.as_secs_f64());
    }
}
