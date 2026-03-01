use std::time::Instant;

use risc0_zkvm::{get_prover_server, ExecutorEnv, ExecutorImpl, ProverOpts, ReceiptKind, VerifierContext};
#[cfg(any(feature = "cuda", feature = "rocm"))]
use risc0_circuit_rv32im;
use risc0_zkvm_methods::{bench::BenchmarkSpec, BENCH_ELF, BENCH_ID};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iters: u32 = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let po2: Option<u32> = args.get(2)
        .and_then(|s| s.parse().ok());

    // Receipt kind: "composite" (default), "succinct", or "groth16"
    let receipt_kind = match args.get(3).map(|s| s.as_str()) {
        Some("succinct") => ReceiptKind::Succinct,
        Some("groth16") => ReceiptKind::Groth16,
        _ => ReceiptKind::Composite,
    };

    // Support loading an external guest ELF for apples-to-apples benchmarking
    let external_elf = std::env::var("BENCH_ELF_PATH").ok().map(|p| {
        eprintln!("Loading external guest ELF from {p}");
        std::fs::read(&p).unwrap_or_else(|e| panic!("Failed to read {p}: {e}"))
    });
    let elf: &[u8] = external_elf.as_deref().unwrap_or(BENCH_ELF);

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
    let session = ExecutorImpl::from_elf(env, elf).unwrap().run().unwrap();
    eprintln!(
        "  segments: {}, total_cycles: {}, user_cycles: {}",
        session.segments.len(),
        session.total_cycles,
        session.user_cycles,
    );

    let opts = ProverOpts::default()
        .with_receipt_kind(receipt_kind)
        .with_hashfn("poseidon2".to_string());
    eprintln!("Receipt kind: {:?}", receipt_kind);
    let prover = get_prover_server(&opts).unwrap();
    let ctx = VerifierContext::default();

    // Pre-load GPU modules so the first kernel launch doesn't stall ~100ms.
    #[cfg(feature = "cuda")]
    risc0_circuit_rv32im::prove::cuda_warmup();
    #[cfg(feature = "rocm")]
    risc0_circuit_rv32im::prove::rocm_warmup();

    let kind_str = match receipt_kind {
        ReceiptKind::Composite => "composite",
        ReceiptKind::Succinct => "succinct",
        ReceiptKind::Groth16 => "groth16",
        _ => "unknown",
    };
    eprintln!("Proving {} segments ({kind_str})...", session.segments.len());
    let t0 = Instant::now();
    let prove_info = prover.prove_session(&ctx, &session).unwrap();
    let elapsed = t0.elapsed();

    if external_elf.is_none() {
        eprintln!("Verifying receipt...");
        prove_info
            .receipt
            .verify(BENCH_ID)
            .expect("receipt verification failed");
        eprintln!("  OK");
    } else {
        eprintln!("Skipping verification (external ELF, BENCH_ID mismatch)");
    }

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
