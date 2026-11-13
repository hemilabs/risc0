use std::time::Instant;

use risc0_zkvm::{
    get_prover_server, ExecutorEnv, ExecutorImpl, InnerReceipt, ProverOpts, Receipt, ReceiptKind,
    VerifierContext,
};
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

    // Always run STARK proving in composite mode so we can time each compression stage
    // (composite -> succinct -> groth16) independently. The `receipt_kind` argument now
    // controls how far up the wrapper ladder we climb, not what prove_session emits.
    let opts = ProverOpts::default()
        .with_receipt_kind(ReceiptKind::Composite)
        .with_hashfn("poseidon2".to_string());
    eprintln!("Target receipt kind: {:?}", receipt_kind);
    let prover = get_prover_server(&opts).unwrap();
    let ctx = VerifierContext::default();

    // Pre-load GPU modules so the first kernel launch doesn't stall ~100ms.
    #[cfg(feature = "cuda")]
    risc0_circuit_rv32im::prove::cuda_warmup();
    #[cfg(feature = "rocm")]
    risc0_circuit_rv32im::prove::rocm_warmup();

    // Preload the Groth16 circom graph + SRS off the critical path.
    if receipt_kind == ReceiptKind::Groth16 {
        let t = Instant::now();
        risc0_groth16::prove::preload_graph().expect("groth16 preload_graph");
        eprintln!("groth16 preload_graph: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Stage 1: composite STARK proving ---
    eprintln!("Proving {} segments (composite)...", session.segments.len());
    let t_e2e = Instant::now();
    let t0 = Instant::now();
    let prove_info = prover.prove_session(&ctx, &session).unwrap();
    let composite_secs = t0.elapsed().as_secs_f64();
    let composite_receipt = prove_info
        .receipt
        .inner
        .composite()
        .expect("prove_session should return a composite receipt")
        .clone();
    let stats = prove_info.stats.clone();
    let journal = prove_info.receipt.journal.bytes.clone();
    eprintln!(
        "  composite prove: {composite_secs:.2}s ({} segments, {:.1}ms/segment)",
        session.segments.len(),
        composite_secs * 1000.0 / session.segments.len() as f64,
    );

    // --- Stage 2: composite -> succinct ---
    let (succinct_secs, succinct_receipt) = if receipt_kind != ReceiptKind::Composite {
        if receipt_kind == ReceiptKind::Groth16 {
            // Preload SRS while GPU is idle right after STARK; done inline here so it's
            // accounted for in total e2e time.
            let t = Instant::now();
            risc0_groth16::prove::preload_srs().expect("groth16 preload_srs");
            eprintln!("groth16 preload_srs: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
        }
        let t1 = Instant::now();
        let sr = prover
            .composite_to_succinct(&composite_receipt)
            .expect("composite_to_succinct");
        let s = t1.elapsed().as_secs_f64();
        eprintln!("  composite->succinct: {s:.2}s");
        (s, Some(sr))
    } else {
        (0.0, None)
    };

    // --- Stage 3: succinct -> groth16 ---
    let (groth16_secs, final_receipt) = match receipt_kind {
        ReceiptKind::Composite => (0.0, prove_info.receipt.clone()),
        ReceiptKind::Succinct => {
            let r = Receipt::new(
                InnerReceipt::Succinct(succinct_receipt.clone().unwrap()),
                journal.clone(),
            );
            (0.0, r)
        }
        ReceiptKind::Groth16 => {
            let t2 = Instant::now();
            let g = prover
                .succinct_to_groth16(succinct_receipt.as_ref().unwrap())
                .expect("succinct_to_groth16");
            let s = t2.elapsed().as_secs_f64();
            eprintln!("  succinct->groth16: {s:.2}s");
            let r = Receipt::new(InnerReceipt::Groth16(g), journal.clone());
            (s, r)
        }
        _ => unreachable!(),
    };

    let e2e_secs = t_e2e.elapsed().as_secs_f64();

    if external_elf.is_none() {
        eprintln!("Verifying final receipt...");
        final_receipt
            .verify(BENCH_ID)
            .expect("receipt verification failed");
        eprintln!("  OK");
    } else {
        eprintln!("Skipping verification (external ELF, BENCH_ID mismatch)");
    }

    let seal_len = match &final_receipt.inner {
        InnerReceipt::Composite(_) => None,
        InnerReceipt::Succinct(s) => Some(s.get_seal_bytes().len()),
        InnerReceipt::Groth16(g) => Some(g.seal.len()),
        _ => None,
    };

    let throughput = iters as f64 / e2e_secs;
    let kind_str = match receipt_kind {
        ReceiptKind::Composite => "composite",
        ReceiptKind::Succinct => "succinct",
        ReceiptKind::Groth16 => "groth16",
        _ => "unknown",
    };

    println!();
    println!("=== SHA256 Benchmark ({iters} iterations, target={kind_str}) ===");
    println!("  composite STARK:     {composite_secs:6.2}s  ({} segments)", session.segments.len());
    if receipt_kind != ReceiptKind::Composite {
        println!("  composite->succinct: {succinct_secs:6.2}s");
    }
    if receipt_kind == ReceiptKind::Groth16 {
        println!("  succinct->groth16:   {groth16_secs:6.2}s");
    }
    println!("  ---------------------------------");
    println!("  total e2e:           {e2e_secs:6.2}s");
    println!("  throughput (e2e):    {throughput:.0} SHA256/s");
    println!("  total_cycles:        {}", stats.total_cycles);
    println!("  user_cycles:         {}", stats.user_cycles);
    if let Some(n) = seal_len {
        println!("  final seal bytes:    {n}");
    }
}
