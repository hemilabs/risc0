use std::time::Instant;

use risc0_zkvm::{
    get_prover_server, ExecutorEnv, ExecutorImpl, ProverOpts, ReceiptKind, VerifierContext,
};
use risc0_zkvm_methods::{bench::BenchmarkSpec, BENCH_ELF, BENCH_ID};

fn main() {
    let sha_iters: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let kind_arg = std::env::args().nth(2).unwrap_or_else(|| "composite".into());
    let kind = match kind_arg.as_str() {
        "composite" => ReceiptKind::Composite,
        "succinct" => ReceiptKind::Succinct,
        other => {
            eprintln!("unknown ReceiptKind '{other}', defaulting to composite");
            ReceiptKind::Composite
        }
    };

    let spec = BenchmarkSpec::HashBytesIter {
        buf: vec![0u8; 64],
        iters: sha_iters,
    };

    eprintln!("Executing guest ({sha_iters} SHA256 iterations) -> {kind:?} receipt");
    let env = ExecutorEnv::builder().write(&spec).unwrap().build().unwrap();
    let session = ExecutorImpl::from_elf(env, BENCH_ELF).unwrap().run().unwrap();
    eprintln!(
        "  segments: {}, total_cycles: {}, user_cycles: {}",
        session.segments.len(),
        session.total_cycles,
        session.user_cycles,
    );

    let opts = ProverOpts::default()
        .with_hashfn("poseidon2".to_string())
        .with_receipt_kind(kind);
    let prover = get_prover_server(&opts).unwrap();
    let ctx = VerifierContext::default();

    let t0 = Instant::now();
    let prove_info = prover.prove_session(&ctx, &session).unwrap();
    let prove_elapsed = t0.elapsed();

    let receipt = prove_info.receipt;

    let t1 = Instant::now();
    receipt.verify(BENCH_ID).expect("receipt verify failed");
    let verify_elapsed = t1.elapsed();

    println!();
    println!(
        "prove:  {:.2}s  ({} segments, {:.1}ms/seg)",
        prove_elapsed.as_secs_f64(),
        session.segments.len(),
        prove_elapsed.as_secs_f64() * 1000.0 / session.segments.len() as f64,
    );
    println!("verify: {:.3}s  receipt_kind={:?}", verify_elapsed.as_secs_f64(), kind);
    println!("stats:  {:?}", prove_info.stats);
    let bytes = bincode::serialize(&receipt).unwrap();
    println!("receipt size: {} bytes", bytes.len());
}
