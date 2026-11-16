//! P31-14 receipt producer: proves a guest with this tree's prover (`--features metal`) and writes
//! its composite, succinct and Groth16 receipts for `stock-verify31`.
//!
//! ```text
//! p31-14-produce receipts <out dir> <case> [<elf .bin> <u32 input>]
//!   case fibonacci | merkle : a zk-miner guest (ProgramBinary .bin), input written as 4 LE bytes
//!                              (write_slice), as the zk-miner worker's --benchmark does
//!   case compose            : multi_test SysVerify of two proven Echo assumptions (resolve)
//!   case keccak             : multi_test KeccakUnion(1) (a keccak assumption)
//! p31-14-produce wrap-timing <elf .bin> <u32 input> <reps>
//!   one succinct receipt, then <reps> Groth16 compressions in this process (identity_p254 +
//!   shrink_wrap; the library prints `[succinct_to_groth16]` per wrap, `[recursion_prove]` with
//!   RISC0_VERBOSE=1); `P14_RELEASE_BETWEEN=1` frees the Groth16 key and HAL pools between wraps
//! ```
//!
//! Writes `<out>/<case>.{composite,succinct,groth16}.receipt` (bincode 1.3 `Receipt`, the 3.1
//! crate's own receipt encoding), `<case>.image_id` and `<case>.journal` (hex), and prints
//! `[P31-14]` lines (stage times, Metal peak of the process' device allocation, sampled every
//! 2 ms). Every receipt is also verified with this tree's verifier before it is written.
//! Segment po2: `P14_SEG_PO2`, else the worker's tier for `RISC0_METAL_MAX_WORKING_SET_BYTES`
//! (< 6 GiB 18, < 12 GiB 19), else the executor default.

use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use risc0_zkvm::{
    compute_image_id, get_prover_server,
    sha::{Digest, Digestible},
    ExecutorEnv, ExecutorEnvBuilder, ProverOpts, Receipt,
};
use risc0_zkvm_methods::{multi_test::MultiTestSpec, MULTI_TEST_ELF, MULTI_TEST_ID};

struct Sampler {
    dev: metal::Device,
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    handle: Option<JoinHandle<()>>,
}

impl Sampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let handle = {
            let (stop, peak) = (stop.clone(), peak.clone());
            std::thread::spawn(move || {
                let dev = metal::Device::system_default().expect("Metal device");
                // P14_TRACE_ALLOC=1: print every change of the allocation by >= 1 MiB (to stderr,
                // in the order of the library's own stage lines).
                let trace = std::env::var("P14_TRACE_ALLOC").as_deref() == Ok("1");
                let (t0, mut last) = (Instant::now(), 0u64);
                while !stop.load(Ordering::Relaxed) {
                    let now = dev.current_allocated_size();
                    peak.fetch_max(now, Ordering::Relaxed);
                    if trace && now.abs_diff(last) >= 1 << 20 {
                        eprintln!(
                            "[P31-14 alloc] t={:.3}s {:.1} MiB ({:+.1})",
                            t0.elapsed().as_secs_f64(),
                            now as f64 / (1u64 << 20) as f64,
                            (now as f64 - last as f64) / (1u64 << 20) as f64
                        );
                        last = now;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
            })
        };
        Self {
            dev: metal::Device::system_default().expect("Metal device"),
            stop,
            peak,
            handle: Some(handle),
        }
    }

    /// The device's allocation now (MiB): what a stage leaves resident for the next one.
    fn resident(&self, stage: &str) -> f64 {
        let mib = self.dev.current_allocated_size() as f64 / (1u64 << 20) as f64;
        println!("[P31-14] resident after {stage}: {mib:.1} MiB");
        mib
    }

    /// Peak MiB since the last call.
    fn take(&self) -> f64 {
        std::thread::sleep(Duration::from_millis(5));
        self.peak.swap(0, Ordering::Relaxed) as f64 / (1u64 << 20) as f64
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn segment_po2() -> Option<u32> {
    if let Ok(v) = std::env::var("P14_SEG_PO2") {
        return Some(v.trim().parse().expect("P14_SEG_PO2"));
    }
    const GIB: u64 = 1 << 30;
    let limit: u64 = std::env::var("RISC0_METAL_MAX_WORKING_SET_BYTES")
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(match limit {
        l if l < 6 * GIB => 18,
        l if l < 12 * GIB => 19,
        _ => 20,
    })
}

fn builder<'a>() -> ExecutorEnvBuilder<'a> {
    let mut b = ExecutorEnv::builder();
    if let Some(po2) = segment_po2() {
        b.segment_limit_po2(po2);
    }
    b
}

/// The env, ELF and image id of a case.
fn case_env(case: &str, rest: &[String]) -> Result<(ExecutorEnv<'static>, Vec<u8>, Digest)> {
    match case {
        "fibonacci" | "merkle" => {
            let [elf, input] = rest else {
                bail!("{case} needs <elf .bin> <u32 input>");
            };
            let elf = std::fs::read(elf).with_context(|| format!("read {elf}"))?;
            let input: u32 = input.parse()?;
            let image_id = compute_image_id(&elf)?;
            let env = builder().write_slice(&input.to_le_bytes()).build()?;
            Ok((env, elf, image_id))
        }
        "compose" => {
            let opts = ProverOpts::composite();
            let prover = get_prover_server(&opts)?;
            let mut assumptions = Vec::new();
            for msg in [b"execution A".to_vec(), b"execution B".to_vec()] {
                let env = builder()
                    .write(&MultiTestSpec::Echo { bytes: msg })?
                    .build()?;
                assumptions.push(prover.prove(env, MULTI_TEST_ELF)?.receipt);
            }
            let mut b = builder();
            for a in &assumptions {
                b.add_assumption(a.clone());
            }
            let env = b
                .write(&MultiTestSpec::SysVerify(vec![
                    (MULTI_TEST_ID.into(), b"execution A".to_vec()),
                    (MULTI_TEST_ID.into(), b"execution B".to_vec()),
                ]))?
                .build()?;
            Ok((env, MULTI_TEST_ELF.to_vec(), MULTI_TEST_ID.into()))
        }
        "keccak" => {
            let env = builder().write(&MultiTestSpec::KeccakUnion(1))?.build()?;
            Ok((env, MULTI_TEST_ELF.to_vec(), MULTI_TEST_ID.into()))
        }
        _ => bail!("unknown case {case}"),
    }
}

fn write(out: &Path, name: &str, receipt: &Receipt) -> Result<PathBuf> {
    let path = out.join(name);
    std::fs::write(&path, bincode::serialize(receipt)?)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

fn receipts(out: &Path, case: &str, rest: &[String]) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let sampler = Sampler::start();
    prebuild(&sampler)?;
    let (env, elf, image_id) = case_env(case, rest)?;
    let pre_mib = sampler.take();

    let t = Instant::now();
    let info = get_prover_server(&ProverOpts::composite())?.prove(env, &elf)?;
    let composite = info.receipt;
    let prove_s = t.elapsed().as_secs_f64();
    let prove_mib = sampler.take();
    sampler.resident("composite");
    composite.verify(image_id)?;
    let segs = composite.inner.composite()?.segments.len();
    write(out, &format!("{case}.composite.receipt"), &composite)?;
    std::fs::write(
        out.join(format!("{case}.image_id")),
        hex::encode(image_id.as_bytes()),
    )?;
    std::fs::write(
        out.join(format!("{case}.journal")),
        hex::encode(&composite.journal.bytes),
    )?;

    let opts = ProverOpts::succinct();
    let t = Instant::now();
    let succinct = get_prover_server(&opts)?.compress(&opts, &composite)?;
    let succinct_s = t.elapsed().as_secs_f64();
    let succinct_mib = sampler.take();
    sampler.resident("succinct");
    succinct.verify(image_id)?;
    write(out, &format!("{case}.succinct.receipt"), &succinct)?;

    // What the Groth16 stage starts from: the wrap itself drops this thread's cached provers
    // before identity_p254 and again before the key loads, so release them here too and report
    // what stays (process-wide state such as pipeline objects), the headroom the key load sees.
    risc0_zkvm::release_hal_buffers();
    let wrap_start_mib = sampler.resident("succinct, this thread's provers released");
    let budget_mib = std::env::var("RISC0_METAL_MAX_WORKING_SET_BYTES")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(|b| b as f64 / (1u64 << 20) as f64);
    if let Some(b) = budget_mib {
        println!(
            "[P31-14] groth16 stage headroom before the key load: {:.1} MiB (budget {b:.0} MiB)",
            b - wrap_start_mib
        );
    }

    // P14_DIAG_IDENTITY=1: also run identity_p254 alone first and report what it leaves
    // allocated after this thread's provers are released (diagnostics, not timed).
    if std::env::var("P14_DIAG_IDENTITY").as_deref() == Ok("1") {
        let opts = ProverOpts::succinct();
        get_prover_server(&opts)?.identity_p254(succinct.inner.succinct()?)?;
        sampler.resident("identity_p254");
        risc0_zkvm::release_hal_buffers();
        sampler.resident("identity_p254, this thread's provers released");
        sampler.take();
    }

    let opts = ProverOpts::groth16();
    let t = Instant::now();
    let groth16 = get_prover_server(&opts)?.compress(&opts, &succinct)?;
    let groth16_s = t.elapsed().as_secs_f64();
    let groth16_mib = sampler.take();
    groth16.verify(image_id)?;
    write(out, &format!("{case}.groth16.receipt"), &groth16)?;
    if groth16.journal.bytes != composite.journal.bytes
        || groth16.claim()?.digest() != composite.claim()?.digest()
    {
        bail!("the Groth16 receipt's claim or journal differs from the composite receipt's");
    }

    println!(
        "[P31-14] case={case} image_id={} segments={segs} user_cycles={} total_cycles={} \
         journal={}B budget={} seg_po2={:?} | composite {prove_s:.2}s peak {prove_mib:.0} MiB | \
         succinct {succinct_s:.2}s peak {succinct_mib:.0} MiB | groth16 {groth16_s:.2}s peak \
         {groth16_mib:.0} MiB | setup peak {pre_mib:.0} MiB | all verified locally",
        hex::encode(image_id.as_bytes()),
        info.stats.user_cycles,
        info.stats.total_cycles,
        composite.journal.bytes.len(),
        std::env::var("RISC0_METAL_MAX_WORKING_SET_BYTES").unwrap_or_else(|_| "default".into()),
        segment_po2(),
    );
    Ok(())
}

fn wrap_timing(rest: &[String]) -> Result<()> {
    let [elf, input, reps] = rest else {
        bail!("wrap-timing needs <elf .bin> <u32 input> <reps>");
    };
    let reps: usize = reps.parse()?;
    let (env, elf, image_id) = case_env("fibonacci", &[elf.clone(), input.clone()])?;
    let opts = ProverOpts::succinct();
    let succinct = get_prover_server(&opts)?.prove(env, &elf)?.receipt;
    succinct.verify(image_id)?;
    let opts = ProverOpts::groth16();
    for rep in 0..reps {
        let t = Instant::now();
        let g = get_prover_server(&opts)?.compress(&opts, &succinct)?;
        let s = t.elapsed().as_secs_f64();
        g.verify(image_id)?;
        println!("[P31-14] wrap rep={rep} ms={:.1}", s * 1000.0);
        // P14_RELEASE_BETWEEN=1: free the cached Groth16 key and HAL pools between wraps (as a
        // worker does between jobs), so later wraps fit a 4 GiB budget. Not timed.
        if std::env::var("P14_RELEASE_BETWEEN").as_deref() == Ok("1") {
            risc0_zkvm::release_hal_buffers();
            risc0_groth16::prove::release_srs()?;
        }
    }
    Ok(())
}

/// `P14_PREBUILD` (`,`-separated): build these process-wide Metal pipeline sets at startup, in
/// order, and print what each leaves allocated. `keccak`: the keccak PSOs (otherwise built by the
/// first keccak proof); `groth16`: the Groth16 prover's PSOs (otherwise built by the wrap);
/// `rv32im`, `recursion`: those circuits' warmups. `worker`: all four on parallel threads, as the
/// zk-miner worker's `warm_up_pipelines` does at startup (so a 4 GiB wrap sees the PSO footprint
/// a worker has).
fn prebuild(sampler: &Sampler) -> Result<()> {
    let Ok(list) = std::env::var("P14_PREBUILD") else {
        return Ok(());
    };
    for what in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match what {
            "keccak" => risc0_circuit_keccak::prove::metal_warmup(),
            "recursion" => risc0_circuit_recursion::prove::metal_warmup(),
            "rv32im" => risc0_circuit_rv32im::prove::metal_warmup(),
            "groth16" => {
                risc0_groth16_sys::metal::metal_warmup()?;
            }
            "worker" => std::thread::scope(|s| {
                s.spawn(risc0_circuit_rv32im::prove::metal_warmup);
                s.spawn(risc0_circuit_recursion::prove::metal_warmup);
                s.spawn(risc0_circuit_keccak::prove::metal_warmup);
                s.spawn(|| risc0_groth16_sys::metal::metal_warmup().map(drop));
            }),
            _ => bail!("P14_PREBUILD: unknown {what}"),
        }
        sampler.resident(&format!("prebuilding the {what} PSOs"));
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("receipts") if args.len() >= 3 => receipts(Path::new(&args[1]), &args[2], &args[3..]),
        Some("wrap-timing") => wrap_timing(&args[1..]),
        _ => bail!(
            "usage: p31-14-produce receipts <out dir> <case> [<elf .bin> <u32 input>] | \
             wrap-timing <elf .bin> <u32 input> <reps>"
        ),
    }
}
