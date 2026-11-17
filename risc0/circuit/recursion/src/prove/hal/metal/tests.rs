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

//! C9 integration tests of the recursion Metal HAL (PLAN 4.3 row C9): the translators' `--check`
//! round trips, a metallib smoke test, bit-identical seals against the CPU prover (deterministic
//! ZK noise, [with_noise_seed]), the 256-bit seal-check noise seed (B9-SEC), identity_p254
//! through the CPU fallback with the host-chain check on, and the CPU HAL witgen leaving the
//! caller's trace untouched. The per-phase equivalence and
//! fault-path tests live with their phases (`c6_tests`, `c7_tests`, `c8_tests`).

use std::{
    collections::{BTreeSet, VecDeque},
    path::PathBuf,
    process::Command,
    sync::atomic::Ordering,
};

use risc0_circuit_recursion_sys::{RawPreflightTrace, StepMode};
use risc0_core::field::baby_bear::BabyBearElem;
use risc0_zkp::{
    adapter::{CircuitInfo as _, TapsProvider as _},
    field::Elem as _,
    hal::metal::MetalHalPoseidon2,
};
use test_log::test;

use super::*;
use crate::{
    prove::{
        hal::{cpu::CpuCircuitHal, CircuitWitnessGenerator as _},
        preflight::Preflight,
        witgen::host_ctrl_for,
        zkr::get_zkr,
    },
    CircuitImpl, CIRCUIT,
};

fn test_input() -> VecDeque<u32> {
    // Two Poseidon2 digests (8 words each), as zkvm's `test_zkr` adds them.
    (0..16u32).map(|i| i * 0x101 + 7).collect()
}

fn translator_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../recursion-sys/kernels/cuda")
}

/// Run a translator with `--check` (in-memory translation, pinned rewrite counts, residual scan,
/// round trip; nothing written). PLAN 4.3 row C9 / C4 note: the build enforces only the pins.
fn run_translator_check(script: &str, extra: &[&str]) {
    let dir = translator_dir();
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let out = Command::new(&python)
        .arg(dir.join(script))
        .arg("--check")
        .args(extra)
        .current_dir(&dir)
        .output()
        .unwrap_or_else(|e| panic!("running {python} {script}: {e}"));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    eprintln!("{script} --check {extra:?}:\n{stdout}{stderr}");
    assert!(
        out.status.success(),
        "{script} --check {extra:?} failed ({}):\n{stdout}\n{stderr}",
        out.status
    );
}

#[test]
fn translator_steps_check() {
    run_translator_check("cuda2metal_steps.py", &[]);
}

#[test]
fn translator_eval_check_monolith() {
    run_translator_check("cuda2metal_eval.py", &[]);
}

#[test]
fn translator_eval_check_split() {
    run_translator_check("cuda2metal_eval.py", &["--split"]);
}

/// The embedded metallib holds exactly the entry points `METAL_KERNEL_NAMES` lists (no PSO is
/// built: cold eval_check PSOs take tens of seconds), and the base kernels load.
#[test]
fn metallib_smoke() {
    let device = metal::Device::system_default().expect("no Metal device");
    let library = device
        .new_library_with_data(METAL_LIB)
        .expect("METAL_LIB loads");
    let in_lib: BTreeSet<String> = library.function_names().into_iter().collect();
    let listed: BTreeSet<String> = METAL_KERNEL_NAMES.iter().map(|s| s.to_string()).collect();
    assert_eq!(listed.len(), METAL_KERNEL_NAMES.len(), "duplicate names");
    let missing: Vec<_> = listed.difference(&in_lib).collect();
    assert!(
        missing.is_empty(),
        "listed but not in METAL_LIB: {missing:?}"
    );
    let unlisted: Vec<_> = in_lib.difference(&listed).collect();
    assert!(
        unlisted.is_empty(),
        "in METAL_LIB but not listed: {unlisted:?}"
    );
    for name in EvalKernels::ALL.iter().flat_map(|v| v.names()) {
        assert!(listed.contains(*name), "{name}");
    }
    let kernels = RecursionKernels::get(&device).unwrap();
    for name in [
        "parStepExec",
        "parStepVerifyWom",
        "parStepComputeAccum",
        "parStepVerifyAccum",
    ] {
        assert!(kernels.base.get(name).is_some(), "{name}");
    }
    eprintln!(
        "C9 metallib: {} bytes, {} entry points",
        METAL_LIB.len(),
        in_lib.len()
    );
}

/// The seal check is not vacuous: without a noise seed two CPU proofs of the same input differ
/// (ZK noise); with the same seed they are identical; with different seeds they differ.
#[test]
fn noise_seed_controls_the_seal() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    let program = get_zkr("test_recursion_circuit.zkr", 16).unwrap();
    let cpu = crate::prove::hal::cpu::recursion_prover("poseidon2").unwrap();
    let a = cpu.prove(&program, test_input(), None).unwrap();
    let b = cpu.prove(&program, test_input(), None).unwrap();
    assert_ne!(a.seal, b.seal, "unseeded seals should differ");
    let c = with_noise_seed(7, || cpu.prove(&program, test_input(), None)).unwrap();
    let d = with_noise_seed(7, || cpu.prove(&program, test_input(), None)).unwrap();
    let e = with_noise_seed(8, || cpu.prove(&program, test_input(), None)).unwrap();
    assert_eq!(c.seal, d.seal);
    assert_ne!(c.seal, e.seal);
    // The seed is restored: unseeded again afterwards.
    let f = cpu.prove(&program, test_input(), None).unwrap();
    assert_ne!(f.seal, c.seal);
}

/// Full Metal proofs (all three circuit phases on the GPU) with deterministic noise give the CPU
/// prover's seal bit for bit. Prints wall time and peak device memory per phase (PLAN 9.1).
fn seal_matches_cpu(hashfn: &str, po2: usize) {
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    let program = get_zkr("test_recursion_circuit.zkr", po2).unwrap();
    let metal = metal_recursion_prover(hashfn).unwrap();
    let cpu = crate::prove::hal::cpu::recursion_prover(hashfn).unwrap();
    // Warm (PSOs, ctrl caches, pool) so the timing below is a steady-state proof.
    metal.prove(&program, test_input(), None).unwrap();
    cpu.prove(&program, test_input(), None).unwrap();
    let probe = MetalHalPoseidon2::new();
    probe.reset_peak_allocated_size();
    for a in &PHASE_ALLOC {
        a.store(0, Ordering::Relaxed);
    }
    let calls = phase_calls();
    let ok = SEAL_CHECKS_OK.load(Ordering::Relaxed);
    let r = prove_seal_checked(metal.as_ref(), cpu.as_ref(), &program, test_input(), None).unwrap();
    assert_eq!(SEAL_CHECKS_OK.load(Ordering::Relaxed), ok + 1);
    let after = phase_calls();
    for i in 0..3 {
        assert!(after[i] > calls[i], "phase {i} not served by the Metal HAL");
    }
    let alloc = phase_alloc();
    eprintln!(
        "C9 {hashfn} po2 {po2}: seal {} words == CPU; device alloc at witgen {} B, accum {} B, \
         eval_check {} B, proof peak {} B",
        r.seal.len(),
        alloc[0],
        alloc[1],
        alloc[2],
        probe.peak_allocated_size()
    );
}

#[test]
fn seal_matches_cpu_poseidon2_po2_16() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    seal_matches_cpu("poseidon2", 16);
}

#[test]
fn seal_matches_cpu_poseidon2_po2_18() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    seal_matches_cpu("poseidon2", 18); // zkvm RECURSION_PO2: the lift/join shape
}

#[test]
fn seal_matches_cpu_sha256_po2_16() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    seal_matches_cpu("sha-256", 16);
}

/// A seal mismatch is reported (not vacuous): the "CPU" side proves with another seed.
#[test]
fn seal_check_detects_a_difference() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    struct Reseeded(Box<dyn RecursionProver>);
    impl RecursionProver for Reseeded {
        fn prove(
            &self,
            program: &Program,
            input: VecDeque<u32>,
            preflight: Option<Preflight>,
        ) -> Result<RecursionReceipt> {
            with_noise_seed(1, || self.0.prove(program, input, preflight))
        }
    }
    let program = get_zkr("test_recursion_circuit.zkr", 16).unwrap();
    let metal = metal_recursion_prover("poseidon2").unwrap();
    let cpu = Reseeded(crate::prove::hal::cpu::recursion_prover("poseidon2").unwrap());
    let err = prove_seal_checked(metal.as_ref(), &cpu, &program, test_input(), None).unwrap_err();
    assert!(format!("{err:#}").contains("seal differs"), "{err:#}");
}

/// B9-SEC (host only): each noise stream's ChaCha key depends on every one of the 256 seed bits
/// and on the stream, so no 64-bit value selects the noise of a seeded proof. Before B9-SEC the
/// key was `seed_from_u64(seed ^ stream * K)` of a 64-bit seed.
#[test]
fn b9_sec_noise_key_uses_every_seed_bit() {
    use crate::prove::witgen::{noise_seed_from_u64, noise_stream_key, seeded_noise_rng};
    use rand::RngCore as _;
    let seed = crate::prove::witgen::os_noise_seed().unwrap();
    assert_ne!(
        seed,
        crate::prove::witgen::os_noise_seed().unwrap(),
        "fresh OS seeds"
    );
    for stream in [1u64, 2] {
        let key = noise_stream_key(&seed, stream);
        let first = seeded_noise_rng(&seed, stream).next_u64();
        for bit in 0..256 {
            let mut s = seed;
            s[bit / 8] ^= 1 << (bit % 8);
            assert_ne!(
                noise_stream_key(&s, stream),
                key,
                "stream {stream}, seed bit {bit}"
            );
            assert_ne!(
                seeded_noise_rng(&s, stream).next_u64(),
                first,
                "stream {stream}, bit {bit}"
            );
        }
    }
    assert_ne!(noise_stream_key(&seed, 1), noise_stream_key(&seed, 2));
    // The test expansion of a u64 is domain separated from the stream keys.
    assert_ne!(noise_seed_from_u64(7), noise_seed_from_u64(8));
    assert_ne!(
        noise_stream_key(&noise_seed_from_u64(7), 1),
        noise_seed_from_u64(7)
    );
}

/// B9-SEC: the receipt `RISC0_METAL_REC_SEAL_CHECK` returns (Metal, checked against the CPU) has
/// ZK noise that is a function of a fresh 256-bit OS seed and of no 64-bit value: the CPU re-proof
/// under that seed reproduces its seal; flipping one seed bit outside the first 64 changes it; the
/// seed's low 64 bits used as a `u64` test seed do not reproduce it; two seal checks use
/// different seeds and give different seals. (The noise of a receipt is not reproducible from
/// any u64 because the 192 high seed bits, each of which changes the key
/// ([b9_sec_noise_key_uses_every_seed_bit]), are OS entropy no u64 path can set: `with_noise_seed`
/// is test-only and hashes its u64 under a separate domain.)
#[test]
fn b9_sec_seal_checked_receipt_noise_is_256_bit() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    let program = get_zkr("test_recursion_circuit.zkr", 16).unwrap();
    let metal = metal_recursion_prover("poseidon2").unwrap();
    let cpu = crate::prove::hal::cpu::recursion_prover("poseidon2").unwrap();
    let last_seed = || {
        LAST_SEAL_CHECK_SEED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("prove_seal_checked records its seed")
    };
    let r1 =
        prove_seal_checked(metal.as_ref(), cpu.as_ref(), &program, test_input(), None).unwrap();
    let s1 = last_seed();
    let r2 =
        prove_seal_checked(metal.as_ref(), cpu.as_ref(), &program, test_input(), None).unwrap();
    let s2 = last_seed();
    assert_ne!(s1, s2, "each seal check draws a fresh seed");
    assert_ne!(r1.seal, r2.seal, "each seal check has fresh noise");
    assert!(
        s1[8..].iter().any(|&b| b != 0),
        "seed bits above 64 are OS entropy"
    );
    let reprove = |seed: crate::prove::witgen::NoiseSeed| {
        with_noise_seed32(seed, || cpu.prove(&program, test_input(), None))
            .unwrap()
            .seal
    };
    assert_eq!(
        reprove(s1),
        r1.seal,
        "the receipt is the function of its 256-bit seed"
    );
    for bit in [64usize, 128, 255] {
        let mut s = s1;
        s[bit / 8] ^= 1 << (bit % 8);
        assert_ne!(reprove(s), r1.seal, "seed bit {bit} changes the noise");
    }
    let lo = u64::from_le_bytes(s1[..8].try_into().unwrap());
    let u = with_noise_seed(lo, || cpu.prove(&program, test_input(), None)).unwrap();
    assert_ne!(
        u.seal, r1.seal,
        "the seed's low 64 bits do not reproduce the receipt"
    );
    assert_ne!(crate::prove::witgen::noise_seed_from_u64(lo), s1);
    eprintln!(
        "B9-SEC: seal-checked receipt reproduced only by its 256-bit OS seed ({} seal words)",
        r1.seal.len()
    );
}

/// The CPU HAL witgen runs on a copy of the cycles: the caller's trace is unchanged and a second
/// witgen over it gives the same witness (the host-chain / prewitgen checks rerun it).
#[test]
fn cpu_witgen_keeps_trace() {
    let po2 = 16;
    let program = get_zkr("test_recursion_circuit.zkr", po2).unwrap();
    let mut pf = Preflight::new(test_input());
    for (cycle, row) in program.code_by_row().enumerate() {
        pf.step(cycle, row).unwrap();
    }
    let total = 1usize << po2;
    let ctrl = host_ctrl_for(&program);
    let cycles_before = pf.trace.cycles.clone();
    let raw = RawPreflightTrace {
        wom: pf.trace.wom.as_ptr(),
        cycles: pf.trace.cycles.as_ptr(),
        iops: pf.trace.iops.as_ptr(),
        num_woms: pf.trace.wom.len() as u32,
        num_cycles: program.code_rows() as u32,
        num_iops: pf.trace.iops.len() as u32,
    };
    let run = || {
        let data = vec![BabyBearElem::INVALID; total * CIRCUIT.data_size()];
        let global = vec![BabyBearElem::INVALID; CircuitImpl::OUTPUT_SIZE];
        let hal = risc0_zkp::hal::cpu::CpuHal::<risc0_core::field::baby_bear::BabyBear>::new(
            risc0_zkp::core::hash::poseidon2::Poseidon2HashSuite::new_suite(),
        );
        use risc0_zkp::hal::{Buffer as _, Hal as _};
        let (c, d, g) = (
            hal.copy_from_elem("ctrl", &ctrl),
            hal.copy_from_elem("data", &data),
            hal.copy_from_elem("global", &global),
        );
        CpuCircuitHal
            .generate_witness(StepMode::Parallel, total as u32, &raw, &c, &d, &g)
            .unwrap();
        let raw = |v: Vec<BabyBearElem>| -> Vec<u32> {
            v.iter().map(|e| e.as_u32_montgomery()).collect()
        };
        (raw(d.to_vec()), raw(g.to_vec()))
    };
    let a = run();
    assert_eq!(
        pf.trace.cycles, cycles_before,
        "CPU witgen changed the trace"
    );
    let b = run();
    assert!(a == b, "second CPU witgen over the same trace differs");
}

/// identity_p254 runs on the CPU fallback (no Metal Poseidon254 until Z10) through the public
/// [crate::prove::Prover] path (prepare: preflight + host chains), with the host-chain check on.
/// Before C9 the check's second CPU witgen read the iopIdx the first one had advanced and failed
/// ("eqz failed at: zirgen/circuit/recursion/wom.cpp:78"), which kept zkvm's
/// `test_recursion_host_chains_check` ignored on Metal.
#[test]
fn p254_cpu_fallback_with_host_chains_check() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    for hashfn in ["poseidon_254", "poseidon2"] {
        let program = get_zkr("test_recursion_circuit.zkr", 16).unwrap();
        let before = crate::prove::set_host_chains_check(true);
        let mut prover = crate::prove::Prover::new(program.clone(), hashfn);
        prover.add_input(test_input().make_contiguous());
        let r = prover.prepare().and_then(|_| prover.run());
        let after = crate::prove::set_host_chains_check(false);
        let r = r.unwrap();
        let cpu = crate::prove::hal::cpu::recursion_prover(hashfn).unwrap();
        let c = cpu.prove(&program, test_input(), None).unwrap();
        assert_eq!(r.output, c.output);
        let globals = CircuitImpl::OUTPUT_SIZE;
        assert_eq!(r.seal[..globals], c.seal[..globals]);
        eprintln!("C9 {hashfn}: host chain checks {}", after - before);
        assert!(after > before, "{hashfn}: no host-chain check ran");
    }
}

/// R17 audit: repeated Metal proofs of one input with a fixed noise seed are bit-identical with
/// each other and with the CPU prover's seal (`R17_AUDIT_REPS` proofs, default 10, per hash
/// function in `R17_AUDIT_HASHFNS`, default `poseidon2,sha-256`, at `R17_AUDIT_PO2`, default
/// 18). The whole pipeline runs, so every recursion kernel (steps, accum incl.
/// `prefixProductsChunks`, eval_check, the zkp kernels) is covered. Prints `R17AUDIT` k/n lines.
#[test]
#[ignore]
fn r17_seal_repeat() {
    crate::prove::hal::metal::require_heavy_gpu_run("r17_seal_repeat");
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let reps: usize = env("R17_AUDIT_REPS", "10").parse().unwrap();
    let po2: usize = env("R17_AUDIT_PO2", "18").parse().unwrap();
    let program = get_zkr("test_recursion_circuit.zkr", po2).unwrap();
    let mut bad = vec![];
    for hashfn in env("R17_AUDIT_HASHFNS", "poseidon2,sha-256").split(',') {
        let metal = metal_recursion_prover(hashfn).unwrap();
        let cpu = crate::prove::hal::cpu::recursion_prover(hashfn).unwrap();
        let want = with_noise_seed(7, || cpu.prove(&program, test_input(), None))
            .unwrap()
            .seal;
        let (mut vs_first, mut vs_cpu) = (0, 0);
        let mut first: Option<Vec<u32>> = None;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            let seal = with_noise_seed(7, || metal.prove(&program, test_input(), None))
                .unwrap()
                .seal;
            if seal != want {
                vs_cpu += 1;
            }
            match &first {
                None => first = Some(seal),
                Some(f) => vs_first += usize::from(*f != seal),
            }
        }
        println!(
            "R17AUDIT recursion seal {hashfn} po2 {po2}: {vs_first}/{reps} reps differ from rep 0, \
             {vs_cpu}/{reps} differ from CPU ({:.0} ms/proof)",
            t.elapsed().as_secs_f64() * 1e3 / reps as f64
        );
        if vs_first + vs_cpu > 0 {
            bad.push(hashfn.to_string());
        }
    }
    assert!(bad.is_empty(), "R17AUDIT recursion seals differ: {bad:?}");
}

/// P31-10 A/B bench (ignored; heavy, run by hand): real lift/join/resolve proofs replayed from
/// `C11_CAPTURES` (optional `C11_ONLY` filter), each round in the order zkvm makes them, with the
/// program handed over as [get_zkr] hands it: `RISC0_METAL_REC_ZKR_CACHE` on (default) passes
/// the same `Arc<Program>` to every proof of a program (the P31-10 cache); `0` builds a new one
/// per proof from its words, as stock 3.1's `get_zkr` does ([Program::from_encoded] inside the
/// timed region; the prover then re-checks the code against its ctrl caches). `P3110_REPS`
/// rounds (default 7) after one warm round. Prints `P3110CAP <file> ...` and a `P3110BENCH`
/// line with the per-round sum and the per-kind (lift/join) medians.
#[test]
#[ignore]
fn p31_10_bench_zkr_cache() {
    use std::time::Instant;
    crate::prove::hal::metal::require_heavy_gpu_run("p31_10_bench_zkr_cache");
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    let cache_on = std::env::var("RISC0_METAL_REC_ZKR_CACHE").as_deref() != Ok("0");
    let reps: usize = std::env::var("P3110_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let dir = std::env::var("C11_CAPTURES").expect("C11_CAPTURES=<capture dir>");
    let only = std::env::var("C11_ONLY").unwrap_or_default();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "rcap"))
        .filter(|p| only.is_empty() || p.to_string_lossy().contains(&only))
        .collect();
    files.sort();
    let caps: Vec<_> = files
        .iter()
        .map(|p| {
            let c = capture::read(p).unwrap();
            // The program's words, as the zkr holds them (canonical).
            let words: Vec<u32> = c.program.code.iter().map(|&e| u32::from(e)).collect();
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                c,
                words,
            )
        })
        .collect();
    assert!(!caps.is_empty(), "no captures");
    let mut provers: std::collections::HashMap<String, Box<dyn RecursionProver>> =
        Default::default();
    for (_, c, _) in &caps {
        if !provers.contains_key(&c.hashfn) {
            provers.insert(c.hashfn.clone(), metal_recursion_prover(&c.hashfn).unwrap());
        }
    }
    let preflight = |c: &capture::Capture| {
        let p = c.preflight()?;
        if crate::prove::host_chains_backend() {
            *p.host_chains.lock().unwrap() = crate::prove::witgen::host_chains(&c.program, &p);
        }
        crate::prove::stage_metal(&p);
        Some(p)
    };
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); caps.len()];
    let mut rounds = Vec::new();
    for rep in 0..=reps {
        let mut sum = 0.0;
        for (i, (_, c, words)) in caps.iter().enumerate() {
            let pf = preflight(c);
            let prover = &provers[&c.hashfn];
            let t = Instant::now();
            let fresh;
            let program = if cache_on {
                &c.program
            } else {
                fresh = Arc::new(Program::from_encoded(words, c.program.po2));
                &fresh
            };
            let r = prover.prove_shared(program, c.input.clone(), pf).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(r);
            if rep > 0 {
                times[i].push(ms);
                sum += ms;
            }
        }
        if rep > 0 {
            rounds.push(sum);
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (mut lift, mut join) = (Vec::new(), Vec::new());
    for (i, (name, c, _)) in caps.iter().enumerate() {
        let t = &mut times[i];
        let m = med(t);
        println!(
            "P3110CAP cache={cache_on} {name} rows={} median={m:.2} min={:.2} max={:.2}",
            c.program.code_rows(),
            t[0],
            t[t.len() - 1]
        );
        // The 3.0 captures: lifts have 134164 rows, join/resolve ~259.6K (C11 note).
        if c.hashfn == "poseidon2"
            && c.program.code_rows() < 200_000
            && c.program.code_rows() > 130_000
        {
            lift.extend(t.iter().copied());
        } else if c.program.code_rows() > 250_000 {
            join.extend(t.iter().copied());
        }
    }
    let (lo, hi) = (
        rounds.iter().cloned().fold(f64::MAX, f64::min),
        rounds.iter().cloned().fold(0.0, f64::max),
    );
    let lm = if lift.is_empty() {
        f64::NAN
    } else {
        med(&mut lift)
    };
    let jm = if join.is_empty() {
        f64::NAN
    } else {
        med(&mut join)
    };
    println!(
        "P3110BENCH cache={cache_on} n={} round_median={:.2} min={lo:.2} max={hi:.2} \
         lift_median={lm:.2} join_median={jm:.2} reps={reps}",
        caps.len(),
        med(&mut rounds)
    );
}
