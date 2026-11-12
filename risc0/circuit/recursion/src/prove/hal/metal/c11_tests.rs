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

//! C11 (recursion optimization round 1): profile and A/B harnesses (ignored; run by hand) and
//! the correctness tests of the C11 changes.

use std::{collections::VecDeque, sync::Arc, time::Instant};

use test_log::test;

use super::*;
use crate::prove::{witgen::with_noise_seed, zkr::get_zkr};

fn test_input() -> VecDeque<u32> {
    (0..16u32).map(|i| i * 0x101 + 7).collect()
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Whole-proof wall time of the Metal prover on `test_recursion_circuit.zkr` (env `C11_HASHFN`,
/// `C11_PO2` default 18 = zkvm RECURSION_PO2, `C11_REPS` default 7, `C11_ZKR`), after one warm
/// proof. One line `C11BENCH ...` with the median/min/max for the A/B driver. With
/// `RISC0_METAL_REC_PROFILE=1` every proof also prints its per-phase profile.
#[test]
#[ignore]
fn c11_bench_proofs() {
    let _guard = global_state_test_lock();
    let hashfn = std::env::var("C11_HASHFN").unwrap_or_else(|_| "poseidon2".into());
    let zkr = std::env::var("C11_ZKR").unwrap_or_else(|_| "test_recursion_circuit.zkr".into());
    let po2 = env_usize("C11_PO2", 18);
    let reps = env_usize("C11_REPS", 7);
    let program = get_zkr(&zkr, po2).unwrap();
    let prover = metal_recursion_prover(&hashfn).unwrap();
    let t = Instant::now();
    prover.prove(&program, test_input(), None).unwrap();
    let first = t.elapsed().as_secs_f64() * 1e3;
    let mut times = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        prover.prove(&program, test_input(), None).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if profile::enabled() {
            let sum: f64 = profile::entries()
                .iter()
                .filter(|(l, _)| !l.starts_with("witgen."))
                .map(|(_, ms)| ms)
                .sum();
            eprintln!(
                "[rec_metal_profile] total={ms:.2} rest(fri)={:.2}",
                ms - sum
            );
        }
        times.push(ms);
    }
    let (lo, hi) = (
        times.iter().cloned().fold(f64::MAX, f64::min),
        times.iter().cloned().fold(0.0, f64::max),
    );
    let med = median(&mut times);
    println!(
        "C11BENCH hashfn={hashfn} zkr={zkr} po2={po2} rows={} first={first:.1} median={med:.2} \
         min={lo:.2} max={hi:.2} n={reps}",
        program.code_rows()
    );
}

/// The captures in `C11_CAPTURES` (a `RISC0_METAL_REC_CAPTURE` directory), sorted by name.
fn load_captures() -> Vec<(String, capture::Capture)> {
    let dir = std::env::var("C11_CAPTURES").expect("C11_CAPTURES=<capture dir>");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "rcap"))
        .collect();
    files.sort();
    let only = std::env::var("C11_ONLY").unwrap_or_default();
    files
        .into_iter()
        .filter(|p| only.is_empty() || p.to_string_lossy().contains(&only))
        .map(|p| {
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                capture::read(&p).unwrap(),
            )
        })
        .collect()
}

/// A preflight for a replay, prepared as zkvm's `prepare()` does (trace and host chains off the
/// timed path).
fn replay_preflight(c: &capture::Capture) -> Option<Preflight> {
    let p = c.preflight()?;
    if crate::prove::host_chains_backend() {
        let t = Instant::now();
        let chains = crate::prove::witgen::host_chains(&c.program, &p);
        if std::env::var_os("C12_CHAIN_STATS").is_some() {
            eprintln!(
                "C12CHAINS chains={} rows={} ms={:.2}",
                chains.as_ref().map_or(0, |h| h.heads.len()),
                chains.as_ref().map_or(0, |h| h.cycles.len()),
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        *p.host_chains.lock().unwrap() = chains;
    }
    Some(p)
}

/// Replay real lift/join/resolve/identity_p254 proofs (`C11_CAPTURES`, optional `C11_ONLY`
/// substring filter) `C11_REPS` times after a warm pass. One `C11CAP <file> ...` line per
/// capture (median/min/max ms) and a `C11BENCH captures ...` line with the per-round sum.
#[test]
#[ignore]
fn c11_bench_captures() {
    let _guard = global_state_test_lock();
    let reps = env_usize("C11_REPS", 5);
    let caps = load_captures();
    assert!(!caps.is_empty(), "no captures");
    let mut provers: std::collections::HashMap<String, Box<dyn RecursionProver>> =
        Default::default();
    for (_, c) in &caps {
        if !provers.contains_key(&c.hashfn) {
            provers.insert(c.hashfn.clone(), metal_recursion_prover(&c.hashfn).unwrap());
        }
    }
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); caps.len()];
    let mut rounds = Vec::new();
    for rep in 0..=reps {
        let mut sum = 0.0;
        for (i, (name, c)) in caps.iter().enumerate() {
            let pf = replay_preflight(c);
            let prover = &provers[&c.hashfn];
            let t = Instant::now();
            let r = prover
                .prove_shared(&c.program, c.input.clone(), pf)
                .unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(r);
            if profile::enabled() {
                let s: f64 = profile::entries()
                    .iter()
                    .filter(|(l, _)| !l.starts_with("witgen."))
                    .map(|(_, ms)| ms)
                    .sum();
                eprintln!(
                    "[rec_metal_profile] {name} total={ms:.2} rest(fri)={:.2}",
                    ms - s
                );
            }
            if rep > 0 {
                times[i].push(ms);
                sum += ms;
            }
        }
        if rep > 0 {
            rounds.push(sum);
        }
    }
    for (i, (name, c)) in caps.iter().enumerate() {
        let t = &mut times[i];
        let (lo, hi) = (
            t.iter().cloned().fold(f64::MAX, f64::min),
            t.iter().cloned().fold(0.0, f64::max),
        );
        println!(
            "C11CAP {name} rows={} median={:.2} min={lo:.2} max={hi:.2}",
            c.program.code_rows(),
            median(t)
        );
    }
    let (lo, hi) = (
        rounds.iter().cloned().fold(f64::MAX, f64::min),
        rounds.iter().cloned().fold(0.0, f64::max),
    );
    println!(
        "C11BENCH captures n={} median={:.2} min={lo:.2} max={hi:.2} reps={reps}",
        caps.len(),
        median(&mut rounds)
    );
}

/// Replayed captures: the seeded Metal proof equals the CPU proof (C9 seal check) for each.
#[test]
#[ignore]
fn c11_captures_seal_match_cpu() {
    let _guard = global_state_test_lock();
    for (name, c) in load_captures() {
        let metal = metal_recursion_prover(&c.hashfn).unwrap();
        let cpu = crate::prove::hal::cpu::recursion_prover(&c.hashfn).unwrap();
        prove_seal_checked(
            metal.as_ref(),
            cpu.as_ref(),
            &c.program,
            c.input.clone(),
            replay_preflight(&c),
        )
        .unwrap();
        eprintln!("C11 capture {name}: seal == CPU");
    }
}

/// Seeded Metal proof equals the CPU proof (C9's seal check) on the test program at `po2`, for
/// `hashfn`, under whatever C11 knobs the process has.
pub(crate) fn seal_matches_cpu(hashfn: &str, po2: usize) {
    // Serialized with the other whole-proof tests: the working-set budget is device-wide, so
    // concurrent proofs add up under RISC0_METAL_MAX_WORKING_SET_BYTES (C10 finding 6).
    let _guard = global_state_test_lock();
    let program = get_zkr("test_recursion_circuit.zkr", po2).unwrap();
    let metal = metal_recursion_prover(hashfn).unwrap();
    let cpu = crate::prove::hal::cpu::recursion_prover(hashfn).unwrap();
    prove_seal_checked(metal.as_ref(), cpu.as_ref(), &program, test_input(), None).unwrap();
    // Determinism: two seeded Metal proofs are identical.
    let a = with_noise_seed(5, || metal.prove(&program, test_input(), None)).unwrap();
    let b = with_noise_seed(5, || metal.prove(&program, test_input(), None)).unwrap();
    assert!(a.seal == b.seal, "two seeded Metal proofs differ");
}

#[test]
fn c11_seal_matches_cpu_poseidon2_po2_16() {
    seal_matches_cpu("poseidon2", 16);
}

#[test]
fn c11_seal_matches_cpu_poseidon2_po2_18() {
    seal_matches_cpu("poseidon2", 18);
}

#[test]
fn c11_seal_matches_cpu_sha256_po2_16() {
    seal_matches_cpu("sha-256", 16);
}

/// Cold/warm PSO creation per kernel set (run with `MTL_SHADER_CACHE_SIZE=0` for cold, K5's
/// method): base kernels, monolith eval_check, split eval_check, sequential (as loaded today) and
/// the split parts on parallel threads. Builds private `MetalKernels`, not the process statics.
#[test]
#[ignore]
fn c11_pso_times() {
    let _guard = global_state_test_lock();
    let device = metal::Device::system_default().unwrap();
    let base: Vec<&str> = METAL_KERNEL_NAMES
        .iter()
        .copied()
        .filter(|n| !is_eval_kernel(n))
        .collect();
    let time = |names: &[&str]| {
        let t = Instant::now();
        Pipelines::new(&device, METAL_LIB, names, 1).unwrap();
        t.elapsed().as_secs_f64()
    };
    let which = std::env::var("C11_PSO_SETS")
        .unwrap_or_else(|_| "base,split,split_par,monolith,base_pipe,split_pipe".into());
    for set in which.split(',') {
        let s = match set {
            "base" => time(&base),
            "split" => time(EVAL_SPLIT),
            "monolith" => time(EVAL_MONOLITH),
            "base_par" | "split_par" => {
                let names: &[&str] = if set == "base_par" { &base } else { EVAL_SPLIT };
                let t = Instant::now();
                std::thread::scope(|sc| {
                    for n in names {
                        let device = &device;
                        sc.spawn(move || Pipelines::new(device, METAL_LIB, &[n], 1).unwrap());
                    }
                });
                t.elapsed().as_secs_f64()
            }
            "base_pipe" | "split_pipe" => {
                let names: &[&str] = if set == "base_pipe" {
                    &base
                } else {
                    EVAL_SPLIT
                };
                let t = Instant::now();
                Pipelines::new(&device, METAL_LIB, names, pso_threads()).unwrap();
                t.elapsed().as_secs_f64()
            }
            _ => panic!("unknown set {set}"),
        };
        println!("C11PSO {set}: {s:.2} s");
    }
}

/// eval_check launch-width sweep at po2 `C11_PO2` (default 18): every width of `C11_EVAL_TGS`
/// (0 = default) on the selected variant's PSOs, or on the split kernels of `C11_EVAL_LIB` (a
/// metallib built with another `max_total_threads_per_threadgroup` attribute), interleaved
/// `C11_REPS` times; each result must equal the CPU eval_check. Prints the median encode+sync.
#[test]
#[ignore]
fn c11_eval_tg_sweep() {
    use risc0_zkp::{
        core::log2_ceil,
        field::{baby_bear::BabyBearElem, map_pow, Elem as _, ExtElem as _, RootsOfUnity as _},
        hal::Hal as _,
        INV_RATE,
    };
    let _guard = global_state_test_lock();
    let po2 = env_usize("C11_PO2", 18);
    let reps = env_usize("C11_REPS", 7);
    let hal = Arc::new(MetalHalPoseidon2::new());
    let c = MetalCircuitHal::new(hal.clone()).unwrap();
    let alt;
    let (variant, kernels): (EvalKernels, &Pipelines) = match std::env::var("C11_EVAL_LIB") {
        Ok(path) => {
            let bytes = std::fs::read(&path).unwrap();
            alt = Pipelines::new(&hal.device, &bytes, EVAL_SPLIT, 4).unwrap();
            (EvalKernels::Split, &alt)
        }
        Err(_) => c.eval_kernels().unwrap(),
    };
    for n in variant.names() {
        let pso = kernels.pipeline(n);
        eprintln!(
            "C11TG {n}: maxTotalThreadsPerThreadgroup {} width {} static tg mem {}",
            pso.max_total_threads_per_threadgroup(),
            pso.thread_execution_width(),
            pso.static_threadgroup_memory_length()
        );
    }
    let p = crate::testutil::EvalCheckParams::new(po2);
    let pows = map_pow(p.poly_mix, crate::info::POLY_MIX_POWERS);
    let rou = BabyBearElem::ROU_FWD[po2 + log2_ceil(INV_RATE)];
    let want = eval_check::cpu_eval_check(
        &p.code, &p.data, &p.accum, &p.mix, &p.out, &pows, rou, po2, p.domain,
    );
    let code = hal.copy_from_elem("code", &p.code);
    let data = hal.copy_from_elem("data", &p.data);
    let accum = hal.copy_from_elem("accum", &p.accum);
    let mix = hal.copy_from_elem("mix", &p.mix);
    let out = hal.copy_from_elem("out", &p.out);
    let groups = [&accum, &code, &data];
    let globals = [&mix, &out];
    let check = hal.alloc_elem("check", BabyBearExtElem::EXT_SIZE * p.domain);
    let tgs: Vec<u64> = std::env::var("C11_EVAL_TGS")
        .unwrap_or_else(|_| "0,32,64,128,256,512,1024".into())
        .split(',')
        .map(|t| t.trim().parse().unwrap())
        .collect();
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); tgs.len()];
    for rep in 0..=reps {
        for (i, &tg) in tgs.iter().enumerate() {
            check.view_mut(|v| v.fill(BabyBearElem::new(12345)));
            let t = Instant::now();
            with_tg(TgKind::Eval, tg, || {
                c.encode_eval_check(
                    variant, kernels, &check, &groups, &globals, &pows, po2, p.steps,
                )
            });
            hal.sync();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let got = check.to_vec();
            assert!(got == want, "tg {tg}: eval_check differs from the CPU");
            if rep > 0 {
                times[i].push(ms);
            }
        }
    }
    let _ = BabyBearElem::ZERO;
    for (i, tg) in tgs.iter().enumerate() {
        let t = &mut times[i];
        let (lo, hi) = (
            t.iter().cloned().fold(f64::MAX, f64::min),
            t.iter().cloned().fold(0.0, f64::max),
        );
        println!(
            "C11TG eval {variant:?} lib={} tg={tg}: median {:.2} ms min {lo:.2} max {hi:.2} (== CPU)",
            std::env::var("C11_EVAL_LIB").unwrap_or_else(|_| "builtin".into()),
            median(t)
        );
    }
}

#[test]
fn c11_row_passes_seal_matches_cpu_po2_16() {
    witgen::with_row_passes(true, || seal_matches_cpu("poseidon2", 16));
}

#[test]
fn c11_row_passes_seal_matches_cpu_po2_18() {
    witgen::with_row_passes(true, || seal_matches_cpu("poseidon2", 18));
}

/// The row passes against the generic INVALID fill + zeroize on a raw buffer, including live
/// counts that are not multiples of 4 and the tail fill.
#[test]
fn c11_row_passes_match_generic() {
    use crate::prove::hal::CircuitWitnessGenerator as _;
    use risc0_zkp::{
        field::{baby_bear::BabyBearElem, Elem as _},
        hal::Hal as _,
    };
    let _guard = global_state_test_lock();
    let hal = Arc::new(MetalHalPoseidon2::new());
    let c = MetalCircuitHal::new(hal.clone()).unwrap();
    let rows = 1usize << 10;
    let cols = 7usize;
    let inv = BabyBearElem::INVALID.as_u32_montgomery();
    for live in [0usize, 1, 3, 4, 5, 511, 1000, 1023, 1024] {
        let buf = hal.alloc_elem("rows", rows * cols);
        buf.view_mut(|v| v.fill(BabyBearElem::new(77)));
        c.init_rows(&buf, rows, live).unwrap();
        let v: Vec<u32> = buf.to_vec().iter().map(|e| e.as_u32_montgomery()).collect();
        for (i, &x) in v.iter().enumerate() {
            let want = if i % rows < live { inv } else { 0 };
            assert_eq!(x, want, "init live {live} elem {i}");
        }
        // Witgen-like writes into some live rows, then the tail and the zeroize.
        buf.view_mut(|v| {
            for col in 0..cols {
                for r in (0..live).step_by(3) {
                    v[col * rows + r] = BabyBearElem::new((col * 1000 + r) as u32 + 1);
                }
            }
        });
        let before: Vec<u32> = buf.to_vec().iter().map(|e| e.as_u32_montgomery()).collect();
        let zk = 4usize.min(rows - live);
        c.fill_tail(&buf, rows, zk, BabyBearElem::new(9)).unwrap();
        c.zeroize_rows(&buf, rows, live).unwrap();
        let after: Vec<u32> = buf.to_vec().iter().map(|e| e.as_u32_montgomery()).collect();
        let nine = BabyBearElem::new(9).as_u32_montgomery();
        for i in 0..rows * cols {
            let r = i % rows;
            let want = if r >= rows - zk {
                nine
            } else if r < live && before[i] == inv {
                0
            } else {
                before[i]
            };
            assert_eq!(after[i], want, "live {live} elem {i}");
        }
    }
    assert!(c.init_rows(&hal.alloc_elem("odd", 3 * 6), 6, 1).is_err());
}
