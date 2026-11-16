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
    crate::prove::hal::metal::require_heavy_gpu_run("c11_bench_proofs");
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
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

/// A preflight for a replay, prepared as zkvm's `prepare()` does (trace, host chains and the C14
/// Metal staging off the timed path).
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
    // C14: and staged in a shared Metal buffer, as `prepare()` does (`RISC0_METAL_REC_STAGE`).
    crate::prove::stage_metal(&p);
    Some(p)
}

/// Replay real lift/join/resolve/identity_p254 proofs (`C11_CAPTURES`, optional `C11_ONLY`
/// substring filter) `C11_REPS` times after a warm pass. One `C11CAP <file> ...` line per
/// capture (median/min/max ms) and a `C11BENCH captures ...` line with the per-round sum.
#[test]
#[ignore]
fn c11_bench_captures() {
    crate::prove::hal::metal::require_heavy_gpu_run("c11_bench_captures");
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
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
    crate::prove::hal::metal::require_heavy_gpu_run("c11_captures_seal_match_cpu");
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
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
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    seal_matches_cpu("poseidon2", 16);
}

#[test]
fn c11_seal_matches_cpu_poseidon2_po2_18() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    seal_matches_cpu("poseidon2", 18);
}

#[test]
fn c11_seal_matches_cpu_sha256_po2_16() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    seal_matches_cpu("sha-256", 16);
}

/// Cold/warm PSO creation per kernel set (run with `MTL_SHADER_CACHE_SIZE=0` for cold, K5's
/// method): base kernels, monolith eval_check, split eval_check, sequential (as loaded today) and
/// the split parts on parallel threads. Builds private `MetalKernels`, not the process statics.
#[test]
#[ignore]
fn c11_pso_times() {
    crate::prove::hal::metal::require_heavy_gpu_run("c11_pso_times");
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
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
    crate::prove::hal::metal::require_heavy_gpu_run("c11_eval_tg_sweep");
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
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
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    witgen::with_row_passes(true, || seal_matches_cpu("poseidon2", 16));
}

#[test]
fn c11_row_passes_seal_matches_cpu_po2_18() {
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    witgen::with_row_passes(true, || seal_matches_cpu("poseidon2", 18));
}

/// The row passes against the generic INVALID fill + zeroize on a raw buffer, including live
/// counts that are not multiples of 4, with the ZK tail written as on the prover (P31-09): a copy
/// of independent values (`eltwise_copy_elem_slice`), never a fill.
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
        let tail: Vec<BabyBearElem> = (0..zk * cols)
            .map(|i| BabyBearElem::new(9 + i as u32))
            .collect();
        hal.eltwise_copy_elem_slice(&buf, &tail, cols, zk, 0, zk, rows - zk, rows);
        c.zeroize_rows(&buf, rows, live).unwrap();
        let after: Vec<u32> = buf.to_vec().iter().map(|e| e.as_u32_montgomery()).collect();
        for i in 0..rows * cols {
            let (col, r) = (i / rows, i % rows);
            let want = if r >= rows - zk {
                tail[col * zk + r - (rows - zk)].as_u32_montgomery()
            } else if r < live && before[i] == inv {
                0
            } else {
                before[i]
            };
            assert_eq!(after[i], want, "live {live} elem {i}");
        }
    }
    assert!(c.init_rows(&hal.alloc_elem("odd", 3 * 6), 6, 1).is_err());
    // The Metal HAL has no tail fill (P31-09): the trait default refuses.
    let buf = hal.alloc_elem("fill", rows * cols);
    assert!(c.fill_tail(&buf, rows, 4, BabyBearElem::new(9)).is_err());
}

// ---------------------------------------------------------------------------------------------
// P31-09: independent recursion ZK noise (3.1 `f68f86020`) in every Metal path.

mod p31_09 {
    use std::collections::HashSet;

    use test_log::test;

    use risc0_zkp::{
        adapter::{CircuitInfo as _, TapsProvider as _},
        core::hash::poseidon2::Poseidon2HashSuite,
        field::{
            baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
            Elem as _,
        },
        hal::{cpu::CpuHal, Buffer as _, Hal},
        ZK_CYCLES,
    };

    use super::*;
    use crate::{
        prove::{
            hal::{cpu::CpuCircuitHal, CircuitAccumulator, CircuitWitnessGenerator},
            witgen::{self as wg, WitnessGenerator},
        },
        CircuitImpl, CIRCUIT,
    };

    const PO2: usize = 16;

    fn preflight(program: &Program) -> Preflight {
        let mut pf = Preflight::new(test_input());
        for (cycle, row) in program.code_by_row().enumerate() {
            pf.step(cycle, row).unwrap();
        }
        pf
    }

    /// `data` and `accum` (raw words) after the witgen and the accum of one proof, as the
    /// prover leaves them before committing the accum (noise in, zeroized).
    fn witness<H, C>(hal: &H, c: &C, program: &Program) -> (Vec<u32>, Vec<u32>)
    where
        H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
        C: CircuitWitnessGenerator<H> + CircuitAccumulator<H>,
    {
        let pf = preflight(program);
        let w = WitnessGenerator::new(hal, c, program, &pf, None).unwrap();
        let mix: Vec<BabyBearElem> = (0..CircuitImpl::MIX_SIZE as u32)
            .map(|i| BabyBearElem::new(i * 7919 + 3))
            .collect();
        w.accum(hal, c, &mix).unwrap();
        let raw = |v: Vec<BabyBearElem>| v.iter().map(|e| e.as_u32_montgomery()).collect();
        (raw(w.data.to_vec()), raw(w.accum.to_vec()))
    }

    fn metal_witness(row_passes: bool, program: &Program) -> (Vec<u32>, Vec<u32>) {
        let hal = Arc::new(MetalHalPoseidon2::new());
        let c = MetalCircuitHal::new(hal.clone()).unwrap();
        witgen::with_row_passes(row_passes, || witness(hal.as_ref(), &c, program))
    }

    fn cpu_witness(program: &Program) -> (Vec<u32>, Vec<u32>) {
        let hal: CpuHal<BabyBear> = CpuHal::new(Poseidon2HashSuite::new_suite());
        witness(&hal, &CpuCircuitHal, program)
    }

    /// The last ZK_CYCLES rows of every one of `cols` columns, column by column.
    fn tail(v: &[u32], cols: usize) -> Vec<u32> {
        let rows = 1usize << PO2;
        assert_eq!(v.len(), cols * rows);
        (0..cols)
            .flat_map(|c| {
                v[c * rows + rows - ZK_CYCLES..(c + 1) * rows]
                    .iter()
                    .copied()
            })
            .collect()
    }

    /// `v` holds `ZK_CYCLES * cols` noise cells with independent values: as many distinct values
    /// as uniform draws give (n(n-1)/2p expected colliding pairs: ~4 for the 131072 data cells),
    /// no INVALID, and every column has more than one value. A repeated noise value (the bug
    /// 3.1 fixed) gives 1 distinct value.
    fn assert_independent(what: &str, v: &[u32], cols: usize) -> usize {
        let n = v.len();
        assert_eq!(n, ZK_CYCLES * cols, "{what}: noise cell count");
        let inv = BabyBearElem::INVALID.as_u32_montgomery();
        assert!(v.iter().all(|&x| x != inv), "{what}: INVALID in the noise");
        let distinct = v.iter().collect::<HashSet<_>>().len();
        let p = 2013265921f64;
        let expect = n as f64 * (n as f64 - 1.0) / (2.0 * p);
        let bound = (8.0 * expect).ceil() as usize + 16;
        assert!(distinct > 1, "{what}: one repeated noise value");
        assert!(
            n - distinct <= bound,
            "{what}: {n} cells, only {distinct} distinct (expected ~{expect:.1} collisions)"
        );
        for c in 0..cols {
            let col = &v[c * ZK_CYCLES..(c + 1) * ZK_CYCLES];
            let d = col.iter().collect::<HashSet<_>>().len();
            assert!(
                d + 2 >= ZK_CYCLES,
                "{what}: column {c} has {d} distinct values"
            );
        }
        eprintln!(
            "P31-09 {what}: {n} noise cells ({cols} cols x {ZK_CYCLES} rows), {distinct} distinct \
             ({} repeats, ~{expect:.2} expected)",
            n - distinct
        );
        distinct
    }

    /// The distinctness check is not vacuous: 3.0's noise (one value repeated over the tail) and a
    /// per-column repeat fail it. Host only.
    #[test]
    fn p31_09_repeated_noise_is_caught() {
        let cols = 12;
        let n = ZK_CYCLES * cols;
        let x = BabyBearElem::new(1234).as_u32_montgomery();
        let one = std::panic::catch_unwind(|| assert_independent("repeated", &vec![x; n], cols));
        assert!(one.is_err(), "one repeated value must fail");
        let per_col: Vec<u32> = (0..n).map(|i| (i / ZK_CYCLES) as u32 + 1).collect();
        let col = std::panic::catch_unwind(|| assert_independent("per column", &per_col, cols));
        assert!(col.is_err(), "a repeated value per column must fail");
        let ok: Vec<u32> = (0..n as u32).map(|i| i * 3 + 1).collect();
        assert_eq!(assert_independent("distinct", &ok, cols), n);
    }

    /// Unseeded Metal proofs (default noise, `rand::rng()`): the data and accum noise rows hold
    /// independent values, the two tails share none beyond chance, and two proofs differ.
    fn noise_rows_independent(row_passes: bool) {
        let _heavy = crate::prove::hal::metal::heavy_test_lock();
        let _guard = global_state_test_lock();
        let program = get_zkr("test_recursion_circuit.zkr", PO2).unwrap();
        let mode = if row_passes { "row passes" } else { "generic" };
        let (d, a) = metal_witness(row_passes, &program);
        let (dt, at) = (
            tail(&d, CIRCUIT.data_size()),
            tail(&a, CIRCUIT.accum_size()),
        );
        let nd = assert_independent(&format!("{mode} data"), &dt, CIRCUIT.data_size());
        let na = assert_independent(&format!("{mode} accum"), &at, CIRCUIT.accum_size());
        let both: HashSet<_> = dt.iter().chain(at.iter()).collect();
        assert!(
            both.len() + 16 >= nd + na,
            "{mode}: data and accum noise overlap"
        );
        // Fresh noise per proof.
        let (d2, a2) = metal_witness(row_passes, &program);
        let (dt2, at2) = (
            tail(&d2, CIRCUIT.data_size()),
            tail(&a2, CIRCUIT.accum_size()),
        );
        let same = dt.iter().zip(&dt2).filter(|(x, y)| x == y).count()
            + at.iter().zip(&at2).filter(|(x, y)| x == y).count();
        assert!(same < 16, "{mode}: two proofs share {same} noise cells");
        // Everything but the noise is the same witness.
        let rows = 1usize << PO2;
        let live = |i: usize| i % rows < rows - ZK_CYCLES;
        assert!(d
            .iter()
            .zip(&d2)
            .enumerate()
            .all(|(i, (x, y))| !live(i) || x == y));
        assert!(a
            .iter()
            .zip(&a2)
            .enumerate()
            .all(|(i, (x, y))| !live(i) || x == y));
    }

    #[test]
    fn p31_09_noise_rows_independent_generic() {
        noise_rows_independent(false);
    }

    #[test]
    fn p31_09_noise_rows_independent_row_passes() {
        noise_rows_independent(true);
    }

    /// The stream [crate::prove::witgen::with_noise_seed] must give, recomputed here from
    /// the stock `BabyBearElem::random` (n draws in order from the seeded `StdRng`).
    fn seeded_stream(seed: u64, stream: u64, n: usize) -> Vec<u32> {
        use rand::SeedableRng as _;
        let mut rng =
            rand::rngs::StdRng::seed_from_u64(seed ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        (0..n)
            .map(|_| BabyBearElem::random(&mut rng).as_u32_montgomery())
            .collect()
    }

    /// Under a noise seed (RISC0_HIP_REC_ZCHECK style, on the whole proof witness): Metal with the
    /// row passes == Metal generic == CPU HAL, for `data` and `accum` element for element, with
    /// the noise drawn ahead (default) or in place; the tails hold the seeded stream in the
    /// layout `noise[col * ZK_CYCLES + r]` -> row `rows - ZK_CYCLES + r` (the stock 3.1 copy).
    #[test]
    fn p31_09_seeded_noise_row_passes_eq_generic_eq_cpu() {
        let _heavy = crate::prove::hal::metal::heavy_test_lock();
        let _guard = global_state_test_lock();
        let program = get_zkr("test_recursion_circuit.zkr", PO2).unwrap();
        let seed = 0x5eed_0931;
        let run = |f: &dyn Fn() -> (Vec<u32>, Vec<u32>), overlap: bool| {
            wg::with_noise_overlap(overlap, || with_noise_seed(seed, f))
        };
        let early0 = wg::EARLY_NOISE_USED.load(std::sync::atomic::Ordering::Relaxed);
        let generic = run(&|| metal_witness(false, &program), true);
        let rowp = run(&|| metal_witness(true, &program), true);
        let cpu = run(&|| cpu_witness(&program), true);
        let early1 = wg::EARLY_NOISE_USED.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(early1 - early0, 3, "noise drawn ahead on the helper thread");
        let inplace = run(&|| metal_witness(true, &program), false);
        let cpu_inplace = run(&|| cpu_witness(&program), false);
        assert_eq!(
            wg::EARLY_NOISE_USED.load(std::sync::atomic::Ordering::Relaxed),
            early1,
            "overlap off draws in place"
        );
        let first_diff = |a: &[u32], b: &[u32]| a.iter().zip(b).position(|(x, y)| x != y);
        for (name, other) in [
            ("row passes", &rowp),
            ("cpu", &cpu),
            ("row passes, in place", &inplace),
            ("cpu, in place", &cpu_inplace),
        ] {
            assert_eq!(
                first_diff(&generic.0, &other.0),
                None,
                "data: generic vs {name}"
            );
            assert_eq!(
                first_diff(&generic.1, &other.1),
                None,
                "accum: generic vs {name}"
            );
        }
        let (dt, at) = (
            tail(&generic.0, CIRCUIT.data_size()),
            tail(&generic.1, CIRCUIT.accum_size()),
        );
        assert_eq!(dt, seeded_stream(seed, 1, ZK_CYCLES * CIRCUIT.data_size()));
        assert_eq!(at, seeded_stream(seed, 2, ZK_CYCLES * CIRCUIT.accum_size()));
        assert_independent("seeded data", &dt, CIRCUIT.data_size());
        assert_independent("seeded accum", &at, CIRCUIT.accum_size());
        // A different seed gives different noise.
        let other = wg::with_noise_overlap(true, || {
            with_noise_seed(seed + 1, || metal_witness(false, &program))
        });
        assert_ne!(tail(&other.0, CIRCUIT.data_size()), dt);
        eprintln!("P31-09 seeded: Metal row passes == Metal generic == CPU (data + accum), drawn ahead and in place");
    }

    /// C9 seal equality (Metal == CPU under `with_noise_seed`) with the row passes forced off
    /// (the `c11_row_passes_seal_matches_cpu_*` tests force them on).
    #[test]
    fn p31_09_seal_matches_cpu_row_passes_off_po2_16() {
        let _heavy = crate::prove::hal::metal::heavy_test_lock();
        witgen::with_row_passes(false, || super::seal_matches_cpu("poseidon2", 16));
    }

    #[test]
    fn p31_09_seal_matches_cpu_row_passes_off_po2_18() {
        let _heavy = crate::prove::hal::metal::heavy_test_lock();
        witgen::with_row_passes(false, || super::seal_matches_cpu("poseidon2", 18));
    }

    /// The same with the noise drawn in place (`RISC0_RECURSION_NOISE_OVERLAP=0`), both modes.
    #[test]
    fn p31_09_seal_matches_cpu_noise_in_place() {
        let _heavy = crate::prove::hal::metal::heavy_test_lock();
        for rp in [false, true] {
            wg::with_noise_overlap(false, || {
                witgen::with_row_passes(rp, || super::seal_matches_cpu("poseidon2", 16))
            });
        }
    }
}
