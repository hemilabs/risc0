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

//! C15 (regrouped eval_check split): the in-process interleaved A/B harness on replayed captures
//! (ignored; run by hand).

use std::time::Instant;

use test_log::test;

use super::*;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

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

/// A preflight for a replay, prepared as zkvm's `prepare()` does (as `c11_bench_captures`).
fn replay_preflight(c: &capture::Capture) -> Option<Preflight> {
    let p = c.preflight()?;
    if crate::prove::host_chains_backend() {
        *p.host_chains.lock().unwrap() = crate::prove::witgen::host_chains(&c.program, &p);
    }
    Some(p)
}

fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[s.len() / 2], s[0], s[s.len() - 1])
}

/// In-process interleaved A/B of two eval_check variants on the replayed captures
/// (`C11_CAPTURES`, optional `C11_ONLY`), one cached prover per hash suite: per capture and round
/// both arms prove it back to back, in ABBA order (the second arm on a capture gains from the
/// first's warm-up), so both see the same GPU contention. `C15_ARMS=<a>,<b>`
/// (`RISC0_METAL_REC_KERNELS` names; default `lz,rg`: the C13 default against C15's) and
/// `C15_ROUNDS` (default 8) rounds after a warm round. Prints per-round sums (`C15PAIR`), their
/// medians (`C15AB`), the per-proof paired median delta, and with `RISC0_METAL_REC_PROFILE=1`
/// the eval_check phase per arm (`C15EC`).
#[test]
#[ignore]
fn c15_ab_eval_in_process() {
    crate::prove::hal::metal::require_heavy_gpu_run("c15_ab_eval_in_process");
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    use super::eval_check::test_override::with_variant;
    let _guard = global_state_test_lock();
    let rounds = env_usize("C15_ROUNDS", 8);
    let spec = std::env::var("C15_ARMS").unwrap_or_else(|_| "lz,rg".into());
    let arms: Vec<EvalKernels> = spec
        .split(',')
        .map(|s| EvalKernels::parse(s).unwrap_or_else(|| panic!("C15_ARMS: unknown {s:?}")))
        .collect();
    assert_eq!(arms.len(), 2, "C15_ARMS=<a>,<b>");
    let caps = load_captures();
    assert!(!caps.is_empty(), "no captures");
    let mut provers: std::collections::HashMap<String, Box<dyn RecursionProver>> =
        Default::default();
    for (_, c) in &caps {
        if !provers.contains_key(&c.hashfn) {
            provers.insert(c.hashfn.clone(), recursion_prover(&c.hashfn).unwrap());
        }
    }
    let mut sums: [Vec<f64>; 2] = [vec![], vec![]];
    let mut ec: [Vec<f64>; 2] = [vec![], vec![]];
    let mut deltas = vec![];
    let mut wins = 0;
    for round in 0..=rounds {
        let mut sum = [0.0f64; 2];
        for (_, c) in &caps {
            let order: [usize; 2] = if round % 2 == 0 { [0, 1] } else { [1, 0] };
            let mut ms = [0.0f64; 2];
            for arm in order {
                let pf = replay_preflight(c);
                let t = Instant::now();
                let r = with_variant(arms[arm], || {
                    provers[&c.hashfn].prove_shared(&c.program, c.input.clone(), pf)
                })
                .unwrap();
                ms[arm] = t.elapsed().as_secs_f64() * 1e3;
                std::hint::black_box(r);
                if round > 0 && profile::enabled() {
                    if let Some((_, e)) =
                        profile::entries().iter().find(|(l, _)| *l == "eval_check")
                    {
                        ec[arm].push(*e);
                    }
                }
            }
            sum[0] += ms[0];
            sum[1] += ms[1];
            if round > 0 {
                deltas.push(ms[1] - ms[0]);
            }
        }
        if round > 0 {
            wins += (sum[1] < sum[0]) as usize;
            println!("C15PAIR {round} A={:.1} B={:.1}", sum[0], sum[1]);
            sums[0].push(sum[0]);
            sums[1].push(sum[1]);
        }
    }
    for (i, v) in sums.iter().enumerate() {
        let (m, lo, hi) = stats(v);
        println!(
            "C15AB {} ({:?}) round median={m:.1} [{lo:.1}, {hi:.1}]",
            ["A", "B"][i],
            arms[i]
        );
    }
    let (dm, dlo, dhi) = stats(&deltas);
    println!(
        "C15AB B faster in {wins}/{rounds} rounds; per proof B-A median={dm:+.2} [{dlo:+.2}, \
         {dhi:+.2}] ms over {} proofs",
        deltas.len()
    );
    for (i, v) in ec.iter().enumerate() {
        if !v.is_empty() {
            let (m, lo, _) = stats(v);
            println!(
                "C15EC {} eval_check n={} median={m:.2} min={lo:.2}",
                ["A", "B"][i],
                v.len()
            );
        }
    }
}

/// PSO creation time of eval_check variants (run with `MTL_SHADER_CACHE_SIZE=0` for cold, K5's
/// method): `C15_PSO_SETS` (default `lz,rg`), each built as the HAL builds it (on
/// `RISC0_METAL_REC_PSO_THREADS` threads). Builds private `Pipelines`, not the process statics.
#[test]
#[ignore]
fn c15_pso_times() {
    crate::prove::hal::metal::require_heavy_gpu_run("c15_pso_times");
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    let _guard = global_state_test_lock();
    let device = metal::Device::system_default().unwrap();
    let sets = std::env::var("C15_PSO_SETS").unwrap_or_else(|_| "lz,rg".into());
    for set in sets.split(',') {
        let v = EvalKernels::parse(set).unwrap_or_else(|| panic!("C15_PSO_SETS: unknown {set:?}"));
        let t = Instant::now();
        Pipelines::new(&device, METAL_LIB, v.names(), pso_threads()).unwrap();
        println!("C15PSO {set}: {:.2} s", t.elapsed().as_secs_f64());
    }
}

/// A kernel arm of [c15_eval_kernel_bench]: an [EvalKernels] name, or `lib=<metallib>` (the
/// `ec_rg_part_*` kernels of another metallib in index order, e.g. a regrouping under test built
/// offline with `cuda2metal_eval.py --compile --metallib`). Returns the dispatch order and PSOs.
fn kernel_arm(device: &metal::DeviceRef, spec: &str) -> (Vec<&'static str>, &'static Pipelines) {
    let t0 = Instant::now();
    let (names, k): (Vec<&'static str>, Pipelines) = if let Some(path) = spec.strip_prefix("lib=") {
        let bytes: &'static [u8] = Box::leak(std::fs::read(path).unwrap().into_boxed_slice());
        let lib = device.new_library_with_data(bytes).unwrap();
        let mut parts: Vec<(usize, &'static str)> = lib
            .function_names()
            .into_iter()
            .filter_map(|n| {
                let i = n.strip_prefix("ec_rg_part_")?.parse().ok()?;
                Some((i, &*Box::leak(n.into_boxed_str())))
            })
            .collect();
        parts.sort();
        assert!(
            parts.iter().enumerate().all(|(i, (j, _))| i == *j),
            "{path}: part gaps"
        );
        let names: Vec<&'static str> = parts.into_iter().map(|(_, n)| n).collect();
        let k = Pipelines::new(device, bytes, &names, 8).unwrap();
        (names, k)
    } else {
        let v = EvalKernels::parse(spec).unwrap_or_else(|| panic!("C15_KB_ARMS: unknown {spec:?}"));
        (
            v.names().to_vec(),
            Pipelines::new(device, METAL_LIB, v.names(), 8).unwrap(),
        )
    };
    println!(
        "C15KB PSOs {spec}: {} parts, {:.2} s",
        names.len(),
        t0.elapsed().as_secs_f64()
    );
    (names, Box::leak(Box::new(k)))
}

/// eval_check kernel A/B on random canonical inputs (C12's method, in process): every arm of
/// `C15_KB_ARMS` (default `lz,rg,lz,rg`; an arm given twice is an A-vs-A control) once per round,
/// order rotated each round, `C15_KB_ROUNDS` (15) rounds after a warm round, po2 `C15_KB_PO2`
/// (18). Each arm's output must equal the CPU eval_check. With `RISC0_METAL_PROFILE=1` every
/// dispatch is its own command buffer and the GPU time per part is printed (`C15KBP`); else the
/// encode+sync wall time. Prints per-arm median [min, max] (`C15KB`) and per-round deltas against
/// the first arm (`C15KBD`).
#[test]
#[ignore]
fn c15_eval_kernel_bench() {
    crate::prove::hal::metal::require_heavy_gpu_run("c15_eval_kernel_bench");
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    use risc0_zkp::{
        core::log2_ceil,
        field::{
            baby_bear::{BabyBearElem, BabyBearExtElem},
            map_pow, ExtElem as _, RootsOfUnity as _,
        },
        hal::{metal::MetalHalPoseidon2, Hal as _},
        INV_RATE,
    };
    let _guard = global_state_test_lock();
    let po2 = env_usize("C15_KB_PO2", 18);
    let rounds = env_usize("C15_KB_ROUNDS", 15);
    let spec = std::env::var("C15_KB_ARMS").unwrap_or_else(|_| "lz,rg,lz,rg".into());
    let specs: Vec<String> = spec.split(',').map(|s| s.trim().to_string()).collect();
    let prof = std::env::var(risc0_zkp::hal::metal::PROFILE_ENV).is_ok_and(|v| v == "1");
    let hal = std::sync::Arc::new(MetalHalPoseidon2::new());
    let c = MetalCircuitHal::new(hal.clone()).unwrap();
    let mut built: Vec<(String, (Vec<&'static str>, &'static Pipelines))> = Vec::new();
    for s in &specs {
        if !built.iter().any(|(n, _)| n == s) {
            built.push((s.clone(), kernel_arm(&hal.device, s)));
        }
    }
    let arms: Vec<(Vec<&'static str>, &'static Pipelines)> = specs
        .iter()
        .map(|s| built.iter().find(|(n, _)| n == s).unwrap().1.clone())
        .collect();
    let p = crate::testutil::EvalCheckParams::new(po2);
    let pows = map_pow(p.poly_mix, crate::info::POLY_MIX_POWERS);
    let rou = BabyBearElem::ROU_FWD[po2 + log2_ceil(INV_RATE)];
    let want = super::eval_check::cpu_eval_check(
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
    let n = arms.len();
    let mut tot: Vec<Vec<f64>> = vec![vec![]; n];
    let mut parts: Vec<std::collections::BTreeMap<&'static str, Vec<f64>>> =
        vec![Default::default(); n];
    for round in 0..=rounds {
        for j in 0..n {
            let a = (j + round) % n;
            let (names, kernels) = (&arms[a].0, arms[a].1);
            check.view_mut(|v| v.fill(BabyBearElem::new(12345)));
            hal.sync();
            let _ = risc0_zkp::hal::metal::profile_take();
            let t0 = Instant::now();
            c.encode_eval_parts(
                names, kernels, &check, &groups, &globals, &pows, po2, p.steps,
            );
            hal.sync();
            let wall = t0.elapsed().as_secs_f64() * 1e3;
            let entries = risc0_zkp::hal::metal::profile_take();
            if round == 0 {
                assert!(check.to_vec() == want, "{}: differs from the CPU", specs[a]);
                continue;
            }
            if prof {
                let mut sum = 0.0;
                for (name, e) in entries {
                    if name.starts_with("ec_") || name == "eval_check" {
                        sum += e.gpu_secs * 1e3;
                        parts[a].entry(name).or_default().push(e.gpu_secs * 1e3);
                    }
                }
                tot[a].push(sum);
            } else {
                tot[a].push(wall);
            }
        }
        if round > 0 {
            let r: Vec<String> = (0..n)
                .map(|a| format!("{}={:.2}", specs[a], tot[a].last().unwrap()))
                .collect();
            println!("C15KBR {round} {}", r.join(" "));
        }
    }
    let unit = if prof { "GPU ms" } else { "wall ms" };
    for a in 0..n {
        let (m, lo, hi) = stats(&tot[a]);
        println!(
            "C15KB po2 {po2} {} [{a}] {unit} median={m:.2} [{lo:.2}, {hi:.2}]",
            specs[a]
        );
        if prof {
            let v: Vec<String> = parts[a]
                .iter()
                .map(|(k, v)| format!("{k}={:.2}", stats(v).0))
                .collect();
            println!("C15KBP {} [{a}] {}", specs[a], v.join(" "));
        }
    }
    for a in 1..n {
        let d: Vec<f64> = (0..rounds).map(|r| tot[a][r] - tot[0][r]).collect();
        let faster = d.iter().filter(|&&x| x < 0.0).count();
        let (m, lo, hi) = stats(&d);
        println!(
            "C15KBD {} [{a}] - {} [0]: median={m:+.2} [{lo:+.2}, {hi:+.2}] ms, faster in {faster}/{rounds}",
            specs[a], specs[0]
        );
    }
}
