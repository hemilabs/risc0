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

//! C13 (recursion ctrl groups persisted across jobs): the job-shaped A/B harness (ignored; run
//! by hand) and the correctness tests of the persisted ctrl groups.

use std::{sync::Arc, time::Instant};

use risc0_core::field::baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem};
use risc0_zkp::{
    hal::{Buffer as _, Hal},
    prove::poly_group::PolyGroup,
};
use test_log::test;

use super::*;
use crate::prove::witgen::with_noise_seed;

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

fn replay_preflight(c: &capture::Capture) -> Option<Preflight> {
    let p = c.preflight()?;
    if crate::prove::host_chains_backend() {
        *p.host_chains.lock().unwrap() = crate::prove::witgen::host_chains(&c.program, &p);
    }
    Some(p)
}

/// Job-shaped replay: like `c11_bench_captures`, but every round starts from **fresh provers**
/// (new HALs, empty device ctrl caches), as a worker's job does after `release_hal_buffers` /
/// the pre-identity_p254 release dropped the thread's cached provers. Round 0 is the warm pass
/// (PSOs, and with C13 the first host snapshots). `C13_ROUNDS` (default 5) timed rounds.
/// Prints `C13CAP <file> ...` per capture, `C13P254 ...` (the poseidon_254 captures' sum, i.e.
/// identity_p254) and `C11BENCH jobs ...` (the per-round sum, for the A/B driver), plus the
/// device peak of the timed rounds.
#[test]
#[ignore]
fn c13_bench_jobs() {
    let _guard = global_state_test_lock();
    let rounds = env_usize("C13_ROUNDS", 5);
    let caps = load_captures();
    assert!(!caps.is_empty(), "no captures");
    let probe = MetalHalPoseidon2::new();
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); caps.len()];
    let mut sums = Vec::new();
    let mut p254 = Vec::new();
    for round in 0..=rounds {
        if round == 1 {
            probe.reset_peak_allocated_size();
        }
        let mut provers: std::collections::HashMap<String, Box<dyn RecursionProver>> =
            Default::default();
        let (mut sum, mut sum_p254) = (0.0, 0.0);
        for (i, (name, c)) in caps.iter().enumerate() {
            if !provers.contains_key(&c.hashfn) {
                // The HAL's factory (with RISC0_METAL_REC_SEAL_CHECK=1, every proof is
                // re-proved on the CPU and compared).
                provers.insert(c.hashfn.clone(), recursion_prover(&c.hashfn).unwrap());
            }
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
            if round > 0 {
                times[i].push(ms);
                sum += ms;
                if c.hashfn == "poseidon_254" {
                    sum_p254 += ms;
                }
            }
        }
        if round > 0 {
            sums.push(sum);
            p254.push(sum_p254);
        }
    }
    for (i, (name, c)) in caps.iter().enumerate() {
        let t = &mut times[i];
        let (lo, hi) = (
            t.iter().cloned().fold(f64::MAX, f64::min),
            t.iter().cloned().fold(0.0, f64::max),
        );
        println!(
            "C13CAP {name} rows={} median={:.2} min={lo:.2} max={hi:.2}",
            c.program.code_rows(),
            median(t)
        );
    }
    let mm = |v: &mut Vec<f64>| {
        let (lo, hi) = (
            v.iter().cloned().fold(f64::MAX, f64::min),
            v.iter().cloned().fold(0.0, f64::max),
        );
        (median(v), lo, hi)
    };
    let (m, lo, hi) = mm(&mut p254);
    println!("C13P254 median={m:.2} min={lo:.2} max={hi:.2}");
    let (bytes, groups, restores, snaps) = crate::prove::ctrl_persist::stats();
    println!(
        "C13PERSIST enabled={} host_bytes={bytes} groups={groups} restores={restores} snapshots={snaps}",
        crate::prove::ctrl_persist::enabled()
    );
    let (m, lo, hi) = mm(&mut sums);
    println!(
        "C11BENCH jobs n={} median={m:.2} min={lo:.2} max={hi:.2} rounds={rounds} peak={}",
        caps.len(),
        probe.peak_allocated_size()
    );
}

#[test]
fn c13_persist_knob_parses() {
    use crate::prove::ctrl_persist::{mode_wants, parse_persist, Mode};
    assert_eq!(parse_persist(""), Some(Mode::P254));
    assert_eq!(parse_persist("1"), Some(Mode::P254));
    assert_eq!(parse_persist(" On "), Some(Mode::P254));
    assert_eq!(parse_persist("all"), Some(Mode::All));
    assert_eq!(parse_persist("0"), Some(Mode::Off));
    assert_eq!(parse_persist("off"), Some(Mode::Off));
    assert_eq!(parse_persist("maybe"), None);
    let p254 = MetalHalPoseidon254::new().get_hash_suite().name.clone();
    let p2 = MetalHalPoseidon2::new().get_hash_suite().name.clone();
    assert!(mode_wants(Mode::P254, &p254));
    assert!(!mode_wants(Mode::P254, &p2));
    assert!(mode_wants(Mode::All, &p2));
    assert!(!mode_wants(Mode::Off, &p254));
}

/// Commit `program`'s ctrl group on `hal` (as a proof does) and return it.
fn commit_ctrl_group<H>(hal: &H, program: &crate::prove::Program) -> PolyGroup<H>
where
    H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
{
    let ctrl = hal.copy_from_elem("ctrl", &crate::prove::witgen::host_ctrl_for(program));
    let mut prover = risc0_zkp::prove::Prover::new(hal, crate::taps::TAPSET);
    prover.set_po2(program.po2);
    prover.commit_group(crate::REGISTER_GROUP_CTRL, &ctrl);
    prover
        .get_group(crate::REGISTER_GROUP_CTRL)
        .unwrap()
        .clone()
}

/// A group rebuilt from its host copy on another HAL of the same type equals the committed one
/// word for word: coefficients, the re-expanded evaluated domain, every Merkle node, root and
/// top layer (poseidon2 and poseidon_254, po2 14 and 18).
fn group_from_host_bit_exact<H>(new_hal: impl Fn() -> H)
where
    H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
{
    for po2 in [14, 18] {
        let program = crate::prove::zkr::get_zkr("test_recursion_circuit.zkr", po2).unwrap();
        let a = new_hal();
        let group = commit_ctrl_group(&a, &program);
        let host = group.to_host();
        assert_eq!(host.bytes(), group.host_bytes());
        let b = new_hal();
        let rebuilt = PolyGroup::from_host(&b, &host);
        assert!(
            rebuilt.to_host().same_as(&host),
            "po2 {po2}: rebuilt group differs"
        );
        assert_eq!(rebuilt.merkle.root(), group.merkle.root());
        assert!(
            rebuilt.evaluated.as_ref().unwrap().to_vec()
                == group.evaluated.as_ref().unwrap().to_vec(),
            "po2 {po2}: re-expanded evaluated domain differs"
        );
        assert!(
            rebuilt.coeffs.as_ref().unwrap().to_vec() == group.coeffs.as_ref().unwrap().to_vec(),
            "po2 {po2}: coefficients differ"
        );
        eprintln!(
            "C13 {} po2 {po2}: rebuilt ctrl group == committed ({} host bytes)",
            a.get_hash_suite().name,
            host.bytes()
        );
    }
}

#[test]
fn c13_group_from_host_bit_exact_poseidon2() {
    let _guard = global_state_test_lock();
    group_from_host_bit_exact(MetalHalPoseidon2::new);
}

#[test]
fn c13_group_from_host_bit_exact_p254() {
    let _guard = global_state_test_lock();
    group_from_host_bit_exact(MetalHalPoseidon254::new);
}

/// A persisted group is only handed to a program with exactly its code: a program whose code
/// differs only where the cache key does not sample it gets the same key, but no group.
#[test]
fn c13_persisted_group_needs_same_code() {
    use crate::prove::{ctrl_cache_key, ctrl_persist};
    let _guard = global_state_test_lock();
    if !ctrl_persist::enabled() {
        return;
    }
    ctrl_persist::clear();
    let program = crate::prove::zkr::get_zkr("test_recursion_circuit.zkr", 14).unwrap();
    let key = ctrl_cache_key(&program);
    let hal = MetalHalPoseidon2::new();
    let group = commit_ctrl_group(&hal, &program);
    let code = Arc::new(program.code.clone());
    let hashfn = hal.get_hash_suite().name.clone();
    ctrl_persist::store(
        &hashfn,
        key,
        code.clone(),
        Some(&program),
        &group,
        100.0,
        usize::MAX,
    );
    // Same program (pointer and by value) -> the entry's code, and the group.
    let c = ctrl_persist::code_for(key, &program, Some(&program)).unwrap();
    assert!(Arc::ptr_eq(&c, &code));
    let copy = (*program).clone();
    let c = ctrl_persist::code_for(key, &copy, None).unwrap();
    assert!(Arc::ptr_eq(&c, &code));
    assert!(ctrl_persist::lookup(&hashfn, key, &c).is_some());
    // Another hash suite has no group for it.
    assert!(ctrl_persist::lookup("sha-256", key, &c).is_none());
    // One code word changed where the key does not sample it: same key, no group.
    let mut other = (*program).clone();
    let n = other.code.len();
    let stride = (n / 256).max(1);
    let i = (1024..n - 1024).find(|i| i % stride != 0).unwrap();
    other.code[i] += BabyBearElem::new(1);
    assert_eq!(ctrl_cache_key(&other), key);
    assert!(ctrl_persist::code_for(key, &other, None).is_none());
    // A code Arc that is not the entry's (another program's copy) never matches.
    assert!(ctrl_persist::lookup(&hashfn, key, &Arc::new(other.code.clone())).is_none());
    ctrl_persist::clear();
}

/// Eviction is deterministic: Poseidon254 groups first, then larger groups; a group larger than
/// the cap is refused; a larger group is not starved by smaller ones already kept.
#[test]
fn c13_persist_cap_and_eviction() {
    use crate::prove::{ctrl_cache_key, ctrl_persist};
    let _guard = global_state_test_lock();
    if !ctrl_persist::enabled() {
        return;
    }
    ctrl_persist::clear();
    let program = crate::prove::zkr::get_zkr("test_recursion_circuit.zkr", 14).unwrap();
    let key = ctrl_cache_key(&program);
    let code = Arc::new(program.code.clone());
    let hal = MetalHalPoseidon2::new();
    let group = commit_ctrl_group(&hal, &program);
    let bytes = group.host_bytes() + code.len() * 4;
    assert!(!ctrl_persist::would_keep(
        "poseidon2",
        key,
        bytes,
        bytes - 1
    ));
    assert!(ctrl_persist::would_keep("poseidon2", key, bytes, bytes));
    // The real HashSuite names (commit_ctrl passes `hal.get_hash_suite().name`), not the
    // prover-selection strings.
    let p254 = MetalHalPoseidon254::new().get_hash_suite().name.clone();
    let p2 = hal.get_hash_suite().name.clone();
    // Room for one: a Poseidon254 group wins over a larger poseidon2 one (the suite bonus, not
    // the size or key tie-break), whatever the timings.
    let cap = bytes + bytes / 4;
    ctrl_persist::store(&p2, key, code.clone(), None, &group, 50.0, cap);
    assert!(ctrl_persist::would_keep(&p254, key, bytes - 1, cap));
    let small = crate::prove::zkr::get_zkr("test_recursion_circuit.zkr", 13).unwrap();
    let small_key = ctrl_cache_key(&small);
    let small_code = Arc::new(small.code.clone());
    let small_group = commit_ctrl_group(&MetalHalPoseidon254::new(), &small);
    let small_bytes = small_group.host_bytes() + small_code.len() * 4;
    assert!(small_bytes < bytes && small_bytes + bytes > cap);
    assert!(ctrl_persist::would_keep(&p254, small_key, small_bytes, cap));
    ctrl_persist::store(
        &p254,
        small_key,
        small_code.clone(),
        None,
        &small_group,
        1.0,
        cap,
    );
    assert!(ctrl_persist::lookup(&p2, key, &code).is_none());
    assert!(ctrl_persist::lookup(&p254, small_key, &small_code).is_some());
    // A lower-priority group that does not fit next to it is refused.
    assert!(!ctrl_persist::would_keep(&p2, key, bytes, cap));
    // Already kept: not again.
    assert!(!ctrl_persist::would_keep(
        &p254,
        small_key,
        small_bytes,
        usize::MAX
    ));
    ctrl_persist::clear();
    // Same suite: a larger group is kept although the smaller one already held leaves no room
    // for both (it would be evicted: no starvation); a smaller one that does not fit is refused.
    ctrl_persist::store(
        "poseidon2",
        key,
        code.clone(),
        None,
        &group,
        9.0,
        usize::MAX,
    );
    let other = (key.0, key.1, key.2 ^ 1);
    assert!(ctrl_persist::would_keep(
        "poseidon2",
        other,
        2 * bytes,
        2 * bytes + bytes / 2
    ));
    assert!(!ctrl_persist::would_keep(
        "poseidon2",
        other,
        bytes / 2,
        bytes + bytes / 4
    ));
    ctrl_persist::clear();
}

/// Job-shaped proofs: a fresh prover (as after the worker's release) rebuilds the ctrl group
/// the previous prover kept, and its seeded proof equals the CPU prover's (C9 seal check).
fn fresh_prover_seal_matches_cpu(hashfn: &str, po2: usize) {
    use crate::prove::ctrl_persist::{self, Mode};
    let _guard = global_state_test_lock();
    ctrl_persist::clear();
    // `all` (unless the process runs with the knob at 0), so the poseidon2 path is covered too.
    let off = !ctrl_persist::enabled();
    ctrl_persist::set_test_mode(Some(if off { Mode::Off } else { Mode::All }));
    let program = crate::prove::zkr::get_zkr("test_recursion_circuit.zkr", po2).unwrap();
    let input =
        || -> std::collections::VecDeque<u32> { (0..16u32).map(|i| i * 0x101 + 7).collect() };
    let first = metal_recursion_prover(hashfn).unwrap();
    let a = with_noise_seed(3, || first.prove_shared(&program, input(), None)).unwrap();
    drop(first);
    let (_, _, restores0, snaps) = ctrl_persist::stats();
    let second = metal_recursion_prover(hashfn).unwrap();
    let cpu = crate::prove::hal::cpu::recursion_prover(hashfn).unwrap();
    prove_seal_checked(second.as_ref(), cpu.as_ref(), &program, input(), None).unwrap();
    let b = with_noise_seed(3, || second.prove_shared(&program, input(), None)).unwrap();
    assert!(a.seal == b.seal, "committed vs rebuilt ctrl: seals differ");
    let (bytes, groups, restores, _) = ctrl_persist::stats();
    if !off {
        assert!(snaps >= 1, "nothing was kept");
        assert_eq!(
            restores,
            restores0 + 1,
            "the fresh prover did not rebuild the ctrl group"
        );
    } else {
        assert_eq!(restores, restores0);
    }
    eprintln!(
        "C13 {hashfn} po2 {po2}: fresh prover seal == CPU; persisted {groups} groups, {bytes} B"
    );
    ctrl_persist::set_test_mode(None);
    ctrl_persist::clear();
}

#[test]
fn c13_fresh_prover_seal_matches_cpu_poseidon2() {
    fresh_prover_seal_matches_cpu("poseidon2", 18);
}

#[test]
fn c13_fresh_prover_seal_matches_cpu_p254() {
    fresh_prover_seal_matches_cpu("poseidon_254", 16);
}

/// In-process interleaved A/B of the persisted ctrl groups on job-shaped rounds (fresh provers
/// per round, as `c13_bench_jobs`): A = persistence off (the pre-C13 path: every round commits
/// every ctrl), B = on; `C13_PAIRS` (default 7) pairs A B A B .. after a warm round of each.
/// Both arms see the same GPU contention, which a binary A/B on a shared GPU cannot promise.
/// Prints `C13AB <arm> ...` lines (round sum and identity_p254 median/min/max). `C13_AA=1` runs
/// both arms with persistence off (the A-vs-A spread).
#[test]
#[ignore]
fn c13_ab_jobs_in_process() {
    use crate::prove::ctrl_persist::{set_test_mode, Mode};
    let _guard = global_state_test_lock();
    let pairs = env_usize("C13_PAIRS", 7);
    let b_off = env_usize("C13_AA", 0) == 1;
    let caps = load_captures();
    assert!(!caps.is_empty(), "no captures");
    let round = || -> (f64, f64) {
        let mut provers: std::collections::HashMap<String, Box<dyn RecursionProver>> =
            Default::default();
        let (mut sum, mut p254) = (0.0, 0.0);
        for (_, c) in &caps {
            if !provers.contains_key(&c.hashfn) {
                provers.insert(c.hashfn.clone(), recursion_prover(&c.hashfn).unwrap());
            }
            let pf = replay_preflight(c);
            let t = Instant::now();
            let r = provers[&c.hashfn]
                .prove_shared(&c.program, c.input.clone(), pf)
                .unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(r);
            sum += ms;
            if c.hashfn == "poseidon_254" {
                p254 += ms;
            }
        }
        (sum, p254)
    };
    let b_mode = if b_off { Mode::Off } else { Mode::P254 };
    for m in [Mode::Off, b_mode] {
        set_test_mode(Some(m));
        round();
    }
    let (mut a, mut b) = (vec![], vec![]);
    let mut wins = 0;
    for i in 0..pairs {
        set_test_mode(Some(Mode::Off));
        let ra = round();
        set_test_mode(Some(b_mode));
        let rb = round();
        wins += (rb.0 < ra.0) as usize;
        println!(
            "C13PAIR {i} A sum={:.1} p254={:.1} | B sum={:.1} p254={:.1}",
            ra.0, ra.1, rb.0, rb.1
        );
        a.push(ra);
        b.push(rb);
    }
    set_test_mode(None);
    for (arm, v) in [("A(off)", &a), ("B(on)", &b)] {
        let mut s: Vec<f64> = v.iter().map(|x| x.0).collect();
        let mut p: Vec<f64> = v.iter().map(|x| x.1).collect();
        let (s_lo, s_hi) = (
            s.iter().cloned().fold(f64::MAX, f64::min),
            s.iter().cloned().fold(0.0, f64::max),
        );
        let (p_lo, p_hi) = (
            p.iter().cloned().fold(f64::MAX, f64::min),
            p.iter().cloned().fold(0.0, f64::max),
        );
        println!(
            "C13AB {arm} sum median={:.1} [{s_lo:.1}, {s_hi:.1}] p254 median={:.1} [{p_lo:.1}, {p_hi:.1}]",
            median(&mut s),
            median(&mut p)
        );
    }
    println!("C13AB B faster in {wins}/{pairs} pairs");
}

/// In-process interleaved A/B of the eval_check mix-power loads (C13 sub-track b) on the replayed
/// captures with one cached prover per hash suite (`c11_bench_captures`' shape): per proof, A =
/// the C12 four-load lazy split (`ec_lz32_part_*`), B = the uint4 form (`ec_lz_part_*`), so both
/// see the same GPU contention; the order alternates per round (ABBA). `C13_REPS` (default 8)
/// rounds after a warm round of each.
/// `C13_AA=1` runs A in both arms. Prints `C13UAB` lines (per-round sums, B faster count).
#[test]
#[ignore]
fn c13_ab_uint4_in_process() {
    use super::eval_check::test_override::with_variant;
    let _guard = global_state_test_lock();
    let reps = env_usize("C13_REPS", 8);
    let aa = env_usize("C13_AA", 0) == 1;
    let caps = load_captures();
    assert!(!caps.is_empty(), "no captures");
    let mut provers: std::collections::HashMap<String, Box<dyn RecursionProver>> =
        Default::default();
    for (_, c) in &caps {
        if !provers.contains_key(&c.hashfn) {
            provers.insert(c.hashfn.clone(), recursion_prover(&c.hashfn).unwrap());
        }
    }
    let arms = [
        EvalKernels::SplitLzU32,
        if aa {
            EvalKernels::SplitLzU32
        } else {
            EvalKernels::SplitLz
        },
    ];
    let mut sums = [vec![], vec![]];
    // With RISC0_METAL_REC_PROFILE=1: every proof's eval_check phase (ms) per arm.
    let mut ec: [Vec<f64>; 2] = [vec![], vec![]];
    let mut wins = 0;
    let mut n = 0;
    for rep in 0..=reps {
        let mut sum = [0.0f64; 2];
        for (_, c) in &caps {
            // ABBA: the arm that runs second on a capture gains from the first's warm-up (an
            // A-vs-A in fixed order measured the second ~6 % faster), so the order alternates.
            let order: [usize; 2] = if rep % 2 == 0 { [0, 1] } else { [1, 0] };
            for arm in order {
                let v = arms[arm];
                let pf = replay_preflight(c);
                let t = Instant::now();
                let r = with_variant(v, || {
                    provers[&c.hashfn].prove_shared(&c.program, c.input.clone(), pf)
                })
                .unwrap();
                sum[arm] += t.elapsed().as_secs_f64() * 1e3;
                std::hint::black_box(r);
                if rep > 0 && profile::enabled() {
                    if let Some((_, ms)) =
                        profile::entries().iter().find(|(l, _)| *l == "eval_check")
                    {
                        ec[arm].push(*ms);
                    }
                }
            }
        }
        if rep > 0 {
            wins += (sum[1] < sum[0]) as usize;
            n += 1;
            println!("C13UPAIR {rep} A={:.1} B={:.1}", sum[0], sum[1]);
            sums[0].push(sum[0]);
            sums[1].push(sum[1]);
        }
    }
    for (arm, v) in ["A(lz32)", if aa { "B(lz32)" } else { "B(uint4)" }]
        .iter()
        .zip(sums.iter_mut())
    {
        let (lo, hi) = (
            v.iter().cloned().fold(f64::MAX, f64::min),
            v.iter().cloned().fold(0.0, f64::max),
        );
        println!(
            "C13UAB {arm} round median={:.1} [{lo:.1}, {hi:.1}]",
            median(v)
        );
    }
    println!("C13UAB B faster in {wins}/{n} rounds");
    for (arm, v) in ["A", "B"].iter().zip(ec.iter_mut()) {
        if !v.is_empty() {
            let lo = v.iter().cloned().fold(f64::MAX, f64::min);
            let n = v.len();
            let m = median(v);
            println!(
                "C13UEC {arm} eval_check n={n} median={m:.2} min={lo:.2} p25={:.2}",
                v[n / 4]
            );
        }
    }
}
