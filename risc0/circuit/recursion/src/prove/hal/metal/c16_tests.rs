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

//! C16 (b): `prewarm_p254_ctrl`, identity_p254's ctrl group kept before the first wrap of a
//! process. Correctness tests, and the first-job harness (ignored; one process per arm).

use std::time::Instant;

use risc0_zkp::hal::Hal;
use test_log::test;

use super::*;
use crate::prove::{
    ctrl_cache_key,
    ctrl_persist::{self, Mode},
    prewarm_p254_ctrl, P254Prewarm,
};

/// The recursion programs' po2 (risc0-zkvm's `RECURSION_PO2`).
const PO2: usize = 18;

/// Restores the C13 test mode and drops the kept groups when a test ends (also on failure).
struct ModeGuard;
impl Drop for ModeGuard {
    fn drop(&mut self) {
        ctrl_persist::set_test_mode(None);
        ctrl_persist::clear();
    }
}

/// The prewarm commits identity_p254's ctrl on a temporary prover and keeps exactly the group a
/// commit on a fresh HAL builds (bit for bit); a second call finds it kept; a new prover then
/// rebuilds it instead of committing it.
#[test]
fn c16_prewarm_p254_kept_then_restored() {
    let _guard = global_state_test_lock();
    let _mode = ModeGuard;
    ctrl_persist::clear();
    ctrl_persist::set_test_mode(Some(Mode::P254));
    let r = prewarm_p254_ctrl(PO2).unwrap();
    if std::env::var(ctrl_persist::PREWARM_ENV).is_ok_and(|v| v.trim() == "0") {
        assert!(matches!(r, P254Prewarm::Skipped(_)), "{r:?}");
        return;
    }
    let P254Prewarm::Kept { ms, bytes } = r else {
        panic!("prewarm did not keep the group: {r:?}");
    };
    eprintln!("C16 prewarm: kept {bytes} B in {ms:.1} ms");
    assert_eq!(prewarm_p254_ctrl(PO2).unwrap(), P254Prewarm::AlreadyKept);

    // The kept copy equals a commit of the same program on a fresh HAL.
    let program = crate::prove::zkr::get_zkr("identity.zkr", PO2).unwrap();
    let key = ctrl_cache_key(&program);
    let hal = MetalHalPoseidon254::new();
    let hashfn = hal.get_hash_suite().name.clone();
    let code = ctrl_persist::code_for(key, &program, Some(&program)).expect("kept code");
    let kept = ctrl_persist::lookup(&hashfn, key, &code).expect("kept group");
    let ctrl = hal.copy_from_elem("ctrl", &crate::prove::witgen::host_ctrl_for(&program));
    let mut prover = risc0_zkp::prove::Prover::new(&hal, crate::taps::TAPSET);
    prover.set_po2(program.po2);
    prover.commit_group(crate::REGISTER_GROUP_CTRL, &ctrl);
    let fresh = prover
        .get_group(crate::REGISTER_GROUP_CTRL)
        .unwrap()
        .to_host();
    assert!(
        fresh.same_as(&kept),
        "prewarmed group differs from a fresh commit"
    );
    assert_eq!(kept.bytes() + code.len() * 4, bytes);

    // A new p254 prover (what the first wrap builds) rebuilds it.
    let (_, _, restores0, snaps0) = ctrl_persist::stats();
    let p = crate::prove::recursion_prover("poseidon_254").unwrap();
    p.prewarm(&program).unwrap();
    let (_, groups, restores, snaps) = ctrl_persist::stats();
    assert_eq!(
        restores,
        restores0 + 1,
        "the new prover did not rebuild the kept group"
    );
    assert_eq!(
        snaps, snaps0,
        "the new prover committed and snapshotted again"
    );
    assert_eq!(groups, 1);
}

/// With persistence off nothing is committed or kept.
#[test]
fn c16_prewarm_p254_skipped_when_persist_off() {
    let _guard = global_state_test_lock();
    let _mode = ModeGuard;
    ctrl_persist::clear();
    ctrl_persist::set_test_mode(Some(Mode::Off));
    let r = prewarm_p254_ctrl(PO2).unwrap();
    assert!(matches!(r, P254Prewarm::Skipped(_)), "{r:?}");
    let (bytes, groups, _, _) = ctrl_persist::stats();
    assert_eq!((bytes, groups), (0, 0));
}

/// First identity_p254 of a process (run once per process, interleave the arms by hand): a
/// poseidon2 capture first (the lifts and joins before the wrap warm the recursion pipelines),
/// then identity_p254 on a fresh prover, after `metal_warmup` (the worker's start-up pipeline build). `C16_PREWARM=1` calls `prewarm_p254_ctrl` at the start
/// (as a worker would at start-up) and prints its outcome. `C11_CAPTURES` (the C11 capture dir)
/// must hold `0000_poseidon2*` and `0005_poseidon_254*`. Prints `C16FIRST prewarm=.. ms=..`.
/// With `RISC0_METAL_REC_SEAL_CHECK=1` the proofs are checked against the CPU prover.
#[test]
#[ignore]
fn c16_first_job() {
    let _guard = global_state_test_lock();
    let dir = std::env::var("C11_CAPTURES").expect("C11_CAPTURES=<capture dir>");
    let read = |prefix: &str| {
        let p = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.file_name().unwrap().to_string_lossy().starts_with(prefix))
            .unwrap_or_else(|| panic!("no {prefix} capture in {dir}"));
        capture::read(&p).unwrap()
    };
    let prewarm = std::env::var("C16_PREWARM").is_ok_and(|v| v == "1");
    // The worker builds the recursion and zkp pipelines at start-up (`metal_warmup`).
    let t = Instant::now();
    crate::prove::metal_warmup();
    let warmup_ms = t.elapsed().as_secs_f64() * 1e3;
    let probe = MetalHalPoseidon2::new();
    let before = probe.allocated_size();
    probe.reset_peak_allocated_size();
    let t = Instant::now();
    let outcome = prewarm.then(|| prewarm_p254_ctrl(PO2).unwrap());
    let prewarm_ms = t.elapsed().as_secs_f64() * 1e3;
    let (peak, after) = (probe.peak_allocated_size(), probe.allocated_size());
    let replay = |c: &capture::Capture| {
        let pf = c.preflight().inspect(|p| {
            if crate::prove::host_chains_backend() {
                *p.host_chains.lock().unwrap() = crate::prove::witgen::host_chains(&c.program, p);
            }
        });
        let prover = recursion_prover(&c.hashfn).unwrap();
        let t = Instant::now();
        let r = prover
            .prove_shared(&c.program, c.input.clone(), pf)
            .unwrap();
        std::hint::black_box(r);
        t.elapsed().as_secs_f64() * 1e3
    };
    let lift_ms = replay(&read("0000_poseidon2"));
    let p254_ms = replay(&read("0005_poseidon_254"));
    println!(
        "C16FIRST prewarm={prewarm} warmup_ms={warmup_ms:.1} outcome={outcome:?} prewarm_ms={prewarm_ms:.1} \
         device_before={before} prewarm_peak={peak} device_after={after} lift_ms={lift_ms:.1} \
         p254_ms={p254_ms:.1}"
    );
}

/// The GPU's "Device Utilization %" (IORegistry), or None (as zkp's tests_z9).
fn gpu_utilization() -> Option<u32> {
    let out = std::process::Command::new("ioreg")
        .args(["-r", "-d", "1", "-c", "IOAccelerator"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let key = "\"Device Utilization %\"=";
    let i = text.find(key)? + key.len();
    text[i..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

/// With `C16_QUIET=1`, wait (up to 60 s) until two readings 250 ms apart show the GPU below 50 %
/// busy: other agents' GPU work otherwise inflates single samples several-fold.
fn wait_gpu_quiet() {
    if !std::env::var("C16_QUIET").is_ok_and(|v| v == "1") {
        return;
    }
    let t0 = Instant::now();
    let mut quiet = 0;
    while t0.elapsed().as_secs() < 60 && quiet < 2 {
        match gpu_utilization() {
            Some(u) if u >= 50 => quiet = 0,
            _ => quiet += 1,
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

fn load_avg() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// In-process A/B of Poseidon254 kernel settings on whole identity_p254 proofs: each proof gets
/// a fresh p254 prover (as a worker's wrap does; its ctrl group is restored from C13's copy), built
/// after the arm's environment is applied (`C16_A` / `C16_B`: `;`-separated `NAME=value`, read by
/// the HAL when it is created; an empty value unsets). `C16_PAIRS` (default 9) pairs in ABBA
/// order after one warm proof per arm. Prints `C16AB` lines: per-arm median / min / max and the
/// per-pair B - A deltas. `C11_CAPTURES` must hold `0005_poseidon_254*`. `C16_QUIET=1` waits for
/// a quiet GPU before each pair.
#[test]
#[ignore]
fn c16_ab_p254_in_process() {
    let _guard = global_state_test_lock();
    let dir = std::env::var("C11_CAPTURES").expect("C11_CAPTURES=<capture dir>");
    let path = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("0005_poseidon_254")
        })
        .expect("no identity_p254 capture");
    let c = capture::read(&path).unwrap();
    let arms: Vec<Vec<(String, String)>> = ["C16_A", "C16_B"]
        .iter()
        .map(|k| {
            std::env::var(k)
                .unwrap_or_default()
                .split(';')
                .filter(|kv| !kv.trim().is_empty())
                .map(|kv| {
                    let (n, v) = kv.split_once('=').expect("NAME=value");
                    (n.trim().to_string(), v.trim().to_string())
                })
                .collect()
        })
        .collect();
    let pairs = std::env::var("C16_PAIRS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9usize);
    let prove = |arm: usize| -> f64 {
        for (n, v) in &arms[arm] {
            if v.is_empty() {
                std::env::remove_var(n);
            } else {
                std::env::set_var(n, v);
            }
        }
        let pf = c.preflight().inspect(|p| {
            if crate::prove::host_chains_backend() {
                *p.host_chains.lock().unwrap() = crate::prove::witgen::host_chains(&c.program, p);
            }
        });
        let prover = recursion_prover(&c.hashfn).unwrap();
        let t = Instant::now();
        let r = prover
            .prove_shared(&c.program, c.input.clone(), pf)
            .unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(r);
        // Restore the defaults for the other arm.
        for (n, _) in &arms[arm] {
            std::env::remove_var(n);
        }
        ms
    };
    prove(0);
    prove(1);
    let mut t = [Vec::new(), Vec::new()];
    let mut d = Vec::new();
    for i in 0..pairs {
        wait_gpu_quiet();
        let order = if i % 2 == 0 { [0, 1] } else { [1, 0] };
        let mut ms = [0.0; 2];
        for arm in order {
            ms[arm] = prove(arm);
            t[arm].push(ms[arm]);
        }
        d.push(ms[1] - ms[0]);
        println!(
            "C16AB pair {i}: A {:.1} B {:.1} B-A {:+.1} load {}",
            ms[0],
            ms[1],
            ms[1] - ms[0],
            load_avg()
        );
    }
    let stat = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (s[s.len() / 2], s[0], s[s.len() - 1])
    };
    for (name, v) in [("A", &t[0]), ("B", &t[1]), ("B-A", &d)] {
        let (m, lo, hi) = stat(v);
        println!(
            "C16AB {name}: median {m:.1} [{lo:.1}, {hi:.1}] n={}",
            v.len()
        );
    }
    println!(
        "C16AB B faster in {}/{} pairs",
        d.iter().filter(|x| **x < 0.0).count(),
        d.len()
    );
}
