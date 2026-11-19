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
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
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
    let program = crate::prove::zkr::get_zkr_shared("identity.zkr", PO2).unwrap();
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

/// A headroom source that reports `first` on its first call and `later` on every later one
/// (W7-C1: a Prove fills the budget between the check before the GPU lock and the lock).
struct FlipHeadroom {
    calls: std::sync::atomic::AtomicUsize,
    first: (u64, u64),
    later: (u64, u64),
}

impl FlipHeadroom {
    fn new(first: (u64, u64), later: (u64, u64)) -> Self {
        Self {
            calls: Default::default(),
            first,
            later,
        }
    }
    fn calls(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl ctrl_persist::Headroom for FlipHeadroom {
    fn device_memory(&self) -> Option<(u64, u64)> {
        let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Some(if n == 0 { self.first } else { self.later })
    }
}

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

/// W7-C1 (LEADS2 S5): the prewarm re-checks its device headroom and C13's cap once it holds
/// `exclusive()`. A fake headroom source flips between the two checks: the default order skips
/// without committing; the pre-W7-C1 order (`early`) checks once and commits anyway. No GPU work
/// (the commit is a stub).
#[test]
fn w7c1_prewarm_rechecks_gate_under_lock() {
    let _guard = global_state_test_lock();
    let _mode = ModeGuard;
    ctrl_persist::clear();
    ctrl_persist::set_test_mode(Some(Mode::P254));
    let roomy = (4 * GIB, 0);
    let full = (4 * GIB, 4 * GIB - 400 * MIB);
    let run = |src: &FlipHeadroom, early: bool| {
        let committed = std::cell::Cell::new(false);
        let r = ctrl_persist::prewarm_p254_ctrl_with(PO2, src, early, |_| {
            committed.set(true);
            Ok(())
        })
        .unwrap();
        (r, committed.get())
    };

    // The budget fills while the prewarm waits for the lock.
    let src = FlipHeadroom::new(roomy, full);
    let (r, committed) = run(&src, false);
    assert!(
        matches!(&r, P254Prewarm::Skipped(why) if why.starts_with("no headroom under the GPU lock")),
        "{r:?}"
    );
    assert!(!committed, "committed beside a full budget");
    assert_eq!(src.calls(), 2);

    // The working-set limit shrinks so that the group no longer fits C13's cap (a 32nd of it).
    if std::env::var(ctrl_persist::PERSIST_BYTES_ENV).is_err() {
        let src = FlipHeadroom::new(roomy, (2 * GIB, 0));
        let (r, committed) = run(&src, false);
        assert!(
            matches!(&r, P254Prewarm::Skipped(why) if why.contains("cap under the GPU lock")),
            "{r:?}"
        );
        assert!(!committed);
    }

    // No room before the lock: skipped at once, the lock is never taken.
    let src = FlipHeadroom::new(full, full);
    let (r, committed) = run(&src, false);
    assert!(
        matches!(&r, P254Prewarm::Skipped(why) if why.starts_with("no headroom before the GPU lock")),
        "{r:?}"
    );
    assert!(!committed);
    assert_eq!(src.calls(), 1);

    // Room both times: commits (the stub keeps nothing, hence "not kept").
    let src = FlipHeadroom::new(roomy, roomy);
    let (r, committed) = run(&src, false);
    assert!(committed);
    assert_eq!(r, P254Prewarm::Skipped("the group was not kept".into()));
    assert_eq!(src.calls(), 2);

    // `RISC0_METAL_P254_PREWARM_GATE=early`: one check only, so the flip goes unnoticed.
    let src = FlipHeadroom::new(roomy, full);
    let (_, committed) = run(&src, true);
    assert!(committed, "the early-only order should not re-check");
    assert_eq!(src.calls(), 1);
}

/// W7-C1 (LEADS2 S5) on the GPU, at a 4 GiB budget
/// (`RISC0_METAL_MAX_WORKING_SET_BYTES=4294967296`; heavy, ignored): thread A holds
/// `exclusive()` (as a Prove does for its whole call); thread B's prewarm passes the check before
/// the lock and blocks on it; A then allocates to within 400 MiB of the budget and lets go. B must
/// report `Skipped` and A's allocation must stand (`W7C1_RACE_MARGIN_MIB` changes the 400 MiB). With `W7C1_RACE_EARLY=1` the same race is
/// repeated with the pre-W7-C1 order and its outcome printed (the failure being fixed). Finally,
/// with the memory returned, the prewarm keeps the group.
#[test]
#[ignore]
fn w7c1_prewarm_race_at_budget() {
    crate::prove::hal::metal::require_heavy_gpu_run("w7c1_prewarm_race_at_budget");
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    let _guard = global_state_test_lock();
    let _mode = ModeGuard;
    ctrl_persist::clear();
    ctrl_persist::set_test_mode(Some(Mode::P254));
    let (limit, _) = risc0_zkp::hal::metal::device_memory().expect("Metal device");
    assert!(
        limit <= 8 * GIB,
        "set RISC0_METAL_MAX_WORKING_SET_BYTES (4 GiB): limit is {} MiB",
        limit >> 20
    );
    // Load identity.zkr first, so that B reaches the lock quickly.
    crate::prove::zkr::get_zkr_shared("identity.zkr", PO2).unwrap();
    // A's margin under the limit (default 400 MiB; M9's 4 GiB run had 154 MiB).
    let margin: u64 = std::env::var("W7C1_RACE_MARGIN_MIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let race = |early: bool| -> (anyhow::Result<P254Prewarm>, u64) {
        use std::sync::atomic::Ordering::SeqCst;
        let passed = &ctrl_persist::test_hooks::EARLY_GATE_PASSED;
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let before = passed.load(SeqCst);
        let a = std::thread::spawn(move || {
            let gpu = risc0_zkp::hal::metal::exclusive();
            held_tx.send(()).unwrap();
            let t = Instant::now();
            while passed.load(SeqCst) == before {
                assert!(t.elapsed().as_secs() < 60, "B never passed the early gate");
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            // B is at (or about to block on) `exclusive()`.
            std::thread::sleep(std::time::Duration::from_millis(100));
            let device = metal::Device::system_default().unwrap();
            let (limit, allocated) = risc0_zkp::hal::metal::device_memory().unwrap();
            let want = limit - margin * MIB - allocated;
            let buf = device.new_buffer(want, metal::MTLResourceOptions::StorageModePrivate);
            let (_, filled) = risc0_zkp::hal::metal::device_memory().unwrap();
            // Fix round 1: prepare() staging (no lock) gives way inside the 512 MiB margin too,
            // on the real device count: a new staging buffer here would sit beside the "segment
            // pool". (Pooled buffers that fit are reused, so drain them first.)
            super::staging::drain_pool();
            let wom = vec![risc0_zkp::field::baby_bear::BabyBearExtElem::default(); 1 << 16];
            assert!(
                super::staging::MetalStaged::new(&wom, &[], &[], None).is_none(),
                "staged within the margin"
            );
            drop(gpu);
            // Keep the "segment pool" until B is done.
            done_rx.recv().unwrap();
            drop(buf);
            filled
        });
        held_rx.recv().unwrap();
        let r = if early {
            ctrl_persist::prewarm_p254_ctrl_with(
                PO2,
                &ctrl_persist::MetalDeviceHeadroom,
                true,
                |program| {
                    let prover = crate::prove::recursion_prover("poseidon_254")?;
                    prover.prewarm(program)
                },
            )
        } else {
            prewarm_p254_ctrl(PO2)
        };
        done_tx.send(()).unwrap();
        let filled = a.join().unwrap();
        (r, filled)
    };

    let (r, filled) = race(false);
    eprintln!(
        "W7C1RACE order=default limit={} MiB filled={} MiB outcome={r:?}",
        limit >> 20,
        filled >> 20
    );
    assert!(filled + 512 * MIB > limit, "A did not fill the budget");
    let r = r.expect("the prewarm failed");
    assert!(
        matches!(&r, P254Prewarm::Skipped(why) if why.starts_with("no headroom under the GPU lock")),
        "{r:?}"
    );
    assert_eq!(ctrl_persist::stats().1, 0, "a group was kept");

    if std::env::var("W7C1_RACE_EARLY").is_ok_and(|v| v == "1") {
        let (r, filled) = race(true);
        eprintln!(
            "W7C1RACE order=early limit={} MiB filled={} MiB outcome={r:?}",
            limit >> 20,
            filled >> 20
        );
        ctrl_persist::clear();
    }

    // With the memory back, the prewarm keeps the group.
    let r = prewarm_p254_ctrl(PO2).unwrap();
    eprintln!("W7C1RACE after: {r:?}");
    assert!(matches!(r, P254Prewarm::Kept { .. }), "{r:?}");
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
    crate::prove::hal::metal::require_heavy_gpu_run("c16_first_job");
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
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
    crate::prove::hal::metal::require_heavy_gpu_run("c16_ab_p254_in_process");
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
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

/// W7-C1: build cost of zkp's p254 PSO variants (C16 open issue: the 4 PSOs C16 added, `w64u` x2
/// and the 4-lane x2), measured without touching zkp: the zkp metallib at `W7C1_ZKP_METALLIB` is
/// loaded through the public concurrent loader with no archive (run with
/// `MTL_SHADER_CACHE_SIZE=0` for a cold compile), building `W7C1_PSO_SET` = `all` (zkp's
/// `KERNEL_NAMES`, what a HAL builds today), `no_w64u` (all but the unrolled pair: what a lazy
/// non-default variant would build eagerly), `w64u` or `lanes` (one pair alone). One set per
/// process. Prints `W7C1PSO set=.. n=.. ms=..`.
#[test]
#[ignore]
fn w7c1_p254_pso_cold_cost() {
    crate::prove::hal::metal::require_heavy_gpu_run("w7c1_p254_pso_cold_cost");
    let _heavy = crate::prove::hal::metal::heavy_test_lock();
    let path = std::env::var("W7C1_ZKP_METALLIB").expect("W7C1_ZKP_METALLIB=<zkp metallib>");
    let bytes = std::fs::read(&path).unwrap();
    let set = std::env::var("W7C1_PSO_SET").unwrap_or_else(|_| "all".into());
    let all = risc0_zkp::hal::metal::KERNEL_NAMES;
    let w64u = ["p254_fold_w64u", "p254_rows_w64u"];
    let lanes = ["p254_fold4_w64", "p254_rows4_w64"];
    let names: Vec<&str> = match set.as_str() {
        "all" => all.to_vec(),
        "no_w64u" => all.iter().copied().filter(|n| !w64u.contains(n)).collect(),
        "w64u" => w64u.to_vec(),
        "lanes" => lanes.to_vec(),
        s => panic!("W7C1_PSO_SET={s:?}"),
    };
    let device = metal::Device::system_default().unwrap();
    let t = Instant::now();
    let k = risc0_zkp::hal::metal::archive::ArchivedKernels::new_in_threads(
        None, &device, "w7c1", &bytes, &names, None,
    )
    .unwrap();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    std::hint::black_box(k);
    println!("W7C1PSO set={set} n={} ms={ms:.0}", names.len());
}
