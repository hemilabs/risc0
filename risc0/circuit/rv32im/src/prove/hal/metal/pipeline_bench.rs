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

//! R14 (docs/metal-port/notes/R14.md): an interleaved A/B bench of the rv32im segment pipeline as
//! zkvm drives it (`prover_impl.rs`): the next segment's preflight on a background thread while
//! `prove_begin` proves the current one, `prove_end` after the last. Every variant runs the same
//! segments in the same process, interleaved per round (A B A B ...), and every seal is compared
//! word for word with the first variant's seal of that segment (fixed `rand_z`), so a variant
//! that changes results fails the bench.
//!
//! Inputs (env):
//! - `R14_SEGMENTS`: comma list of `postcard` `(u32, Segment)` dumps (R11's fix-2 segments);
//!   else `R14_PO2` (default 20) simple_loop segments, `R14_NSEG` of them (default 2).
//! - `R14_VARIANTS`: `name:KEY=VAL+KEY=VAL;name2:...` (`KEY=` removes the variable); default
//!   `A:;B:` (A-vs-A spread).
//! - `R14_PAIRS` (default 7) rounds after one warm-up round.
//!
//! Run alone: `<bin> pipeline_bench --ignored --nocapture --test-threads 1`. It sets process
//! environment variables between rounds, so it refuses to run unless the harness runs one test
//! at a time (PLAN 8.7: a `setenv` racing another test's `getenv` is a data race), and it holds
//! the rv32im Metal test lock for its whole run.

use std::{
    sync::{atomic::Ordering, Arc},
    time::Instant,
};

use risc0_binfmt::MemoryImage;
use risc0_core::field::Elem as _;
use test_log::test;

use super::R14_STATS;
use crate::{
    execute::{
        segment::Segment,
        testutil::{self, NullSyscall, DEFAULT_SESSION_LIMIT},
    },
    prove::{PreflightResults, SegmentProver},
    zirgen::circuit::ExtVal,
    MAX_INSN_CYCLES,
};

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn load_segments() -> Vec<(String, Segment)> {
    let mut out = Vec::new();
    for path in std::env::var("R14_SEGMENTS")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.trim().is_empty())
    {
        let bytes = std::fs::read(path.trim()).unwrap();
        let ((_index, seg), _) = postcard::take_from_bytes::<(u32, Segment)>(&bytes).unwrap();
        let name = std::path::Path::new(path.trim())
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        out.push((name, seg));
    }
    if out.is_empty() {
        let po2: usize = env_or("R14_PO2", 20);
        let n: usize = env_or("R14_NSEG", 2);
        let program = testutil::kernel::simple_loop((n as u32) << po2 >> 2);
        let segs = testutil::execute(
            MemoryImage::new_kernel(program),
            po2,
            MAX_INSN_CYCLES,
            DEFAULT_SESSION_LIMIT,
            &NullSyscall,
            None,
        )
        .unwrap()
        .segments;
        for (i, s) in segs.into_iter().take(n).enumerate() {
            out.push((format!("simple_loop{i}"), s));
        }
    }
    out
}

type Variant = (String, Vec<(String, String)>);

fn parse_variants(spec: &str) -> Vec<Variant> {
    spec.split(';')
        .filter(|v| !v.trim().is_empty())
        .map(|v| {
            let (name, kvs) = v.split_once(':').unwrap_or((v, ""));
            let kvs = kvs
                .split('+')
                .filter(|kv| !kv.trim().is_empty())
                .map(|kv| {
                    let (k, val) = kv.split_once('=').expect("KEY=VAL");
                    (k.trim().to_string(), val.trim().to_string())
                })
                .collect();
            (name.trim().to_string(), kvs)
        })
        .collect()
}

/// Apply a variant's environment; returns the previous values to restore.
fn apply(kvs: &[(String, String)]) -> Vec<(String, Option<String>)> {
    kvs.iter()
        .map(|(k, v)| {
            let old = std::env::var(k).ok();
            // SAFETY (edition 2021 API): pipeline_bench asserts a single-threaded harness and sets
            // variables only between rounds, while no prover thread runs.
            if v.is_empty() {
                std::env::remove_var(k);
            } else {
                std::env::set_var(k, v);
            }
            (k.clone(), old)
        })
        .collect()
}

fn restore(old: Vec<(String, Option<String>)>) {
    for (k, v) in old {
        match v {
            Some(v) => std::env::set_var(&k, v),
            None => std::env::remove_var(&k),
        }
    }
}

/// One pass over all segments the way zkvm drives the prover; returns the wall time of the pass,
/// each prove_begin's wall time, and the seals in segment order.
fn run_pass(
    prover: &dyn SegmentProver,
    segs: &Arc<Vec<(String, Segment)>>,
) -> (f64, Vec<f64>, Vec<Vec<u32>>) {
    let rand_z = ExtVal::ONE;
    let pre = |i: usize| PreflightResults::new(&segs[i].1, rand_z).unwrap();
    let mut seals = Vec::new();
    let mut begins = Vec::new();
    // `R14_PREFLIGHT=upfront`: every preflight before the timed pass (no CPU preflight thread
    // next to the GPU work), else zkvm's background thread.
    let upfront = std::env::var("R14_PREFLIGHT").is_ok_and(|v| v == "upfront");
    let mut all: Vec<Option<PreflightResults>> = if upfront {
        (0..segs.len()).map(|i| Some(pre(i))).collect()
    } else {
        Vec::new()
    };
    let mut next = if upfront { None } else { Some(pre(0)) };
    let t = Instant::now();
    for i in 0..segs.len() {
        let cur = if upfront {
            all[i].take().unwrap()
        } else {
            next.take().unwrap()
        };
        let bg = (!upfront && i + 1 < segs.len()).then(|| {
            let segs = segs.clone();
            std::thread::spawn(move || PreflightResults::new(&segs[i + 1].1, rand_z).unwrap())
        });
        let tb = Instant::now();
        if let Some(seal) = prover.prove_begin(cur).unwrap() {
            seals.push(seal);
        }
        begins.push(tb.elapsed().as_secs_f64());
        next = bg.map(|h| h.join().unwrap());
    }
    seals.push(prover.prove_end().unwrap());
    (t.elapsed().as_secs_f64(), begins, seals)
}

fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[s.len() / 2], s[0], s[s.len() - 1])
}

fn load_average() -> String {
    std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

#[test]
#[ignore = "measurement: run alone with --ignored --nocapture --test-threads 1"]
fn pipeline_bench() {
    assert!(
        super::tests::harness_is_single_threaded(),
        "pipeline_bench sets environment variables: run it with --test-threads 1"
    );
    let _lock = super::tests::metal_test_lock();
    let segs = Arc::new(load_segments());
    let variants =
        parse_variants(&std::env::var("R14_VARIANTS").unwrap_or_else(|_| "A:;B:".into()));
    let pairs: usize = env_or("R14_PAIRS", 7);
    let prover = super::segment_prover().unwrap();
    // Device-wide `currentAllocatedSize` (the pool's reuse across segments shows as a flat line).
    let hal = risc0_zkp::hal::metal::MetalHalPoseidon2::new();
    eprintln!(
        "[r14] {} segments ({}), variants {:?}, {pairs} rounds, load {}",
        segs.len(),
        segs.iter()
            .map(|s| s.0.as_str())
            .collect::<Vec<_>>()
            .join(","),
        variants.iter().map(|v| v.0.as_str()).collect::<Vec<_>>(),
        load_average()
    );
    let mut reference: Option<Vec<Vec<u32>>> = None;
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
    let mut stat_sums: Vec<[u64; 4]> = vec![[0; 4]; variants.len()];
    for round in 0..=pairs {
        for (vi, (name, kvs)) in variants.iter().enumerate() {
            let old = apply(kvs);
            for s in R14_STATS.iter() {
                s.store(0, Ordering::Relaxed);
            }
            let (secs, begins, seals) = run_pass(prover.as_ref(), &segs);
            let st: Vec<u64> = R14_STATS
                .iter()
                .map(|s| s.load(Ordering::Relaxed))
                .collect();
            restore(old);
            for (i, seal) in seals.iter().enumerate() {
                crate::verify(seal).unwrap_or_else(|e| panic!("{name} seg {i}: {e:?}"));
            }
            match &reference {
                None => reference = Some(seals),
                Some(r) => {
                    for (i, (a, b)) in r.iter().zip(&seals).enumerate() {
                        assert!(a == b, "variant {name}: seal of segment {i} differs");
                    }
                }
            }
            eprintln!(
                "[r14] round {round} {name}: {:.0} ms ({:.0} ms/seg; prove_begin ms {:?}) \
                 outside prove_begin {:.1} ms; eval host-only {:.1} ms, eval wait {:.1} ms, \
                 preflight copy {:.1} ms; allocated {} MiB; load {}",
                secs * 1e3,
                secs * 1e3 / segs.len() as f64,
                begins.iter().map(|b| (b * 1e3).round()).collect::<Vec<_>>(),
                (secs - begins.iter().sum::<f64>()) * 1e3,
                st[0] as f64 / 1e6,
                st[1] as f64 / 1e6,
                st[3] as f64 / 1e6,
                hal.allocated_size() >> 20,
                load_average()
            );
            if round > 0 {
                times[vi].push(secs);
                for k in 0..4 {
                    stat_sums[vi][k] += st[k];
                }
            }
        }
    }
    for (vi, (name, _)) in variants.iter().enumerate() {
        let (med, min, max) = stats(&times[vi]);
        let n = (times[vi].len() * segs.len()) as f64;
        eprintln!(
            "[r14] {name}: pass median {:.0} ms [min {:.0}, max {:.0}], per segment median {:.1} ms; \
             per segment: eval host-only {:.2} ms, eval wait {:.2} ms, \
             preflight copy {:.2} ms",
            med * 1e3,
            min * 1e3,
            max * 1e3,
            med * 1e3 / segs.len() as f64,
            stat_sums[vi][0] as f64 / 1e6 / n,
            stat_sums[vi][1] as f64 / 1e6 / n,
            stat_sums[vi][3] as f64 / 1e6 / n,
        );
    }
}
