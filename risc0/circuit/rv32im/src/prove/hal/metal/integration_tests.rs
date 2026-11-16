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

//! R11 integration gates for the rv32im Metal segment prover (docs/metal-port/PLAN.md 4.2):
//!
//! - [seal_matches_cpu]: with a fixed `rand_z` (the `witgen_debug` value, `ExtVal::ONE`) the
//!   Metal segment prover's seal equals the CPU segment prover's, word for word, and verifies on
//!   the CPU. Under `--features witgen_debug` the same holds through `SegmentProver::prove`.
//! - [seal_matches_cpu_large] (ignored, slow, huge class): the same at po2 20 (simple_loop) and on
//!   dumped real-guest segments (`R11_SEGMENTS`).
//! - [prove_po2_sweep] (ignored, measurement): Metal wall time and the peak GPU memory of each
//!   prover phase ([super::PHASE_MARKS]) per po2 of `R11_PO2S`, optionally against the CPU prover.
//!   Over a `RISC0_METAL_MAX_WORKING_SET_BYTES` budget the failing allocation is reported.
//!
//! W7-R1 (LEADS2 S1) memory-safe defaults, see docs/metal-port/notes/W7-R1.md:
//! - the ignored tests panic unless run under `bin/gpu-run`'s exclusive lock (`GPU_RUN_MODE=heavy`
//!   or `huge`; `R11_NO_GPU_RUN_CHECK=1` skips the check). `bin/gpu-run --huge` (exclusive, with a
//!   memory admission check for a ~20 GB process) is the recommended mode for the CPU-plus-Metal
//!   po2 20 runs (`seal_matches_cpu_large`, the sweep with a CPU prover at po2 >= 20) and is
//!   required for the ~20 GB configurations: po2 21, `R11_KEEP_PROVER=1` or
//!   `R11_EQ_KEEP_PROVERS=1` with a po2 >= 20 case;
//! - the sweep defaults to `R11_PO2S=20` with no CPU prover at po2 >= 20 (`R11_CPU=1` forces it
//!   on at every po2, `R11_CPU=0` off); po2 21 (~19 GiB) needs `R11_ALLOW_21=1`; the Metal and CPU
//!   provers are dropped and rebuilt for every po2, so each po2 starts with an empty buffer pool
//!   and the sweep's peak is its largest po2's, not the sum (`R11_KEEP_PROVER=1`: one prover for
//!   the whole sweep, the old behaviour);
//! - `seal_matches_cpu_large` proves each case with the CPU prover first and drops it before the
//!   Metal prover of that case exists, then drops that too (peak = the larger of the two, not the
//!   sum); `R11_EQ_KEEP_PROVERS=1` keeps both for the whole run (the old behaviour);
//! - a test that runs the CPU prover runs itself again in a child with `MallocLargeCache=0`, so the
//!   CPU prover's freed buffers leave the footprint ([super::tests::rerun_without_malloc_large_cache];
//!   `R11_KEEP_MALLOC_CACHE=1` skips that).
//!
//! Usage (release, `--features metal`, from the crate directory; `<bin>` = the lib test binary):
//! - `bin/gpu-run --tag <item> -- <bin> prove_po2_sweep --ignored --nocapture` (po2 20; BASELINES:
//!   `R11_PO2S=20 R11_CPU=0 R11_REPS=4`);
//! - `R11_PO2S=14,15,16,17,18,19,20 bin/gpu-run --tag <item> -- <bin> prove_po2_sweep --ignored
//!   --nocapture` (the sweep; CPU at po2 < 20);
//! - `R11_EQ_PO2S= R11_SEGMENTS=<seg.postcard> bin/gpu-run --huge --tag <item> -- <bin>
//!   seal_matches_cpu_large --ignored --nocapture` (one dumped segment per process is the
//!   smallest footprint; a list is proved one case at a time);
//! - `R11_ALLOW_21=1 R11_PO2S=21 R11_CPU=0 bin/gpu-run --huge --no-budget --tag <item> -- <bin>
//!   prove_po2_sweep --ignored --nocapture` (po2 21 alone, ~18.6 GiB).
//!
//! Tests that rerun themselves with `MallocLargeCache=0` do their work in a child process:
//! `/usr/bin/time -l` around them reports only the parent (~2.3 MB). Use the in-process
//! `footprint ... (process peak ...)` lines, or the parent's `child (MallocLargeCache=0) peak
//! footprint` line.
//!
//! Which circuit phases run on the GPU is the production default (all three since R17;
//! `RISC0_METAL_RV32IM_CPU_PHASES=witgen` puts witgen on the CPU kernel, see witgen.rs); run these
//! once per configuration.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::atomic::Ordering,
    time::Instant,
};

use risc0_binfmt::{MemoryImage, Program};
use risc0_core::field::Elem as _;
use risc0_zkp::hal::metal::MetalHalPoseidon2;
use test_log::test;

use super::{PHASE_MARKS, PHASE_MARKS_ON};
use crate::{
    execute::{
        segment::Segment,
        testutil::{self, NullSyscall, DEFAULT_SESSION_LIMIT},
        DEFAULT_SEGMENT_LIMIT_PO2,
    },
    prove::{hal::cpu, PreflightResults, SegmentProver},
    zirgen::circuit::ExtVal,
    MAX_INSN_CYCLES,
};

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn segments(program: Program, po2: usize) -> Vec<Segment> {
    // Below po2 15 the executor needs the lower bound (zkvm uses it the same way).
    let max_insn = if (1usize << po2) > MAX_INSN_CYCLES {
        MAX_INSN_CYCLES
    } else {
        crate::MAX_INSN_CYCLES_LOWER_PO2
    };
    testutil::execute(
        MemoryImage::new_kernel(program),
        po2,
        max_insn,
        DEFAULT_SESSION_LIMIT,
        &NullSyscall,
        None,
    )
    .unwrap()
    .segments
}

/// Some riscv-tests programs (the mul/div majors, loads/stores, shifts), by name.
fn riscv_programs(names: &[&str]) -> Vec<(String, Program)> {
    use std::io::Read;

    use flate2::read::GzDecoder;
    use tar::Archive;

    let bytes = include_bytes!("../../testdata/riscv-tests.tgz");
    let mut tar = Archive::new(GzDecoder::new(&bytes[..]));
    let mut found = Vec::new();
    for entry in tar.entries().unwrap() {
        let mut entry = entry.unwrap();
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let name = entry
            .path()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        if !names.contains(&name.as_str()) {
            continue;
        }
        let mut elf = Vec::new();
        entry.read_to_end(&mut elf).unwrap();
        found.push((name, Program::load_elf(&elf, u32::MAX).unwrap()));
    }
    assert_eq!(found.len(), names.len(), "riscv-tests.tgz cases");
    found
}

fn phases_label() -> &'static str {
    if !super::witgen::cpu_witgen() {
        "all three phases on the GPU"
    } else {
        "witgen on the CPU kernel, accum + eval_check on the GPU"
    }
}

/// Metal seal == CPU seal for a fixed `rand_z`, and both verify.
#[test]
fn seal_matches_cpu() {
    // W7-R1: ~3.5 GB of footprint (CPU and Metal provers of 14 programs): one large test at a time.
    let _lock = super::tests::large_test_lock();
    let metal = super::segment_prover().unwrap();
    let cpu = cpu::segment_prover().unwrap();
    let mut programs: Vec<(String, Program)> = vec![
        ("basic".into(), testutil::kernel::basic()),
        ("multi_read".into(), testutil::kernel::multi_read()),
        (
            "simple_loop(40000)".into(),
            testutil::kernel::simple_loop(40000),
        ),
    ];
    programs.extend(riscv_programs(&[
        "mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu", "sb", "lhu", "sra",
    ]));
    let rand_z = ExtVal::ONE; // crate::prove::preflight_rand_z under witgen_debug
    let mut checked = 0;
    for (name, program) in programs {
        for segment in segments(program, DEFAULT_SEGMENT_LIMIT_PO2) {
            let t = Instant::now();
            let seal_cpu = cpu
                .prove_core(PreflightResults::new(&segment, rand_z).unwrap())
                .unwrap();
            let t_cpu = t.elapsed();
            let t = Instant::now();
            let seal_metal = metal
                .prove_core(PreflightResults::new(&segment, rand_z).unwrap())
                .unwrap();
            let t_metal = t.elapsed();
            crate::verify(&seal_cpu).unwrap();
            crate::verify(&seal_metal)
                .unwrap_or_else(|e| panic!("{name} segment {}: Metal seal: {e:?}", segment.index));
            let diff = seal_cpu.iter().zip(&seal_metal).position(|(a, b)| a != b);
            assert!(
                seal_cpu.len() == seal_metal.len() && diff.is_none(),
                "{name} segment {} po2 {}: Metal seal differs from the CPU seal \
                 (len {} vs {}, first differing word {diff:?})",
                segment.index,
                segment.po2,
                seal_cpu.len(),
                seal_metal.len()
            );
            #[cfg(feature = "witgen_debug")]
            {
                // SegmentProver::prove draws rand_z from preflight_rand_z(): fixed here.
                let a = cpu.prove(&segment).unwrap();
                let b = metal.prove(&segment).unwrap();
                assert_eq!(a, b, "{name}: prove() seals differ under witgen_debug");
                assert_eq!(
                    a, seal_cpu,
                    "{name}: witgen_debug rand_z is not ExtVal::ONE"
                );
            }
            tracing::info!(
                "[r11] {name} segment {} po2 {}: seals equal ({} words), cpu {:.0} ms, metal {:.0} ms",
                segment.index,
                segment.po2,
                seal_metal.len(),
                t_cpu.as_secs_f64() * 1e3,
                t_metal.as_secs_f64() * 1e3
            );
            checked += 1;
        }
    }
    eprintln!(
        "[r11] seal_matches_cpu ({}): {checked} segments",
        phases_label()
    );
}

/// W7-R1: the "huge class" banner of the heavy tests (one process can reach most of a 36 GB Mac).
fn huge_banner(test: &str, what: &str) {
    let mode = std::env::var("GPU_RUN_MODE").unwrap_or_else(|_| "unset".into());
    eprintln!(
        "[w7-r1] {test}: huge class ({what}): run alone with `bin/gpu-run --huge --tag <item> -- \
         ...` (GPU_RUN_MODE={mode})"
    );
}

/// This process' footprint and lifetime peak footprint, for the log.
fn footprint_line() -> String {
    let (now, peak) = super::tests::process_footprint();
    format!(
        "footprint {:.2} GiB (process peak {:.2} GiB)",
        gib(now),
        gib(peak)
    )
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// One case of [seal_matches_cpu_large]: a simple_loop po2 or a segment file, loaded on demand so
/// that only one case's segment is in memory at a time.
enum LargeCase {
    Loop(usize),
    File(String),
}

impl LargeCase {
    fn name(&self) -> String {
        match self {
            Self::Loop(po2) => format!("simple_loop po2 {po2}"),
            Self::File(path) => path.clone(),
        }
    }

    fn load(&self) -> Segment {
        match self {
            Self::Loop(po2) => segments(testutil::kernel::simple_loop((1u32 << po2) / 4), *po2)
                .into_iter()
                .next()
                .unwrap(),
            Self::File(path) => {
                let bytes = std::fs::read(path).unwrap();
                let ((_index, seg), _) =
                    postcard::take_from_bytes::<(u32, Segment)>(&bytes).unwrap();
                seg
            }
        }
    }
}

/// Large-segment seal equality (ignored: the CPU prover takes about a minute per po2 20 segment):
/// a full simple_loop segment at each po2 of `R11_EQ_PO2S` (default 20) and every segment file in
/// `R11_SEGMENTS` (comma list of `postcard` dumps of `risc0_zkvm::Segment`, the format of
/// `witgen_segment_file`; e.g. the rv32im-torture / sha256-chain / memory-merkle po2 20 segments
/// of notes/R11.md). The Metal seal must equal the CPU seal word for word for a fixed `rand_z` and
/// verify; `R11_EQ_REPS` (default 1) Metal proofs per segment.
///
/// W7-R1 (huge class): each case is proved by a fresh CPU prover that is dropped before that
/// case's fresh Metal prover is built, and the Metal prover is dropped before the next case, so
/// the process peak is the larger of one CPU and one Metal po2 20 proof, not their sum plus the
/// Metal pool of every earlier case (R23's review measured 19.9-21 GB per 2-segment process and
/// 27.97 GB with po2 13..20 mixed). `R11_EQ_KEEP_PROVERS=1` keeps both provers for the whole run
/// (the old behaviour, for comparison).
#[test]
#[ignore = "slow, huge class: po2 20 CPU proofs; run alone under bin/gpu-run --huge with --ignored --nocapture"]
fn seal_matches_cpu_large() {
    // The old keep-both-provers mode reaches ~20 GB: only under --huge's admission check. The
    // large-test lock is held here, in the parent, across the MallocLargeCache=0 child's run, so
    // no other heavy test of this binary overlaps it (W7-R1 fix round 2).
    let _lock = super::tests::begin_heavy_test(
        "seal_matches_cpu_large",
        if env_or("R11_EQ_KEEP_PROVERS", 0) == 1 {
            super::tests::GpuRunClass::Huge
        } else {
            super::tests::GpuRunClass::Heavy
        },
    );
    if super::tests::rerun_without_malloc_large_cache(&format!(
        "{}::seal_matches_cpu_large",
        module_path!()
    )) {
        return;
    }
    huge_banner(
        "seal_matches_cpu_large",
        "a po2 20 CPU proof and a po2 20 Metal proof per case, one after the other",
    );
    let reps = env_or("R11_EQ_REPS", 1usize);
    let keep = env_or("R11_EQ_KEEP_PROVERS", 0) == 1;
    let mut cases: Vec<LargeCase> = Vec::new();
    for po2 in std::env::var("R11_EQ_PO2S")
        .unwrap_or_else(|_| "20".into())
        .split(',')
        .filter(|s| !s.trim().is_empty())
    {
        cases.push(LargeCase::Loop(po2.trim().parse().unwrap()));
    }
    for path in std::env::var("R11_SEGMENTS")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.trim().is_empty())
    {
        cases.push(LargeCase::File(path.trim().to_string()));
    }
    if cases.len() > 1 && !keep {
        eprintln!(
            "[w7-r1] seal_matches_cpu_large: {} cases in one process, proved one at a time with \
             fresh provers (one case per process is the smallest footprint)",
            cases.len()
        );
    }
    let kept = keep.then(|| {
        (
            super::segment_prover().unwrap(),
            cpu::segment_prover().unwrap(),
        )
    });
    let rand_z = ExtVal::ONE;
    let mut bad = Vec::new();
    for case in &cases {
        let name = case.name();
        let segment = case.load();
        let t = Instant::now();
        let seal_cpu = {
            let fresh = kept.is_none().then(|| cpu::segment_prover().unwrap());
            let cpu = kept
                .as_ref()
                .map_or_else(|| fresh.as_ref().unwrap(), |k| &k.1);
            cpu.prove_core(PreflightResults::new(&segment, rand_z).unwrap())
                .unwrap()
            // A fresh CPU prover (and its HAL's buffers) is dropped here.
        };
        let t_cpu = t.elapsed().as_secs_f64();
        crate::verify(&seal_cpu).unwrap();
        eprintln!("[w7-r1] {name}: after the CPU proof: {}", footprint_line());
        let fresh = kept.is_none().then(|| super::segment_prover().unwrap());
        let metal = kept
            .as_ref()
            .map_or_else(|| fresh.as_ref().unwrap(), |k| &k.0);
        for rep in 0..reps {
            let t = Instant::now();
            let res = metal.prove_core(PreflightResults::new(&segment, rand_z).unwrap());
            let t_metal = t.elapsed().as_secs_f64();
            let verdict = match res {
                Err(e) => format!("Metal prove error: {e:#}"),
                Ok(seal) => match crate::verify(&seal) {
                    Err(e) => format!("Metal seal does not verify: {e:?}"),
                    Ok(()) if seal != seal_cpu => format!(
                        "Metal seal differs from the CPU seal (first word {:?})",
                        seal_cpu.iter().zip(&seal).position(|(a, b)| a != b)
                    ),
                    Ok(()) => String::new(),
                },
            };
            eprintln!(
                "[r11] {name} po2 {} rep {rep}: {} (cpu {:.0} ms, metal {:.0} ms)",
                segment.po2,
                if verdict.is_empty() {
                    "seals equal, verify"
                } else {
                    verdict.as_str()
                },
                t_cpu * 1e3,
                t_metal * 1e3
            );
            if !verdict.is_empty() {
                bad.push(format!("{name} rep {rep}: {verdict}"));
            }
        }
        drop(fresh);
        drop(segment);
        eprintln!(
            "[w7-r1] {name}: after the Metal proof{}: {}",
            if keep { "" } else { " and its prover's drop" },
            footprint_line()
        );
    }
    eprintln!(
        "[r11] seal_matches_cpu_large ({}): {} cases x {reps}, {} bad; {}",
        phases_label(),
        cases.len(),
        bad.len(),
        footprint_line()
    );
    assert!(bad.is_empty(), "{bad:#?}");
}

fn mb(b: u64) -> f64 {
    b as f64 / (1u64 << 20) as f64
}

/// Prove `segment` with the Metal prover, catching the HAL's budget panic; returns the time and
/// the phase marks, or the error text.
fn metal_prove(
    hal: &MetalHalPoseidon2,
    prover: &dyn SegmentProver,
    segment: &Segment,
) -> Result<(f64, Vec<super::PhaseMark>, Vec<u32>), String> {
    PHASE_MARKS.lock().unwrap().clear();
    PHASE_MARKS_ON.store(true, Ordering::Relaxed);
    super::phase_mark(hal, "prove start");
    let t = Instant::now();
    let res = catch_unwind(AssertUnwindSafe(|| prover.prove(segment)));
    let secs = t.elapsed().as_secs_f64();
    super::phase_mark(hal, "prove end");
    PHASE_MARKS_ON.store(false, Ordering::Relaxed);
    let marks = std::mem::take(&mut *PHASE_MARKS.lock().unwrap());
    match res {
        Ok(Ok(seal)) => Ok((secs, marks, seal)),
        Ok(Err(e)) => Err(format!("{e:#}")),
        Err(p) => Err(p
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "panic".into())),
    }
}

/// The sweep's po2s (`R11_PO2S`, default `20`); po2 21 (~19 GiB of Metal memory plus the host
/// side) only with `R11_ALLOW_21=1`.
fn sweep_po2s() -> Vec<usize> {
    let po2s: Vec<usize> = std::env::var("R11_PO2S")
        .unwrap_or_else(|_| "20".into())
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().unwrap())
        .collect();
    assert!(
        !po2s.iter().any(|&p| p >= 21) || env_or("R11_ALLOW_21", 0) == 1,
        "R11_PO2S={po2s:?}: po2 21 needs ~19 GiB of GPU memory plus the host side; set \
         R11_ALLOW_21=1 to run it (alone, under `bin/gpu-run --huge`, with --no-budget or a \
         budget above 20 GiB)"
    );
    po2s
}

/// The gpu-run mode the sweep needs: `--huge` for the ~20 GB configurations (po2 21; one kept
/// prover across a sweep reaching po2 20; a CPU prover beside the Metal one at po2 >= 20), the
/// plain heavy lock (or `--huge`) otherwise.
fn sweep_class(po2s: &[usize]) -> super::tests::GpuRunClass {
    let max = po2s.iter().copied().max().unwrap_or(0);
    let keep = env_or("R11_KEEP_PROVER", 0) == 1;
    if max >= 21
        || (keep && max >= 20 && po2s.len() > 1)
        || po2s.iter().any(|&p| p >= 20 && sweep_cpu(p))
    {
        super::tests::GpuRunClass::Huge
    } else {
        super::tests::GpuRunClass::Heavy
    }
}

/// Whether the sweep runs the CPU prover at `po2`: `R11_CPU=1` / `0` at every po2, unset only
/// below po2 20 (a po2 20 CPU proof takes about a minute and ~10 GB beside the Metal prover).
fn sweep_cpu(po2: usize) -> bool {
    match std::env::var("R11_CPU") {
        Ok(v) => v.trim() == "1",
        Err(_) => po2 < 20,
    }
}

/// po2 sweep: Metal (`R11_REPS`, default 2) and optionally CPU wall time, the peak GPU memory per
/// phase, and (W7-R1) the device's `currentAllocatedSize` and the process footprint after each
/// po2, before and after its provers are dropped (`R11_SETTLE_MS`: again that much later).
/// Defaults: po2 20, no CPU prover at po2 >= 20,
/// fresh provers per po2 (`R11_KEEP_PROVER=1`: one Metal and one CPU prover for the whole sweep,
/// whose pool then keeps the buffers of every po2, the old behaviour).
#[test]
#[ignore = "measurement, heavy: po2 20 by default; run alone under bin/gpu-run with --ignored --nocapture"]
fn prove_po2_sweep() {
    let po2s = sweep_po2s();
    // Gate, then the large-test lock for the whole sweep (also across the CPU-sweep child).
    let _lock = super::tests::begin_heavy_test("prove_po2_sweep", sweep_class(&po2s));
    if po2s.iter().any(|&p| sweep_cpu(p))
        && super::tests::rerun_without_malloc_large_cache(&format!(
            "{}::prove_po2_sweep",
            module_path!()
        ))
    {
        return;
    }
    let reps = env_or("R11_REPS", 2usize);
    let keep = env_or("R11_KEEP_PROVER", 0) == 1;
    if po2s.iter().any(|&p| p >= 20) {
        huge_banner(
            "prove_po2_sweep",
            "~9.3 GiB of Metal memory per po2 20 segment, ~19 GiB at po2 21",
        );
    }
    // Reads the device-wide `currentAllocatedSize` and the process-wide peak for the phase marks;
    // its own pool stays empty (no allocations through it).
    let hal = MetalHalPoseidon2::new();
    let limit = hal.working_set_limit();
    eprintln!(
        "[r11] sweep: {}, working-set limit {:.0} MiB, po2s {po2s:?}, {}, {}",
        phases_label(),
        mb(limit),
        if keep {
            "one prover for the whole sweep (R11_KEEP_PROVER=1)"
        } else {
            "fresh provers per po2"
        },
        footprint_line()
    );
    let mut kept_metal = None;
    let mut kept_cpu = None;
    for po2 in po2s {
        // simple_loop: ~4 cycles per iteration; the first segment is a full 2^po2 one.
        let segs = segments(testutil::kernel::simple_loop((1u32 << po2) / 4), po2);
        let segment = &segs[0];
        assert_eq!(segment.po2 as usize, po2);
        let fresh = (!keep).then(|| super::segment_prover().unwrap());
        if keep && kept_metal.is_none() {
            kept_metal = Some(super::segment_prover().unwrap());
        }
        let metal = fresh.as_ref().or(kept_metal.as_ref()).unwrap();
        for rep in 0..reps {
            match metal_prove(&hal, metal.as_ref(), segment) {
                Ok((secs, marks, seal)) => {
                    crate::verify(&seal).unwrap();
                    let mut line = String::new();
                    let mut max = 0;
                    for w in marks.windows(2) {
                        // The peak of the interval that ends at w[1].
                        let what = match (w[0].label, w[1].label) {
                            ("prove start", "witgen start") => "preflight+alloc",
                            ("witgen start", "witgen end") => "witgen",
                            ("witgen end", "accum start") => "data commit",
                            ("accum start", "accum end") => "accum",
                            ("accum end", "eval_check start") => "accum commit",
                            ("eval_check start", "eval_check end") => "eval_check",
                            ("eval_check end", "prove end") => "fri+tail",
                            _ => "?",
                        };
                        max = max.max(w[1].peak);
                        line += &format!(
                            " {what} {:.0} MiB/{:.0} ms;",
                            mb(w[1].peak),
                            (w[1].at - w[0].at).as_secs_f64() * 1e3
                        );
                    }
                    let end = marks.last().unwrap();
                    eprintln!(
                        "[r11] po2={po2} metal rep={rep}: {:.0} ms, seal verifies, peak {:.0} MiB, \
                         held after {:.0} MiB |{line}",
                        secs * 1e3,
                        mb(max),
                        mb(end.allocated)
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[r11] po2={po2} metal rep={rep}: FAILED (limit {:.0} MiB): {}",
                        mb(limit),
                        e.lines().next().unwrap_or("")
                    );
                    break;
                }
            }
        }
        let held = hal.allocated_size();
        let before_drop = footprint_line();
        drop(fresh);
        eprintln!(
            "[w7-r1] po2={po2} after the Metal proofs: currentAllocatedSize {:.0} MiB, {before_drop}; \
             {}: currentAllocatedSize {:.0} MiB, {}",
            mb(held),
            if keep {
                "prover kept"
            } else {
                "after dropping the prover"
            },
            mb(hal.allocated_size()),
            footprint_line()
        );
        // W7-R1: whether the footprint of the freed buffers decays with time (measurement only).
        let settle = env_or("R11_SETTLE_MS", 0u64);
        if settle > 0 {
            std::thread::sleep(std::time::Duration::from_millis(settle));
            eprintln!(
                "[w7-r1] po2={po2} {settle} ms later: currentAllocatedSize {:.0} MiB, {}",
                mb(hal.allocated_size()),
                footprint_line()
            );
        }
        if sweep_cpu(po2) {
            let fresh = (!keep).then(|| cpu::segment_prover().unwrap());
            if keep && kept_cpu.is_none() {
                kept_cpu = Some(cpu::segment_prover().unwrap());
            }
            let cpu = fresh.as_ref().or(kept_cpu.as_ref()).unwrap();
            let t = Instant::now();
            let seal = cpu.prove(segment).unwrap();
            let secs = t.elapsed().as_secs_f64();
            crate::verify(&seal).unwrap();
            let before_drop = footprint_line();
            drop(fresh);
            eprintln!(
                "[r11] po2={po2} cpu: {:.0} ms, seal verifies; {before_drop}; {}: {}",
                secs * 1e3,
                if keep {
                    "prover kept"
                } else {
                    "after dropping the prover"
                },
                footprint_line()
            );
        }
    }
}
