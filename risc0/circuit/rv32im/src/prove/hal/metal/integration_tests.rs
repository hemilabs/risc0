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
//! - [seal_matches_cpu_large] (ignored, slow): the same at po2 20 (simple_loop) and on dumped
//!   real-guest segments (`R11_SEGMENTS`).
//! - [prove_po2_sweep] (ignored, measurement): po2 14..21 (`R11_PO2S`), Metal vs CPU wall time and
//!   the peak GPU memory of each prover phase ([super::PHASE_MARKS]); `R11_CPU=0` skips the CPU
//!   prover. Over a `RISC0_METAL_MAX_WORKING_SET_BYTES` budget the failing allocation is reported.
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

/// Large-segment seal equality (ignored: the CPU prover takes about a minute per po2 20 segment):
/// a full simple_loop segment at each po2 of `R11_EQ_PO2S` (default 20) and every segment file in
/// `R11_SEGMENTS` (comma list of `postcard` dumps of `risc0_zkvm::Segment`, the format of
/// `witgen_segment_file`; e.g. the rv32im-torture / sha256-chain / memory-merkle po2 20 segments
/// of notes/R11.md). The Metal seal must equal the CPU seal word for word for a fixed `rand_z` and
/// verify; `R11_EQ_REPS` (default 1) Metal proofs per segment.
#[test]
#[ignore = "slow: po2 20 CPU proofs; run with --ignored --nocapture"]
fn seal_matches_cpu_large() {
    let metal = super::segment_prover().unwrap();
    let cpu = cpu::segment_prover().unwrap();
    let reps = env_or("R11_EQ_REPS", 1usize);
    let mut cases: Vec<(String, Segment)> = Vec::new();
    for po2 in std::env::var("R11_EQ_PO2S")
        .unwrap_or_else(|_| "20".into())
        .split(',')
        .filter(|s| !s.trim().is_empty())
    {
        let po2: usize = po2.trim().parse().unwrap();
        let seg = segments(testutil::kernel::simple_loop((1u32 << po2) / 4), po2)
            .into_iter()
            .next()
            .unwrap();
        cases.push((format!("simple_loop po2 {po2}"), seg));
    }
    for path in std::env::var("R11_SEGMENTS")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.trim().is_empty())
    {
        let bytes = std::fs::read(path.trim()).unwrap();
        let ((_index, seg), _) = postcard::take_from_bytes::<(u32, Segment)>(&bytes).unwrap();
        cases.push((path.trim().to_string(), seg));
    }
    let rand_z = ExtVal::ONE;
    let mut bad = Vec::new();
    for (name, segment) in &cases {
        let t = Instant::now();
        let seal_cpu = cpu
            .prove_core(PreflightResults::new(segment, rand_z).unwrap())
            .unwrap();
        let t_cpu = t.elapsed().as_secs_f64();
        crate::verify(&seal_cpu).unwrap();
        for rep in 0..reps {
            let t = Instant::now();
            let res = metal.prove_core(PreflightResults::new(segment, rand_z).unwrap());
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
    }
    eprintln!(
        "[r11] seal_matches_cpu_large ({}): {} cases x {reps}, {} bad",
        phases_label(),
        cases.len(),
        bad.len()
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

/// po2 sweep: Metal (2 reps) vs CPU wall time and the peak GPU memory per phase.
#[test]
#[ignore = "measurement: po2 14..21, run alone with --ignored --nocapture"]
fn prove_po2_sweep() {
    let po2s: Vec<usize> = std::env::var("R11_PO2S")
        .unwrap_or_else(|_| "14,15,16,17,18,19,20,21".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let with_cpu = env_or("R11_CPU", 1) == 1;
    let reps = env_or("R11_REPS", 2usize);
    let metal = super::segment_prover().unwrap();
    let cpu = cpu::segment_prover().unwrap();
    let hal = MetalHalPoseidon2::new();
    let limit = hal.working_set_limit();
    eprintln!(
        "[r11] sweep: {}, working-set limit {:.0} MiB",
        phases_label(),
        mb(limit)
    );
    for po2 in po2s {
        // simple_loop: ~4 cycles per iteration; the first segment is a full 2^po2 one.
        let segs = segments(testutil::kernel::simple_loop((1u32 << po2) / 4), po2);
        let segment = &segs[0];
        assert_eq!(segment.po2 as usize, po2);
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
        if with_cpu {
            let t = Instant::now();
            let seal = cpu.prove(segment).unwrap();
            let secs = t.elapsed().as_secs_f64();
            crate::verify(&seal).unwrap();
            eprintln!("[r11] po2={po2} cpu: {:.0} ms, seal verifies", secs * 1e3);
        }
    }
}
