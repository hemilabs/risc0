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

//! Item K3: the Metal eval_check translator (build_metal_eval.rs), tested directly.
//!
//! - `stage_layout`, `pinned_counts`, `rejects_changed_generator`: no toolchain needed.
//! - `msl_compiles` (ignored: minutes of `xcrun metal`): compiles the stages for the counts in
//!   `K3_STAGE_COUNTS` (default `58,29,16,8`) with `K3_JOBS` (default 6) compilers at once, prints
//!   the front-end time of each stage file and links a metallib per count under
//!   `$CARGO_TARGET_TMPDIR/k3/<count>/` (used for PSO timing, see docs/metal-port/notes/K3.md).
//! - `metallib_has_every_stage` (`--features metal`): every `eval_stage_<k>` is in `METAL_LIB`.

#![allow(dead_code)]

#[path = "../build_metal_eval.rs"]
mod build_metal_eval;

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
    time::Instant,
};

use build_metal_eval::{read_cuda_sources, translate, RewriteCounts, Translation};

fn manifest() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// The eval_check_<n>.cu files as `(path, text)`.
type Sources = Vec<(String, String)>;

fn inputs() -> (String, Sources) {
    read_cuda_sources(&manifest().join("kernels/cuda"))
}

#[test]
fn stage_layout() {
    let (header, sources) = inputs();
    for requested in [58, 29, 16, 8, 4, 2, 1, 100] {
        let tr = translate(&header, &sources, requested);
        let s = requested.min(58);
        assert_eq!(tr.chain_len, 58);
        assert_eq!(tr.stages.len(), s, "requested {requested}");
        assert_eq!(tr.files.len(), s);
        // Every function in exactly one stage, in chain order, poly_fp first, keccak_0 last.
        let chain: Vec<String> = tr
            .stages
            .iter()
            .flat_map(|st| st.functions.clone())
            .collect();
        let mut expected = vec!["poly_fp".to_string()];
        expected.extend((0..=56).rev().map(|n| format!("keccak_{n}")));
        assert_eq!(chain, expected);
        for (k, (st, (file, text))) in tr.stages.iter().zip(&tr.files).enumerate() {
            assert_eq!(st.name, format!("eval_stage_{k}"));
            assert_eq!(file, &format!("eval_stage_{k}.metal"));
            assert!(text.contains(&format!("kernel void eval_stage_{k}(")));
            assert!(!st.functions.is_empty());
            // Only the last stage writes the check; every other one ends in carry stores.
            let last = k + 1 == s;
            assert_eq!(text.contains("write_check("), last, "stage {k}");
            assert_eq!(text.contains("carry_store("), !last, "stage {k}");
            assert_eq!(text.contains("carry_load("), k > 0, "stage {k}");
            assert!(!text.contains("__device__") && !text.contains("__constant__"));
            assert!(!text.contains("size_t"));
        }
        assert_eq!(tr.counts.cuts, s - 1);
        assert_eq!(tr.counts.calls, 57 - (s - 1));
        // Balanced: no stage more than the largest function above an even share.
        let total: usize = tr.stages.iter().map(|st| st.lines).sum();
        assert_eq!(total, tr.counts.body_lines);
        let max = tr.stages.iter().map(|st| st.lines).max().unwrap();
        assert!(
            max <= total / s + 3500,
            "requested {requested}: largest stage {max} lines"
        );
        let rs = tr.rust_consts();
        assert!(rs.contains(&format!("pub const EVAL_STAGES: usize = {s};")));
        assert!(rs.contains(&format!("\"eval_stage_{}\"]", s - 1)));
    }
}

/// Pinned rewrite counts and sizes for today's generated code: a generator change that keeps the
/// shapes but changes the code shows up here (update the numbers after checking the new output).
#[test]
fn pinned_counts() {
    let (header, sources) = inputs();
    let tr: Translation = translate(&header, &sources, 58);
    assert_eq!(
        tr.counts,
        RewriteCounts {
            functions: 58,
            scratch_params: 57,
            input_params: 62,
            poly_mix_reads: 13490,
            calls: 0,
            cuts: 57,
            constants: 2,
            body_lines: 90630,
        }
    );
    assert_eq!(tr.scratch_elems, 8237);
    assert_eq!(tr.max_column, 1095);
    assert_eq!(
        tr.constants,
        vec![
            ("INV_RATE".to_string(), 4),
            ("kNumPolyMixPows".to_string(), 2175)
        ]
    );
    assert_eq!(tr.carry_words, 20);
    let rs = tr.rust_consts();
    assert!(rs.contains("pub const EVAL_SCRATCH_ELEMS: u32 = 8237;"));
    assert!(rs.contains("pub const EVAL_CARRY_WORDS: u32 = 20;"));
    assert!(rs.contains("pub const EVAL_NUM_POLY_MIX_POWS: u32 = 2175;"));
    assert!(rs.contains("pub const EVAL_INV_RATE: u32 = 4;"));
    assert!(rs.contains("pub const EVAL_BUF_PARAMS: u64 = 9;"));
}

/// A changed generator must fail the translation loudly, not produce other code.
#[test]
fn rejects_changed_generator() {
    let (header, sources) = inputs();
    let poly = sources
        .iter()
        .position(|(_, t)| t.contains("FpExt poly_fp("))
        .unwrap();
    let mutate_source = |i: usize, from: &str, to: &str| -> Vec<(String, String)> {
        let mut s = sources.clone();
        assert!(
            s[i].1.contains(from),
            "test setup: `{from}` not in {}",
            s[i].0
        );
        s[i].1 = s[i].1.replacen(from, to, 1);
        s
    };
    let cases: Vec<(&str, String, Sources)> = vec![
        (
            "unknown line",
            header.clone(),
            mutate_source(
                0,
                "  uint32_t mask = size - 1;\n",
                "  uint32_t mask = size - 1;\n  Fp q = 7;\n",
            ),
        ),
        (
            "`Fp *` spelling",
            header.clone(),
            mutate_source(0, "Fp* arg0", "Fp *arg0"),
        ),
        (
            "second array",
            header.clone(),
            mutate_source(
                0,
                "  uint32_t mask = size - 1;\n",
                "  uint32_t mask = size - 1;\n  Fp x99999[4];\n",
            ),
        ),
        (
            "64-bit op",
            header.clone(),
            mutate_source(
                0,
                "  uint32_t mask = size - 1;\n",
                "  uint32_t mask = size - 1;\n  Fp x99999 = x1 / x2;\n",
            ),
        ),
        (
            "unknown constant",
            header.replacen(
                "constexpr size_t INV_RATE = 4;",
                "constexpr size_t INV_RATE = 4;\nconstexpr size_t FOO = 1;",
                1,
            ),
            sources.clone(),
        ),
        (
            "unknown parameter type",
            header.clone(),
            mutate_source(poly, "const Fp* accum) {", "const FpExt* accum) {"),
        ),
        ("call isn't a tail call", header.clone(), {
            let mut s = sources.clone();
            let (i, at) = s
                .iter()
                .enumerate()
                .find_map(|(i, (_, t))| t.find("  return x").map(|at| (i, at)))
                .unwrap();
            s[i].1.insert_str(at, "  Fp x99999(1);\n");
            s
        }),
        ("missing file", header.clone(), sources[1..].to_vec()),
        (
            "broken chain",
            header.clone(),
            mutate_source(poly, "auto x", "auto y"),
        ),
        (
            "write to an input",
            header.clone(),
            mutate_source(
                poly,
                "  uint32_t mask = size - 1;\n",
                "  uint32_t mask = size - 1;\n  data[0] = data[1];\n",
            ),
        ),
    ];
    for (what, header, sources) in cases {
        let r = catch_unwind(AssertUnwindSafe(|| translate(&header, &sources, 58)));
        assert!(r.is_err(), "{what}: translation should have panicked");
    }
}

fn toolchain_present() -> bool {
    Command::new("xcrun")
        .args(["--sdk", "macosx", "--find", "metal"])
        .output()
        .is_ok_and(|o| o.status.success())
}

fn include_dirs() -> [PathBuf; 2] {
    [
        manifest().join("kernels/metal"),
        manifest().join("../../build_kernel/kernels/metal"),
    ]
}

/// Compile the stages with `xcrun metal` (front end + AIR; the GPU back end runs at PSO creation).
#[test]
#[ignore = "minutes of xcrun metal; run explicitly (see the module doc)"]
fn msl_compiles() {
    if !toolchain_present() {
        println!("SKIPPED: no xcrun metal (source /Users/user/zkminer/env.sh)");
        return;
    }
    let counts: Vec<usize> = std::env::var("K3_STAGE_COUNTS")
        .unwrap_or_else(|_| "58,29,16,8".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let jobs: usize = std::env::var("K3_JOBS")
        .ok()
        .and_then(|j| j.parse().ok())
        .unwrap_or(6);
    let extra: Vec<String> = std::env::var("K3_METAL_FLAGS")
        .map(|f| f.split_whitespace().map(String::from).collect())
        .unwrap_or_default();
    let (header, sources) = inputs();
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("k3");
    for count in counts {
        let tr = translate(&header, &sources, count);
        let dir = root.join(count.to_string());
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut work = Vec::new();
        for ((name, text), st) in tr.files.iter().zip(&tr.stages) {
            let src = dir.join(name);
            std::fs::write(&src, text).unwrap();
            work.push((src, st.clone()));
        }
        let queue = Arc::new(Mutex::new(work));
        let results = Arc::new(Mutex::new(Vec::new()));
        let started = Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..jobs {
                let queue = queue.clone();
                let results = results.clone();
                let extra = &extra;
                scope.spawn(move || loop {
                    let Some((src, st)) = queue.lock().unwrap().pop() else {
                        break;
                    };
                    let air = src.with_extension("air");
                    let mut cmd = Command::new("xcrun");
                    cmd.args(["--sdk", "macosx", "metal", "-std=metal3.2", "-O2", "-c"])
                        .arg(&src)
                        .arg("-o")
                        .arg(&air);
                    for inc in include_dirs() {
                        cmd.arg("-I").arg(inc);
                    }
                    cmd.args(extra);
                    let t = Instant::now();
                    let out = cmd.output().unwrap();
                    let secs = t.elapsed().as_secs_f64();
                    assert!(
                        out.status.success(),
                        "{}: {}",
                        src.display(),
                        String::from_utf8_lossy(&out.stderr)
                    );
                    let warnings = String::from_utf8_lossy(&out.stderr)
                        .matches("warning:")
                        .count();
                    results.lock().unwrap().push((st, secs, air, warnings));
                });
            }
        });
        let wall = started.elapsed().as_secs_f64();
        let mut results = std::mem::take(&mut *results.lock().unwrap());
        results.sort_by_key(|(st, ..)| st.name[11..].parse::<usize>().unwrap());
        let lib = dir.join("eval.metallib");
        let mut link = Command::new("xcrun");
        link.args(["--sdk", "macosx", "metallib", "-o"]).arg(&lib);
        link.args(results.iter().map(|(_, _, air, _)| air));
        let t = Instant::now();
        assert!(link.status().unwrap().success(), "metallib link");
        let link_secs = t.elapsed().as_secs_f64();
        println!(
            "== {count} stages: {} files, {jobs} jobs, wall {wall:.1} s, link {link_secs:.1} s, {}",
            results.len(),
            lib.display()
        );
        let sum: f64 = results.iter().map(|r| r.1).sum();
        for (st, secs, _, warnings) in &results {
            println!(
                "{:>14} {:>6} lines {:>8.2} s  {} warnings  {}",
                st.name,
                st.lines,
                secs,
                warnings,
                st.functions.join(",")
            );
        }
        let (big, big_secs) = results
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(st, s, ..)| (st.name.clone(), *s))
            .unwrap();
        let biggest = results.iter().max_by_key(|r| r.0.lines).unwrap();
        println!(
            "== {count} stages: sum {sum:.1} s, slowest {big} {big_secs:.2} s, largest {} ({} lines) {:.2} s",
            biggest.0.name, biggest.0.lines, biggest.1
        );
    }
}

/// The metallib cargo built (`--features metal`) has every stage kernel the build generated.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn metallib_has_every_stage() {
    let lib = risc0_circuit_keccak_sys::METAL_LIB;
    let Some(stages) = option_env!("KECCAK_METAL_EVAL_STAGES") else {
        // RISC0_SKIP_BUILD_KERNELS: no generated sources, empty metallib.
        assert!(lib.is_empty());
        println!("SKIPPED: kernels not built");
        return;
    };
    let stages: usize = stages.parse().unwrap();
    let expected = std::env::var(build_metal_eval::STAGES_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(build_metal_eval::DEFAULT_STAGES)
        .min(58);
    assert_eq!(
        stages,
        expected,
        "stage count built vs {}",
        build_metal_eval::STAGES_ENV
    );
    let has = |name: &str| {
        let pat = format!("{name}\0");
        lib.windows(pat.len()).any(|w| w == pat.as_bytes())
    };
    for k in 0..stages {
        assert!(
            has(&format!("eval_stage_{k}")),
            "eval_stage_{k} missing from METAL_LIB"
        );
    }
    assert!(
        !has(&format!("eval_stage_{stages}")),
        "stale eval_stage_{stages} in METAL_LIB"
    );
    assert!(has("scatter_preflight"));
    println!(
        "METAL_LIB ({} bytes) has eval_stage_0..{}",
        lib.len(),
        stages - 1
    );
}
