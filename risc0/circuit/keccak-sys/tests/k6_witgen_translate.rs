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

//! Item K6: the Metal witgen translator (build_metal_witgen.rs), tested directly.
//!
//! - `arm_layout`, `pinned_counts`, `rejects_changed_generator`: no toolchain needed.
//! - `msl_compiles` (ignored: a minute of `xcrun metal`): compiles every arm file and the
//!   whole-tree probe, prints the front-end time of each, and links
//!   `$CARGO_TARGET_TMPDIR/k6/{arms,whole_tree}.metallib` (used for PSO timing, see
//!   docs/metal-port/notes/K6.md). `K6_METAL_FLAGS` adds compiler flags (e.g.
//!   `-DR0_KECCAK_WITGEN_INLINE`).
//! - `metallib_has_every_arm` (`--features metal`): every `step_arm_<K>` is in `METAL_LIB`.

#![allow(dead_code)]

#[path = "../build_metal_witgen.rs"]
mod build_metal_witgen;

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use build_metal_witgen::{read_inputs, translate, Counts, Inputs, ARMS};

fn manifest() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn inputs() -> Inputs {
    read_inputs(
        &manifest().join("kernels/cuda"),
        &manifest().join("kernels/metal"),
    )
}

const ARM_FUNCTIONS: [u32; ARMS] = [25, 26, 27, 28, 29, 30, 31, 32, 33, 50, 51, 52];

#[test]
fn arm_layout() {
    let tr = translate(&inputs());
    assert_eq!(tr.arms.len(), ARMS);
    assert_eq!(tr.files.len(), ARMS);
    // step_Top_24's subtree (the next control state) is in every arm.
    let base: Vec<String> = std::iter::once(0)
        .chain(1..=24)
        .map(|k| format!("step_Top_{k}"))
        .collect();
    for (ct, (arm, (file, text))) in tr.arms.iter().zip(&tr.files).enumerate() {
        let k = ARM_FUNCTIONS[ct];
        assert_eq!(arm.cycle_type, ct);
        assert_eq!(arm.callee, format!("step_Top_{k}"));
        assert_eq!(arm.kernel, format!("step_arm_{k}"));
        assert_eq!(file, &format!("witgen_arm_{k}.metal"));
        assert!(text.contains(&format!(
            "R0_KECCAK_WITGEN_KERNEL(step_arm_{k}, keccak_witgen::step_Top_arm_{k})"
        )));
        for f in &base {
            assert!(arm.functions.contains(f), "{file}: {f}");
        }
        assert!(arm.functions.contains(&arm.callee));
        // No other arm's function, and every other arm call replaced by the fault.
        for (other_ct, &other) in ARM_FUNCTIONS.iter().enumerate() {
            let def = format!("void step_Top_{other}(thread ExecContext& ctx");
            assert_eq!(
                text.contains(&def),
                other_ct == ct,
                "{file}: step_Top_{other}"
            );
        }
        assert_eq!(
            text.matches("ctx.fault |= WITGEN_FAULT_WRONG_ARM;").count(),
            ARMS - 1
        );
        assert!(!text.contains("__device__") && !text.contains("INVOKE_EXTERN"));
        assert!(!text.contains("ExecContext& ctx,") || text.contains("thread ExecContext& ctx"));
        assert_eq!(
            text.matches("ExecContext& ctx").count(),
            text.matches("thread ExecContext& ctx").count()
        );
        // Only the generated header comments survive.
        let comments = text
            .lines()
            .filter(|l| l.trim_start().starts_with("//"))
            .count();
        assert!(comments <= 4, "{file}: {comments} comment lines");
    }
    // The SHA arm (50) is the largest: 17 functions of its own.
    let sha = &tr.arms[9];
    assert_eq!(sha.functions.len(), base.len() + 17);
    let rs = tr.rust_consts();
    assert!(rs.contains(
        "pub const WITGEN_ARM_KERNELS: [&str; 12] = [\"step_arm_25\", \"step_arm_26\", \
         \"step_arm_27\", \"step_arm_28\", \"step_arm_29\", \"step_arm_30\", \"step_arm_31\", \
         \"step_arm_32\", \"step_arm_33\", \"step_arm_50\", \"step_arm_51\", \"step_arm_52\"];"
    ));
    assert!(rs.contains("pub const WITGEN_BUF_PARAMS: usize = 6;"));
    assert!(rs.contains("pub const WITGEN_FAULT_WRONG_ARM: u32 = 4;"));
    // The whole-tree probe has every function and the unmodified step_Top.
    assert!(tr
        .whole_tree
        .contains("R0_KECCAK_WITGEN_KERNEL(step_whole_tree, keccak_witgen::step_Top)"));
    for f in &tr.order {
        assert!(tr
            .whole_tree
            .contains(&format!("void {f}(thread ExecContext& ctx")));
    }
    assert!(!tr.whole_tree.contains("WITGEN_FAULT_WRONG_ARM;"));
}

/// Pinned counts and sizes for today's generated code: a generator change that keeps the shapes
/// but changes the code shows up here (update the numbers after checking the new output).
#[test]
fn pinned_counts() {
    let tr = translate(&inputs());
    assert_eq!(
        tr.counts,
        Counts {
            files: 16,
            functions: 54,
            body_lines: 34712,
            comment_lines: 16986,
            gets: 13830,
            sets: 15456,
            set_globals: 16,
            externs: 102,
            calls: 53,
            data_cols: 1096,
            global_cols: 17,
        }
    );
    let lines: Vec<usize> = tr.arms.iter().map(|a| a.lines).collect();
    assert_eq!(
        lines,
        vec![1331, 2315, 2251, 3004, 2259, 3536, 3536, 3799, 3757, 9826, 2143, 1299]
    );
}

/// A changed generator must fail the translation loudly, not produce other code.
#[test]
fn rejects_changed_generator() {
    let base = inputs();
    let top = base
        .steps
        .iter()
        .position(|(_, t)| t.contains("void step_Top(ExecContext& ctx"))
        .unwrap();
    let with = |f: &dyn Fn(&mut Inputs)| {
        let mut i = Inputs {
            defs: base.defs.clone(),
            decls: base.decls.clone(),
            steps: base.steps.clone(),
            header: base.header.clone(),
        };
        f(&mut i);
        i
    };
    let mutate = |idx: usize, from: &'static str, to: &'static str| {
        with(&move |i: &mut Inputs| {
            assert!(
                i.steps[idx].1.contains(from),
                "test setup: `{from}` not in {}",
                i.steps[idx].0
            );
            i.steps[idx].1 = i.steps[idx].1.replacen(from, to, 1);
        })
    };
    let cases: Vec<(&str, Inputs)> = vec![
        (
            "column past kRegCountData",
            mutate(
                top,
                "set(ctx, data0, 932, x2);",
                "set(ctx, data0, 1096, x2);",
            ),
        ),
        (
            "unknown builtin",
            mutate(
                top,
                "set(ctx, data0, 933, isz(x3));",
                "set(ctx, data0, 933, isnz(x3));",
            ),
        ),
        (
            "setGlobal on the data buffer",
            mutate(
                top,
                "set(ctx, data0, 932, x2);",
                "setGlobal(ctx, data0, 932, x2);",
            ),
        ),
        (
            "get on the global buffer",
            mutate(
                top,
                "Val x3 = get(ctx, data0, 932, 0);",
                "Val x3 = get(ctx, global1, 0, 0);",
            ),
        ),
        (
            "local used before declaration",
            mutate(
                top,
                "set(ctx, data0, 932, x2);",
                "set(ctx, data0, 932, x99);",
            ),
        ),
        (
            "local declared twice",
            mutate(
                top,
                "Val x3 = get(ctx, data0, 932, 0);",
                "Val x2 = get(ctx, data0, 932, 0);",
            ),
        ),
        (
            "unknown extern",
            mutate(
                top,
                "INVOKE_EXTERN(ctx, getCycle)",
                "INVOKE_EXTERN(ctx, isFirstCycle)",
            ),
        ),
        (
            "unknown operator",
            mutate(
                top,
                "set(ctx, data0, 933, isz(x3));",
                "set(ctx, data0, 933, (x3 / x3));",
            ),
        ),
        (
            "an arm removed",
            mutate(
                top,
                "  if (to_size_t(x29)) {\n    step_Top_52(ctx, data0);\n  }\n",
                "",
            ),
        ),
        (
            "arm condition not a one-hot read",
            mutate(top, "  if (to_size_t(x29)) {", "  if (to_size_t(x5)) {"),
        ),
        (
            "declaration differs",
            with(&|i: &mut Inputs| {
                i.decls.1 = i.decls.1.replacen(
                    "step_Top_52(ExecContext& ctx, MutableBuf arg0);",
                    "step_Top_52(ExecContext& ctx, MutableBuf arg0, GlobalBuf arg1);",
                    1,
                )
            }),
        ),
        (
            "missing file",
            with(&|i: &mut Inputs| {
                i.steps.remove(1);
            }),
        ),
        (
            "64-bit literal",
            mutate(
                top,
                "set(ctx, data0, 933, isz(x3));",
                "set(ctx, data0, 933, Val(4294967296));",
            ),
        ),
        (
            "fault bits removed from witgen.h",
            with(&|i: &mut Inputs| {
                i.header.1 = i.header.1.replace("#define WITGEN_FAULT_", "#define X_")
            }),
        ),
    ];
    for (what, inputs) in cases {
        let r = catch_unwind(AssertUnwindSafe(|| translate(&inputs)));
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

/// Compile the arm files and the whole-tree probe with `xcrun metal` (front end + AIR).
#[test]
#[ignore = "a minute of xcrun metal; run explicitly (see the module doc)"]
fn msl_compiles() {
    if !toolchain_present() {
        println!("SKIPPED: no xcrun metal (source /Users/user/zkminer/env.sh)");
        return;
    }
    let extra: Vec<String> = std::env::var("K6_METAL_FLAGS")
        .map(|f| f.split_whitespace().map(String::from).collect())
        .unwrap_or_default();
    let tr = translate(&inputs());
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("k6");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut files: Vec<(String, String)> = tr.files.clone();
    files.push(("witgen_whole_tree.metal".into(), tr.whole_tree.clone()));
    let results: Vec<(String, f64, PathBuf, usize)> = std::thread::scope(|s| {
        let handles: Vec<_> = files
            .iter()
            .map(|(name, text)| {
                let dir = &dir;
                let extra = &extra;
                s.spawn(move || {
                    let src = dir.join(name);
                    std::fs::write(&src, text).unwrap();
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
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    assert!(out.status.success(), "{}: {stderr}", src.display());
                    (name.clone(), secs, air, stderr.matches("warning:").count())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (name, secs, _, warnings) in &results {
        println!("{name:>28} {secs:>7.2} s  {warnings} warnings");
        assert_eq!(*warnings, 0, "{name}");
    }
    let link = |out: &str, airs: Vec<&PathBuf>| {
        let lib = dir.join(out);
        let ok = Command::new("xcrun")
            .args(["--sdk", "macosx", "metallib", "-o"])
            .arg(&lib)
            .args(airs)
            .status()
            .unwrap()
            .success();
        assert!(ok, "metallib link {out}");
        println!("linked {}", lib.display());
    };
    let (whole, arms): (Vec<_>, Vec<_>) = results
        .iter()
        .partition(|r| r.0 == "witgen_whole_tree.metal");
    link("arms.metallib", arms.iter().map(|r| &r.2).collect());
    link("whole_tree.metallib", whole.iter().map(|r| &r.2).collect());
}

/// The metallib cargo built (`--features metal`) has every arm kernel.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn metallib_has_every_arm() {
    use risc0_circuit_keccak_sys::metal_witgen::{WITGEN_ARM_KERNELS, WITGEN_KERNEL_NAMES};
    let lib = risc0_circuit_keccak_sys::METAL_LIB;
    if lib.is_empty() {
        // RISC0_SKIP_BUILD_KERNELS: no generated sources, empty metallib.
        println!("SKIPPED: kernels not built");
        return;
    }
    let has = |name: &str| {
        let pat = format!("{name}\0");
        lib.windows(pat.len()).any(|w| w == pat.as_bytes())
    };
    let tr = translate(&inputs());
    let names: std::collections::BTreeSet<String> = tr.kernel_names().into_iter().collect();
    let built: std::collections::BTreeSet<String> =
        WITGEN_KERNEL_NAMES.iter().map(|s| s.to_string()).collect();
    assert_eq!(names, built, "the HAL's names vs the translator's");
    for (ct, name) in WITGEN_ARM_KERNELS.iter().enumerate() {
        assert_eq!(*name, format!("step_arm_{}", ARM_FUNCTIONS[ct]));
        assert!(has(name), "{name} missing from METAL_LIB");
    }
    assert!(
        !has("step_whole_tree"),
        "the probe kernel is not part of the metallib"
    );
    assert!(has("scatter_preflight"));
    println!(
        "METAL_LIB ({} bytes) has the {} witgen arms",
        lib.len(),
        ARMS
    );
}
