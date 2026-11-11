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

//! The rv32im Metal generators' own checks as cargo tests (Metal port item R16), so that
//! `cargo test -p risc0-circuit-rv32im-sys --features metal` runs them:
//!
//! - `cuda2metal.py --selftest`: every translator pass on snippets of the real sources, compiled
//!   with `xcrun metal`, the pins and their sensitivity, the refusals (needs the Metal toolchain:
//!   `source env.sh`);
//! - the CI drift checks of every generator (`--check`), which cover the Metal translations
//!   although their outputs are never checked in (they live in the build's `OUT_DIR`);
//! - every generator command this build ran (build.rs lists them in `RV32IM_METAL_GENERATORS`)
//!   re-run with `--check` against this build's outputs, and the same check failing on a stale
//!   copy of each output.
//!
//! The scripts run with `RISC0_PYTHON` if set, else `python3`. None of these tests touches
//! process-wide state, so they run under the default parallel harness.

#![cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]

use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};

fn crate_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn python() -> Command {
    let mut cmd = Command::new(std::env::var("RISC0_PYTHON").unwrap_or_else(|_| "python3".into()));
    cmd.current_dir(crate_dir())
        .env("PYTHONDONTWRITEBYTECODE", "1");
    cmd
}

fn run<S: AsRef<std::ffi::OsStr>>(args: &[S]) -> Output {
    let mut cmd = python();
    cmd.args(args);
    cmd.output()
        .unwrap_or_else(|e| panic!("could not run {cmd:?}: {e}"))
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn run_ok<S: AsRef<std::ffi::OsStr> + std::fmt::Debug>(args: &[S]) -> String {
    let out = run(args);
    let log = text(&out);
    assert!(
        out.status.success(),
        "{args:?} failed ({}):\n{log}",
        out.status
    );
    log
}

/// The generator commands of this build (`$OUT_DIR/metal_generators.txt`).
fn build_generators() -> Vec<Vec<String>> {
    let list = std::fs::read_to_string(env!("RV32IM_METAL_GENERATORS"))
        .expect("metal_generators.txt (written by build.rs)");
    list.lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.split('\t').map(String::from).collect())
        .collect()
}

fn skipped_build() -> bool {
    // RISC0_SKIP_BUILD_KERNELS: empty metallib, no generator ran.
    build_generators().is_empty()
}

fn with_check(cmd: &[String]) -> Vec<String> {
    let mut cmd = cmd.to_vec();
    if !cmd.iter().any(|a| a == "--check") {
        cmd.push("--check".into());
    }
    cmd
}

#[test]
fn cuda2metal_selftest() {
    let log = run_ok(&["kernels/cuda/cuda2metal.py", "--selftest"]);
    assert!(
        log.contains("cuda2metal selftest: all checks passed"),
        "{log}"
    );
}

/// What CI runs without a Metal build: every generator's `--check`, each of which now includes
/// its Metal translation (pins) besides its checked-in outputs.
#[test]
fn generator_ci_checks() {
    let log = run_ok(&["kernels/cuda/cuda2metal.py", "check"]);
    assert!(log.contains("cuda2metal check: all up to date"), "{log}");
    let log = run_ok(&["kernels/cuda/accum_tu_gen.py", "--check"]);
    assert!(
        log.contains("--metal translation matches PINNED_METAL"),
        "{log}"
    );
    // to_metal.py's pin is checked in this mode too (TO_METAL_EXPECT).
    run_ok(&["kernels/cuda/evalfast/regen.py", "--check"]);
    // R17 owns witgen_tu_gen.py; its --check covers witgen_tu.cu only (see the R16 note).
    run_ok(&["kernels/cuda/witgen_tu_gen.py", "--check"]);
}

/// The generated sources this build compiled are what the generators produce from the sources
/// now (each command re-run with `--check`).
#[test]
fn build_outputs_up_to_date() {
    if skipped_build() {
        eprintln!("RISC0_SKIP_BUILD_KERNELS build: no generator ran, nothing to check");
        return;
    }
    let cmds = build_generators();
    let scripts: Vec<&str> = cmds.iter().map(|c| c[0].as_str()).collect();
    for want in [
        "kernels/cuda/cuda2metal.py",
        "kernels/cuda/accum_tu_gen.py",
        "kernels/cuda/witgen_tu_gen.py",
        "kernels/cuda/evalfast/regen.py",
    ] {
        assert!(scripts.contains(&want), "{want} not in {scripts:?}");
    }
    for cmd in cmds.iter() {
        run_ok(&with_check(cmd));
    }
}

/// The `--check` runs above can fail: each generator's output, copied and changed by one line,
/// is reported out of date.
#[test]
fn check_detects_stale_output() {
    if skipped_build() {
        eprintln!("RISC0_SKIP_BUILD_KERNELS build: no generator ran, nothing to check");
        return;
    }
    let gen_dir = PathBuf::from(env!("RV32IM_METAL_GENERATORS"))
        .parent()
        .unwrap()
        .join("metal");
    let gen = gen_dir.to_str().unwrap();
    // generator -> its main output in gen_dir
    let outputs = [
        ("kernels/cuda/accum_tu_gen.py", "accum_tu.metal"),
        ("kernels/cuda/witgen_tu_gen.py", "witgen_tu.metal"),
        ("kernels/cuda/cuda2metal.py", "eval_check.metal"),
        ("kernels/cuda/evalfast/regen.py", "eval_check_ef.metal"),
    ];
    let mut checked = 0;
    for cmd in build_generators() {
        if !cmd.iter().any(|a| a == gen) {
            continue; // `cuda2metal.py incs -o kernels/metal --check`: checked-in files
        }
        let (_, file) = outputs
            .iter()
            .find(|(s, _)| *s == cmd[0])
            .unwrap_or_else(|| panic!("no known output for {cmd:?}"));
        let tmp = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("r16_stale")
            .join(file.replace('.', "_"));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        for entry in std::fs::read_dir(&gen_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_file() {
                std::fs::copy(&path, tmp.join(path.file_name().unwrap())).unwrap();
            }
        }
        let tmp_s = tmp.to_str().unwrap();
        let cmd: Vec<String> = cmd
            .iter()
            .map(|a| {
                if a == gen {
                    tmp_s.to_string()
                } else {
                    a.clone()
                }
            })
            .collect();
        let cmd = with_check(&cmd);
        // The unchanged copy passes ...
        run_ok(&cmd);
        // ... one extra line fails.
        let target = tmp.join(file);
        let mut body = std::fs::read_to_string(&target).unwrap();
        body.push_str("// stale\n");
        std::fs::write(&target, body).unwrap();
        let out = run(&cmd);
        assert!(
            !out.status.success(),
            "{cmd:?} accepted a changed {file}:\n{}",
            text(&out)
        );
        let _ = std::fs::remove_dir_all(&tmp);
        checked += 1;
    }
    assert_eq!(
        checked,
        outputs.len(),
        "generators writing into OUT_DIR/metal"
    );
}
