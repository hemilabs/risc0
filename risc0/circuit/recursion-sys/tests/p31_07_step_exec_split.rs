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

//! P31-07 (docs/metal-port/notes/P31-07.md): the step_exec split generator that the Metal build
//! runs (`kernels/cuda/split_step_exec_metal.py`).
//!
//! - `--selftest`: re-splitting upstream 3.0's monolithic `step_exec.cu` (the xz fixture in
//!   `kernels/cuda/testdata`) gives the hemilabs F30 split byte for byte, structural changes
//!   (renamed or moved arms, a missing micro instruction or constant, an already split input) are
//!   refused, a statement placed anywhere the split does not copy (between arms, in the tail
//!   outside an arm-guarded block, between macro sub-arms, before micro inst[0]) is refused instead
//!   of being dropped, a statement that the split would emit more than once per cycle (a call,
//!   a witness write or a witness read in macro_ops' write_addr prologue, which runs in the
//!   dispatcher and in all 9 sub-arm functions) is refused instead of being duplicated, the same
//!   statement inside an arm or tail block is kept, and the current
//!   `step_exec.cu` splits into the 19 noinline functions with its pinned sha256;
//! - the steps translator's `--check` on that split (pins, residual scan, round trip).
//!
//! Pure Python, no GPU and no Metal toolchain; runs without any feature.

use std::{path::PathBuf, process::Command};

fn run(script: &str, args: &[&str]) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels/cuda");
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let out = Command::new(&python)
        .arg("-I")
        .arg(dir.join(script))
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("running {python} {script}: {e}"));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    eprintln!("{script} {args:?}:\n{stdout}{stderr}");
    assert!(
        out.status.success(),
        "{script} {args:?} failed ({}):\n{stdout}\n{stderr}",
        out.status
    );
}

#[test]
fn split_generator_selftest() {
    run("split_step_exec_metal.py", &["--selftest"]);
}

#[test]
fn steps_translator_check_on_split() {
    run("cuda2metal_steps.py", &["--check", "--only", "step_exec"]);
}
