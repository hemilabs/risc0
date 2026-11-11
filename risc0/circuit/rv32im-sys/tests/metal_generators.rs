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
//! process-wide state, so they run under the default parallel harness; scratch copies are per
//! process (`Scratch`), so overlapping `cargo test` runs of this crate do not collide either.

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
    // R17: witgen_tu_gen.py --check also translates --metal in memory (pins, no-loop check).
    let log = run_ok(&["kernels/cuda/witgen_tu_gen.py", "--check"]);
    assert!(
        log.contains("--metal translates") && log.contains("map lowering unroll"),
        "{log}"
    );
}

/// `witgen_tu_gen.py --check` fails when the Metal translation no longer matches PINNED_METAL,
/// and the translation refuses a loop in the call graph (R17: the shape of the Apple GPU
/// lost-store defect, notes/R17.md), whatever the pins say.
#[test]
fn witgen_check_detects_pin_mismatch_and_loops() {
    let script = r#"
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, 'kernels/cuda')
import witgen_tu_gen as g
import cuda2metal as c2m
g.gen_metal()
g.gen_metal(map_lowering='loop')
g.PINNED_METAL['counts']['lambda.map.unroll'] += 1
try:
    g.main_cuda(True)
except SystemExit as e:
    assert e.code not in (None, 0), e.code
    print('pin mismatch rejected')
else:
    raise AssertionError('--check accepted a changed pin')
g.PINNED_METAL['counts']['lambda.map.unroll'] -= 1
orig = c2m.translate
def with_loop(*a, **k):
    t, c = orig(*a, **k)
    i = t.rindex('} // namespace rv32im_metal')
    return t[:i] + 'static void c2m_extra() { for (uint i = 0; i < 2; i++) {} }\n' + t[i:], c
c2m.translate = with_loop
try:
    g.gen_metal(no_pin=True)
except c2m.TranslateError as e:
    assert 'loop' in str(e), e
    print('loop rejected')
else:
    raise AssertionError('a loop in the call graph was accepted')
"#;
    let log = run_ok(&["-c", script]);
    assert!(
        log.contains("pin mismatch rejected") && log.contains("loop rejected"),
        "{log}"
    );
}

/// R17 fix round 1: what the witgen TU actually compiles -- witgen_tu.metal preprocessed with this
/// build's include path (`xcrun metal -E`) -- has no loop outside the Metal SDK beyond the ones
/// reviewed in witgen_tu_gen.py's SCAN_LOOP_ALLOW, includes the generated witgen_fpext.h (the
/// shared fpext.h with its 4-element loops unrolled) and not the shared fpext.h. Then the scanner
/// must reject an extra loop in a header, and gen_fpext a loop it cannot unroll.
#[test]
fn witgen_tu_preprocessed_loops_reviewed() {
    if skipped_build() {
        eprintln!("RISC0_SKIP_BUILD_KERNELS build: no generator ran, nothing to check");
        return;
    }
    let out_dir = PathBuf::from(env!("RV32IM_METAL_GENERATORS"))
        .parent()
        .unwrap()
        .to_path_buf();
    let gen_dir = out_dir.join("metal");
    let scratch = Scratch::new("witgen_tu_i");
    let pre = scratch.0.join("witgen_tu.i");
    let out = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-E", "-std=metal3.2", "-I"])
        .arg(out_dir.join("_sys_"))
        .arg("-I")
        .arg(crate_dir().join("kernels/metal"))
        .arg("-I")
        .arg(&gen_dir)
        .arg(gen_dir.join("witgen_tu.metal"))
        .arg("-o")
        .arg(&pre)
        .output()
        .expect("xcrun metal -E (source env.sh)");
    assert!(out.status.success(), "metal -E failed:\n{}", text(&out));
    let body = std::fs::read_to_string(&pre).unwrap();
    assert!(
        body.contains("/witgen_fpext.h\""),
        "witgen_fpext.h not included"
    );
    assert!(
        !body.contains("_sys_/fpext.h\""),
        "the shared fpext.h is included"
    );
    // R17b: Fp from witgen_fp.h (every function always_inline), not the shared fp.h.
    assert!(body.contains("/witgen_fp.h\""), "witgen_fp.h not included");
    assert!(
        !body.contains("_sys_/fp.h\""),
        "the shared fp.h is included"
    );
    let log = run_ok(&[
        "kernels/cuda/witgen_tu_gen.py",
        "--scan-loops",
        pre.to_str().unwrap(),
    ]);
    assert!(log.contains("all reviewed"), "{log}");
    let script = r#"
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, 'kernels/cuda')
sys.argv = ['x']
import witgen_tu_gen as g
import cuda2metal as c2m
pre = open(PRE).read()
found, bad = g.scan_loops(pre)
assert not bad, bad
extra = pre + '\n# 1 "/x/kernels/metal/witgen.h"\ninline void f() { for (int j = 0; j < 2; j++) { g(); } }\n'
found, bad = g.scan_loops(extra)
assert any('witgen.h' in b for b in bad), bad
print('extra loop rejected')
src = open(g.FPEXT_SRC_DEFAULT).read()
g.gen_fpext(src)
try:
    g.gen_fpext(src.replace('constexpr inline FpExt zeroize() {',
                            'constexpr inline FpExt zeroize() {\n    for (int j = 0; j < 4; j++) {}', 1),
                no_pin=True)
except c2m.TranslateError as e:
    assert 'loop' in str(e), e
    print('fpext loop rejected')
g.PINNED_FPEXT['loop.unroll4'] += 1
try:
    g.gen_fpext(src)
except c2m.TranslateError as e:
    print('fpext pin rejected')
"#
    .replace("PRE", &format!("{:?}", pre.to_str().unwrap()));
    let log = run_ok(&["-c", &script]);
    assert!(
        log.contains("extra loop rejected")
            && log.contains("fpext loop rejected")
            && log.contains("fpext pin rejected"),
        "{log}"
    );
    drop(scratch);
}

/// The audit IR scanner (tools/metal_audit/ir_scan.py).
fn ir_scan_py() -> PathBuf {
    crate_dir().join("../../../tools/metal_audit/ir_scan.py")
}

/// `xcrun metal -S -emit-llvm` of `src_dir/witgen_tu.metal` with the build's include path and
/// `flags` (an -O level, -D...) into `ll`.
fn emit_witgen_ll(out_dir: &Path, src_dir: &Path, flags: &[&str], ll: &Path) {
    let out = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.2",
            "-Wno-unused-variable",
        ])
        .args(["-S", "-emit-llvm"])
        .args(flags)
        .arg("-c")
        .arg(src_dir.join("witgen_tu.metal"))
        .arg("-I")
        .arg(out_dir.join("_sys_"))
        .arg("-I")
        .arg(crate_dir().join("kernels/metal"))
        .arg("-I")
        .arg(src_dir)
        .arg("-o")
        .arg(ll)
        .output()
        .expect("xcrun metal -S -emit-llvm (source env.sh)");
    assert!(
        out.status.success(),
        "metal -emit-llvm {flags:?} failed:\n{}",
        text(&out)
    );
}

fn build_out_dir() -> PathBuf {
    PathBuf::from(env!("RV32IM_METAL_GENERATORS"))
        .parent()
        .unwrap()
        .to_path_buf()
}

/// R17 round 2 + R17b: the witgen TU as built, compiled to LLVM IR at -O2 (shipped), -O0, -Oz
/// and with compiler-decided inlining (`WITGEN_TU_FNATTR=` at -O2, -O0, -Oz):
/// - `--scan-ir`: no Fp / FpExt operation and no FpExt helper (inv_ext_chain, inv_0(FpExt),
///   loadExt, storeExt, ef_ext_mul) is defined out of line (round 2: the round-1 copy of fpext.h
///   lost elems[3] of PoseidonState.zcheck at -O0/-Oz), only the three kernels have external
///   linkage, and no call leaves the TU (R17b: the metallib linker could bind a linkonce_odr call
///   to another TU's copy; a stale accum_tu.air broke witgen that way);
/// - `ir_scan.py --fail-tiers ABS`: no tier A (constructor-initialized private aggregate written
///   after a call and read, in a loop or straight line), B (any aggregate written after a call in
///   a loop and read after it) or S (an out-of-line function writing a by-value aggregate after
///   or through a call, the round-2 trigger) site.
/// Positive controls: the round-1 copy (`--fpext-calls outline`, -O0) fails `--scan-ir`; the
/// original map-loop lowering (`--map-lowering loop`, -O2) fails the tier scan; gen_fpext and
/// gen_fp refuse a constexpr line they did not mark always_inline.
#[test]
fn witgen_tu_fpext_always_inline_in_ir() {
    if skipped_build() {
        eprintln!("RISC0_SKIP_BUILD_KERNELS build: no generator ran, nothing to check");
        return;
    }
    let out_dir = build_out_dir();
    let gen_dir = out_dir.join("metal");
    let scratch = Scratch::new("witgen_tu_ll");
    let settings: [(&str, &[&str]); 6] = [
        ("O2", &["-O2"]),
        ("O0", &["-O0"]),
        ("Oz", &["-Oz"]),
        ("auto", &["-O2", "-DWITGEN_TU_FNATTR="]),
        ("autoO0", &["-O0", "-DWITGEN_TU_FNATTR="]),
        ("autoOz", &["-Oz", "-DWITGEN_TU_FNATTR="]),
    ];
    for (tag, flags) in settings {
        let ll = scratch.0.join(format!("witgen_tu_{tag}.ll"));
        emit_witgen_ll(&out_dir, &gen_dir, flags, &ll);
        let log = run_ok(&[
            "kernels/cuda/witgen_tu_gen.py",
            "--scan-ir",
            ll.to_str().unwrap(),
        ]);
        assert!(
            log.contains("no out-of-line FpExt operation")
                && log.contains("no call outside the TU"),
            "{tag}: {log}"
        );
        let log = run_ok(&[
            "-I",
            ir_scan_py().to_str().unwrap(),
            "--tiers",
            "ABS",
            "--fail-tiers",
            "ABS",
            ll.to_str().unwrap(),
        ]);
        eprintln!("{tag}: {}", log.lines().nth(1).unwrap_or(""));
        std::fs::remove_file(&ll).unwrap();
    }
    // Positive control 1: the round-1 copies (no always_inline), compiled at -O0, are flagged.
    let base = scratch.0.join("outline");
    std::fs::create_dir_all(&base).unwrap();
    run_ok(&[
        "kernels/cuda/witgen_tu_gen.py",
        "--metal",
        "-o",
        base.to_str().unwrap(),
        "--fpext-src",
        out_dir.join("witgen_fpext_src.h").to_str().unwrap(),
        "--fp-src",
        out_dir.join("witgen_fp_src.h").to_str().unwrap(),
        "--fpext-calls",
        "outline",
    ]);
    let ll = base.join("witgen_tu.ll");
    emit_witgen_ll(&out_dir, &base, &["-O0"], &ll);
    let out = run(&[
        "kernels/cuda/witgen_tu_gen.py",
        "--scan-ir",
        ll.to_str().unwrap(),
    ]);
    assert!(
        !out.status.success() && text(&out).contains("out of line: _ZmlL5FpExt"),
        "--scan-ir did not flag the round-1 fpext copy:\n{}",
        text(&out)
    );
    // Positive control 2: the map-loop lowering at -O2 has tier-A/B sites (ToBits<5> & co).
    let loop_dir = scratch.0.join("loop");
    std::fs::create_dir_all(&loop_dir).unwrap();
    run_ok(&[
        "kernels/cuda/witgen_tu_gen.py",
        "--metal",
        "-o",
        loop_dir.to_str().unwrap(),
        "--fpext-src",
        out_dir.join("witgen_fpext_src.h").to_str().unwrap(),
        "--fp-src",
        out_dir.join("witgen_fp_src.h").to_str().unwrap(),
        "--map-lowering",
        "loop",
    ]);
    let ll = loop_dir.join("witgen_tu.ll");
    emit_witgen_ll(&out_dir, &loop_dir, &["-O2"], &ll);
    let out = run(&[
        "-I",
        ir_scan_py().to_str().unwrap(),
        "--tiers",
        "AB",
        "--fail-tiers",
        "AB",
        ll.to_str().unwrap(),
    ]);
    assert!(
        !out.status.success() && text(&out).contains("exec_ToBits_5"),
        "ir_scan.py did not flag the map-loop lowering:\n{}",
        text(&out)
    );
    let script = r#"
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, 'kernels/cuda')
sys.argv = ['x']
import witgen_tu_gen as g
import cuda2metal as c2m
src = open(g.FPEXT_SRC_DEFAULT).read()
txt, _, bad = g.gen_fpext(src)
assert not bad and txt.count(g.FPEXT_ALWAYS_INLINE) == g.PINNED_FPEXT['fn.always_inline'], bad
try:
    g.gen_fpext(src.replace('  constexpr Fp constPart() const', '  inline constexpr Fp constPart() const', 1), no_pin=True)
except c2m.TranslateError as e:
    assert 'always_inline' in str(e), e
    print('unmarked constexpr rejected')
fsrc = open(g.FP_SRC_DEFAULT).read()
txt, _, bad = g.gen_fp(fsrc)
assert not bad and txt.count(g.FPEXT_ALWAYS_INLINE) == g.PINNED_FP['fn.always_inline'], bad
try:
    g.gen_fp(fsrc.replace('  constexpr uint32_t asRaw() const', '  inline constexpr uint32_t asRaw() const', 1), no_pin=True)
except c2m.TranslateError as e:
    assert 'always_inline' in str(e), e
    print('unmarked fp constexpr rejected')
"#;
    let log = run_ok(&["-c", script]);
    assert!(
        log.contains("unmarked constexpr rejected")
            && log.contains("unmarked fp constexpr rejected"),
        "{log}"
    );
    drop(scratch);
}

/// R17b: the linked metallib's witgen kernels (par_stepExecW, fwd_stepExec, rev_stepExec)
/// contain only the witgen TU's own code: the build's metallib is disassembled
/// (`metal-objdump -d`) and every function of the three kernel modules must be one the witgen
/// TU (compiled here at -O2 with the build's flags) defines, with the same instruction count
/// (R17's -O0 build linked accum_tu's load / loadExt / inRange / inv_chain instead of its own).
/// The scanner's logic is also checked on a synthetic disassembly with a foreign function and
/// a foreign body.
#[test]
fn witgen_metallib_links_only_witgen_code() {
    if skipped_build() {
        eprintln!("RISC0_SKIP_BUILD_KERNELS build: no generator ran, nothing to check");
        return;
    }
    let out_dir = build_out_dir();
    let scratch = Scratch::new("witgen_metallib");
    let ll = scratch.0.join("witgen_tu.ll");
    emit_witgen_ll(&out_dir, &out_dir.join("metal"), &["-O2"], &ll);
    let log = run_ok(&[
        "kernels/cuda/witgen_tu_gen.py",
        "--scan-metallib",
        env!("RV32IM_METAL_PATH"),
        ll.to_str().unwrap(),
    ]);
    assert!(
        log.contains("par_stepExecW") && log.contains("every function from the witgen TU"),
        "{log}"
    );
    let script = r#"
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, 'kernels/cuda')
sys.argv = ['x']
import witgen_tu_gen as g
import re
ll = open(LL).read()
fns = re.findall(r'(?ms)^define .*?^}$', ll)[:3]
dis = ''.join('0x0000 -- %s:\n' % k + '\n'.join(fns) + '\n' for k in g.WITGEN_KERNELS)
mods, bad = g.scan_metallib(dis, ll)
assert not bad and all(len(v) == 3 for v in mods.values()), (bad, mods)
foreign = 'define internal void @_ZN12rv32im_metal7loadExtERNS_11ExecContextENS_11BoundLayoutINS_3RegEEEm() {\n  ret void\n}\n'
mods, bad = g.scan_metallib(dis + foreign, ll)
assert len(bad) == 1 and 'loadExt' in bad[0], bad
print('foreign function rejected')
grown = fns[0].replace('\n', '\n  %x = add i32 0, 0\n', 1)
mods, bad = g.scan_metallib(dis.replace(fns[0], grown, 1), ll)
assert len(bad) == 1 and 'another TU' in bad[0], bad
print('foreign body rejected')
"#
    .replace("LL", &format!("{:?}", ll.to_str().unwrap()));
    let log = run_ok(&["-c", &script]);
    assert!(
        log.contains("foreign function rejected") && log.contains("foreign body rejected"),
        "{log}"
    );
    drop(scratch);
}

/// `accum_tu_gen.py --check` fails when the Metal translation no longer matches PINNED_METAL
/// (before R16 it never translated --metal in --check and passed regardless): one pinned
/// rewrite count is changed in memory and the --check entry point must exit non-zero.
#[test]
fn accum_check_detects_pin_mismatch() {
    let script = r#"
import sys
sys.path.insert(0, 'kernels/cuda')
import accum_tu_gen as g
assert g.metal_pin_check() == [], g.metal_pin_check()
for mode in g.PINNED_METAL:
    pins = g.PINNED_METAL[mode]
    key = sorted(pins)[0]
    pins[key] += 1
    bad = g.metal_pin_check()
    assert any(mode in b for b in bad), (mode, bad)
    pins[key] -= 1
key = sorted(g.PINNED_METAL['always'])[0]
g.PINNED_METAL['always'][key] += 1
try:
    g.main_cuda(True)
except SystemExit as e:
    assert e.code not in (None, 0), e.code
    print('pin mismatch rejected')
else:
    raise AssertionError('--check accepted a changed pin')
"#;
    let log = run_ok(&["-c", script]);
    assert!(log.contains("pin mismatch rejected"), "{log}");
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

/// A scratch directory under CARGO_TARGET_TMPDIR unique to this process and `tag`; removed
/// when dropped (also while unwinding from a failed assertion).
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("r16_stale-{}", std::process::id()))
            .join(tag);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
        // The per-process parent, once its last output dir is gone.
        if let Some(parent) = self.0.parent() {
            let _ = std::fs::remove_dir(parent);
        }
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
        // Unique to this process and output, removed on success and on panic: concurrent
        // `cargo test` runs of this crate share CARGO_TARGET_TMPDIR (R16 fix round 1).
        let scratch = Scratch::new(&file.replace('.', "_"));
        let tmp = scratch.0.clone();
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
        drop(scratch);
        checked += 1;
    }
    assert_eq!(
        checked,
        outputs.len(),
        "generators writing into OUT_DIR/metal"
    );
}

/// Regression for R16 fix round 1: `check_detects_stale_output` used one fixed scratch dir, so
/// two overlapping `cargo test` runs of this crate corrupted each other's copies. Runs three more
/// copies of that test (this test binary, filtered to it) at the same time as each other and as
/// the in-harness copy; all must pass.
#[test]
fn stale_check_concurrent_processes() {
    const CHILD: &str = "R16_STALE_CHECK_CHILD";
    if std::env::var_os(CHILD).is_some() || skipped_build() {
        return;
    }
    let exe = std::env::current_exe().unwrap();
    // Staggered starts, so that one copy's steps (copy, check, append, check, remove) overlap
    // the others' at different points; started in lockstep they write identical copies and the
    // old shared dir did not fail.
    let children: Vec<_> = [0u64, 1000, 2500]
        .iter()
        .map(|&delay_ms| {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            Command::new(&exe)
                .args(["check_detects_stale_output", "--exact", "--test-threads=1"])
                .env(CHILD, "1")
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap_or_else(|e| panic!("could not run {exe:?}: {e}"))
        })
        .collect();
    for child in children {
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "concurrent check_detects_stale_output failed ({}):\n{}",
            out.status,
            text(&out)
        );
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("1 passed"),
            "{}",
            text(&out)
        );
    }
}
