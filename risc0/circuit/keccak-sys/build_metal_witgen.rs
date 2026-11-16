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

//! The Metal witgen translator (item K6, docs/metal-port/PLAN.md 4.4 and 8.2), a `#[path]` module
//! of build.rs (and of keccak-sys/tests/k6_witgen_translate.rs, which tests [translate] directly).
//!
//! The generated CUDA step code (`kernels/cuda/steps_<n>.cu`, 54 `__device__` functions, ~35k
//! lines) is one tree under `step_Top`, which CUDA runs whole in one `par_stepExec` thread per
//! cycle. `step_Top` is: a prologue (the cycle counter, columns 932..934), `step_Top_0` on the
//! first cycle or `step_Top_24` (the next control state, from the previous row's one-hot) on the
//! others, the one-hot of the cycle type (columns 0..11), and then one call per cycle type: the
//! "arms" `step_Top_K`, K in 25..33, 50, 51, 52. A cycle runs exactly one arm.
//!
//! So instead of one kernel for the whole tree, this emits one kernel per arm (`step_arm_<K>`,
//! one `.metal` file each in `$OUT_DIR/metal`): `step_Top` with every other arm's call replaced by
//! a fault (`WITGEN_FAULT_WRONG_ARM`), plus only the functions it reaches. The HAL groups the
//! cycles by (cycle_type, sub_type) and dispatches each group with its cycle type's arm kernel.
//! Everything else is the CUDA code as written; the rewrites are:
//! - `__device__ void f(ExecContext& ctx, MutableBuf a, GlobalBuf b)` becomes
//!   `R0_KECCAK_WITGEN_FN void f(thread ExecContext& ctx, MutableBuf a, GlobalBuf b)` inside
//!   `namespace keccak_witgen` (MutableBuf/GlobalBuf are by-value structs in
//!   kernels/metal/witgen.h, which also defines `get`/`set`/`setGlobal` and the builtins);
//! - `INVOKE_EXTERN(ctx, name...` becomes `extern_name(ctx...`;
//! - comment lines are dropped.
//!
//! Every token must be one the generator emits today (the builtins, `xN` locals, the function's
//! parameters, `step_Top_N` calls, integer literals, `( ) , ; = + - * { }`); every column literal
//! must be below the buffer's register count (defs.cu.inc), every `get`/`set` must name a
//! `MutableBuf` parameter and every `setGlobal` a `GlobalBuf` one; calls must match the callee's
//! parameter kinds; locals must be declared before use and once; `step_Top` must have exactly the
//! structure above. Anything else panics with the offending line, so a changed generator fails
//! the build instead of producing a kernel that means something else.
//!
//! Also written, not compiled into the metallib: `$OUT_DIR/witgen_probe/witgen_whole_tree.metal`,
//! the whole tree as one kernel with every function `noinline` (the plan's PSO probe, K6 note).

// Some items are only used by keccak-sys/tests/k6_witgen_translate.rs.
#![allow(dead_code)]

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    path::{Path, PathBuf},
};

/// Cycle types (arms of `step_Top`).
pub(crate) const ARMS: usize = 12;

/// Kernel bindings (kernels/metal/witgen.h), in order.
pub(crate) const BUFFERS: [&str; 7] = [
    "DATA",
    "GLOBAL",
    "RUN_ORDER",
    "PREIMAGE_IDX",
    "PREIMAGES",
    "FAULT",
    "PARAMS",
];

/// Size of `WitgenParams` (witgen.h static_asserts it too).
pub(crate) const PARAMS_SIZE: usize = 32;

/// The generated witgen `.metal` sources to compile into the keccak metallib (empty under
/// `RISC0_SKIP_BUILD_KERNELS`), written under `gen_dir` (`$OUT_DIR/metal`); `kernel_dir`
/// (`kernels/metal`, on the include path) holds `witgen.h`. Always writes
/// `$OUT_DIR/keccak_metal_witgen.rs` (kernel names, arm table, bindings and fault bits for the
/// HAL, see [Translation::rust_consts]) and points `KECCAK_METAL_WITGEN_RS` at it.
pub(crate) fn metal_witgen_sources(kernel_dir: &Path, gen_dir: &Path) -> Vec<PathBuf> {
    let cuda_dir = kernel_dir
        .parent()
        .expect("kernels/metal has a parent")
        .join("cuda");
    let inputs = read_inputs(&cuda_dir, kernel_dir);
    for path in inputs.paths() {
        println!("cargo:rerun-if-changed={path}");
    }
    let tr = translate(&inputs);

    std::fs::create_dir_all(gen_dir).unwrap();
    let keep: BTreeSet<String> = tr.files.iter().map(|(n, _)| n.clone()).collect();
    for entry in std::fs::read_dir(gen_dir).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if name.starts_with("witgen_arm_") && name.ends_with(".metal") && !keep.contains(&name) {
            std::fs::remove_file(gen_dir.join(&name)).unwrap();
        }
    }
    let mut paths = Vec::new();
    for (name, text) in &tr.files {
        let path = gen_dir.join(name);
        write_if_changed(&path, text);
        paths.push(path);
    }

    let out_dir = gen_dir.parent().expect("gen_dir is under OUT_DIR");
    let probe_dir = out_dir.join("witgen_probe");
    std::fs::create_dir_all(&probe_dir).unwrap();
    write_if_changed(&probe_dir.join("witgen_whole_tree.metal"), &tr.whole_tree);

    let rs = out_dir.join("keccak_metal_witgen.rs");
    write_if_changed(&rs, &tr.rust_consts());
    println!("cargo:rustc-env=KECCAK_METAL_WITGEN_RS={}", rs.display());

    if std::env::var("RISC0_SKIP_BUILD_KERNELS").is_ok() {
        return Vec::new();
    }
    paths
}

fn write_if_changed(path: &Path, text: &str) {
    if std::fs::read_to_string(path).ok().as_deref() != Some(text) {
        std::fs::write(path, text).unwrap();
    }
}

/// The translator's inputs, as `(path, text)`.
pub(crate) struct Inputs {
    /// `kernels/cuda/defs.cu.inc` (register counts).
    pub defs: (String, String),
    /// `kernels/cuda/steps.cuh` (declarations).
    pub decls: (String, String),
    /// `kernels/cuda/steps_<n>.cu`, in `n` order.
    pub steps: Vec<(String, String)>,
    /// `kernels/metal/witgen.h` (fault bits).
    pub header: (String, String),
}

impl Inputs {
    fn paths(&self) -> Vec<&str> {
        let mut v = vec![
            self.defs.0.as_str(),
            self.decls.0.as_str(),
            self.header.0.as_str(),
        ];
        v.extend(self.steps.iter().map(|(p, _)| p.as_str()));
        v
    }
}

fn read(path: PathBuf) -> (String, String) {
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    (path.display().to_string(), text)
}

pub(crate) fn read_inputs(cuda_dir: &Path, metal_dir: &Path) -> Inputs {
    let mut files: Vec<(u32, PathBuf)> = std::fs::read_dir(cuda_dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.ok()?.path();
            let n = p
                .file_name()?
                .to_str()?
                .strip_prefix("steps_")?
                .strip_suffix(".cu")?
                .parse()
                .ok()?;
            Some((n, p))
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no kernels/cuda/steps_<n>.cu");
    Inputs {
        defs: read(cuda_dir.join("defs.cu.inc")),
        decls: read(cuda_dir.join("steps.cuh")),
        steps: files.into_iter().map(|(_, p)| read(p)).collect(),
        header: read(metal_dir.join("witgen.h")),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// `MutableBuf argN` (the `data` buffer).
    Mutable,
    /// `GlobalBuf argN` (the `global` buffer).
    Global,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Param {
    pub kind: Kind,
    pub name: String,
}

#[derive(Clone, Debug)]
pub(crate) struct Function {
    pub name: String,
    pub params: Vec<Param>,
    /// Code lines (comments dropped), already rewritten for MSL.
    pub body: Vec<String>,
    /// Functions called, in order of appearance.
    pub calls: Vec<String>,
}

/// One arm of `step_Top`: the kernel for cycles of `cycle_type`.
#[derive(Clone, Debug)]
pub(crate) struct Arm {
    pub cycle_type: usize,
    /// `step_Top_K`, the function the arm calls.
    pub callee: String,
    /// `step_arm_K`, the kernel.
    pub kernel: String,
    /// Functions in the kernel's file (besides its step_Top variant), in source order.
    pub functions: Vec<String>,
    /// Code lines in the kernel's file.
    pub lines: usize,
}

/// What the translator rewrote (pinned by keccak-sys/tests/k6_witgen_translate.rs).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Counts {
    pub files: usize,
    pub functions: usize,
    pub body_lines: usize,
    pub comment_lines: usize,
    pub gets: usize,
    pub sets: usize,
    pub set_globals: usize,
    pub externs: usize,
    pub calls: usize,
    pub data_cols: usize,
    pub global_cols: usize,
}

pub(crate) struct Translation {
    pub functions: BTreeMap<String, Function>,
    /// Function names in source order.
    pub order: Vec<String>,
    /// By cycle type.
    pub arms: Vec<Arm>,
    /// `(file name, text)` of each arm kernel, by cycle type.
    pub files: Vec<(String, String)>,
    /// The whole-tree probe kernel (`step_whole_tree`), not part of the metallib.
    pub whole_tree: String,
    /// `(name, value)` of witgen.h's `WITGEN_FAULT_*` bits.
    pub faults: Vec<(String, u32)>,
    pub counts: Counts,
}

const EXTERNS: [&str; 3] = ["getCycle", "getPreimage", "nextPreimage"];
const NAMESPACE_OPEN: &str = "namespace risc0::circuit::keccak::cuda {";
const NAMESPACE_CLOSE: &str = "} // namespace risc0::circuit::keccak::cuda";
const TOP: &str = "step_Top";

fn fail(path: &str, line: usize, text: &str, why: &str) -> ! {
    panic!("keccak witgen translator: {path}:{line}: {why}: {text:?}")
}

/// `constexpr size_t <name> = <n>;` from defs.cu.inc.
fn reg_count(defs: &(String, String), name: &str) -> usize {
    let prefix = format!("constexpr size_t {name} = ");
    let mut found = None;
    for l in defs.1.lines() {
        if let Some(v) = l.strip_prefix(&prefix) {
            let v = v
                .strip_suffix(';')
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| fail(&defs.0, 0, l, "unparsable register count"));
            assert!(found.is_none(), "{}: {name} defined twice", defs.0);
            found = Some(v);
        }
    }
    found.unwrap_or_else(|| panic!("keccak witgen translator: {}: no {name}", defs.0))
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Tok {
    Ident(String),
    Num(u64),
    Punct(char),
}

fn tokenize(path: &str, n: usize, line: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let b = line.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c == ' ' {
            i += 1;
        } else if c.is_ascii_alphabetic() || c == '_' {
            let s = i;
            while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(Tok::Ident(line[s..i].to_string()));
        } else if c.is_ascii_digit() {
            let s = i;
            while i < b.len() && (b[i] as char).is_ascii_digit() {
                i += 1;
            }
            let v: u64 = line[s..i].parse().unwrap();
            if v > u32::MAX as u64 {
                fail(path, n, line, "integer literal above u32");
            }
            if i < b.len() && ((b[i] as char).is_ascii_alphabetic() || b[i] == b'_') {
                fail(path, n, line, "integer literal with a suffix");
            }
            out.push(Tok::Num(v));
        } else if "(),;=+-*{}".contains(c) {
            out.push(Tok::Punct(c));
            i += 1;
        } else {
            fail(path, n, line, &format!("unexpected character {c:?}"));
        }
    }
    out
}

fn is_local(s: &str) -> bool {
    s.len() > 1 && s.starts_with('x') && s[1..].bytes().all(|c| c.is_ascii_digit())
}

fn is_step(s: &str) -> bool {
    s == TOP
        || s.strip_prefix("step_Top_")
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()))
}

/// Parse `ExecContext& ctx, MutableBuf a, GlobalBuf b` (CUDA) into the parameters after ctx.
fn parse_params(path: &str, n: usize, line: &str, list: &str) -> Vec<Param> {
    let mut parts = list.split(", ");
    if parts.next() != Some("ExecContext& ctx") {
        fail(path, n, line, "first parameter is not `ExecContext& ctx`");
    }
    let params: Vec<Param> = parts
        .map(|p| {
            let (kind, name) = if let Some(name) = p.strip_prefix("MutableBuf ") {
                (Kind::Mutable, name)
            } else if let Some(name) = p.strip_prefix("GlobalBuf ") {
                (Kind::Global, name)
            } else {
                fail(path, n, line, "parameter is not a MutableBuf or GlobalBuf");
            };
            if name.is_empty() || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
                fail(path, n, line, "bad parameter name");
            }
            Param {
                kind,
                name: name.to_string(),
            }
        })
        .collect();
    if params.is_empty() || params[0].kind != Kind::Mutable {
        fail(path, n, line, "first buffer parameter is not a MutableBuf");
    }
    let names: BTreeSet<&str> = params.iter().map(|p| p.name.as_str()).collect();
    if names.len() != params.len() {
        fail(path, n, line, "duplicate parameter name");
    }
    params
}

/// `__device__ void <name>(<params>)` + `suffix`, or None.
fn parse_signature(path: &str, n: usize, line: &str, suffix: &str) -> Option<(String, Vec<Param>)> {
    let rest = line.strip_prefix("__device__ void ")?;
    let rest = rest.strip_suffix(suffix).unwrap_or_else(|| {
        fail(
            path,
            n,
            line,
            &format!("signature does not end in {suffix:?}"),
        )
    });
    let (name, list) = rest
        .split_once('(')
        .unwrap_or_else(|| fail(path, n, line, "no parameter list"));
    if !is_step(name) {
        fail(path, n, line, "function is not step_Top or step_Top_<n>");
    }
    let list = list
        .strip_suffix(')')
        .unwrap_or_else(|| fail(path, n, line, "parameter list does not end in ')'"));
    Some((name.to_string(), parse_params(path, n, line, list)))
}

/// Check one code line of `f` (tokens and their meaning) and return it rewritten for MSL.
#[allow(clippy::too_many_arguments)]
fn translate_line(
    path: &str,
    n: usize,
    line: &str,
    params: &[Param],
    locals: &mut BTreeSet<String>,
    calls: &mut Vec<(usize, String, Vec<String>)>,
    counts: &mut Counts,
    data_cols: usize,
    global_cols: usize,
) -> String {
    let toks = tokenize(path, n, line);
    let ident = |i: usize| match toks.get(i) {
        Some(Tok::Ident(s)) => Some(s.as_str()),
        _ => None,
    };
    let punct = |i: usize, c: char| toks.get(i) == Some(&Tok::Punct(c));
    let num = |i: usize| match toks.get(i) {
        Some(Tok::Num(v)) => Some(*v),
        _ => None,
    };
    let param = |s: &str| params.iter().find(|p| p.name == s);
    let mut i = 0;
    while i < toks.len() {
        if let Tok::Ident(s) = &toks[i] {
            let s = s.as_str();
            match s {
                "get" | "set" | "setGlobal" => {
                    // get(ctx, P, COL, BACK) / set(ctx, P, COL, ... / setGlobal(ctx, P, COL, ...
                    let (kind, bound) = if s == "setGlobal" {
                        (Kind::Global, global_cols)
                    } else {
                        (Kind::Mutable, data_cols)
                    };
                    let p = ident(i + 4).and_then(param);
                    if !(punct(i + 1, '(')
                        && ident(i + 2) == Some("ctx")
                        && punct(i + 3, ',')
                        && p.is_some_and(|p| p.kind == kind)
                        && punct(i + 5, ','))
                    {
                        fail(path, n, line, &format!("unexpected {s} call"));
                    }
                    let col = num(i + 6)
                        .unwrap_or_else(|| fail(path, n, line, "column is not a literal"));
                    if col as usize >= bound || !punct(i + 7, ',') {
                        fail(
                            path,
                            n,
                            line,
                            &format!("column {col} out of range (< {bound})"),
                        );
                    }
                    if s == "get" {
                        if num(i + 8).is_none() || !punct(i + 9, ')') {
                            fail(path, n, line, "get's back is not a literal");
                        }
                        counts.gets += 1;
                    } else if s == "set" {
                        counts.sets += 1;
                    } else {
                        counts.set_globals += 1;
                    }
                    i += 8;
                    continue;
                }
                "INVOKE_EXTERN" => {
                    let name = ident(i + 4).unwrap_or("");
                    if !(punct(i + 1, '(')
                        && ident(i + 2) == Some("ctx")
                        && punct(i + 3, ',')
                        && EXTERNS.contains(&name))
                    {
                        fail(path, n, line, "unexpected INVOKE_EXTERN");
                    }
                    let ok = if name == "getPreimage" {
                        // INVOKE_EXTERN(ctx, getPreimage, Val(N)): N < 25 words * 4 halves.
                        punct(i + 5, ',')
                            && ident(i + 6) == Some("Val")
                            && punct(i + 7, '(')
                            && num(i + 8).is_some_and(|v| v < 100)
                            && punct(i + 9, ')')
                            && punct(i + 10, ')')
                    } else {
                        punct(i + 5, ')')
                    };
                    if !ok {
                        fail(
                            path,
                            n,
                            line,
                            &format!("unexpected arguments of extern {name}"),
                        );
                    }
                    counts.externs += 1;
                    i += 5;
                    continue;
                }
                "Val" => {
                    // A declaration `Val xN =`, or a literal `Val(N)` / conversion `Val(expr)`.
                    if let Some(x) = ident(i + 1) {
                        if !is_local(x) || !punct(i + 2, '=') {
                            fail(path, n, line, "unexpected declaration");
                        }
                        if !locals.insert(x.to_string()) {
                            fail(path, n, line, &format!("{x} declared twice"));
                        }
                        i += 2;
                        continue;
                    }
                    if !punct(i + 1, '(') {
                        fail(path, n, line, "unexpected use of Val");
                    }
                }
                "return" => {
                    if i != 0 || !punct(1, ';') || toks.len() != 2 {
                        fail(path, n, line, "unexpected return");
                    }
                }
                "if" => {
                    // `if (cond) {` on its own line.
                    if i != 0 || !punct(1, '(') || !punct(toks.len() - 1, '{') {
                        fail(path, n, line, "unexpected if");
                    }
                }
                _ if is_step(s) => {
                    // `step_Top_N(ctx, a, b);` on its own line.
                    let ok = i == 0
                        && punct(1, '(')
                        && ident(2) == Some("ctx")
                        && punct(toks.len() - 2, ')')
                        && punct(toks.len() - 1, ';');
                    if !ok {
                        fail(path, n, line, "unexpected call");
                    }
                    let mut args = Vec::new();
                    let mut j = 3;
                    while j < toks.len() - 2 {
                        if !punct(j, ',') {
                            fail(path, n, line, "unexpected call arguments");
                        }
                        let a = ident(j + 1)
                            .filter(|a| param(a).is_some())
                            .unwrap_or_else(|| {
                                fail(path, n, line, "call argument is not a parameter")
                            });
                        args.push(a.to_string());
                        j += 2;
                    }
                    calls.push((n, s.to_string(), args));
                    counts.calls += 1;
                    i = toks.len();
                    continue;
                }
                _ if is_local(s) => {
                    if !locals.contains(s) {
                        fail(path, n, line, &format!("{s} used before its declaration"));
                    }
                }
                "ctx" | "bitAnd" | "isz" | "inv_0" | "to_size_t" => {}
                _ if param(s).is_some() => {}
                _ => fail(path, n, line, &format!("unexpected identifier {s}")),
            }
        }
        i += 1;
    }
    let mut out = line.to_string();
    for e in EXTERNS {
        out = out.replace(
            &format!("INVOKE_EXTERN(ctx, {e}"),
            &format!("extern_{e}(ctx"),
        );
    }
    out
}

/// `#define WITGEN_FAULT_<X> <n>u` lines of witgen.h.
fn fault_bits(header: &(String, String)) -> Vec<(String, u32)> {
    let mut faults = Vec::new();
    for l in header.1.lines() {
        if let Some(rest) = l.strip_prefix("#define WITGEN_FAULT_") {
            let mut it = rest.split_whitespace();
            let name = it.next().unwrap();
            let v: u32 = it
                .next()
                .and_then(|v| v.strip_suffix('u'))
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| fail(&header.0, 0, l, "unparsable fault bit"));
            assert!(v.is_power_of_two(), "{}: {l}: not one bit", header.0);
            faults.push((format!("WITGEN_FAULT_{name}"), v));
        }
    }
    assert!(!faults.is_empty(), "{}: no WITGEN_FAULT_ bits", header.0);
    let all = faults.iter().fold(0u32, |a, (_, v)| {
        assert!(a & v == 0, "{}: fault bits overlap", header.0);
        a | v
    });
    assert!(all != 0);
    faults
}

/// The `if (cond) { call; }` blocks at the top level of step_Top: `(if line index, close line
/// index, condition, callee)`, indices into `body`.
fn top_blocks(path: &str, f: &Function) -> Vec<(usize, usize, String, String)> {
    let mut blocks = Vec::new();
    let mut i = 0;
    while i < f.body.len() {
        let l = &f.body[i];
        if l.starts_with("  if (") {
            let cond = l
                .strip_prefix("  if (")
                .and_then(|c| c.strip_suffix(") {"))
                .unwrap_or_else(|| fail(path, 0, l, "step_Top: unexpected if"));
            let close = i + 2;
            if f.body.get(close).map(String::as_str) != Some("  }") {
                fail(path, 0, l, "step_Top: an if block that is not one call");
            }
            let call = f.body[i + 1].trim();
            let callee = call
                .split_once('(')
                .map(|(c, _)| c)
                .filter(|c| is_step(c))
                .unwrap_or_else(|| fail(path, 0, call, "step_Top: an if block that is not a call"));
            blocks.push((i, close, cond.to_string(), callee.to_string()));
            i = close + 1;
        } else {
            if l.starts_with("   ") || l.starts_with("  }") {
                fail(path, 0, l, "step_Top: nested code outside the call blocks");
            }
            i += 1;
        }
    }
    blocks
}

/// Translate the CUDA step code (see the module doc).
pub(crate) fn translate(inputs: &Inputs) -> Translation {
    let data_cols = reg_count(&inputs.defs, "kRegCountData");
    let global_cols = reg_count(&inputs.defs, "kRegCountGlobal");
    let faults = fault_bits(&inputs.header);
    let mut counts = Counts {
        files: inputs.steps.len(),
        data_cols,
        global_cols,
        ..Default::default()
    };

    let mut functions: BTreeMap<String, Function> = BTreeMap::new();
    let mut order = Vec::new();
    let mut all_calls: Vec<(String, usize, String, String, Vec<String>)> = Vec::new();
    for (path, text) in &inputs.steps {
        let mut lines = text.lines().enumerate().map(|(i, l)| (i + 1, l));
        let mut seen_open = false;
        let mut seen_close = false;
        while let Some((n, line)) = lines.next() {
            if let Some((name, params)) = parse_signature(path, n, line, " {") {
                if !seen_open || seen_close {
                    fail(path, n, line, "function outside the namespace");
                }
                let mut body = Vec::new();
                let mut locals = BTreeSet::new();
                let mut calls = Vec::new();
                let mut depth = 0i32;
                loop {
                    let (n, l) = lines
                        .next()
                        .unwrap_or_else(|| fail(path, n, line, "unterminated function"));
                    if l == "}" {
                        break;
                    }
                    if l.trim_start().starts_with("//") {
                        counts.comment_lines += 1;
                        continue;
                    }
                    if l.trim().is_empty() {
                        fail(path, n, l, "blank line in a function");
                    }
                    if l.contains("//") {
                        fail(path, n, l, "trailing comment");
                    }
                    depth += l.matches('{').count() as i32 - l.matches('}').count() as i32;
                    if depth < 0 {
                        fail(path, n, l, "unbalanced braces");
                    }
                    body.push(translate_line(
                        path,
                        n,
                        l,
                        &params,
                        &mut locals,
                        &mut calls,
                        &mut counts,
                        data_cols,
                        global_cols,
                    ));
                }
                if depth != 0 {
                    fail(path, n, line, "unbalanced braces");
                }
                if body.last().map(|l| l.trim()) != Some("return;") {
                    fail(path, n, line, "function does not end in return");
                }
                counts.body_lines += body.len();
                for (cn, callee, args) in &calls {
                    all_calls.push((
                        path.clone(),
                        *cn,
                        name.clone(),
                        callee.clone(),
                        args.clone(),
                    ));
                }
                let f = Function {
                    name: name.clone(),
                    params,
                    body,
                    calls: calls.into_iter().map(|(_, c, _)| c).collect(),
                };
                if functions.insert(name.clone(), f).is_some() {
                    fail(path, n, line, "function defined twice");
                }
                order.push(name);
            } else if line == NAMESPACE_OPEN && !seen_open {
                seen_open = true;
            } else if line == NAMESPACE_CLOSE && seen_open && !seen_close {
                seen_close = true;
            } else if !(line.is_empty()
                || line.starts_with("//")
                || line == "#include \"steps.cuh\""
                || line == "#include \"witgen.h\"")
            {
                fail(path, n, line, "unexpected line outside a function");
            }
        }
        if !seen_open || !seen_close {
            fail(path, 0, "", "missing namespace");
        }
    }
    counts.functions = functions.len();

    // Calls: to a defined function, with arguments of the callee's parameter kinds.
    for (path, n, caller, callee, args) in &all_calls {
        let f = functions
            .get(callee)
            .unwrap_or_else(|| fail(path, *n, callee, "call of an undefined function"));
        let c = &functions[caller];
        let kinds: Vec<Kind> = args
            .iter()
            .map(|a| c.params.iter().find(|p| &p.name == a).unwrap().kind)
            .collect();
        let want: Vec<Kind> = f.params.iter().map(|p| p.kind).collect();
        if kinds != want {
            fail(
                path,
                *n,
                callee,
                "call arguments do not match the parameters",
            );
        }
        if callee == TOP {
            fail(path, *n, callee, "call of step_Top");
        }
    }

    // Declarations (steps.cuh) are exactly the definitions.
    let mut declared = BTreeMap::new();
    for (i, l) in inputs.decls.1.lines().enumerate() {
        if let Some(rest) = l.strip_prefix("extern ") {
            let (name, params) = parse_signature(&inputs.decls.0, i + 1, rest, ";")
                .unwrap_or_else(|| fail(&inputs.decls.0, i + 1, l, "unexpected declaration"));
            declared.insert(name, params);
        }
    }
    let defined: BTreeMap<String, Vec<Param>> = functions
        .iter()
        .map(|(n, f)| (n.clone(), f.params.clone()))
        .collect();
    assert_eq!(
        declared, defined,
        "keccak witgen translator: steps.cuh declarations differ from the steps_<n>.cu definitions"
    );

    // step_Top's structure.
    let top = functions
        .get(TOP)
        .unwrap_or_else(|| panic!("keccak witgen translator: no step_Top"));
    let top_path = &inputs.steps[0].0;
    let want_params = [Kind::Mutable, Kind::Global];
    if top.params.iter().map(|p| p.kind).collect::<Vec<_>>() != want_params {
        fail(
            top_path,
            0,
            TOP,
            "step_Top's parameters are not (MutableBuf, GlobalBuf)",
        );
    }
    let blocks = top_blocks(top_path, top);
    if blocks.len() != 2 + ARMS || blocks[0].3 != "step_Top_0" || blocks[1].3 != "step_Top_24" {
        fail(
            top_path,
            0,
            &format!("{:?}", blocks.iter().map(|b| &b.3).collect::<Vec<_>>()),
            "step_Top is not prologue + step_Top_0/step_Top_24 + 12 arms",
        );
    }
    // Each arm's condition is the one-hot column of its cycle type, read back right after it was
    // set: `Val xA = get(ctx, data0, J, 0);` ... `if (to_size_t(xA)) {`.
    let data0 = &top.params[0].name;
    let mut arm_of_type: Vec<Option<(usize, String)>> = vec![None; ARMS];
    for (bi, (_, _, cond, callee)) in blocks.iter().enumerate().skip(2) {
        let x = cond
            .strip_prefix("to_size_t(")
            .and_then(|c| c.strip_suffix(')'))
            .filter(|x| is_local(x))
            .unwrap_or_else(|| fail(top_path, 0, cond, "arm condition is not to_size_t(xN)"));
        let prefix = format!("  Val {x} = get(ctx, {data0}, ");
        let def = top
            .body
            .iter()
            .find_map(|l| l.strip_prefix(&prefix))
            .and_then(|r| r.strip_suffix(", 0);"))
            .and_then(|c| c.parse::<usize>().ok())
            .unwrap_or_else(|| fail(top_path, 0, x, "arm condition is not a one-hot column read"));
        if def >= ARMS || arm_of_type[def].is_some() {
            fail(
                top_path,
                0,
                cond,
                "arm conditions are not the 12 one-hot columns",
            );
        }
        if blocks.iter().filter(|b| &b.3 == callee).count() != 1 {
            fail(top_path, 0, callee, "arm function called twice");
        }
        arm_of_type[def] = Some((bi, callee.clone()));
    }
    let arm_of_type: Vec<(usize, String)> = arm_of_type.into_iter().map(Option::unwrap).collect();

    let reach = |roots: &[&str]| -> BTreeSet<String> {
        let mut seen = BTreeSet::new();
        let mut stack: Vec<String> = roots.iter().map(|s| s.to_string()).collect();
        while let Some(f) = stack.pop() {
            if seen.insert(f.clone()) {
                stack.extend(functions[&f].calls.iter().cloned());
            }
        }
        seen
    };
    // Every function is reachable from step_Top (nothing dead, nothing left out of every arm).
    let all = reach(&[TOP]);
    if all.len() != functions.len() {
        let dead: Vec<_> = functions.keys().filter(|f| !all.contains(*f)).collect();
        panic!("keccak witgen translator: functions not reachable from step_Top: {dead:?}");
    }

    let emit_fn = |out: &mut String, f: &Function, name: &str| {
        let params: Vec<String> = f
            .params
            .iter()
            .map(|p| match p.kind {
                Kind::Mutable => format!("MutableBuf {}", p.name),
                Kind::Global => format!("GlobalBuf {}", p.name),
            })
            .collect();
        writeln!(
            out,
            "R0_KECCAK_WITGEN_FN void {name}(thread ExecContext& ctx, {}) {{",
            params.join(", ")
        )
        .unwrap();
        for l in &f.body {
            writeln!(out, "{l}").unwrap();
        }
        writeln!(out, "}}").unwrap();
    };
    let emit_decl = |out: &mut String, f: &Function| {
        let params: Vec<String> = f
            .params
            .iter()
            .map(|p| match p.kind {
                Kind::Mutable => format!("MutableBuf {}", p.name),
                Kind::Global => format!("GlobalBuf {}", p.name),
            })
            .collect();
        writeln!(
            out,
            "R0_KECCAK_WITGEN_FN void {}(thread ExecContext& ctx, {});",
            f.name,
            params.join(", ")
        )
        .unwrap();
    };
    let preamble = |out: &mut String, what: &str| {
        writeln!(
            out,
            "// Generated by keccak-sys/build_metal_witgen.rs from kernels/cuda/steps_*.cu: {what}.\n\
             // Do not edit.\n"
        )
        .unwrap();
    };

    let mut arms = Vec::new();
    let mut files = Vec::new();
    for (cycle_type, (block, callee)) in arm_of_type.iter().enumerate() {
        let k = callee.strip_prefix("step_Top_").unwrap();
        let kernel = format!("step_arm_{k}");
        let variant = format!("step_Top_arm_{k}");
        let used = reach(&["step_Top_0", "step_Top_24", callee]);
        let fns: Vec<String> = order
            .iter()
            .filter(|f| used.contains(*f))
            .cloned()
            .collect();

        let mut top_arm = top.clone();
        for (bi, (open, _, _, _)) in blocks.iter().enumerate().skip(2) {
            if bi != *block {
                top_arm.body[open + 1] = "    ctx.fault |= WITGEN_FAULT_WRONG_ARM;".to_string();
            }
        }

        let mut out = String::new();
        preamble(
            &mut out,
            &format!("the {kernel} kernel, for cycles of cycle type {cycle_type}"),
        );
        writeln!(out, "#include \"witgen.h\"\n\nnamespace keccak_witgen {{\n").unwrap();
        for f in &fns {
            emit_decl(&mut out, &functions[f]);
        }
        writeln!(out).unwrap();
        for f in &fns {
            emit_fn(&mut out, &functions[f], f);
        }
        writeln!(
            out,
            "// step_Top (kernels/cuda/steps_0.cu) for cycle type {cycle_type}: only arm {callee} is \
             here;\n// a cycle whose one-hot selects another arm sets WITGEN_FAULT_WRONG_ARM."
        )
        .unwrap();
        emit_fn(&mut out, &top_arm, &variant);
        writeln!(
            out,
            "\n}} // namespace keccak_witgen\n\nR0_KECCAK_WITGEN_KERNEL({kernel}, keccak_witgen::{variant})"
        )
        .unwrap();
        let lines = out.lines().count();
        arms.push(Arm {
            cycle_type,
            callee: callee.clone(),
            kernel: kernel.clone(),
            functions: fns,
            lines,
        });
        files.push((format!("witgen_arm_{k}.metal"), out));
    }

    // The whole tree in one kernel, every function noinline (probe only).
    let mut whole_tree = String::new();
    preamble(
        &mut whole_tree,
        "the whole step_Top tree as one kernel, step_whole_tree (PSO probe; not in the metallib)",
    );
    writeln!(
        whole_tree,
        "#include \"witgen.h\"\n\nnamespace keccak_witgen {{\n"
    )
    .unwrap();
    for f in &order {
        emit_decl(&mut whole_tree, &functions[f]);
    }
    writeln!(whole_tree).unwrap();
    for f in &order {
        emit_fn(&mut whole_tree, &functions[f], f);
    }
    writeln!(
        whole_tree,
        "\n}} // namespace keccak_witgen\n\nR0_KECCAK_WITGEN_KERNEL(step_whole_tree, keccak_witgen::step_Top)"
    )
    .unwrap();

    Translation {
        functions,
        order,
        arms,
        files,
        whole_tree,
        faults,
        counts,
    }
}

impl Translation {
    /// The kernel names, by cycle type.
    pub(crate) fn kernel_names(&self) -> Vec<String> {
        self.arms.iter().map(|a| a.kernel.clone()).collect()
    }

    /// `$OUT_DIR/keccak_metal_witgen.rs`, included by keccak-sys's src/lib.rs.
    pub(crate) fn rust_consts(&self) -> String {
        let mut s = String::new();
        writeln!(
            s,
            "// Generated by keccak-sys/build_metal_witgen.rs (item K6). Do not edit.\n"
        )
        .unwrap();
        let names: Vec<String> = self
            .arms
            .iter()
            .map(|a| format!("{:?}", a.kernel))
            .collect();
        writeln!(
            s,
            "/// The witgen arm kernel of each cycle type (index = cycle type).\n\
             pub const WITGEN_ARM_KERNELS: [&str; {ARMS}] = [{}];",
            names.join(", ")
        )
        .unwrap();
        writeln!(
            s,
            "/// Every witgen kernel in `METAL_LIB`.\npub const WITGEN_KERNEL_NAMES: &[&str] = &WITGEN_ARM_KERNELS;"
        )
        .unwrap();
        for (i, b) in BUFFERS.iter().enumerate() {
            writeln!(s, "pub const WITGEN_BUF_{b}: usize = {i};").unwrap();
        }
        writeln!(s, "pub const WITGEN_NARGS: usize = {};", BUFFERS.len()).unwrap();
        writeln!(s, "/// `WitgenParams` (kernels/metal/witgen.h).").unwrap();
        writeln!(s, "pub const WITGEN_PARAMS_SIZE: usize = {PARAMS_SIZE};").unwrap();
        for (name, v) in &self.faults {
            writeln!(s, "pub const {name}: u32 = {v};").unwrap();
        }
        let fault_names: Vec<String> = self
            .faults
            .iter()
            .map(|(n, v)| format!("({v}, {:?})", n.trim_start_matches("WITGEN_FAULT_")))
            .collect();
        writeln!(
            s,
            "/// `(bit, name)` of every fault bit.\npub const WITGEN_FAULTS: &[(u32, &str)] = &[{}];",
            fault_names.join(", ")
        )
        .unwrap();
        writeln!(
            s,
            "/// Register counts the translator checked every column literal against.\n\
             pub const WITGEN_DATA_COLS: usize = {};\npub const WITGEN_GLOBAL_COLS: usize = {};",
            self.counts.data_cols, self.counts.global_cols
        )
        .unwrap();
        s
    }
}
