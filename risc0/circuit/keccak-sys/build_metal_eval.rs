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

//! The Metal eval_check translator (item K3, docs/metal-port/PLAN.md 4.4 and 8.2), a `#[path]`
//! module of build.rs (and of keccak-sys/tests/k3_*.rs, which test [translate] directly).
//!
//! The generated CUDA constraint code (`kernels/cuda/eval_check.cuh` and `eval_check_<n>.cu`,
//! ~91k lines) is one chain of tail calls, `poly_fp -> keccak_56 -> ... -> keccak_0`, threading
//! `poly_fp`'s intermediate array `Fp x13[8237]` down as `Fp* argN` and reading the polynomial
//! mix powers from a `__constant__` global. As one Metal kernel it would never finish compiling
//! (plan 4.4 risks), so the chain is cut into `eval_stage_0 .. eval_stage_{S-1}` kernels, one
//! `.metal` file each, `S` = `RISC0_KECCAK_METAL_EVAL_STAGES` (default 58: one function a stage).
//!
//! The rewrites are those of build.rs's `write_intel_translation_unit` (the precedent), made for
//! MSL, on the parsed functions rather than on text:
//! - the scratch array becomes this thread's `ScratchRef` (`kernels/metal/eval_check_prelude.h`:
//!   element `i` at `scratch[i * stride + t]`), and every writable `Fp* argN` a `ScratchRef argN`;
//!   read-only `const Fp* argN` (classified first, so it never matches the `Fp*` rule) becomes
//!   `const device Fp* argN`;
//! - every function takes an `EvalCtx ectx` after `size` (poly mix, scratch, carry), every call
//!   passes it, and `poly_mix[` reads `ectx.poly_mix[` (a bound `device` buffer: 34.8 KB is too
//!   large for `constant` set_bytes);
//! - `__device__` is dropped: the chain is cut on the parsed functions first (the CUDA headers are
//!   what identify a function), then each is emitted as `R0_KECCAK_EVAL_FN FpExt name(...)`;
//! - the header's `constexpr size_t` constants become `constant constexpr uint32_t` (MSL program
//!   scope needs `constant`; `size_t` would make every column index 64-bit, which Apple GPUs
//!   emulate);
//! - at a cut, the call into the next stage's first function becomes explicit carry stores of its
//!   `FpExt` arguments (`carry_store(ectx, slot, xN)`, slot order = parameter order); that stage's
//!   kernel loads them back (`carry_load`) and passes the scratch and the kernel's own input
//!   buffers for the other parameters: every `const Fp*` parameter is traced back to the
//!   `poly_fp` parameter (`ctrl`, `out`, `data`, `mix` or `accum`) it always is, so no pointer is
//!   ever stored in the carry.
//!
//! Every input line must have one of the shapes the generator emits today; anything else panics
//! with the offending line, so a changed generator fails the build instead of producing a kernel
//! that means something else.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::Write as _,
    path::{Path, PathBuf},
};

/// Default number of stage kernels: one per function of the chain (plan 2.0).
pub(crate) const DEFAULT_STAGES: usize = 58;

/// Environment variable selecting the stage count (read at build time).
pub(crate) const STAGES_ENV: &str = "RISC0_KECCAK_METAL_EVAL_STAGES";

/// `poly_fp`'s read-only parameters, in order: the kernels bind them as buffers 1..=5.
pub(crate) const INPUTS: [&str; 5] = ["ctrl", "out", "data", "mix", "accum"];

/// The generated eval_check `.metal` sources to compile into the keccak metallib, written under
/// `gen_dir` (`$OUT_DIR/metal`); `kernel_dir` (`kernels/metal`, on the include path) holds
/// `eval_check_prelude.h`. Also writes `$OUT_DIR/keccak_metal_eval.rs` (stage names and sizes for
/// the HAL, see [Translation::rust_consts]) and points `KECCAK_METAL_EVAL_RS` at it. Called by
/// build.rs's `build_metal_kernels` only when kernels are built (not under
/// `RISC0_SKIP_BUILD_KERNELS`).
pub(crate) fn metal_eval_sources(kernel_dir: &Path, gen_dir: &Path) -> Vec<PathBuf> {
    let cuda_dir = kernel_dir
        .parent()
        .expect("kernels/metal has a parent")
        .join("cuda");
    let (header, sources) = read_cuda_sources(&cuda_dir);
    println!(
        "cargo:rerun-if-changed={}",
        cuda_dir.join("eval_check.cuh").display()
    );
    for (path, _) in &sources {
        println!("cargo:rerun-if-changed={path}");
    }
    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join("eval_check_prelude.h").display()
    );

    let tr = translate(&header, &sources, stages_from_env());
    if tr.stages.len() != tr.requested_stages {
        println!(
            "cargo:warning={STAGES_ENV}={} exceeds the chain's {} functions; using {}",
            tr.requested_stages,
            tr.chain_len,
            tr.stages.len()
        );
    }

    std::fs::create_dir_all(gen_dir).unwrap();
    // Drop stage files of an earlier build with more stages.
    let keep: BTreeSet<String> = tr
        .files
        .iter()
        .chain(&tr.strict_files)
        .map(|(n, _)| n.clone())
        .collect();
    for entry in std::fs::read_dir(gen_dir).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        let stage = name.starts_with("eval_stage_") || name.starts_with("eval_strict_stage_");
        if stage && name.ends_with(".metal") && !keep.contains(&name) {
            std::fs::remove_file(gen_dir.join(&name)).unwrap();
        }
    }
    let mut paths = Vec::new();
    for (name, text) in tr.files.iter().chain(&tr.strict_files) {
        let path = gen_dir.join(name);
        write_if_changed(&path, text);
        paths.push(path);
    }

    let out_dir = gen_dir.parent().expect("gen_dir is under OUT_DIR");
    let rs = out_dir.join("keccak_metal_eval.rs");
    write_if_changed(&rs, &tr.rust_consts());
    println!("cargo:rustc-env=KECCAK_METAL_EVAL_RS={}", rs.display());
    println!(
        "cargo:rustc-env=KECCAK_METAL_EVAL_STAGES={}",
        tr.stages.len()
    );
    paths
}

/// Under `RISC0_SKIP_BUILD_KERNELS` (no stage kernels in the empty metallib): writes a
/// `$OUT_DIR/keccak_metal_eval.rs` with no stages (sizes 1) and points
/// `KECCAK_METAL_EVAL_RS` at it, so `src/lib.rs` (`metal_eval`, `METAL_KERNEL_NAMES`) and the HAL
/// still compile. Same items as [Translation::rust_consts].
#[allow(dead_code)] // unused where tests include this file
pub(crate) fn write_skip_consts(out_dir: &Path) {
    let mut s = String::from(
        "// Written by keccak-sys/build_metal_eval.rs under RISC0_SKIP_BUILD_KERNELS: no stages.\n\
         pub const EVAL_STAGES: usize = 0;\n\
         pub const EVAL_STAGE_NAMES: [&str; 0] = [];\n\
         pub const EVAL_STRICT_STAGE_NAMES: [&str; 0] = [];\n\
         pub const EVAL_STAGE_BINDINGS: [u32; 0] = [];\n",
    );
    for name in [
        "SCRATCH_ELEMS",
        "MAX_STRIDE",
        "CARRY_WORDS",
        "MAX_COLUMN",
        "INV_RATE",
        "NUM_POLY_MIX_POWS",
    ] {
        // 1, not 0: the HAL's bounds checks stay meaningful expressions (clippy) and nothing
        // runs with these values (no stages).
        writeln!(s, "pub const EVAL_{name}: u32 = 1;").unwrap();
    }
    for (i, b) in BUFS.iter().enumerate() {
        writeln!(s, "pub const EVAL_BUF_{b}: u64 = {i};").unwrap();
    }
    writeln!(s, "pub const EVAL_PARAMS_SIZE: usize = 32;").unwrap();
    let rs = out_dir.join("keccak_metal_eval.rs");
    write_if_changed(&rs, &s);
    println!("cargo:rustc-env=KECCAK_METAL_EVAL_RS={}", rs.display());
    println!("cargo:rustc-env=KECCAK_METAL_EVAL_STAGES=0");
}

/// The stage count from [STAGES_ENV] (a positive integer), else [DEFAULT_STAGES].
pub(crate) fn stages_from_env() -> usize {
    println!("cargo:rerun-if-env-changed={STAGES_ENV}");
    match std::env::var(STAGES_ENV) {
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => panic!("{STAGES_ENV}={v:?}: expected a positive integer"),
        },
        Err(_) => DEFAULT_STAGES,
    }
}

fn write_if_changed(path: &Path, text: &str) {
    if std::fs::read_to_string(path).ok().as_deref() != Some(text) {
        std::fs::write(path, text).unwrap();
    }
}

/// `kernels/cuda/eval_check.cuh` and the `eval_check_<n>.cu` files in `n` order, as
/// `(path, contents)`.
pub(crate) fn read_cuda_sources(cuda_dir: &Path) -> (String, Vec<(String, String)>) {
    let header = std::fs::read_to_string(cuda_dir.join("eval_check.cuh"))
        .unwrap_or_else(|e| panic!("{}/eval_check.cuh: {e}", cuda_dir.display()));
    let mut files: Vec<(u32, PathBuf)> = std::fs::read_dir(cuda_dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.ok()?.path();
            let n = p
                .file_name()?
                .to_str()?
                .strip_prefix("eval_check_")?
                .strip_suffix(".cu")?
                .parse()
                .ok()?;
            Some((n, p))
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no kernels/cuda/eval_check_<n>.cu");
    let sources = files
        .into_iter()
        .map(|(_, p)| {
            let text = std::fs::read_to_string(&p).unwrap();
            (p.display().to_string(), text)
        })
        .collect();
    (header, sources)
}

/// A parameter of a generated function, after `idx` and `size`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// `Fp* argN`: always the scratch array.
    Scratch,
    /// `FpExt argN`.
    Ext,
    /// `const Fp* argN`: one of [INPUTS].
    Input,
}

#[derive(Clone, Debug)]
struct Param {
    kind: Kind,
    name: String,
}

#[derive(Clone, Debug)]
struct Call {
    /// Index of the line in `body`.
    line: usize,
    result: String,
    callee: String,
    args: Vec<String>,
}

#[derive(Clone, Debug)]
struct Function {
    name: String,
    params: Vec<Param>,
    body: Vec<String>,
    call: Option<Call>,
    /// The scratch array's length, if this function declares it.
    array: Option<(String, u32)>,
    /// Types of the parameters and locals (filled by `check_body`).
    locals: HashMap<String, Ty>,
}

/// One stage kernel.
#[derive(Clone, Debug)]
#[allow(dead_code)] // Read by the k3 tests (build.rs uses only the names).
pub(crate) struct Stage {
    /// `eval_stage_<k>`.
    pub name: String,
    /// The chain functions it runs, caller first.
    pub functions: Vec<String>,
    /// Lines of generated constraint code in those functions.
    pub lines: usize,
    /// Carry words it reads (its first function's `FpExt` parameters, 4 words each).
    pub carry_in_words: usize,
    /// The buffer bindings (`EVAL_BUF_*` indices) it may access, as a bit mask (K4): the HAL
    /// binds only these, so the Metal validation layer sees no unused bindings. A superset of
    /// what the compiled kernel reads (any mention of an input counts), never a subset.
    pub bindings: u32,
}

/// Binding indices (kernels/metal/eval_check_prelude.h), as [Translation::rust_consts] emits.
pub(crate) const BUFS: [&str; 10] = [
    "CHECK", "CTRL", "OUT", "DATA", "MIX", "ACCUM", "POLY_MIX", "SCRATCH", "CARRY", "PARAMS",
];

fn buf_bit(name: &str) -> u32 {
    let upper = name.to_ascii_uppercase();
    1 << BUFS
        .iter()
        .position(|b| *b == upper)
        .unwrap_or_else(|| panic!("eval_check: no binding {name}"))
}

/// Whether `line` mentions the identifier `ident` (not as part of a longer identifier).
fn mentions(line: &str, ident: &str) -> bool {
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    line.match_indices(ident).any(|(i, _)| {
        !line[..i].chars().next_back().is_some_and(is_ident)
            && !line[i + ident.len()..].chars().next().is_some_and(is_ident)
    })
}

/// What [translate] produces.
#[derive(Clone, Debug)]
#[allow(dead_code)] // `counts` is read by the k3 tests.
pub(crate) struct Translation {
    /// `(file name, MSL)`, one per stage, in stage order (kernels `eval_stage_<k>`, with the K7
    /// lazy multiply-accumulate).
    pub files: Vec<(String, String)>,
    /// The same stages without the lazy rewrite (K3/K4's code; kernels
    /// `eval_strict_stage_<k>`, files `eval_strict_stage_<k>.metal`): the runtime switch back
    /// (`RISC0_KECCAK_METAL_EVAL_MACC=strict`), built only when selected.
    pub strict_files: Vec<(String, String)>,
    pub stages: Vec<Stage>,
    pub requested_stages: usize,
    /// Functions in the chain (58 today).
    pub chain_len: usize,
    /// Length of the scratch array (`Fp x13[8237]` today): scratch words per thread.
    pub scratch_elems: u32,
    /// Carry words per thread: the largest `carry_in_words`, at least 1.
    pub carry_words: usize,
    /// The header's `constexpr` constants (`INV_RATE`, `kNumPolyMixPows`).
    pub constants: Vec<(String, u64)>,
    /// The largest column `c` of a `buf[c * size + ...]` read: the HAL must check
    /// `(max_column + 1) * domain` fits in u32 (the kernels index in 32 bits, as CUDA does).
    pub max_column: u32,
    /// Pinned rewrite counts (see the k3 tests).
    pub counts: RewriteCounts,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RewriteCounts {
    /// Function definitions (each gains `EvalCtx ectx`).
    pub functions: usize,
    /// `Fp* argN` parameters made `ScratchRef`.
    pub scratch_params: usize,
    /// `const Fp* argN` parameters made `const device Fp*`.
    pub input_params: usize,
    /// `poly_mix[` reads made `ectx.poly_mix[`.
    pub poly_mix_reads: usize,
    /// Calls that pass `ectx` (calls within a stage).
    pub calls: usize,
    /// Calls cut into carry stores (= stages - 1).
    pub cuts: usize,
    /// `constexpr size_t` made `constant constexpr uint32_t`.
    pub constants: usize,
    /// Lines of constraint code (function bodies).
    pub body_lines: usize,
    /// K7 lazy multiply-accumulate (see [lazy_plan]): `FpExt xB = xA + poly_mix[N] * xC;` lines
    /// rewritten into 64-bit per-limb accumulators,
    pub lazy_macs: usize,
    /// the chains they form (each ends in one Montgomery reduction per limb),
    pub lazy_chains: usize,
    /// and the folds that keep the accumulators below 2^64.
    pub lazy_folds: usize,
}

impl Translation {
    /// Rust constants for the HAL (`include!(env!("KECCAK_METAL_EVAL_RS"))`): the kernels'
    /// names and the sizes of the buffers they need. Binding indices and `EvalParams` are
    /// documented in `kernels/metal/eval_check_prelude.h`.
    pub(crate) fn rust_consts(&self) -> String {
        let mut s = String::from(
            "// Generated by keccak-sys/build_metal_eval.rs. Do not edit.\n\
             // Must match kernels/metal/eval_check_prelude.h.\n",
        );
        let names: Vec<String> = self
            .stages
            .iter()
            .map(|st| format!("{:?}", st.name))
            .collect();
        writeln!(
            s,
            "/// Number of eval_check stage kernels (run in order, all on the same chunk)."
        )
        .unwrap();
        writeln!(s, "pub const EVAL_STAGES: usize = {};", self.stages.len()).unwrap();
        writeln!(s, "/// The stage kernels' entry points, in launch order.").unwrap();
        writeln!(
            s,
            "pub const EVAL_STAGE_NAMES: [&str; {}] = [{}];",
            self.stages.len(),
            names.join(", ")
        )
        .unwrap();
        let strict: Vec<String> = (0..self.stages.len())
            .map(|k| format!("\"eval_strict_stage_{k}\""))
            .collect();
        writeln!(
            s,
            "/// The same stages without the K7 lazy multiply-accumulate (runtime switch\n\
             /// `RISC0_KECCAK_METAL_EVAL_MACC=strict`; not in METAL_KERNEL_NAMES: built on demand)."
        )
        .unwrap();
        writeln!(
            s,
            "pub const EVAL_STRICT_STAGE_NAMES: [&str; {}] = [{}];",
            self.stages.len(),
            strict.join(", ")
        )
        .unwrap();
        writeln!(
            s,
            "/// Scratch words per thread (scratch buffer: this many * `stride` Fp)."
        )
        .unwrap();
        writeln!(
            s,
            "pub const EVAL_SCRATCH_ELEMS: u32 = {};",
            self.scratch_elems
        )
        .unwrap();
        writeln!(
            s,
            "/// Largest `stride` (chunk capacity) the kernels' 32-bit scratch indexing allows."
        )
        .unwrap();
        writeln!(
            s,
            "pub const EVAL_MAX_STRIDE: u32 = {};",
            (1u64 << 32) / self.scratch_elems.max(self.carry_words as u32) as u64
        )
        .unwrap();
        writeln!(
            s,
            "/// Carry words per thread (carry buffer: this many * `stride` u32)."
        )
        .unwrap();
        writeln!(s, "pub const EVAL_CARRY_WORDS: u32 = {};", self.carry_words).unwrap();
        writeln!(
            s,
            "/// Largest column read as `buf[c * domain + ...]`: require"
        )
        .unwrap();
        writeln!(
            s,
            "/// `(EVAL_MAX_COLUMN + 1) * domain <= 2^32` (32-bit indexing, as CUDA)."
        )
        .unwrap();
        writeln!(s, "pub const EVAL_MAX_COLUMN: u32 = {};", self.max_column).unwrap();
        for (name, value) in &self.constants {
            writeln!(s, "pub const EVAL_{}: u32 = {value};", upper_snake(name)).unwrap();
        }
        writeln!(s, "/// Buffer bindings shared by every stage kernel.").unwrap();
        for (i, b) in BUFS.iter().enumerate() {
            writeln!(s, "pub const EVAL_BUF_{b}: u64 = {i};").unwrap();
        }
        let masks: Vec<String> = self
            .stages
            .iter()
            .map(|st| format!("{:#05x}", st.bindings))
            .collect();
        writeln!(
            s,
            "/// Per stage, a bit mask of the bindings (`1 << EVAL_BUF_*`) it may access; the\n\
             /// launcher binds only these (a superset of what the kernel reads)."
        )
        .unwrap();
        writeln!(
            s,
            "pub const EVAL_STAGE_BINDINGS: [u32; {}] = [{}];",
            self.stages.len(),
            masks.join(", ")
        )
        .unwrap();
        writeln!(s, "/// `sizeof(EvalParams)`.").unwrap();
        writeln!(s, "pub const EVAL_PARAMS_SIZE: usize = 32;").unwrap();
        s
    }
}

fn upper_snake(name: &str) -> String {
    let name = name
        .strip_prefix('k')
        .filter(|r| r.starts_with(char::is_uppercase))
        .unwrap_or(name);
    let mut out = String::new();
    for (i, c) in name.char_indices() {
        if c.is_uppercase() && i > 0 && !name[..i].ends_with(|p: char| p.is_uppercase() || p == '_')
        {
            out.push('_');
        }
        out.push(c.to_ascii_uppercase());
    }
    out
}

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        && !s.as_bytes()[0].is_ascii_digit()
}

fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// Every maximal run of digits replaced by `N`.
fn normalize(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_digits = false;
    for c in line.chars() {
        if c.is_ascii_digit() {
            if !in_digits {
                out.push('N');
            }
            in_digits = true;
        } else {
            out.push(c);
            in_digits = false;
        }
    }
    out
}

/// Lines the generator puts outside functions and declarations.
fn is_boilerplate(trimmed: &str) -> bool {
    trimmed.is_empty()
        || trimmed == "// This code is automatically generated"
        || trimmed == "#pragma once"
        || trimmed == "#include \"supra/fp.h\""
        || trimmed == "#include \"eval_check.cuh\""
        || trimmed == "#include <cstdint>"
        || trimmed == "namespace risc0::circuit::keccak::cuda {"
        || trimmed == "} // namespace risc0::circuit::keccak::cuda"
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Split `name(a, b, c)` (no trailing `;`) into the name and the comma-separated parts.
fn split_call(s: &str) -> Option<(&str, Vec<&str>)> {
    let (name, rest) = s.split_once('(')?;
    let inner = rest.strip_suffix(')')?;
    Some((name.trim(), inner.split(',').map(str::trim).collect()))
}

fn parse_params(name: &str, list: &[&str], what: &str) -> Vec<Param> {
    assert!(
        list.len() >= 2 && list[0] == "uint32_t idx" && list[1] == "uint32_t size",
        "eval_check: {what} {name} doesn't start with (uint32_t idx, uint32_t size, ...): {list:?}"
    );
    list[2..]
        .iter()
        .map(|p| {
            // `const Fp*` first: it also ends in `Fp* name`.
            let (kind, pname) = if let Some(n) = p.strip_prefix("const Fp* ") {
                (Kind::Input, n)
            } else if let Some(n) = p.strip_prefix("Fp* ") {
                (Kind::Scratch, n)
            } else if let Some(n) = p.strip_prefix("FpExt ") {
                (Kind::Ext, n)
            } else {
                panic!("eval_check: {what} {name}: unexpected parameter `{p}`")
            };
            assert!(
                is_ident(pname),
                "eval_check: {what} {name}: parameter `{p}`"
            );
            Param {
                kind,
                name: pname.to_string(),
            }
        })
        .collect()
}

struct Header {
    /// Declared functions and their parameters.
    decls: BTreeMap<String, Vec<(Kind, String)>>,
    constants: Vec<(String, u64)>,
}

fn parse_header(header: &str) -> Header {
    let mut decls = BTreeMap::new();
    let mut constants = Vec::new();
    let mut poly_mix = false;
    let mut stmt = String::new();
    for line in header.lines() {
        let trimmed = line.trim();
        if stmt.is_empty() && is_boilerplate(trimmed) {
            continue;
        }
        assert!(
            !trimmed.contains("Fp *"),
            "eval_check.cuh: unexpected `Fp *` spelling: {line}"
        );
        stmt.push(' ');
        stmt.push_str(trimmed);
        if !trimmed.ends_with(';') {
            continue;
        }
        let s = collapse_ws(&stmt);
        stmt.clear();
        let s = s.strip_suffix(';').unwrap();
        if let Some(decl) = s.strip_prefix("extern __device__ FpExt ") {
            let (name, list) = split_call(decl)
                .unwrap_or_else(|| panic!("eval_check.cuh: unexpected declaration `{s}`"));
            assert!(is_ident(name), "eval_check.cuh: declaration `{s}`");
            let params = parse_params(name, &list, "declaration of")
                .into_iter()
                .map(|p| (p.kind, p.name))
                .collect();
            assert!(
                decls.insert(name.to_string(), params).is_none(),
                "eval_check.cuh: {name} declared twice"
            );
        } else if let Some(c) = s.strip_prefix("constexpr size_t ") {
            let (name, value) = c
                .split_once(" = ")
                .unwrap_or_else(|| panic!("eval_check.cuh: unexpected constant `{s}`"));
            assert!(
                is_ident(name) && is_digits(value),
                "eval_check.cuh: constant `{s}`"
            );
            let value: u64 = value.parse().unwrap();
            assert!(
                value <= u32::MAX as u64,
                "eval_check.cuh: {name} doesn't fit u32"
            );
            constants.push((name.to_string(), value));
        } else if s == "extern __constant__ FpExt poly_mix[kNumPolyMixPows]" {
            assert!(!poly_mix, "eval_check.cuh: poly_mix declared twice");
            poly_mix = true;
        } else {
            panic!("eval_check.cuh: unexpected statement `{s};`");
        }
    }
    assert!(
        stmt.trim().is_empty(),
        "eval_check.cuh: unterminated `{stmt}`"
    );
    assert!(poly_mix, "eval_check.cuh: no poly_mix declaration");
    let names: Vec<&str> = constants.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        ["INV_RATE", "kNumPolyMixPows"],
        "eval_check.cuh: unexpected constants"
    );
    Header { decls, constants }
}

/// Parse the definitions in one eval_check_<n>.cu.
fn parse_definitions(file: &str, text: &str, out: &mut Vec<Function>) {
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let trimmed = lines[i].trim();
        assert!(
            !lines[i].contains("Fp *"),
            "{file}:{}: unexpected `Fp *` spelling: {}",
            i + 1,
            lines[i]
        );
        if is_boilerplate(trimmed) {
            i += 1;
            continue;
        }
        assert!(
            lines[i].starts_with("__device__ FpExt"),
            "{file}:{}: unexpected line outside a function: {}",
            i + 1,
            lines[i]
        );
        // The signature: up to the line ending in `) {`.
        let mut sig = String::new();
        loop {
            assert!(i < lines.len(), "{file}: unterminated signature");
            sig.push(' ');
            sig.push_str(lines[i].trim());
            i += 1;
            if lines[i - 1].trim_end().ends_with(") {") {
                break;
            }
        }
        let sig = collapse_ws(&sig);
        let decl = sig
            .strip_prefix("__device__ FpExt ")
            .and_then(|s| s.strip_suffix(" {"))
            .unwrap_or_else(|| panic!("{file}: unexpected signature `{sig}`"));
        let (name, list) =
            split_call(decl).unwrap_or_else(|| panic!("{file}: unexpected signature `{sig}`"));
        assert!(is_ident(name), "{file}: signature `{sig}`");
        let params = parse_params(name, &list, "definition of");
        let mut body = Vec::new();
        while i < lines.len() && lines[i] != "}" {
            assert!(
                !lines[i].contains("Fp *"),
                "{file}:{}: unexpected `Fp *` spelling: {}",
                i + 1,
                lines[i]
            );
            // Blank lines (the generator puts one before the tail call's `return`) are dropped.
            if !lines[i].trim().is_empty() {
                body.push(lines[i].trim().to_string());
            }
            i += 1;
        }
        assert!(i < lines.len(), "{file}: {name} has no closing brace");
        i += 1;
        out.push(Function {
            name: name.to_string(),
            params,
            body,
            call: None,
            array: None,
            locals: HashMap::new(),
        });
    }
}

/// Local variable types, for checking call arguments.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    Fp,
    Ext,
    Scratch,
    Input,
}

/// Check every body line has a known shape; find the call, the scratch array and the largest
/// column index. Returns the number of `poly_mix[` reads.
fn check_body(f: &mut Function, max_column: &mut u32) -> usize {
    let mut tys: HashMap<String, Ty> = HashMap::new();
    for p in &f.params {
        let ty = match p.kind {
            Kind::Scratch => Ty::Scratch,
            Kind::Ext => Ty::Ext,
            Kind::Input => Ty::Input,
        };
        tys.insert(p.name.clone(), ty);
    }
    let name = f.name.clone();
    let ty_of = |tys: &HashMap<String, Ty>, v: &str, line: &str| -> Ty {
        *tys.get(v)
            .unwrap_or_else(|| panic!("eval_check: {name}: `{v}` undeclared in `{line}`"))
    };
    let declare = |tys: &mut HashMap<String, Ty>, v: &str, ty: Ty, line: &str| {
        assert!(is_ident(v), "eval_check: {name}: `{line}`");
        assert!(
            tys.insert(v.to_string(), ty).is_none(),
            "eval_check: {name}: `{v}` declared twice"
        );
    };
    let mut poly_mix = 0;
    let n = f.body.len();
    assert!(n >= 2, "eval_check: {name}: empty body");
    assert_eq!(
        f.body[0], "uint32_t mask = size - 1;",
        "eval_check: {name}: first line"
    );
    for (li, line) in f.body.iter().enumerate().skip(1) {
        let line = line.as_str();
        // Tail: `return xN;` is the last line, and only there.
        if let Some(v) = line
            .strip_prefix("return ")
            .and_then(|r| r.strip_suffix(';'))
        {
            assert_eq!(li, n - 1, "eval_check: {name}: `{line}` before the end");
            assert_eq!(
                ty_of(&tys, v, line),
                Ty::Ext,
                "eval_check: {name}: returns `{v}`"
            );
            continue;
        }
        assert!(
            li < n - 1,
            "eval_check: {name}: doesn't end in `return xN;`"
        );
        // `auto xR = callee(idx, size, args...);`
        if let Some(rest) = line.strip_prefix("auto ") {
            let (result, call) = rest
                .split_once(" = ")
                .unwrap_or_else(|| panic!("eval_check: {name}: unexpected line `{line}`"));
            let (callee, args) = call
                .strip_suffix(';')
                .and_then(split_call)
                .unwrap_or_else(|| panic!("eval_check: {name}: unexpected line `{line}`"));
            assert!(is_ident(callee), "eval_check: {name}: `{line}`");
            assert!(
                args.len() >= 2 && args[0] == "idx" && args[1] == "size",
                "eval_check: {name}: call without (idx, size, ...): `{line}`"
            );
            assert!(f.call.is_none(), "eval_check: {name}: more than one call");
            // A tail call: the next and last line returns its result.
            assert_eq!(
                f.body[n - 1],
                format!("return {result};"),
                "eval_check: {name}: the call to {callee} isn't a tail call"
            );
            assert_eq!(
                li,
                n - 2,
                "eval_check: {name}: the call to {callee} isn't a tail call"
            );
            let args: Vec<String> = args[2..].iter().map(|a| a.to_string()).collect();
            for a in &args {
                ty_of(&tys, a, line);
            }
            declare(&mut tys, result, Ty::Ext, line);
            f.call = Some(Call {
                line: li,
                result: result.to_string(),
                callee: callee.to_string(),
                args,
            });
            continue;
        }
        // `Fp xN[M];`: the scratch array.
        if let Some((var, len)) = line
            .strip_prefix("Fp ")
            .and_then(|d| d.strip_suffix("];"))
            .and_then(|d| d.split_once('['))
            .filter(|(v, l)| v.starts_with('x') && is_digits(&v[1..]) && is_digits(l))
        {
            assert!(f.array.is_none(), "eval_check: {name}: two arrays");
            f.array = Some((var.to_string(), len.parse().expect("array length")));
            declare(&mut tys, var, Ty::Scratch, line);
            continue;
        }

        // Straight-line arithmetic, reads and writes: match the shape, with each operand replaced
        // by its type and digits by `N`. (The variable being declared isn't typed yet: it stays
        // `xN`.)
        let role_shape = {
            let mut s = String::new();
            let mut word = String::new();
            let flush = |word: &mut String, s: &mut String| {
                if word.is_empty() {
                    return;
                }
                let w = std::mem::take(word);
                let role = if is_digits(&w) {
                    "N".to_string()
                } else {
                    match tys.get(&w) {
                        Some(Ty::Fp) => "FP".to_string(),
                        Some(Ty::Ext) => "EXT".to_string(),
                        Some(Ty::Scratch) => "SCRATCH".to_string(),
                        Some(Ty::Input) => "INPUT".to_string(),
                        None => w,
                    }
                };
                s.push_str(&role);
            };
            for c in line.chars() {
                if c.is_ascii_alphanumeric() || c == '_' {
                    word.push(c);
                } else {
                    flush(&mut word, &mut s);
                    s.push(c);
                }
            }
            flush(&mut word, &mut s);
            normalize(&s)
        };
        let declared = |prefix: &str| -> &str {
            line.strip_prefix(prefix)
                .and_then(|r| r.split([' ', '(']).next())
                .unwrap()
        };
        match role_shape.as_str() {
            "Fp xN = FP * FP;"
            | "Fp xN = FP + FP;"
            | "Fp xN = FP - FP;"
            | "Fp xN(N);"
            | "Fp xN = SCRATCH[N];"
            | "Fp xN = INPUT[N];" => {
                declare(&mut tys, declared("Fp "), Ty::Fp, line);
            }
            "Fp xN = INPUT[N * size + ((idx - INV_RATE * N) & mask)];" => {
                let col: u32 = line
                    .split_once('[')
                    .and_then(|(_, r)| r.split_once(" * size"))
                    .and_then(|(c, _)| c.parse().ok())
                    .expect("column");
                *max_column = (*max_column).max(col);
                declare(&mut tys, declared("Fp "), Ty::Fp, line);
            }
            "FpExt xN = FpExt(N);" => {
                declare(&mut tys, declared("FpExt "), Ty::Ext, line);
            }
            "FpExt xN = EXT + poly_mix[N] * FP;"
            | "FpExt xN = EXT + FP * FP * poly_mix[N];"
            | "FpExt xN = EXT + FP * EXT * poly_mix[N];" => {
                poly_mix += 1;
                declare(&mut tys, declared("FpExt "), Ty::Ext, line);
            }
            "SCRATCH[N] = FP;" => {}
            _ => panic!("eval_check: {name}: unexpected line `{line}` (shape `{role_shape}`)"),
        }
    }
    f.locals = tys;
    poly_mix
}

/// Emit `f` as MSL; `cut` replaces its tail call by carry stores (the callee starts the next
/// stage).
fn emit_function(
    f: &Function,
    cut: Option<&Function>,
    lazy: bool,
    counts: &mut RewriteCounts,
) -> String {
    let mut s = String::new();
    let plan = lazy.then(|| lazy_plan(f));
    // Upper bound (exclusive) of each lazy accumulator's 64-bit value.
    let mut bound: HashMap<String, u128> = HashMap::new();
    let params: Vec<String> = f
        .params
        .iter()
        .map(|p| match p.kind {
            Kind::Scratch => format!("ScratchRef {}", p.name),
            Kind::Ext => format!("FpExt {}", p.name),
            Kind::Input => format!("const device Fp* {}", p.name),
        })
        .collect();
    writeln!(
        s,
        "R0_KECCAK_EVAL_FN FpExt {}(uint32_t idx, uint32_t size, EvalCtx ectx, {}) {{",
        f.name,
        params.join(", ")
    )
    .unwrap();
    for (li, line) in f.body.iter().enumerate() {
        if let Some(call) = f.call.as_ref().filter(|c| c.line == li) {
            match cut {
                Some(head) => {
                    writeln!(
                        s,
                        "  // {} starts the next stage: carry its FpExt arguments.",
                        head.name
                    )
                    .unwrap();
                    let mut slot = 0;
                    for (p, a) in head.params.iter().zip(&call.args) {
                        if p.kind == Kind::Ext {
                            writeln!(s, "  carry_store(ectx, {slot}, {a});").unwrap();
                            slot += 1;
                        }
                    }
                    s.push_str("  return FpExt();\n");
                    counts.cuts += 1;
                }
                None => {
                    writeln!(
                        s,
                        "  auto {} = {}(idx, size, ectx, {});",
                        call.result,
                        call.callee,
                        call.args.join(", ")
                    )
                    .unwrap();
                    counts.calls += 1;
                }
            }
            continue;
        }
        if cut.is_some() && li == f.body.len() - 1 {
            // `return xR;` of the cut call: already returned.
            continue;
        }
        if let Some((var, _)) = f
            .array
            .as_ref()
            .filter(|(v, len)| *line == format!("Fp {v}[{len}];"))
        {
            writeln!(s, "  ScratchRef {var} = ectx.scratch;").unwrap();
            continue;
        }
        if let Some((b, a, n, c)) = plan.as_ref().and_then(|_| parse_mac(line)) {
            let lazy_vars = plan.as_ref().unwrap();
            counts.lazy_macs += 1;
            writeln!(s, "  FpExt pm_{b} = ectx.poly_mix[{n}];").unwrap();
            let (prev, mut bd) = match bound.get(a) {
                Some(&bd) if lazy_vars.contains(a) => (a.to_string(), bd),
                _ => {
                    counts.lazy_chains += 1;
                    let start = format!("s_{b}");
                    for j in 0..4 {
                        writeln!(
                            s,
                            "  uint64_t {start}_l{j} = keccak_eval::lazy_start({a}.elems[{j}]);"
                        )
                        .unwrap();
                    }
                    (start, LAZY_START_BOUND)
                }
            };
            let fold = bd + LAZY_PROD_BOUND > 1u128 << 64;
            if fold {
                counts.lazy_folds += 1;
                bd = LAZY_FOLDED_BOUND;
            }
            for j in 0..4 {
                let acc = if fold {
                    format!("keccak_eval::lazy_fold({prev}_l{j})")
                } else {
                    format!("{prev}_l{j}")
                };
                writeln!(
                    s,
                    "  uint64_t {b}_l{j} = {acc} + keccak_eval::lazy_prod(pm_{b}.elems[{j}], {c});"
                )
                .unwrap();
            }
            bound.insert(b.to_string(), bd + LAZY_PROD_BOUND);
            if !lazy_vars.contains(b) {
                writeln!(
                    s,
                    "  FpExt {b} = keccak_eval::lazy_ext({b}_l0, {b}_l1, {b}_l2, {b}_l3);"
                )
                .unwrap();
            }
            continue;
        }
        let out = line.replace("poly_mix[", "ectx.poly_mix[");
        s.push_str("  ");
        s.push_str(&out);
        s.push('\n');
    }
    s.push_str("}\n");
    s
}

/// BabyBear's modulus.
const LAZY_P: u128 = 15 * (1 << 27) + 1;
/// Exclusive bound of one product of two raw (Montgomery, < P) words.
const LAZY_PROD_BOUND: u128 = (LAZY_P - 1) * (LAZY_P - 1) + 1;
/// Exclusive bound of a chain's start, `raw << 32` (raw < P).
const LAZY_START_BOUND: u128 = (LAZY_P - 1) * (1 << 32) + 1;
/// Exclusive bound after `lazy_fold` (the high word reduced mod P).
const LAZY_FOLDED_BOUND: u128 = LAZY_P * (1 << 32);

/// `FpExt xB = xA + poly_mix[N] * xC;` (the multiply-accumulate shape) as `(xB, xA, N, xC)`.
fn parse_mac(line: &str) -> Option<(&str, &str, &str, &str)> {
    let rest = line.strip_prefix("FpExt ")?;
    let (b, rhs) = rest.split_once(" = ")?;
    let (a, rhs) = rhs.split_once(" + poly_mix[")?;
    let (n, c) = rhs.split_once("] * ")?;
    let c = c.strip_suffix(';')?;
    (is_ident(b) && is_ident(a) && is_digits(n) && is_ident(c)).then_some((b, a, n, c))
}

/// K7 lazy multiply-accumulate. `FpExt xB = xA + poly_mix[N] * xC;` is 4 Montgomery products
/// and 4 modular additions. Chains of them (xA used only as the accumulator of the next one)
/// are emitted as 4 `uint64_t` accumulators per step instead: the chain starts from
/// `raw(xA) << 32` (= xA·R² in Montgomery terms, R = 2^32), each step adds the 64-bit product
/// `raw(poly_mix[N].elems[j]) * raw(xC)` (= p·c·R²), the high word is reduced mod P whenever the
/// next product could overflow 2^64 (`lazy_fold`, no multiply), and the chain's last value is
/// Montgomery-reduced once per limb (`lazy_ext`). The result is the same field element, in
/// canonical form, so the output is bit-identical; it saves about half of the multiplies of
/// these lines (13.4k of the ~70k Montgomery products a cycle; eval_check is ALU-bound, K7 note).
/// Returns the chain members whose value never has to exist as an `FpExt`: defined by a
/// multiply-accumulate and used exactly once, as the accumulator of another one.
fn lazy_plan(f: &Function) -> BTreeSet<String> {
    let mut uses: HashMap<&str, usize> = HashMap::new();
    let mut acc_uses: HashMap<&str, usize> = HashMap::new();
    let mut defined = BTreeSet::new();
    for line in &f.body {
        let mut words = line
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .filter(|w| is_ident(w) && (w.starts_with('x') || w.starts_with("arg")))
            .peekable();
        // The declared variable of `Fp xN ...`, `FpExt xN ...`, `auto xN = ...` is no use.
        if ["Fp ", "FpExt ", "auto "]
            .iter()
            .any(|p| line.starts_with(p))
        {
            words.next();
        }
        for w in words {
            *uses.entry(w).or_default() += 1;
        }
        if let Some((b, a, _, _)) = parse_mac(line) {
            *acc_uses.entry(a).or_default() += 1;
            defined.insert(b);
        }
    }
    acc_uses
        .into_iter()
        .filter(|(v, n)| *n == 1 && uses.get(v) == Some(&1) && defined.contains(v))
        .map(|(v, _)| v.to_string())
        .collect()
}

/// Translate the CUDA eval_check (`header` = eval_check.cuh, `sources` = the eval_check_<n>.cu
/// files as `(name, text)` in order) into `stages` Metal stage kernels (fewer if the chain is
/// shorter). Panics if the input doesn't have the expected shape.
pub(crate) fn translate(header: &str, sources: &[(String, String)], stages: usize) -> Translation {
    assert!(stages > 0, "eval_check: stage count must be positive");
    let header = parse_header(header);
    let mut functions = Vec::new();
    for (file, text) in sources {
        parse_definitions(file, text, &mut functions);
    }

    let mut counts = RewriteCounts {
        constants: header.constants.len(),
        ..Default::default()
    };
    let mut max_column = 0;
    for f in functions.iter_mut() {
        counts.poly_mix_reads += check_body(f, &mut max_column);
        counts.body_lines += f.body.len();
        counts.functions += 1;
        counts.scratch_params += f.params.iter().filter(|p| p.kind == Kind::Scratch).count();
        counts.input_params += f.params.iter().filter(|p| p.kind == Kind::Input).count();
    }

    // Definitions match declarations one to one.
    let defined: BTreeSet<&str> = functions.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        defined.len(),
        functions.len(),
        "eval_check: a function is defined twice"
    );
    let declared: BTreeSet<&str> = header.decls.keys().map(String::as_str).collect();
    assert_eq!(
        defined, declared,
        "eval_check: definitions don't match eval_check.cuh"
    );
    for f in &functions {
        let params: Vec<(Kind, String)> =
            f.params.iter().map(|p| (p.kind, p.name.clone())).collect();
        assert_eq!(
            &params, &header.decls[&f.name],
            "eval_check: {}'s definition doesn't match its declaration",
            f.name
        );
    }

    // The chain from poly_fp.
    let index: HashMap<&str, usize> = functions
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name.as_str(), i))
        .collect();
    let root = *index.get("poly_fp").expect("eval_check: no poly_fp");
    let poly_fp = &functions[root];
    let root_params: Vec<(Kind, &str)> = poly_fp
        .params
        .iter()
        .map(|p| (p.kind, p.name.as_str()))
        .collect();
    assert_eq!(
        root_params,
        INPUTS.iter().map(|n| (Kind::Input, *n)).collect::<Vec<_>>(),
        "eval_check: poly_fp's parameters changed"
    );
    let mut chain = vec![root];
    while let Some(call) = &functions[*chain.last().unwrap()].call {
        let next = *index
            .get(call.callee.as_str())
            .unwrap_or_else(|| panic!("eval_check: call to undefined {}", call.callee));
        assert!(
            !chain.contains(&next),
            "eval_check: the chain loops at {}",
            call.callee
        );
        chain.push(next);
    }
    assert_eq!(
        chain.len(),
        functions.len(),
        "eval_check: functions outside the chain from poly_fp"
    );

    // The scratch array: declared once, by poly_fp.
    let arrays: Vec<&Function> = functions.iter().filter(|f| f.array.is_some()).collect();
    assert_eq!(arrays.len(), 1, "eval_check: expected exactly one Fp array");
    assert_eq!(
        arrays[0].name, "poly_fp",
        "eval_check: the Fp array isn't poly_fp's"
    );
    let scratch_elems = arrays[0].array.as_ref().unwrap().1;

    // Check call arguments against the callee's parameters and trace every `const Fp*`
    // parameter to the poly_fp input it is.
    let mut roots: HashMap<(usize, String), String> = HashMap::new();
    for n in INPUTS {
        roots.insert((root, n.to_string()), n.to_string());
    }
    for w in chain.windows(2) {
        let (caller, callee) = (&functions[w[0]], &functions[w[1]]);
        let call = caller.call.as_ref().unwrap();
        assert_eq!(
            call.args.len(),
            callee.params.len(),
            "eval_check: {} passes {} arguments to {}",
            caller.name,
            call.args.len(),
            callee.name
        );
        for (p, a) in callee.params.iter().zip(&call.args) {
            let want = match p.kind {
                Kind::Scratch => Ty::Scratch,
                Kind::Ext => Ty::Ext,
                Kind::Input => Ty::Input,
            };
            assert_eq!(
                caller.locals.get(a),
                Some(&want),
                "eval_check: {} passes `{a}` as {}'s `{}`",
                caller.name,
                callee.name,
                p.name
            );
            if p.kind == Kind::Input {
                let r = roots[&(w[0], a.clone())].clone();
                roots.insert((w[1], p.name.clone()), r);
            }
        }
    }

    // Cut the chain into stages of about equal line counts, exactly min(stages, chain) of them:
    // cut after a function once the running total reaches the next 1/S, or when each remaining
    // function must be its own stage.
    let lens: Vec<usize> = chain.iter().map(|&i| functions[i].body.len()).collect();
    let total: usize = lens.iter().sum();
    let s = stages.min(chain.len());
    let mut cuts: Vec<usize> = Vec::new(); // cut after chain[i]
    let mut done = 0;
    for (i, len) in lens.iter().enumerate().take(chain.len() - 1) {
        done += len;
        let need = s - 1 - cuts.len();
        if need == 0 {
            break;
        }
        let remaining = chain.len() - 1 - i;
        if done * s >= total * (cuts.len() + 1) || remaining == need {
            cuts.push(i);
        }
    }
    assert_eq!(cuts.len(), s - 1, "eval_check: stage cutting");
    let mut bounds = vec![0];
    bounds.extend(cuts.iter().map(|c| c + 1));
    bounds.push(chain.len());

    let mut files = Vec::new();
    let mut strict_files = Vec::new();
    let mut out_stages = Vec::new();
    let mut carry_words = 1;
    for k in 0..s {
        let members = &chain[bounds[k]..bounds[k + 1]];
        let head = &functions[members[0]];
        let last = k + 1 == s;
        let ext_in = if k == 0 {
            0
        } else {
            head.params.iter().filter(|p| p.kind == Kind::Ext).count()
        };
        carry_words = carry_words.max(4 * ext_in);
        let name = format!("eval_stage_{k}");
        let lines: usize = members.iter().map(|&i| functions[i].body.len()).sum();
        let fn_names: Vec<String> = members.iter().map(|&i| functions[i].name.clone()).collect();

        // The stage's file, lazy (default kernels) or strict (the K3/K4 code, runtime switch).
        let stage_text = |lazy: bool, kname: &str, counts: &mut RewriteCounts| -> String {
            let mut body = String::new();
            // Callees first, so no forward declarations are needed.
            for (j, &fi) in members.iter().enumerate().rev() {
                let f = &functions[fi];
                let cut =
                    (j + 1 == members.len() && !last).then(|| &functions[chain[bounds[k + 1]]]);
                body.push('\n');
                body.push_str(&emit_function(f, cut, lazy, counts));
            }
            // The header's constants this stage uses (an unused one would warn).
            let consts: String = header
                .constants
                .iter()
                .filter(|(n, _)| body.contains(n.as_str()))
                .map(|(n, v)| format!("constant constexpr uint32_t {n} = {v};\n"))
                .collect();

            let mut text = String::new();
            writeln!(
                text,
                "// Generated by keccak-sys/build_metal_eval.rs from kernels/cuda/eval_check.cuh and\n\
                 // eval_check_<n>.cu. Do not edit.\n\
                 // Stage {k} of {s}{}: {} ({lines} lines of constraint code).\n\
                 #include \"eval_check_prelude.h\"\n\n\
                 namespace keccak_eval {{\n\n{consts}",
                if lazy { "" } else { " (strict: no lazy multiply-accumulate)" },
                fn_names.join(" -> ")
            )
            .unwrap();
            text.push_str(&body);
            text.push_str("\n} // namespace keccak_eval\n\n");

            // The kernel.
            let args: Vec<String> = if k == 0 {
                INPUTS.iter().map(|s| s.to_string()).collect()
            } else {
                let mut slot = 0;
                head.params
                    .iter()
                    .map(|p| match p.kind {
                        Kind::Scratch => "ectx.scratch".to_string(),
                        Kind::Input => roots[&(members[0], p.name.clone())].clone(),
                        Kind::Ext => {
                            let a = format!("keccak_eval::carry_load(ectx, {slot})");
                            slot += 1;
                            a
                        }
                    })
                    .collect()
            };
            writeln!(
                text,
                "R0_KECCAK_EVAL_KERNEL_ATTRS\n\
                 kernel void {kname}(R0_KECCAK_EVAL_KERNEL_PARAMS) {{\n\
                 \x20 R0_KECCAK_EVAL_KERNEL_PROLOGUE\n\
                 \x20 FpExt tot = keccak_eval::{}(cycle, p.domain, ectx, {});",
                head.name,
                args.join(", ")
            )
            .unwrap();
            if last {
                text.push_str("  keccak_eval::write_check(check, p, cycle, tot);\n");
            } else {
                text.push_str("  (void)tot;\n");
            }
            text.push_str("}\n");
            text
        };
        // The strict copy's rewrites are not counted again.
        let strict_name = format!("eval_strict_stage_{k}");
        let strict = stage_text(false, &strict_name, &mut RewriteCounts::default());
        strict_files.push((format!("{strict_name}.metal"), strict));
        let text = stage_text(true, &name, &mut counts);
        files.push((format!("{name}.metal"), text));

        // The bindings this stage may touch. Inputs: a member's `const Fp*` parameter mentioned
        // on a line other than its tail call (a call within the stage hands it to the callee,
        // which is a member too; a cut call does not pass inputs: the next kernel binds them).
        let mut bindings = buf_bit("params") | buf_bit("scratch");
        if last {
            bindings |= buf_bit("check");
        }
        if !last || ext_in > 0 {
            bindings |= buf_bit("carry");
        }
        for &fi in members {
            let f = &functions[fi];
            let call_line = f.call.as_ref().map(|c| c.line);
            let read = |ident: &str| {
                f.body
                    .iter()
                    .enumerate()
                    .any(|(li, line)| Some(li) != call_line && mentions(line, ident))
            };
            for p in f.params.iter().filter(|p| p.kind == Kind::Input) {
                if read(&p.name) {
                    bindings |= buf_bit(&roots[&(fi, p.name.clone())]);
                }
            }
            if read("poly_mix") {
                bindings |= buf_bit("poly_mix");
            }
        }
        out_stages.push(Stage {
            name,
            functions: fn_names,
            lines,
            carry_in_words: 4 * ext_in,
            bindings,
        });
    }

    Translation {
        files,
        strict_files,
        stages: out_stages,
        requested_stages: stages,
        chain_len: chain.len(),
        scratch_elems,
        carry_words,
        constants: header.constants,
        max_column,
        counts,
    }
}
