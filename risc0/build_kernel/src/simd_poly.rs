//! A circuit's CPU constraint polynomial, 16 cycles at a time.
//!
//! The generated `poly_fp` sources (rv32im's `rust_poly_fp_<n>.cpp`, recursion's `poly_fp.cpp`)
//! evaluate the polynomial for one cycle: ~25,000 field
//! operations of straight-line code, megabytes of it, run once per cycle of the 4x-extended
//! domain. Here it is rewritten so that every value is 16 cycles wide (`FpV` / `FpExtV` in
//! risc0-sys's `cxx/fp_simd.h`, one AVX-512 register or two AVX2 ones): the same code runs once per
//! 16 cycles, its trace reads become vector loads of 16 consecutive rows, and its constants and
//! mix powers stay scalars that are broadcast. The results are bit-identical to the scalar code.
//!
//! The rewrite is mechanical and done at build time, so it follows the generated scalar code.
//! What it relies on, and checks:
//! - every function is `FpExt name(size_t cycle, size_t steps, FpExt* poly_mix, ...)`, in
//!   `namespace risc0::circuit::<circuit>`, and all but `poly_fp` are called exactly once, and
//!   its pointer parameters are either trace buffers (`args[i]`, passed on
//!   by name) or the caller's local arrays (`Fp xN[..]` / `FpExt xN[..]`);
//! - trace reads that depend on the row are `<buffer>[<expr with cycle>]`, which for a 16-row
//!   block starting at `cycle` read 16 consecutive elements unless the block wraps around the
//!   end of the column (`ld` in fp_simd.h handles both);
//! - only local arrays are written.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    /// A trace buffer: stays scalar, read 16 rows at a time.
    Buf,
    /// A caller's `Fp` array.
    FpArr,
    /// A caller's `FpExt` array.
    ExtArr,
    /// An `FpExt` value.
    Ext,
}

struct Func {
    name: String,
    params: Vec<(String, String)>,
}

fn parse_def(line: &str) -> Option<Func> {
    let rest = line.strip_prefix("FpExt ")?;
    let body = rest
        .strip_suffix(") {")
        .or_else(|| rest.strip_suffix(");"))?;
    let (name, params) = body.split_once('(')?;
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    let params = params
        .split(", ")
        .map(|p| {
            let (t, n) = p.rsplit_once(' ').expect("parameter");
            (t.to_string(), n.to_string())
        })
        .collect();
    Some(Func {
        name: name.to_string(),
        params,
    })
}

fn strip_comments(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find("/*") {
        out.push_str(&rest[..i]);
        let j = rest[i..].find("*/").expect("unterminated comment");
        rest = &rest[i + j + 2..];
    }
    out.push_str(rest);
    out
}

fn is_ident(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `args[N]`
fn is_args_elem(s: &str) -> bool {
    s.strip_prefix("args[")
        .and_then(|r| r.strip_suffix(']'))
        .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
}

/// Rewrites `<buf>[<expr with cycle>]` into `ld(<buf>, <expr>, mask)`, checking that `<buf>` is a
/// trace buffer.
fn rewrite_loads(line: &str, kinds: &HashMap<String, Kind>) -> String {
    let mut out = String::new();
    let mut copied = 0;
    let mut search = 0;
    while let Some(rel) = line[search..].find('[') {
        let open = search + rel;
        let close = open + line[open..].find(']').expect("unbalanced [");
        search = close + 1;
        let index = &line[open + 1..close];
        if !index.contains("cycle") {
            continue;
        }
        // The base: `args[N]` or the identifier right before the `[`.
        let before = &line[..open];
        let base_start = if before.ends_with(']') {
            before.rfind("args[").expect("args[")
        } else {
            before
                .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .map_or(0, |i| i + 1)
        };
        let base = &line[base_start..open];
        if !is_args_elem(base) {
            assert_eq!(
                kinds.get(base),
                Some(&Kind::Buf),
                "row-dependent read of {base}, not a trace buffer: {line}"
            );
        }
        out.push_str(&line[copied..base_start]);
        out.push_str(&format!("ld({base}, {index}, mask)"));
        copied = close + 1;
    }
    out.push_str(&line[copied..]);
    out
}

fn signature(f: &Func, kinds: &HashMap<String, Kind>) -> String {
    if f.name == "poly_fp" {
        return "FpExtV poly_fp(size_t cycle, size_t steps, const FpExt* poly_mix, \
                const Fp* const* args)"
            .to_string();
    }
    let params: Vec<String> = f.params[3..]
        .iter()
        .map(|(_, n)| {
            let t = match kinds[n] {
                Kind::Buf => "const Fp* ",
                Kind::FpArr => "FpV* ",
                Kind::ExtArr => "FpExtV* ",
                Kind::Ext => "const FpExtV& ",
            };
            format!("{t}{n}")
        })
        .collect();
    format!(
        "FpExtV {}(size_t cycle, size_t steps, const FpExt* poly_mix, {})",
        f.name,
        params.join(", ")
    )
}

/// Writes the 16-wide sources for namespace `ns` into `out_dir`: one per scalar source, plus the
/// `extern "C"` entry `<entry>_x16_<ns>`. Returns their paths.
pub fn generate(
    sources: &[PathBuf],
    circuit: &str,
    entry: &str,
    ns: &str,
    out_dir: &Path,
) -> Vec<PathBuf> {
    let texts: Vec<String> = sources
        .iter()
        .map(|p| std::fs::read_to_string(p).unwrap())
        .collect();

    // Pass 1: definitions, their local arrays, and the call that reaches each.
    let mut funcs: HashMap<String, Func> = HashMap::new();
    let mut local_arrays: HashMap<String, HashMap<String, Kind>> = HashMap::new();
    let mut calls: HashMap<String, (String, Vec<String>)> = HashMap::new();
    for text in &texts {
        let mut cur: Option<String> = None;
        for line in text.lines() {
            if line.ends_with(") {") {
                if let Some(f) = parse_def(line) {
                    local_arrays.insert(f.name.clone(), HashMap::new());
                    cur = Some(f.name.clone());
                    funcs.insert(f.name.clone(), f);
                    continue;
                }
            }
            let Some(fname) = cur.clone() else { continue };
            if line == "}" {
                cur = None;
                continue;
            }
            // `  Fp x33[1007];` / `  FpExt x34[144];`
            for (prefix, kind) in [("  Fp ", Kind::FpArr), ("  FpExt ", Kind::ExtArr)] {
                if let Some(decl) = line.strip_prefix(prefix).and_then(|r| r.strip_suffix("];")) {
                    if let Some((name, n)) = decl.split_once('[') {
                        if is_ident(name) && n.chars().all(|c| c.is_ascii_digit()) {
                            local_arrays
                                .get_mut(&fname)
                                .unwrap()
                                .insert(name.into(), kind);
                        }
                    }
                }
            }
            // `  auto xN = <circuit>_K(cycle, steps, poly_mix, a, b, ...);`
            if let Some(i) = line.find("(cycle, steps, poly_mix, ") {
                let callee_start = line[..i]
                    .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .map_or(0, |j| j + 1);
                let callee = line[callee_start..i].to_string();
                let args = strip_comments(
                    line[i + "(cycle, steps, poly_mix, ".len()..]
                        .strip_suffix(");")
                        .expect("call"),
                );
                let args = args.split(',').map(|a| a.trim().to_string()).collect();
                assert!(
                    calls.insert(callee.clone(), (fname, args)).is_none(),
                    "{callee} is called more than once"
                );
            }
        }
    }
    assert!(funcs.contains_key("poly_fp"), "no poly_fp in the sources");

    // Pass 2: each parameter's kind, from what its one caller passes.
    let mut kinds: HashMap<String, HashMap<String, Kind>> = HashMap::new();
    kinds.insert("poly_fp".into(), HashMap::new());
    fn resolve(
        name: &str,
        funcs: &HashMap<String, Func>,
        calls: &HashMap<String, (String, Vec<String>)>,
        locals: &HashMap<String, HashMap<String, Kind>>,
        kinds: &mut HashMap<String, HashMap<String, Kind>>,
    ) {
        if kinds.contains_key(name) {
            return;
        }
        let (caller, args) = calls
            .get(name)
            .unwrap_or_else(|| panic!("{name} is never called"));
        resolve(caller, funcs, calls, locals, kinds);
        let params = &funcs[name].params[3..];
        assert_eq!(params.len(), args.len(), "arity of {name}");
        let mut k = HashMap::new();
        for ((t, n), a) in params.iter().zip(args) {
            let kind = if t == "FpExt" {
                Kind::Ext
            } else if is_args_elem(a) {
                Kind::Buf
            } else if let Some(&kind) = kinds[caller].get(a.as_str()) {
                kind
            } else if let Some(&kind) = locals[caller].get(a.as_str()) {
                kind
            } else {
                panic!("{name}: cannot tell what {t} {n} is from `{a}` in {caller}");
            };
            let expected_ptr = match t.as_str() {
                "Fp*" => matches!(kind, Kind::Buf | Kind::FpArr),
                "FpExt*" => kind == Kind::ExtArr,
                "FpExt" => kind == Kind::Ext,
                _ => false,
            };
            assert!(expected_ptr, "{name}: {t} {n} gets a {kind:?}");
            k.insert(n.clone(), kind);
        }
        kinds.insert(name.to_string(), k);
    }
    let names: Vec<String> = funcs.keys().cloned().collect();
    for name in &names {
        resolve(name, &funcs, &calls, &local_arrays, &mut kinds);
    }

    // Pass 3: rewrite.
    std::fs::create_dir_all(out_dir).unwrap();
    let mut outputs = Vec::new();
    for (src, text) in sources.iter().zip(&texts) {
        let mut out = String::with_capacity(text.len() + text.len() / 8);
        let mut cur: Option<String> = None;
        // The function being rewritten: its signature and body, emitted at its end.
        let mut def_sig = String::new();
        let mut body: Vec<String> = Vec::new();
        for line in text.lines() {
            // fp.h / fpext.h come with fp_simd.h; the standard headers stay.
            if line == "#include \"fp.h\"" {
                out.push_str("#include \"fp_simd.h\"\n");
                continue;
            }
            if line == "#include \"fpext.h\"" {
                continue;
            }
            if line == format!("namespace risc0::circuit::{circuit} {{") {
                out.push_str(&format!(
                    "using namespace risc0;\nusing namespace risc0::simd;\n\
                     namespace risc0::circuit::{circuit}::{ns} {{\n"
                ));
                continue;
            }
            if line.starts_with("} // namespace") {
                out.push_str(&format!(
                    "}} // namespace risc0::circuit::{circuit}::{ns}\n"
                ));
                continue;
            }
            if line.starts_with("FpExt ") {
                if let Some(f) = parse_def(line) {
                    let sig = signature(&f, &kinds[&f.name]);
                    if line.ends_with(") {") {
                        def_sig = sig;
                        body.clear();
                        cur = Some(f.name);
                    } else {
                        out.push_str(&sig);
                        out.push_str(";\n");
                    }
                    continue;
                }
            }
            let Some(fname) = &cur else {
                out.push_str(line);
                out.push('\n');
                continue;
            };
            if line == "}" {
                let calls_out = calls.values().any(|(caller, _)| caller == fname);
                if fname == "poly_fp" && !calls_out && body.len() > 2 * CHUNK_STATEMENTS {
                    out.push_str(&chunked_poly_fp(&def_sig, &body));
                } else {
                    out.push_str(&def_sig);
                    out.push_str(" {\n");
                    for l in &body {
                        out.push_str(l);
                        out.push('\n');
                    }
                    out.push_str("}\n");
                }
                cur = None;
                continue;
            }
            let fk = &kinds[fname];
            let mut l = line.to_string();
            if let Some(r) = l.strip_prefix("  Fp ").filter(|r| r.ends_with("];")) {
                l = format!("  FpV {r}");
            } else if let Some(r) = l.strip_prefix("  FpExt ").filter(|r| r.ends_with("];")) {
                l = format!("  FpExtV {r}");
            } else if let Some(r) = l.strip_prefix("  FpExt ").filter(|r| r.contains(" = ")) {
                l = format!("  FpExtV {r}");
            }
            if l.contains("cycle") && l.contains('[') {
                l = rewrite_loads(&l, fk);
            }
            // Only local arrays are written.
            if let Some((lhs, _)) = l.trim_start().split_once("] = ") {
                if let Some((base, _)) = lhs.split_once('[') {
                    if let Some(k) = fk.get(base) {
                        assert!(
                            matches!(k, Kind::FpArr | Kind::ExtArr),
                            "{fname} writes to {base}: {line}"
                        );
                    }
                }
            }
            body.push(l);
        }
        assert!(
            out.contains("#include \"fp_simd.h\""),
            "{} does not include fp.h",
            src.display()
        );
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let path = out_dir.join(format!("{stem}_x16_{ns}.cpp"));
        write_if_changed(&path, &out);
        outputs.push(path);
    }

    let entry = format!(
        r#"#include "fp_simd.h"
#include <cstring>
#include <exception>

namespace risc0::circuit::{circuit}::{ns} {{
risc0::simd::FpExtV poly_fp(size_t cycle, size_t steps, const risc0::FpExt* poly_mix,
                            const risc0::Fp* const* args);
}} // namespace risc0::circuit::{circuit}::{ns}

// The polynomial at the 16 cycles from `cycle` (a multiple of 16) into result[0..16].
extern "C" const char* {entry}_x16_{ns}(size_t cycle,
                                       size_t steps,
                                       const risc0::FpExt* poly_mix,
                                       const risc0::Fp* const* args,
                                       risc0::FpExt* result) {{
  try {{
    risc0::simd::FpExtV v = risc0::circuit::{circuit}::{ns}::poly_fp(cycle, steps, poly_mix, args);
    alignas(64) uint32_t lanes[4][risc0::simd::kLanes];
    for (size_t i = 0; i < 4; i++) {{
      risc0::simd::vstore(lanes[i], v.e[i].v);
    }}
    for (size_t j = 0; j < risc0::simd::kLanes; j++) {{
      result[j] = risc0::FpExt(risc0::Fp::fromRaw(lanes[0][j]),
                               risc0::Fp::fromRaw(lanes[1][j]),
                               risc0::Fp::fromRaw(lanes[2][j]),
                               risc0::Fp::fromRaw(lanes[3][j]));
    }}
  }} catch (const std::exception& err) {{
    return strdup(err.what());
  }}
  return nullptr;
}}
"#
    );
    let path = out_dir.join(format!("poly_fp_x16_{ns}_entry.cpp"));
    write_if_changed(&path, &entry);
    outputs.push(path);
    outputs
}

/// Statements per part of a `poly_fp` that is split (`chunked_poly_fp`).
const CHUNK_STATEMENTS: usize = 1500;

/// The `x<N>` names in `expr`.
fn value_names(expr: &str) -> Vec<&str> {
    let b = expr.as_bytes();
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut names = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'x' && (i == 0 || !ident(b[i - 1])) {
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 && (j == b.len() || !ident(b[j])) {
                names.push(&expr[i..j]);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    names
}

/// `expr` with each name in `vars` read from the frame (`f.<name>`).
fn qualify(expr: &str, vars: &std::collections::HashSet<String>) -> String {
    let mut out = String::with_capacity(expr.len() + 16);
    let mut last = 0;
    for name in value_names(expr) {
        let at = name.as_ptr() as usize - expr.as_ptr() as usize;
        if vars.contains(name) {
            out.push_str(&expr[last..at]);
            out.push_str("f.");
            out.push_str(name);
            last = at + name.len();
        }
    }
    out.push_str(&expr[last..]);
    out
}

/// A `poly_fp` without calls that is too long for compilers to optimize in reasonable time as one
/// function (recursion's ~12,000 statements take clang over 10 minutes), as parts of
/// `CHUNK_STATEMENTS` statements. Its constants move out to namespace scope; a value a later part
/// reads is kept in a frame the parts share. `body` is the rewritten body.
fn chunked_poly_fp(sig: &str, body: &[String]) -> String {
    use std::collections::{HashMap, HashSet};

    #[derive(Clone, Copy, PartialEq, PartialOrd)]
    enum Ty {
        Scalar,
        Vector,
        Ext,
    }
    let mut consts = Vec::new();
    // (name, type, expression) of each value, then the returned name.
    let mut defs: Vec<(String, Ty, String)> = Vec::new();
    let mut ret = None;
    let mut ty: HashMap<String, Ty> = HashMap::new();
    for line in body {
        let l = line.trim();
        if l.is_empty() || l.starts_with("//") || l == "size_t mask = steps - 1;" {
            continue;
        }
        if l.starts_with("constexpr ") {
            let name = value_names(l)[0];
            ty.insert(name.to_string(), Ty::Scalar);
            consts.push(l.to_string());
            continue;
        }
        if let Some(r) = l.strip_prefix("return ").and_then(|r| r.strip_suffix(';')) {
            ret = Some(r.to_string());
            continue;
        }
        let (decl, expr) = l
            .strip_suffix(';')
            .and_then(|l| l.split_once(" = "))
            .unwrap_or_else(|| panic!("poly_fp: unexpected statement `{l}`"));
        let (t, name) = decl.split_once(' ').expect("declaration");
        let expr = strip_comments(expr);
        let t = match t {
            "FpExtV" => Ty::Ext,
            "auto" if expr.starts_with("ld(") => Ty::Vector,
            "auto" if expr.starts_with("args[") => Ty::Scalar,
            "auto" => value_names(&expr)
                .iter()
                .map(|n| {
                    *ty.get(*n)
                        .unwrap_or_else(|| panic!("poly_fp: {n} used before defined"))
                })
                .fold(Ty::Scalar, |a, b| if b > a { b } else { a }),
            _ => panic!("poly_fp: unexpected declaration `{l}`"),
        };
        ty.insert(name.to_string(), t);
        defs.push((name.to_string(), t, expr));
    }
    let ret = ret.expect("poly_fp returns nothing");

    let parts: Vec<&[(String, Ty, String)]> = defs.chunks(CHUNK_STATEMENTS).collect();
    let part_of: HashMap<&str, usize> = parts
        .iter()
        .enumerate()
        .flat_map(|(i, p)| p.iter().map(move |(n, _, _)| (n.as_str(), i)))
        .collect();
    // Values a later part reads, and the one returned.
    let mut shared: HashSet<String> = HashSet::new();
    for (i, part) in parts.iter().enumerate() {
        for (_, _, expr) in part.iter() {
            for n in value_names(expr) {
                if part_of.get(n).is_some_and(|&d| d < i) {
                    shared.insert(n.to_string());
                }
            }
        }
    }
    shared.insert(ret.clone());

    let mut out = String::new();
    for c in &consts {
        out.push_str(c);
        out.push('\n');
    }
    out.push_str("struct PolyFpFrame {\n");
    for (name, t, _) in &defs {
        if shared.contains(name) {
            let t = match t {
                Ty::Scalar => "Fp",
                Ty::Vector => "FpV",
                Ty::Ext => "FpExtV",
            };
            out.push_str(&format!("  {t} {name};\n"));
        }
    }
    out.push_str("};\n");
    for (i, part) in parts.iter().enumerate() {
        out.push_str(&format!(
            "#if defined(_MSC_VER) && !defined(__clang__)\n__declspec(noinline)\n#else\n\
             __attribute__((noinline))\n#endif\n\
             static void poly_fp_part_{i}(size_t cycle, size_t steps, const FpExt* poly_mix, \
             const Fp* const* args, PolyFpFrame& f) {{\n  size_t mask = steps - 1;\n  \
             (void)cycle; (void)mask; (void)poly_mix; (void)args;\n"
        ));
        let earlier: HashSet<String> = shared
            .iter()
            .filter(|n| part_of.get(n.as_str()).is_some_and(|&d| d < i))
            .cloned()
            .collect();
        for (name, t, expr) in part.iter() {
            let decl = if *t == Ty::Ext { "FpExtV" } else { "auto" };
            out.push_str(&format!("  {decl} {name} = {};\n", qualify(expr, &earlier)));
            if shared.contains(name) {
                out.push_str(&format!("  f.{name} = {name};\n"));
            }
        }
        out.push_str("}\n");
    }
    out.push_str(sig);
    out.push_str(" {\n  PolyFpFrame f;\n");
    for i in 0..parts.len() {
        out.push_str(&format!(
            "  poly_fp_part_{i}(cycle, steps, poly_mix, args, f);\n"
        ));
    }
    out.push_str(&format!("  return f.{ret};\n}}\n"));
    out
}

/// Leaves an unchanged file alone, so its object is not rebuilt for nothing.
fn write_if_changed(path: &Path, contents: &str) {
    if std::fs::read_to_string(path).ok().as_deref() != Some(contents) {
        std::fs::write(path, contents).unwrap();
    }
}

/// Builds the 16-wide polynomial of a circuit into static libraries `<lib>_avx512` and
/// `<lib>_avx2` (`<entry>_x16_avx512`, `<entry>_x16_avx2`), with `<entry>_x16_available()`
/// telling the prover whether they are there. x86-64 only; elsewhere, and on Windows without
/// clang-cl (MSVC takes hours on these files), only `available()` (returning false) is built and
/// the prover uses the scalar polynomial.
///
/// `include_dirs` must reach `fp.h`, `fpext.h` and `fp_simd.h` (risc0-sys's `cxx`).
pub fn build_x16_poly(
    sources: &[PathBuf],
    circuit: &str,
    entry: &str,
    lib: &str,
    include_dirs: &[PathBuf],
) {
    println!("cargo:rerun-if-env-changed=RISC0_CLANG_CL");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join(lib);
    std::fs::create_dir_all(&out_dir).unwrap();
    let x86_64 = std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64");
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    let clang_cl = if msvc && x86_64 {
        find_clang_cl()
    } else {
        None
    };
    let available = std::env::var("RISC0_SKIP_BUILD_KERNELS").is_err()
        && x86_64
        && (!msvc || clang_cl.is_some());
    if x86_64 && msvc && clang_cl.is_none() {
        println!(
            "cargo:warning=clang-cl not found (set RISC0_CLANG_CL or install LLVM): the CPU \
             prover of {circuit} uses the scalar constraint polynomial"
        );
    }
    let avail = out_dir.join("x16_available.cpp");
    write_if_changed(
        &avail,
        &format!("extern \"C\" bool {entry}_x16_available() {{ return {available}; }}\n"),
    );
    if !available {
        cc::Build::new().cpp(true).file(&avail).compile(lib);
        return;
    }

    std::thread::scope(|scope| {
        for (ns, flag) in [("avx512", "-mavx512f"), ("avx2", "-mavx2")] {
            let (out_dir, clang_cl, avail) = (&out_dir, &clang_cl, &avail);
            scope.spawn(move || {
                let files = generate(sources, circuit, entry, ns, &out_dir.join(ns));
                let mut build = cc::Build::new();
                build
                    .cpp(true)
                    .debug(false)
                    .warnings(false)
                    .includes(include_dirs)
                    .define("RISC0_SIMD_NS", ns)
                    .files(&files);
                if ns == "avx512" {
                    build.file(avail);
                }
                if msvc {
                    build
                        .compiler(clang_cl.as_ref().unwrap())
                        .define("NOMINMAX", None)
                        .define("_CRT_NONSTDC_NO_DEPRECATE", None)
                        .flag("/std:c++20")
                        .flag("/EHsc")
                        .flag(flag);
                } else {
                    build.flag("-std=c++17").flag(flag).flag_if_supported("-g0");
                }
                build.compile(&format!("{lib}_{ns}"));
            });
        }
    });
}

/// clang-cl: `RISC0_CLANG_CL`, else on PATH, else LLVM's default install directory.
fn find_clang_cl() -> Option<PathBuf> {
    let works = |p: &Path| {
        std::process::Command::new(p)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    };
    if let Some(p) = std::env::var_os("RISC0_CLANG_CL") {
        let p = PathBuf::from(p);
        return works(&p).then_some(p);
    }
    [
        PathBuf::from("clang-cl"),
        PathBuf::from(r"C:\Program Files\LLVM\bin\clang-cl.exe"),
    ]
    .into_iter()
    .find(|p| works(p))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    /// The rewrite accepts both circuits' generated polynomials (and checks them as it goes).
    #[test]
    fn generates_for_both_circuits() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let out = PathBuf::from(
            std::env::var_os("SIMD_POLY_TEST_OUT").unwrap_or_else(|| std::env::temp_dir().into()),
        )
        .join("simd_poly_test");
        let rv32im: Vec<PathBuf> = (0..4)
            .map(|i| {
                root.join(format!(
                    "circuit/rv32im-sys/kernels/cxx/rust_poly_fp_{i}.cpp"
                ))
            })
            .collect();
        let files = super::generate(
            &rv32im,
            "rv32im_v2",
            "rv32im_entry",
            "avx2",
            &out.join("rv32im"),
        );
        assert_eq!(files.len(), 5);
        let recursion = [root.join("circuit/recursion-sys/kernels/cxx/poly_fp.cpp")];
        let files = super::generate(
            &recursion,
            "recursion",
            "recursion_entry",
            "avx2",
            &out.join("recursion"),
        );
        assert_eq!(files.len(), 2);
    }
}
