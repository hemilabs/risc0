// Copyright 2025 RISC Zero, Inc.
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

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

use risc0_build_kernel::{KernelBuild, KernelType};

#[cfg(all(
    feature = "metal",
    any(feature = "cuda", feature = "rocm", feature = "intel")
))]
compile_error!(
    "Feature 'metal' is mutually exclusive with 'cuda', 'rocm' and 'intel'. Enable only one GPU backend."
);

fn main() {
    if metal_enabled() {
        build_metal_kernels();
    }

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda_kernels();
    }

    if env::var("CARGO_FEATURE_ROCM").is_ok() {
        build_rocm_kernels();
    }

    if env::var("CARGO_FEATURE_INTEL").is_ok() {
        build_intel_kernels();
    }
}

fn build_intel_kernels() {
    println!("cargo:rerun-if-changed=kernels/intel");
    let icpx = risc0_build_kernel::find_icpx();
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    // Ahead of time for the build's Intel targets (`RISC0_INTEL_SYCL_TARGETS`, as rv32im, recursion
    // and keccak are), with SPIR-V for any other GPU. Built as SPIR-V alone, the driver compiled the
    // prover on the card at its first Groth16 proof (about 2 minutes on an Arc B580). The options are
    // those that compile used, so the code is the same, only built earlier.
    let args: Vec<std::ffi::OsString> = ["-fsycl", "-std=c++17", "-O3", "-Ikernels/intel"]
        .iter()
        .map(Into::into)
        .chain([risc0_build_kernel::intel_sycl_targets_arg().into()])
        .collect();
    let output = risc0_build_kernel::intel_build_library(
        &icpx,
        &args,
        &[PathBuf::from("kernels/intel/ffi.cpp")],
        &out_dir,
        "risc0_groth16_intel",
    );
    if !output.status.success() {
        panic!(
            "icpx failed for the Intel Groth16 prover:
{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    risc0_build_kernel::intel_link_library(&out_dir, "risc0_groth16_intel");
}

fn build_cuda_kernels() {
    let mut build = KernelBuild::new(KernelType::Cuda);
    if env::var("CARGO_FEATURE_SETUP").is_ok() {
        build.flag("-DSRS_READ_COEFFS");
    }
    build
        .files(["kernels/cuda/ffi.cu"])
        .deps(["kernels/cuda"])
        .flag("-D__ADX__")
        .include(env::var("DEP_BLST_C_SRC").unwrap())
        .include(env::var("DEP_SPPARK_ROOT").unwrap())
        .flag(&thp_pinned_define())
        .compile("risc0_groth16_cuda");

    println!("cargo:rustc-link-lib=static=blst");
}

fn build_rocm_kernels() {
    let output = "risc0_groth16_cuda";

    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=RISC0_HIP_ARCH");
    for entry in std::fs::read_dir("kernels/cuda").unwrap() {
        let path = entry.unwrap().path();
        println!("cargo:rerun-if-changed={}", path.display());
    }

    env::set_var("HIP_PLATFORM", "amd");

    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    // The sppark headers these kernels include (MSM, NTT): re-run when one changes, file by file,
    // as a directory would make cargo recurse into sppark's `rust/sppark` symlink loop.
    for file in risc0_build_kernel::kernel_source_files(Path::new(&sppark_root)) {
        println!("cargo:rerun-if-changed={}", file.display());
    }
    let hipcc = risc0_build_kernel::find_hipcc();
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let kernel_dir = std::fs::canonicalize("kernels/cuda").unwrap();

    let include_cuda2hip = format!("{sppark_root}/util/cuda2hip.hpp");
    let blst_src = env::var("DEP_BLST_C_SRC").unwrap();
    let arches = risc0_build_kernel::hip_arches();
    let mut base_flags: Vec<&str> = vec![
        "-x",
        "hip",
        "-std=c++17",
        "-O3",
        "-Wno-unused-function",
        "-Wno-unused-parameter",
        "-Wno-missing-braces",
        "-mllvm",
        "-amdgpu-early-inline-all=false",
        "-include",
        &include_cuda2hip,
        "-I",
        kernel_dir.to_str().unwrap(),
        "-I",
        &blst_src,
        "-I",
        &sppark_root,
        "-D__ADX__",
    ];
    // Windows hipcc rebuilds the command line for clang and drops unescaped quotes, which would
    // leave the header name bare; escaped, they reach clang as quotes.
    let thp_define = if cfg!(windows) {
        thp_pinned_define().replace('"', "\\\"")
    } else {
        thp_pinned_define()
    };
    base_flags.push(&thp_define);

    if env::var("CARGO_FEATURE_SETUP").is_ok() {
        base_flags.push("-DSRS_READ_COEFFS");
    }
    base_flags.extend(risc0_build_kernel::hip_host_flags());

    let obj = out_dir.join(format!(
        "groth16_ffi.{}",
        risc0_build_kernel::object_extension()
    ));
    let ffi_cu = kernel_dir.join("ffi.cu");
    risc0_build_kernel::hip_compile(&hipcc, &base_flags, &ffi_cu, &obj, &arches, None);

    // On MSVC, whose linker resolves symbols across libraries in any order, archive the object
    // alone and link blst beside it, as the CUDA build does.
    if risc0_build_kernel::is_msvc_target() {
        let _ = std::fs::remove_file(out_dir.join(format!("{output}.lib")));
        risc0_build_kernel::add_to_static_lib(&out_dir, output, &[obj]);
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static={output}");
        println!("cargo:rustc-link-lib=static=blst");
        risc0_build_kernel::emit_rocm_lib_link();
        return;
    }

    // Merge blst objects into our archive so that the linker can resolve
    // blst symbols referenced by host-side fp2_t/xyzz_t/jacobian_t code
    // in the same archive pass (avoids link-order issues with static libs).
    let build_dir = out_dir.parent().unwrap().parent().unwrap();
    let blst_lib_names: &[&str] = if cfg!(target_os = "windows") {
        &["blst.lib", "libblst.a"]
    } else {
        &["libblst.a"]
    };
    let mut blst_lib = None;
    for entry in std::fs::read_dir(build_dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name().to_string_lossy().starts_with("blst-") {
            for lib_name in blst_lib_names {
                let candidate = entry.path().join("out").join(lib_name);
                if candidate.exists() {
                    blst_lib = Some(candidate);
                    break;
                }
            }
            if blst_lib.is_some() {
                break;
            }
        }
    }
    let blst_lib = blst_lib.expect("could not find blst library in build directory");

    let lib_path = out_dir.join(format!("lib{output}.a"));
    let _ = std::fs::remove_file(&lib_path);

    let ar = risc0_build_kernel::find_ar_tool();

    // Create archive with our HIP object
    let status = Command::new(&ar)
        .arg("rcs")
        .arg(&lib_path)
        .arg(&obj)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");

    // Extract blst objects and merge them into our archive
    let blst_tmp = out_dir.join("blst_objs");
    let _ = std::fs::remove_dir_all(&blst_tmp);
    std::fs::create_dir_all(&blst_tmp).unwrap();
    let status = Command::new(&ar)
        .arg("x")
        .arg(&blst_lib)
        .current_dir(&blst_tmp)
        .status()
        .expect("failed to extract blst objects");
    assert!(status.success(), "ar extract failed");

    for entry in std::fs::read_dir(&blst_tmp).unwrap() {
        let obj_path = entry.unwrap().path();
        let status = Command::new(&ar)
            .arg("rcs")
            .arg(&lib_path)
            .arg(&obj_path)
            .status()
            .expect("failed to add blst object to archive");
        assert!(status.success(), "ar append failed");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={output}");

    // Link HIP runtime
    risc0_build_kernel::emit_rocm_lib_link();
}

/// `-DRISC0_THP_PINNED_H="<risc0-sys>/kernels/zkp/cuda/thp_pinned.h"` (see pinned_upload.cuh). That
/// directory can't go on the include path: its `cuda.h` would shadow the toolkit's.
fn thp_pinned_define() -> String {
    let root = PathBuf::from(env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap());
    let path = root.join("thp_pinned.h");
    println!("cargo:rerun-if-changed={}", path.display());
    // Forward slashes: inside the quoted header name a Windows backslash would be read as an
    // escape by the preprocessor (both work as separators there).
    let path = path.to_string_lossy().replace('\\', "/");
    format!("-DRISC0_THP_PINNED_H=\"{path}\"")
}

/// The `metal` feature on macOS/iOS: compile every `kernels/metal/*.metal` into
/// `metal_kernels_groth16.metallib` and point `GROTH16_METAL_PATH` at it for `src/lib.rs`
/// (`METAL_LIB`). Headers in `kernels/metal` (`bn254.h`, ...) are dependencies of every source.
///
/// The entry-point list `METAL_KERNEL_NAMES` is generated from the sources
/// (`$OUT_DIR/metal_kernel_names.rs`, see [metal_kernel_names]), so a kernel item adds a
/// `.metal` file without editing this script or `src/lib.rs`.
///
/// With no `.metal` source yet (the Groth16 kernels land in items G2..G6), the metallib is an
/// empty file and the name list is empty; nothing loads the library until the prover exists.
fn build_metal_kernels() {
    const OUTPUT: &str = "metal_kernels_groth16";
    const KERNEL_DIR: &str = "kernels/metal";

    // A directory: cargo re-runs this script when any file under it changes or appears.
    println!("cargo:rerun-if-changed={KERNEL_DIR}");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let sources = metal_sources(Path::new(KERNEL_DIR));

    let mut names = Vec::new();
    for src in &sources {
        let text = std::fs::read_to_string(src)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", src.display()));
        names.extend(metal_kernel_names(&text));
    }
    let mut seen = std::collections::HashSet::new();
    for name in &names {
        assert!(
            seen.insert(name.as_str()),
            "Metal kernel entry point `{name}` is defined twice in {KERNEL_DIR}"
        );
    }
    let list: String = names.iter().map(|n| format!("{n:?}, ")).collect();
    write_if_changed(
        &out_dir.join("metal_kernel_names.rs"),
        &format!("&[{list}]\n"),
    );

    let metallib = if sources.is_empty() {
        // No kernels yet: `metallib` cannot link zero objects, so embed an empty file.
        let path = out_dir.join(OUTPUT).with_extension("metallib");
        write_if_changed(&path, "");
        println!("cargo:{OUTPUT}={}", path.display());
        path
    } else {
        KernelBuild::new(KernelType::Metal)
            .files(&sources)
            .deps([KERNEL_DIR])
            .flag("-std=metal3.2")
            .flag("-O2")
            .compile(OUTPUT);
        // KernelBuild writes `skip-<output>` (empty) under RISC0_SKIP_BUILD_KERNELS.
        let name = if env::var("RISC0_SKIP_BUILD_KERNELS").is_ok() {
            format!("skip-{OUTPUT}")
        } else {
            OUTPUT.to_string()
        };
        out_dir.join(name).with_extension("metallib")
    };
    println!("cargo:rustc-env=GROTH16_METAL_PATH={}", metallib.display());
}

/// The `.metal` files directly in `dir`, sorted (generated sources, if any, would go to
/// `$OUT_DIR/metal` and be added by the item that generates them).
fn metal_sources(dir: &Path) -> Vec<PathBuf> {
    let mut sources: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "metal"))
                .collect()
        })
        .unwrap_or_default();
    sources.sort();
    sources
}

fn write_if_changed(path: &Path, contents: &str) {
    // An unchanged file keeps its mtime, so nothing downstream rebuilds.
    if std::fs::read(path).ok().as_deref() != Some(contents.as_bytes()) {
        std::fs::write(path, contents)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
    }
}

/// The metallib entry points defined by MSL source `text`, in order:
/// - `kernel void NAME(...)` outside a template gives `NAME`;
/// - an explicit instantiation carrying `[[host_name("NAME")]]` gives `NAME`;
/// - a `template <...> kernel void NAME(...)` definition gives nothing (it is not a function in
///   the library until instantiated).
///
/// Comments, string literals other than the host_name argument, and preprocessor conditionals are
/// not interpreted beyond stripping comments, so a kernel inside `#if 0` is still listed; the
/// runtime's `load_library` then fails fast on the missing name, which is the intended check.
fn metal_kernel_names(text: &str) -> Vec<String> {
    let code = strip_comments(text);
    let bytes = code.as_bytes();
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut names = Vec::new();
    // Start of the current declaration: just after the last `;`, `{` or `}` at any depth.
    let mut decl_start = 0;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b';' || b == b'{' || b == b'}' {
            decl_start = i + 1;
            i += 1;
            continue;
        }
        let at_word_start = i == 0 || !is_ident(bytes[i - 1]);
        if at_word_start && code[i..].starts_with("kernel") {
            let after = i + "kernel".len();
            if after < bytes.len() && !is_ident(bytes[after]) {
                let rest = code[after..].trim_start();
                if let Some(rest) = rest.strip_prefix("void") {
                    let rest = rest.trim_start();
                    let len = rest.bytes().take_while(|&c| is_ident(c)).count();
                    let name = &rest[..len];
                    let decl = &code[decl_start..i];
                    if let Some(host) = host_name(decl) {
                        names.push(host);
                    } else if !contains_word(decl, "template") && !name.is_empty() {
                        names.push(name.to_string());
                    }
                }
            }
            i = after;
            continue;
        }
        i += 1;
    }
    names
}

/// The argument of `host_name("...")` in `decl`, if any.
fn host_name(decl: &str) -> Option<String> {
    let at = decl.find("host_name")?;
    let rest = decl[at + "host_name".len()..]
        .trim_start()
        .strip_prefix('(')?;
    let rest = rest.trim_start().strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn contains_word(text: &str, word: &str) -> bool {
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    text.match_indices(word).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + word.len()..].chars().next();
        !before.is_some_and(is_ident) && !after.is_some_and(is_ident)
    })
}

/// `text` with `//` and `/* */` comments replaced by spaces (string literals are respected).
fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut in_str = false;
    while let Some(c) = chars.next() {
        if in_str {
            out.push(c);
            if c == '\\' {
                if let Some(n) = chars.next() {
                    out.push(n);
                }
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match (c, chars.peek()) {
            ('"', _) => {
                in_str = true;
                out.push(c);
            }
            ('/', Some('/')) => {
                for n in chars.by_ref() {
                    if n == '\n' {
                        out.push('\n');
                        break;
                    }
                }
            }
            ('/', Some('*')) => {
                chars.next();
                let mut prev = ' ';
                for n in chars.by_ref() {
                    if prev == '*' && n == '/' {
                        break;
                    }
                    prev = n;
                }
                out.push(' ');
            }
            _ => out.push(c),
        }
    }
    out
}

/// Whether to build the Metal kernels: the `metal` feature on a macOS/iOS target (on any other
/// target the guard in src/lib.rs rejects the feature).
fn metal_enabled() -> bool {
    env::var("CARGO_FEATURE_METAL").is_ok()
        && env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos" || os == "ios")
}
