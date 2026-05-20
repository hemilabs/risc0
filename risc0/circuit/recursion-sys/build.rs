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

/// Build-cache stamp utilities for the Intel SYCL recursion kernel.
///
/// Replaces the previous `"built"` literal stamp (which never invalidated on
/// source changes) with content-addressed SHA-256 hashing. One stamp covers
/// the recursion eval_check compilation. Stamp file format is a small TOML:
///
///     hash         = "<sha256 hex>"
///     built_at     = "<utc rfc3339 timestamp>"
///     icpx_version = "<first line of `icpx --version`>"
///     status       = "ok" | "failed"
///
/// Cache hit requires all of: artifact present, stamp parses, hash matches,
/// status is "ok". Anything else triggers a rebuild.
mod stamp {
    use sha2::{Digest, Sha256};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn update_len(h: &mut Sha256, n: u64) {
        h.update(n.to_le_bytes());
    }

    #[allow(dead_code)] // kept as a convenience entry-point for callers that
                       // don't need label/content separation
    pub fn hash(files: &[&Path], extras: &[&str]) -> String {
        let pairs: Vec<(PathBuf, PathBuf)> = files
            .iter()
            .map(|p| (p.to_path_buf(), p.to_path_buf()))
            .collect();
        hash_labeled(&pairs, extras)
    }

    pub fn hash_labeled(items: &[(PathBuf, PathBuf)], extras: &[&str]) -> String {
        let mut sorted: Vec<&(PathBuf, PathBuf)> = items.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let mut h = Sha256::new();
        h.update(b"STAMP-V2\0");
        for (label, content_path) in sorted {
            let path_bytes = label.as_os_str().as_encoded_bytes();
            h.update(b"FILE");
            update_len(&mut h, path_bytes.len() as u64);
            h.update(path_bytes);
            match std::fs::read(content_path) {
                Ok(b) => {
                    h.update(b"OK");
                    update_len(&mut h, b.len() as u64);
                    h.update(&b);
                }
                Err(_) => {
                    h.update(b"MISS");
                    update_len(&mut h, 0);
                }
            }
        }
        for s in extras {
            let bytes = s.as_bytes();
            h.update(b"EXTRA");
            update_len(&mut h, bytes.len() as u64);
            h.update(bytes);
        }
        format!("{:x}", h.finalize())
    }

    pub fn icpx_version(icpx: &Path) -> String {
        let out = match Command::new(icpx).arg("--version").output() {
            Ok(o) => o,
            Err(e) => return format!("unknown-exec:{}", e.raw_os_error().unwrap_or(-1)),
        };
        let stdout = String::from_utf8_lossy(&out.stdout);
        let first = stdout.lines().next().map(|s| s.trim()).unwrap_or("");
        let sanitized: String = first.chars().filter(|c| !c.is_control()).collect();
        if !out.status.success() {
            return format!("unknown-exit:{}", out.status.code().unwrap_or(-1));
        }
        if sanitized.is_empty() {
            return "unknown".into();
        }
        sanitized
    }

    pub fn now() -> String {
        Command::new("date")
            .arg("-u")
            .arg("+%Y-%m-%dT%H:%M:%SZ")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".into())
    }

    pub struct Stamp {
        pub hash: String,
        pub status: String,
    }

    pub fn read(path: &Path) -> Option<Stamp> {
        let s = std::fs::read_to_string(path).ok()?;
        let mut hash = None;
        let mut status = None;
        for line in s.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                let k = k.trim();
                let v = v.trim().trim_matches('"');
                match k {
                    "hash" => hash = Some(v.to_string()),
                    "status" => status = Some(v.to_string()),
                    _ => {}
                }
            }
        }
        Some(Stamp {
            hash: hash?,
            status: status?,
        })
    }

    pub fn write(path: &Path, hash_value: &str, icpx_version: &str, status: &str) {
        let now = now();
        let escape = |s: &str| -> String {
            let cleaned: String = s.chars().filter(|c| !c.is_control()).collect();
            cleaned.replace('\\', "\\\\").replace('"', "\\\"")
        };
        let body = format!(
            "hash = \"{}\"\nbuilt_at = \"{}\"\nicpx_version = \"{}\"\nstatus = \"{}\"\n",
            escape(hash_value),
            escape(&now),
            escape(icpx_version),
            escape(status),
        );
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp = {
            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            let name = path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "stamp".into());
            parent.join(format!("{name}.{pid}.{nanos}.tmp"))
        };
        std::fs::write(&tmp, body).expect("failed to write build-cache stamp tmp");
        std::fs::rename(&tmp, path).expect("failed to rename build-cache stamp");
    }

    pub fn need_rebuild(artifact: &Path, stamp_path: &Path, expected_hash: &str) -> bool {
        need_rebuild_multi(&[artifact], stamp_path, expected_hash)
    }

    pub fn need_rebuild_multi(
        artifacts: &[&Path],
        stamp_path: &Path,
        expected_hash: &str,
    ) -> bool {
        if artifacts.iter().any(|p| !p.exists()) {
            return true;
        }
        match read(stamp_path) {
            None => true,
            Some(s) => s.hash != expected_hash || s.status != "ok",
        }
    }
}

const RECURSION_INTEL_STAMP_VERSION: &str = "recursion-v2";

/// Glob `*.h`/`*.hpp`/`*.cuh` from an absolute external include root
/// (`DEP_RISC0_SYS_CXX_ROOT`) so upstream `fp.h`/`fpext.h` content edits
/// invalidate the stamp. Returns (label, content_path) pairs where the
/// label is a portable virtual `cxx_root/<basename>` path so the cache key
/// stays stable across machines with different absolute workspace roots.
fn cxx_root_headers(cxx_root: &str) -> Vec<(PathBuf, PathBuf)> {
    let mut out = Vec::new();
    for ext in &["*.h", "*.hpp", "*.cuh"] {
        let pattern = format!("{cxx_root}/{ext}");
        if let Ok(paths) = glob::glob(&pattern) {
            for p in paths.filter_map(Result::ok) {
                if let Some(name) = p.file_name() {
                    let label = PathBuf::from("cxx_root").join(name);
                    out.push((label, p));
                }
            }
        }
    }
    out
}

/// Inputs for the recursion eval_check kernel: poly_fp + the SYCL kernel
/// wrapper + transitive cxx_root headers + local cxx headers/.inc fragments
/// (poly_fp.cpp transitively includes them).
fn compute_recursion_intel_hash(cxx_root: &str, icpx_version: &str) -> String {
    let mut in_tree: Vec<PathBuf> = vec![
        PathBuf::from("kernels/intel/eval_check.cpp"),
        PathBuf::from("kernels/cxx/poly_fp.cpp"),
        PathBuf::from("build.rs"),
    ];
    in_tree.extend(glob_paths("kernels/cxx/*.h"));
    in_tree.extend(glob_paths("kernels/cxx/*.h.inc"));
    in_tree.extend(glob_paths("kernels/cxx/*.cpp.inc"));
    let mut pairs: Vec<(PathBuf, PathBuf)> =
        in_tree.into_iter().map(|p| (p.clone(), p)).collect();
    pairs.extend(cxx_root_headers(cxx_root));
    stamp::hash_labeled(
        &pairs,
        &[icpx_version, RECURSION_INTEL_STAMP_VERSION, "recursion"],
    )
}

#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("Features 'cuda' and 'rocm' are mutually exclusive. Enable only one GPU backend.");

fn main() {
    build_cpu_kernels();

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

fn build_cpu_kernels() {
    rerun_if_changed("kernels/cxx");
    KernelBuild::new(KernelType::Cpp)
        .files(glob_paths("kernels/cxx/*.cpp"))
        .include(env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap())
        .compile("risc0_recursion_cpu");
}

fn build_cuda_kernels() {
    rerun_if_changed("kernels/cuda");

    let cuda_root = env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap();
    let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();

    // Split build: step/ffi kernels get -maxrregcount=96 for higher occupancy,
    // while eval_check/supra kernels keep unrestricted registers.
    KernelBuild::new(KernelType::Cuda)
        .files(glob_paths("kernels/cuda/step_*.cu"))
        .file("kernels/cuda/ffi.cu")
        .deps(["kernels/cuda"])
        .flag("-DFEATURE_BABY_BEAR")
        .flag("-maxrregcount=96")
        .include(&cuda_root)
        .include(&cxx_root)
        .include(&sppark_root)
        .compile("risc0_recursion_cuda_step");

    KernelBuild::new(KernelType::Cuda)
        .file("kernels/cuda/eval_check.cu")
        .file("kernels/cuda/ffi_supra.cu")
        .file("kernels/cuda/sppark.cu")
        .deps(["kernels/cuda"])
        .flag("-DFEATURE_BABY_BEAR")
        .include(&cuda_root)
        .include(&cxx_root)
        .include(&sppark_root)
        .compile("risc0_recursion_cuda");
}

fn build_rocm_kernels() {
    let output = "risc0_recursion_cuda";
    rerun_if_changed("kernels/cuda");
    println!("cargo:rerun-if-env-changed=RISC0_HIP_ARCH");

    env::set_var("HIP_PLATFORM", "amd");

    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    let cuda_root = env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap();
    let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
    let hipcc = risc0_build_kernel::find_hipcc();
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let kernel_dir = std::fs::canonicalize("kernels/cuda").unwrap();

    // Single-TU approach: amalgamate all .cu files EXCEPT ffi_supra.cu and sppark.cu
    // (which use sppark types that conflict with risc0's fpext.h in the same TU).
    let cu_files = glob_paths("kernels/cuda/*.cu");
    // Three-group approach:
    // Group A (risc0 types): ffi.cu + step_*.cu — amalgamated
    // Group B (sppark eval): ffi_supra.cu + eval_check.cu — amalgamated
    // Group C (sppark host): sppark.cu — compiled separately (gpu_t.cuh's fmt()
    //   can't compile for device when combined with device kernels)
    let mut risc0_amalg = String::new();
    let mut supra_amalg = String::new();
    let mut separate_files = Vec::new();
    for cu in &cu_files {
        let name = cu.file_name().unwrap().to_str().unwrap();
        let abs = std::fs::canonicalize(cu).unwrap();
        let line = format!("#include \"{}\"\n", abs.display());
        if name == "sppark.cu" {
            separate_files.push(cu.clone());
        } else if name == "ffi_supra.cu" || name == "eval_check.cu" {
            supra_amalg.push_str(&line);
        } else {
            risc0_amalg.push_str(&line);
        }
    }

    let risc0_amalg_path = out_dir.join("risc0_kernels_rocm.cu");
    std::fs::write(&risc0_amalg_path, &risc0_amalg).unwrap();
    let supra_amalg_path = out_dir.join("supra_kernels_rocm.cu");
    std::fs::write(&supra_amalg_path, &supra_amalg).unwrap();

    let include_cuda2hip = format!("{sppark_root}/util/cuda2hip.hpp");
    let kernel_dir_str = kernel_dir.to_str().unwrap();
    let arches = risc0_build_kernel::hip_arches();
    let base_flags: Vec<&str> = vec![
        "-x", "hip",
        "-std=c++17", "-O3", "-fPIC",
        "-Wno-unused-function", "-Wno-unused-parameter", "-Wno-missing-braces",
        "-mllvm", "-amdgpu-early-inline-all=false",
        "-DFEATURE_BABY_BEAR",
        "-include", &include_cuda2hip,
        "-I", &cuda_root,
        "-I", &cxx_root,
        "-I", &sppark_root,
        "-I", kernel_dir_str,
    ];

    let mut obj_files = Vec::new();
    for (amalg_path, obj_name) in [
        (&risc0_amalg_path, "risc0_kernels_rocm.o"),
        (&supra_amalg_path, "supra_kernels_rocm.o"),
    ] {
        let obj = out_dir.join(obj_name);
        risc0_build_kernel::hip_compile(&hipcc, &base_flags, amalg_path, &obj, &arches, None);
        obj_files.push(obj);
    }

    // Compile separate files (e.g. sppark.cu) individually — these contain host-only
    // code (gpu_t.cuh) that can't be in the same TU as device kernels.
    for cu in &separate_files {
        let name = cu.file_name().unwrap().to_str().unwrap();
        let obj_name = format!("{}.o", name.trim_end_matches(".cu"));
        let obj = out_dir.join(&obj_name);
        let abs = std::fs::canonicalize(cu).unwrap();
        risc0_build_kernel::hip_compile(&hipcc, &base_flags, &abs, &obj, &arches, None);
        obj_files.push(obj);
    }

    // Device-link: with -fgpu-rdc, device code is compiled as bitcode in each .o file.
    // We must link all device code together into a single GPU code object so that
    // cross-TU device function calls resolve and __noinline__ functions stay separate.
    let device_linked = out_dir.join("device_linked_rocm.o");
    {
        let mut cmd = Command::new(&hipcc);
        cmd.arg("--offload-arch=native")
            .arg("-fgpu-rdc")
            .arg("-fPIC")
            .arg("-r")
            .arg("-nostdlib");
        for obj in &obj_files {
            cmd.arg(obj);
        }
        cmd.arg("-o").arg(&device_linked);
        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("failed to run hipcc device link: {e}"));
        assert!(status.success(), "hipcc device link failed");
    }

    // Archive: with -fgpu-rdc, the device-linked .o contains both the host stubs
    // and the fully linked GPU code, so we only archive it (not the individual .o files,
    // which would cause duplicate symbol errors).
    let lib_path = out_dir.join(format!("lib{output}.a"));
    let _ = std::fs::remove_file(&lib_path);
    let ar = risc0_build_kernel::find_ar_tool();
    let mut ar_cmd = Command::new(&ar);
    ar_cmd.arg("rcs").arg(&lib_path).arg(&device_linked);
    let status = ar_cmd.status().expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={output}");

    // Link against HIP runtime
    risc0_build_kernel::emit_rocm_lib_link();
}

#[allow(dead_code)]
fn build_intel_kernels() {
    rerun_if_changed("kernels/intel");
    rerun_if_changed("kernels/cxx");

    let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();

    // Find icpx compiler
    let icpx = env::var("RISC0_ICPX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let oneapi = PathBuf::from("/opt/intel/oneapi/compiler/latest/bin/icpx");
            if oneapi.exists() { oneapi } else { PathBuf::from("icpx") }
        });

    // Cache directory anchored on the <profile> dir under target/, regardless
    // of profile name (so custom profiles still cache). OUT_DIR layout is
    // `<target>/<profile>/build/<crate>-<hash>/out`.
    let cache_dir = out_dir
        .ancestors()
        .find(|p| {
            p.parent()
                .and_then(|q| q.file_name())
                .map(|n| n == std::ffi::OsStr::new("target"))
                .unwrap_or(false)
        })
        .map(|p| p.join("intel_recursion_cache"))
        .unwrap_or_else(|| out_dir.join("intel_recursion_cache"));
    std::fs::create_dir_all(&cache_dir).unwrap();

    let so_path = cache_dir.join("librisc0_recursion_intel.so");

    // Compute SHA-256 over the recursion eval_check inputs. Stamp invalidates
    // when any of poly_fp.cpp, eval_check.cpp, build.rs, or the icpx version
    // change.
    let icpx_version = stamp::icpx_version(&icpx);
    let recursion_hash = compute_recursion_intel_hash(&cxx_root, &icpx_version);

    // Check if we can skip rebuild: artifact present, stamp parses, hash
    // matches, status was "ok" last build.
    let stamp_path = cache_dir.join("intel_eval_check.stamp");
    let need_rebuild = stamp::need_rebuild(&so_path, &stamp_path, &recursion_hash);

    if need_rebuild {
        eprintln!("Building Intel SYCL eval_check kernel for recursion...");

        let mut cmd = Command::new(&icpx);
        cmd.arg("-shared")
            .arg("-fPIC")
            .arg("-fsycl")
            .arg("-std=c++17")
            .arg("-O1") // -O1 with -cl-opt-disable for ocloc
            .arg("-Xs").arg("-options -cl-opt-disable")
            .arg("-Wno-unused-parameter")
            .arg("-Wno-unused-function")
            .arg("-Wno-unused-variable")
            .arg("-Wno-sign-compare")
            .arg(format!("-I{cxx_root}"))
            .arg("-Ikernels/cxx");

        // Create an amalgamation file that includes poly_fp.cpp + kernel wrapper
        // in a single translation unit (required for SYCL device code).
        // The recursion poly_fp.cpp is a single monolithic function (~24K lines),
        // roughly half the size of rv32im's 52K. We mark the function itself as
        // noinline so it compiles as a separate device function.
        let amalg_path = out_dir.join("intel_recursion_eval_check_amalg.cpp");
        let mut amalg = String::new();
        amalg.push_str("// Auto-generated amalgamation for SYCL device code\n");
        amalg.push_str("#include \"fp.h\"\n");
        amalg.push_str("#include \"fpext.h\"\n");
        amalg.push_str("#include <cstdint>\n");
        amalg.push_str("namespace risc0::circuit::recursion {\n");
        amalg.push_str("constexpr size_t kInvRate = 4;\n");

        // Read poly_fp.cpp and extract the function body, skipping the preamble
        let src = std::fs::read_to_string("kernels/cxx/poly_fp.cpp").unwrap();
        if let Some(ns_start) = src.find("namespace risc0::circuit::recursion {") {
            let body_start = ns_start + "namespace risc0::circuit::recursion {".len();
            if let Some(body_end) = src.rfind('}') {
                let body = &src[body_start..body_end];
                // Inject __attribute__((noinline)) on the function definition.
                // The function definition starts with "FpExt poly_fp(" at line start.
                let mut modified = String::new();
                let mut injected = false;
                for line in body.lines() {
                    if !injected && line.starts_with("FpExt poly_fp(") && line.contains('{') {
                        // This is the function definition - inject noinline
                        modified.push_str("__attribute__((noinline)) ");
                        injected = true;
                    } else if !injected && line.starts_with("FpExt poly_fp(") {
                        // Forward declaration — skip, we already have it in header
                        // Actually keep it but don't inject noinline on declarations
                    }
                    modified.push_str(line);
                    modified.push('\n');
                }
                amalg.push_str(&modified);
            }
        }
        amalg.push_str("} // namespace risc0::circuit::recursion\n");
        // Append the kernel wrapper
        amalg.push_str(&std::fs::read_to_string("kernels/intel/eval_check.cpp").unwrap());
        std::fs::write(&amalg_path, &amalg).unwrap();

        cmd.arg(&amalg_path)
            .arg("-o")
            .arg(&so_path)
            .arg("-fsycl-targets=intel_gpu_bmg_g31"); // AOT for BMG

        eprintln!("  Running: {:?}", cmd);
        let output = cmd.output().expect("Failed to run icpx");
        if !output.status.success() {
            stamp::write(&stamp_path, &recursion_hash, &icpx_version, "failed");
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("Intel recursion eval_check compilation failed:\n{}", stderr);
        }
        stamp::write(&stamp_path, &recursion_hash, &icpx_version, "ok");
        eprintln!("  Built {}", so_path.display());
    } else {
        eprintln!("Using cached Intel recursion eval_check kernel");
    }

    // Link
    println!("cargo:rustc-link-search=native={}", cache_dir.display());
    println!("cargo:rustc-link-lib=dylib=risc0_recursion_intel");

    // RPATH for runtime
    let intel_lib = PathBuf::from("/opt/intel/oneapi/compiler/latest/lib");
    if intel_lib.exists() {
        println!("cargo:rustc-link-search=native={}", intel_lib.display());
    }
}

fn rerun_if_changed<P: AsRef<Path>>(path: P) {
    println!("cargo:rerun-if-changed={}", path.as_ref().display());
}

fn glob_paths(pattern: &str) -> Vec<PathBuf> {
    glob::glob(pattern).unwrap().map(|x| x.unwrap()).collect()
}
