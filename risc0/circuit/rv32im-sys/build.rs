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
    collections::hash_map::DefaultHasher,
    env,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    process::Command,
};

use risc0_build_kernel::{KernelBuild, KernelType};

#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("Features 'cuda' and 'rocm' are mutually exclusive. Enable only one GPU backend.");

fn main() {
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda_kernels();
    }

    if env::var("CARGO_FEATURE_ROCM").is_ok() {
        build_rocm_kernels();
    }

    if env::var("CARGO_FEATURE_INTEL").is_ok() {
        build_intel_kernels();
    }

    build_cpu_kernels();
}

fn build_cpu_kernels() {
    rerun_if_changed("kernels/cxx");
    KernelBuild::new(KernelType::Cpp)
        .files(glob_paths("kernels/cxx/*.cpp"))
        .deps(glob_paths("kernels/cxx/*.h"))
        .deps(glob_paths("kernels/cxx/*.cpp.inc"))
        .deps(glob_paths("kernels/cxx/*.h.inc"))
        .include(env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap())
        .compile("risc0_rv32im_cpu");
}

fn build_cuda_kernels() {
    let output = "risc0_rv32im_cuda";

    println!("cargo:rerun-if-env-changed=NVCC_APPEND_FLAGS");
    println!("cargo:rerun-if-env-changed=NVCC_PREPEND_FLAGS");
    println!("cargo:rerun-if-env-changed=SCCACHE_RECACHE");
    rerun_if_changed("kernels/cuda");

    env::set_var("SCCACHE_IDLE_TIMEOUT", "0");

    if env::var("RISC0_SKIP_BUILD_KERNELS").is_ok() {
        let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
        let out_path = out_dir.join(format!("lib{output}-skip.a"));
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&out_path)
            .unwrap();
        println!("cargo:{}={}", output, out_path.display());
        return;
    }

    let cuda_root = env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap();
    let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    let use_native_arch =
        env::var_os("NVCC_PREPEND_FLAGS").is_none() && env::var_os("NVCC_APPEND_FLAGS").is_none();

    // Step 1: Compile eval_check_combined.cu WITHOUT -dc (standalone mode).
    // This allows NVCC to inline the 20 device functions in the poly_fp call
    // chain, eliminating cross-function ABI overhead (register save/restore,
    // parameter passing through local memory). With -dc, NVCC generates
    // relocatable code that cannot inline across function boundaries.
    //
    // CACHING: eval_check compilation takes ~24 minutes. We hash all eval_check
    // source files and skip recompilation when only other .cu files changed.
    // Use a stable cache dir (target/release/) instead of OUT_DIR which changes
    // with each cargo build hash.
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let cache_dir = out_dir
        .ancestors()
        .find(|p| p.ends_with("release") || p.ends_with("debug"))
        .map(|p| p.join("eval_check_cache"))
        .unwrap_or_else(|| out_dir.join("eval_check_cache"));
    std::fs::create_dir_all(&cache_dir).unwrap();
    let eval_check_cached = cache_dir.join("eval_check_combined_standalone.o");
    let eval_check_stamp = cache_dir.join("eval_check_hash.stamp");
    let eval_check_obj = out_dir.join("eval_check_combined_standalone.o");
    let current_hash = eval_check_source_hash();
    let cached_hash = std::fs::read_to_string(&eval_check_stamp).unwrap_or_default();
    let need_rebuild = current_hash != cached_hash || !eval_check_cached.exists();
    if need_rebuild {
        eprintln!("eval_check: source changed (or first build), compiling standalone...");
        let mut cmd = Command::new("nvcc");
        cmd.current_dir("kernels/cuda")
            .arg("-ccbin=c++")
            .arg("-std=c++17")
            .arg("-Xcompiler")
            .arg("-O3,-ffunction-sections,-fdata-sections,-fPIC")
            .arg("-Xcompiler")
            .arg("-Wno-unused-function,-Wno-unused-parameter")
            .arg("-m64")
            .arg("-Xptxas")
            .arg("-O3")
            .arg("-diag-suppress=177")
            .arg("-diag-suppress=550")
            .arg("-diag-suppress=2922")
            .arg("-I")
            .arg(&cuda_root)
            .arg("-I")
            .arg(&cxx_root)
            .arg("-I")
            .arg(&sppark_root);
        if use_native_arch {
            cmd.arg("-arch=native");
        }
        cmd.arg("-c") // compile only, NO --device-c
            .arg("eval_check_combined.cu")
            .arg("-o")
            .arg(&eval_check_cached);
        let status = cmd.status().expect("failed to run nvcc for eval_check_combined.cu");
        assert!(
            status.success(),
            "nvcc failed for eval_check_combined.cu (standalone mode)"
        );
        std::fs::write(&eval_check_stamp, &current_hash).unwrap();
    } else {
        eprintln!("eval_check: source unchanged, reusing cached object");
    }
    // Copy cached object to OUT_DIR for this build.
    std::fs::copy(&eval_check_cached, &eval_check_obj).unwrap();

    // Step 2: Compile remaining .cu files with -dc (separate compilation) via cc crate.
    // Exclude eval_check files (compiled standalone above).
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .cudart("static")
        .debug(false)
        .flag("-diag-suppress=177")
        .flag("-diag-suppress=550")
        .flag("-diag-suppress=2922")
        .flag("-std=c++17")
        .flag("-Xcompiler")
        .flag("-Wno-unused-function,-Wno-unused-parameter")
        .flag("-Xcompiler")
        .flag("-O3")
        .flag("-Xptxas")
        .flag("-O3")
        .include(&cuda_root)
        .include(&cxx_root)
        .include(&sppark_root);
    if use_native_arch {
        build.flag("-arch=native");
    }
    let cuda_files: Vec<PathBuf> = glob_paths("kernels/cuda/*.cu")
        .into_iter()
        .filter(|p| {
            let name = p.file_name().unwrap().to_str().unwrap();
            !matches!(
                name,
                "eval_check_0.cu"
                    | "eval_check_1.cu"
                    | "eval_check_2.cu"
                    | "eval_check_3.cu"
                    | "eval_check_combined.cu"
                    | "eval_check_kernel.cu"  // RDC-only, not used by NVIDIA
                    | "witgen_combined.cu"
            )
        })
        .collect();
    build.files(cuda_files).compile(output);

    // Step 3: Add standalone objects to the archive.
    let archive = out_dir.join(format!("lib{output}.a"));
    let ar = risc0_build_kernel::find_ar_tool();
    let status = Command::new(&ar)
        .arg("rcs")
        .arg(&archive)
        .arg(&eval_check_obj)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed to add standalone objects");
}

fn build_rocm_kernels() {
    let output = "risc0_rv32im_cuda";

    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=RISC0_HIP_ARCH");
    println!("cargo:rerun-if-env-changed=SCCACHE_RECACHE");
    rerun_if_changed("kernels/cuda");

    env::set_var("SCCACHE_IDLE_TIMEOUT", "0");
    env::set_var("HIP_PLATFORM", "amd");

    if env::var("RISC0_SKIP_BUILD_KERNELS").is_ok() {
        let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
        let out_path = out_dir.join(format!("lib{output}-skip.a"));
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&out_path)
            .unwrap();
        println!("cargo:{}={}", output, out_path.display());
        return;
    }

    let cuda_root = env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap();
    let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    let hipcc = risc0_build_kernel::find_hipcc();

    // Step 1: Compile eval_check_combined.cu standalone with hipcc.
    // -mllvm -amdgpu-early-inline-all=false prevents OOM on this large kernel
    // (21 functions, 2154 column reads).
    //
    // CACHING: eval_check compilation is very slow. We hash all eval_check
    // source files and skip recompilation when only other .cu files changed.
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let kernel_dir = std::fs::canonicalize("kernels/cuda").unwrap();
    let cache_dir = out_dir
        .ancestors()
        .find(|p| p.ends_with("release") || p.ends_with("debug"))
        .map(|p| p.join("eval_check_cache_rocm"))
        .unwrap_or_else(|| out_dir.join("eval_check_cache_rocm"));
    std::fs::create_dir_all(&cache_dir).unwrap();
    let eval_check_cached = cache_dir.join("eval_check_combined_standalone.o");
    let eval_check_stamp = cache_dir.join("eval_check_hash_rocm.stamp");
    let eval_check_obj = out_dir.join("eval_check_combined_standalone.o");
    let current_hash = eval_check_source_hash();
    let cached_hash = std::fs::read_to_string(&eval_check_stamp).unwrap_or_default();
    let need_rebuild = current_hash != cached_hash || !eval_check_cached.exists();
    if need_rebuild {
        eprintln!("eval_check (rocm): source changed (or first build), compiling standalone...");
        let include_cuda2hip = format!("{}/util/cuda2hip.hpp", &sppark_root);
        let eval_check_flags: Vec<&str> = vec![
            "-x", "hip",
            "-std=c++17", "-O3", "-fPIC",
            "-Wno-unused-function", "-Wno-unused-parameter",
            "-mllvm", "-amdgpu-early-inline-all=false",
            "-mllvm", "-amdgpu-use-aa-in-codegen",
            "-mllvm", "-amdgpu-schedule-metric-bias=0",
            "-mllvm", "-amdgpu-internalize-symbols",
            "-mllvm", "-amdgpu-schedule-relaxed-occupancy",
            "-include", &include_cuda2hip,
            "-I", &cuda_root,
            "-I", &cxx_root,
            "-I", &sppark_root,
        ];
        risc0_build_kernel::hip_compile(
            &hipcc,
            &eval_check_flags,
            Path::new("eval_check_combined.cu"),
            &eval_check_cached,
            &risc0_build_kernel::hip_arches(),
            Some(Path::new("kernels/cuda")),
        );
        std::fs::write(&eval_check_stamp, &current_hash).unwrap();
    } else {
        eprintln!("eval_check (rocm): source unchanged, reusing cached object");
    }
    // Copy cached object to OUT_DIR for this build.
    std::fs::copy(&eval_check_cached, &eval_check_obj).unwrap();

    // Step 2: Single-TU amalgamation for remaining .cu files.
    // This avoids -fgpu-rdc and the problematic HIP device link step entirely.
    // Exclude eval_check files (compiled standalone above), witgen_combined.cu,
    // and ffi_supra.cu (uses sppark types that conflict with risc0's fpext.h).
    let cuda_files: Vec<PathBuf> = glob_paths("kernels/cuda/*.cu")
        .into_iter()
        .filter(|p| {
            let name = p.file_name().unwrap().to_str().unwrap();
            !matches!(
                name,
                "eval_check_0.cu"
                    | "eval_check_1.cu"
                    | "eval_check_2.cu"
                    | "eval_check_3.cu"
                    | "eval_check_combined.cu"
                    | "eval_check_kernel.cu"
                    | "witgen_combined.cu"
                    | "ffi_supra.cu"
            )
        })
        .collect();
    let separate_files: Vec<PathBuf> = glob_paths("kernels/cuda/ffi_supra.cu");

    let amalg_path = out_dir.join("remaining_kernels_rocm.cu");
    let mut amalg = String::new();
    for cu in &cuda_files {
        let abs = std::fs::canonicalize(cu).unwrap();
        amalg.push_str(&format!("#include \"{}\"\n", abs.display()));
    }
    std::fs::write(&amalg_path, &amalg).unwrap();

    // Cache remaining_kernels: hash source files, skip recompilation if unchanged
    let remaining_cached = cache_dir.join("remaining_kernels_rocm.o");
    let remaining_stamp = cache_dir.join("remaining_kernels_hash_rocm.stamp");
    let remaining_obj = out_dir.join("remaining_kernels_rocm.o");
    let remaining_hash = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        // Hash .cu files
        for cu in &cuda_files {
            let abs = std::fs::canonicalize(cu).unwrap();
            if let Ok(content) = std::fs::read_to_string(&abs) {
                content.hash(&mut hasher);
            }
        }
        // Also hash header files (.h, .cuh, .inc) since they affect compilation
        for pattern in &["kernels/cuda/*.h", "kernels/cuda/*.cuh", "kernels/cuda/*.inc"] {
            for hdr in glob_paths(pattern) {
                let abs = std::fs::canonicalize(&hdr).unwrap();
                if let Ok(content) = std::fs::read_to_string(&abs) {
                    content.hash(&mut hasher);
                }
            }
        }
        // Hash target architectures so cache invalidates when arch list changes
        risc0_build_kernel::hip_arches().hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    };
    let remaining_prev = std::fs::read_to_string(&remaining_stamp).unwrap_or_default();
    if remaining_hash != remaining_prev || !remaining_cached.exists() {
        eprintln!("remaining_kernels (rocm): source changed, compiling...");
        let include_cuda2hip = format!("{sppark_root}/util/cuda2hip.hpp");
        let kernel_dir_str = kernel_dir.to_str().unwrap();
        let remaining_flags: Vec<&str> = vec![
            "-x", "hip",
            "-std=c++17", "-O3", "-fPIC",
            "-Wno-unused-function", "-Wno-unused-parameter",
            "-mllvm", "-amdgpu-early-inline-all=false",
            "-include", &include_cuda2hip,
            "-I", &cuda_root,
            "-I", &cxx_root,
            "-I", &sppark_root,
            "-I", kernel_dir_str,
        ];
        risc0_build_kernel::hip_compile(
            &hipcc,
            &remaining_flags,
            &amalg_path,
            &remaining_cached,
            &risc0_build_kernel::hip_arches(),
            None,
        );
        std::fs::write(&remaining_stamp, &remaining_hash).unwrap();
    } else {
        eprintln!("remaining_kernels (rocm): source unchanged, reusing cached object");
    }
    std::fs::copy(&remaining_cached, &remaining_obj).unwrap();

    // Compile ffi_supra.cu separately (uses sppark types, can't be in same TU)
    // Also cached
    let mut all_objs = vec![eval_check_obj.clone(), remaining_obj];
    for cu in &separate_files {
        let stem = cu.file_stem().unwrap().to_str().unwrap();
        let obj = out_dir.join(format!("{stem}.o"));
        let cached_obj = cache_dir.join(format!("{stem}.o"));
        let cached_stamp = cache_dir.join(format!("{stem}_hash_rocm.stamp"));
        let src_hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            if let Ok(content) = std::fs::read_to_string(cu) {
                content.hash(&mut hasher);
            }
            // Hash target architectures so cache invalidates when arch list changes
            risc0_build_kernel::hip_arches().hash(&mut hasher);
            format!("{:016x}", hasher.finish())
        };
        let prev_hash = std::fs::read_to_string(&cached_stamp).unwrap_or_default();
        if src_hash != prev_hash || !cached_obj.exists() {
            eprintln!("{stem} (rocm): source changed, compiling...");
            let include_cuda2hip = format!("{sppark_root}/util/cuda2hip.hpp");
            let kernel_dir_str = kernel_dir.to_str().unwrap();
            let sep_flags: Vec<&str> = vec![
                "-x", "hip",
                "-std=c++17", "-O3", "-fPIC",
                "-Wno-unused-function", "-Wno-unused-parameter",
                "-mllvm", "-amdgpu-early-inline-all=false",
                "-include", &include_cuda2hip,
                "-I", &cuda_root,
                "-I", &cxx_root,
                "-I", &sppark_root,
                "-I", kernel_dir_str,
            ];
            risc0_build_kernel::hip_compile(
                &hipcc,
                &sep_flags,
                cu,
                &cached_obj,
                &risc0_build_kernel::hip_arches(),
                None,
            );
            std::fs::write(&cached_stamp, &src_hash).unwrap();
        } else {
            eprintln!("{stem} (rocm): source unchanged, reusing cached object");
        }
        std::fs::copy(&cached_obj, &obj).unwrap();
        all_objs.push(obj);
    }

    // Step 3: Archive all objects
    let archive = out_dir.join(format!("lib{output}.a"));
    let _ = std::fs::remove_file(&archive);
    let ar = risc0_build_kernel::find_ar_tool();
    let mut ar_cmd = Command::new(&ar);
    ar_cmd.arg("rcs").arg(&archive);
    for obj in &all_objs {
        ar_cmd.arg(obj);
    }
    let status = ar_cmd.status().expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={output}");

    // Link against HIP runtime
    risc0_build_kernel::emit_rocm_lib_link();
}

fn rerun_if_changed<P: AsRef<Path>>(path: P) {
    println!("cargo:rerun-if-changed={}", path.as_ref().display());
}

fn glob_paths(pattern: &str) -> Vec<PathBuf> {
    glob::glob(pattern).unwrap().map(|x| x.unwrap()).collect()
}

/// Hash the contents of all files relevant to eval_check compilation.
/// Returns a hex string that changes when any eval_check source changes.
fn eval_check_source_hash() -> String {
    let mut hasher = DefaultHasher::new();
    let files = [
        "kernels/cuda/eval_check_combined.cu",
        "kernels/cuda/eval_check_0.cu",
        "kernels/cuda/eval_check_1.cu",
        "kernels/cuda/eval_check_2.cu",
        "kernels/cuda/eval_check_3.cu",
        "kernels/cuda/eval_check.cuh",
    ];
    for f in &files {
        if let Ok(contents) = std::fs::read(f) {
            f.hash(&mut hasher);
            contents.hash(&mut hasher);
        }
    }
    // Also hash include dirs and arch flag so cache invalidates on toolchain change.
    if let Ok(v) = env::var("DEP_RISC0_SYS_CUDA_ROOT") {
        v.hash(&mut hasher);
    }
    if let Ok(v) = env::var("DEP_RISC0_SYS_CXX_ROOT") {
        v.hash(&mut hasher);
    }
    if let Ok(v) = env::var("DEP_SPPARK_ROOT") {
        v.hash(&mut hasher);
    }
    // Hash target architectures so cache invalidates when arch list changes
    risc0_build_kernel::hip_arches().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
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

    // Cache directory for the expensive eval_check compilation
    let cache_dir = out_dir
        .ancestors()
        .find(|p| p.ends_with("release") || p.ends_with("debug"))
        .map(|p| p.join("intel_rv32im_cache"))
        .unwrap_or_else(|| out_dir.join("intel_rv32im_cache"));
    std::fs::create_dir_all(&cache_dir).unwrap();

    let so_path = cache_dir.join("librisc0_rv32im_intel.so");

    // Check if we can skip rebuild
    let stamp_path = cache_dir.join("intel_eval_check.stamp");
    let need_rebuild = !so_path.exists() || {
        let stamp = std::fs::read_to_string(&stamp_path).unwrap_or_default();
        stamp.is_empty() // Always rebuild if no stamp (TODO: hash sources)
    };

    if need_rebuild {
        eprintln!("Building Intel SYCL eval_check kernel...");

        let mut cmd = Command::new(&icpx);
        cmd.arg("-shared")
            .arg("-fPIC")
            .arg("-fsycl")
            .arg("-std=c++17")
            .arg("-Os") // -Os + noinline + 256 GRF: best eval_check perf (18MB vs 30MB -O1, 2x fewer icache misses)
            // Tell ocloc to skip expensive optimization passes via -cl-opt-disable.
            // Without this, ocloc takes 3+ hours for the 52K-line kernel.
            // AOT compilation for BMG. With __noinline__ on the 20 sub-functions,
            // ocloc should handle this in reasonable time (est. 15-30 min).
            .arg("-Wno-unused-parameter")
            .arg("-Wno-unused-function")
            .arg("-Wno-unused-variable")
            .arg("-Wno-sign-compare")
            .arg(format!("-I{cxx_root}"))
            .arg("-Ikernels/cxx");

        // Create an amalgamation file that includes all poly_fp sources
        // in a single translation unit (required for SYCL device code).
        // We handle the kInvRate redefinition by including all files in
        // a namespace wrapper with the constant defined once.
        let amalg_path = out_dir.join("intel_eval_check_amalg.cpp");
        let mut amalg = String::new();
        amalg.push_str("// Auto-generated amalgamation for SYCL device code\n");
        amalg.push_str("#include \"fp.h\"\n");
        amalg.push_str("#include \"fpext.h\"\n");
        amalg.push_str("#include <cstdint>\n");
        amalg.push_str("namespace risc0::circuit::rv32im_v2 {\n");
        amalg.push_str("constexpr size_t kInvRate = 4;\n");
        // Include the function bodies but skip their preamble (includes + kInvRate).
        // CRITICAL: inject __attribute__((noinline)) before each rv32im_v2_* function
        // DEFINITION (not declarations). Without noinline, LLVM/IGC tries to inline
        // all 52K lines into one mega-function, causing ocloc to take 3+ hours.
        // Tested selective inlining (pairs) — SLOWER due to increased per-function
        // register pressure. All-noinline with 256 GRF is optimal.
        for i in 0..4 {
            let src = std::fs::read_to_string(
                format!("kernels/cxx/rust_poly_fp_{i}.cpp")
            ).unwrap();
            if let Some(ns_start) = src.find("namespace risc0::circuit::rv32im_v2 {") {
                let body_start = ns_start + "namespace risc0::circuit::rv32im_v2 {".len();
                if let Some(body_end) = src.rfind('}') {
                    let body = &src[body_start..body_end];
                    let mut modified = String::new();
                    for line in body.lines() {
                        if line.starts_with("FpExt rv32im_v2_") && line.contains('{') {
                            modified.push_str("__attribute__((noinline)) ");
                        }
                        modified.push_str(line);
                        modified.push('\n');
                    }
                    amalg.push_str(&modified);
                }
            }
        }
        amalg.push_str("} // namespace risc0::circuit::rv32im_v2\n");

        // 2-way multi-pass: run gen_multipass.py to add _pass1 variants and poly_fp_pass2
        let mono_path = out_dir.join("intel_eval_check_mono.cpp");
        std::fs::write(&mono_path, &amalg).unwrap();
        let multipass_script = PathBuf::from("kernels/intel/gen_multipass.py");
        if multipass_script.exists() {
            eprintln!("  Running gen_multipass.py for 2-way eval_check split...");
            let mp_output = Command::new("python3")
                .arg(&multipass_script)
                .arg(&mono_path)
                .arg(&amalg_path)
                .output()
                .expect("Failed to run gen_multipass.py");
            if !mp_output.status.success() {
                let stderr = String::from_utf8_lossy(&mp_output.stderr);
                eprintln!("  gen_multipass.py failed (using monolithic fallback):\n{}", stderr);
                // Fall back to monolithic
                let mut mono = std::fs::read_to_string(&mono_path).unwrap();
                mono.push_str(&std::fs::read_to_string("kernels/intel/eval_check.cpp").unwrap());
                std::fs::write(&amalg_path, &mono).unwrap();
            } else {
                let stderr = String::from_utf8_lossy(&mp_output.stderr);
                eprintln!("{}", stderr);
                // Append the kernel wrapper. Set RISC0_BUILD_MULTIPASS=1 to enable
                // the multi-pass kernel (currently has correctness bugs).
                let mut amalg_content = std::fs::read_to_string(&amalg_path).unwrap();
                if std::env::var("RISC0_BUILD_MULTIPASS").is_ok() {
                    amalg_content.push_str("\n#define MULTIPASS_ENABLED\n");
                    eprintln!("  MULTIPASS_ENABLED defined (RISC0_BUILD_MULTIPASS set)");
                }
                amalg_content.push_str(&std::fs::read_to_string("kernels/intel/eval_check.cpp").unwrap());
                std::fs::write(&amalg_path, &amalg_content).unwrap();
            }
        } else {
            eprintln!("  gen_multipass.py not found, using monolithic eval_check");
            amalg.push_str(&std::fs::read_to_string("kernels/intel/eval_check.cpp").unwrap());
            std::fs::write(&amalg_path, &amalg).unwrap();
        }

        cmd.arg(&amalg_path)
            .arg("-o")
            .arg(&so_path)
            .arg("-fsycl-targets=intel_gpu_bmg_g31") // AOT with noinline sub-functions
            // Force 256 GRF mode: doubles register file from 8KB to 16KB per thread,
            // dramatically reducing the 42KB spill overhead. Trades occupancy (8→4 threads/EU)
            // for fewer spills — net win since kernel is spill-bound, not compute-bound.
            .arg("-Xs").arg("-options -cl-intel-256-GRF-per-thread");

        eprintln!("  Running: {:?}", cmd);
        let output = cmd.output().expect("Failed to run icpx");
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("Intel eval_check compilation failed:\n{}", stderr);
        }
        std::fs::write(&stamp_path, "built").unwrap();
        eprintln!("  Built {}", so_path.display());
    } else {
        eprintln!("Using cached Intel eval_check kernel");
    }

    // Link eval_check
    println!("cargo:rustc-link-search=native={}", cache_dir.display());
    println!("cargo:rustc-link-lib=dylib=risc0_rv32im_intel");

    // ========================================================================
    // Build Intel SYCL witgen kernel (separate .so, ~34s compile)
    // ========================================================================
    let witgen_so = cache_dir.join("librisc0_rv32im_intel_witgen.so");
    let witgen_stamp = cache_dir.join("intel_witgen.stamp");
    let witgen_rebuild = !witgen_so.exists() || {
        let stamp = std::fs::read_to_string(&witgen_stamp).unwrap_or_default();
        stamp.is_empty()
    };

    if witgen_rebuild {
        eprintln!("Building Intel SYCL witgen kernel...");

        // Create witgen amalgamation: Intel headers + steps.cpp body + ffi_witgen.cpp
        let witgen_amalg_path = out_dir.join("intel_witgen_amalg.cpp");
        let mut witgen_amalg = String::new();
        witgen_amalg.push_str("// Auto-generated SYCL witgen amalgamation\n");
        witgen_amalg.push_str("#include \"steps.h\"\n");
        witgen_amalg.push_str("#include \"witgen.h\"\n");
        witgen_amalg.push_str("\n");
        witgen_amalg.push_str("namespace risc0::circuit::rv32im_v2::intel {\n");

        // Read steps.cpp, extract body, inject noinline
        let steps_src = std::fs::read_to_string("kernels/cxx/steps.cpp").unwrap();
        let ns_marker = "namespace risc0::circuit::rv32im_v2::cpu {";
        if let Some(ns_start) = steps_src.find(ns_marker) {
            let body_start = ns_start + ns_marker.len();
            if let Some(body_end) = steps_src.rfind('}') {
                let body = &steps_src[body_start..body_end];
                let mut noinline_count = 0;
                for line in body.lines() {
                    let s = line.trim_start();

                    // Inject noinline on function definitions
                    if !s.is_empty() && !s.starts_with("//") && !s.starts_with('#')
                        && !s.starts_with("namespace") && !s.starts_with("using")
                        && !s.starts_with('}') && !s.starts_with('{')
                        && !s.starts_with("if") && !s.starts_with("for")
                        && !s.starts_with("while") && !s.starts_with("switch")
                        && !s.starts_with("else") && !s.starts_with("return")
                        && !s.starts_with("auto") && !s.starts_with("Val ")
                        && !s.starts_with("ExtVal") && !s.starts_with("size_t")
                        && s.contains('(') && s.trim_end().ends_with('{')
                        && (s.contains("Struct ") || s.starts_with("void step_")
                            || s.starts_with("ComponentStruct "))
                    {
                        witgen_amalg.push_str("__attribute__((noinline)) ");
                        noinline_count += 1;
                    }
                    witgen_amalg.push_str(line);
                    witgen_amalg.push('\n');
                }
                eprintln!("  Injected noinline on {} functions", noinline_count);
            }
        }
        witgen_amalg.push_str("} // namespace risc0::circuit::rv32im_v2::intel\n\n");
        // Append the kernel wrapper + extern implementations
        witgen_amalg.push_str(
            &std::fs::read_to_string("kernels/intel/ffi_witgen.cpp").unwrap()
        );
        std::fs::write(&witgen_amalg_path, &witgen_amalg).unwrap();

        let mut cmd = Command::new(&icpx);
        cmd.arg("-shared")
            .arg("-fPIC")
            .arg("-fsycl")
            .arg("-std=c++17")
            .arg("-Os") // -Os: different optimization passes to avoid icpx -O1 accum miscompilation
            .arg("-Xs").arg("-options -cl-opt-disable")
            .arg("-Wno-unused-parameter")
            .arg("-Wno-unused-function")
            .arg("-Wno-unused-variable")
            .arg("-Wno-sign-compare")
            .arg("-Wno-unused-but-set-variable")
            .arg(format!("-Ikernels/intel"))
            .arg(format!("-I{cxx_root}"))
            .arg("-Ikernels/cxx")
            .arg(&witgen_amalg_path)
            .arg("-o")
            .arg(&witgen_so)
            .arg("-fsycl-targets=intel_gpu_bmg_g31");

        eprintln!("  Running: {:?}", cmd);
        let output = cmd.output().expect("Failed to run icpx for witgen");
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("Intel witgen compilation failed (non-fatal, using CPU fallback):\n{}", stderr);
            // Don't panic — witgen can fall back to CPU
        } else {
            std::fs::write(&witgen_stamp, "built").unwrap();
            eprintln!("  Built {}", witgen_so.display());
        }
    } else {
        eprintln!("Using cached Intel witgen kernel");
    }

    // Link witgen .so (if it exists)
    if witgen_so.exists() {
        println!("cargo:rustc-link-lib=dylib=risc0_rv32im_intel_witgen");
    }

    // RPATH for runtime
    let intel_lib = PathBuf::from("/opt/intel/oneapi/compiler/latest/lib");
    if intel_lib.exists() {
        println!("cargo:rustc-link-search=native={}", intel_lib.display());
    }
}
