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
                    | "witgen_combined.cu"
            )
        })
        .collect();
    build.files(cuda_files).compile(output);

    // Step 3: Add standalone objects to the archive.
    let archive = out_dir.join(format!("lib{output}.a"));
    let status = Command::new("ar")
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
    let hipcc = env::var("HIPCC").unwrap_or_else(|_| "hipcc".to_string());

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
    let mut ar_cmd = Command::new("ar");
    ar_cmd.arg("rcs").arg(&archive);
    for obj in &all_objs {
        ar_cmd.arg(obj);
    }
    let status = ar_cmd.status().expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={output}");

    // Link against HIP runtime
    if let Ok(hip_path) = env::var("HIP_PATH") {
        println!("cargo:rustc-link-search=native={}/lib", hip_path);
    } else if std::path::Path::new("/opt/rocm/lib").exists() {
        println!("cargo:rustc-link-search=native=/opt/rocm/lib");
    }
    println!("cargo:rustc-link-lib=amdhip64");
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
