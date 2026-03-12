// Copyright 2024 RISC Zero, Inc.
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
        .include(env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap())
        .compile("risc0_keccak_cpu");
}

fn build_cuda_kernels() {
    let output = "risc0_keccak_cuda";

    println!("cargo:rerun-if-env-changed=NVCC_APPEND_FLAGS");
    println!("cargo:rerun-if-env-changed=NVCC_PREPEND_FLAGS");
    println!("cargo:rerun-if-env-changed=SCCACHE_RECACHE");
    rerun_if_changed("kernels/cuda");

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

    env::set_var("SCCACHE_IDLE_TIMEOUT", "0");

    let cuda_root = env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap();
    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let kernel_dir = std::fs::canonicalize("kernels/cuda").unwrap();

    // Build 1: witgen/steps/ffi files with --device-c (need cross-TU device linking)
    let witgen_files: Vec<PathBuf> = glob_paths("kernels/cuda/*.cu")
        .into_iter()
        .filter(|p| {
            let name = p.file_name().unwrap().to_str().unwrap();
            !name.starts_with("eval_check") && name != "ffi_supra.cu"
        })
        .collect();

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
        .include(&cuda_root)
        .include(&sppark_root);
    if env::var_os("NVCC_PREPEND_FLAGS").is_none() && env::var_os("NVCC_APPEND_FLAGS").is_none() {
        build.flag("-arch=native");
    }
    build.files(witgen_files).compile("risc0_keccak_cuda_witgen");

    // Build 2: eval_check as a single TU WITHOUT --device-c (whole program mode).
    // This enables inlining of the deep poly_fp chain (keccak_0..keccak_47),
    // reducing per-thread stack from ~82KB to ~8KB and preventing CUDA OOM at po2=18.
    let eval_check_obj = out_dir.join("eval_check_combined.o");
    let mut nvcc = Command::new("nvcc");
    nvcc.arg("-std=c++17")
        .arg("-O3")
        .arg("-Xcompiler").arg("-fPIC")
        .arg("-diag-suppress=177")
        .arg("-diag-suppress=550")
        .arg("-diag-suppress=2922")
        .arg("-Xcompiler").arg("-Wno-unused-function,-Wno-unused-parameter")
        .arg(format!("-I{}", cuda_root))
        .arg(format!("-I{}", sppark_root))
        .arg(format!("-I{}", kernel_dir.display()));
    if env::var_os("NVCC_PREPEND_FLAGS").is_none() && env::var_os("NVCC_APPEND_FLAGS").is_none() {
        nvcc.arg("-arch=native");
    }
    nvcc.arg("-c")
        .arg("kernels/cuda/eval_check_combined.cu")
        .arg("-o").arg(&eval_check_obj);

    let status = nvcc.status().expect("failed to run nvcc");
    assert!(status.success(), "nvcc failed to compile eval_check_combined.cu");

    // Archive into static lib
    let lib_path = out_dir.join(format!("lib{output}.a"));
    let _ = std::fs::remove_file(&lib_path);
    let mut ar = Command::new("ar");
    ar.arg("rcs").arg(&lib_path).arg(&eval_check_obj);
    let status = ar.status().expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={output}");
}

fn build_rocm_kernels() {
    let output = "risc0_keccak_cuda";

    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=RISC0_HIP_ARCH");
    println!("cargo:rerun-if-env-changed=SCCACHE_RECACHE");
    rerun_if_changed("kernels/cuda");

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

    env::set_var("SCCACHE_IDLE_TIMEOUT", "0");
    env::set_var("HIP_PLATFORM", "amd");

    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    let cuda_root = env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap();
    let hipcc = risc0_build_kernel::find_hipcc();
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let kernel_dir = std::fs::canonicalize("kernels/cuda").unwrap();

    // Two-group single-TU approach:
    // Group A (risc0 types): ffi.cu + step_*.cu + empty.cu
    // Group B (sppark types): ffi_supra.cu + eval_check_*.cu
    // These can't share a TU because sppark's bb31_4_t conflicts with risc0's fpext.h.
    // Within each group, files share a TU so cross-TU __device__ calls resolve without -fgpu-rdc.
    let cu_files = glob_paths("kernels/cuda/*.cu");
    let mut risc0_amalg = String::new();
    let mut supra_amalg = String::new();
    for cu in &cu_files {
        let name = cu.file_name().unwrap().to_str().unwrap();
        let abs = std::fs::canonicalize(cu).unwrap();
        let line = format!("#include \"{}\"\n", abs.display());
        if name == "ffi_supra.cu" || name.starts_with("eval_check") {
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
        "-std=c++17", "-O2", "-fPIC",
        "-Wno-unused-function", "-Wno-unused-parameter", "-Wno-missing-braces",
        "-mllvm", "-amdgpu-early-inline-all=false",
        "-include", &include_cuda2hip,
        "-I", &cuda_root,
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

    // Archive
    let lib_path = out_dir.join(format!("lib{output}.a"));
    let _ = std::fs::remove_file(&lib_path);
    let ar = risc0_build_kernel::find_ar_tool();
    let mut ar_cmd = Command::new(&ar);
    ar_cmd.arg("rcs").arg(&lib_path);
    for obj in &obj_files {
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
