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
        .include(env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap())
        .include(env::var("DEP_SPPARK_ROOT").unwrap());
    if env::var_os("NVCC_PREPEND_FLAGS").is_none() && env::var_os("NVCC_APPEND_FLAGS").is_none() {
        build.flag("-arch=native");
    }
    build.files(glob_paths("kernels/cuda/*.cu")).compile(output);
}

fn build_rocm_kernels() {
    let output = "risc0_keccak_cuda";

    println!("cargo:rerun-if-env-changed=HIPCC");
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
    let hipcc = env::var("HIPCC").unwrap_or_else(|_| "hipcc".to_string());
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

    let mut obj_files = Vec::new();
    // Compile risc0-types amalgamation
    for (amalg_path, obj_name) in [
        (&risc0_amalg_path, "risc0_kernels_rocm.o"),
        (&supra_amalg_path, "supra_kernels_rocm.o"),
    ] {
        let obj = out_dir.join(obj_name);
        let status = Command::new(&hipcc)
            .arg("-x").arg("hip")
            .arg("-std=c++17")
            .arg("-O1")
            .arg("-fPIC")
            .arg("-Wno-unused-function")
            .arg("-Wno-unused-parameter")
            .arg("-Wno-missing-braces")
            .arg("--offload-arch=native")
            .arg("-include").arg(format!("{sppark_root}/util/cuda2hip.hpp"))
            .arg("-I").arg(&cuda_root)
            .arg("-I").arg(&sppark_root)
            .arg("-I").arg(&kernel_dir)
            .arg("-c").arg(amalg_path)
            .arg("-o").arg(&obj)
            .status()
            .unwrap_or_else(|e| panic!("failed to run hipcc for {obj_name}: {e}"));
        assert!(status.success(), "hipcc failed for {obj_name}");
        obj_files.push(obj);
    }

    // Archive
    let lib_path = out_dir.join(format!("lib{output}.a"));
    let _ = std::fs::remove_file(&lib_path);
    let mut ar_cmd = Command::new("ar");
    ar_cmd.arg("rcs").arg(&lib_path);
    for obj in &obj_files {
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
