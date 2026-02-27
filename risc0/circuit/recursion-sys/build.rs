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
    KernelBuild::new(KernelType::Cuda)
        .files(glob_paths("kernels/cuda/*.cu"))
        .deps(["kernels/cuda"])
        .flag("-DFEATURE_BABY_BEAR")
        .include(env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap())
        .include(env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap())
        .include(env::var("DEP_SPPARK_ROOT").unwrap())
        .compile("risc0_recursion_cuda");
}

fn build_rocm_kernels() {
    let output = "risc0_recursion_cuda";
    rerun_if_changed("kernels/cuda");

    env::set_var("HIP_PLATFORM", "amd");

    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    let cuda_root = env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap();
    let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
    let hipcc = env::var("HIPCC").unwrap_or_else(|_| "hipcc".to_string());
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

    let mut obj_files = Vec::new();
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
            .arg("-DFEATURE_BABY_BEAR")
            .arg("-include").arg(format!("{sppark_root}/util/cuda2hip.hpp"))
            .arg("-I").arg(&cuda_root)
            .arg("-I").arg(&cxx_root)
            .arg("-I").arg(&sppark_root)
            .arg("-I").arg(&kernel_dir)
            .arg("-c").arg(amalg_path)
            .arg("-o").arg(&obj)
            .status()
            .unwrap_or_else(|e| panic!("failed to run hipcc for {obj_name}: {e}"));
        assert!(status.success(), "hipcc failed for {obj_name}");
        obj_files.push(obj);
    }

    // Compile separate files (e.g. sppark.cu) individually — these contain host-only
    // code (gpu_t.cuh) that can't be in the same TU as device kernels.
    for cu in &separate_files {
        let name = cu.file_name().unwrap().to_str().unwrap();
        let obj_name = format!("{}.o", name.trim_end_matches(".cu"));
        let obj = out_dir.join(&obj_name);
        let abs = std::fs::canonicalize(cu).unwrap();
        let status = Command::new(&hipcc)
            .arg("-x").arg("hip")
            .arg("-std=c++17")
            .arg("-O1")
            .arg("-fPIC")
            .arg("-Wno-unused-function")
            .arg("-Wno-unused-parameter")
            .arg("-Wno-missing-braces")
            .arg("--offload-arch=native")
            .arg("-DFEATURE_BABY_BEAR")
            .arg("-include").arg(format!("{sppark_root}/util/cuda2hip.hpp"))
            .arg("-I").arg(&cuda_root)
            .arg("-I").arg(&cxx_root)
            .arg("-I").arg(&sppark_root)
            .arg("-I").arg(&kernel_dir)
            .arg("-c").arg(&abs)
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
