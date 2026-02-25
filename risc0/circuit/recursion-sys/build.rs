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
};

use risc0_build_kernel::{KernelBuild, KernelType};

fn main() {
    build_cpu_kernels();

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda_kernels();
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

    // Build step/ffi kernels with -maxrregcount=96 for higher occupancy.
    // eval_check is excluded because capping its registers causes severe regression.
    let step_files: Vec<PathBuf> = glob_paths("kernels/cuda/*.cu")
        .into_iter()
        .filter(|p| {
            let name = p.file_name().unwrap().to_str().unwrap();
            !matches!(name, "eval_check.cu" | "ffi_supra.cu")
        })
        .collect();
    KernelBuild::new(KernelType::Cuda)
        .files(step_files)
        .deps(["kernels/cuda"])
        .flag("-DFEATURE_BABY_BEAR")
        .flag("-Xptxas")
        .flag("-O3")
        .flag("-maxrregcount=96")
        .include(&cuda_root)
        .include(&cxx_root)
        .include(&sppark_root)
        .compile("risc0_recursion_cuda");

    // Build eval_check + ffi_supra without -maxrregcount (eval_check needs REG:255).
    KernelBuild::new(KernelType::Cuda)
        .files(["kernels/cuda/eval_check.cu", "kernels/cuda/ffi_supra.cu"])
        .deps(["kernels/cuda"])
        .flag("-DFEATURE_BABY_BEAR")
        .flag("-Xptxas")
        .flag("-O3")
        .include(&cuda_root)
        .include(&cxx_root)
        .include(&sppark_root)
        .compile("risc0_recursion_cuda_eval_check");
}

fn rerun_if_changed<P: AsRef<Path>>(path: P) {
    println!("cargo:rerun-if-changed={}", path.as_ref().display());
}

fn glob_paths(pattern: &str) -> Vec<PathBuf> {
    glob::glob(pattern).unwrap().map(|x| x.unwrap()).collect()
}
