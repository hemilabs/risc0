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

use std::{env, path::PathBuf, process::Command};

use risc0_build_kernel::{KernelBuild, KernelType};

fn main() {
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda_kernels();
    }

    if env::var("CARGO_FEATURE_ROCM").is_ok() {
        build_rocm_kernels();
    }
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
    let hipcc = risc0_build_kernel::find_hipcc();
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let kernel_dir = std::fs::canonicalize("kernels/cuda").unwrap();

    let include_cuda2hip = format!("{sppark_root}/util/cuda2hip.hpp");
    let blst_src = env::var("DEP_BLST_C_SRC").unwrap();
    let arches = risc0_build_kernel::hip_arches();
    let mut base_flags: Vec<&str> = vec![
        "-x", "hip",
        "-std=c++17", "-O3", "-fPIC",
        "-Wno-unused-function", "-Wno-unused-parameter", "-Wno-missing-braces",
        "-mllvm", "-amdgpu-early-inline-all=false",
        "-include", &include_cuda2hip,
        "-I", kernel_dir.to_str().unwrap(),
        "-I", &blst_src,
        "-I", &sppark_root,
        "-D__ADX__",
    ];

    if env::var("CARGO_FEATURE_SETUP").is_ok() {
        base_flags.push("-DSRS_READ_COEFFS");
    }

    let obj = out_dir.join("groth16_ffi.o");
    let ffi_cu = kernel_dir.join("ffi.cu");
    risc0_build_kernel::hip_compile(&hipcc, &base_flags, &ffi_cu, &obj, &arches, None);

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
        .arg("rcs").arg(&lib_path).arg(&obj)
        .status().expect("failed to run ar");
    assert!(status.success(), "ar failed");

    // Extract blst objects and merge them into our archive
    let blst_tmp = out_dir.join("blst_objs");
    let _ = std::fs::remove_dir_all(&blst_tmp);
    std::fs::create_dir_all(&blst_tmp).unwrap();
    let status = Command::new(&ar)
        .arg("x").arg(&blst_lib)
        .current_dir(&blst_tmp)
        .status().expect("failed to extract blst objects");
    assert!(status.success(), "ar extract failed");

    for entry in std::fs::read_dir(&blst_tmp).unwrap() {
        let obj_path = entry.unwrap().path();
        let status = Command::new(&ar)
            .arg("rcs").arg(&lib_path).arg(&obj_path)
            .status().expect("failed to add blst object to archive");
        assert!(status.success(), "ar append failed");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={output}");

    // Link HIP runtime
    risc0_build_kernel::emit_rocm_lib_link();
}
