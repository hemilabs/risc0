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
    let cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let rocm = env::var("CARGO_FEATURE_ROCM").is_ok();

    // Set the GPU builds' environment before any thread starts: the CPU build below runs
    // concurrently and reads the environment.
    if (cuda || rocm) && env::var("RISC0_SKIP_BUILD_KERNELS").is_err() {
        env::set_var("SCCACHE_IDLE_TIMEOUT", "0");
        if rocm {
            env::set_var("HIP_PLATFORM", "amd");
        }
    }

    // The CPU kernels (rust_poly_fp_<n>.cpp, ~7 MB of generated C++ each, up to ~20 min of -O3
    // per file) and the GPU kernels share nothing, so build them concurrently instead of one
    // after the other: each takes about as long as the other.
    std::thread::scope(|scope| {
        scope.spawn(build_cpu_kernels);

        if cuda {
            build_cuda_kernels();
        }

        if rocm {
            build_rocm_kernels();
        }
    });
}

fn build_cpu_kernels() {
    rerun_if_changed("kernels/cxx");
    // rust_poly_fp_<n>.cpp, the generated constraint polynomial, are ~7 MB each and take far
    // longer to compile than everything else here: start them at once, each on its own thread.
    let (poly_fp, rest): (Vec<PathBuf>, Vec<PathBuf>) =
        glob_paths("kernels/cxx/*.cpp").into_iter().partition(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("rust_poly_fp_"))
        });
    KernelBuild::new(KernelType::Cpp)
        .files(rest)
        .files_on_own_threads(poly_fp)
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

    let cuda_root = env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap();
    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    rerun_if_include_roots_changed(&[&cuda_root, &sppark_root]);
    let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let kernel_dir = std::fs::canonicalize("kernels/cuda").unwrap();
    let msvc = risc0_build_kernel::is_msvc_target();
    let use_native_arch =
        env::var_os("NVCC_PREPEND_FLAGS").is_none() && env::var_os("NVCC_APPEND_FLAGS").is_none();

    // Step 1: eval_check as one translation unit compiled WITHOUT --device-c, so the generated
    // constraint chain inlines, and with its scratch array in global memory instead of on the
    // stack. See kernels/cuda/eval_check_scratch_kernel.cuh for why.
    let tu = out_dir.join("eval_check_scratch.cu");
    write_scratch_eval_check(&kernel_dir, &tu);
    let obj_ext = risc0_build_kernel::object_extension();
    let eval_check_obj = out_dir.join(format!("eval_check_scratch.{obj_ext}"));
    // A stable cache dir (target/<profile>/) rather than OUT_DIR, which changes with the crate's
    // fingerprint: compiling this translation unit takes minutes.
    let cache_dir = out_dir
        .ancestors()
        .find(|p| p.ends_with("release") || p.ends_with("debug"))
        .map(|p| p.join("keccak_eval_check_cache"))
        .unwrap_or_else(|| out_dir.join("keccak_eval_check_cache"));
    std::fs::create_dir_all(&cache_dir).unwrap();

    let mut cmd = risc0_build_kernel::nvcc_command();
    cmd.current_dir(&kernel_dir);
    if !msvc {
        cmd.arg("-ccbin=c++")
            .arg("-Xcompiler")
            .arg("-O3,-ffunction-sections,-fdata-sections,-fPIC")
            .arg("-Xcompiler")
            .arg("-Wno-unused-function,-Wno-unused-parameter");
    }
    cmd.arg(risc0_build_kernel::cuda_cxx_std_flag())
        .arg("-m64")
        .arg("-Xptxas")
        .arg("-O3")
        .arg("-diag-suppress=177")
        .arg("-diag-suppress=550")
        .arg("-diag-suppress=2922")
        .arg("-I")
        .arg(&kernel_dir)
        .arg("-I")
        .arg(&cuda_root)
        .arg("-I")
        .arg(&sppark_root);
    if use_native_arch {
        cmd.arg("-arch=native");
    }
    cmd.arg("-c") // compile only, NO --device-c
        .arg(&tu);

    // Key the cache on everything the compile can read (the generated translation unit, the
    // headers it includes, the include roots), the whole command and nvcc's environment, as
    // rv32im-sys does for its eval_check.
    let native_key = if use_native_arch {
        risc0_build_kernel::native_gpu_arch_key(false)
    } else {
        String::new()
    };
    let local_sources: Vec<PathBuf> = vec![
        tu.clone(),
        kernel_dir.join("eval_check_scratch_prelude.cuh"),
        kernel_dir.join("eval_check_scratch_kernel.cuh"),
    ];
    let local_sources: Vec<&Path> = local_sources.iter().map(PathBuf::as_path).collect();
    let sppark = Path::new(&sppark_root);
    let sources_key = risc0_build_kernel::kernel_sources_key(
        &[Path::new(&cuda_root), sppark],
        &local_sources,
        |path| {
            path.starts_with(sppark)
                || !matches!(
                    path.extension().and_then(|e| e.to_str()),
                    Some("c" | "cc" | "cpp" | "cu")
                )
        },
    );
    let key = risc0_build_kernel::hash_key(&[
        "keccak-cuda-eval-check-scratch-v1",
        &sources_key,
        &risc0_build_kernel::command_key(&cmd),
        &risc0_build_kernel::nvcc_environment_key(),
        &native_key,
    ]);
    // eval_check is a single ptxas run of ~25 min, the -dc compile of the step files below
    // about 8: run it on its own thread, concurrently with them, instead of before them.
    std::thread::scope(|scope| {
        scope.spawn(|| {
            risc0_build_kernel::cached_compile(
                &cache_dir,
                "eval_check_scratch",
                obj_ext,
                &key,
                &eval_check_obj,
                |out| {
                    let status = cmd
                        .arg("-o")
                        .arg(out)
                        .status()
                        .expect("failed to run nvcc for eval_check_scratch.cu");
                    assert!(status.success(), "nvcc failed for eval_check_scratch.cu");
                },
            );
        });

        // Step 2: everything else with --device-c, as before. eval_check_*.cu and ffi_supra.cu
        // are in the translation unit above (ROCm still builds ffi_supra.cu).
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .cudart("static")
            .debug(false)
            .flag("-diag-suppress=177")
            .flag("-diag-suppress=550")
            .flag("-diag-suppress=2922")
            .flag(risc0_build_kernel::cuda_cxx_std_flag())
            .include(&cuda_root)
            .include(&sppark_root);
        // GCC/Clang warning flags; cl.exe, nvcc's host compiler on MSVC, rejects them.
        if msvc {
            risc0_build_kernel::msvc_cuda_build_setup(&mut build);
        } else {
            build
                .flag("-Xcompiler")
                .flag("-Wno-unused-function,-Wno-unused-parameter");
        }
        if use_native_arch {
            build.flag("-arch=native");
        }
        let cuda_files: Vec<PathBuf> = glob_paths("kernels/cuda/*.cu")
            .into_iter()
            .filter(|p| {
                let name = p.file_name().unwrap().to_str().unwrap();
                !(name == "ffi_supra.cu" || name.starts_with("eval_check_"))
            })
            .collect();
        build.files(cuda_files).compile(output);
    });

    // Step 3: add the eval_check object to the archive.
    risc0_build_kernel::add_to_static_lib(&out_dir, output, &[eval_check_obj]);
}

/// Write the CUDA eval_check translation unit to `dest`: the generated constraint code
/// (eval_check.cuh, then eval_check_<n>.cu) between eval_check_scratch_prelude.cuh and
/// eval_check_scratch_kernel.cuh, with its scratch array moved to global memory. Every writable
/// `Fp* argN` parameter becomes a `ScratchRef argN`, and `poly_fp`'s `Fp xN[M];` becomes this
/// thread's slice of the launch's scratch buffer, `M` elements long (`kKeccakScratchElems`).
/// The header's `extern` declaration of `poly_mix` becomes its definition.
///
/// Panics if the generated code doesn't have that shape (one array, and no writable `Fp*` but
/// the parameters that carry it), rather than build something that means something else.
fn write_scratch_eval_check(kernel_dir: &Path, dest: &Path) {
    let mut bodies: Vec<(u32, PathBuf)> = glob_paths("kernels/cuda/eval_check_*.cu")
        .into_iter()
        .filter_map(|p| {
            let name = p.file_name()?.to_str()?;
            let n = name
                .strip_prefix("eval_check_")?
                .strip_suffix(".cu")?
                .parse()
                .ok()?;
            Some((n, p))
        })
        .collect();
    bodies.sort();
    assert!(!bodies.is_empty(), "no kernels/cuda/eval_check_<n>.cu");
    let mut files = vec![kernel_dir.join("eval_check.cuh")];
    files.extend(bodies.into_iter().map(|(_, p)| p));

    let mut tu = String::from(
        "// Generated by keccak-sys/build.rs (`write_scratch_eval_check`) from kernels/cuda/\n\
         // eval_check.cuh and eval_check_<n>.cu. Do not edit.\n\
         #include \"eval_check_scratch_prelude.cuh\"\n",
    );
    let mut array: Option<(String, u64)> = None;
    let mut params = 0;
    let mut poly_mix = false;
    for file in &files {
        let text = std::fs::read_to_string(file).unwrap();
        let name = file.display();
        assert!(
            !text.contains("Fp *"),
            "{name}: unexpected `Fp *` spelling in the generated eval_check code"
        );
        tu.push_str(&format!(
            "// --- {} ---\n",
            file.file_name().unwrap().to_string_lossy()
        ));
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed == "#pragma once" || trimmed == "#include \"eval_check.cuh\"" {
                continue;
            }
            // The header declares `poly_mix` for --device-c builds; a whole-program build
            // defines it here, before its first use, and not again.
            if trimmed == "extern __constant__ FpExt poly_mix[kNumPolyMixPows];" {
                assert!(!poly_mix, "{name}: poly_mix declared twice");
                poly_mix = true;
                tu.push_str("__constant__ FpExt poly_mix[kNumPolyMixPows];\n");
                continue;
            }
            // `Fp x13[8237];`: the scratch array.
            let array_decl = trimmed
                .strip_prefix("Fp x")
                .and_then(|d| d.strip_suffix("];"))
                .and_then(|d| d.split_once('['))
                .filter(|(var, len)| {
                    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
                    digits(var) && digits(len)
                });
            if let Some((var, len)) = array_decl {
                assert!(
                    array.is_none(),
                    "{name}: more than one Fp array in eval_check"
                );
                array = Some((format!("x{var}"), len.parse().expect("array length")));
                tu.push_str(&format!(
                    "  ScratchRef x{var} = scratch_for_this_thread();\n"
                ));
                continue;
            }
            let mut out = String::with_capacity(line.len());
            let mut rest = line;
            while let Some(i) = rest.find("Fp*") {
                let (before, after) = rest.split_at(i);
                out.push_str(before);
                let ident_before = before
                    .chars()
                    .last()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
                if ident_before || out.ends_with("const ") {
                    out.push_str("Fp*");
                } else {
                    assert!(
                        after.starts_with("Fp* arg"),
                        "{name}: writable `Fp*` that isn't a scratch parameter: {line}"
                    );
                    out.push_str("ScratchRef");
                    params += 1;
                }
                rest = &after["Fp*".len()..];
            }
            out.push_str(rest);
            tu.push_str(&out);
            tu.push('\n');
        }
    }
    let (_, len) = array.expect("no Fp array in the generated eval_check code");
    assert!(
        params > 0,
        "no scratch parameters in the generated eval_check code"
    );
    assert!(
        poly_mix,
        "no poly_mix declaration in the generated eval_check code"
    );
    tu.push_str(&format!(
        "namespace risc0::circuit::keccak::cuda {{\n\
         constexpr uint32_t kKeccakScratchElems = {len};\n\
         }}\n\
         #include \"eval_check_scratch_kernel.cuh\"\n"
    ));
    if std::fs::read_to_string(dest).ok().as_deref() != Some(tu.as_str()) {
        std::fs::write(dest, tu).unwrap();
    }
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
        "-x",
        "hip",
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-Wno-unused-function",
        "-Wno-unused-parameter",
        "-Wno-missing-braces",
        "-mllvm",
        "-amdgpu-early-inline-all=false",
        "-include",
        &include_cuda2hip,
        "-I",
        &cuda_root,
        "-I",
        &sppark_root,
        "-I",
        kernel_dir_str,
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

/// The include roots the CUDA build compiles against: re-run this script when a file in them
/// changes (as a path dependency), so the cache key is recomputed. File by file, as a directory
/// would make cargo recurse into sppark's `rust/sppark` symlink loop.
fn rerun_if_include_roots_changed(roots: &[&str]) {
    for root in roots {
        for file in risc0_build_kernel::kernel_source_files(Path::new(root)) {
            rerun_if_changed(file);
        }
    }
}
