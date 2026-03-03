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
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use rayon::prelude::*;
use sha2::{Digest, Sha256};
use tempfile::tempdir_in;

const METAL_INCS: &[(&str, &str)] = &[
    ("fp.h", include_str!("../kernels/metal/fp.h")),
    ("fpext.h", include_str!("../kernels/metal/fpext.h")),
];

#[derive(Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum KernelType {
    Cpp,
    Cuda,
    Metal,
}

pub struct KernelBuild {
    kernel_type: KernelType,
    flags: Vec<String>,
    files: Vec<PathBuf>,
    inc_dirs: Vec<PathBuf>,
    deps: Vec<PathBuf>,
}

impl KernelBuild {
    pub fn new(kernel_type: KernelType) -> Self {
        Self {
            kernel_type,
            flags: Vec::new(),
            files: Vec::new(),
            inc_dirs: Vec::new(),
            deps: Vec::new(),
        }
    }

    /// Add a directory to the `-I` or include path for headers
    pub fn include<P: AsRef<Path>>(&mut self, dir: P) -> &mut KernelBuild {
        self.inc_dirs.push(dir.as_ref().to_path_buf());
        self
    }

    /// Add an arbitrary flag to the invocation of the compiler
    pub fn flag(&mut self, flag: &str) -> &mut KernelBuild {
        self.flags.push(flag.to_string());
        self
    }

    /// Add a file which will be compiled
    pub fn file<P: AsRef<Path>>(&mut self, p: P) -> &mut KernelBuild {
        self.files.push(p.as_ref().to_path_buf());
        self
    }

    /// Add files which will be compiled
    pub fn files<P>(&mut self, p: P) -> &mut KernelBuild
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        for file in p.into_iter() {
            self.file(file);
        }
        self
    }

    /// Add a file which will be compiled
    pub fn file_opt<P: AsRef<Path>>(&mut self, _p: P, _opt: usize) -> &mut KernelBuild {
        self
    }

    /// Add files which will be compiled
    pub fn files_opt<P>(&mut self, _p: P, _opt: usize) -> &mut KernelBuild
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        self
    }

    /// Add a dependency
    pub fn dep<P: AsRef<Path>>(&mut self, p: P) -> &mut KernelBuild {
        self.deps.push(p.as_ref().to_path_buf());
        self
    }

    /// Add dependencies
    pub fn deps<P>(&mut self, p: P) -> &mut KernelBuild
    where
        P: IntoIterator,
        P::Item: AsRef<Path>,
    {
        for file in p.into_iter() {
            self.dep(file);
        }
        self
    }

    pub fn compile(&mut self, output: &str) {
        println!("cargo:rerun-if-env-changed=RISC0_SKIP_BUILD_KERNELS");
        for src in self.files.iter() {
            rerun_if_changed(src);
        }
        for dep in self.deps.iter() {
            rerun_if_changed(dep);
        }
        match &self.kernel_type {
            KernelType::Cpp => self.compile_cpp(output),
            KernelType::Cuda => self.compile_cuda(output),
            KernelType::Metal => self.compile_metal(output),
        }
    }

    fn compile_cpp(&mut self, output: &str) {
        if env::var("RISC0_SKIP_BUILD_KERNELS").is_ok() {
            return;
        }

        // It's *highly* recommended to install `sccache` and use this combined with
        // `RUSTC_WRAPPER=/path/to/sccache` to speed up rebuilds of C++ kernels
        cc::Build::new()
            .cpp(true)
            .debug(false)
            .files(&self.files)
            .includes(&self.inc_dirs)
            .flag_if_supported("/std:c++17")
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-fno-var-tracking")
            .flag_if_supported("-fno-var-tracking-assignments")
            .flag_if_supported("-g0")
            .compile(output);
    }

    fn compile_cuda(&mut self, output: &str) {
        println!("cargo:rerun-if-env-changed=NVCC_APPEND_FLAGS");
        println!("cargo:rerun-if-env-changed=NVCC_PREPEND_FLAGS");
        println!("cargo:rerun-if-env-changed=RISC0_CUDART_LINKAGE");
        println!("cargo:rerun-if-env-changed=NVCC_CCBIN");

        for inc_dir in self.inc_dirs.iter() {
            rerun_if_changed(inc_dir);
        }

        if env::var("RISC0_SKIP_BUILD_KERNELS").is_ok() {
            let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
            let out_path = out_dir.join(format!("lib{output}-skip.a"));
            fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&out_path)
                .unwrap();
            println!("cargo:{}={}", output, out_path.display());
            return;
        }

        let mut build = cc::Build::new();

        for file in self.files.iter() {
            build.file(file);
        }

        for inc in self.inc_dirs.iter() {
            build.include(inc);
        }

        for flag in self.flags.iter() {
            build.flag(flag);
        }

        if env::var_os("NVCC_PREPEND_FLAGS").is_none() && env::var_os("NVCC_APPEND_FLAGS").is_none()
        {
            build.flag("-arch=native");
        }

        let cudart = env::var("RISC0_CUDART_LINKAGE").unwrap_or("static".to_string());

        build
            .cuda(true)
            .cudart(&cudart)
            .debug(false)
            .ccbin(env::var("NVCC_CCBIN").is_err())
            .flag("-diag-suppress=177")
            .flag("-diag-suppress=2922")
            .flag("-Xcudafe")
            .flag("--display_error_number")
            .flag("-Xcompiler")
            .flag("-Wno-missing-braces,-Wno-unused-function")
            .compile(output);
    }

    fn compile_metal(&mut self, output: &str) {
        let target = env::var("TARGET").unwrap();
        let sdk_name = if target.ends_with("ios") {
            "iphoneos"
        } else if target.ends_with("ios-sim") {
            "iphonesimulator"
        } else if target.ends_with("darwin") {
            "macosx"
        } else {
            panic!("unsupported target: {target}")
        };

        self.cached_compile(
            output,
            "metallib",
            METAL_INCS,
            &[],
            &[sdk_name.to_string()],
            |out_dir, out_path, sys_inc_dir, _flags| {
                let files: Vec<_> = self.files.iter().map(|x| x.as_path()).collect();

                let air_paths: Vec<_> = files
                    .into_par_iter()
                    .map(|src| {
                        let air_path = out_dir.join(src).with_extension("").with_extension("air");
                        if let Some(parent) = air_path.parent() {
                            fs::create_dir_all(parent).unwrap();
                        }
                        let mut cmd = Command::new("xcrun");
                        cmd.args(["--sdk", sdk_name]);
                        cmd.arg("metal");
                        cmd.arg("-o").arg(&air_path);
                        cmd.arg("-c").arg(src);
                        cmd.arg("-I").arg(sys_inc_dir);
                        cmd.arg("-Wno-unused-variable");
                        for inc_dir in self.inc_dirs.iter() {
                            cmd.arg("-I").arg(inc_dir);
                        }
                        println!("Running: {cmd:?}");
                        let status = cmd.status().unwrap();
                        if !status.success() {
                            panic!("Could not build metal kernels");
                        }
                        air_path
                    })
                    .collect();

                let result = Command::new("xcrun")
                    .args(["--sdk", sdk_name])
                    .arg("metallib")
                    .args(air_paths)
                    .arg("-o")
                    .arg(out_path)
                    .status()
                    .unwrap();
                if !result.success() {
                    panic!("Could not build metal kernels");
                }
            },
        );
    }

    fn cached_compile<F: Fn(&Path, &Path, &Path, &[String])>(
        &self,
        output: &str,
        extension: &str,
        assets: &[(&str, &str)],
        flags: &[String],
        tags: &[String],
        inner: F,
    ) {
        let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
        if env::var("RISC0_SKIP_BUILD_KERNELS").is_ok() {
            let out_path = out_dir
                .join("skip-".to_string() + output)
                .with_extension(extension);
            fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&out_path)
                .unwrap();
            println!("cargo:{}={}", output, out_path.display());
            return;
        }

        let out_path = out_dir.join(output).with_extension(extension);
        let sys_inc_dir = out_dir.join("_sys_");

        let cache_dir = risc0_cache();
        if !cache_dir.is_dir() {
            fs::create_dir_all(&cache_dir).unwrap();
        }

        let temp_dir = tempdir_in(&cache_dir).unwrap();
        let mut hasher = Hasher::new();
        for flag in flags {
            hasher.add_flag(flag);
        }
        for tag in tags {
            hasher.add_flag(tag);
        }
        for src in self.files.iter() {
            hasher.add_file(src);
        }
        for (name, contents) in assets {
            let path = sys_inc_dir.join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&path, contents).unwrap();
            hasher.add_file(path);
        }
        for dep in self.deps.iter() {
            hasher.add_file(dep);
        }
        let digest = hasher.finalize();
        let cache_path = cache_dir.join(digest).with_extension(extension);
        if !cache_path.is_file() {
            let tmp_dir = temp_dir.path();
            let tmp_path = tmp_dir.join(output).with_extension(extension);
            inner(tmp_dir, &tmp_path, &sys_inc_dir, flags);
            fs::rename(tmp_path, &cache_path).unwrap();
        }
        fs::copy(cache_path, &out_path).unwrap();

        println!("cargo:{}={}", output, out_path.display());
    }
}

fn risc0_cache() -> PathBuf {
    directories::ProjectDirs::from("com.risczero", "RISC Zero", "risc0")
        .unwrap()
        .cache_dir()
        .into()
}

struct Hasher {
    sha: Sha256,
}

impl Hasher {
    pub fn new() -> Self {
        Self { sha: Sha256::new() }
    }

    pub fn add_flag(&mut self, flag: &str) {
        self.sha.update(flag);
    }

    pub fn add_file<P: AsRef<Path>>(&mut self, path: P) {
        let bytes = fs::read(path).unwrap();
        self.sha.update(bytes);
    }

    pub fn finalize(self) -> String {
        hex::encode(self.sha.finalize())
    }
}

fn rerun_if_changed<P: AsRef<Path>>(path: P) {
    println!("cargo:rerun-if-changed={}", path.as_ref().display());
}

// ---------------------------------------------------------------------------
// HIP multi-arch parallel compilation
// ---------------------------------------------------------------------------

/// Resolve `RISC0_HIP_ARCH` from the environment, defaulting to `"native"`.
pub fn hip_arches() -> String {
    env::var("RISC0_HIP_ARCH").unwrap_or_else(|_| "native".to_string())
}

/// Compile a HIP source file to an object, potentially for multiple GPU
/// architectures in parallel.
///
/// * `hipcc`    — path to the hipcc compiler.
/// * `flags`    — all compiler flags **except** `--offload-arch`, `-c`, and `-o`.
/// * `source`   — the `.cu` / `.hip` source file.
/// * `output`   — destination object file.
/// * `arches`   — comma-separated arch list (e.g. `"gfx1100,gfx1201"`) or `"native"`.
/// * `work_dir` — optional working directory for hipcc.
///
/// When `arches` contains multiple targets, device code is compiled in parallel
/// (one hipcc process per arch) and merged with `clang-offload-bundler`.
/// Falls back to a single sequential hipcc invocation if the parallel pipeline
/// encounters an error.
pub fn hip_compile(
    hipcc: &str,
    flags: &[&str],
    source: &Path,
    output: &Path,
    arches: &str,
    work_dir: Option<&Path>,
) {
    let arch_list: Vec<&str> = arches.split(',').map(|s| s.trim()).collect();

    if arch_list.len() <= 1 || arches == "native" {
        hip_compile_single(hipcc, flags, source, output, arches, work_dir);
        return;
    }

    // Try parallel compilation; fall back to sequential on failure.
    if let Err(e) = hip_compile_parallel(hipcc, flags, source, output, &arch_list, work_dir) {
        eprintln!(
            "hip_compile: parallel build failed ({e:#}), falling back to sequential for {}",
            source.display()
        );
        let joined = arch_list.join(",");
        hip_compile_single(hipcc, flags, source, output, &joined, work_dir);
    }
}

/// Single-invocation hipcc compilation (original behaviour).
fn hip_compile_single(
    hipcc: &str,
    flags: &[&str],
    source: &Path,
    output: &Path,
    arches: &str,
    work_dir: Option<&Path>,
) {
    let mut cmd = Command::new(hipcc);
    if let Some(dir) = work_dir {
        cmd.current_dir(dir);
    }
    for flag in flags {
        cmd.arg(flag);
    }
    cmd.arg(format!("--offload-arch={arches}"));
    cmd.arg("-c").arg(source).arg("-o").arg(output);

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("hipcc failed to start: {e}"));
    assert!(
        status.success(),
        "hipcc failed for {}",
        source.display()
    );
}

/// Parallel multi-arch compilation:
///   1. Host code compiled once (`--offload-host-only`).
///   2. Device code compiled per-arch in parallel (`--offload-device-only`).
///   3. Results merged with `clang-offload-bundler`.
fn hip_compile_parallel(
    hipcc: &str,
    flags: &[&str],
    source: &Path,
    output: &Path,
    arch_list: &[&str],
    work_dir: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = output
        .parent()
        .ok_or("output path has no parent directory")?;
    let stem = output
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("hip_kernel");

    eprintln!(
        "hip_compile: parallel build for {} ({} arches: {})",
        source.display(),
        arch_list.len(),
        arch_list.join(", "),
    );

    // Collect flags into owned Strings so we can share across threads.
    let flags_owned: Vec<String> = flags.iter().map(|s| s.to_string()).collect();
    let hipcc_owned = hipcc.to_string();
    let source_owned = source.to_path_buf();
    let work_dir_owned = work_dir.map(|d| d.to_path_buf());

    // 1. Host-only compilation (fast — no device codegen).
    let host_obj = out_dir.join(format!("{stem}_host.o"));
    {
        let mut cmd = Command::new(hipcc);
        if let Some(dir) = work_dir {
            cmd.current_dir(dir);
        }
        for flag in flags {
            cmd.arg(flag);
        }
        cmd.arg("--offload-host-only");
        cmd.arg(format!("--offload-arch={}", arch_list[0]));
        cmd.arg("-c").arg(source).arg("-o").arg(&host_obj);

        let status = cmd.status()?;
        if !status.success() {
            return Err(format!("hipcc --offload-host-only failed for {}", source.display()).into());
        }
    }

    // 2. Device-only compilation — one per arch, in parallel via rayon.
    let dev_results: Vec<Result<PathBuf, String>> = arch_list
        .par_iter()
        .map(|arch| {
            let dev_obj = out_dir.join(format!("{stem}_dev_{arch}.o"));
            let mut cmd = Command::new(&hipcc_owned);
            if let Some(ref dir) = work_dir_owned {
                cmd.current_dir(dir);
            }
            for flag in &flags_owned {
                cmd.arg(flag);
            }
            cmd.arg("--offload-device-only");
            cmd.arg(format!("--offload-arch={arch}"));
            cmd.arg("-c").arg(&source_owned).arg("-o").arg(&dev_obj);

            let status = cmd
                .status()
                .map_err(|e| format!("hipcc device-only ({arch}) failed to start: {e}"))?;
            if !status.success() {
                return Err(format!("hipcc --offload-device-only --offload-arch={arch} failed"));
            }
            Ok(dev_obj)
        })
        .collect();

    // Collect results, bail on any failure.
    let dev_objs: Vec<PathBuf> = dev_results
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    // 3. Bundle host + all device objects into one fat binary.
    let bundler = find_offload_bundler(hipcc);
    let mut cmd = Command::new(&bundler);
    cmd.arg("-type=o");

    // Host input first.
    cmd.arg(format!("-input={}", host_obj.display()));
    let mut targets = "host-x86_64-unknown-linux-gnu".to_string();

    // Device inputs.
    for (arch, dev_obj) in arch_list.iter().zip(&dev_objs) {
        cmd.arg(format!("-input={}", dev_obj.display()));
        targets.push_str(&format!(",hipv4-amdgcn-amd-amdhsa--{arch}"));
    }

    cmd.arg(format!("-output={}", output.display()));
    cmd.arg(format!("-targets={targets}"));

    eprintln!("hip_compile: bundling {} arches → {}", arch_list.len(), output.display());
    let status = cmd.status()?;
    if !status.success() {
        return Err("clang-offload-bundler failed".into());
    }

    // Cleanup intermediate files.
    let _ = fs::remove_file(&host_obj);
    for obj in &dev_objs {
        let _ = fs::remove_file(obj);
    }

    Ok(())
}

/// Locate `clang-offload-bundler`, searching ROCm paths and hipcc's neighbourhood.
fn find_offload_bundler(hipcc: &str) -> String {
    // 1. Explicit env var.
    if let Ok(path) = env::var("CLANG_OFFLOAD_BUNDLER") {
        return path;
    }

    // 2. Standard ROCm install.
    let rocm_path = "/opt/rocm/llvm/bin/clang-offload-bundler";
    if Path::new(rocm_path).is_file() {
        return rocm_path.to_string();
    }

    // 3. Adjacent to hipcc.
    if let Ok(resolved) = fs::canonicalize(hipcc) {
        if let Some(bin_dir) = resolved.parent() {
            // hipcc is usually in /opt/rocm/bin/ ; bundler in /opt/rocm/llvm/bin/
            let candidate = bin_dir.join("../llvm/bin/clang-offload-bundler");
            if let Ok(canon) = fs::canonicalize(&candidate) {
                if canon.is_file() {
                    return canon.to_string_lossy().to_string();
                }
            }
        }
    }

    // 4. PATH fallback.
    "clang-offload-bundler".to_string()
}
