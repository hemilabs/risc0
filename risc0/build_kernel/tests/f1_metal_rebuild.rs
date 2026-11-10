// Copyright 2026 RISC Zero, Inc.
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

//! End to end through `KernelBuild::compile` with `KernelType::Metal` (macOS with the Metal
//! toolchain only): touching a header under an `-I` dir, or changing a flag, rebuilds the
//! metallib; `.air` files of a generated (absolute) source land under `<OUT_DIR>/<output>/`; a
//! directory dependency doesn't panic; the shared headers (threadgroup Fp/FpExt, cuda2metal.h,
//! sppark_compat.h) compile. One test, since it sets process environment variables.

#![cfg(target_os = "macos")]

use std::{env, fs, process::Command};

use risc0_build_kernel::{KernelBuild, KernelType};

fn metal_available() -> bool {
    env::var_os("METAL_TC").is_some()
        || Command::new("xcrun")
            .args(["--sdk", "macosx", "--find", "metal"])
            .output()
            .is_ok_and(|o| o.status.success())
}

fn contains(haystack: &[u8], needle: &str) -> bool {
    haystack
        .windows(needle.len())
        .any(|w| w == needle.as_bytes())
}

const KERNEL: &str = r#"
#include "sub/names.h"
#include "fpext.h"
#include "cuda2metal.h"
#include "sppark_compat.h"

kernel void KNAME(device Fp* out [[buffer(0)]], uint lid [[thread_index_in_threadgroup]]) {
  threadgroup Fp x[8];
  threadgroup FpExt y[4];
  x[lid & 7] = Fp(lid);
  y[lid & 3] = FpExt(Fp(lid));
  __syncthreads();
  out[lid] = x[(lid + 1) & 7] * y[0].elems[0];
}
"#;

const NAMES: &str = r#"
#ifdef F1_FLAG
#define KNAME f1_rebuild_flag_NONCE
#else
#define KNAME f1_rebuild_HEADER_NONCE
#endif
"#;

#[test]
fn metal_rebuilds_on_header_and_flag_changes() {
    if !metal_available() {
        eprintln!("SKIPPED: no Metal toolchain (source /Users/user/zkminer/env.sh)");
        return;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let nonce = format!("{}_{nanos}", std::process::id());
    let root = env::temp_dir().join(format!("f1-rebuild-{nonce}"));
    let out_dir = root.join("out");
    let inc = root.join("inc");
    let dep = root.join("depdir");
    fs::create_dir_all(out_dir.join("gen")).unwrap();
    fs::create_dir_all(inc.join("sub")).unwrap();
    fs::create_dir_all(dep.join("nested")).unwrap();
    fs::write(dep.join("nested/x.h"), "// dep\n").unwrap();
    // A generated source, by absolute path under OUT_DIR.
    let src = out_dir.join("gen/k.metal");
    fs::write(&src, KERNEL).unwrap();
    let names = inc.join("sub/names.h");
    let write_names = |header: &str| {
        fs::write(
            &names,
            NAMES.replace("NONCE", &nonce).replace("HEADER", header),
        )
        .unwrap()
    };

    env::set_var("OUT_DIR", &out_dir);
    env::set_var("TARGET", "aarch64-apple-darwin");
    env::remove_var("RISC0_SKIP_BUILD_KERNELS");

    let output = "f1_rebuild_kernels";
    let lib = out_dir.join(format!("{output}.metallib"));
    let compile = |flags: &[&str]| -> Vec<u8> {
        let mut b = KernelBuild::new(KernelType::Metal);
        b.file(&src).include(&inc).dep(&dep);
        for f in flags {
            b.flag(f);
        }
        b.compile(output);
        fs::read(&lib).unwrap()
    };

    write_names("a");
    let first = compile(&[]);
    assert!(contains(&first, &format!("f1_rebuild_a_{nonce}")));

    // The .air of the absolute generated source is under <OUT_DIR>/<output>/, not beside it.
    assert!(out_dir.join(output).join("gen/k.air").is_file());
    assert!(!out_dir.join("gen/k.air").exists());

    // Touch (change) the header under the -I dir: rebuilt.
    write_names("b");
    let second = compile(&[]);
    assert!(contains(&second, &format!("f1_rebuild_b_{nonce}")));
    assert!(!contains(&second, &format!("f1_rebuild_a_{nonce}")));

    // A flag reaches the compiler and changes the key: rebuilt.
    let third = compile(&["-DF1_FLAG=1"]);
    assert!(contains(&third, &format!("f1_rebuild_flag_{nonce}")));

    // Unchanged inputs: the cached metallib comes back unchanged.
    let fourth = compile(&[]);
    assert_eq!(second, fourth);

    let _ = fs::remove_dir_all(&root);
}
