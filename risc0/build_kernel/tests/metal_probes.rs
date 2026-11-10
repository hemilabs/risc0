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

//! Compile-only checks for the Day-0 Metal probes in `kernels/metal/probes/` (plan item D0).
//!
//! Every `*.metal` probe carries a `// EXPECT: ACCEPT` or `// EXPECT: REJECT` line; this test
//! compiles each one with `xcrun metal -std=metal3.2 -O2` and asserts the expectation, so a
//! toolchain update that changes any recorded MSL fact fails here. Runtime facts (PSO limits,
//! dispatch behaviour, timings) are measured by the standalone runner in `probes/runner/` and
//! recorded in `probes/README.md`.
//!
//! Requires the Metal toolchain on PATH (`source /Users/user/zkminer/env.sh` on the dev Mac).

#![cfg(any(target_os = "macos", target_os = "ios"))]

use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};

/// Probes that must exist, with their expected compile outcome. Keeps the README table and the
/// directory in sync: removing or flipping a probe requires editing this list.
const EXPECTED: &[(&str, bool)] = &[
    ("assert_release", true),
    ("atomic_ulong", false),
    ("designated_init", true),
    ("dispatch", true),
    ("extern_def", true),
    ("extern_use", true),
    ("gpu_address", true),
    ("host_name", true),
    ("maxthreads_768", true),
    ("maxthreads_768_post", false),
    ("mul_bench", true),
    ("mulhi64", true),
    ("simd_shuffle", true),
    ("tg_dynamic", true),
    ("tg_static_32k", true),
    ("tg_static_36k", true),
    ("watchdog", true),
];

fn probes_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/metal/probes")
}

fn out_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("risc0-metal-probes-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn xcrun(args: &[&std::ffi::OsStr]) -> Output {
    Command::new("xcrun").args(args).output().unwrap_or_else(|e| {
        panic!("failed to run xcrun ({e}); source /Users/user/zkminer/env.sh to put the Metal toolchain on PATH")
    })
}

fn toolchain_present() {
    let o = xcrun(&["metal".as_ref(), "--version".as_ref()]);
    assert!(
        o.status.success(),
        "`xcrun metal` is not available: {}\nsource /Users/user/zkminer/env.sh to put the Metal toolchain on PATH",
        String::from_utf8_lossy(&o.stderr)
    );
}

fn compile(src: &Path, air: &Path) -> Output {
    xcrun(&[
        "metal".as_ref(),
        "-std=metal3.2".as_ref(),
        "-O2".as_ref(),
        "-c".as_ref(),
        src.as_os_str(),
        "-o".as_ref(),
        air.as_os_str(),
    ])
}

fn expectation(src: &Path) -> bool {
    let text = std::fs::read_to_string(src).unwrap();
    let line = text
        .lines()
        .find_map(|l| l.trim().strip_prefix("// EXPECT:"))
        .unwrap_or_else(|| panic!("{} has no `// EXPECT:` line", src.display()));
    match line.trim() {
        "ACCEPT" => true,
        "REJECT" => false,
        other => panic!("{}: bad expectation `{other}`", src.display()),
    }
}

#[test]
fn metal_probes_compile() {
    toolchain_present();
    let out = out_dir("compile");
    let mut found: Vec<String> = std::fs::read_dir(probes_dir())
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension()? == "metal")
                .then(|| p.file_stem().unwrap().to_string_lossy().into_owned())
        })
        .collect();
    found.sort();
    let listed: Vec<&str> = EXPECTED.iter().map(|(n, _)| *n).collect();
    assert_eq!(found, listed, "probe directory and EXPECTED list differ");

    let mut failures = vec![];
    for (stem, accept) in EXPECTED {
        let src = probes_dir().join(format!("{stem}.metal"));
        assert_eq!(
            expectation(&src),
            *accept,
            "{stem}: EXPECT line disagrees with EXPECTED"
        );
        let o = compile(&src, &out.join(format!("{stem}.air")));
        let accepted = o.status.success();
        println!("{stem}: {}", if accepted { "ACCEPT" } else { "REJECT" });
        if accepted != *accept {
            failures.push(format!(
                "{stem}: expected {}, got {}\n{}",
                if *accept { "ACCEPT" } else { "REJECT" },
                if accepted { "ACCEPT" } else { "REJECT" },
                String::from_utf8_lossy(&o.stderr)
            ));
        }
    }
    let _ = std::fs::remove_dir_all(&out);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn metal_probes_reject_diagnostics() {
    toolchain_present();
    let out = out_dir("diag");
    for (stem, needle) in [
        (
            "maxthreads_768_post",
            "'max_total_threads_per_threadgroup' attribute cannot be applied to types",
        ),
        (
            "atomic_ulong",
            "no matching function for call to 'atomic_fetch_add_explicit'",
        ),
    ] {
        let o = compile(
            &probes_dir().join(format!("{stem}.metal")),
            &out.join(format!("{stem}.air")),
        );
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(
            !o.status.success() && err.contains(needle),
            "{stem}: expected `{needle}`, got:\n{err}"
        );
    }
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn metal_probes_extern_link() {
    toolchain_present();
    let out = out_dir("extern");
    let use_air = out.join("extern_use.air");
    let def_air = out.join("extern_def.air");
    assert!(compile(&probes_dir().join("extern_use.metal"), &use_air)
        .status
        .success());
    assert!(compile(&probes_dir().join("extern_def.metal"), &def_air)
        .status
        .success());

    let both = out.join("both.metallib");
    let o = xcrun(&[
        "metallib".as_ref(),
        use_air.as_os_str(),
        def_air.as_os_str(),
        "-o".as_ref(),
        both.as_os_str(),
    ]);
    assert!(
        o.status.success(),
        "use+def link failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );

    let alone = out.join("alone.metallib");
    let o = xcrun(&[
        "metallib".as_ref(),
        use_air.as_os_str(),
        "-o".as_ref(),
        alone.as_os_str(),
    ]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        !o.status.success() && err.contains("Undefined symbol"),
        "linking extern_use.air alone should fail with an undefined symbol, got status {:?}: {err}",
        o.status
    );
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn metal_probes_assert_is_void() {
    toolchain_present();
    // Preprocess only: with just <metal_stdlib>, assert(...) must expand to ((void) 0).
    let src = probes_dir().join("assert_release.metal");
    let o = xcrun(&[
        "metal".as_ref(),
        "-std=metal3.2".as_ref(),
        "-E".as_ref(),
        src.as_os_str(),
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let pp = String::from_utf8_lossy(&o.stdout);
    let body = pp
        .lines()
        .skip_while(|l| !l.contains("kernel void assert_release"))
        .take(4)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        body.contains("((void) 0)"),
        "assert did not expand to ((void) 0):\n{body}"
    );
    assert!(
        !body.contains("always false"),
        "assert condition survived preprocessing:\n{body}"
    );
}
