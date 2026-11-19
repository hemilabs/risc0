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

//! The Metal cache key (`KernelBuild::metal_cache_key`), flag defaults and `.air` placement.
//! Pure Rust: no Metal toolchain needed.

// W10-GATE: Metal-only (upstream release-3.1 has no such test; other builds do not run it).
#![cfg(any(target_os = "macos", target_os = "ios"))]

use std::{
    fs,
    path::{Path, PathBuf},
};

use risc0_build_kernel::{metal_air_rel_path, KernelBuild, KernelType, METAL_INCS};

/// A fresh directory under the system temp dir, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("f1-{tag}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write(path: &Path, contents: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}

/// A tree: `src/k.metal`, include dir `inc` with a header nested in `inc/sub`, and a dependency
/// directory `dep` with a nested file.
fn tree(tmp: &TempDir) -> (PathBuf, PathBuf, PathBuf) {
    let src = tmp.0.join("src/k.metal");
    let inc = tmp.0.join("inc");
    let dep = tmp.0.join("dep");
    write(&src, "#include \"sub/h.h\"\nkernel void k() {}\n");
    write(&inc.join("sub/h.h"), "#define X 1\n");
    write(&inc.join("top.h"), "#define Y 1\n");
    write(&dep.join("nested/d.h"), "#define Z 1\n");
    (src, inc, dep)
}

fn build(src: &Path, inc: &Path, dep: &Path, flags: &[&str]) -> KernelBuild {
    let mut b = KernelBuild::new(KernelType::Metal);
    b.file(src).include(inc).dep(dep);
    for f in flags {
        b.flag(f);
    }
    b
}

const TC: &str = "Apple metal version test";

#[test]
fn key_is_stable() {
    let tmp = TempDir::new("stable");
    let (src, inc, dep) = tree(&tmp);
    let a = build(&src, &inc, &dep, &[]).metal_cache_key("macosx", TC);
    let b = build(&src, &inc, &dep, &[]).metal_cache_key("macosx", TC);
    assert_eq!(a, b);
    assert_eq!(a.len(), 64);
}

#[test]
fn key_changes_with_flags() {
    let tmp = TempDir::new("flags");
    let (src, inc, dep) = tree(&tmp);
    let base = build(&src, &inc, &dep, &[]).metal_cache_key("macosx", TC);
    let with_def = build(&src, &inc, &dep, &["-DFOO=1"]).metal_cache_key("macosx", TC);
    let other_def = build(&src, &inc, &dep, &["-DFOO=2"]).metal_cache_key("macosx", TC);
    let o0 = build(&src, &inc, &dep, &["-O0"]).metal_cache_key("macosx", TC);
    let std4 = build(&src, &inc, &dep, &["-std=metal4.0"]).metal_cache_key("macosx", TC);
    let keys = [&base, &with_def, &other_def, &o0, &std4];
    for (i, a) in keys.iter().enumerate() {
        for b in &keys[i + 1..] {
            assert_ne!(a, b);
        }
    }
    // The defaults are part of the effective flags: spelling them out is the same build.
    let explicit = build(&src, &inc, &dep, &["-std=metal3.2", "-O2"]).metal_cache_key("macosx", TC);
    assert_eq!(base, explicit);
}

#[test]
fn key_changes_with_sdk_and_toolchain() {
    let tmp = TempDir::new("tc");
    let (src, inc, dep) = tree(&tmp);
    let b = build(&src, &inc, &dep, &[]);
    let base = b.metal_cache_key("macosx", TC);
    assert_ne!(base, b.metal_cache_key("iphoneos", TC));
    assert_ne!(
        base,
        b.metal_cache_key("macosx", "Apple metal version other")
    );
}

#[test]
fn key_changes_when_a_nested_include_header_changes() {
    let tmp = TempDir::new("inc");
    let (src, inc, dep) = tree(&tmp);
    let before = build(&src, &inc, &dep, &[]).metal_cache_key("macosx", TC);
    write(&inc.join("sub/h.h"), "#define X 2\n");
    let after = build(&src, &inc, &dep, &[]).metal_cache_key("macosx", TC);
    assert_ne!(
        before, after,
        "a header under an -I dir (nested) must change the key"
    );
    // A new header in an include dir changes it too (it can shadow another).
    write(&inc.join("sub/new.h"), "\n");
    let added = build(&src, &inc, &dep, &[]).metal_cache_key("macosx", TC);
    assert_ne!(after, added);
}

#[test]
fn key_changes_with_sources_and_directory_deps() {
    let tmp = TempDir::new("deps");
    let (src, inc, dep) = tree(&tmp);
    let base = build(&src, &inc, &dep, &[]).metal_cache_key("macosx", TC);
    write(&dep.join("nested/d.h"), "#define Z 2\n");
    let dep_changed = build(&src, &inc, &dep, &[]).metal_cache_key("macosx", TC);
    assert_ne!(
        base, dep_changed,
        "a file under a directory dep must change the key"
    );
    write(&src, "#include \"sub/h.h\"\nkernel void k2() {}\n");
    let src_changed = build(&src, &inc, &dep, &[]).metal_cache_key("macosx", TC);
    assert_ne!(dep_changed, src_changed);
}

#[test]
fn input_files_cover_include_dirs_and_dep_dirs() {
    let tmp = TempDir::new("inputs");
    let (src, inc, dep) = tree(&tmp);
    let files = build(&src, &inc, &dep, &[]).metal_input_files();
    for f in [
        src.clone(),
        inc.join("sub/h.h"),
        inc.join("top.h"),
        dep.join("nested/d.h"),
    ] {
        assert!(files.contains(&f), "{} missing from {files:?}", f.display());
    }
    assert!(files.iter().all(|f| f.is_file()));
}

#[test]
fn metal_flags_defaults() {
    let b = KernelBuild::new(KernelType::Metal);
    assert_eq!(b.metal_flags(), ["-std=metal3.2", "-O2"]);

    let mut b = KernelBuild::new(KernelType::Metal);
    b.flag("-DX=1");
    assert_eq!(b.metal_flags(), ["-std=metal3.2", "-O2", "-DX=1"]);

    let mut b = KernelBuild::new(KernelType::Metal);
    b.flag("-std=metal4.0").flag("-O0");
    assert_eq!(b.metal_flags(), ["-std=metal4.0", "-O0"]);

    let mut b = KernelBuild::new(KernelType::Metal);
    b.flag("-Os");
    assert_eq!(b.metal_flags(), ["-std=metal3.2", "-Os"]);
}

#[test]
fn air_paths_stay_under_the_output_dir() {
    let out = Path::new("/build/out");
    // Generated source under OUT_DIR: relative to OUT_DIR.
    assert_eq!(
        metal_air_rel_path(Path::new("/build/out/metal/x.metal"), out),
        PathBuf::from("metal/x.air")
    );
    // Relative source: kept.
    assert_eq!(
        metal_air_rel_path(Path::new("kernels/zkp/metal/ntt.metal"), out),
        PathBuf::from("kernels/zkp/metal/ntt.air")
    );
    // Other absolute source: root dropped, so joining keeps it under the output dir.
    let rel = metal_air_rel_path(Path::new("/elsewhere/a/b.metal"), out);
    assert!(rel.is_relative());
    assert_eq!(rel, PathBuf::from("elsewhere/a/b.air"));
    // `..` can't climb out of the output dir.
    let rel = metal_air_rel_path(Path::new("../x/y.metal"), out);
    assert!(rel
        .components()
        .all(|c| matches!(c, std::path::Component::Normal(_))));
}

#[test]
fn metal_incs_ship_the_compat_headers() {
    let names: Vec<_> = METAL_INCS.iter().map(|(n, _)| *n).collect();
    assert_eq!(
        names,
        ["fp.h", "fpext.h", "cuda2metal.h", "sppark_compat.h"]
    );
}
