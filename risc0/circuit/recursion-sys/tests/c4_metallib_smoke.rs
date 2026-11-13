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

//! C4 (docs/metal-port/PLAN.md 4.3): smoke test of the complete recursion metallib that build.rs
//! assembles (hand-written `rec_kernels.metal` + the step and eval_check kernels translated into
//! `$OUT_DIR/metal` at build time).
//!
//! - the library's function names are exactly [METAL_KERNEL_NAMES] (nothing missing, nothing
//!   unlisted, no duplicates), and the translated names agree with the translators'
//!   `--list-kernels` output;
//! - every listed kernel gives a compute pipeline state, created once each, with its creation
//!   time (cold on a first load of this metallib content), max threads, SIMD width and static
//!   threadgroup memory (≤ 32768 B) printed, plus the device's `currentAllocatedSize`.
//!
//! The crate has no `metal` dependency, so (like C1's test) this drives Metal through a minimal
//! `objc_msgSend` binding. Run with
//! `cargo test -p risc0-circuit-recursion-sys --features metal --test c4_metallib_smoke -- --nocapture`
//! to see the timing table.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::{
    collections::BTreeSet,
    ffi::{c_char, c_void, CStr},
    path::Path,
    process::Command,
    time::Instant,
};

use risc0_circuit_recursion_sys::{METAL_KERNEL_NAMES, METAL_LIB};

type Id = *mut c_void;
type Sel = *const c_void;

#[link(name = "objc")]
extern "C" {
    fn objc_msgSend();
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_getClass(name: *const c_char) -> Id;
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

#[link(name = "Metal", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> Id;
}

#[link(name = "Foundation", kind = "framework")]
extern "C" {}

extern "C" {
    // libdispatch (libSystem); a NULL destructor (DISPATCH_DATA_DESTRUCTOR_DEFAULT) copies.
    fn dispatch_data_create(
        buffer: *const c_void,
        size: usize,
        queue: *mut c_void,
        destructor: *mut c_void,
    ) -> Id;
}

fn sel(name: &str) -> Sel {
    debug_assert!(name.ends_with('\0'));
    unsafe { sel_registerName(name.as_ptr() as *const c_char) }
}

/// `msg!(Ret; obj, "selector:with:", a => TypeA, b => TypeB)`
macro_rules! msg {
    ($ret:ty; $obj:expr, $sel:literal $(, $a:expr => $t:ty)* $(,)?) => {{
        #[allow(clippy::missing_transmute_annotations)]
        let f: unsafe extern "C" fn(Id, Sel $(, $t)*) -> $ret =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f($obj, sel(concat!($sel, "\0")) $(, $a)*)
    }};
}

fn nsstring(s: &str) -> Id {
    let c = std::ffi::CString::new(s).unwrap();
    unsafe {
        msg!(Id; objc_getClass(c"NSString".as_ptr()), "stringWithUTF8String:",
             c.as_ptr() => *const c_char)
    }
}

fn rust_string(s: Id) -> String {
    unsafe {
        let p: *const c_char = msg!(*const c_char; s, "UTF8String");
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

fn describe(err: Id) -> String {
    if err.is_null() {
        return "(no error object)".into();
    }
    unsafe { rust_string(msg!(Id; err, "localizedDescription")) }
}

struct Lib {
    device: Id,
    library: Id,
}

impl Lib {
    fn load() -> Self {
        assert!(
            !METAL_LIB.is_empty(),
            "METAL_LIB is empty (RISC0_SKIP_BUILD_KERNELS?)"
        );
        unsafe {
            let device = MTLCreateSystemDefaultDevice();
            assert!(!device.is_null(), "no Metal device");
            let data = dispatch_data_create(
                METAL_LIB.as_ptr() as *const c_void,
                METAL_LIB.len(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            let mut err: Id = std::ptr::null_mut();
            let library =
                msg!(Id; device, "newLibraryWithData:error:", data => Id, &mut err => *mut Id);
            assert!(
                !library.is_null(),
                "newLibraryWithData failed: {}",
                describe(err)
            );
            Lib { device, library }
        }
    }

    /// `[library functionNames]`.
    fn function_names(&self) -> Vec<String> {
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let arr: Id = msg!(Id; self.library, "functionNames");
            let n: u64 = msg!(u64; arr, "count");
            let names = (0..n)
                .map(|i| rust_string(msg!(Id; arr, "objectAtIndex:", i => u64)))
                .collect();
            objc_autoreleasePoolPop(pool);
            names
        }
    }

    fn allocated(&self) -> u64 {
        unsafe { msg!(u64; self.device, "currentAllocatedSize") }
    }
}

fn python() -> Command {
    match std::env::var("RISC0_PYTHON") {
        Ok(p) => Command::new(p),
        Err(_) => Command::new("python3"),
    }
}

/// `<script> --list-kernels [extra]`, run from `kernels/cuda` (as build.rs runs the translators).
fn list_kernels(script: &str, extra: &[&str]) -> Vec<String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/cuda");
    let out = python()
        .current_dir(&dir)
        .arg(script)
        .arg("--list-kernels")
        .args(extra)
        .output()
        .unwrap_or_else(|e| panic!("could not run python for {script}: {e}"));
    assert!(
        out.status.success(),
        "{script} --list-kernels failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect()
}

const C1_NAMES: &[&str] = &[
    "injectWomBacks",
    "fillOneExt",
    "prefixProductsChunks",
    "prefixProductsTotals",
    "prefixProductsFixup",
    "recInitRows",
    "recFillTail",
    "recZeroizeRows",
    "markHostChains",
    "scatterHostChains",
];

#[test]
fn c4_kernel_names_match_rec_kernels_and_translators() {
    let listed: Vec<&str> = METAL_KERNEL_NAMES.to_vec();
    let unique: BTreeSet<&str> = listed.iter().copied().collect();
    assert_eq!(
        unique.len(),
        listed.len(),
        "duplicate METAL_KERNEL_NAMES entry"
    );

    let mut expected: Vec<String> = C1_NAMES.iter().map(|s| s.to_string()).collect();
    expected.extend(list_kernels("cuda2metal_steps.py", &[]));
    expected.extend(list_kernels(
        "cuda2metal_eval.py",
        &["--split", "--split-lz", "--split-rg"],
    ));
    assert_eq!(
        listed, expected,
        "METAL_KERNEL_NAMES must be rec_kernels.metal's kernels followed by the translators' \
         --list-kernels output"
    );
}

#[test]
fn c4_metallib_has_every_kernel_and_builds_each_pipeline_once() {
    let lib = Lib::load();

    // Exactly the listed functions: nothing missing, nothing unlisted.
    let in_lib: BTreeSet<String> = lib.function_names().into_iter().collect();
    let listed: BTreeSet<String> = METAL_KERNEL_NAMES.iter().map(|s| s.to_string()).collect();
    let missing: Vec<_> = listed.difference(&in_lib).collect();
    let unlisted: Vec<_> = in_lib.difference(&listed).collect();
    assert!(
        missing.is_empty(),
        "listed but not in METAL_LIB: {missing:?}"
    );
    assert!(
        unlisted.is_empty(),
        "in METAL_LIB but not listed: {unlisted:?}"
    );

    let before = lib.allocated();
    eprintln!(
        "METAL_LIB {} B, {} kernels; currentAllocatedSize before PSOs {} B",
        METAL_LIB.len(),
        METAL_KERNEL_NAMES.len(),
        before
    );
    eprintln!(
        "{:<22} {:>10} {:>8} {:>5} {:>9}",
        "kernel", "pso_s", "max_thr", "simd", "static_tg"
    );
    let mut total = 0.0f64;
    let mut psos = Vec::new();
    for name in METAL_KERNEL_NAMES {
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let func = msg!(Id; lib.library, "newFunctionWithName:", nsstring(name) => Id);
            assert!(!func.is_null(), "{name}: newFunctionWithName returned nil");
            let mut err: Id = std::ptr::null_mut();
            let t = Instant::now();
            let pso = msg!(Id; lib.device, "newComputePipelineStateWithFunction:error:",
                           func => Id, &mut err => *mut Id);
            let dt = t.elapsed().as_secs_f64();
            assert!(
                !pso.is_null(),
                "{name}: PSO creation failed: {}",
                describe(err)
            );
            let max: u64 = msg!(u64; pso, "maxTotalThreadsPerThreadgroup");
            let simd: u64 = msg!(u64; pso, "threadExecutionWidth");
            let tg: u64 = msg!(u64; pso, "staticThreadgroupMemoryLength");
            eprintln!("{name:<22} {dt:>10.3} {max:>8} {simd:>5} {tg:>9}");
            total += dt;
            assert!(
                tg <= 32768,
                "{name}: static threadgroup memory {tg} > 32768"
            );
            assert!(max > 0, "{name}: maxTotalThreadsPerThreadgroup is 0");
            // Every translated kernel and the scan carry max_total_threads_per_threadgroup(256).
            if !C1_NAMES.contains(name) || *name == "prefixProductsChunks" {
                assert!(max <= 256, "{name}: max threads {max} > the declared 256");
            }
            msg!((); func, "release");
            psos.push(pso);
            objc_autoreleasePoolPop(pool);
        }
    }
    let after = lib.allocated();
    eprintln!(
        "total PSO creation {total:.3} s; currentAllocatedSize after {} PSOs {} B (+{} B)",
        psos.len(),
        after,
        after.saturating_sub(before)
    );
    for pso in psos {
        unsafe { msg!((); pso, "release") }
    }
}
