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

//! The shared MSL headers (`METAL_INCS`) compile with the Metal toolchain (macOS only):
//! `threadgroup`/`device`/`constant` Fp and FpExt objects and operands, constexpr field
//! arithmetic checked with `static_assert`, every `cuda2metal.h` alias and the `sppark_compat.h`
//! names; and the existing zkp kernels compile against them and link into a metallib.

#![cfg(target_os = "macos")]

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use risc0_build_kernel::{find_metal_toolchain, METAL_INCS};

fn metal_available() -> bool {
    env::var_os("METAL_TC").is_some()
        || Command::new("xcrun")
            .args(["--sdk", "macosx", "--find", "metal"])
            .output()
            .is_ok_and(|o| o.status.success())
}

/// A fresh directory holding the shared headers, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = env::temp_dir().join(format!("f1-hdr-{tag}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(dir.join("inc")).unwrap();
        for (name, contents) in METAL_INCS {
            fs::write(dir.join("inc").join(name), contents).unwrap();
        }
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Compile `srcs` with `-std=metal3.2 -O2 -I <headers> [-I extra]` and link them into a
/// metallib, panicking with the compiler output on failure.
fn compile_and_link(scratch: &Scratch, srcs: &[PathBuf], extra_inc: Option<&Path>) {
    let tc = find_metal_toolchain();
    let mut airs = Vec::new();
    for src in srcs {
        let air = scratch
            .0
            .join(src.file_name().unwrap())
            .with_extension("air");
        let mut cmd = tc.metal_command();
        cmd.args([
            "-std=metal3.2",
            "-O2",
            "-Werror",
            "-Wno-unused-variable",
            "-I",
        ])
        .arg(scratch.0.join("inc"));
        if let Some(inc) = extra_inc {
            cmd.arg("-I").arg(inc);
        }
        cmd.arg("-c").arg(src).arg("-o").arg(&air);
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "{} failed to compile:\n{}",
            src.display(),
            String::from_utf8_lossy(&out.stderr)
        );
        airs.push(air);
    }
    let lib = scratch.0.join("out.metallib");
    let out = tc
        .metallib_command()
        .args(&airs)
        .arg("-o")
        .arg(&lib)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "metallib failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(lib.is_file());
}

fn compile_sources(tag: &str, sources: &[(&str, &str)]) {
    if !metal_available() {
        eprintln!("SKIPPED: no Metal toolchain (source /Users/user/zkminer/env.sh)");
        return;
    }
    let scratch = Scratch::new(tag);
    let srcs: Vec<_> = sources
        .iter()
        .map(|(name, src)| {
            let path = scratch.0.join(name);
            fs::write(&path, src).unwrap();
            path
        })
        .collect();
    compile_and_link(&scratch, &srcs, None);
}

/// threadgroup/device/constant Fp and FpExt objects and operands.
const ADDRESS_SPACES: &str = r#"
#include <metal_stdlib>
using namespace metal;
#include "fpext.h"

constant Fp kC = Fp(7);
constant FpExt kE = FpExt(Fp(1), Fp(2), Fp(3), Fp(4));
constant Fp kTab[3] = {Fp(1), Fp(2), Fp(3)};

kernel void probe_tg(device Fp* out [[buffer(0)]],
                     device FpExt* outE [[buffer(1)]],
                     device FpExt* io [[buffer(2)]],
                     constant Fp& cp [[buffer(3)]],
                     threadgroup Fp* tgp [[threadgroup(0)]],
                     uint lid [[thread_index_in_threadgroup]]) {
  threadgroup Fp x[8];
  threadgroup FpExt y[4];
  x[lid & 7] = Fp(lid);
  y[lid & 3] = FpExt(Fp(lid));
  tgp[lid] = Fp(lid) * Fp(3);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // constant Fp copied to a thread local, then multiplied (the documented convention)
  Fp c = kC;
  Fp m = c * x[(lid + 1) & 7];
  // by-value operators also take threadgroup/constant/device operands directly
  Fp d = x[lid & 7] * kC + cp - tgp[lid] * kTab[lid % 3];
  FpExt e = y[(lid + 1) & 3] * kE + y[lid & 3] * x[0] - kE;
  FpExt ee = io[lid] * y[0];
  x[lid & 7] += m;
  x[lid & 7] -= d;
  x[lid & 7] *= c;
  y[lid & 3] += e;
  y[lid & 3] -= ee;
  y[lid & 3] *= kC;
  y[lid & 3] *= e;
  io[lid] *= e;
  io[lid] -= ee;
  io[lid] *= c;
  bool eq = (x[0] == kC) || (y[0] != kE) || (kTab[0] < x[1]) || (io[0] == y[1]);
  out[lid] = m + d + (eq ? Fp(1) : Fp(0)) + (-x[lid & 7]) + (5 + c) + inv(x[1]);
  outE[lid] = e + ee + (-y[0]) + Fp(2) * y[1] + inv(y[2]) + pow(kE, 3) + FpExt(kC);
}
"#;

/// constexpr arithmetic and constant-object construction, checked at compile time.
const STATIC_CHECKS: &str = r#"
#include <metal_stdlib>
using namespace metal;
#include "fpext.h"
constant Fp kC = Fp(7);
constant FpExt kE = FpExt(Fp(1), Fp(2), Fp(3), Fp(4));
constant FpExt kE1 = FpExt(9u);
constant FpExt kE2 = FpExt(Fp::invalid());
static_assert((Fp(3) * Fp(5)).asUInt32() == 15, "mul");
static_assert((Fp(3) - Fp(5)).asUInt32() == Fp::P - 2, "sub");
static_assert((-Fp(1)).asUInt32() == Fp::P - 1, "neg");
static_assert((2 + Fp(5)).asUInt32() == 7, "int lhs");
static_assert(Fp(kC).asUInt32() == 7, "constant Fp");
static_assert(FpExt(kE).elems[3].asUInt32() == 4, "constant FpExt");
static_assert(FpExt(kE1).elems[0].asUInt32() == 9 && FpExt(kE1).elems[1].asUInt32() == 0, "constant FpExt(u)");
static_assert(FpExt(kE2).elems[0].asRaw() == Fp::INVALID, "raw preserved");
static_assert((inv(Fp(5)) * Fp(5)).asUInt32() == 1, "inv");
static_assert((inv(FpExt(Fp(1), Fp(2), Fp(3), Fp(4))) * FpExt(Fp(1), Fp(2), Fp(3), Fp(4))) == FpExt(1u), "inv ext");
static_assert(Fp(3) < Fp(5) && Fp(5) >= Fp(5) && Fp(6) > Fp(5) && Fp(5) <= Fp(5) && Fp(1) != Fp(2), "cmp");
kernel void k(device uint* o [[buffer(0)]]) { o[0] = 0; }
"#;

/// Every cuda2metal.h alias and the sppark_compat.h names.
const COMPAT: &str = r#"
#include "cuda2metal.h"
#include "sppark_compat.h"
using namespace metal;

static __device__ __forceinline__ uint twice(uint x) { return 2 * x; }
__device__ __noinline__ uint thrice(uint x) { return 3 * x; }
__device__ inline uint four(uint x) { return 4 * x; }

[[max_total_threads_per_threadgroup(256)]]
kernel void compat_k(device uint* __restrict__ out [[buffer(0)]],
                     device fr_t* f [[buffer(1)]],
                     device md4_t* e [[buffer(2)]],
                     device atomic_uint* fault [[buffer(3)]],
                     device int* ictr [[buffer(4)]],
                     threadgroup uint* tg [[threadgroup(0)]],
                     uint gid [[thread_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
  uint bad = 0;
  uint v = gid;
  v += __shfl_down_sync(0xffffffffu, v, 1);
  v += __shfl_up_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_sync(0xffffffffu, v, 0);
  v += __shfl_down(v, 4);
  ulong w = __shfl_down_sync(0xffffffffu, ulong(v), 1);
  uint b = __ballot_sync(0xffffffffu, lane & 1);
  __syncwarp();
  __syncwarp(0xffffffffu);
  tg[lane] = v;
  __syncthreads();
  __syncthreads_device();
  v += __popc(b) + __popcll(w) + __clz(v) + __clzll(w) + __brev(v) + __ffs(v) + __ffs(0);
  v += __umulhi(v, 0x12345678u) + uint(__umul64hi(w, w));
  v += __ldg(&out[0]) + __ldcg(&out[1]) + twice(v) + thrice(v) + four(v) + warpSize + WARP_SZ;
  uint old = atomicAdd(&out[2], 1u);
  old += atomicOr(&out[3], 4u);
  old += atomicCAS(&out[4], 0u, 1u);
  old += atomicAdd(fault, 0u);
  old += uint(atomicAdd(ictr, 1)) + uint(atomicCAS(ictr, 0, 2));
  old += atomicAdd(&tg[0], 1u) + atomicCAS(&tg[1], 0u, 1u) + atomicOr(&tg[2], 1u);
  asm("");
  __builtin_assume(v != 0);
  v = __builtin_elementwise_min(v, old);
  v = opq(v);
  assert(v == 7);
  R0_FAULT(bad, v < 100);
  printf("%d\n", v);
  r0::array<uint, 4> arr = {1, 2, 3, 4};
  cuda::std::array<fr_t, 2> fa = {fr_t(1), fr_t(2)};
  v += arr[gid & 3] + arr.size() + fa[1].asUInt32();
  // sppark compat
  fr_t x = f[gid];
  bb31_t y = csel(x, fr_t(3), gid & 1);
  md4_t z = e[gid];
  index_t i = gid;
  zero(f[i]);
  zero(e[i]);
  fr_t q; zero(q);
  md4_t r; zero(r);
  y = y ^ 5u;
  z = (z ^ (uint32_t)7) + csel(z, r, 0);
  y = reciprocal(y) + q;
  z = reciprocal(z);
  f[i] = y;
  e[i] = z;
  out[gid] = v;
  r0_fault_flush(fault, bad);
}
"#;

#[test]
fn address_space_objects_and_operands_compile() {
    compile_sources("as", &[("address_spaces.metal", ADDRESS_SPACES)]);
}

#[test]
fn constexpr_field_arithmetic_static_asserts() {
    compile_sources("sa", &[("static_checks.metal", STATIC_CHECKS)]);
}

#[test]
fn cuda2metal_and_sppark_compat_compile() {
    compile_sources("compat", &[("compat.metal", COMPAT)]);
}

#[test]
fn zkp_kernels_compile_with_the_shared_headers() {
    if !metal_available() {
        eprintln!("SKIPPED: no Metal toolchain (source /Users/user/zkminer/env.sh)");
        return;
    }
    let zkp = Path::new(env!("CARGO_MANIFEST_DIR")).join("../sys/kernels/zkp/metal");
    let mut srcs: Vec<_> = fs::read_dir(&zkp)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "metal"))
        .collect();
    srcs.sort();
    assert!(
        srcs.len() >= 7,
        "expected the zkp kernels in {}",
        zkp.display()
    );
    let scratch = Scratch::new("zkp");
    compile_and_link(&scratch, &srcs, Some(&zkp));
}
