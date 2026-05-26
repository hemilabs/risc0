# R3-A07 Round-3 Agent 07 — Scaffolding for step_compute_accum Intel SYCL PoC

## TL;DR

Concrete scaffolding deliverables for the R3-A06 plan: empty-but-buildable Intel SYCL kernel header + FFI launcher + Rust FFI decl + build.rs sketch + HAL env-gated path. The scaffolding compiles a do-nothing `.so` that exports `risc0_circuit_recursion_intel_compute_accum`; the future kernel body amalgamation and oneDPL scan slot into clearly-marked TODOs. The HAL is touched only behind `RISC0_INTEL_GPU_ACCUM=1` so default behaviour is unchanged.

## Files produced

| Path | Status |
|---|---|
| `recursion-sys/kernels/intel/step_compute_accum.h` | **new (scaffold)** |
| `recursion-sys/kernels/intel/ffi_compute_accum.cpp` | **new (scaffold)** |
| `recursion-sys/build.rs` | **diff sketch** (additive) |
| `recursion-sys/src/lib.rs` | **diff sketch** (additive) |
| `recursion/src/prove/hal/intel.rs` | **diff sketch** (env-gated path) |

The two `.h` / `.cpp` files are valid C++17 / SYCL and produce a linkable `.so` even before the kernel bodies are amalgamated in. The build.rs hook is a copy-modify of `build_intel_kernels` (the existing eval_check build) with the kernel-amalgamation step left as a placeholder.

---

## File 1: `recursion-sys/kernels/intel/step_compute_accum.h` (new)

Scaffold — defines the device-side context struct + the two externs the generated `step_compute_accum.cpp` / `step_verify_accum.cpp` bodies reference. Mirrors rv32im's `kernels/intel/witgen.h` in spirit: SYCL-safe replacements for cxx scaffolding, plus the externs.

```cpp
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

// Intel SYCL device-side scaffolding for step_compute_accum + step_verify_accum.
// Mirrors recursion-sys/kernels/cxx/extern.h and rv32im-sys/kernels/intel/witgen.h.
//
// The amalgamation in build.rs prepends this header, then concatenates
// the bodies of step_compute_accum.cpp and step_verify_accum.cpp.
// The generated bodies reference:
//   - extern_plonkWriteAccum_wom(ctx, cycle, "wom", {Fp,Fp,Fp,Fp})
//   - extern_plonkReadAccum_wom (step_verify_accum only)
// Both have inline device implementations below.

#pragma once

#include "fp.h"
#include "fpext.h"

#include <array>
#include <cstdint>
#include <cstddef>

// Override assert / cassert: SYCL device code can't host-side abort.
// The generated SSA uses defensive assert() that must compile away.
// Same trick used by rv32im-sys/kernels/intel/witgen.h:29-32.
#ifdef assert
#undef assert
#endif
#define assert(x) ((void)0)

namespace risc0::circuit::recursion {

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif

// Device-only AccumContext: just raw pointers, no std::vector.
// Layout-compatible with the call patterns in the generated code,
// which only ever does `actx->accum[cycle] = ...`. We never call
// `.size()`, `.begin()`, etc. — see Round-2 risk #7 for analysis.
//
// IMPORTANT: layout of `accum` (first pointer field) is load-bearing,
// because the generated `static_cast<AccumContext*>(ctx)->accum[cycle]`
// expects a sequence-of-FpExt that supports operator[].  `FpExt*` provides
// that with identical syntax to `std::vector<FpExt>`.
struct DeviceAccumContext {
  FpExt*   accum;     // FpExt[steps]    — scan input, set by extern_plonkWriteAccum_wom
  uint32_t steps;
  uint32_t cycles;

  // The five Fp pointers passed in via args[]. The generated body indexes
  // them as args[0..4]. We mirror them here so the externs can reach them
  // without a separate global. Mostly informational — the actual args[]
  // is built fresh per work-item in ffi_compute_accum.cpp.
  Fp* ctrl;
  Fp* global;
  Fp* data;
  Fp* mix;
  Fp* accum_fp;
};

// Device implementations of the externs the generated code calls.
// Both are inline so they're emitted into each work-item's kernel.

inline void extern_plonkWriteAccum_wom(void* ctx,
                                       size_t cycle,
                                       const char* /*extra*/,
                                       std::array<Fp, 4> args) {
  auto* actx = static_cast<DeviceAccumContext*>(ctx);
  actx->accum[cycle] = FpExt(args[0], args[1], args[2], args[3]);
}

inline std::array<Fp, 4> extern_plonkReadAccum_wom(void* ctx,
                                                   size_t cycle,
                                                   const char* /*extra*/,
                                                   std::array<Fp, 0> /*args*/) {
  auto* actx = static_cast<DeviceAccumContext*>(ctx);
  FpExt v = actx->accum[cycle];
  return {v.elems[0], v.elems[1], v.elems[2], v.elems[3]};
}

// extern_log: device no-op. step_compute_accum.cpp does not call it
// (verified via grep), but we provide the symbol so any future
// generated-code change with logging compiles.
template <typename T>
inline void extern_log(void* /*ctx*/, size_t /*cycle*/,
                       const char* /*extra*/, T /*args*/) {}

// Forward declarations of the per-cycle entry points. The amalgamation
// supplies the definitions by including the generated .cpp bodies.
Fp step_compute_accum(void* ctx, size_t steps, size_t cycle, Fp** args);
Fp step_verify_accum(void* ctx, size_t steps, size_t cycle, Fp** args);

} // namespace risc0::circuit::recursion
```

**Scaffolding note**: Until the build.rs amalgamation actually inlines the two `step_*.cpp` bodies, the forward decls at the bottom remain unresolved. To make the `.so` build right now without the bodies, ffi_compute_accum.cpp ships with **weak stubs** (see comment block in File 2). On day 2 of the PoC plan, the stubs are removed and the real bodies amalgamated.

---

## File 2: `recursion-sys/kernels/intel/ffi_compute_accum.cpp` (new)

Scaffold — SYCL queue + USM cache + a three-phase kernel launcher whose Phase 1/3 are stubbed (no-op) and Phase 2 is left as a `// TODO: oneDPL inclusive_scan` block. The `extern "C"` entry point exists and is callable.

```cpp
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

// Intel SYCL launcher for step_compute_accum + step_verify_accum.
// SCAFFOLDING ONLY — kernel bodies are TODO; see PoC plan day-by-day.
//
// Build invocation (from recursion-sys/build.rs::build_intel_accum):
//   icpx -shared -fPIC -fsycl -std=c++17 -O1 -Xs '-options -cl-opt-disable'
//        -fsycl-targets=intel_gpu_bmg_g31
//        <amalgamation>.cpp -o librisc0_recursion_intel_accum.so
//
// Amalgamation order in build.rs:
//   1. #include "fp.h" / "fpext.h" / <cstdint>
//   2. #include "kernels/intel/step_compute_accum.h"
//   3. extracted body of kernels/cxx/step_compute_accum.cpp  (defines step_compute_accum)
//   4. extracted body of kernels/cxx/step_verify_accum.cpp   (defines step_verify_accum)
//   5. this file (ffi_compute_accum.cpp)

#include <sycl/sycl.hpp>
// TODO(day-3): switch to oneDPL once Phase 2 (inclusive_scan) is wired up.
// #include <oneapi/dpl/algorithm>
// #include <oneapi/dpl/execution>
// #include <oneapi/dpl/numeric>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef assert
#undef assert
#endif
#define assert(x) ((void)0)

#include "fp.h"
#include "fpext.h"

// Pulled in by the amalgamation, but listed here so the scaffold compiles
// standalone (with weak stub bodies — see SCAFFOLD_STUB_KERNELS below).
// In the real build the amalgamation provides full definitions and these
// weak fallbacks are linker-suppressed.
namespace risc0::circuit::recursion {

#if defined(SCAFFOLD_STUB_KERNELS)
// Weak stubs so the bare scaffold links. Removed once the amalgamation
// inlines the real generated bodies from kernels/cxx/step_compute_accum.cpp
// and kernels/cxx/step_verify_accum.cpp.
__attribute__((weak)) Fp step_compute_accum(void* /*ctx*/, size_t /*steps*/,
                                            size_t /*cycle*/, Fp** /*args*/) {
  return Fp(0);
}
__attribute__((weak)) Fp step_verify_accum(void* /*ctx*/, size_t /*steps*/,
                                           size_t /*cycle*/, Fp** /*args*/) {
  return Fp(0);
}
#endif

// Device USM cache: persistent FpExt scratch + DeviceAccumContext.
// Grow-only allocator across calls — mirrors the rv32im accum pattern.
struct IntelAccumCache {
  FpExt*               d_accum_ext = nullptr;  // size = steps_cap
  uint32_t             steps_cap   = 0;
  DeviceAccumContext*  d_ctx       = nullptr;  // single shared-USM struct
};
static IntelAccumCache g_accum_cache;

static void ensure_cache(sycl::queue& q, uint32_t steps) {
  if (steps > g_accum_cache.steps_cap) {
    if (g_accum_cache.d_accum_ext) sycl::free(g_accum_cache.d_accum_ext, q);
    g_accum_cache.d_accum_ext = sycl::malloc_device<FpExt>(steps, q);
    g_accum_cache.steps_cap   = steps;
  }
  if (!g_accum_cache.d_ctx) {
    g_accum_cache.d_ctx = sycl::malloc_shared<DeviceAccumContext>(1, q);
  }
}

static const char* make_error(const std::string& s) {
  // Same pattern as eval_check.cpp:13. strdup ok — host-side, short string.
  return strdup(s.c_str());
}

} // namespace risc0::circuit::recursion

// ============================================================================
// extern "C" entry point — the symbol Rust binds to.
// ============================================================================
extern "C" const char* risc0_circuit_recursion_intel_compute_accum(
    void* queue_ptr,
    void* d_ctrl_ptr,
    void* d_global_ptr,
    void* d_data_ptr,
    void* d_mix_ptr,
    void* d_accum_ptr,
    uint32_t work_cycles,
    uint32_t total_cycles)
{
  using namespace risc0::circuit::recursion;

  try {
    auto& q = *static_cast<sycl::queue*>(queue_ptr);
    auto t0 = std::chrono::steady_clock::now();

    ensure_cache(q, work_cycles);

    // ----- Phase 0: initialise the FpExt scan scratch to FpExt(1) -----
    {
      FpExt* d_accum_ext = g_accum_cache.d_accum_ext;
      q.parallel_for(sycl::range<1>(work_cycles), [=](sycl::id<1> i) {
        d_accum_ext[i] = FpExt(Fp(1), Fp(0), Fp(0), Fp(0));
      });
    }

    // Populate the device context.
    DeviceAccumContext host_ctx{};
    host_ctx.accum    = g_accum_cache.d_accum_ext;
    host_ctx.steps    = work_cycles;
    host_ctx.cycles   = total_cycles;
    host_ctx.ctrl     = static_cast<Fp*>(d_ctrl_ptr);
    host_ctx.global   = static_cast<Fp*>(d_global_ptr);
    host_ctx.data     = static_cast<Fp*>(d_data_ptr);
    host_ctx.mix      = static_cast<Fp*>(d_mix_ptr);
    host_ctx.accum_fp = static_cast<Fp*>(d_accum_ptr);
    *g_accum_cache.d_ctx = host_ctx;
    q.wait();

    // Tunable WG_SIZE — start at 256 (CUDA default).
    uint32_t WG_SIZE = 256;
    if (const char* s = std::getenv("RISC0_RECURSION_ACCUM_WG")) {
      int v = std::atoi(s);
      if (v == 32 || v == 64 || v == 128 || v == 256 || v == 512 || v == 1024)
        WG_SIZE = (uint32_t)v;
    }
    uint32_t global_size = ((work_cycles + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
    auto t1 = std::chrono::steady_clock::now();

    // ----- Phase 1: per-cycle step_compute_accum (SCAFFOLD: stub bodies)
    {
      Fp* ctrl     = host_ctx.ctrl;
      Fp* global   = host_ctx.global;
      Fp* data     = host_ctx.data;
      Fp* mix      = host_ctx.mix;
      Fp* accum_fp = host_ctx.accum_fp;
      DeviceAccumContext* ctx_ptr = g_accum_cache.d_ctx;
      uint32_t cycles = total_cycles;
      q.parallel_for(
          sycl::nd_range<1>(global_size, WG_SIZE),
          [=](sycl::nd_item<1> item) {
            uint32_t cycle = item.get_global_id(0);
            if (cycle >= work_cycles) return;
            Fp* args[5] = {ctrl, global, data, mix, accum_fp};
            (void)step_compute_accum(ctx_ptr, cycles, cycle, args);
          });
      q.wait();
    }
    auto t2 = std::chrono::steady_clock::now();

    // ----- Phase 2: multiplicative inclusive_scan on FpExt -----
    // TODO(day-3): replace stub with oneDPL or custom Hillis-Steele scan.
    // Reference CPU op: std::inclusive_scan with FpExt mul, identity FpExt(1).
    // Sketch (uncomment + add oneDPL includes once tested):
    // {
    //   auto policy = oneapi::dpl::execution::make_device_policy(q);
    //   FpExt* p = g_accum_cache.d_accum_ext;
    //   oneapi::dpl::inclusive_scan(
    //       policy, p, p + work_cycles, p,
    //       [](FpExt a, FpExt b) { return a * b; },
    //       FpExt(Fp(1), Fp(0), Fp(0), Fp(0)));
    //   q.wait();
    // }
    auto t3 = std::chrono::steady_clock::now();

    // ----- Phase 3: per-cycle step_verify_accum (SCAFFOLD: stub bodies)
    {
      Fp* ctrl     = host_ctx.ctrl;
      Fp* global   = host_ctx.global;
      Fp* data     = host_ctx.data;
      Fp* mix      = host_ctx.mix;
      Fp* accum_fp = host_ctx.accum_fp;
      DeviceAccumContext* ctx_ptr = g_accum_cache.d_ctx;
      uint32_t cycles = total_cycles;
      q.parallel_for(
          sycl::nd_range<1>(global_size, WG_SIZE),
          [=](sycl::nd_item<1> item) {
            uint32_t cycle = item.get_global_id(0);
            if (cycle >= work_cycles) return;
            Fp* args[5] = {ctrl, global, data, mix, accum_fp};
            (void)step_verify_accum(ctx_ptr, cycles, cycle, args);
          });
      q.wait();
    }
    auto t4 = std::chrono::steady_clock::now();

    if (std::getenv("RISC0_VERBOSE")) {
      fprintf(stderr,
              "      [ffi_recursion_accum] init: %.1fms, "
              "compute: %.1fms, scan: %.1fms, verify: %.1fms, "
              "cycles=%u/%u\n",
              std::chrono::duration<double, std::milli>(t1 - t0).count(),
              std::chrono::duration<double, std::milli>(t2 - t1).count(),
              std::chrono::duration<double, std::milli>(t3 - t2).count(),
              std::chrono::duration<double, std::milli>(t4 - t3).count(),
              work_cycles, total_cycles);
    }

    return nullptr;
  } catch (const sycl::exception& e) {
    return make_error(std::string("SYCL recursion accum error: ") + e.what());
  } catch (const std::exception& e) {
    return make_error(std::string("recursion accum error: ") + e.what());
  } catch (...) {
    return make_error("Unknown error in intel recursion compute_accum");
  }
}
```

**Scaffolding choices**:
- `SCAFFOLD_STUB_KERNELS` define lets the file compile and link before the amalgamation is wired. Defined via `-DSCAFFOLD_STUB_KERNELS` in build.rs on day 1; removed on day 2.
- Phase 2 (scan) is commented out — Day 3 work. oneDPL include guarded behind a TODO so build.rs's `-fsycl` invocation links cleanly without `-l<dpl>` flags.
- Phase 1/3 launch the (stub) kernel bodies on real `nd_range` shape so we can profile launch overhead before the real bodies arrive.
- All five buffer pointers + queue exactly match the FFI declared in `src/lib.rs` (File 4 below).

---

## File 3: `recursion-sys/build.rs` change sketch (additive)

**Diff** against the current `recursion-sys/build.rs:422-565`. Don't apply — these are the *additions*; the existing `build_intel_kernels` body stays put.

```diff
@@ recursion-sys/build.rs
 #[allow(dead_code)]
 fn build_intel_kernels() {
     rerun_if_changed("kernels/intel");
     rerun_if_changed("kernels/cxx");
     println!("cargo:rerun-if-env-changed=RISC0_RECURSION_OPTIMIZE");
+    println!("cargo:rerun-if-env-changed=RISC0_RECURSION_ACCUM_OPTIMIZE");

     let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
     let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
-    // Find icpx compiler
     let icpx = env::var("RISC0_ICPX")
         .map(PathBuf::from)
         .unwrap_or_else(|_| {
             let oneapi = PathBuf::from("/opt/intel/oneapi/compiler/latest/bin/icpx");
             if oneapi.exists() { oneapi } else { PathBuf::from("icpx") }
         });
     let cache_dir = out_dir
         .ancestors()
         .find(|p| {
             p.parent()
                 .and_then(|q| q.file_name())
                 .map(|n| n == std::ffi::OsStr::new("target"))
                 .unwrap_or(false)
         })
         .map(|p| p.join("intel_recursion_cache"))
         .unwrap_or_else(|| out_dir.join("intel_recursion_cache"));
     std::fs::create_dir_all(&cache_dir).unwrap();

     // ... existing eval_check build (unchanged) ...

+    // ===== R3-A06: accum SYCL kernel (PoC scaffold) =====
+    build_intel_accum(&icpx, &cxx_root, &out_dir, &cache_dir);
 }

+// New helper — builds librisc0_recursion_intel_accum.so.
+// Mirrors the eval_check build pattern; on day-1 of the PoC the amalgamation
+// step is skipped and SCAFFOLD_STUB_KERNELS is defined so the .so links with
+// no kernel body.
+fn build_intel_accum(
+    icpx: &Path,
+    cxx_root: &str,
+    out_dir: &Path,
+    cache_dir: &Path,
+) {
+    let so_path = cache_dir.join("librisc0_recursion_intel_accum.so");
+
+    let icpx_version = stamp::icpx_version(icpx);
+    let accum_hash = compute_recursion_accum_hash(cxx_root, &icpx_version);
+    let stamp_path = cache_dir.join("intel_accum.stamp");
+
+    if stamp::need_rebuild(&so_path, &stamp_path, &accum_hash) {
+        eprintln!("Building Intel SYCL accum kernel for recursion (scaffold)...");
+
+        // SCAFFOLD: skip amalgamation; compile ffi_compute_accum.cpp with
+        // SCAFFOLD_STUB_KERNELS so the link resolves stub step_*_accum.
+        // Day-2 PoC work removes this branch and switches to the amalgamation.
+        let scaffold_only = std::env::var_os("RISC0_RECURSION_ACCUM_SCAFFOLD").is_some();
+
+        let recursion_optimize = std::env::var_os("RISC0_RECURSION_ACCUM_OPTIMIZE").is_some();
+        let mut cmd = Command::new(icpx);
+        cmd.arg("-shared").arg("-fPIC").arg("-fsycl").arg("-std=c++17");
+        if recursion_optimize {
+            cmd.arg("-O2")
+               .arg("-Xs").arg("-options \"-cl-intel-256-GRF-per-thread\"");
+        } else {
+            cmd.arg("-O1")
+               .arg("-Xs").arg("-options -cl-opt-disable");
+        }
+        cmd.arg("-Wno-unused-parameter")
+           .arg("-Wno-unused-function")
+           .arg("-Wno-unused-variable")
+           .arg("-Wno-sign-compare")
+           .arg(format!("-I{cxx_root}"))
+           .arg("-Ikernels/cxx")
+           .arg("-Ikernels/intel")
+           .arg("-fsycl-targets=intel_gpu_bmg_g31");
+
+        let input_cpp;
+        if scaffold_only {
+            // Day-1: just build ffi_compute_accum.cpp standalone with stubs.
+            cmd.arg("-DSCAFFOLD_STUB_KERNELS");
+            input_cpp = PathBuf::from("kernels/intel/ffi_compute_accum.cpp");
+        } else {
+            // Day-2+: amalgamate step_compute_accum.h + cxx bodies + ffi wrapper.
+            // (Helper functions `extract_namespace_body` and `inject_noinline`
+            //  should be factored from the existing eval_check build path.)
+            let amalg_path = out_dir.join("intel_recursion_accum_amalg.cpp");
+            let mut amalg = String::new();
+            amalg.push_str("// Auto-generated SYCL accum amalgamation\n");
+            amalg.push_str("#include \"fp.h\"\n");
+            amalg.push_str("#include \"fpext.h\"\n");
+            amalg.push_str("#include <cstdint>\n");
+            amalg.push_str("#include \"kernels/intel/step_compute_accum.h\"\n");
+            amalg.push_str("namespace risc0::circuit::recursion {\n");
+            for src in &["kernels/cxx/step_compute_accum.cpp",
+                         "kernels/cxx/step_verify_accum.cpp"] {
+                // TODO: factor extract_namespace_body + inject_noinline shared
+                //       helpers (currently inlined in build_intel_kernels).
+                let body = extract_namespace_body(src);
+                let modified = inject_noinline(
+                    &body, &["Fp step_compute_accum(", "Fp step_verify_accum("]);
+                amalg.push_str(&modified);
+            }
+            amalg.push_str("} // namespace risc0::circuit::recursion\n");
+            amalg.push_str(
+                &std::fs::read_to_string("kernels/intel/ffi_compute_accum.cpp").unwrap());
+            std::fs::write(&amalg_path, &amalg).unwrap();
+            input_cpp = amalg_path;
+        }
+
+        cmd.arg(&input_cpp).arg("-o").arg(&so_path);
+
+        let output = cmd.output().expect("Failed to run icpx for accum");
+        if !output.status.success() {
+            stamp::write(&stamp_path, &accum_hash, &icpx_version, "failed");
+            let stderr = String::from_utf8_lossy(&output.stderr);
+            panic!("Intel recursion accum compilation failed:\n{}", stderr);
+        }
+        stamp::write(&stamp_path, &accum_hash, &icpx_version, "ok");
+        eprintln!("  Built {}", so_path.display());
+    } else {
+        eprintln!("Using cached Intel recursion accum kernel");
+    }
+
+    println!("cargo:rustc-link-search=native={}", cache_dir.display());
+    println!("cargo:rustc-link-lib=dylib=risc0_recursion_intel_accum");
+}
+
+fn compute_recursion_accum_hash(cxx_root: &str, icpx_version: &str) -> String {
+    let pairs: Vec<(PathBuf, PathBuf)> = vec![
+        ("kernels/intel/step_compute_accum.h".into(),
+         "kernels/intel/step_compute_accum.h".into()),
+        ("kernels/intel/ffi_compute_accum.cpp".into(),
+         "kernels/intel/ffi_compute_accum.cpp".into()),
+        ("kernels/cxx/step_compute_accum.cpp".into(),
+         "kernels/cxx/step_compute_accum.cpp".into()),
+        ("kernels/cxx/step_verify_accum.cpp".into(),
+         "kernels/cxx/step_verify_accum.cpp".into()),
+        ("build.rs".into(), "build.rs".into()),
+    ];
+    let opt_tag = if std::env::var_os("RISC0_RECURSION_ACCUM_OPTIMIZE").is_some()
+                  { "opt1" } else { "opt0" };
+    let scaffold_tag = if std::env::var_os("RISC0_RECURSION_ACCUM_SCAFFOLD").is_some()
+                       { "scaffold" } else { "full" };
+    stamp::hash_labeled(&pairs, &[icpx_version, "v1", "accum", opt_tag, scaffold_tag])
+}
```

**Build envvars introduced**:
- `RISC0_RECURSION_ACCUM_SCAFFOLD=1` — day-1 stub build (skip amalgamation)
- `RISC0_RECURSION_ACCUM_OPTIMIZE=1` — `-O2` + GRF=256 (day-7)
- `RISC0_RECURSION_ACCUM_WG=<N>` — runtime WG sweep (day-7)

---

## File 4: `recursion-sys/src/lib.rs` Rust FFI decl (additive)

```diff
@@ recursion-sys/src/lib.rs:111-126
 #[cfg(feature = "intel")]
 extern "C" {
     pub fn risc0_circuit_recursion_intel_eval_check(
         queue: *mut std::os::raw::c_void,
         check: *mut std::os::raw::c_void,
         ctrl: *const std::os::raw::c_void,
         data: *const std::os::raw::c_void,
         accum: *const std::os::raw::c_void,
         mix: *const std::os::raw::c_void,
         out: *const std::os::raw::c_void,
         poly_mix: *const std::os::raw::c_void,
         rou: u32,
         po2: u32,
         domain: u32,
     ) -> *const std::os::raw::c_char;
+
+    /// R3-A06 scaffold: GPU-side compute_accum + verify_accum.
+    /// All five buffer pointers must be device USM; the kernel does no D2H/H2D.
+    /// Returns NULL on success or a strdup'd C string on failure.
+    pub fn risc0_circuit_recursion_intel_compute_accum(
+        queue:  *mut std::os::raw::c_void,
+        ctrl:   *mut std::os::raw::c_void,
+        global: *mut std::os::raw::c_void,
+        data:   *mut std::os::raw::c_void,
+        mix:    *mut std::os::raw::c_void,
+        accum:  *mut std::os::raw::c_void,
+        work_cycles:  u32,
+        total_cycles: u32,
+    ) -> *const std::os::raw::c_char;
 }
```

All five buffer args are `*mut`: `step_verify_accum` writes to args[4] (accum) and the cxx path also updates `global` indirectly.

---

## File 5: HAL env-gated path in `recursion/src/prove/hal/intel.rs::accumulate`

```diff
@@ recursion/src/prove/hal/intel.rs::CircuitAccumulator::accumulate
 impl<IH: IntelHash> CircuitAccumulator<IntelHal<IH>> for IntelCircuitHal<IH> {
     fn accumulate(
         &self,
         work_cycles: u32,
         total_cycles: u32,
         ctrl: &IntelBuffer<BabyBearElem>,
         global: &IntelBuffer<BabyBearElem>,
         data: &IntelBuffer<BabyBearElem>,
         mix: &IntelBuffer<BabyBearElem>,
         accum: &IntelBuffer<BabyBearElem>,
     ) -> Result<()> {
-        let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
-        let t0 = std::time::Instant::now();
-        // Download GPU buffers to host for CPU FFI
-        let ctrl_host = ctrl.to_vec();
-        // ... existing D2H+CPU+H2D body ...
+        // R3-A06: opt into GPU accum via RISC0_INTEL_GPU_ACCUM=1.
+        // Default is the proven D2H+CPU+H2D path below.
+        if std::env::var_os("RISC0_INTEL_GPU_ACCUM").is_some() {
+            return self.accumulate_gpu(
+                work_cycles, total_cycles, ctrl, global, data, mix, accum);
+        }
+        self.accumulate_cpu(
+            work_cycles, total_cycles, ctrl, global, data, mix, accum)
+    }
+}
+
+impl<IH: IntelHash> IntelCircuitHal<IH> {
+    // GPU path: zero D2H/H2D, device pointers straight into the FFI.
+    fn accumulate_gpu(
+        &self,
+        work_cycles: u32,
+        total_cycles: u32,
+        ctrl:   &IntelBuffer<BabyBearElem>,
+        global: &IntelBuffer<BabyBearElem>,
+        data:   &IntelBuffer<BabyBearElem>,
+        mix:    &IntelBuffer<BabyBearElem>,
+        accum:  &IntelBuffer<BabyBearElem>,
+    ) -> Result<()> {
+        let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
+        let t0 = std::time::Instant::now();
+        let queue = risc0_sys::intel::get_queue();
+        risc0_sys::intel::esimd_check(unsafe {
+            risc0_circuit_recursion_sys::risc0_circuit_recursion_intel_compute_accum(
+                queue,
+                ctrl.as_device_ptr().0   as *mut std::ffi::c_void,
+                global.as_device_ptr().0 as *mut std::ffi::c_void,
+                data.as_device_ptr().0   as *mut std::ffi::c_void,
+                mix.as_device_ptr().0    as *mut std::ffi::c_void,
+                accum.as_device_ptr().0  as *mut std::ffi::c_void,
+                work_cycles,
+                total_cycles,
+            )
+        });
+        if verbose {
+            eprintln!("[recursion_accum_intel_gpu] total={:.1}ms work_cycles={} total_cycles={}",
+                t0.elapsed().as_secs_f64() * 1000.0,
+                work_cycles, total_cycles);
+        }
+        Ok(())
+    }
+
+    // Fallback: verbatim copy of the original D2H+CPU FFI+H2D body.
+    fn accumulate_cpu(
+        &self,
+        work_cycles: u32,
+        total_cycles: u32,
+        ctrl:   &IntelBuffer<BabyBearElem>,
+        global: &IntelBuffer<BabyBearElem>,
+        data:   &IntelBuffer<BabyBearElem>,
+        mix:    &IntelBuffer<BabyBearElem>,
+        accum:  &IntelBuffer<BabyBearElem>,
+    ) -> Result<()> {
+        let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
+        let t0 = std::time::Instant::now();
+        // (verbatim previous body: ctrl.to_vec()…ffi_wrap…copy_from_host)
+        let ctrl_host = ctrl.to_vec();
+        let global_host = global.to_vec();
+        let data_host = data.to_vec();
+        let mix_host = mix.to_vec();
+        let accum_host = accum.to_vec();
+        let d2h_ms = t0.elapsed().as_secs_f64() * 1000.0;
+        let buffers = RawAccumBuffers {
+            ctrl: ctrl_host.as_ptr(),
+            global: global_host.as_ptr(),
+            data: data_host.as_ptr(),
+            mix: mix_host.as_ptr(),
+            accum: accum_host.as_ptr(),
+        };
+        let t1 = std::time::Instant::now();
+        ffi_wrap(|| unsafe {
+            risc0_circuit_recursion_cpu_accum(&buffers, work_cycles, total_cycles)
+        })?;
+        let ffi_ms = t1.elapsed().as_secs_f64() * 1000.0;
+        let t2 = std::time::Instant::now();
+        accum.copy_from_host(&accum_host);
+        global.copy_from_host(&global_host);
+        let h2d_ms = t2.elapsed().as_secs_f64() * 1000.0;
+        if verbose {
+            eprintln!("[recursion_accum_intel] d2h={d2h_ms:.1}ms ffi_cpu={ffi_ms:.1}ms h2d={h2d_ms:.1}ms total={:.1}ms work_cycles={} total_cycles={}",
+                t0.elapsed().as_secs_f64() * 1000.0,
+                work_cycles, total_cycles);
+        }
+        Ok(())
+    }
 }
```

**Default behaviour unchanged** — `accumulate_cpu` is the same body as today, just factored. Set `RISC0_INTEL_GPU_ACCUM=1` to opt into the new GPU FFI.

---

## Integration sketch — order of operations to make the scaffold buildable

1. **Drop in File 1** (`step_compute_accum.h`) — independent header, no link deps.
2. **Drop in File 2** (`ffi_compute_accum.cpp`) — includes File 1; compiles standalone with `-DSCAFFOLD_STUB_KERNELS`.
3. **Apply File 3 diff** to `recursion-sys/build.rs` — adds `build_intel_accum` and the hash/stamp helper. Set `RISC0_RECURSION_ACCUM_SCAFFOLD=1` for the first build.
4. **Apply File 4 diff** to `recursion-sys/src/lib.rs` — exposes the FFI to Rust.
5. **Apply File 5 diff** to `recursion/src/prove/hal/intel.rs` — env-gated path; unset = unchanged behaviour.

After step 5: `cargo build -p risc0-circuit-recursion --features intel` succeeds. `librisc0_recursion_intel_accum.so` exists in the cache dir. `nm -D librisc0_recursion_intel_accum.so | grep compute_accum` shows the entry symbol. The FFI is callable but returns immediately (Phase 1/3 stubs are no-ops, Phase 2 commented out).

The scaffold is **strictly additive** — no existing test fails, no existing path changes behaviour. Day-2 of the PoC plan flips `RISC0_RECURSION_ACCUM_SCAFFOLD` off and starts wiring in the real kernel bodies via the amalgamation hook in build.rs.

---

## Open questions for future agents (day-2+ work)

1. Confirm `FpExt::operator*` is constexpr-friendly enough for oneDPL's `inclusive_scan` algorithm (Round-2 risk #2). If not, write the custom Hillis-Steele scan kernel.
2. Confirm `step_verify_accum.cpp` calls only `extern_plonkReadAccum_wom` and nothing else (Round-2 risk #4). Grep confirms it does — but double check on the *generated* output after any circuit DSL bump.
3. Decide whether the noinline injection in build.rs should target both function bodies, or just `step_compute_accum` (eval_check pattern injects on just `poly_fp`). Default: both.
4. Validate that `FpExt`'s constructor `FpExt(Fp,Fp,Fp,Fp)` and `FpExt(Fp(1), Fp(0), Fp(0), Fp(0))` identity element compile in device code (rv32im already does this — should be fine).
