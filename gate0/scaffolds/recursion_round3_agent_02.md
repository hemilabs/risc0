# R3-A06 Round-3 Agent 02 — Apply R2-03 freebie patch

## Status: APPLIED

The RISC0_RECURSION_EVAL_CHECK_WG env knob has been applied to recursion
eval_check exactly as specified in `/tmp/recursion_round2_agent_03.md`.

## File modified
`/home/user/risc0-intel/risc0/risc0/circuit/recursion-sys/kernels/intel/eval_check.cpp`

## Lines edited
- **Before:** line 43 contained `constexpr uint32_t WG_SIZE = 1024;` (single
  line); line 44 contained the `global_size` computation.
- **After:** lines 43-55 contain the env-driven WG selection (6 comment lines +
  7 code lines); line 56 contains the unchanged `global_size` computation.
- Net delta: +13 lines (1 replaced, 13 added).

## Verification
```
$ sed -n '43,56p' .../recursion-sys/kernels/intel/eval_check.cpp
        // Default WG=1024 chosen because recursion poly_fp is compiled with
        // `-O1 -cl-opt-disable` (see recursion-sys/build.rs), which keeps
        // register pressure low enough that the widest workgroup the device
        // supports is occupancy-optimal. If recursion ever moves to -O2 or
        // RISC0_RECURSION_OPTIMIZE=1, the spill profile may force WG<=512.
        // Runtime-tunable via RISC0_RECURSION_EVAL_CHECK_WG for sweeps.
        uint32_t WG_SIZE = 1024;
        if (const char* s = std::getenv("RISC0_RECURSION_EVAL_CHECK_WG")) {
            int v = std::atoi(s);
            if (v == 16 || v == 32 || v == 64 || v == 128 || v == 256 || v == 512 || v == 1024) {
                WG_SIZE = (uint32_t)v;
            }
        }
        uint32_t global_size = ((domain + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
```

## Pattern fidelity vs rv32im reference (rv32im-sys/eval_check.cpp:32-39)
- Same `std::getenv` + `std::atoi` + whitelist guard structure.
- Same `uint32_t WG_SIZE = <default>;` (non-const so the env branch can rewrite).
- Two recursion-specific deltas (per R2-03):
  1. Default = 1024 (not 512) — recursion uses `-cl-opt-disable`, register
     pressure is low, 1024 fits the resource budget.
  2. Whitelist includes 1024 — rv32im caps at 512 due to `simd_size=16`
     and `eu_thread_count=4`; recursion has no such flags.
- `<cstdlib>` not explicitly included — same as rv32im; transitive include via
  `<sycl/sycl.hpp>` / `<cstring>` is sufficient (rv32im compiles cleanly today
  with the same lack of explicit `<cstdlib>` include).

## Build status
Not rebuilt (per task instructions). The recursion eval_check `.so` will
require one rebuild before sweeping; after that, env-only.

## Compile-time risk
None expected. The diff is identical structure to rv32im's working knob; no
new headers; no new symbols; only `WG_SIZE` changed from `constexpr` to
`uint32_t` — every downstream use is `sycl::nd_range<1>(global_size, WG_SIZE)`
which accepts a runtime value (rv32im proves this works).

## Behavior at default (env unset or invalid)
`WG_SIZE` stays 1024 — byte-identical to baseline, satisfying R2-03
acceptance criterion #1.

## Files cited
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion-sys/kernels/intel/eval_check.cpp:43-56` — patched section
- `/home/user/risc0-intel/risc0/risc0/circuit/rv32im-sys/kernels/intel/eval_check.cpp:32-39` — reference pattern
- `/tmp/recursion_round2_agent_03.md` — R2 diff this round 3 applied
