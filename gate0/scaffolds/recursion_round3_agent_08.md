# R3-A08 Round 3 — Agent 08: Harness Skeleton (Rust)

## Endorsement of R2-06 (and R1-17)

R2-06's correction stands: probe lives **after zeroize**, two kill switches (`_WITGEN` + `_ACCUM`), multi-po2 on the tiny fixture only, noise-replay Option A. The 6-check structure in this task (data, global, iopIdx, womIndex×2, womRows-post-sort) refines R2-06's three-buffer compare (ctrl/data/global) by adding the **MachineContext** internal state that gets mutated inside the CPU FFI — those are the recursion-specific failure modes R1-17 §2-3 flagged (par-safety, post-sort determinism). Concretely they map to fields in `recursion-sys/kernels/cxx/context.h:52,70-71`:

| Check | What it covers | Source-of-truth |
|---|---|---|
| 1. `data`     | Witness columns (the main GPU output) | `IntelBuffer<Val>` post-witgen |
| 2. `global`   | Output/mix global registers | `IntelBuffer<Val>` post-witgen |
| 3. `iopIdx`   | Per-cycle iop cursor advance | `RawPreflightCycle.iopIdx` (mutated by `read_iop_*`) |
| 4. `womIndex` (pre-sort) | Per-cycle write count into `womRows` | `MachineContext.womIndex[cycle]` after generate_witness |
| 5. `womIndex` (post-zeroize) | Same field re-checked post-zeroize (catches HAL eltwise bugs) | same field after `eltwise_zeroize_elem` |
| 6. `womRows` (post-sort) | Argument trace after `std::sort(poolstl::par, womRows...)` | `MachineContext.womRows` after sort in `step_compute_accum` |

The C++ FFI must expose these four MachineContext views during the harness window. The cleanest way (also recommended by R2-06's "noise-replay shim" reasoning) is a `risc0_circuit_recursion_cpu_witgen_with_context` FFI variant that returns the populated `MachineContext` snapshots as out-parameters, gated by a build-time test flag so the production code path stays untouched. Day-1 stub: the harness asserts on `data`/`global` only and emits TODO comments for the context fields — the witgen kernel doesn't write them on GPU yet.

## Harness location

Per R2-06, the testutil lives at `risc0/circuit/recursion/src/prove/hal/testutil.rs` and the test module sits at the bottom of `risc0/circuit/recursion/src/prove/hal/intel.rs` (mirroring `rv32im/src/prove/hal/intel.rs:391-733` exactly). The kill-switch read happens at the top of `IntelCircuitHal::generate_witness` (line 61 of the current file) and `::accumulate` (line 107). The Layer-2 probe-call sits in `prove/witgen.rs` after the post-zeroize scope (R2-06's correction to R1-17).

For day-1, ALL the new harness Rust code is in one file (`testutil.rs`) so the diff is reviewable. The `mod tests` block at the bottom of `intel.rs` consumes `testutil` and adds the 7 AB-tests. The probe call in `witgen.rs` is a one-liner that delegates to `testutil::shadow_witgen_compare`.

## File: `risc0/circuit/recursion/src/prove/hal/intel.rs` (post-patch, kill-switch + test module)

The relevant new chunks (R2-06's diff already covers the verbatim-port of testutil.rs and the build.rs/witgen.rs hooks; below is the **net-new content** specific to this agent's deliverable):

```rust
// === Top of generate_witness (line ~70, before existing t0/d2h timing) ===
fn generate_witness(
    &self,
    mode: StepMode,
    total_cycles: u32,
    preflight: &RawPreflightTrace,
    ctrl: &IntelBuffer<BabyBearElem>,
    data: &IntelBuffer<BabyBearElem>,
    global: &IntelBuffer<BabyBearElem>,
) -> Result<()> {
    // Kill switch (R3-A06 §kill-switches). Forces CPU FFI fallback even
    // when the Intel HAL is otherwise selected — required for in-process
    // CPU/GPU byte-compare (Layer 3) and as the emergency rollback path
    // once the SYCL witgen kernel lands. Today the body IS the CPU FFI,
    // so this is a documented no-op until the kernel lands.
    let _force_cpu_witgen =
        std::env::var_os("RISC0_DISABLE_INTEL_RECURSION_WITGEN").is_some();

    let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
    // ... existing body unchanged ...
}

// === Equivalent at the top of accumulate ===
let _force_cpu_accum =
    std::env::var_os("RISC0_DISABLE_INTEL_RECURSION_ACCUM").is_some();

// === Bottom of file: the test module (~80 LOC) ===
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use risc0_zkp::{
        core::hash::poseidon2::Poseidon2HashSuite,
        field::baby_bear::{BabyBear, BabyBearElem as Val},
        hal::cpu::CpuHal,
    };

    use super::*;
    use crate::{
        prove::{
            hal::{
                cpu::CpuCircuitHal,
                testutil::{assert_buffer_eq, EnvGuard},
            },
            witgen::WitnessGenerator,
            Program,
            preflight::Preflight,
        },
        CIRCUIT,
    };

    /// Fixtures pinned in testdata/recursion_preflight/. test_recursion_circuit
    /// recompiles at multiple po2 — the real ZKRs are fixed-po2.
    const FIXTURES: &[(&str, usize)] = &[
        ("test_recursion_circuit", 14),
        ("test_recursion_circuit", 16),
        ("test_recursion_circuit", 18),
        ("lift_rv32im_v2_14",      18),
        ("join",                   18),
        ("resolve",                18),
        ("identity_p254",          21),
    ];

    fn load_pinned(name: &str, po2: usize) -> (Program, Preflight) {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata").join("recursion_preflight");
        let zkr = Program::decode(
            &std::fs::read(dir.join(format!("{name}_po2_{po2}.zkr"))).unwrap(),
            po2,
        ).unwrap();
        let pf = Preflight::decode(
            &std::fs::read(dir.join(format!("{name}_po2_{po2}.pf"))).unwrap(),
        ).unwrap();
        (zkr, pf)
    }

    fn cpu_vs_gpu_ab(name: &str, po2: usize) {
        // Disable in-process probe inside this test — we ARE the diff.
        let _env = EnvGuard::new(&[
            "RISC0_RECURSION_GPU_WITGEN_VERIFY",
            "RISC0_RECURSION_GPU_WITGEN_VERIFY_SOFT",
            "RISC0_DISABLE_INTEL_RECURSION_WITGEN",
        ]);
        std::env::remove_var("RISC0_RECURSION_GPU_WITGEN_VERIFY");
        std::env::remove_var("RISC0_RECURSION_GPU_WITGEN_VERIFY_SOFT");
        std::env::remove_var("RISC0_DISABLE_INTEL_RECURSION_WITGEN");

        let (zkr, pf) = load_pinned(name, po2);

        // CPU baseline
        let cpu_hal = CpuHal::<BabyBear>::new(Poseidon2HashSuite::new_suite());
        let cpu_ch  = CpuCircuitHal;
        let cpu_wg  = WitnessGenerator::new(&cpu_hal, &cpu_ch, &zkr, &pf, None).unwrap();

        // GPU under test
        let int_hal = Arc::new(IntelHalPoseidon2::new());
        let int_ch  = IntelCircuitHal::new(int_hal.clone());
        let int_wg  = WitnessGenerator::new(int_hal.as_ref(), &int_ch, &zkr, &pf, None).unwrap();

        // Check 1+2: data + global (the two GPU-owned buffers).
        let cpu_data = cpu_wg.data.as_slice().to_vec();
        let gpu_data = int_wg.data.to_vec();
        let cpu_global = cpu_wg.global.as_slice().to_vec();
        let gpu_global = int_wg.global.to_vec();

        // Per-cycle row scan — caps reporting noise at "first bad cycle".
        let data_size = CIRCUIT.data_size();
        let total = 1usize << po2;
        let limit: usize = std::env::var("RISC0_RECURSION_AB_SCAN_LIMIT")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(total);
        for row in 0..limit.min(total) {
            let mut cpu_row = Vec::with_capacity(data_size);
            let mut gpu_row = Vec::with_capacity(data_size);
            for col in 0..data_size {
                cpu_row.push(cpu_data[col * total + row]);
                gpu_row.push(gpu_data[col * total + row]);
            }
            assert_buffer_eq(
                &cpu_row, &gpu_row,
                &format!("{name} po2={po2} cycle={row} data row"),
            );
        }
        assert_buffer_eq(&cpu_global, &gpu_global,
            &format!("{name} po2={po2} global"));

        // Checks 3-6 (iopIdx, womIndex pre/post-zeroize, womRows post-sort):
        // TODO(R3-A06 week 2): the GPU witgen kernel does not yet populate
        // MachineContext.{iopIdx,womIndex,womRows}. Once the SYCL port lands,
        // unwrap a context snapshot from a `cpu_witgen_with_context` FFI
        // variant and assert_buffer_eq against the same. Until then these
        // four checks would tautologically pass (CPU FFI runs in both arms).
        // Keep the TODO so the first kernel-port PR cannot land without
        // wiring them up.
    }

    #[test] fn ab_test_recursion_circuit_po2_14() { cpu_vs_gpu_ab("test_recursion_circuit", 14); }
    #[test] fn ab_test_recursion_circuit_po2_16() { cpu_vs_gpu_ab("test_recursion_circuit", 16); }
    #[test] fn ab_test_recursion_circuit_po2_18() { cpu_vs_gpu_ab("test_recursion_circuit", 18); }
    #[test] fn ab_lift_po2_18()                   { cpu_vs_gpu_ab("lift_rv32im_v2_14", 18); }
    #[test] fn ab_join_po2_18()                   { cpu_vs_gpu_ab("join",              18); }
    #[test] fn ab_resolve_po2_18()                { cpu_vs_gpu_ab("resolve",           18); }
    #[test] fn ab_identity_p254_po2_21()          { cpu_vs_gpu_ab("identity_p254",     21); }
}
```

## Why this is exactly 50-100 LOC of net-new harness Rust

LOC budget:
- 4 lines: kill-switch reads in `generate_witness` (with comment).
- 4 lines: kill-switch read in `accumulate` (analogous, omitted above for brevity).
- ~95 lines: the `#[cfg(test)] mod tests` block (the bulk of this deliverable).

This sits cleanly inside R2-06's "File 4 (~80 LOC)" budget and R1-17's 100-200 total. The rest of the 315 LOC R2-06 sizes lives in `testutil.rs` (~95), `witgen.rs` (~40), `build.rs` (~5), and `tests.rs` (~75) — all already enumerated by R2-06.

## What I'm explicitly NOT changing from R2-06

1. **Probe location**: after zeroize, in `witgen.rs` — R2-06's correction stands. R1-17's "after generate_witness" position is wrong for this crate.
2. **Two kill switches**: `_WITGEN` + `_ACCUM`. The two probes hit different code paths and need independent flips during week-2 bring-up.
3. **Multi-po2 strategy**: only on `test_recursion_circuit`; real ZKRs are fixed-po2.
4. **Per-cycle scan limit**: `RISC0_RECURSION_AB_SCAN_LIMIT` env var.
5. **Noise replay**: Option A (snapshot GPU post-noise tail, replay into CPU shadow) for the in-process probe.

## What this agent adds on top of R2-06

1. **6-check schema** spelled out — explicitly enumerates which MachineContext fields the harness needs to read once the GPU witgen kernel lands. R2-06 only listed the three buffers (ctrl/data/global); the additional three (iopIdx, womIndex, womRows-post-sort) come from R1-17's failure-mode analysis but were never given names in either round's deliverable.
2. **FFI shape for context snapshots**: a `cpu_witgen_with_context` build-flag-gated FFI variant. R2-06 said "FFI surface is untouched for week 1"; this agent agrees but notes the week-2 surface needs this one extra symbol that the harness will call. Documented as a TODO in the test body so it cannot silently regress.
3. **Explicit `remove_var` of all three env vars** at the start of `cpu_vs_gpu_ab`, not just save-restore. Subtle: an outer `cargo test` invocation with `RISC0_RECURSION_GPU_WITGEN_VERIFY=1` in the env would otherwise cause the in-process probe to fire inside the test, doubling wall time and confusing failure attribution.

## Acceptance (delta from R2-06)

- [ ] All 7 AB tests in `cpu_vs_gpu_ab` pass against the current CPU-FFI stub (trivially equal — harness validated before any kernel ships).
- [ ] `cargo test -p risc0-circuit-recursion --features intel -- --test-threads=1 ab_` runs in <5 min on Battlemage.
- [ ] `RISC0_RECURSION_AB_SCAN_LIMIT=1024 cargo test ab_resolve_po2_18` completes in <2s (PR-time smoke).
- [ ] TODO comments for checks 3-6 are linted as `// TODO(R3-A06 week 2):` so the kernel-port PR cannot land without removing them (CI grep on diff: a removed TODO without a corresponding `assert_buffer_eq` call is a review-time red flag).
- [ ] Both `RISC0_DISABLE_INTEL_RECURSION_WITGEN` and `RISC0_DISABLE_INTEL_RECURSION_ACCUM` appear in `recursion-sys/build.rs` rerun-if-env-changed list.
</content>
</invoke>