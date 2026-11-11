# R3-A06 Round-3 Agent 14 — Open Questions After 34 Reports

Synthesizes what 17 R1 + 17 R2 reviews CANNOT answer from analysis alone, plus the cheapest experiment for each.

The reviews converged on most numbers (e.g. `RECURSION_PO2=18`, 99 s lift+join, eval_check 46% of a lift, accum-first ordering, harness-first sequencing). What follows is everything they explicitly left unmeasured or speculated, sorted by blast radius.

---

## Tier-1 open questions — must answer BEFORE committing to the full port

### Q1. Does IGC compile the 15K-LOC `step_compute_accum.cpp` amalgamation? (UNTESTED)

**State of knowledge.** R1-11 / R2-04 / R2-11 establish that `RISC0_RECURSION_OPTIMIZE=1` on the 24K-LOC `poly_fp.cpp` hangs ocloc for 1h24m with no result (one-function monolithic SSA hits IGC's O(N²) pre-RA scheduler). R1-04, R1-16 flag this as the single biggest risk for the bigger 40K-LOC `step_exec.cpp`. **No reviewer has actually tried compiling either `step_compute_accum.cpp` (15K LOC) or `step_exec.cpp` (40K LOC) under SYCL/icpx yet.**

If `step_compute_accum.cpp` hangs even at `-O1 -cl-opt-disable`, the entire R3-A06 plan must restructure around `poly_fp.cpp`'s split-first-then-port approach — adding 2-4 days minimum per kernel.

**Cheapest experiment (≤ 4 hours wall, mostly unattended):**
1. Copy `step_compute_accum.cpp` verbatim into a stub amalgamation under `recursion-sys/kernels/intel/_probe_accum/`, mirroring rv32im's `kernels/intel/ffi_witgen.cpp` wrap pattern. No real kernel, just a `q.parallel_for(item)` that calls `step_compute_accum(ctx, steps, item.get_global_id(0), args)`.
2. Provide empty device stubs for the single extern (`extern_plonkWriteAccum_wom` → no-op or write to a USM `accum_ext[]`).
3. Build with three flag sets, bounded 90 min each:
   - `-O1 -cl-opt-disable` (recursion eval_check's current setting; the safe baseline)
   - `-O2` (no `-cl-intel-256-GRF-per-thread`)
   - `-O2 -cl-intel-256-GRF-per-thread` (rv32im's setting)
4. Record: wall time, ocloc exit code, zebin `private_size` / `spill_size` / `grf_count`.

**Decision gate.** If any of (1)/(2) compile in <30 min, the accum PoC path is unblocked — proceed. If all three hang or produce >256 KB scratch, fall back to splitting `step_compute_accum.cpp` (R2-04's mechanical-split scaffolding can be adapted; ~2-4 days extra). If `-O1 -cl-opt-disable` itself hangs, the entire plan is in trouble — re-scope to "host-side oneDPL sort/scan + Phase A quick wins only".

### Q2. What's the actual GPU eval_check time in recursion? (ESTIMATED, not isolated)

**State of knowledge.** R1-13 estimated 5-15 ms per lift (extrapolating from rv32im's 2.7 s/seg at PO2=20 → 64K-domain units → recursion at PO2=14 65536-domain). R2-08 ACTUALLY MEASURED IT and found **1479.6 ms** for one lift on Intel B70 — **roughly 100× R1-13's estimate**. The discrepancy is because R1-13 used the wrong domain (PO2=14 = 65536 cycles) while recursion really runs at PO2=18 = 262144 cycles, and PO2=18 inflates eval_check's 4× domain to 1048576 cells, dwarfed further by recursion's `-O1 -cl-opt-disable` which leaves register pressure poor.

But R2-08 ran ONE 4-lift configuration. We don't know:
- Per-lift eval_check time variance across the 42-lift Succinct chain.
- Whether each ZKR program (lift_v2_14 / join / resolve / identity_p254) has different eval_check times.
- Whether the time changes after the `RISC0_RECURSION_EVAL_CHECK_WG` knob is swept (R3-A06 #02 already applied the env knob; nobody has run the sweep).

R2-08's 46.4%-of-lift number is *the* load-bearing fact for whether the rest of R3-A06 matters: if eval_check dominates and is unmoveable, witgen+accum porting saves at most ~13% E2E even at 100× speedup.

**Cheapest experiment (≤ 30 min wall):**
1. With the eval_check WG knob already applied (per `/tmp/recursion_round3_agent_02.md`), run:
   ```
   for WG in 256 512 1024; do
     RISC0_RECURSION_EVAL_CHECK_WG=$WG RISC0_VERBOSE=1 \
       ./target/release/examples/prove_and_verify 100000 succinct 2>&1 \
       | grep -E "start_finalize|finalize " > /tmp/wg_${WG}.log
   done
   ```
2. Compare per-lift `eval_check_launch=` median, P95, sum across all 42 lifts.
3. Sanity: each WG should produce identical seal bytes (RECURSION_PO2=18 eval_check is bit-exact).

**Decision gate.** If sweep moves eval_check by >5% E2E (i.e. >5 s total), the win lands without any port work — promote WG sweep to PR #1. If sweep is ≤1% (mirrors the R2-03 prediction), reconfirm that step_compute_accum / step_exec are the only meaningful levers and proceed with accum-first PoC.

### Q3. Is the 9.6× per-lift parallelism (R2-14) achievable or just theoretical?

**State of knowledge.** R2-14 computed a 9.6× per-lift speedup at PO2=18 by extrapolating from the parallelism model: 39,322 par-safe heads × B70's ~8192 lane saturation. R1-06 independently computed ~4.2× per-lift using a chain-bounded model. R2-08 measured the actual baseline (3189 ms per lift on Intel B70 = 1480 ms eval_check + 1151 ms data_commit + 189 ms witgen FFI + 141 ms accum FFI + ...) — the witgen+accum CPU FFI is only **330 ms / 3189 ms = 10.4% of a lift**. Best-case 100× witgen+accum speedup saves 330 ms × 42 = 13.9 s = **5.6% Succinct E2E**, not the 30%+ R2-14 projects.

R2-14's calculation didn't account for R2-08's measurement that data_commit (1151 ms) + eval_check (1480 ms) + accum_commit (50 ms) = **2681 ms of GPU work** that already exists and that the port doesn't touch. R2-14 also used R1-06's 2.4 s/lift estimate, which is from CUDA reference, not from B70.

**Cheapest experiment (≤ 4 hours wall):**
1. Run `prove_and_verify 100000 succinct` with `RISC0_DISABLE_INTEL_RECURSION_WITGEN=1` (the env knob R1-17 specifies as part of the harness) — but this knob doesn't exist yet; gate is item Q9 below.
2. Alternative: ablate by **skipping the CPU witgen FFI entirely** for one experimental run (will produce a wrong seal; only used to upper-bound speedup). Compare lift wall vs baseline.
3. Better alternative: capture R2-08's detailed timing for **all 42 lifts** (not just lift 1) and re-aggregate. The 1480 ms eval_check could be 1× lift0 (cold) or could be the steady state.

**Decision gate.** The honest upper bound for R3-A06's gain — after accounting for what's already on GPU — is: `(witgen_d2h + witgen_h2d + witgen_cpu + accum_d2h + accum_cpu + accum_h2d) × 42 lifts / 244 s prove`. From R2-08 that's `(105+21+49+100+30+2) ms × 42 / 244 s = ~5.4% Succinct E2E`. **Until someone measures this against the full chain, R2-14's 30% number is unsupported**, and the plan's 20-35% range is at risk.

### Q4. For step_compute_accum specifically: how fast IS the GPU version vs CPU's 29.7 ms? (UNMEASURED)

**State of knowledge.** R2-08 measured CPU accum FFI at **29.7 ms** per lift (not 141 ms — that's the wall including d2h+h2d). R1-03 estimated GPU accum at "5-10 ms/lift × 42 = 2-4 s saved" but never built a kernel. R2-14 estimated "computeAccum ~80 ms" GPU at PO2=18 — counterintuitively *slower* than CPU.

The CPU is winning by being multithreaded on a 32-core box with a 15K-LOC straight-line SSA function. GPU can win only if:
- Per-cycle work fits in registers without spilling to scratch.
- The single extern (`extern_plonkWriteAccum_wom`) is inlined as a USM write.
- `calcPrefixProducts` (multiplicative inclusive_scan on FpExt) runs in <5 ms (oneDPL).

None of this has been measured. Until it is, the accum port could land at parity or slower — see R2-A04 / `project_stacked_patches_v1.md` for prior art where a "guaranteed win" landed at parity + SIGABRT.

**Cheapest experiment (≤ 1 day wall, builds Q1's stub PoC further):**
1. Once Q1's amalgamation compiles, wrap it in a real SYCL kernel that writes to a USM `accum_ext[]` buffer.
2. Add the oneDPL `inclusive_scan` with FpExt multiplicative monoid (identity FpExt(1,0,0,0)) for `calcPrefixProducts`.
3. Compare wall vs CPU: measure `accum FFI total - (d2h+h2d)` on CPU vs the GPU kernel wall (no D2H needed; buffers already on device).
4. Sweep WG sizes 64 / 128 / 256 (per `project_accum_wg_sweep.md`, default 256 was optimal for rv32im accum — likely also true for recursion).

**Decision gate.** If GPU accum is ≥1.5× faster than CPU's 30 ms (i.e. ≤20 ms), the accum port is worth pursuing. If GPU is at parity or slower (≥30 ms), the **plan's accum-first sequencing loses its primary justification** — the PoC is no longer "easier kernel proves win earlier"; it becomes "easier kernel proves loss earlier". Either way, this measurement saves weeks downstream.

### Q5. Is there a newer IGC that fixes the 24K-line `poly_fp.cpp` hang without splitting?

**State of knowledge.** R1-11 documents stock IGC 2.30.1 hanging 1h24m on the `-O2 -cl-intel-256-GRF-per-thread` path. R2-11 cites `project_tier4_igc_flags.md` (NF7 flags from Agent B aren't recognized by 2.30.1). The patched IGC at `/home/user/igc-rebuild/build/IGC/Release/libigc.so.2.30.0+0` fixes the witgen amalgamation SIGSEGV but R1-11 specifically notes "PreCompiledFuncImport fix is necessary but not sufficient" — the recursion hang is a different pathology.

Whether a newer IGC (≥2.31) ships scheduler improvements that handle one-function SSA blobs is **unknown to any reviewer**. The MEMORY's `project_tier4_igc_flags.md` notes that NF7's IGC env vars (`IGC_VISAPreSchedRPThreshold`, etc.) are post-2.30.1 features.

**Cheapest experiment (≤ 1 day wall, mostly download/build):**
1. Check Intel's `intel/intel-graphics-compiler` GitHub for any 2.31+ tag. If `intel/compute-runtime` ships pre-built debs for ≥25.x, install side-by-side.
2. If only source: `git clone intel/intel-graphics-compiler` at HEAD, build `libigc.so.2` (~30 min on 32-core), `LD_LIBRARY_PATH` it.
3. Re-run the failed `RISC0_RECURSION_OPTIMIZE=1` build, bounded 2 hours wall.
4. If success: zebin diff vs `-O1 -cl-opt-disable` baseline (`private_size`, `spill_size`, `grf_count`).

**Decision gate.** If a newer IGC compiles `poly_fp.cpp` at `-O2 -cl-intel-256-GRF-per-thread` in <30 min, the **C (poly_fp split) PR can be dropped** from R2-15's stacked plan, saving 2-4 days. If it still hangs, the split is mandatory and budget stays as Round-1 Agent 16 said.

---

## Tier-2 open questions — answer DURING the Phase A quick wins

### Q6. Does the differential harness (R1-17 / R2-06) need any features beyond rv32im's?

**State of knowledge.** R2-06 provides the concrete diff and identifies one correction to R1-17 (probe must run AFTER noise/zeroize, not before). Three recursion-specific failure modes flagged but unverified:
- **Multiple ZKR programs** (lift_v2_14, join, resolve, identity, union, +povw variants) — need per-program fixtures.
- **par-safety stress** — preflight where >50% cycles are not par-safe (recursion's natural state per R1-06).
- **WomRow sort determinism** — verify CPU+GPU produce byte-equal sort output.

Beyond those, R2-06's harness is verbatim port of rv32im's `intel.rs:392-733`. But rv32im's harness only tests `eval_check`, not `witgen` or `accum`. Whether `assert_buffer_eq` correctly identifies the FIRST mismatching cycle (vs e.g. the first mismatching ROW of a multi-row cycle) at PO2=18 (262K rows × 128 cols data = 33 M cells) is unknown — at this scale, naive `iter().zip().position()` is fine (~50 ms scan) but a CYCLE-level diff would need additional plumbing.

R1-17 already specifies cycle-level diff at Layer 1: `for row in 0..total { assert_buffer_eq(&d_cpu[lo..hi], &d_gpu[lo..hi], &format!("...cycle={row}...")) }`. So this is already designed; it's the SECOND-order question of "is one fixture enough, or do we need 5 fixtures × 3 PO2s = 15 unit tests in CI" that's open.

**Cheapest experiment (≤ 2 days wall, part of harness landing):**
1. Land R2-06's diff verbatim.
2. Run all 6 fixtures (lift_v2_14/18, join, resolve, identity_p254 at PO2=21, union if covered) against current CPU FFI (CPU-vs-CPU, trivially passes).
3. Confirm each runs in <30 s.
4. If any takes >2 min, slim the harness — likely union at PO2=22 won't fit in CI budget; restrict to lift_v2_14 + join at PO2=18 for fast CI and run the full grid nightly.

**Decision gate.** Harness landing is precondition for ANY kernel work; this isn't really an open question, just a sizing question. The answer is "do exactly what R2-06 specs and add the 3 recursion-specific assertions from R2-10 (WomRow sort byte-equal, womIndex post-scan byte-equal, par-safety stress)".

### Q7. Will the WomRow sort actually produce byte-equal output across CPU `std::sort(poolstl::par)` and `oneapi::dpl::sort`?

**State of knowledge.** R1-14 / R1-07 / R2-09 / R2-10 establish:
- The 5-tuple key uniquely identifies all distinct rows (key = whole payload + sentinel).
- True ties only occur on byte-identical rows → `injectWomBacks` reads `womRows[idx-1]` and gets identical content regardless of tie order.
- CPU `std::sort(poolstl::par, ...)` is non-stable.
- oneDPL ships `sort` (non-stable).
- The CUDA port uses `thrust::sort` (non-stable) and is byte-identical to CPU.

But "byte-identical to CPU on CUDA" is empirical evidence FOR CUDA, not for oneDPL on Intel. The R2-10 risk-R6 mitigation requires a byte-equal post-sort check between CPU and GPU. If they diverge — e.g. because oneDPL's internal merge sort produces a different tie-break order under partial keys that LOOK distinct under `kInvalidPattern` sentinel rows — the whole witgen port loses bit-exactness.

**Cheapest experiment (≤ 4 hours wall):**
1. Add a minimal SYCL probe to `recursion-sys/kernels/cxx/ffi.cpp::verifyWom` (gated by env var):
   ```cpp
   if (getenv("RISC0_VERIFY_GPU_SORT")) {
       // sort once on CPU, snapshot bytes
       std::vector<WomArgumentRow> cpu_sorted = womRows;
       std::sort(poolstl::par, cpu_sorted.begin(), cpu_sorted.end());
       // sort same input on GPU via oneDPL
       auto policy = oneapi::dpl::execution::make_device_policy(*q);
       sycl::malloc_device + memcpy + oneapi::dpl::sort(policy, ...) + memcpy back
       // byte-compare
       if (memcmp(cpu_sorted.data(), gpu_sorted.data(), sz)) abort();
   }
   ```
2. Run on a single lift_v2_14 preflight. If byte-equal: foundation work in R2-15 PR #3 (D — oneDPL substitution) is safe.

**Decision gate.** If divergent, either (a) use `stable_sort` on both sides to canonicalize ordering, OR (b) extend the key to break all ties deterministically (CUDA's strategy is the secondary identifier). This is a 1-day correctness fix that's MUCH cheaper to discover NOW than during the integration tail.

### Q8. Is the patched IGC sufficient for an `step_exec.cpp` amalgamation of 40K LOC?

**State of knowledge.** R2-11 documents the patched IGC fixes `PreCompiledFuncImport::replaceFunc` SIGSEGV (rv32im witgen amalgamation). The recursion port produces a **second** witgen amalgamation, of LARGER size (40K LOC step_exec + 15K LOC accum = ~55K LOC after wrapping). Whether the same SIGSEGV path is hit is unknown — `PreCompiledFuncImport` is invoked on any kernel using vendor intrinsics, and the recursion step_exec uses the same FpExt intrinsics. **If the bug surfaces but the patched IGC doesn't cover the new amalgamation's specific code path**, we're back to "rebuild IGC with a SECOND patch".

R2-11 Scenario D explicitly flags this: "the recursion amalgamation hits a DIFFERENT IGC bug". The 1h24m `RISC0_RECURSION_OPTIMIZE=1` hang IS that different bug (different from PreCompiledFuncImport SIGSEGV). Two distinct IGC pathologies are now in scope.

**Cheapest experiment (≤ 1 day wall, depends on Q1):**
1. Once Q1's `step_compute_accum.cpp` stub PoC builds, scale up by **fusing in** `step_exec.cpp` (or, more conservatively, build it as a separate `.so` to isolate variables).
2. Run with `LD_LIBRARY_PATH=/home/user/igc-rebuild/build/IGC/Release/:$LD_LIBRARY_PATH` (the patched IGC) and again WITHOUT (stock 2.30.1).
3. Capture exit code, wall time, kernel symbols loaded.

**Decision gate.** If patched IGC builds it, the port can proceed; document the patched binary as a Week-0 prerequisite. If patched IGC hangs (different bug), open Q1's split path AND check whether the patch is reusable.

### Q9. Does R3-A06's plan need `RISC0_DISABLE_INTEL_RECURSION_WITGEN=1` kill switch RIGHT NOW (Phase A) or only after kernel exists?

**State of knowledge.** R1-17 / R2-06 both specify this env var, but it has no purpose until there's an actual GPU witgen kernel to disable. R2-13 measured intermediate verify cost at ~10-13 ms / verify (not 50-200 ms as R1-05 estimated), and confirmed `composite_to_succinct` already bypasses them. The "1.5-6% Succinct E2E" from default-off skip-verify is therefore **0.4-0.5%** in practice — R1-05's main quick win is effectively already implemented.

This means R2-15's PR #1 (Bundle A+B, "land today, 1.5-6% E2E win") is mostly already done. The remaining Phase A delta is just A (the eval_check WG knob, already applied per `/tmp/recursion_round3_agent_02.md`).

**Cheapest experiment (≤ 1 hour wall):**
1. Run `prove_and_verify 100000 succinct` with `RISC0_VERBOSE=1` on current main.
2. grep for `[lift]` lines — confirm none mention `verify=` (i.e. composite-to-succinct already bypasses verify).
3. Time a run with `RISC0_SKIP_INTERMEDIATE_VERIFY=1` vs unset. If delta is <1 s, R2-13 stands: the "lift+join intermediates" wave has already crashed.

**Decision gate.** If confirmed already-bypassed, R2-15 PR #1 simplifies to just the A knob (sub-1% E2E ceiling) and the headline Phase A gain shrinks accordingly. **Then the path forward becomes "do Phase B accum-first PoC unless Q1-Q5 say no"** — Phase A's residual gain is too small to be the stopping point.

---

## Tier-3 open questions — known unknowns, lower priority

### Q10. What is the actual par-safe fraction in lift_rv32im_v2_X programs (R2-17 §"par-safe audit")?

R1-06 estimated 10-30% par-safe based on op-class fractions; R2-17 §"par-safe audit" suggested a 1-day measurement to gate the port. Nobody has run it. Mechanism is trivial — drop a counter into `parStepExec` and run one lift.

**Cheapest experiment (≤ 30 min):** Add the 8-line counter R2-17 specifies to `ffi.cpp::parStepExec`, rebuild, run one Succinct prove, print per-ZKR par-safe ratios. Resolves R1-06 vs R2-14 disagreement on whether 10-30% (chain-bound) or 85% (saturation-bound) is right.

### Q11. Does USM shared make any difference on B70 for recursion buffers? (R1-10 said "barely")

R1-10 measured USM shared as ~1% E2E for accum's small buffers. But that's at PO2=14 assumed cycles; at PO2=18 actual, the buffers are 16× bigger. R1-10's "below the noise floor" claim deserves a re-measurement at production size.

**Cheapest experiment (≤ 2 hours):** Patch `recursion/src/prove/hal/intel.rs` to swap `malloc_device → malloc_shared` for ctrl/global/data/mix/accum (5-line diff). Run 42-seg fib. Compare wall.

### Q12. Cross-segment pipelining (R2-07 "6.5% E2E") — actual or theoretical?

R2-07 designed a 2-thread architecture for lift(N+1) CPU overlap with lift(N) GPU. R2-08 measured the actual GPU/CPU split per lift (GPU 92%, CPU FFI 2.5%, PCIe 7.1%). With CPU only 2.5% per lift, the overlap window is **~80 ms per lift × 41 overlaps = 3.3 s = 1.3% Succinct E2E**, not the 6.5% R1-12 estimated.

**Cheapest experiment (≤ 1 day):** Implement R2-07's 2-thread sketch on the existing CPU FFI path (no new SYCL kernels needed). Compare wall.

**Decision gate.** If <2% gain, drop it — not worth the queue-management complexity. If >4%, fold into PR #1.

### Q13. Does the recursion-sys build emit `cargo:rerun-if-env-changed` for the new R3-A06 env vars?

R2-03 confirms the eval_check WG knob doesn't need rerun-if-env-changed (env read at runtime, not build). But R1-17 / R2-06's `RISC0_DISABLE_INTEL_RECURSION_WITGEN` and `RISC0_RECURSION_GPU_WITGEN_VERIFY` are also runtime knobs — so likewise no rerun needed. Confirmed but not verified by reviewers.

**Cheapest experiment (≤ 5 min):** Read `recursion-sys/build.rs` lines around 425, 305. Confirm pattern matches the rerun-if-env-changed precedent for `RISC0_RECURSION_OPTIMIZE`.

### Q14. Will the rv32im `BufferObj`/`witgen.h` scaffolding port verbatim to recursion?

R1-02 / R2-15 both assume "yes" — recursion's `Fp** args` indexing is actually simpler than rv32im's `BoundLayout<T>` discriminated union. R1-15 confirms 8 effective extern signatures map 1:1 (or simpler than) rv32im's 12+. But "verbatim port" isn't proven until someone tries.

**Cheapest experiment** — bundled with Q1 (the accum stub PoC necessarily exercises this).

### Q15. Are the patched-IGC stamp files at risk of staleness across the dual amalgamation (eval_check + witgen)?

R2-11 §1.1 documents rv32im's `try_recover_witgen_so()` silently copying stale .so. Recursion will now have THREE .so files: `librisc0_recursion_intel_eval_check.so` (exists), `librisc0_recursion_intel_witgen.so` (new), `librisc0_recursion_intel_accum.so` (new from R2-05). All share an IGC binary. If the build script's stamp logic doesn't account for IGC binary hash, an IGC upgrade silently uses outdated kernels.

**Cheapest experiment (≤ 1 hour):** Audit recursion-sys/build.rs's stamp-write logic for `sha256(libigc.so.2)` inclusion. Mirror rv32im's R2-A08-era stamp pattern if missing. This is a hardening pass, not a discovery.

---

## Summary table — open questions ranked by blast radius

| # | Question | Tier | Wall to answer | Decision gate |
|---|---|:-:|:-:|---|
| Q1 | Does IGC compile step_compute_accum.cpp 15K? | 1 | 4 h | If hang → split-first path; if compile → unblock |
| Q2 | What's real eval_check time across all 42 lifts? | 1 | 30 min | If WG sweep wins >5% → promote to PR #1 |
| Q3 | Is 9.6× per-lift actually 5.6%, not 30%, E2E? | 1 | 4 h | If <10% E2E → re-scope plan or focus on eval_check |
| Q4 | Is GPU accum faster than CPU's 30 ms? | 1 | 1 d | If not → kill accum-first sequencing |
| Q5 | Newer IGC unblocks RISC0_RECURSION_OPTIMIZE? | 1 | 1 d | If yes → drop split PR |
| Q6 | Does R2-06 harness cover all recursion failure modes? | 2 | 2 d | Land harness w/ recursion-specific assertions |
| Q7 | oneDPL sort byte-equal to CPU std::sort? | 2 | 4 h | If not → stable_sort or extend key |
| Q8 | Patched IGC sufficient for 40K step_exec amalg? | 2 | 1 d | If not → second patch round |
| Q9 | Is R1-05's intermediate-verify win already realized? | 2 | 1 h | Re-scope Phase A if yes |
| Q10 | Actual par-safe fraction in lift programs? | 3 | 30 min | Sizes the chain-vs-saturation argument |
| Q11 | USM shared at PO2=18 — still negligible? | 3 | 2 h | If <1% E2E → drop; else fold in |
| Q12 | Cross-segment pipelining actually 6.5% or 1.3%? | 3 | 1 d | If <2% → drop |
| Q13 | rerun-if-env-changed plumbing for new vars? | 3 | 5 min | Trivial hygiene |
| Q14 | Does rv32im witgen.h scaffolding port verbatim? | 3 | (Q1) | Folded into Q1 |
| Q15 | IGC binary hash in build.rs stamp? | 3 | 1 h | Hardening; do before merge |

---

## The bottom line

**Five Tier-1 questions can be answered in 2 working days TOTAL** (Q2 30 min + Q1 4 h + Q3 4 h + Q5 1 d + Q4 1 d, with Q1 prerequisite to Q4). That budget would convert the plan's review-by-analysis confidence into measurement-backed confidence — and several of the Tier-1 answers (especially Q3 if it confirms ≤6% E2E ceiling, or Q5 if a new IGC version unblocks `poly_fp.cpp`) can REPLACE the entire 7-11 week port plan with a 1-2 day patch.

The 34 R1+R2 reports converged on what to build. The most important open question they LEFT is whether the building is worth it on B70, where the measured per-lift breakdown (R2-08) shows GPU work is already 92% of a lift. **Until Q3 is measured, the plan is operating on an assumed-CPU-bound regime that R2-08 contradicts.**
