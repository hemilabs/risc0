# R3-A06 Recursion Port — FINAL ROADMAP (Round-3 Agent 09 Synthesis)

**Synthesis source:** all 17 Round-1 + 17 Round-2 reports under `/tmp/recursion_round{1,2}_agent_*.md`.

**Status of plan:** PIVOT — original "3-5 week monolithic step_exec port → 20-35% E2E" framing is materially aspirational. Reality is staged 6-PR rollout, 9-14 engineer-weeks for full delivery, **15-25% E2E ceiling (10-15% conservative)**.

---

## 0. TL;DR

1. **Win ceiling — 15-25% Succinct E2E, 10-15% conservative.** R2-01's lowered 6-10% is a per-mechanism floor that excludes the non-step_exec parallel phases that R2-14/R1-06 model. The realistic mid-point is ~15%.
2. **Effort — 9-14 focused engineer-weeks** for the full step_exec port (per R2-11). R2-15's 6-PR rollout reaches ~10-12% by Week 8-10 and trends to the full ceiling around month 3+.
3. **Sequence — A→B PoC first, then step_exec.** Accum-first (R2-05) is the smallest viable PoC and de-risks the toolchain before committing 6-9 weeks to step_exec.
4. **Today's PR (PR1) — R2-03 WG knob alone.** Cross-segment pipelining (R2-07) deserves its own focused PR (PR3).
5. **NO-GO conditions — Phase B IGC failure on 15K-LOC accum amalgamation; or Phase A retiring ≥10% E2E (residual port delta no longer worth 6-9 weeks).**

---

## 1. Corrected Win Ceiling

### 1a. The R2-01 vs R2-14 contradiction

| Agent | E2E ceiling | Mechanism | Quality |
|---|---:|---|---|
| **R2-01** (FRI re-measurement) | **6-10%** | Only witgen+CPU-accum movable = 26% of per-op = 11% of E2E; if GPU kernels match CUDA, save ~9-10%; PCIe-only save = ~6%. | Rigorous empirical, but assumes only the CPU-FFI portion changes. |
| **R2-14** (PO2 correction) | **28-32%** | po2=18 (262K cycles, 16× more than plan thought) makes GPU saturation unambiguous; if Intel kernels reach CUDA-class throughput per-cycle, 9.6× per-lift speedup. | Aspirational; assumes CUDA-class throughput per kernel on Intel B70. |
| **R2-08** (per-lift breakdown) | (orthogonal) | At po2=18, eval_check is already 46% of per-lift wall time and **already on GPU (ESIMD)**. Witgen CPU-FFI is only 1.5% per lift; PCIe d2h+h2d is 7%. | Definitive on-the-box measurement (Lift seg 1: 3189 ms total, ~92% GPU, 2.5% CPU-FFI, 7% PCIe). |
| **R2-16 SCORECARD** | **15-25% (target 18%)** | Synthesis of the spread; lower bound R2-01-style ~10-12%, upper bound R2-14-style ~25-30%. | Final aggregated. |

**Why R2-01 is the floor, not the ceiling:**
R2-01 measured CPU-FFI witgen (49 ms) + CPU-FFI accum (30 ms) + the PCIe round-trips around them (228 ms) ≈ 307 ms / lift. That is the **avoided-PCIe-and-CPU-glue** path. R2-14's larger ceiling counts the full witgen+accum semantic block (~1 s/lift on CUDA — they include preflight, sort/scan, verifyWom, computeAccum, verifyAccum collectively) being able to reach CUDA-like throughput. The two are not contradictory; they are different surface areas.

**Why R2-14 is the upper bound, not the realistic target:**
- R1-16: Intel B70 is 6.2× slower than RTX 4090 on eval_check. Per-kernel CUDA parity is not the historical pattern on Intel.
- R2-11 (rv32im history): tuning tail is 6+ weeks of micro-experiments, most negative; do not bake CUDA-class throughput into the headline.
- R2-08: 92% of lift wall-time is already GPU; the un-ported sliver is small in absolute terms. Even a 10× speedup on that sliver alone caps the win below R2-14's 28-32%.

### 1b. Final position

| Confidence band | Win ceiling | Source |
|---|---:|---|
| **Conservative (high confidence)** | **10-15% Succinct E2E** | R2-16 lower bound; R2-01's avoided-PCIe-plus-glue band; aligns with most reviewers landing 6-10% on the *PR-by-PR* delivery path. |
| **Target (planning)** | **15-20% E2E** | R2-16 mid-point; assumes GPU witgen reaches ~1.5-2× CPU baseline, not CUDA parity. |
| **Optimistic ceiling** | **20-25% E2E** | R2-14 conservative scenario + chain-walk spill discount; requires Intel kernels reaching ~80% of CUDA per-cycle. |
| Aspirational only | 25-35% E2E | R2-14 best case + ESIMD on par-safe slice (R2-17 Alt-6); requires multiple Phase-D wins. Mark as stretch, not commitment. |

**Communicate as: target 15-20%, ceiling 25%, conservative floor 10-12%.** Replace plan's "20-35%" headline.

---

## 2. PR-by-PR Ordering

**Ordering rationale** (from R2-15 + corrections):

1. **Quick wins ship visible perf in days.** R2-13 retired R1-05's "skip-verify" lever (already pulled by default in pipelined `composite_to_succinct` — EV is now 0-0.4%, **drop from menu**). The remaining shippable quick wins are R2-03 (WG knob, freebie) and R2-07 (cross-segment pipelining, 6.5%). These also do not block any later work.
2. **Foundation work follows the dependency graph.** R2-04 poly_fp split unlocks `RISC0_RECURSION_OPTIMIZE=1` for any future eval_check tuning; R2-09 oneDPL sort/scan + WomRow USM scaffolding pays into the step_compute_accum and step_exec ports.
3. **PoC before full port.** R2-05 step_compute_accum PoC (15K LOC, 1 effective extern, fully parallel) validates the SYCL toolchain on recursion before committing 6-9 weeks to step_exec (40K LOC, 7 externs, par-safety gated).
4. **step_exec is the long pole.** R2-11 estimates 7-11 weeks even *after* the PoC scaffolding exists.

### PR1 — Today (Day 0): R2-03 WG knob

**Already applied** (per R3-A02 status: `/tmp/recursion_round3_agent_02.md`).

- **Scope:** `RISC0_RECURSION_EVAL_CHECK_WG` env knob in `recursion-sys/kernels/intel/eval_check.cpp:43-56`. Default 1024 (byte-identical at default). Whitelist {16,32,64,128,256,512,1024}.
- **EV:** 0-0.3% Succinct E2E (R2-03 / R1-13). Pure tuning surface, freebie.
- **Effort:** 30 min code + 3 min rebuild + 10 min sweep = 45 min total. **Done.**
- **Acceptance:** byte-identical proof at default; whitelisted WGs complete without error; invalid values fall back to 1024.

**Note on R3-04 cross-segment pipelining (R2-07 Tier 1):** Although the task brief floats bundling pipelining into PR1, R2-07 specifies this as a "1 week, ~50 LOC change to `composite_to_succinct`" and explicitly recommends it as a **standalone PR**. Bundling muddies bit-exactness validation (a regression in either change becomes harder to localize). **Defer to PR3 below.**

### PR2 — This week (Day 4-7): R2-04 poly_fp split

- **Scope:** Mechanical post-hoc split of `recursion-sys/kernels/cxx/poly_fp.cpp` (24,753 lines monolithic) into ~10 noinline sub-functions of ~2,500 lines each, distributed across `rust_poly_fp_0.cpp` + `rust_poly_fp_1.cpp` mirroring rv32im's pattern. Modify `recursion-sys/build.rs:421-565` to consume the split files; drop `-cl-opt-disable` default.
- **Pre-PR diagnostic** (1-2h): try `-O1 without -cl-opt-disable` to confirm the structural split is actually needed (vs. a flag fix). Almost certainly fails on the monolithic SSA, but cheap insurance.
- **EV:** 0% direct on E2E; **unblocks `RISC0_RECURSION_OPTIMIZE=1`** for eval_check tuning. Per R2-04 modeling, 1-3% E2E once the kernel rebuilds with -O2 -cl-intel-256-GRF-per-thread; per R2-13 / R1-13 a more conservative <1% E2E from recursion eval_check optimization. Critical-path value is "unblocks a permanently locked tuning surface", not the immediate win.
- **Effort:** 2-4 days (R2-04 estimate); bound at 5-6 days. **Empirical finding from R2-04:** at every candidate cut line, only 2-9 live `FpExt` values cross — sub-function ABI threading is small, mirroring rv32im exactly.
- **Bit-exactness gate:** byte-for-byte proof equality on a 1-segment Succinct run, both with and without `RISC0_RECURSION_OPTIMIZE=1`. Static check: every non-whitespace line of the original body appears in exactly one sub-function modulo SSA rename.
- **Off-ramp:** if mechanical split doesn't go bit-exact by Day 6, escalate to "regenerate from zirgen with split flag" (R1-11 option A). Don't bundle further work into this PR.

### PR3 — 1-2 weeks: R2-07 cross-segment pipelining (Tier 1)

- **Scope:** Hook the bg-CPU-preflight pattern (already used by rv32im SegmentProver) into recursion's `composite_to_succinct` fold (`zkvm/src/host/server/prove/mod.rs:378-396`). Use existing `prepare()` API (`host/recursion/prove/mod.rs:1013-1017`) plus existing `with_queue_override` and the `EVAL_CHECK_QUEUE` from `sys/src/intel.rs:93-156`. Spawn lift(N+1)'s `prepare()` (CPU-only preflight) on a background thread while lift(N)+join(N) GPU work runs.
- **EV:** ~6.5% Succinct E2E (R2-07 Tier 1; matches R1-12's earlier estimate). On 244 s baseline = ~16 s saved.
- **Effort:** 1 week, ~50 LOC change to one file. R2-07 confirms all infrastructure already exists for rv32im and is re-usable.
- **Bit-exactness gate:** Tier 1 produces **byte-identical seals** (pure scheduling change, no transcript change). Test by re-running the existing regression suite.
- **Risks:**
  - `SegmentReceipt::clone()` 4-8 MB × 42 segs = 200 MB alloc per fold. Mitigate with `Arc<SegmentReceipt>`.
  - `thread_local!` caches (PROVER_CACHE, MERKLE_ROOT_CACHE, CTRL_CACHE) are per-thread; bg thread only does preflight (no cache miss), so Tier 1 unaffected.
- **Why this is PR3, not PR1:** Although a quick win in calendar time, it touches the orchestrator's correctness surface (thread spawning, fixture ownership, env-var precedence with R3-02's harness from PR4) and benefits from independent CI / review focus. Bundling with the freebie WG knob (PR1) increases revert risk.

### PR4 — 2-3 weeks: R2-06 / R1-17 differential harness + R2-09 oneDPL sort/scan + WomRow USM scaffolding

**Bundle of two preparatory items:**

#### PR4a — Bit-exactness harness (R2-06 + R1-17)

- **Scope:** Verbatim port of rv32im's `EnvGuard` + `assert_check_eq` + golden mechanism to a new `recursion/src/prove/hal/testutil.rs` (~95 LOC). Add Layer-2 in-process probe to `WitnessGenerator::new`/`accum` (`witgen.rs`, ~40 LOC). Add kill switches `RISC0_DISABLE_INTEL_RECURSION_WITGEN` / `_ACCUM` to `IntelCircuitHal::generate_witness`/`accumulate` (~20 LOC). Add multi-fixture AB tests for the 7-fixture matrix (test_recursion_circuit po2 14/16/18, lift po2 18, join po2 18, resolve po2 18, identity_p254 po2 21). Add 5 `rerun-if-env-changed` lines to `build.rs`. Add Layer-3 seal-equality E2E test in `zkvm/src/host/recursion/tests.rs`.
- **EV:** 0% direct. **Critical path:** any later kernel PR (PR5+) without this harness gets cycle-level divergence localized "at the end of the run" instead of "at cycle 8723 column 47" — months of debug.
- **Effort:** 2-3 engineer-days for the verbatim-port lines, +1-2 days for the noise-replay shim (R2-06 Option A) and the 7-fixture preflight dumps. Total ~315 LOC across 6 files.
- **R2-06's corrections to R1-17:** (a) probe goes AFTER zeroize, not after generate_witness; (b) two kill switches not one; (c) noise-replay shim mandatory; (d) `RISC0_RECURSION_AB_SCAN_LIMIT` caps the per-cycle scan for sub-second smoke tests.

#### PR4b — R2-09 oneDPL sort/scan + WomRow USM (flavor B)

- **Scope:** Move `MachineContext::womRows` (2.36M × 20 B = 45 MB at po2=18) and `womIndex` (262K × 4 B) to USM device memory. Replace `std::sort(poolstl::par, womRows...)` + `std::exclusive_scan(poolstl::par, womIndex...)` + `injectWomBacks()` (recursion-sys/kernels/cxx/ffi.cpp:130-138) with `oneapi::dpl::sort` + `oneapi::dpl::exclusive_scan` + a 262K `parallel_for` for inject. New file: `recursion-sys/kernels/intel/sort_wom.cpp` (~80 LOC). build.rs adds `-I/opt/intel/oneapi/dpl/latest/include` (likely auto-included by `-fsycl`).
- **EV:** <0.5% E2E direct (poolstl already saturates 32 threads; oneDPL only wins the 30-80 ms sort, dwarfed by 99 s lift+join). **Foundation value is huge:** PR5 step_compute_accum can consume the sorted/scanned womRows on-device with no buffer-layout churn.
- **Effort:** 1 week.
- **Bit-exactness:** non-stable sort is provably safe (5-tuple key, ties only on full content equality; CUDA precedent at ffi.cu:286-312 uses unstable thrust::sort). No `stable_sort` needed.
- **Why bundle 4a+4b:** Harness must exist before any kernel kernel work; oneDPL is foundation for PR5. Both touch build.rs once. Both are pure scaffolding (~zero direct E2E win). Single-PR landing keeps the dependency clean and reduces queue depth.

### PR5 — 4-6 weeks: R2-05 step_compute_accum PoC

- **Scope:** Implement Intel SYCL kernel for `step_compute_accum` + `calcPrefixProducts` (multiplicative inclusive_scan on FpExt via oneDPL) + `step_verify_accum`. 15K LOC C++ amalgamated through new `recursion-sys/kernels/intel/step_compute_accum.h` + `ffi_compute_accum.cpp` + `build.rs::build_intel_accum`. New FFI: `risc0_circuit_recursion_intel_compute_accum`. HAL change in `intel.rs::accumulate` gates on `RISC0_INTEL_GPU_ACCUM=1` (default off; CPU fallback retained).
- **EV:** 1-2% E2E direct (recursion accum is ~5% of lift wall time per R2-08; this kernel removes the CPU-FFI ping-pong for it). **Foundation value:** validates SYCL build chain, FFI shape, USM pattern, oneDPL FpExt scan compatibility, and differential harness on the smaller-than-step_exec kernel.
- **Effort:** 5-8 working days for skeleton + bit-exact PoC, +1-1.5 weeks bit-exact debugging tail.
- **GO/NO-GO checkpoint at end of PR5:** If IGC 2.30.1 (or patched IGC at `/home/user/igc-rebuild/build/IGC/Release/`) cannot compile the 15K-LOC amalgamation in <2h, PR6 is high-risk. **NO-GO the full port; ship PR5 as opt-in, polish the Phase A wins (PR1+PR3), call the project complete.**
- **Validation gate:** PR4a harness must pass `RISC0_INTEL_GPU_ACCUM=1 cargo test diff_accum_cpu_vs_gpu` with 0 mismatches on po2=14 and po2=18 fixtures before merge.

### PR6+ — Month 3 onward: step_exec port

**This is the long pole. Phased internally:**

- **PR6 (Week 8-10):** Skeleton + single-cycle bit-exact step_exec kernel on the 40K LOC, 7-extern surface. Same scaffolding pattern as PR5 (witgen.h header with device-safe externs + amalgamation + SYCL wrapper). 7 externs (extern_log = no-op, extern_plonkWrite_wom + extern_plonkRead_wom = 4-element WOM write/read, extern_readCoefficients = TODO sentinel, extern_readIOPBody/Header = IOP tape reads, extern_womRead/Write). All gated by `RISC0_INTEL_GPU_WITGEN=1`, default off. EV at this PR: 1-2% if it lands (gated path).
- **PR7 (Week 10-12):** Parallel-for over cycles using **chain-head ownership pattern** from R2-10 (`if (cycle == 0 || isParSafeExec(cycle)) { step_exec(cycle++); while (cycle < count && !isParSafeExec(cycle)) step_exec(cycle++); }`), mirroring CUDA's `nextStepExec`. 6-check validation harness from R2-10. EV: 3-6% E2E added.
- **PR8+ (Month 3+):** Multi-PO2 sweep (14, 18, 21, 22 per R2-14 / R1-16), HAL integration removing CPU-FFI ping-pong, PO2-specific WG tuning, post-merge tuning tail. EV: 4-8% E2E added, trending to full 15-25% ceiling.

**Effort budget for PR6+ (per R2-11 history-grounded estimate):**
- W6 (helper port): 0.5 wk
- W7 (single-cycle bit-exact): 2-3 wk
- W8 (parallel-for + chain-head walk): 1.5-2 wk
- W9 (HAL + PO2 + tuning): 2-4 wk
- W10 (post-merge tuning tail rv32im-style): +2-3 wk

**Subtotal for step_exec phase: 8-12 weeks.** Combined with PR1-PR5 (1.5 + 5 + 5 + 12 + 25 = ~50 working days = 10 weeks), total **9-14 engineer-weeks for the full delivery.**

---

## 3. Effort Total

Per R2-11 (rv32im-history-grounded), with PR-by-PR accumulation:

| Phase | PRs | Calendar | Engineer-days |
|---|---|---|---:|
| Phase A: visible quick wins | PR1, PR3 | Days 0 → +14 | 1.5 + 5 = **6.5 dev-days** |
| Phase A.5: tuning unblock + scaffolding | PR2 | Days 4-7 | 2-4 dev-days |
| Phase B: PoC | PR4 (harness+oneDPL), PR5 (accum) | Week 3 → 6 | ~20 dev-days |
| **Phase C: step_exec port** | PR6, PR7 | Week 8 → 12 | ~30-50 dev-days |
| Phase D: tuning tail | PR8+ | Week 12 → 20+ | ~15-25 dev-days |
| **TOTAL** | | | **~70-105 dev-days = 9-14 dev-weeks** |

**Optimistic (5-7 weeks):** same engineer who did rv32im, no IGC surprises, no orchestrator rework. Plausible if Phase B's PR5 sails through and the patched IGC handles the step_exec amalgamation without issue.

**Realistic (9-12 weeks):** R2-11 / R2-16 default. Build in 2-3 weeks for the post-merge tuning tail (rv32im evidence).

**Pessimistic (14+ weeks):** IGC pathology on 53K-LOC step_exec.cu (the CUDA reference is 53K lines; the SYCL amalgamation may be larger), recursion-specific IGC bug (the existing 1h24m ocloc hang on `RISC0_RECURSION_OPTIMIZE=1` is a known precedent — see R2-11 §4 Scenario D), multiple bit-exact-debugging rounds.

---

## 4. EV Total

| Cumulative E2E gain | After PR # | Calendar | Source |
|---|---|---|---|
| 0-0.3% | PR1 | Day 0 | R2-03 / R1-13 |
| 0-0.3% | PR2 | Day 4-7 | R2-04 (foundation only; unblocks RISC0_RECURSION_OPTIMIZE) |
| **6.5-7% E2E** | PR3 | Day 10-17 | R2-07 Tier 1 = R1-12 reproduced |
| 6.5-7% E2E | PR4 | Day 17-25 | scaffolding only |
| **7.5-9% E2E** | PR5 | Week 6 | + step_compute_accum 1-2% |
| **10-15% E2E** | PR6+PR7 | Week 8-12 | + step_exec parallel-for ~3-6% |
| **15-25% E2E** | PR8+ | Month 3+ | + tuning tail trending to ceiling |

**Headline targets:**
- **Conservative — 10-15% Succinct E2E** (PR1 → PR7 complete; matches R2-16 lower bound + a fully-landed step_compute_accum and step_exec parallel-for).
- **Target — 15-20% Succinct E2E** (PR1 → PR8+ tuning tail; aligns with R2-16 mid-point).
- **Ceiling — 25% Succinct E2E** (PR1 → PR8+ with ESIMD on par-safe slices per R2-17 Alt-6).

**Communicate as "10-15% conservative, 15-20% target."** Avoid the original "20-35%" headline; it survives only as the aspirational best-case.

---

## 5. Risk Register

| # | Risk | Severity | Likelihood | Mitigation | First-surfaces-at |
|---|---|:-:|:-:|---|:-:|
| **R1** | IGC pathology on step_exec.cu (53K LOC; the existing 1h24m ocloc hang on `RISC0_RECURSION_OPTIMIZE` is a known precedent) | **CRITICAL** | High | Land PR2 poly_fp split first; if step_exec hits a similar hang, escalate to recursion-specific IGC patch (R2-11 §4 Scenario D) — diagnose with `IGC_PRINT_SHADER_AST=1`, bisect with -O1/-O0. | PR6 first amalgamation build |
| **R2** | Patched IGC dependency (`/home/user/igc-rebuild/build/IGC/Release/libigc.so.2`) lost or wrong version | **CRITICAL** | Medium | R2-11 L1: pre-flight check in `recursion-sys/build.rs`; LOUD `cargo:warning=` on stale-recovery (don't repeat rv32im's silent fallback); rebuild path documented (25 min wall time on 32-core box). Patch source archived. | PR4 / PR5 first SYCL build |
| **R3** | Bit-exactness landmines (6 distinct ones per R2-10: iopIdx mutation, womIndex non-atomic, sentinel writes, cross-cycle reads, sort tie-breaking, injectWomBacks ordering) | High | High | Harness from PR4 (Layer 1+2+3, 7 fixtures, AB scan with `RISC0_RECURSION_AB_SCAN_LIMIT`). Per R2-10: `q.memset(d_womRows, 0xff, ...)` mandatory, three separate kernel launches with barrier, chain-head ownership pattern. | PR5 / PR6 bring-up |
| **R4** | Chain-walk register-spill on B70 (chain-bounded step_exec parallelism per R1-06 / R2-14) | Medium | Medium | Declare `poseidon2_state` as `private`; inspect IGC asm dump for spill. Fall-back: 2-pass kernel (par-safe heads via parallel_for, then unsafe-chain walk via second launch). Discount EV 2× if spill confirmed (R2-14: 28% E2E → still positive). | PR7 parallel-for |
| **R5** | USM accumulation / SIGABRT (R2-11 §1.7; rv32im pattern at 40-60% failure rate on consecutive fib runs) | Medium | High | Pre-budget thread_local buffer-pool pass (NOT process-global Mutex; rv32im SIGABRT root cause). Use `prove_and_verify` (stable) instead of `fib` (unstable) for benchmarking. Cooldowns between runs in CI. | PR5 first multi-segment bench |
| **R6** | prove_session deadlock (R2-11 §1.2; rv32im surfaced 4 days post-merge) | Medium | Medium | Multi-segment + multi-PO2 testing (po2={14,18,21,22} × seg-count={1,5,42}) before every merge. Pattern from `project_stacked_patches_v1.md` (5×5 A/B with 25s cooldown). | PR6/PR7 first multi-seg |
| **R7** | Accum-split regression (R2-11 §1.3; rv32im reverted 1 day after merge) | Medium | Medium | Single-seg micro-bench is insufficient. Multi-segment is the gate. Refuse merge until PO2 sweep + multi-seg both green. | PR5 first multi-seg |
| **R8** | CUDA-pattern algorithms don't transfer to Intel (R2-11 §1.6; R2-A04 CUDA-style eval_u Horner kernel measured parity + SIGABRT 60% of fib runs) | Medium | Medium | Do NOT target CUDA parity (R1-16: B70 6.2× slower than 4090 on eval_check). Target 1.5-2× over current CPU baseline (= 1.2-1.6 s/lift). Stretch goal only after the parity baseline is bit-exact. | PR7 tuning |
| **R9** | T3.2-style negative source transform (R2-11 §1.5; bit-exact tree-reduce regressed 2.3× because SSA explosion blew private_size 527 KB → 3091 KB on a spill-bound kernel) | Medium | Low | Mandatory zebin metadata baseline+after diff for every source transform. >20% regression in `private_size` or `spill_size` = abort. | PR8+ tuning |
| **R10** | WG sweep doesn't transfer between kernels (R2-11 §1.8: WG=512 won on eval_check but failed on accum, hung eval_u≥128) | Low | High | Per-kernel WG sweep, not one-time. Budget 1 day × 3 kernels (witgen, accum, eval_check). Default-is-optimal is the common case (accum precedent). | PR8+ tuning |
| **R11** | Phase B PoC succeeds but Phase A already retired ≥10% E2E (PR3 pipelining ~6.5% + PR2 unblock ~1-2% + future Poseidon2/WG knobs could combine to 10-12%) — residual port delta no longer worth 6-9 weeks | Project-level | Medium | **Explicit NO-GO checkpoint at end of PR5.** Re-baseline against pre-port E2E; if residual gap < 8% E2E, ship PR5 as opt-in and stop. | End of PR5 |
| **R12** | Recursion-specific IGC bug (similar in shape to rv32im's `PreCompiledFuncImport::replaceFunc` SIGSEGV but at a different pass — R2-11 §4 Scenario D) | High | Medium | Reproduce on isolated TU; capture stack trace + IR dump; submit upstream + patch locally. ~1 day diagnosis + 1-3 day patch + rebuild (same model as the existing patch). | PR6 first amalgamation |
| **R13** | `RISC0_RECURSION_OPTIMIZE=1` 1h24m ocloc hang is from a DIFFERENT IGC pass than the rv32im one (already known — see R2-11 §4 Scenario D). Phase A can't safely pull in PR2's -O2 default without recursion-specific patching | Medium | High | PR2 keeps default OFF; only flips default after the underlying IGC issue is patched. The split itself is bit-exact at -O1 -cl-opt-disable; -O2 default is a separate flip. | PR2 first -O2 build |
| **R14** | Phase B harness false-positives on noise/zeroize (rand::rng() draws differ between CPU and GPU) | Low | High | R2-06 noise-replay shim (Option A): snapshot GPU noise window, replay into CPU shadow. Implement in PR4a — required for any non-trivial probe assertion. | PR4a first probe run |

**Risk concentration:** R1, R2, R3, R12 are all IGC + bit-exactness debt — the same axis that consumed rv32im's 6-week post-merge tail. **80% of the schedule risk lives in PR5 → PR7.** PR1-PR4 are low-risk by construction.

---

## 6. Decision Points

### DP1 — End of PR2 (Day 7-9)
**Question:** Did the poly_fp mechanical split land bit-exact, and did `RISC0_RECURSION_OPTIMIZE=1` then compile in reasonable time (<60 min)?
- **YES + YES:** Proceed; remaining schedule unchanged.
- **YES + NO (split worked, but -O2 still hangs):** Disable -O2 default; leave PR2's split as scaffolding for the eventual IGC-patched future. Don't block PR3-PR5; they don't depend on -O2.
- **NO (split didn't go bit-exact by Day 6):** Escalate to "regenerate from zirgen with split flag" (R1-11 option A). Bound at +5 days; if still red, drop PR2 from the rollout and proceed without the unblock.

### DP2 — End of PR3 (Day 17)
**Question:** Did Tier 1 pipelining land 5-7% E2E (R2-07 / R1-12 target)?
- **YES (≥5%):** Proceed. PR4-PR5 continue. **Re-baseline** the residual gap for DP4.
- **NO (<2%):** Investigate — does the workload have GPU phases too short for the bg-CPU-preflight to overlap? Or is `prepare()` cheaper than R2-07 modeled (~400 ms)? Re-measure with `RISC0_VERBOSE=1` per-phase breakdown. If structural (e.g., GPU work is too long to mask preflight, which would mean GPU is over-provisioned), the port delta will be smaller too — re-baseline EV expectations downward.
- **Marginal (2-5%):** Proceed. Adjust Phase A residual band downward for DP4 calculation.

### DP3 — End of PR5 (Week 6)
**The single biggest GO/NO-GO.**
**Question 1:** Did IGC 2.30.1 (or patched IGC) compile the 15K-LOC accum amalgamation in <2h?
**Question 2:** Did the bit-exactness harness from PR4a pass `RISC0_INTEL_GPU_ACCUM=1 cargo test diff_accum_cpu_vs_gpu` on po2=14 + po2=18 with 0 mismatches?
- **YES + YES:** Proceed to PR6+. The toolchain is validated; the harder kernel (step_exec, 40K LOC) inherits the same scaffolding.
- **NO on Q1 (IGC fails or hangs):** **NO-GO step_exec.** Ship PR5 as opt-in (CPU fallback remains default); polish Phase A wins (PR1+PR3); call the project complete at ~7-9% E2E.
- **NO on Q2 (bit-exact fails):** Bound at +2 weeks (R2-11 budget for bit-exact tail on the smaller kernel). If still red, NO-GO step_exec — the larger 40K LOC will be strictly harder. Same exit as IGC fail.
- **Aux check (DP3.5): re-measure residual gap.** If Phase A (PR1+PR2+PR3+PR5 step_compute_accum) has already retired ≥10% E2E, the residual port delta (step_exec specifically, ~3-6% E2E) may not be worth 6-9 more weeks. **Soft NO-GO on PR6+** in that case — promote PR5 to default-on (gated re-enable), call project complete.

### DP4 — Mid-PR7 (Week 10-11)
**Question:** Did step_exec single-cycle bit-exact land within R2-11's 2-3 week budget?
- **YES (≤3 weeks):** Proceed to parallel-for + WomRow GPU residency. On schedule for 15-25% E2E.
- **NO (>4 weeks):** Bounded escalation — re-scope to "PR7 lands the gated path with manageable correctness debt, defer parallel-for to PR8+." Communicate slip to stakeholders.

### DP5 — End of PR7 (Week 12)
**Question:** Did the GPU port land bit-exact + perf-positive on po2=14 + po2=18?
- **YES + positive:** Tuning tail (PR8+) proceeds, +2-4 weeks. Target full 15-25% E2E.
- **YES + neutral/negative perf:** Investigate. Likely chain-walk spill (R4) or WG-mistuning (R10). Allocate 1-2 weeks for diagnosis; if no path to positive within 3 weeks, ship gated, declare project complete at the Phase A + step_compute_accum bands (7-9%).
- **NO bit-exact:** Same exit as DP3 NO on Q2.

### DP6 — Anytime: project re-baseline triggers
**Re-baseline triggers** (any of):
1. A higher-priority Intel optimization surfaces (the memory landscape includes Poseidon2 OpenCL default, witgen/accum WG sweeps, etc.).
2. Phase A retires ≥12% E2E (the residual delta is no longer 6-9 weeks of value).
3. The recursion circuit changes upstream (e.g., zirgen regenerate, new ZKR program); the generated step_exec.cpp would need re-porting.
4. patched IGC becomes unmaintainable (next risc0 version requires features not in the patched fork).

In any of those: pause; re-cost; possibly pivot.

---

## 7. What Was Dropped from the Original Plan

- **"Skip-verify intermediates" (R1-05 QW2):** Removed. R2-13 verified the optimized `composite_to_succinct` already bypasses intermediate verifies (it calls `lift_with_opts`/`join_with_opts` directly, not `self.lift()`/`self.join()`). EV is 0-0.4%, not 1.5-6%.
- **"Pipeline gives 2.5× on HIP" framing:** Replaced. R2-07 Tier 1 = ~6.5% on single-GPU Intel; full deferred-finalize parity with rv32im (Tier 3) would yield ~20% but architecturally conflicts with the GPU witgen port itself (post-port, the CPU phases are gone, so there's nothing to overlap).
- **"po2=14, 16K cycles" assumption:** Replaced. R2-14 verified `RECURSION_PO2 = 18` (262K cycles) uniformly across lift, join, resolve, identity, union, and all `*_povw` variants. Plan was confusing `MIN_LIFT_PO2 = 14` (the inner rv32im segment) with the recursion harness size.
- **"FRI 70% of per-op":** Replaced. R2-01 verified real FRI is <1%; the `[recursion_prove] fri=X` log line is misleadingly named (it measures the entire `prover.finalize()` including eval_check, NTT, merkle).
- **"3.4× slower than CUDA, 2.4 s/lift" framing:** Replaced. R1-01 verified per-lift = 1.16 s, per-join = 1.24 s, 42+41 = 83 ops, 99.7 s total. CUDA reference at 0.7 s/op = 1.6-1.8× faster (not 3.4×).
- **"Risk #4: extern_getMemoryTxn":** Removed. R1-15/R1-16: that extern doesn't exist in recursion (it was inadvertently described as an rv32im risk). Recursion has 7 externs, 4 effective no-ops, 3 simple read-mutate-state — 80 LOC total.
- **"WomRow sort needs custom radix":** Removed. R1-14/R2-09: oneDPL ships drop-in, 2.36M rows in ~5-15 ms on Xe2. The custom-radix worry was overstated.

---

## 8. Files referenced

**Critical kernels / sources:**
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion-sys/kernels/intel/eval_check.cpp:43-56` — PR1 patch (already applied)
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion-sys/kernels/cxx/poly_fp.cpp` — PR2 target (24,753 lines monolithic)
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion-sys/kernels/cxx/step_compute_accum.cpp` — PR5 target (15,591 lines)
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion-sys/kernels/cxx/step_exec.cpp` — PR6/PR7 target (40,004 lines)
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion-sys/kernels/cxx/ffi.cpp:60-138` — PR3 (parStepExec chain walk) + PR4b (sort/scan call site)
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion-sys/kernels/cxx/extern.cpp:38-156` — 7 externs reference for PR5/PR6
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion-sys/build.rs:421-565` — PR2 / PR4 / PR5 build infra

**Reference (rv32im patterns to mirror):**
- `/home/user/risc0-intel/risc0/risc0/circuit/rv32im-sys/build.rs:1212-1304` — split mechanism for PR2
- `/home/user/risc0-intel/risc0/risc0/circuit/rv32im-sys/kernels/intel/witgen.h` (299 lines) — scaffolding reference for PR5/PR6
- `/home/user/risc0-intel/risc0/risc0/circuit/rv32im-sys/kernels/intel/ffi_witgen.cpp:374-530` — extern + dispatch reference
- `/home/user/risc0-intel/risc0/risc0/circuit/rv32im-sys/kernels/intel/eval_check.cpp:32-39` — WG env-knob reference for PR1
- `/home/user/risc0-intel/risc0/risc0/circuit/rv32im/src/prove/hal/intel.rs:392-733` — harness reference for PR4a

**HAL integration / Rust side:**
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion/src/prove/hal/intel.rs:60-128` — PR4a kill-switches + PR5/PR6 HAL changes
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion/src/prove/witgen.rs:124-138, 167-182` — PR4a Layer-2 probe site
- `/home/user/risc0-intel/risc0/risc0/zkvm/src/host/recursion/prove/mod.rs:61, 1013-1017` — RECURSION_PO2=18 + prepare() API
- `/home/user/risc0-intel/risc0/risc0/zkvm/src/host/server/prove/prover_impl.rs:1191-1251` — PR3 composite_to_succinct fold site
- `/home/user/risc0-intel/risc0/risc0/sys/src/intel.rs:93-156` — PR3 dual-queue + with_queue_override

**Patched IGC:**
- `/home/user/igc-rebuild/build/IGC/Release/libigc.so.2` — required for PR4+ SYCL builds (R2-11 L1)
- `inteldebug/igc-bug-report/0001-PreCompiledFuncImport-fix-nullptr-arg-push-in-replac.patch` — source patch

**Source memo files (R2 inputs):**
- `/tmp/recursion_round1_agent_{01..17}.md`
- `/tmp/recursion_round2_agent_{01..17}.md` (R2-16 is the cross-cutting SCORECARD synthesis)

---

## 9. Sign-off summary

**GO** on PR1 unconditionally (already applied; freebie).
**GO** on PR2 with 5-day bound + escalation off-ramp.
**GO** on PR3 — independent value, low risk, ~50 LOC.
**GO** on PR4 — required for any kernel work; harness + oneDPL scaffolding bundle.
**CONDITIONAL GO** on PR5 — explicit checkpoint at end (DP3).
**CONDITIONAL GO** on PR6+ — depends on DP3 + DP5 outcomes.

**Headline metrics:**
- Effort: **9-14 engineer-weeks** for full delivery.
- EV: **10-15% conservative, 15-20% target, 25% ceiling** Succinct E2E.
- Calendar: **Months 0 → 3+**, with visible value at each PR (PR3 delivers ~6.5% within 2 weeks).
- Risk concentration: **PR5 → PR7 (Phase B/C)**; PR1-PR4 are low-risk.

**Single biggest decision: DP3** (end of PR5). The PoC's IGC + bit-exact outcome dictates whether the full step_exec port commits 6-9 more weeks or whether the project ships at the ~7-9% E2E Phase A+B band.
