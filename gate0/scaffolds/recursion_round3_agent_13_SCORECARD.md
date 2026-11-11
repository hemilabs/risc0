# R3-A06 Recursion Port — Round-3 Synthesis Scorecard
**Synthesis agent 13 of 17** | Inputs: 17 Round-1 + 17 Round-2 + cross-references

This is the **definitive plan-vs-review scorecard** for the R3-A06 recursion-port plan. Each row tracks one plan claim through Round-1 critique, Round-2 deeper-investigation, and the Round-3 verdict (CONFIRMED, REVISED, UPGRADED, RESOLVED, RETRACTED, or REJECTED). Numbers in parentheses cite the agent(s) that produced the evidence.

---

## Headline summary

| Theme | Plan claim | R1 | R2 | R3 verdict |
|---|---|---|---|---|
| Bottleneck attribution | "Lift+join is 40% of Succinct E2E" | confirmed | confirmed | **CONFIRMED** |
| Win ceiling | 20-35% Succinct E2E | 10-20% | 6-10% (measurement) / 28-32% (po2-corrected) | **RESOLVED → 6-12% realistic; 15-25% optimistic** |
| Effort | 3-5 weeks | 7-11 weeks | 9-14 weeks | **UPGRADED to 9-14 weeks** |
| Order | step_exec first | accum-first PoC | accum-first PoC (5-8 days, gated) | **CONDITIONAL GO on accum-first** |
| Mechanism of win | step_exec parallelism + witgen GPU port | mis-framed (FRI claim wrong) | eval_check is the real dominator (59%/op already on GPU) | **REVISED — the un-ported per-op dominator is eval_check, not FRI** |

---

## 1. Comprehensive scorecard (20 plan claims)

| # | Plan claim (original) | Round-1 review | Round-2 review | Round-3 verdict |
|---|---|---|---|---|
| 1 | **Lift+join = 40% of Succinct E2E (99s of 244s)** | R1-01 reproduced 99.7s / 246.94s = 40.4% | R2-01 reconfirmed (103.4s / 249.65s, 41%); R2-08 broke down lift composition further | **CONFIRMED** — bench reproduces; plan's headline thesis stands |
| 2 | **Per-op cost = 2.4s/lift, 3.4× slower than CUDA** | R1-01: ARITHMETIC ERROR — true cost is 1.16s/lift + 1.24s/join across 83 ops (42 lifts + 41 joins), 1.65× slower than CUDA | R2-08 measured 3375 ms/lift averaged (po2=18; warm-cache lift=3189 ms) | **REJECTED — fix plan headline.** Real per-op figure is 1.16-1.24s; the "3.4×" claim overstates by 2×. R2-08's higher 3.4s figure includes data_commit+eval_check (already-on-GPU), not the witgen+accum slice the port actually addresses |
| 3 | **20-35% Succinct E2E win ceiling** | R1-01: 10-15% (FRI dominance, mis-named — see #4); R1-16: 10-20% (B70 is 6.2× slower than RTX 4090, CUDA parity unrealistic) | R2-01: **6-10%** measurement-backed (witgen+accum is only 26%/op; eval_check at 59%/op stays on GPU); R2-14: **28-32%** assuming the corrected po2=18 GPU-saturation story holds | **RESOLVED → 6-12% realistic, 15-25% optimistic.** R2-01's measurement-backed 6-10% is the most credible floor (it correctly identifies what is/isn't being ported). R2-14's 28-32% assumes the GPU port hits ≥9× speedup on witgen+accum AND assumes that slice is the full 30% of lift — but R2-08 shows it's actually ~7% (witgen 189ms + accum 141ms of 3189 ms). **Adjust expectations to 6-12% E2E** for the witgen+accum port alone; **15-25% requires also attacking eval_check** |
| 4 | **"FRI is 70% of per-op cost" (R1-01's framing, used to anchor #3)** | R1-01 inferred this from `[recursion_prove] fri=850ms` log line | R2-01 **REFUTED**: variable named `fri_ms` but actually times entire `prover.finalize()` (eval_check + check_intt + check_commit + eval_u + combos + actual fri). True `fri_prove` is **6 ms (0.5%/op)**. The real per-op dominator is `eval_check` at 730 ms = **59%/op**, already on GPU | **RETRACTED.** R1-01's "FRI dominance" reasoning was wrong-but-arrived-at-correct-conclusion (#3 ceiling still ~10-15%, but via a different mechanism). The plan's mechanism description should say "eval_check (already GPU) is the per-op dominator; witgen+accum is the un-ported slice" |
| 5 | **Effort = 3-5 weeks** | R1-16: **7-11 weeks** (post-merge rv32im tail = 6 weeks of tuning; bit-exactness debug; PO2 sweep + orchestrator plumbing unbudgeted; tuning tail) | R2-11: **9-14 weeks** (R1-16 + Week 0 patched-IGC verification + recursion-specific IGC pathology already observed at 1h24m hang); R2-15 ordered into 6-PR stacked plan covering Day-0 → Month-3 | **UPGRADED to 9-14 weeks.** R2-11's analysis is most thorough — it folds in the IGC patched-build dependency (Scenarios A/B/C/D), the recursion-specific IGC bug (R3-A06 already hit 1h24m ocloc hang on poly_fp), and the rv32im post-merge tail evidence (6+ weeks). Plan's 3-5 wk is **happy-path-only**; budget 9-14 wk for full delivery |
| 6 | **Port order: step_exec first (40K LOC), eval_check optimization second** | R1-03: PORT ORDER WRONG — start with `step_compute_accum` (15K LOC, 1 extern, fully parallel) as PoC of the same plumbing | R2-05: **detailed 5-8 day PoC plan** for step_compute_accum (F1-F7 file list, day-by-day plan, gated by `RISC0_INTEL_GPU_ACCUM=1`); R2-15 orders into PR sequence with step_compute_accum PoC as PR #5 | **CONDITIONAL GO on accum-first (R2-05 plan).** Strictly safer: 39% the LOC, 1 extern vs 7, unconditionally parallel vs chain-gated. If accum PoC hits IGC compile-time pathology (similar to recursion eval_check's earlier 1h24m hang), bail before committing to step_exec. **Flip the plan's order** |
| 7 | **step_exec is "100s-1000s of cycles in parallel"** | R1-06: chain-bounded; ~10-30% par-safe; effective parallelism ~2400-4900 at po2=14 (NOT 16384) | R2-08: at po2=18, **≥92% of single lift is GPU-parallel** by wall-time (eval_check 46% + data_commit/NTT 36% + accum_commit 1.6% + check_commit 1.9% + FRI 1.1% — all GPU). The 70-85% "non-par-safe" framing conflates within-lift parallelism with cross-lift serialization | **REVISED — both framings have merit on different axes.** Within a lift, GPU-resident work is overwhelmingly parallel (R2-08). Within step_exec specifically, par-safe heads are chain-bounded (R1-06). At po2=18, even chain-bounded parallelism (39K par-safe heads) saturates B70 (R2-14). Net: chain-boundedness is real but not the limiter; **the limiter is wall-clock dominance of already-GPU phases** |
| 8 | **PO2 size for recursion = 14 (16K cycles)** | R1-14: WRONG — `RECURSION_PO2 = 18` (`zkvm/src/host/recursion/prove/mod.rs:61`), 262K cycles; `MIN_LIFT_PO2 = 14` is the inner rv32im segment size that the recursion harness lifts | R2-14: **confirmed po2=18 uniformly** across lift, join, resolve, identity, union, and all `*_povw` variants. 16× more parallelism than plan assumed | **CONFIRMED CORRECTION.** Plan should be edited globally to replace "po2=14, 16K cycles" with "po2=18, 262K cycles". Affects every memory-footprint and parallelism estimate in the plan |
| 9 | **WomRow sort needs GPU radix sort (TBD)** | R1-14: oneDPL provides it (drop-in); confirmed at `/opt/intel/oneapi/dpl/latest/include/oneapi/dpl/`; integration with USM is the real work, not the primitive | R2-09: detailed flavor-B port (move `womRows`/`womIndex` to USM at MachineContext construction; sort+scan+injectWomBacks+stepVerifyWom all on device); kernel-level speedup 4-6×, ~2% E2E in isolation, foundation for full witgen GPU port | **RESOLVED — Risk #3 in the plan is overstated.** Primitive is one line (oneapi::dpl::sort + exclusive_scan). Real work is the USM buffer migration (~80 LOC, 5-7 days). Bit-exactness is **none** (non-stable sort safe; CUDA precedent at `ffi.cu:286-312`) |
| 10 | **Pipelining gives 2.5× on HIP** | R1-01: pipeline doesn't help on single-GPU Intel (~0.8% E2E); R1-12: CPU/GPU phase overlap = ~6.5% E2E (lift N+1 preflight overlaps lift N GPU) | R2-07: **Tier-1 design** = 1-2 weeks for 6.5% E2E (only the preflight overlaps, no GPU contention); Tier-2 = 12% with eval_check_queue overlap; Tier-3 = 20% with deferred-finalize. Most of the infrastructure (`EVAL_CHECK_QUEUE`, `with_queue_override`, `prepare()` API) already exists | **REVISED — pipeline 2.5× is for HIP multi-GPU; on single-GPU Intel it's Tier-1 6.5%.** R2-07's Tier-1 is the right pre-port quick win (1-2 wk, 6.5% E2E). Worth landing BEFORE the full port — but vanishes after full port (single GPU queue) |
| 11 | **Bit-exactness is "highest risk" (1 line)** | R1-07: SIX distinct landmines (iopIdx mutation, womIndex non-atomic, sentinel writes, cross-cycle reads, injectWomBacks ordering, sort tie-breaking) | R2-06: **3-layer harness, 315 LOC, day-by-day** (testutil port from rv32im + kill-switch + Layer-2 in-process probe + Layer-3 seal-equality + AB tests). R2-10: per-risk mitigation table with GPU-side asserts | **UPGRADED — plan severely under-described.** Day-1 harness is REQUIRED before any kernel work (per R1-17, R2-06). Without it, 40K LOC GPU bugs surface as "seal doesn't verify" — months of debug. R2-06's design is shovel-ready; budget Week 1 for harness, 2-3 weeks for W2 (single-cycle bit-exact) |
| 12 | **Validation harness — not in plan** | R1-17: REQUIRED before kernel work; 3 layers (in-process probe, AB tests, E2E seal compare); copy rv32im's EnvGuard/assert_check_eq/golden mechanism | R2-06: **concrete 315-LOC diff across 6 files**, sized for 2-3 engineer-days, day-by-day. Critical correction: probe goes AFTER zeroize (not before) so noise/zeroize HAL ops are also checked | **ADD TO PLAN.** Week-1 deliverable. Without it, the 9-14 week effort estimate is unfounded — bit-exactness debugging unbounded. R2-06's design is mergeable as-is |
| 13 | **Extern complexity: `extern_getMemoryTxn` (risk #4)** | R1-15, R1-16: `extern_getMemoryTxn` **does not exist in recursion** — that's an rv32im extern. Recursion has 7 externs total (4 effective no-ops, 3 simple state mutations); rv32im has 14+. Recursion is materially simpler | R2-10: per-extern mitigation table (R1-R8) with concrete GPU patterns + GPU-side asserts | **REJECTED — plan misdescribed its own risk.** Replace risk #4 with: "7 externs total; `extern_log/readIOPHeader/womWrite/readCoefficients` are no-ops or never reached; `extern_womRead/readIOPBody/plonkWrite_wom` are simple inline device functions over uploaded buffers. ~80 LOC total" |
| 14 | **poly_fp.cpp / `RISC0_RECURSION_OPTIMIZE` worth retrying with patched IGC** | R1-11: **won't work without structural split** — recursion `poly_fp.cpp` is one 24,753-line monolithic SSA function; rv32im splits into 20 noinline sub-fns averaging 2,500 lines each; IGC O(N²) optimizer pass hits 1h24m wall-time with no result | R2-04: **2-4 day mechanical split plan** — 10 sub-functions of ~2,400 lines each via `split_poly_fp.py`; verified that only 2-9 live FpExt values cross any candidate cut (all 10,846 Fp/auto vars are leaf, FpExt-untainted); ABI mirrors rv32im exactly | **CONFIRMED — split is required.** R2-04's plan is shovel-ready (2-4 days). Once split, `-O2 -cl-intel-256-GRF-per-thread` becomes viable. Estimated E2E win: 1-4% per R1-11 / 0.2-0.5% per R1-13 (decomposition disagreement; R1-13 is more measurement-backed) |
| 15 | **eval_check optimization is "1-4% E2E if IGC retry works"** | R1-13: **<0.7% E2E** — per-call recursion eval_check is 5-15ms (recursion has half rv32im's poly_fp + 1/3 the poly_mix); 83 calls × ~10ms = ~1 s out of 244 s | R2-03: ship the `RISC0_RECURSION_EVAL_CHECK_WG` env knob (default 1024, mirrors rv32im pattern; 30 min implementation + 10 min sweep). R2-13: skip-intermediate-verify lever is **already pulled** by ProverImpl's `composite_to_succinct` path; R1-05's 1.5-6% E2E estimate was 3-15× too high | **REVISED DOWN — <1% E2E.** Skip eval_check as a side-quest; ship the WG knob (R2-03) as housekeeping. The skip-intermediate-verify lever (R1-05) is **0%** in production because ProverImpl already bypasses; only the gRPC api/server path retains it |
| 16 | **D2H/H2D cost: 0.2 s / 0.2% E2E** | R1-09: 0.15-0.25 s, 0.07-0.10% E2E (plan's 50 MB/lift overstated; actual ~30 MB) | R2-08 measured: witgen d2h 105ms + h2d 21ms + accum d2h 100ms + h2d 1.5ms = **228 ms PCIe per lift (7.1%)** at po2=18; **54% of witgen+accum wall is pure D2H/H2D shuffling** | **CONFIRMED MAGNITUDE, REFINED SIGNIFICANCE.** Plan was right that 0.2 s/lift D2H is small (R1-09). R2-08 reveals that within the witgen+accum slice, PCIe is over half — meaning a true GPU port (no D2H) saves more than the compute speedup alone. The port's value is **mostly transfer elimination**, not faster compute |
| 17 | **Memory layout: implicit col-major assumption** | R1-08: confirmed col-major both CPU and CUDA; no transpose needed (ctrl already transposed once + cached; data/accum born col-major) | R2-12: **two-tier caching scheme** already correct — content-keyed `(code_rows, po2)` for ctrl/ctrl-PolyGroup/ctrl-host (program-fixed); size-keyed `IntelBufferPool` for data/accum/mix/global/poly_mix_buf/check_poly (per-call). Don't add more caching; **delete the FFI ping-pong** | **CONFIRMED.** Plan's mental model is correct; just call it out explicitly. R2-12's bonus: the per-lift CPU `to_vec()` allocations (~80 MB/lift, ~3.4 GB cumulative over 42 lifts) vanish entirely once the witgen GPU kernel lands — no caching needed, the FFI bridge IS what's being removed |
| 18 | **USM shared as alternative (not in plan)** | R1-10: viable but barely moves wall clock (~1% E2E); B70 supports usm_shared but bottleneck is CPU compute not transfers | (no R2 follow-up; covered by R2-08's PCIe accounting) | **DEFER.** R2-08's measurement (PCIe = 7.1% of lift) makes USM shared a marginal win — keep as Week-1 enabler for incremental porting only |
| 19 | **Pipeline-style 2.5× from HIP**: see #10 | covered above | covered above | covered above |
| 20 | **Quick wins available (not enumerated)** | R1-05: several worth bundling (default-on skip-verify 1.5-6%, CPU/GPU overlap 6.5%, D2H/H2D skips 0.3-0.5%, etc.) | R2-02: 25-LOC default-off intermediate verifies patch (1 dev-day, gated by `intel` cfg + `RISC0_FORCE_VERIFY_INTERMEDIATES=1`); R2-13 **REFUTED** this lever (already pulled by ProverImpl); R2-15 stacked into PR sequence (PR#1 = A+B, Day 0); R2-17 ranked all 6 alternative architectures | **REVISED.** Quick-win menu after R2 refutations: WG knob (R2-03, 30 min, 0-0.3%), default-off intermediate verify (R2-02, 1 day, 0% in optimized path per R2-13 BUT non-zero in gRPC api/server path), poly_fp split (R2-04, 2-4 days, 0.2-1% via unblocking RISC0_RECURSION_OPTIMIZE), Tier-1 pipelining (R2-07, 1-2 wk, 6.5% E2E). **Total realistic quick-win bundle: 7-9% E2E in 2-3 weeks** before any kernel port |

---

## 2. Secondary plan claims (12-row supplementary scorecard)

| # | Plan claim | R1 | R2 | R3 verdict |
|---|---|---|---|---|
| S1 | "CUDA target = 0.7s/lift achievable" | R1-16: B70 is 6.2× slower than RTX 4090 on eval_check; CUDA parity unrealistic | R2-11: R2-A04 CUDA-style eval_u kernel measured parity + SIGABRT — CUDA patterns don't transfer 1:1 to Intel Xe2-HPG | **REVISED to 1.2-1.6s/lift target** — be conservative; market 1.5-2× over current 2.4s baseline |
| S2 | "Helper port = 1 week (W1)" | R1-16: 0.5 wk (recursion externs simpler than rv32im) | R2-11: confirmed 0.5 wk | **REVISED DOWN to 0.5 wk** |
| S3 | "Tuning tail = 1 week (W5)" | R1-16: 2-4 wk (rv32im tail was 6+ wk of micro-experiments) | R2-11: confirmed; also flagged tier-4 IGC flags as dead-end | **UPGRADED to 2-4 wk explicit** |
| S4 | "FRI parallelism is a concern" | R1-01: FRI is on GPU already (`risc0/sys/kernels/zkp/intel/fri_poly_ops.cpp`) | R2-17: confirmed; ESIMD `esimd_fri_fold` already well-tuned | **REJECTED — FRI is already on GPU; nothing to do** |
| S5 | "Multi-GPU parallel lifts" | not in plan | R2-17: hardware-blocked (single B70); ~40% wall-clock on 2-GPU rig if/when second arrives | **PARK as future option**; document host-side parallelism shape |
| S6 | "Replace recursion altogether" | not in plan | R2-17: protocol-blocked (Succinct receipt requires recursion lift→join→...) | **REJECTED — not a plan substitute** |
| S7 | "CUDA wrapper to run NVIDIA kernels" | not in plan | R2-17: physically impossible on Intel B70 | **REJECTED — hardware-incompatible** |
| S8 | "Lift-output cache for repeated inputs" | not in plan | R2-17: 0% on production (unique inputs); +90% on dev/CI loops only | **DEFER — 1-wk side experiment only if a caller proves re-lift pattern** |
| S9 | "Move witgen+accum to ESIMD instead of vanilla SYCL" | not in plan | R2-17: scoped ESIMD on par-safe sub-kernels (sort, scan, FpExt ops, accum tree-reduce) = +5-10% E2E on top of plan's win | **CO-PURSUE — Phase 2 after vanilla SYCL bit-exact reference** |
| S10 | "Default-off intermediate verify (Quick Win #2)" | R1-05: 1.5-6% E2E | R2-13: **0% in production** (ProverImpl already bypasses via direct `lift_with_opts`/`join_with_opts`); 0.4% if gRPC api/server path matters | **REJECTED for production path; KEEP as documentation knob for api/server** |
| S11 | "RISC0_RECURSION_EVAL_CHECK_WG env knob" | R1-13: ship the knob | R2-03: 45-min patch (30 min code + 3 min rebuild + 10 min sweep); 0-0.3% E2E | **SHIP as housekeeping** — freebie, unblocks future post-IGC re-sweeps |
| S12 | "Buffer caching extensions (mix, global, poly_mix)" | R1-08: ctrl already cached via `cached_ctrl_buffer` | R2-12: **nothing else cacheable** (mix/global/poly_mix all Fiat-Shamir-derived, per-call); IntelBufferPool already covers size-keyed pooling | **NO ACTION — two-tier cache scheme is correct as-is** |

---

## 3. Aggregate verdict by category

### Confirmed (plan got it right)
- 40% Succinct E2E share — #1
- col-major memory layout — #17
- D2H/H2D cost magnitude — #16 (but #16 also refines: it's the LARGER share of the witgen+accum slice)

### Revised (plan correct in direction, off in magnitude)
- Win ceiling 20-35% → **6-12% realistic, 15-25% optimistic** (#3)
- Pipeline gain → **6.5% Tier-1 on single-GPU Intel, not 2.5×** (#10)
- Bit-exactness mitigation surface → **6 specific landmines + 3-layer harness, not 1 line** (#11)
- step_exec parallelism → **chain-bounded but GPU-saturating at po2=18** (#7)

### Upgraded (plan under-budgeted)
- Effort: **9-14 weeks**, not 3-5 (#5)
- Tuning tail: **2-4 wk explicit**, not 1 (#S3)
- Validation harness: **MUST BE WEEK-1 DELIVERABLE** (#12)

### Rejected (plan got it wrong)
- Per-op cost framing "2.4s/lift, 3.4× slower" → **1.16s/lift, 1.65× slower** (#2)
- `extern_getMemoryTxn` risk → **doesn't exist in recursion** (#13)
- "FRI is 70%/op" anchor for the win-ceiling reasoning → **0.5%/op; the per-op dominator is eval_check, already GPU** (#4)
- PO2=14 / 16K cycles → **PO2=18 / 262K cycles** (#8)
- WomRow sort needing custom radix → **oneDPL drop-in** (#9)
- Skip-intermediate-verify quick win → **already pulled in optimized path** (#S10)
- FRI parallelism concern → **already on GPU** (#S4)
- CUDA wrapper, replace recursion → **physically impossible/protocol-blocked** (#S6, #S7)

### Conditional / phased GO
- Port order step_exec first → **flip to accum-first PoC** (5-8 days, gated by `RISC0_INTEL_GPU_ACCUM=1` — R2-05 plan) (#6)
- Full step_exec port → **CONDITIONAL** on accum-first PoC clearing IGC pathology (#6, #5, #14)
- poly_fp split → **GO** (2-4 days, R2-04 plan) — unblocks `RISC0_RECURSION_OPTIMIZE` (#14)
- ESIMD witgen+accum → **CO-PURSUE as Phase 2** (#S9)

---

## 4. Round-3 final recommendation: **PIVOT to a 4-phase plan**

**Phase A (1-2 weeks, GO unconditionally) — net-positive regardless of full port**
1. Validation harness (R2-06, 2-3 days) — REQUIRED before any kernel work
2. `RISC0_RECURSION_EVAL_CHECK_WG` env knob (R2-03, 45 min) — housekeeping
3. Tier-1 pipelining (R2-07, 1-2 weeks) — **6.5% E2E**, vanishes post-port
4. (Optional) poly_fp.cpp split (R2-04, 2-4 days) — unblocks `RISC0_RECURSION_OPTIMIZE`

**Phase B (1-2 weeks, CONDITIONAL GO) — accum-first PoC**
1. step_compute_accum port (R2-05, 5-8 days) — gated by `RISC0_INTEL_GPU_ACCUM=1`
2. **GO/NO-GO checkpoint:** IGC must compile 15K-LOC accum body in <2h. If pathological (matching the recursion eval_check 1h24m hang), bail to "polish Phase A wins only"
3. Direct win: 1-2% E2E; **de-risks Phase C**

**Phase C (4-7 weeks, GO IF PHASE B SUCCESS) — full step_exec port**
- step_exec.cpp SYCL kernel (R1-02 GO, simpler than rv32im)
- WomRow sort/scan via oneDPL (R2-09 flavor-B, 1 week)
- Multi-PO2 sweep (14, 18, 21, 22)
- Direct win: **5-8% E2E** (witgen+accum slice eliminated, including PCIe)

**Phase D (2-4 weeks, IF NEEDED) — tuning tail**
- ESIMD layer on par-safe sub-kernels (R2-17 alternative 6)
- WG sweeps per kernel (R2-11 §3.L5: budget 1 day × 3 kernels)
- Only commit if Phase C lands at >6% and marginal cost-per-percent is acceptable

**Bottom-line numbers:**
- **Effort budget: 9-14 engineer-weeks** (R2-11)
- **E2E win target: 8-12% realistic, 15-20% optimistic** (Phase A 6-7% + Phase C 5-8% with overlap)
- **NOT 20-35%** — the plan's headline overstates by ~2-3×

---

## 5. Confidence map

| Claim | Confidence | Supporting agents |
|---|---|---|
| 40% lift+join share | High | R1-01, R2-01, R2-08 |
| Win ceiling 6-12% (realistic) | High | R2-01 (measurement), R1-01, R1-16 |
| Effort 9-14 weeks | High | R1-16, R2-11 (detailed, IGC risks folded in) |
| Accum-first ordering | High | R1-03, R2-05 (concrete plan), R1-15 (extern complexity comparison) |
| Bit-exactness needs 6-landmine check | High | R1-07 (specific landmines), R2-06 (harness design), R2-10 (per-risk mitigation table) |
| eval_check is per-op dominator (not FRI) | High | R2-01 (log decode), R2-08 (per-phase timings) |
| PO2=18 (not 14) | High | R1-14, R2-14, multiple source citations |
| step_exec is chain-bounded but GPU-saturating at po2=18 | High | R1-06 (preflight evidence), R2-08 (wall-time decomposition), R2-14 (39K par-safe heads vs B70 needs ~32K) |
| oneDPL handles sort/scan | High | R1-14, R2-09 (verified in /opt/intel/oneapi) |
| Pipeline Tier-1 6.5% | Medium-High | R1-12, R2-07 (infrastructure inventory) |
| poly_fp split is required for `-O2` | High | R1-11 (1h24m hang reproduction), R2-04 (live-set analysis) |
| Plan's "3.4× slower" arithmetic error | High | R1-01 (verbose log) |
| FRI is already on GPU | High | R2-17, R2-01 (source inspection) |
| ESIMD adds 5-10% on par-safe sub-kernels | Medium | R2-17 (pattern + tree-reduce precedent — note T3.2 was negative on spill-bound kernel; needs zebin baseline) |
| Patched IGC must be Week-0 prereq | High | R2-11 §4 (scenarios A/B/C/D, recursion-specific bug already observed) |

---

## Appendix — Cited evidence files

Primary sources:
- `/tmp/recursion_round1_agent_{01,02,03,05,06,07,08,09,10,11,12,13,14,15,16,17}.md`
- `/tmp/recursion_round2_agent_{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16_SCORECARD,17}.md`

Bench logs:
- `/tmp/r3a06_v2.log` (R2-01's 100k-iter sha Succinct run, RISC0_VERBOSE=1)
- `/tmp/r3a06_trace_full.log`

Plan document: `project_r3_a06_recursion_port_plan.md` (MEMORY)

Related MEMORY entries: `project_igc_patch_required.md`, `project_eval_check_wg512.md`, `project_tree_reduce_negative.md`, `project_accum_wg_sweep.md`, `project_eval_u_gpu_bound.md`, `project_tier4_igc_flags.md`, `project_stacked_patches_v1.md`, `project_fib_sigabrt_root_cause.md`, `reference_risc0_intel_paths.md`.
