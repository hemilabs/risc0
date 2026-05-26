# R3-A06 Round-3 Agent 12 — Phase A Stacked-Wins Realism Audit

## Charter

Phase A = items in the R3-A06 plan that DO NOT depend on porting
`step_exec` / `step_compute_accum` to GPU (i.e. shippable without the
6-9 week full witgen-on-GPU port). Goal: produce a realistic
stacked-EV table for the five Phase A PRs and decide whether they
stack additively or alias each other.

---

## Per-PR audit

### PR 1 — RISC0_RECURSION_EVAL_CHECK_WG knob

| Field | Value |
|---|---|
| Solo EV | **0-0.3% Succinct E2E** |
| Stacks? | **Yes (orthogonal)** |
| Effort | 30 min code + 3 min rebuild + 10 min sweep = **~45 min** |
| Risk | **Low** — env-knob is byte-identical to baseline at default 1024 |

**Source of EV**: `/tmp/recursion_round2_agent_03.md`:140 — "Expected E2E win: 0-0.3% Succinct. Worth doing regardless."
Corroborated by `/tmp/recursion_round1_agent_13.md` (per-call decomposition): recursion eval_check is 5-15 ms × 83 ops = 0.4-1.2 s out of 244 s Succinct = ~0.2-0.5% ceiling; only the sweep delta matters, which is sub-1%.

**Status**: ALREADY APPLIED per `/tmp/recursion_round3_agent_02.md`. Patch in tree at `recursion-sys/kernels/intel/eval_check.cpp:43-55`. One rebuild required, then env-only sweeps.

**Stacking logic**: Independent of every other Phase A item (pure kernel scheduling). Wins or loses on its own and does not change the cost of anything else.

---

### PR 2 — poly_fp.cpp split + RISC0_RECURSION_OPTIMIZE

| Field | Value |
|---|---|
| Solo EV | **1-3% Succinct E2E (very optimistic)** |
| Stacks? | **Yes (orthogonal); modifies same kernel as PR 1** |
| Effort | **2-4 days** mechanical split + 1 day measure |
| Risk | **Med** — bit-exactness sensitive (FpExt chain is order-sensitive); IGC compile-time still unproven |

**Source of EV**: `/tmp/recursion_round1_agent_11.md`:74-98 — "Effort: 2-4 days. Likelihood of unblocking ocloc: very high" (the split itself), gated by an EV-floor argument. `/tmp/recursion_round1_agent_13.md` puts the eval_check ceiling at <0.7% E2E; turning on `-O2 -cl-intel-256-GRF-per-thread` plus 256-GRF + WG=512 raises that ceiling. Round-2 scorecard (`/tmp/recursion_round2_agent_16_SCORECARD.md`:113) caps `eval_check` optimization at "<0.7% E2E" — the 1-3% figure here is optimistic and assumes the split *also* unlocks the broader `-O2` path, but the realistic floor is **<1%**.

**Stacking logic**: Co-modifies recursion eval_check with PR 1; they touch different lines (split = structural; WG knob = runtime). Net EV from stacked PRs 1+2 ≈ same as PR 2 alone because PR 1 is sub-noise. Pure-additive *upper bound* = sum, realistic = max.

**Risk note**: Round-1 agent 11 already identified that `RISC0_RECURSION_OPTIMIZE=1` hit a **1h 24min ocloc hang** on the monolithic 24K-line `poly_fp.cpp`. The split is the prerequisite for *any* future eval_check perf gain — but no gain has yet been measured even hypothetically. EV could easily be 0% if the post-split optimized build doesn't actually outperform the current `-O1 -cl-opt-disable` baseline (Round-1 agent 16 explicitly flags this risk).

---

### PR 3 — Cross-segment pipelining (CPU witgen overlap)

| Field | Value |
|---|---|
| Solo EV | **6.5% Succinct E2E (Tier 1)** |
| Stacks? | **Partial overlap with full port; ANTI-stacks with PR 4 below at Tier 2+** |
| Effort | **1-2 weeks** for Tier 1 only |
| Risk | **Med** — multi-threading, `SegmentReceipt::clone` cost, Send-safety of `Prover` |

**Source of EV**: `/tmp/recursion_round1_agent_12.md`:44, 98 — "save ~0.4 s × 41 = **16 s ≈ 6.5% Succinct E2E** if done right" (the CPU/GPU phase overlap Q2 exception); confirmed in `/tmp/recursion_round2_agent_07.md`:261 (Tier 1 = ~6.5%).

**Stacking logic — CRITICAL**: 
- This win is a **before-port-only** opportunity. R1-12:106 and R2-07:316 both note explicitly: *post full step_exec port, witgen-on-GPU eliminates the CPU phase, so the overlap window shrinks to zero on a single GPU queue*.
- **Stacks fully with PR 1, PR 2**: they're independent kernels and the pipelining doesn't change their cost.
- **Stacks with PR 4 (oneDPL/USM)**: orthogonal mechanisms — pipelining is scheduling, sort/USM is data layout.
- **Partial overlap with PR 5**: pipelining and background-finalize both consume the GPU `EVAL_CHECK_QUEUE`. Tier 1 pipelining only uses CPU on the bg thread, so it doesn't actually contend with PR 5; but a future Tier 2/3 extension (overlapping witgen H2D/D2H on the eval queue) would.
- **ANTI-stacks with the full port (Phase B/C)**: this win vanishes the moment witgen moves to GPU. So if the full port lands, PR 3's 6.5% is *not* a permanent gain — it's a 1-2 week shipped win that becomes obsolete with the port.

**Effort note**: Tier 1 is honest at 1 week; Tier 2 ("also overlap bg lift's witgen FFI") doubles to 2-3 weeks and is the next ~6%; Tier 3 (deferred-finalize parity with rv32im) adds another 2 weeks for another ~7%. Phase A budget should be Tier 1 only.

---

### PR 4 — oneDPL sort + persistent USM scaffolding

| Field | Value |
|---|---|
| Solo EV | **1-3% Succinct E2E (closer to 1%)** |
| Stacks? | **Yes (foundational for full port); standalone EV is small** |
| Effort | **5-7 days** |
| Risk | **Med** — oneDPL's compatibility with `WomArgumentRow` device comparator; FpExt scan unproven |

**Source of EV**: `/tmp/recursion_round1_agent_14.md`:102-104 — *"CPU sort+scan today: maybe 50–100 ms per lift, ~2–4 s across the 42 lifts = 1–2% of Succinct E2E. GPU `oneapi::dpl::sort` of 2.36M elements on Xe2-HPG: typically 5–15 ms. Save 35–85 ms per lift = 1.5–3.5 s total = ~0.6–1.4% E2E."*

Crucially R1-14 line 105 says: *"The sort/scan in isolation is not a meaningful E2E win. It's only worth porting if and when we're already migrating the surrounding code (step_exec, step_verify_mem, step_verify_wom) to GPU so the data is already on device and we avoid a 45 MB D2H/H2D round-trip."*

R2-09 sketches the (A) drop-in version (USM upload/download bracketed around sort) which actually **costs** the round-trip and could be near-zero or negative net; only the (B) "data already on device" version is unambiguously positive — but that requires the full port.

**Stacking logic**:
- **Stacks with PR 1, PR 2**: orthogonal kernels.
- **Stacks with PR 3**: pipelining schedules around it; sort/scan still serial on GPU queue but its cost is small enough that the overlap window is unaffected.
- **Foundation for the full port**: the USM persistent allocation scaffolding (`malloc_device` + `CachedExecBuffers` analog) is the same plumbing the witgen-on-GPU port needs. Lands the FFI surface and the SYCL queue handoff *without* the bit-exactness hazards of the witgen kernels.
- **Risk of zero standalone gain**: if landed as a drop-in port (R2-09 flavor A), the D2H/H2D round-trip cost cancels the GPU sort win. Solo EV is **1% optimistic, 0% pessimistic** in flavor A; flavor B (data already on device) only exists if PR 4 lands *with* the full port.

**Verdict**: Treat PR 4 as **scaffolding work, not a perf PR**. Its real value is de-risking Phase B (accum-first PoC) by validating oneDPL + USM patterns before committing to the larger port.

---

### PR 5 — Background thread for finalize

| Field | Value |
|---|---|
| Solo EV | **~0.4-1% Succinct E2E (estimated)** |
| Stacks? | **Yes (mostly orthogonal)** |
| Effort | **3-5 days** (estimate; not in any agent report directly) |
| Risk | **Med** — `EVAL_CHECK_QUEUE` scheduling, deferred-receipt lifetime |

**Source of EV**: No agent in Rounds 1 or 2 produced a standalone EV measurement for backgrounded finalize on the recursion path. Best precedents:
- `/tmp/recursion_round1_agent_16.md`:84 — rv32im's "backgrounded finalize on eval_check queue" landed Apr 23 for **-84 ms per segment**; for 42 lifts × 84 ms = ~3.5 s = **~1.4% Succinct E2E** if recursion benefits scale identically. But recursion is GPU-different (eval_check kernel is smaller, ~5-15 ms not ~80 ms), so realistic scaling is **0.4-1%**.
- `/tmp/recursion_round2_agent_07.md`:264 — Tier 3 ("deferred finalize for lift, parity with rv32im") = "another ~0.4-0.6 s" saved per lift = **~17 s = ~7% Succinct E2E**. But this is *Tier 3* of PR 3's pipelining, not a standalone PR. It assumes pipelining already retired Tiers 1+2.

**EV uncertainty**: This PR's standalone EV is the **least documented** of the five. Without a per-lift breakdown of how much of the 5-15 ms eval_check + FRI + Merkle paths is *blocking* CPU vs. *overlappable*, the 0.4-1% range is a guess.

**Stacking logic**:
- **Stacks with PR 1, PR 2, PR 4**: orthogonal mechanisms.
- **Partial conflict with PR 3 Tier 1**: PR 5 also uses `EVAL_CHECK_QUEUE` to overlap finalize with the next lift's prep; PR 3 Tier 1 uses CPU-only bg thread for preflight. They don't contend on the *queue*, but they do contend for the *idle slot* on the critical path — the same idle window can only hide one set of work.
- **Stacks with the full port**: post-port, witgen is on GPU and the bg-finalize approach is still useful (rv32im has it post-witgen-port).

**Recommendation**: Investigate during PR 3's measurement phase. If PR 3 Tier 1 retires the bulk of CPU/GPU overlap (~6.5%), PR 5's marginal contribution is likely 0.4-1% on top. If pursued, fold into PR 3's Tier 3 rather than a separate PR.

---

## Stacked-wins TOTAL — Phase A only

### Realistic stack (additive, no over-counting)

| PR | Solo EV (realistic) | Stacked contribution | Cumulative |
|---|---:|---:|---:|
| 1 | 0.2% | 0.2% (independent) | **0.2%** |
| 2 | 0.7% | 0.5% (overlaps with PR 1 on same kernel; max-mode reads as 0.7% but optimistic +0.5%) | **0.7%** |
| 3 (Tier 1 only) | 6.5% | 6.5% (orthogonal) | **7.2%** |
| 4 (drop-in) | 0.5% | 0.5% (orthogonal but small) | **7.7%** |
| 5 | 0.7% | 0.5% (partial overlap with PR 3's idle window) | **8.2%** |

**Realistic Phase A total: ~7-9% Succinct E2E.**

### Optimistic stack (all PRs hit their high end)

| PR | Solo EV (high) | Cumulative |
|---|---:|---:|
| 1 | 0.3% | 0.3% |
| 2 | 3% | 3.3% |
| 3 | 6.5% | 9.8% |
| 4 | 3% | 12.8% |
| 5 | 1% | 13.8% |

**Optimistic Phase A ceiling: ~14% Succinct E2E.**

### Pessimistic stack (drop-in flavors, alias penalties)

| PR | Solo EV (low) | Cumulative |
|---|---:|---:|
| 1 | 0% | 0% |
| 2 | 0% (no compile improvement) | 0% |
| 3 | 4% (Tier 1 partial) | 4% |
| 4 | 0% (D2H/H2D cancels) | 4% |
| 5 | 0.4% | 4.4% |

**Pessimistic Phase A floor: ~4% Succinct E2E.**

---

## Summary recommendation

**Land PR 1 immediately** (already in tree, 45 min work). **Land PR 3 Tier 1 first as the headline win** (6.5%, 1-2 weeks). **Treat PR 4 as scaffolding** for the eventual accum-first PoC, not a perf-only PR. **Land PR 2 only if zebin metadata + bit-exactness harness are in place** (it's the highest-risk, lowest-EV of the five). **Defer PR 5** until after PR 3's measurement informs the residual overlap window.

The honest expected total for Phase A is **~7-9% Succinct E2E** for ~2-3 weeks of focused effort, with a realistic ceiling of ~14% if all upper bounds hit and a floor of ~4% if PR 2 and PR 4 deliver zero. This is consistent with the Round-2 scorecard's "Phase A retires 3-8% E2E" framing (`/tmp/recursion_round2_agent_16_SCORECARD.md`:67-73, 137).

**Critical caveat from Round-1 agent 12**: PR 3's 6.5% is *not permanent* — it vanishes after the full step_exec/step_compute_accum port (Phase C). So if the full port lands at >12% E2E, the realistic *post-port* Phase A residual is only PR 1 + PR 2 + PR 4 + PR 5 = ~1.4-4% additive on top of the port, not the full 7-9%. Phase A and the full port are partially substitutive, not fully additive.

---

## Files cited

- `/tmp/recursion_round1_agent_11.md` — poly_fp.cpp split analysis (PR 2)
- `/tmp/recursion_round1_agent_12.md` — CPU/GPU overlap = 6.5% (PR 3)
- `/tmp/recursion_round1_agent_13.md` — eval_check sub-1% E2E (PR 1, PR 2)
- `/tmp/recursion_round1_agent_14.md` — oneDPL sort EV 1-2% standalone (PR 4)
- `/tmp/recursion_round1_agent_16.md` — rv32im backgrounded finalize precedent (PR 5)
- `/tmp/recursion_round2_agent_03.md` — RISC0_RECURSION_EVAL_CHECK_WG patch and EV (PR 1)
- `/tmp/recursion_round2_agent_07.md` — Pipelining Tier 1/2/3 design (PR 3, PR 5)
- `/tmp/recursion_round2_agent_09.md` — oneDPL sort/scan port mechanics (PR 4)
- `/tmp/recursion_round2_agent_11.md` — rv32im tuning-tail timeline (PR 5 precedent)
- `/tmp/recursion_round2_agent_13.md` — RISC0_SKIP_VERIFY already done at composite_to_succinct
- `/tmp/recursion_round2_agent_14.md` — po2=18 confirmation (affects PR 4 cycle count)
- `/tmp/recursion_round2_agent_16_SCORECARD.md` — Phase A framing (3-8% E2E)
- `/tmp/recursion_round3_agent_02.md` — PR 1 already applied to eval_check.cpp:43-55
