# R3-A06 Recursion Port — Round-3 Synthesis VERDICT

**Agent 16 of 17 | Final GO/NO-GO recommendation**

---

## TL;DR

**PIVOT to Phase A only.** Capture the 5–12% E2E quick wins in 2–4 weeks, defer
the full port indefinitely. The full port's corrected 6–10% Succinct-only
ceiling at 9–14 weeks of risk-heavy work no longer clears the bar set by the
quick wins themselves, and Composite callers (the larger user base) gain
nothing.

---

## Corrected inputs since R2 synthesis

| Dimension | R2 estimate | R3 corrected | Direction |
|---|---:|---:|---|
| Win ceiling (Succinct E2E) | 15–25% (target 18%) | **6–10%** | ▼ ~2.5× lower |
| Effort (focused weeks) | 7–11 wk | **9–14 wk** | ▲ ~1.3× higher |
| Phase A available now | 3–8% E2E in 1–2 wk | **5–12% E2E in 2–4 wk** | ▲ better than expected |
| Composite path baseline | unchanged | **already −10.7%** from prior session | ▲ floor moved up |
| Affected user surface | "Succinct/Groth16" | **Succinct/Groth16 only — Composite gets 0%** | unchanged but more salient |

The "win-to-effort ratio" is now:
- **Full port:** 6–10% Succinct-only / 9–14 wk = **~0.6–1.1 % per engineer-week**, Succinct callers only.
- **Phase A:** 5–12% E2E / 2–4 wk = **~1.5–6 % per engineer-week**, same callers (and several Phase A items help Composite too).
- **Phase A strictly dominates the full port on both axes.**

---

## Option 1 — GO (full port, 9–14 weeks)

### Pros
- Lands the largest single Succinct lever still available on Intel.
- Closes a known architectural gap vs CUDA (step_exec / step_compute_accum on GPU).
- Builds team competence on a SYCL port of a 53K-LOC kernel — valuable for future work.
- Validates the patched IGC fork at production scale.
- If Phase A retires <5% E2E, the full port becomes the only path to meaningful Succinct gains.

### Cons
- **Win ceiling is now 6–10%, not 20–35%.** Plan headline overstates by 3–6×.
- **Effort is now 9–14 weeks.** That's a quarter of an engineer-year for one optimization.
- **Composite callers see 0%.** Composite is the more common production path; recursion only fires when the caller explicitly requests Succinct (and onward to Groth16).
- **Composite already has the −10.7% prior-session win** — the floor under "do nothing on recursion" is already moved.
- **Risk concentration.** Six bit-exactness landmines (R1-07), IGC scale on 53K-LOC monolithic SSA (R1-04/R1-11), unbudgeted PO2 sweeps and tuning tail (R1-16), patched-IGC dependence (memo `project_igc_patch_required.md`). Any one slipping pushes the timeline past 14 weeks.
- **Opportunity cost.** The optimization landscape memo (`project_optimization_landscape.md`) lists Phase 6 eval_check restructure at ~25% E2E for "months" — comparable timeline, larger payoff, broader caller surface. 9–14 weeks committed to recursion forecloses that.
- **No CUDA parity floor.** R1-16 showed B70 is 6.2× slower than RTX 4090 on eval_check; the ported step_exec is unlikely to ever close that gap. We are porting toward a moving CUDA target we can't catch.

### Verdict
**NO on full port.** The corrected ceiling does not justify the corrected effort, especially with Composite unaffected and the Phase 6 eval_check option sitting on the same timeline budget with a larger and broader payoff.

---

## Option 2 — PIVOT to Phase A only (2–4 weeks)

### Pros
- **Strictly best win-to-effort ratio.** 5–12% E2E for 2–4 weeks = 1.5–6 %/week.
- **Some Phase A items help Composite too** (e.g., CPU/GPU phase overlap pattern is a primitive worth having; eval_check WG knob is already applied to recursion per R3-02).
- **Net-positive regardless of full-port decision.** Phase A items don't conflict with a future port; they retire low-hanging fruit either way.
- **Frees 5–10 weeks of engineer time** for the Phase 6 eval_check restructure or other items in `project_optimization_landscape.md`.
- **Validates the harness investment** (R1-17) — even if the port is never built, a recursion bit-exactness harness has long-term value for any future GPU experiments on the recursion path.
- **Reversible.** If Phase A lands <5% E2E AND no other Intel optimization surfaces, we can revisit Phase B+C later with the harness already built.
- **Honest framing for stakeholders.** "We delivered 5–12% Succinct E2E in a month" is a clean story; "we shipped 6–10% in 9–14 weeks after promising 20–35%" is a credibility hit.

### Cons
- **Leaves 6–10% on the table** for Succinct callers. If a high-value Succinct caller demands the gap, we have to come back to it.
- **Doesn't validate the SYCL port toolchain at recursion scale.** Future ports get no transferable plumbing.
- **Some Phase A items are temporary.** CPU/GPU phase overlap (R1-12) vanishes once everything is on GPU; if the port ever lands, the overlap work is discarded.
- **Defer-indefinitely tends to become defer-forever.** Without an explicit re-trigger condition, the residual 6–10% is functionally written off.

### Verdict
**STRONG GO on Phase A only.** This is the recommended option.

---

## Option 3 — NO-GO entirely (stop R3-A06, do other work)

### Pros
- **Maximum reallocation.** All engineer time goes to Phase 6 eval_check (~25% E2E target) or other items in `project_optimization_landscape.md`.
- **Avoids partial-credit risk** where Phase A lands a 2% disappointment and the optics burn capital with no follow-up.
- **No new harness to maintain** if recursion is parked.

### Cons
- **Throws away the 5–12% Phase A win** that costs only 2–4 weeks. That's irrational.
- **Some Phase A work is already done.** R3-02 applied the `RISC0_RECURSION_EVAL_CHECK_WG` knob; abandoning means leaving a partial change in tree without a sweep.
- **Removes the harness** that even a future port (or any future recursion experiment) would need.
- **Wastes the R1+R2+R3 review investment** of 16+ agent-reports.

### Verdict
**NO on full no-go.** Phase A's win-to-effort ratio is too good to discard. NO-GO is only correct if Phase A's *measured* win in flight is <2% and a higher-priority lever surfaces — i.e., a decision deferred to the end of Phase A, not made now.

---

## Comparative scorecard

| Criterion | GO (full) | PIVOT (Phase A) | NO-GO |
|---|:-:|:-:|:-:|
| Realistic E2E win | 6–10% Succinct only | **5–12% Succinct only** | 0% (on recursion) |
| Engineer-weeks | 9–14 | **2–4** | 0 |
| Composite path benefit | none | none (some primitives reusable) | none |
| Risk level | high (IGC, bit-exact, tuning tail) | **low (env knobs, harness)** | n/a |
| Reversibility | low (sunk port cost) | **high** | medium |
| Opportunity cost | Phase 6 eval_check (~25%) | **minimal** | minimal |
| Win-per-week | 0.6–1.1% | **1.5–6%** | 0% |
| Credibility risk if overshoots | high | **low** | low |
| Stakeholder narrative | "delivered 6–10% in 14 wks vs 20–35% promised" | **"delivered 5–12% in a month"** | "deferred to Phase 6" |

PIVOT wins on every measurable axis.

---

## Recommendation: **PIVOT to Phase A only**

### What "Phase A only" means concretely

Execute the four items from the R2 synthesis, in order:

1. **Validation harness** (R1-17, 2–3 days) — REQUIRED first; even Phase A items need it for bit-exactness regression detection. Has long-term value independent of port decision.
2. **Default-on `RISC0_SKIP_VERIFY`** for intermediate lift+join (R1-05 #2, 1 day) → 1.5–6% E2E. Largest single-day item.
3. **CPU/GPU phase overlap** (R1-12, 1–2 weeks) → ~6.5% E2E. Lift N+1 witgen overlaps lift N eval_check. Temporary — vanishes if port ever lands — but pays back inside the first month.
4. **Skip data/ctrl D2H** (R1-05 #1+#10, 1–2 days) → 0.3–0.5% E2E. Cheap, additive.
5. **Already done in R3-02:** `RISC0_RECURSION_EVAL_CHECK_WG` knob applied. Run a one-day WG sweep (16/32/64/128/256/512/1024) and lock the winner. Likely 0–4% E2E.

Total budget: **2–4 focused engineer-weeks**, capturing **5–12% Succinct E2E**.

### Explicit re-trigger conditions for future full-port reconsideration

The full port should be re-opened **only if all three** of these conditions hold simultaneously:

1. **Phase A lands <4% E2E in measured outcome.** (If Phase A retires ≥4%, the residual gap is too small to justify 9–14 weeks.)
2. **A specific Succinct/Groth16 caller surfaces with a workload >30% of throughput.** (Currently Composite dominates production. R3-A06 is a no-op for them.)
3. **A non-recursion lever ≥10% E2E does NOT surface in the next 2 quarters.** (Phase 6 eval_check restructure, Poseidon2 OpenCL default, witgen WG sweeps, etc. all compete for the same engineer time.)

If any one of those three fails, the port stays deferred. This is a deliberate, falsifiable trigger — not a "we'll revisit it eventually" hand-wave.

### What to write into the project memory

Update `project_r3_a06_recursion_port_plan.md` with:
- The 6–10% corrected ceiling (was 20–35%)
- The 9–14 week corrected effort (was 3–5)
- The PIVOT decision and Phase A roadmap
- The three re-trigger conditions above
- Cross-reference to this verdict file

---

## Rationale summary

The Succinct-only constraint is the swing factor. If Composite were affected, 6–10% E2E across the dominant caller path would clear 9–14 weeks easily. But:

- **Composite has its own −10.7% win already booked** from the prior session.
- **R3-A06 only helps Succinct → Groth16 callers**, which are a smaller (though high-value) subset.
- **The marginal Succinct user pays an extra 9–14 weeks of engineer time** for a 6–10% local improvement — and Phase A captures most of that 6–10% in 2–4 weeks anyway.
- **Phase 6 eval_check restructure** (per `project_optimization_landscape.md`) sits in the same multi-month time budget and targets ~25% E2E across *all* paths (Composite + Succinct + Groth16). It is the correct next big lever.

Full-port effort is better redirected to either:
- Phase 6 eval_check (highest-payoff, broadest-surface), or
- The Tier-2 WG sweeps + Poseidon2 OpenCL default from `project_optimization_landscape.md` (small, additive, both paths).

The full R3-A06 port is the **third-best** use of 9–14 weeks. PIVOT.

---

## Files referenced

- `/tmp/recursion_round2_agent_16_SCORECARD.md` — R2 synthesis (15–25% ceiling, 7–11 wk)
- `/tmp/recursion_round2_agent_17.md` — Alternative architectures (ESIMD hybrid, par-safe audit)
- `/tmp/recursion_round3_agent_02.md` — R3-02 freebie patch applied (`RISC0_RECURSION_EVAL_CHECK_WG` knob)
- `/home/user/.claude/projects/-home-user-inteldebug/memory/project_r3_a06_recursion_port_plan.md` — original plan (to be updated)
- `/home/user/.claude/projects/-home-user-inteldebug/memory/project_optimization_landscape.md` — Phase 6 alternative
- `/home/user/.claude/projects/-home-user-inteldebug/memory/project_igc_patch_required.md` — patched IGC dependency
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion-sys/kernels/intel/eval_check.cpp` — file already touched by R3-02

---

## Bottom line

**PIVOT. Phase A only. 2–4 weeks. 5–12% E2E. Defer full port behind three explicit re-trigger conditions.**

Re-running the same 9–14 weeks on Phase 6 eval_check restructure is strictly better for the broader caller base. The R3-A06 full port is a third-best use of that engineer time given (a) the corrected 6–10% ceiling, (b) the Succinct-only constraint, and (c) the existing Composite −10.7% floor.
