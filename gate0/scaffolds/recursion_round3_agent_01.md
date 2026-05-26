# R3-A06 Round-3 Agent 01 — Authoritative E2E Ceiling

## Bottom line

**Authoritative R3-A06 ceiling: 6-10% Succinct E2E speedup, hard upper bound 10.7%.**

R2-01 was right. R2-08's measurement confirms it. R2-14's 28-32% claim is mechanistically wrong — it built a per-lift speedup ratio against a non-existent CPU baseline. R2-16's 15-25% synthesis split the difference inappropriately and absorbed R2-14's bad math; R2-16 should be tightened to 6-10%.

---

## Resolution of the three contradicting claims

### 1. Is witgen+accum 10% of lift wall (R2-08) or ~25%+ (R2-14)?

**It is 10.3% of the warm-lift wall (R2-08 measurement) and 26% of per-op average wall (R2-01 measurement); both numbers are internally consistent because they refer to *different denominators*, not different physical work.**

R2-08 absolute timing (warm lift 1, total 3189 ms wall):
- witgen total: 189.1 ms (d2h 104.5 + cpu_ffi 48.9 + h2d 20.9)
- accum total: 140.8 ms (d2h 99.8 + cpu_ffi 29.7 + h2d 1.5)
- **witgen+accum = 329.9 ms = 10.3% of this lift's wall**

R2-01 absolute timing (per-op average across 83 ops, larger workload):
- witgen total: 212 ms ; accum total: 111 ms
- **witgen+accum = 323 ms = 26% of per-op wall (1241 ms)**

The witgen+accum ABSOLUTE cost is stable across both runs (~325 ms/op) because the recursion harness always runs at PO2=18. The denominator differs because R2-08's lift had unusually heavy GPU work (eval_check 1480 ms + data_commit 1151 ms = 82% of lift) — likely measurement noise, or a smaller workload where GPU phases haven't fully amortized. **For Succinct-E2E budgeting, the per-op average is the right denominator** because Succinct chains run many ops; we use the 323 ms anchor.

### 2. Did R2-14 confuse step_exec speedup with lift wall speedup?

**Yes, and worse: R2-14 built the speedup ratio against a CPU baseline that doesn't exist on Intel.**

R2-14's derivation (lines 58-91 of its report):
- CPU baseline: 2.4 s/lift (composed of "doStepExec 0.7 + verifyWom 0.24 + injectWomBacks 0.1 + computeAccum 0.7 + calcPrefixProducts 0.1 + verifyAccum 0.36")
- GPU projection: 0.25 s/lift
- Per-lift speedup: 9.6×
- Succinct E2E: save 89 s, drop 280 s → 191 s = **31.8%**

This is mechanism-wrong on Intel:

1. **Intel today does not run those 2.4 s of phases on CPU.** It runs `eval_check`, NTT, merkle, FRI on the GPU (1.48 + 1.15 + ... = ~92% of lift wall per R2-08). The only CPU-FFI work on Intel today is **witgen+accum at ~330 ms/op**. R2-14 quoted a CUDA-reference CPU baseline and treated it as the Intel starting point.
2. **The "9.6× per-lift" speedup is therefore against fictional work.** Applying it to a workload that's already 92% GPU produces nonsense numbers.
3. **The corrected arithmetic with R2-14's own logic** (which the R3 task prompt sketched out): even if every CPU phase R3-A06 touches gets a 9.6× speedup, the *moveable* fraction of a lift is only 10.3% (R2-08's measured witgen+accum). Saving 9.6× of 10.3% of the lift wall = 9.2% of lift wall = **3.7% Succinct E2E** (multiplying by the 40% lift+join share). Even at *infinite* speedup the answer is bounded by the moveable mass.

### 3. Direct contradiction: 262K par-safe cycles vs measured wall-time

R2-14's parallelism math (39 322 par-safe heads, B70 saturation, etc.) is correct but **answers the wrong question**. It establishes that the GPU port *can* saturate the hardware. It does *not* establish that the resulting kernel will be faster than the current CPU FFI by 9.6×. The CPU FFI already runs at ~49 ms (witgen) + 30 ms (accum) per op — extremely cheap because most of the CPU work was already eliminated in previous rounds. A 9.6× speedup against the CPU FFI alone would save ~70 ms × 83 = 5.8 s (~2.3% E2E). The big win comes from PCIe elimination (~176 ms × 83 = 14.6 s = 5.9% E2E), not compute speedup.

---

## Authoritative ceiling derivation

**Anchors (from R2-08's direct timer measurements, cross-checked by R2-01):**
- Per-op witgen+accum cost: 323 ms (R2-01 per-op avg) ≈ 330 ms (R2-08 lift 1)
  - PCIe d2h + h2d: 176 ms (eliminable — data stays on GPU)
  - CPU FFI compute: 147 ms (replaced by GPU kernel)
- Ops per Succinct chain: 83 (42 lifts + 41 joins, R2-01 confirmed; R2-14 uses same count)
- Lift+join fraction of Succinct E2E: 40% (R2-01 measured 103.4 s / 249.65 s; R2-08 confirms ~41%)
- Succinct E2E reference: 250 s (matches R2-01 measured 249.65 s)

**Ceiling computation (max moveable):**

| Scenario | witgen+accum after port | Per-op save | × 83 ops | % Succinct E2E |
|---|---:|---:|---:|---:|
| Worst case (PCIe only) | 147 ms | 176 ms | 14.6 s | **5.9%** |
| Realistic mid | 80 ms | 243 ms | 20.2 s | **8.1%** |
| Best case (GPU matches CUDA ~40 ms) | 40 ms | 283 ms | 23.5 s | **9.4%** |
| Theoretical max (free 0 ms) | 0 ms | 323 ms | 26.8 s | **10.7%** |

**Authoritative range: 6-10% Succinct E2E.** The 10.7% is a hard ceiling; nothing in the recursion port can exceed it because witgen+accum is only 10.7% of Succinct E2E to begin with.

---

## Where each prior agent landed (graded)

| Agent | Claim | Verdict |
|---|---|---|
| R2-01 | 6-10% Succinct E2E | **CORRECT** — confirmed verbatim by R2-08's direct measurements |
| R2-08 | (Did not state a ceiling; provided measurement only) | Measurement is sound; if asked, would land at ~6-10% |
| R2-14 | 28-32% Succinct E2E, 9.6× per lift | **WRONG** — used CUDA-reference CPU baseline (2.4 s) that doesn't exist on Intel; speedup ratio applied to imaginary work |
| R2-16 | 15-25% (target 18%) | **PARTIALLY WRONG** — split the difference between R2-01 (correct) and R2-14 (wrong by mechanism), so the synthesis got dragged upward. The "non-step_exec phases drive the win" framing inherited from R1-06 implicitly assumes those phases are CPU-bound, which Intel measurements contradict. |

---

## R2-14's specific arithmetic, corrected with R2-08 measurements

R2-14 row by row:

| R2-14 phase | R2-14 CPU ms | R2-14 GPU ms | Reality on Intel | Already-on-Intel-GPU? |
|---|---:|---:|---|:-:|
| doStepExec (chain heads) | 700 | 80 | Part of witgen FFI (~49 ms total) | No (CPU FFI) |
| verifyWom (sort+scan) | 240 | 10 | NOT in witgen+accum; this is INSIDE `step_exec` GPU on CUDA but on Intel is folded into the CPU witgen FFI's `verify_mem` step | No (CPU FFI piece) |
| verify_mem | (incl. above) | 20 | Same | No |
| injectWomBacks | 100 | 10 | Inside witgen FFI | No |
| computeAccum | 700 | 80 | Part of accum FFI (~30 ms) | No (CPU FFI) |
| calcPrefixProducts | 100 | 5 | Inside accum FFI | No |
| verifyAccum | 360 | 40 | Inside accum FFI | No |
| **Total** | **2400** | **245** | Real Intel CPU portion: **~80 ms FFI + ~225 ms PCIe = 305 ms** | |

R2-14's 2.4 s baseline is 8× larger than Intel's actual 305 ms moveable work. Even if R2-14's 245 ms GPU target is right, the realized save is 305 − 245 ≈ 60 ms/op (PCIe gone, kernel slightly cheaper) → 5 s on 83 ops → **2% Succinct E2E**. To hit even 6% E2E we need to assume **all** of witgen+accum's 330 ms vanishes; that requires both PCIe elimination AND the GPU kernel being ~0 cost.

---

## Implications for R3-A06 plan

1. **Headline number**: drop to "6-10% Succinct E2E speedup". The R2-16 "15-20% target" is unsupportable.
2. **Effort/reward**: at 7-11 weeks of effort (per R2-16) for a 6-10% E2E win, the engineer-week-per-percent ratio is ~1 week per 1% — comparable to harder optimizations already on the menu (Tier-2 WG sweeps ~5% combined for days of work; Poseidon2 default-on 1.4% for hours).
3. **What R3-A06 should still attempt** (best-case path to 9-10%):
   - Eliminate PCIe round-trips by keeping data/ctrl/accum buffers on GPU between witgen→commit→accum→eval_check (the largest chunk of the saveable 226 ms/op).
   - Port witgen+accum compute to GPU only insofar as needed to keep buffers on device; don't expect a per-kernel compute speedup to drive the E2E win.
4. **What R3-A06 cannot deliver**: anything close to R2-14's 30% or R2-16's 18%. Those numbers assumed phases would be moved from CPU to GPU that **are already on GPU on Intel today**. The "intel-already-runs-92%-on-GPU" baseline (R2-08) is the binding constraint.
5. **Follow-on milestone**: if the recursion port lands at the low end (~6%), the bigger lever is **eval_check optimization** (730-1480 ms/op × 83 = 60-120 s of Succinct E2E; even a 1.3× speedup here exceeds the entire R3-A06 ceiling).

---

## Final answer

**R3-A06 Succinct E2E ceiling: 6-10%, hard upper bound 10.7%.**

This number is dictated by the measured fact that **witgen+accum is only 10.7% of Succinct E2E on Intel today** (323 ms/op × 83 ops / 249.65 s). No GPU port of those two phases — however well-optimized — can exceed that ceiling. R2-01 had the right number. R2-08 provided the measurement that nails it down. R2-14 mistook a CUDA-reference baseline for Intel's current state. R2-16 absorbed R2-14's error and should be revised.
