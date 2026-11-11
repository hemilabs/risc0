# R3-A06 Round-3 Agent 03 — Validate R2-08 per-lift breakdown

**Verdict: NEEDS-MEASUREMENT — R2-08's headline numbers are wrong for the
"steady-state single lift" frame they claim. Their data is contention-poisoned
from a 5-seg interleaved pipeline, not a clean per-lift baseline.**

## What I did

1. Confirmed R2-08's instrumentation is **still in source** at
   `/home/user/risc0-intel/risc0/risc0/circuit/recursion/src/prove/hal/intel.rs`
   (lines 70-100, 117-150 — `[recursion_witgen_intel]` and
   `[recursion_accum_intel]` blocks with d2h/ffi_cpu/h2d timing).
2. Ran a clean single-lift bench:
   ```
   RISC0_VERBOSE=1 ./target/release/examples/prove_and_verify 1000 succinct
   ```
   (1 segment, 1 lift, no joins, no concurrent recursion ops). Did this twice
   for stability. Logs: `/tmp/recursion_round3_03_lift.log`,
   `/tmp/recursion_round3_03_lift_v2.log`.
3. Re-read R2-08's raw log `/tmp/succinct_5seg_log.txt` (which they preserved)
   to understand the variance.

## Single-lift measurement (this run, two trials, po2=18, 262144 cycles)

| Phase | Trial 1 (ms) | Trial 2 (ms) | R2-08 lift-1 (ms) |
|---|---:|---:|---:|
| `[recursion_prove] preflight` | 18.5 | 25.5 | 0.0 (cached) |
| `witgen` (d2h+ffi+h2d) | 178.2 | 204.8 | 189.1 |
| · d2h | 79.0 | 97.8 | 104.5 |
| · ffi_cpu | 49.9 | 54.4 | 48.9 |
| · h2d | 13.3 | 12.4 | 20.9 |
| `ctrl` poly_group | 39.4 | 39.2 | 0.0 (cached) |
| `data_commit` (data poly_group) | **56.1** | **56.1** | **1151.3** |
| · data bitrev (NTT) | 11.5 | 11.6 | **1027.9** |
| · data merkle | 38.3 | 38.3 | 121.5 |
| `accum` (d2h+ffi+h2d) | 123.5 | 141.3 | 140.8 |
| `accum_commit` | 13.2 | 13.1 | 50.1 |
| `eval_check_launch` (GPU sync) | **766.9** | **771.7** | **1479.6** |
| `finalize` (check_intt/commit/eval_u/combos/fri) | 99.3 | 97.7 | ~120 |
| **lift total (`recursion_prove` total)** | **1296.1** | **1350.2** | **3189.0** |

## Discrepancy analysis

R2-08 reports 3189 ms / lift on lift #1. My isolated single-lift wall is
**1296–1350 ms**. That is **2.4× faster**, and the per-component split is
~20× off on `data_commit` and ~2× off on `eval_check_launch`.

Pulling from R2-08's own raw log (`/tmp/succinct_5seg_log.txt`):

| op idx in 5-seg run | data_commit (ms) | eval_check_launch (ms) | total (ms) |
|---:|---:|---:|---:|
| lift 0 (cold)  | 1314.6 | 1917.0 | 4432.6 |
| lift 1 (warm)  | 1151.3 | 1479.6 | 3189.0 |
| join 1 | 926.5 | 1673.9 | 3798.0 |
| lift 2 | 922.3 | 950.2  | 2446.5 |
| join 2 | 466.7 | 1386.7 | 2838.5 |
| lift 3 | 923.5 | 1679.1 | 3420.3 |
| join 3 | 566.1 | 1000.2 | 2412.2 |

`data_commit` ranges 466 → 1314 ms on **nominally identical** po2=18 lifts
(count=128 data cols, size=262144, domain=1048576). The bitrev/NTT kernel
itself jitters between 411 ms and 1570 ms.

In my clean single-lift run it is **11.5 ms**, stable to ±0.1 ms across
trials.

**Root cause:** R2-08's data come from `composite_to_succinct`, where lifts
and joins are pipelined and overlap with each other on the GPU (and, in their
5-seg run, also with the segment-prove finalize threads). The GPU bandwidth /
TDR / driver scheduling becomes the bottleneck — `bitrev` (the forward
batch-NTT) is bandwidth-bound and degrades dramatically under contention.
This is **not** intrinsic per-lift cost; it is queue-contention cost.

## Cross-check against R2-01

R2-01 (`/tmp/recursion_round2_agent_01.md`) reports a fresh
100K-iter run, 42 lifts + 41 joins = 83 ops, lift+join total **103.4 s**,
per-op average **~1241 ms**:

> | witgen 212 | commits 69 | accum 111 | eval_check 730 | check+misc 12 |
> | eval_u 27 | combos 41 | fri 6 | **total ~1241 ms** |

That matches my single-lift numbers within noise:
- witgen 178–205 vs 212 ✓
- accum 124–141 vs 111 ✓
- eval_check 767–772 vs 730 ✓
- combos 47–48 vs 41 ✓
- fri (real fri_prove) 7 vs 6 ✓

And it is grossly **inconsistent** with R2-08's "3189 ms/lift" — if that held,
83 ops × 3.19 s = **265 s of lift/join**, but R2-01 measured **103.4 s**
aggregate. R2-08's per-op number is ~2.5× too high.

## What is salvageable from R2-08

The **CPU-side and constant-time pieces** of R2-08 reproduce cleanly:
- witgen breakdown (d2h ≈ 100, ffi_cpu ≈ 50, h2d ≈ 15 ms) ✓
- accum breakdown (d2h ≈ 100, ffi_cpu ≈ 25-30, h2d ≈ 2 ms) ✓
- merkle 38 ms (no contention) ✓
- preflight cold cost ✓

The **GPU-heavy pieces** are the ones inflated by contention:
- data bitrev (forward batch-NTT): nominal 11.5 ms → 400-1570 ms under load
- eval_check (ESIMD): nominal 770 ms → 950-1900 ms under load
- accum_commit merkle/bitrev: nominal 13 ms → 50-122 ms under load

## Corrected per-lift breakdown (po2=18, isolated)

| Phase | Wall (ms) | % of lift |
|---|---:|---:|
| witgen (CPU FFI + d2h/h2d) | 190 | 14 % |
| ctrl poly_group (first lift only) | 39 | 3 % |
| data_commit (NTT + merkle) | 56 | 4 % |
| accum (CPU FFI + d2h/h2d) | 130 | 10 % |
| accum_commit | 13 | 1 % |
| eval_check (GPU sync) | 770 | 57 % |
| finalize (check_intt/commit, eval_u, combos, fri_prove) | 100 | 7 % |
| Rust glue / IOP / preflight | ~25 | 2 % |
| **TOTAL** | **~1323** | **100 %** |

**eval_check is the per-op dominator at ~58 %**, matching R2-01's table.
R2-08's apportionment (eval_check 46 % / data_commit 36 %) **swaps the two
dominators** because their data_commit was contention-inflated 20×.

## Impact on R3-01's E2E ceiling resolution

R2-08's data was being used to argue that **NTT+merkle on the recursion path
(36 %)** is a comparable target to eval_check (46 %), making R3-A06 (CPU FFI
port) less attractive against an NTT-attack alternative. **That is wrong.**
The true split is:

- eval_check ≈ 58 % of per-lift wall
- witgen + accum CPU FFI ≈ 24 % (R3-A06's actual target)
- NTT + merkle commits ≈ 8 %
- finalize (excl. eval_check) ≈ 8 %

So:
- An R3-A06 port that fully eliminates witgen+accum CPU FFI saves at most
  ~24 % of per-op = 26 s of the 103 s lift/join = **~10 % Succinct E2E**
  (matches R2-01's revised estimate).
- An NTT-attack alternative tops out at ~8 %/op = **~3 % E2E** — not
  competitive.
- Closing the remaining gap requires attacking eval_check (multi-stream
  overlap with next-seg witgen, or kernel rework).

## Files modified / left in place

- No source changes by this agent. R2-08's instrumentation in
  `/home/user/risc0-intel/risc0/risc0/circuit/recursion/src/prove/hal/intel.rs`
  is retained — it is correctly written and the new lift logs depend on it.
- `prove_and_verify` example retains R2-08's optional 3rd-arg
  `segment_limit_po2` patch. I did not exercise it (default po2=20 was used,
  which still produced po2=18 lifts via the recursion fast path).

## Recommendation to R3-01

Use the **R2-01-aligned numbers** (eval_check 58 %, witgen/accum 24 %,
commits 8 %, finalize-rest 8 %, per-op ≈ 1.24 s) as the authoritative
ceiling baseline. **Do not** use R2-08's headline 46/36/5/4 split — those
percentages were measured under GPU contention from pipelined lifts/joins,
not at steady-state isolated lift cost.

## Reproduction

```
cd /home/user/risc0-intel/risc0
source /opt/intel/oneapi/setvars.sh
LD_LIBRARY_PATH=/home/user/risc0-intel/risc0/target/release/build/risc0-sys-c50db953e109914f/out:\
/home/user/risc0-intel/risc0/target/release/intel_recursion_cache:\
/home/user/risc0-intel/risc0/target/release/intel_rv32im_cache:\
/home/user/risc0-intel/risc0/target/release/intel_rv32im_cache_default:$LD_LIBRARY_PATH \
RISC0_VERBOSE=1 ./target/release/examples/prove_and_verify 1000 succinct
```

(1 segment, 1 lift, no concurrent ops → clean per-lift measurement.)
