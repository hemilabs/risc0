# R3-A06 Round-3 Agent 15 — Risk Register for Phase A Items

Synthesis of risks for the six concrete Phase A items recommended by R3-09's roadmap
(R2-A16 scorecard + R3-A01 ceiling + R2-11 history → consolidated Phase A list):

1. RISC0_RECURSION_EVAL_CHECK_WG knob (R2-A03)
2. poly_fp.cpp split + RISC0_RECURSION_OPTIMIZE (R1-A11 / R2-A04)
3. Cross-segment pipelining — CPU/GPU phase overlap (R1-A12)
4. oneDPL sort + persistent USM for WomRow (R2-A09)
5. Differential harness (R3-A08 / R2-A06 — gates the step_compute_accum PoC)
6. Full step_exec port (R3-A06 main lever)

Per-item structure: what could go wrong, likelihood, blast radius, detection, mitigation,
with R1 + R2 citations supporting each.

R3-A01's authoritative ceiling caps the total upside at **6-10% Succinct E2E** (hard ceiling
10.7% — witgen+accum is 10.7% of Succinct E2E on Intel today). Many of these risks
threaten not just correctness but the favorable EV/effort ratio.

---

## Item 1 — `RISC0_RECURSION_EVAL_CHECK_WG` knob (R2-A03)

**Status:** APPLIED (R3-A02 confirms the patch is in recursion-sys/kernels/intel/
eval_check.cpp:43-55). One-time recursion eval_check rebuild pending.

### Risk 1.1 — WG=256 spills more than WG=1024 under `-cl-opt-disable`
- **What goes wrong:** Recursion is compiled with `-O1 -cl-opt-disable` (recursion-sys/
  build.rs default). `-cl-opt-disable` keeps register pressure low precisely because
  it inhibits the aggressive IGC passes; smaller WG won't free spill the way it does on
  rv32im's `-O2 -Os` build. Sweep may show 1024 as the optimum, all smaller WGs
  regressing.
- **Likelihood:** Medium-high. R2-A03 sec. 1 calls out exactly this scenario: "Default
  stays at 1024 ... -cl-opt-disable keeps register pressure low enough that the widest
  workgroup the device supports is occupancy-optimal."
- **Blast radius:** Cosmetic. Default is unchanged at 1024, no proof bytes change at
  default. Lost time = sweep cost (~10 min wall-clock per R2-A03 sec. "Build cost").
- **Detection:** Per-WG wall-time table from the sweep (R2-A03 sec. "Suggested sweep").
  If 1024 wins, document and move on.
- **Mitigation:** No code action; this is the expected outcome. Sweep is a check, not
  a bet.

### Risk 1.2 — Whitelist accepts WG=1024 but a future `RISC0_RECURSION_OPTIMIZE` rebuild
   exceeds resource cap at 1024
- **What goes wrong:** If Item 2 (poly_fp split + `-O2 -cl-intel-256-GRF-per-thread`)
  lands, register/spill pressure rises (rv32im's pattern: simd_size=16, 256-GRF →
  halved WG cap of 512 per round3_agent_03's analysis of `gfx_core_helper_xehp_and_later.inl:99-105`).
  WG=1024 then fails to launch with `CL_INVALID_WORK_GROUP_SIZE`, a silent regression
  if default is still 1024.
- **Likelihood:** Medium IF Item 2 lands; near-zero today.
- **Blast radius:** Recursion prove fails outright on the first lift. Caught in
  CI/dev within minutes.
- **Detection:** Driver error string "Total number of work-items in a work-group
  cannot exceed N" (R3-A03 quoting `enqueue_kernel.h:125-127`).
- **Mitigation:** R2-A03 sec. "Default WG=1024 chosen because... If recursion ever
  moves to -O2 or RISC0_RECURSION_OPTIMIZE=1, the spill profile may force WG<=512"
  is already in the comment block. When Item 2 lands, drop the default to 512 in the
  same diff and remove 1024 from the whitelist.

### Risk 1.3 — Bit-mismatch via barrier semantics at sub-1024 WG values
- **What goes wrong:** Eval_check uses SLM-staged poly_mix (R2-A10 documents this for
  rv32im; recursion's eval_check.cpp doesn't stage poly_mix today, so this risk is
  smaller). If staging is added later behind the same env knob, smaller WG = smaller
  SLM stride; barrier coverage may differ. The pure-WG-change today is bit-safe (R2-A03
  sec. "Risk register: Hangs or non-determinism = None — Pure scheduling change,
  identical math").
- **Likelihood:** Near-zero today.
- **Blast radius:** Seal mismatch.
- **Detection:** Seal-byte equality check (R2-A03 acceptance #1) at every whitelisted
  WG.
- **Mitigation:** Run the 5×{256,512,1024} sweep with full seal-byte comparison
  before declaring any non-1024 default.

### Risk 1.4 — Stale build cache hides the toggle
- **What goes wrong:** `recursion-sys/build.rs` has no `cargo:rerun-if-env-changed=
  RISC0_RECURSION_EVAL_CHECK_WG` because the env-var is read at *runtime* by the
  loaded .so. But IF the first rebuild after the diff is short-circuited (kernel
  cache hit), `eval_check.cpp:43-55` would still embed the old `constexpr` and the
  env var becomes a no-op.
- **Likelihood:** Low IF cache hash includes eval_check.cpp content (R2-A05 sec. F3
  shows the stamp hash mechanism). Verify on first build.
- **Blast radius:** All WG values silently produce the WG=1024 result, masking any
  actual win.
- **Detection:** Print "WG=%u" inside the kernel launcher behind `RISC0_VERBOSE=1`
  during the first sweep; or `nm -D` the .so and confirm `getenv` symbol is present.
- **Mitigation:** `cargo clean -p risc0-circuit-recursion-sys` before the first
  sweep; or `touch recursion-sys/build.rs` (R2-A03 sec. "Risk register" entry).

---

## Item 2 — `poly_fp.cpp` split + `RISC0_RECURSION_OPTIMIZE` (R1-11 / R2-A04)

**Status:** Not started. R3-A06's existing `RISC0_RECURSION_OPTIMIZE=1` experiment
ran ocloc for 1h 24min with no result (R1-11 §"TL;DR") on the 24K-line monolithic
poly_fp.cpp.

### Risk 2.1 — Mechanical split misses an SSA var crossing a cut → bit-mismatch
- **What goes wrong:** The splitter (`split_poly_fp.py` in R2-A04 sec. 4 step 1)
  enumerates live FpExt across cuts assuming all `auto`/`Fp` vars are FpExt-untainted.
  R2-A04 sec. 1 verified this with `recursion_cutpoint_analysis.py`: 0 tainted Fps.
  If the assertion fires false-negative (e.g., an `auto` that the parser misclassified
  as constant, or a regex miss on an indirect FpExt dependency), the sub-function
  bodies are no longer semantically equivalent → poly_fp returns a different FpExt →
  recursion seal mismatch.
- **Likelihood:** Low. R2-A04 sec. 7 cites three layers of defense: empirical
  measurement (0 tainted Fps verified), regex-pattern reuse of `poly_fp_cse.py`
  (mature parser), and bit-exactness checking gate.
- **Blast radius:** Recursion seal mismatch. Caught at runtime; no silent
  miscompute risk because every recursion proof verifies its own STARK.
- **Detection:** Static check (every original line appears in exactly one sub-fn,
  R2-A04 sec. 4 step 4a) plus runtime byte-equality of `FpExt` returned by `poly_fp`
  against CPU/CUDA reference (R2-A04 sec. 4 step 4b).
- **Mitigation:** Build the splitter to fail loudly if the tainted-Fp invariant
  breaks (R2-A04 risk table row 6 "Some FpExt-tainted Fp escapes our taint
  analysis... Assertion in the splitter that the tainted-Fp set is empty fails
  the build if violated"). Keep the original `poly_fp.cpp` in-tree as a reference
  for CPU/CUDA (R2-A04 sec. 6).

### Risk 2.2 — Split doesn't actually reduce IGC compile time below the 1h24m wall
- **What goes wrong:** R1-11 hypothesized the IGC blow-up is intra-function SSA
  chain depth, fixable by splitting. If the *real* pathology is elsewhere (e.g.,
  PromotePrivateArrayToReg cost on the global args[] struct, or InstCombine on the
  shared constants), splitting into N functions doesn't help.
- **Likelihood:** Low-medium. R1-11 sec. "Why specifically does ocloc go exponential"
  identifies pre-RA scheduler register-pressure analysis (O(N²) in live set) and
  GenIR-time CSE/InstCombine as candidates — both are *per-function* passes, so
  splitting should help. But R2-A11 sec. 1.5 (T3.2 tree-reduce 2.3× regression)
  warns: "ANY code-gen transform proposed for the recursion port needs zebin
  metadata baseline+after measurement, not just E2E timing."
- **Blast radius:** 2-4 days of work spent for no compile-time win. Recursion stays
  on `-O1 -cl-opt-disable`. The split itself is still beneficial (~1-3% E2E in
  R2-A04 sec. "TL;DR") only if it then unlocks `-O2 -cl-intel-256-GRF-per-thread`.
- **Detection:** R2-A04 sec. 4 step 5: ocloc completes in 20-40 min vs 84 min today
  with the split. If still >60 min, R2-A04 recommends bumping N from 10 → 15 → 20
  (rv32im uses 20).
- **Mitigation:** Bounded experiments first (R1-11 sec. "Recommended next session
  sequencing"): try `-O1` without `-cl-opt-disable` (2h bounded), then `-Os`
  (2h bounded), then 128-GRF (2h bounded). 6 hours total before committing to
  the multi-day split.

### Risk 2.3 — Split + -O2 + 256-GRF gives the **wrong** scratch / spill profile (T3.2 lesson)
- **What goes wrong:** T3.2 FMA tree-reduce regressed E2E 2.3× even though it was
  mathematically bit-exact — SSA explosion blew `private_size` 527 KB → 3091 KB
  on a spill-bound kernel (memory `project_tree_reduce_negative.md`). Splitting
  poly_fp + enabling `-O2` may move recursion eval_check from a comfortable
  low-spill regime to a high-spill regime if the new sub-function ABI cost
  (3-9 FpExt + 4 buffer ptrs per call × 10 sub-fns) doesn't fit GRF.
- **Likelihood:** Low-medium. R2-A04 sec. "Risks and mitigations" row 5 dismisses
  call ABI cost ("50ns per call × 10 = 500ns vs 200ms kernel work"). But the
  T3.2 precedent means we cannot trust an a-priori analysis.
- **Blast radius:** E2E regression up to ~2×. Caught by an A/B before commit.
- **Detection:** R2-A11 L3 ("Zebin metadata diff is mandatory"): inspect
  before/after `private_size`, `spill_size`, `grf_count`. ANY regression >20% in
  the first two = abort.
- **Mitigation:** Zebin metadata baseline measurement before and after the split
  PR. Run a stacked-patches v1 5×5 A/B across multi-PO2 (14, 18, 21, 22 — R2-A11
  L2) for any default-flag change.

### Risk 2.4 — `RISC0_RECURSION_OPTIMIZE` patched-IGC dependency (silent fallback)
- **What goes wrong:** Memory `project_igc_patch_required.md` records that stock
  IGC 2.30.1 SIGSEGVs on the rv32im witgen amalgamation; `build.rs::
  try_recover_witgen_so()` masks the failure by copying a stale .so. The recursion
  port introduces a SECOND witgen amalgamation when Item 6 lands; for Item 2,
  the *eval_check* amalgamation gets larger if poly_fp is split + -O2 enabled.
  Stock IGC may also SIGSEGV here, and silent fallback would mean later builds
  use the un-split .so.
- **Likelihood:** Medium for Item 6's witgen kernel (R2-A11 sec. 1.1
  "Recursion exposure: High"). Lower for Item 2's poly_fp split — the patched
  IGC has been tested only on rv32im's 52K-LOC witgen amalg, not on a recursion-
  optimized -O2 path.
- **Blast radius:** All "performance" measurements are against a stale binary.
  Silent waste of sweep budget.
- **Detection:** R2-A11 L1 sec. "Mitigation": loud `cargo:warning=RECURSION WITGEN
  BUILD FAILED, USING STALE .so — your source edits did NOT take effect`. Verify
  `intel_recursion.stamp` `status` field reads `ok`, not `stale_recovered`.
- **Mitigation:** Item 2's build.rs change must include an explicit `panic!`
  rather than silent fallback. R2-A11 sec. 4 Scenario D ("recursion amalgamation
  hits a DIFFERENT IGC bug") warns that the recursion port may need its OWN
  patched IGC.

---

## Item 3 — Cross-segment CPU/GPU pipelining (R1-12 / R2-A16 sec. 3)

**Status:** Not started. The "pipelined" override at `prover_impl.rs:1159` is
ALREADY in place (R1-A05 sec. "What's already been done"), but it only overlaps
CPU preflight (~20-25ms) with GPU prove (~1.2s) — R1-A01 sec. "Finding 2" shows
this saves ~0.8% E2E because preflight is so cheap relative to prove. **The
unrealized lever is overlapping lift N+1's CPU witgen (1.5-2s) with lift N's GPU
eval_check (~400ms)**, per R1-12 sec. Q2 "Exception."

### Risk 3.1 — Lift N+1 witgen can't actually start until lift N's data is loaded
   into the GPU buffer pool
- **What goes wrong:** Lifts share the IntelHal singleton's BUFFER_POOL.
  If lift N is still mid-flight when lift N+1's CPU witgen starts, the CPU
  needs the *post-witgen* data buffer for lift N+1's accum FFI — and that
  buffer is currently allocated only after lift N completes its destroy
  cycle (`intel.rs:104-126`). Overlap requires double-buffering the
  HAL's per-lift buffers.
- **Likelihood:** High. This is the structural reason R1-12 Q5 calls out
  "non-trivial (changes ExecContext lifetime / GPU queue scheduling)."
- **Blast radius:** Implementation effort 1-2 weeks; if buffer-pool design
  is wrong, OOM under sustained 42-lift runs (memory
  `project_fib_sigabrt_root_cause.md` records USM accumulation as a known
  driver bug).
- **Detection:** Sustained 5-segment fib runs at PO2=18 under
  `RISC0_VERBOSE=1`; watch for `esimd_malloc_device failed` exceptions
  (the SIGABRT signature from the cited memory note).
- **Mitigation:** Pre-budget a double-buffer design (R2-A11 L6 "buffer-pool
  design pass"). Use thread_local pool, not process-global Mutex (R2-A11
  L6 sec.: "learning from its SIGABRT failure mode").

### Risk 3.2 — Win vanishes when Item 6 (witgen GPU port) lands
- **What goes wrong:** R1-12 Q2: "After the full R3-A06 port (step_exec/accum
  on GPU), lifts become GPU-bound, and pipelining would also fail (single
  GPU queue). So pipelining is a dead end on a single-GPU box in both
  regimes... CPU lift + GPU lift overlap... is a 'before full port'
  opportunity."
- **Likelihood:** Certain IF Item 6 lands at any future date.
- **Blast radius:** All Item-3 engineering work (1-2 weeks) is wasted at
  Item 6 landing. Net E2E uplift of Item 3 lasts only as long as Item 6
  is unfinished.
- **Detection:** Trivial: after Item 6 lands, the wall-clock split for
  lift N would show "witgen GPU 80ms + eval_check GPU 400ms = 480ms" with
  no CPU window to overlap.
- **Mitigation:** **Decision gate** — if Item 6 is realistically going to
  land in this 2-week sprint, **skip Item 3**. R3-A01's ceiling of 6-10%
  E2E for the full port and the explicit Phase B "GO/NO-GO" structure
  (R2-A16 §5) suggest Item 6 is multi-month, so Item 3's 1-2 week
  investment may pay back for ≥2 quarters. But mark explicitly as a
  bridge optimization.

### Risk 3.3 — CPU witgen oversubscribes 32 cores when overlapping with itself
- **What goes wrong:** Lift N+1's CPU witgen uses 32 poolstl threads. If
  lift N's CPU witgen hasn't *fully* completed (e.g., post-witgen verify_wom
  on host), the two run concurrently → 64 threads on 32 cores, cache thrash,
  net slowdown. R1-12 Q2: "we cannot run two lifts in parallel on the same
  32-core CPU — they'd thrash the cache and each lift's poolstl::par_unseq
  would oversubscribe."
- **Likelihood:** High if implementation doesn't carefully sequence the CPU
  phase.
- **Blast radius:** Pipelined version regresses against baseline.
- **Detection:** Wall-clock A/B; the regression is immediate. R2-A11 L2
  "Multi-PO2 + multi-segment A/B from day one" catches.
- **Mitigation:** Strict invariant: lift N CPU work must complete before
  lift N+1 CPU work starts; the overlap is *only* between lift N+1 CPU and
  lift N GPU. Implement as a `Mutex<()>` for the CPU side and a queue for
  the GPU side; the queue's wait events gate the GPU side.

### Risk 3.4 — Memory note about CPU/GPU overlap (~6.5%) doesn't reflect
   post-pipelined-override baseline
- **What goes wrong:** R1-A12 estimated "~16 s saved out of 99 s = ~6.5%
  Succinct E2E" assuming the current pipelined override does NOT already
  overlap GPU phases across lifts. R1-A01 sec. "Finding 2" measured the
  current overlap saves only ~0.8%, suggesting room exists. But the exact
  overlap window depends on per-segment timing variability — if witgen is
  occasionally < 400ms (short PO2) or eval_check > 1.5s (long PO2),
  realized savings will differ.
- **Likelihood:** Medium — the 6.5% estimate is averaged.
- **Blast radius:** Realized E2E gain may be ~2-4% rather than the 6.5%
  estimate, dragging EV per dev-week.
- **Detection:** R2-A11 L2 multi-PO2 sweep (14, 18, 21, 22) before declaring
  Item 3 a win.
- **Mitigation:** Set expectations at "2-6% E2E" not "6.5%" when motivating
  the work to stakeholders.

---

## Item 4 — oneDPL sort + persistent USM for WomRow (R2-A09)

**Status:** Not started. R2-A09 verified oneDPL is installed (`/opt/intel/oneapi/dpl/
latest`) and the sort+scan site is `recursion-sys/kernels/cxx/ffi.cpp:119-134`.
Two flavors: (A) drop-in copy-in/sort/copy-out and (B) persistent USM with
`injectWomBacks` + `doStepVerifyWom` on device.

### Risk 4.1 — Flavor A's PCIe round-trip eats the sort win
- **What goes wrong:** R2-A09 sec. 6: "kernel-only speedup ~4-6× on the sort
  itself. Round-trip eats half of that in flavor (A) → real per-lift saving
  ~25-65 ms." For 42 lifts × ~50 ms ≈ 2 s ≈ **~2% E2E in isolation**. The
  PCIe cost is fundamental, not optimizable.
- **Likelihood:** Certain.
- **Blast radius:** Item 4 standalone (Flavor A) lands ~2% E2E max — within
  the 6-10% R3-A01 ceiling but below the engineering threshold (1 week of
  work for 2% E2E is consistent with the rest of the menu).
- **Detection:** Time the sort+scan + memcpy_htod + memcpy_dtoh before-and-
  after. The PCIe cost is the ~46 MB H2D + 46 MB D2H per lift = ~3.6 ms each
  way at PCIe 4.0 x16 (~26 GB/s).
- **Mitigation:** **Plan flavor B from the start.** R2-A09 sec. 8: "Port as
  flavor (B), not (A)." This requires the womRows/womIndex buffers to live
  in USM across the entire MachineContext lifetime, plus injectWomBacks
  and doStepVerifyWom rewritten as device kernels (the same target as
  Item 6 — flavor B is essentially a Item-6 sub-task).

### Risk 4.2 — `oneapi::dpl::sort` won't accept `WomArgumentRow` as a value type
- **What goes wrong:** R2-A09 risk #2 ("oneDPL doesn't compile/work with FpExt
  as the scan type") warns the struct may lack the trivially-copyable /
  default-constructible markers oneDPL needs. `WomArgumentRow` is 5 ×
  `uint32_t` = 20 B, POD, trivially memcpy-able (R2-A09 sec. 2 confirms). But
  the comparator is custom — oneDPL needs it inlinable on device.
- **Likelihood:** Low. R2-A09 sec. 3b "device-callable comparator" is a plain
  lambda; sec. 4 confirms non-stable sort is OK ("CUDA already uses non-stable
  thrust::sort and is bit-exact with the CPU std::sort(poolstl::par)").
- **Blast radius:** If oneDPL fails, fall back to custom kernel (~50 LOC
  Hillis-Steele scan or a sub-group radix sort). 0.5-1 day of extra work.
- **Detection:** First compile of the SYCL TU containing the oneDPL
  invocation; compile error or runtime exception.
- **Mitigation:** R2-A09 sec. 7 "Compile-time hit: oneDPL is heavy template-
  wise; expect +20-40 s on the SYCL TU." Add to the compile-time budget.
  R2-A09 sec. 6 fallback: 1-WG Hillis-Steele scan kernel.

### Risk 4.3 — Sort is **bit-safe** but `injectWomBacks` order isn't (post-sort)
- **What goes wrong:** Sort itself is safe (R1-A07 R3 sec. "Conclusion: sort
  algorithm choice... is BIT-SAFE under operator< as written"). But
  `injectWomBacks` (`ffi.cpp:142-156`) writes to `data[col*steps + cycle - 1]`
  using `womRows[idx-1]` data. If the device-side injectWomBacks kernel
  doesn't WAIT for sort completion (no `q.wait()` between phases), it reads
  garbage.
- **Likelihood:** Low if SYCL in-order queue is used; high if events are
  manually managed.
- **Blast radius:** All recursion seals diverge from CPU baseline. Caught
  by the differential harness (Item 5).
- **Detection:** R1-A07 R9 "Harness: dump the 5 cells `data[col*steps + cycle -
  1]` for col=0..4 after injectWomBacks but before stepVerifyWom; compare
  against CPU."
- **Mitigation:** Use SYCL in-order queue (already in place per R2-A09 sec.
  3a). Each phase calls `q->wait()` between sort/scan/injectWomBacks/
  doStepVerifyWom.

### Risk 4.4 — Persistent USM doubles peak memory at PO2=22
- **What goes wrong:** R2-A09 sec. 5 corrects the PO2: real working set at
  PO2=18 is 2.36M womRows = 45 MB, plus ~45 MB sort scratch = ~91 MB peak
  transient. At PO2=22 this is 16× larger = ~1.4 GB. B70 HBM is ample
  (16 GB), but persistent USM keeps these allocated across all 83 ops in
  Succinct → 1.4 GB × 2-3 buffers stays resident. May push other
  allocations (eval_check 1 GB+) over a soft limit.
- **Likelihood:** Low at PO2=18 (the default); medium-high at PO2=22.
- **Blast radius:** OOM at large PO2. Multi-PO2 testing (R2-A11 L2) catches.
- **Detection:** Track peak USM allocation across a Succinct prove at each
  PO2. R2-A09 sec. 5 already estimates the working set.
- **Mitigation:** Reuse the buffer pool from R2-A08 (rv32im's pattern, already
  in place at `zkp/src/hal/intel.rs:248-292`). Add LRU eviction for
  womRows/womIndex on memory pressure.

---

## Item 5 — Differential harness (R3-A08 / R2-A06)

**Status:** Skeleton designed (R3-A08 = 50-100 LOC of Rust, F1-F7 deliverables
per R2-A06). Required BEFORE Item 6 step_compute_accum PoC begins (R2-A16 §3
Phase A item 2; R1-A17 "REQUIRED before any kernel work begins").

### Risk 5.1 — Harness covers only `data`/`global`; misses `iopIdx`, `womIndex`,
   `womRows` post-sort
- **What goes wrong:** R3-A08 explicitly enumerates 6 checks but the day-1
  scaffolding only wires `data` and `global` (R3-A08 sec. "Checks 3-6 ...
  TODO(R3-A06 week 2)"). If Item 6 PR lands without the TODO removal, race
  conditions in `iopIdx++` or `womIndex[cycle]++` (R1-A07 R1, R2) are
  silently undetected. R1-A07 R5 documents 4,429 `Fp::invalid()` mentions
  in step_exec.cpp; a wrong write that the CPU assert would catch is silent
  on GPU because `#define assert(x) ((void)0)` (R1-A07 R5 sec. a).
- **Likelihood:** Medium-high if review discipline is lax.
- **Blast radius:** GPU witgen lands as "bit-exact" by harness criteria,
  but a real production workload at PO2=22 with high non-par-safe density
  produces wrong seals.
- **Detection:** R3-A08 acceptance: "TODO comments for checks 3-6 are linted
  as `// TODO(R3-A06 week 2):` so the kernel-port PR cannot land without
  removing them (CI grep on diff: a removed TODO without a corresponding
  `assert_buffer_eq` call is a review-time red flag)."
- **Mitigation:** Treat the TODO grep as a CI gate. R1-A07 sec. "Recommended
  validation harness" enumerates all 6 checks; R3-A08 maps them to specific
  MachineContext fields (`context.h:52,70-71`). Reject any Item 6 PR that
  doesn't populate all 6.

### Risk 5.2 — `EnvGuard` deadlocks because libtest runs parallel tests
- **What goes wrong:** R3-A08 sec. "What this agent adds": "an outer `cargo
  test` invocation with `RISC0_RECURSION_GPU_WITGEN_VERIFY=1` in the env
  would otherwise cause the in-process probe to fire inside the test,
  doubling wall time and confusing failure attribution." Even with
  `--test-threads=1` (R3-A08 acceptance), nested test runs from other
  workspaces could still hit it.
- **Likelihood:** Low.
- **Blast radius:** Confusing test failures, but not silent.
- **Detection:** `--test-threads=1` mandatory invocation; document in
  CONTRIBUTING.
- **Mitigation:** R3-A08 already uses `ENV_LOCK` (process-global Mutex
  pattern from rv32im `intel.rs:341-499`).

### Risk 5.3 — Harness only runs at PO2=14/16/18 — doesn't catch PO2=22 bugs
- **What goes wrong:** R3-A08 sec. "Fixtures" lists PO2 ∈ {14, 16, 18, 18,
  18, 18, 21}. The `identity_p254` fixture covers PO2=21, but PO2=22 is
  used by long-chain Succinct workloads (R1-A14 R2-A16 §3 "Multi-PO2
  sweep (14, 18, 21, 22)").
- **Likelihood:** Medium for PO2=22-specific bugs (e.g., a 22-cycle chain
  exceeding 4096 row buckets, exposing an integer-overflow bug).
- **Blast radius:** Production Succinct seal mismatch at PO2=22 only,
  flaky/random.
- **Detection:** Add a PO2=22 fixture (would require capturing a real
  long-chain join preflight; non-trivial — likely 0.5 day of fixture work).
- **Mitigation:** R2-A11 L2 makes multi-PO2 a hard gate. Until PO2=22
  fixture exists, do NOT declare Item 6 done; the missing PO2 is a known
  gap.

### Risk 5.4 — Capturing the snapshot from a "known-good CPU path" is
   itself slow/flaky
- **What goes wrong:** R2-A05 sec. F7 ("Capturing the snapshot") says
  `RISC0_DUMP_ACCUM_SNAPSHOT=1` is a one-shot capture during a known-good
  CPU run. If the run is non-deterministic (e.g., a poolstl::par sort over
  ties that resolves differently per-thread-schedule), the captured
  snapshot is wrong.
- **Likelihood:** Low. R1-A07 R3 "Sort algorithm choice... is BIT-SAFE under
  operator< as written"; deterministic across thread scheduling.
- **Blast radius:** Differential harness shows false-positive mismatches.
- **Detection:** Capture the snapshot 3 times, byte-compare; if differs,
  the snapshot capture itself is broken.
- **Mitigation:** Document the snapshot SHA-256 in the test file; CI
  validates the in-tree snapshot still matches its documented hash.

---

## Item 6 — Full `step_exec` port (R3-A06 main lever, R2-A16 Phase C)

**Status:** Not started. Conditional GO based on Item 5 (harness) + Item 6's
own Phase B PoC checkpoint (the `step_compute_accum` smaller-kernel
validation, R2-A05). Realistic effort 4-7 weeks (R2-A16 §5).

### Risk 6.1 — IGC compile-time pathology on 53K-LOC step_exec
- **What goes wrong:** R2-A11 sec. 4 Scenario D: "Recursion port may need
  its OWN IGC patch." R1-A11 documented the 24K-LOC poly_fp.cpp hanging
  ocloc for 1h 24min; step_exec.cpp is 40K LOC (R3-A06 sec. 3). Even with
  patched IGC + poly_fp split (Item 2), step_exec amalgamation may hit a
  *different* IGC bug.
- **Likelihood:** Medium-high. R2-A11 sec. 4 Scenario D: "1h+ behavior is
  already a known issue, not a forecast" for the recursion poly_fp; a
  larger step_exec is more exposed.
- **Blast radius:** Phase C blocker. Could push timeline 2-4 weeks for
  IGC bug triage + patch.
- **Detection:** Bounded ocloc time per R1-A11 sec. "Recommended next
  session sequencing" (kill at 90 min); if persistent, capture
  `IGC_PRINT_SHADER_AST=1` + `IGC_DumpToCurrentDir=1` (R2-A11 sec. 4
  Scenario D diagnosis path).
- **Mitigation:** **Start Item 6 with the smaller PoC (`step_compute_accum`,
  15K LOC, R2-A05).** R2-A16 §5 Phase B GO/NO-GO: "if IGC chokes on 15K
  LOC of FpExt SSA (similar to recursion eval_check's earlier 1h24m
  timeout), Phase C is high-risk. Bail to 'polish Phase A wins only' if
  checkpoint fails." Apply the structural split lessons from Item 2 to
  step_exec preemptively.

### Risk 6.2 — `parStepExec` walk for non-par-safe chains not replicated on GPU
- **What goes wrong:** R1-A07 R1 + R8 detail this. Recursion has cross-cycle
  reads (`(cycle - 1)`, ~30+ hits in step_exec.cpp). CPU walks the chain
  via `MachineContext::parStepExec` (`ffi.cpp:60-69`); a naive `parallel_for`
  on GPU sees stale `Fp::invalid()` at `data[cycle - 1]` for non-par-safe
  cycles.
- **Likelihood:** Certain if implemented naively.
- **Blast radius:** All non-par-safe-dense recursion seals diverge from
  CPU. Lift fails.
- **Detection:** R3-A08 Check #1 (data buffer post-witgen) + Check #3 (iopIdx
  post-execution). R1-A07 R8: "After GPU witgen, dump the data buffer
  post-witgen and assert it matches CPU's byte-for-byte at every (col,
  cycle). If any column at cycle N differs from CPU's column at cycle N,
  the par-safe chain or the back-N read was botched."
- **Mitigation:** Mirror CUDA's `nextStepExec` (R1-A07 R1 sec. "The CUDA
  port preserves this (ffi.cu:51-61: `nextStepExec` does the same walk
  on-device)"). Effective parallelism is ~10-30% (R1-A06), capping the
  win.

### Risk 6.3 — `Fp::invalid()` sentinel writes silent on GPU (assert no-op)
- **What goes wrong:** R1-A07 R5 marks this **HIGH — likely BREAKER**.
  4,429 `Fp::invalid()` mentions in step_exec.cpp implement a
  WRITE-ONCE-OR-MATCH semantic; the assert that catches accidental
  conflicting writes is `#define assert(x) ((void)0)` on GPU. A wrong
  mux arm fires → cell holds wrong value → downstream poly_fp + eval_check
  produce a different but plausible proof. Receipt no longer bit-equal
  to CPU; neither side panics.
- **Likelihood:** Medium. Any porting bug in the mux dispatch silently
  produces wrong proofs.
- **Blast radius:** Subtle seal mismatches that pass STARK verification
  but break downstream Succinct/Groth16 cross-verifications. Costly to
  diagnose.
- **Detection:** R1-A07 R5: "after witgen, do `data_cpu == data_gpu`
  element-wise INCLUDING the `Fp::invalid()` sentinel cells (which must
  remain `Fp::invalid()` on both)."
- **Mitigation:** R3-A08 Check #1 explicitly includes sentinel cells.
  Multi-PO2 sweep gates the merge.

### Risk 6.4 — R3-A01's hard ceiling (10.7%) means Item 6 EV is bounded; realized
   may be much lower (PCIe+kernel ~6%)
- **What goes wrong:** R3-A01's authoritative ceiling: witgen+accum is only
  10.7% of Succinct E2E on Intel today (323 ms/op × 83 ops / 249.65 s).
  Even free GPU compute (0 ms) + zero PCIe = max **10.7%** E2E. Realistic
  GPU compute of 80 ms/op + persistent USM (PCIe ~0) = ~243 ms saved/op
  × 83 = 20.2 s = **8.1%** E2E (R3-A01 Table "Realistic mid"). R2-A14's
  28-32% claim and R2-A16's 15-25% target are both arithmetically wrong.
- **Likelihood:** Certain.
- **Blast radius:** Stakeholder credibility risk. 7-11 engineering weeks
  for a 6-10% win is a 0.9-1.4% E2E per dev-week ratio (R3-A01 sec.
  "Implications for R3-A06 plan"), comparable to other items on the menu
  (memory `project_optimization_landscape.md` ranks Tier-2 WG sweeps at
  ~5% combined for days of work).
- **Detection:** Daily E2E timing on `prove_and_verify 100000 succinct`
  with `RISC0_VERBOSE=1` per-phase output. Compare to R3-A01's anchors.
- **Mitigation:** R3-A01 sec. "Implications" #1: "Headline number: drop
  to 6-10% Succinct E2E speedup." Manage expectations BEFORE the project
  starts. Bail to lower-effort items if Phase B PoC measurements project
  <5% E2E.

### Risk 6.5 — USM accumulation / SIGABRT scales with allocations (R2-A11 sec. 1.7)
- **What goes wrong:** Memory `project_fib_sigabrt_root_cause.md` records
  Intel Compute Runtime accumulates USM across processes — driver garbage
  collection is event-driven, not deterministic; 40-60% of consecutive
  fib invocations exit with SIGABRT. Item 6 adds 5 device buffers × 42
  lifts = 210 alloc/free events per Succinct prove. R2-A11 sec. 1.7:
  "USM accumulation could surface faster."
- **Likelihood:** Medium-high under sustained workloads (CI, soak).
- **Blast radius:** Periodic CI failures, especially in CI environments
  without 30-60s cooldowns between processes.
- **Detection:** 1000-iteration soak at end-of-port (modeled on rv32im's
  R2-A16 §6 Day 10).
- **Mitigation:** R2-A11 L6: "Pre-budget a buffer-pool design pass
  (analogous to rv32im's R2-A08, but learning from its SIGABRT failure
  mode — use thread_local pool, not process-global Mutex)." Item 6's
  buffer pool design must be in scope from day one.

### Risk 6.6 — `extern_log` / `std::vector<Fp>` not device-compatible
- **What goes wrong:** R2-A05 sec. F1: "SYCL device code can't use
  `std::vector`. The cxx version uses `std::vector<Fp>` in `extern_log`."
  step_compute_accum doesn't call extern_log (grep verified 0 hits), but
  step_exec.cpp does (R1-A07 R6 lists extern_log among recursion's
  externs). Step_exec port needs a device-side stub.
- **Likelihood:** Certain.
- **Blast radius:** Compile failure on step_exec amalgamation. Caught at
  build time, not runtime — easy to fix once located.
- **Detection:** Compile errors on first amalgamation build.
- **Mitigation:** R2-A05 sec. F1: "templated no-op overload accepting
  `(void*, size_t, const char*, T)` for any T." Apply same pattern.

### Risk 6.7 — Headline numbers (lift=1.16s/op, ops=83) already corrected; old
   plan still cited in social context
- **What goes wrong:** R1-A01 sec. "Finding 1" corrected the plan's 2.4 s/lift
  arithmetic to 1.16 s/lift + 1.24 s/join × 83 ops = 99.7 s. The "3.4×
  slower than CUDA" headline overstates by 2× (true ratio is 1.65×). If
  stakeholder communication still uses the old framing, expectations
  diverge from achievable.
- **Likelihood:** High if not actively corrected.
- **Blast radius:** Reputational; affects go/no-go decisions.
- **Detection:** Read any communication artifact (PR description, slack
  thread) before send.
- **Mitigation:** Update memory note `project_r3_a06_recursion_port_plan.md`
  with R3-A01 + R1-A01 + R2-A16 corrections explicitly. Use
  "**target 6-10% E2E**" not "20-35%" in all communication.

---

## Cross-cutting risks (apply to all items)

### CC1 — Patched IGC dependency lost / wiped
- All items that touch any Intel kernel inherit the requirement (memory
  `project_igc_patch_required.md`). Item 1's eval_check rebuild + Item 6's
  step_exec amalgamation + Item 2's poly_fp split = at least three
  workflows that silently fail with stock IGC.
- **Mitigation:** R2-A11 L1 (pre-flight check) + `cargo:warning=` loud
  fallback. Archive the patch source outside the working directory (memory
  note's Scenario C).

### CC2 — Multi-PO2 not enforced as a gate
- R2-A11 L2: "no patch lands until validated on po2 ∈ {14, 18, 21, 22} ×
  seg-count ∈ {1, 5, 42}." Single-segment benchmarks masked the rv32im
  accum-split regression by 1-2 days (memory record).
- **Mitigation:** Enforce in CI; the rv32im post-merge pattern.

### CC3 — Single-segment benchmarks mask regressions
- Same root cause as CC2.
- **Mitigation:** Stacked-patches v1 5×5 A/B (memory
  `project_stacked_patches_v1.md`) with 25s cooldown is the validation
  template.

### CC4 — Tuning tail consumes 4-8 weeks beyond Phase A landing
- R2-A11 sec. 1.9: rv32im needed 13 distinct optimization commits across
  6 calendar weeks. "Phase A" (correctness) + "Phase B" (tuning) framing
  is mandatory.
- **Mitigation:** R2-A11 L7: "Mark the project as having TWO phases."

### CC5 — Effort underestimated by ~2× per item
- R2-A16 §2 estimate: 7-11 weeks vs plan's 3-5. R2-A11 §5 with IGC risks:
  9-14 weeks.
- **Mitigation:** R2-A16 conditional GO with Phase B GO/NO-GO checkpoint.

---

## Summary table

| Item | Top risk | Likelihood | Blast radius | Detect | Mitigate |
|---|---|---|---|---|---|
| 1: EVAL_CHECK_WG knob | WG=256 spills under -cl-opt-disable | Med-High | Cosmetic | Per-WG wall time | Document, leave default 1024 |
| 1: EVAL_CHECK_WG knob | Future -O2 rebuild exceeds 1024 cap | Med IF Item 2 lands | Launch error | CL_INVALID_WORK_GROUP_SIZE | Coupled change: drop default to 512 when Item 2 lands |
| 2: poly_fp split | Splitter misses tainted Fp var | Low | Seal mismatch | Static line-count + runtime byte-equal | Splitter assertion + golden compare |
| 2: poly_fp split | Split doesn't reduce IGC time below 1h | Low-Med | 2-4 wasted days | Bounded ocloc time | Try -O1/-Os bounded first (6 h budget) |
| 2: poly_fp split | -O2 + 256-GRF blows spill (T3.2 lesson) | Low-Med | E2E up to 2× regression | Zebin metadata diff | Mandatory pre/post zebin measurement |
| 2: poly_fp split | Patched-IGC silent fallback | Med | Stale .so | stamp `status` field | Loud `cargo:warning=` |
| 3: Cross-seg pipelining | Buffer pool can't double-buffer | High | 1-2 wk rework | OOM under soak | thread_local pool, R2-A08-style |
| 3: Cross-seg pipelining | Vanishes when Item 6 lands | Certain IF Item 6 lands | All Item-3 work wasted | Trivial timing comparison | Decision gate: skip if Item 6 imminent |
| 3: Cross-seg pipelining | CPU oversubscribes 32 cores | High if not sequenced | Pipelined regresses | Wall-clock A/B | Strict CPU-side Mutex |
| 3: Cross-seg pipelining | 6.5% may be 2-4% realized | Med | EV erosion | Multi-PO2 timing | Set expectations 2-6%, not 6.5% |
| 4: oneDPL sort + USM | Flavor A PCIe eats win | Certain | ~2% E2E only | Time round-trip | Plan flavor B (USM-resident) from start |
| 4: oneDPL sort + USM | oneDPL doesn't accept WomArgumentRow | Low | 0.5-1d fallback | First compile | Hillis-Steele 50-LOC fallback |
| 4: oneDPL sort + USM | injectWomBacks reads pre-sort data | Low w/ in-order queue | All seals diverge | data[cycle-1] dump | q.wait() between phases |
| 4: oneDPL sort + USM | Peak USM doubles at PO2=22 | Med-high at PO2=22 | OOM | Memory tracking | LRU eviction; buffer pool reuse |
| 5: Differential harness | Day-1 only covers data/global | Med-high | Silent GPU bugs | TODO grep CI gate | Block merge if TODO not removed |
| 5: Differential harness | EnvGuard deadlock | Low | Test-time confusion | --test-threads=1 | ENV_LOCK from rv32im |
| 5: Differential harness | PO2=22 missing fixture | Med | Long-chain seal mismatch | Capture PO2=22 fixture | Mandatory before Item 6 declares done |
| 5: Differential harness | Snapshot non-deterministic | Low | False-positive mismatches | Triple capture | SHA-256 documented in test |
| 6: Full step_exec port | IGC pathology on 40K-LOC step_exec | Med-high | 2-4 week timeline slip | Bounded ocloc, IGC AST dump | PoC step_compute_accum first (Phase B GO/NO-GO) |
| 6: Full step_exec port | parStepExec walk not replicated | Certain if naive | All seals diverge | Per-cycle data buffer compare | Mirror CUDA nextStepExec |
| 6: Full step_exec port | Fp::invalid() sentinel silent on GPU | Med (porting bugs) | Subtle silent mismatches | Sentinel-included element-wise compare | R3-A08 Check #1 + multi-PO2 |
| 6: Full step_exec port | R3-A01 ceiling 6-10%, not 20-35% | Certain | Stakeholder credibility | Per-op timing vs anchor | Update memory note + comms |
| 6: Full step_exec port | USM SIGABRT at scale | Med-high under soak | CI failures | 1000-iter soak | Buffer-pool design (R2-A11 L6) |
| 6: Full step_exec port | extern_log std::vector | Certain | Compile failure | First build | Templated no-op overload |
| 6: Full step_exec port | Old 20-35% framing leaks | High | Reputational | Pre-send review | Memory note correction |

---

## Recommendation summary

| Item | Risk profile | Recommended action |
|---|---|---|
| 1: WG knob | Low risk, low EV (~0-0.3%) | Land it; freebie housekeeping (R2-A03 verdict). |
| 2: poly_fp split | Medium risk (T3.2 precedent), medium EV (1-3% E2E) | Bounded -O1/-Os experiments first (6 h), then 2-4 day split if needed. |
| 3: Cross-seg pipelining | High risk (buffer pool, oversubscription), medium EV (2-6%) | **Conditional GO**: only if Item 6 won't land in 6 months. |
| 4: oneDPL sort + USM | Low risk (proven primitive), low standalone EV (~2%) | Skip Flavor A; co-implement Flavor B with Item 6's USM-resident plumbing. |
| 5: Differential harness | **Pre-requisite, low risk, infinite EV** (gates Items 4B, 6) | **MUST land first**. 50-100 LOC. R3-A08 design ready. |
| 6: Full step_exec port | High risk, ceiling 6-10% per R3-A01 | **Conditional GO** via Phase B PoC checkpoint (step_compute_accum); bail if IGC chokes. |

**Total realistic Phase A delta:** 3-8% Succinct E2E within 6-10 engineer-weeks
(per R2-A16 §3 Phase A 1-2 wks + Phase B 1-2 wks + Phase C 4-7 wks; R2-A11 §5
revises 9-14 weeks all-in with the IGC risk premium).

**Hard authoritative ceiling:** 10.7% Succinct E2E (R3-A01) — nothing in Phase A
can exceed it.

---

## Files cited

- `/tmp/recursion_round1_agent_01.md` (R1-A01: 99s lift+join measurement, 1.65× CUDA gap)
- `/tmp/recursion_round1_agent_05.md` (R1-A05: 12 quick wins, 8 not yet picked)
- `/tmp/recursion_round1_agent_07.md` (R1-A07: 10 bit-exactness risks)
- `/tmp/recursion_round1_agent_11.md` (R1-A11: poly_fp 1h24m hang structural)
- `/tmp/recursion_round1_agent_12.md` (R1-A12: CPU/GPU phase overlap ~6.5% E2E)
- `/tmp/recursion_round2_agent_03.md` (R2-A03: EVAL_CHECK_WG knob diff)
- `/tmp/recursion_round2_agent_04.md` (R2-A04: poly_fp mechanical split plan)
- `/tmp/recursion_round2_agent_05.md` (R2-A05: step_compute_accum PoC, 5-8 days)
- `/tmp/recursion_round2_agent_09.md` (R2-A09: oneDPL sort/scan port)
- `/tmp/recursion_round2_agent_11.md` (R2-A11: rv32im port history lessons)
- `/tmp/recursion_round2_agent_16_SCORECARD.md` (R2-A16: synthesis scorecard, Phase A enumeration)
- `/tmp/recursion_round3_agent_01.md` (R3-A01: authoritative 6-10% E2E ceiling)
- `/tmp/recursion_round3_agent_02.md` (R3-A02: EVAL_CHECK_WG knob APPLIED)
- `/tmp/recursion_round3_agent_05.md` (R3-A05: skip_verify already in place)
- `/tmp/recursion_round3_agent_08.md` (R3-A08: differential harness 50-100 LOC)
- `/home/user/.claude/projects/-home-user-inteldebug/memory/MEMORY.md`:
  - `project_tree_reduce_negative.md` (T3.2 2.3× regression — R2-A11 sec. 1.5)
  - `project_fib_sigabrt_root_cause.md` (USM accumulation — R2-A11 sec. 1.7)
  - `project_igc_patch_required.md` (silent fallback hazard — R2-A11 sec. 1.1)
  - `project_eval_check_wg512.md` (WG=512 rv32im precedent)
  - `project_optimization_landscape.md` (menu of other levers)
