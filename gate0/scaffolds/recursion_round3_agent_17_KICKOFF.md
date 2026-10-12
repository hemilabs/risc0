# Recursion port kickoff — first 48 hours

**Author:** R3-A06 agent 17 (synthesis)
**Date:** 2026-05-20
**Decision context:** R3-A06 plan reviewed across 17 R1 + 16 R2 + 2 R3 reports.
**This is the start-TODAY action list.** Opinionated, concrete, file-path-level.

## Big-picture framing for the first 48 hours

The R2-16 scorecard verdict is **PIVOT** to a phased plan, not a 3-5 week
monolithic step_exec push. The first 48 hours therefore are **Phase A**:
freebies + measurements + harness scaffolding. **Zero kernel code yet.**
Everything in this window is reversible, low-risk, and de-risks the multi-week
port decision (Phase B = accum-first PoC, Phase C = step_exec, Phase D =
tuning).

Three things must happen in 48 hours:
1. **Confirm the WG=1024 default-knob baseline** (R3-02 applied this — verify
   the binary builds and proves byte-identical, sweep WG, measure E2E).
2. **Run the par-safe audit** (R2-17 made this a gate on the whole plan;
   answers "is GPU witgen even worth porting?" in one number).
3. **Land the validation harness skeleton** (R1-17 — `EnvGuard`,
   `assert_buffer_eq`, env-knob plumbing, no kernel code yet) so any kernel
   change in Phase B+ ships with a first-mismatch diagnostic ready to fire.

The deliverable at hour 48 is a **decision-grade memo**: (a) confirmed Phase-A
baseline E2E with WG knob in place; (b) measured par-safe fraction → updates
the EV ceiling for Phase B/C; (c) committed harness skeleton + one passing
self-comparison test (CPU-vs-CPU shadow) proving the plumbing works before any
GPU kernel ever lands.

---

## Day 1 (today, 2026-05-20)

### Hour 1 — Confirm R3-02 (WG knob) is on disk and rebuild
```bash
grep -n "RISC0_RECURSION_EVAL_CHECK_WG" \
  /home/user/risc0-intel/risc0/risc0/circuit/recursion-sys/kernels/intel/eval_check.cpp
# Expected: lines 48-56 contain the env-driven WG_SIZE selection (R3-02 applied this).

cd /home/user/risc0-intel/risc0
cargo clean -p risc0-circuit-recursion-sys
LD_LIBRARY_PATH=/home/user/igc-rebuild/build/IGC/Release \
  cargo build --release -p risc0-circuit-recursion-sys 2>&1 | \
  tee /tmp/recursion_kickoff_h1_build.log
# Watch for: ocloc completes, no PreCompiledFuncImport crash (memory:
# project_igc_patch_required.md — patched IGC required).
```
**Acceptance:** the `librisc0_circuit_recursion-*.so` rebuild lands under
`risc0/target/release/build/risc0-circuit-recursion-sys-*/out/`.

### Hour 2 — Baseline E2E proof (control run, WG unset = default 1024)
```bash
cd /home/user/risc0-intel/risc0
LD_LIBRARY_PATH=/home/user/igc-rebuild/build/IGC/Release \
  RISC0_VERBOSE=1 RISC0_HASHFN=poseidon2 \
  target/release/examples/prove_and_verify 100000 succinct \
  2>&1 | tee /tmp/recursion_kickoff_h2_baseline_wg1024.log
# Capture wall-clock + [prove_session] timings; expect ~244 s for Succinct E2E,
# 99-103 s for lift+join chain (matches R1-01 numbers).
```
**Acceptance:** total prove time recorded; lift+join sub-total recorded;
proof verifies. Save this as the **Phase A baseline** for all downstream comparisons.

### Hour 3 — WG sweep (256 / 512 / 1024)
```bash
for WG in 256 512 1024; do
  LD_LIBRARY_PATH=/home/user/igc-rebuild/build/IGC/Release \
    RISC0_RECURSION_EVAL_CHECK_WG=$WG \
    target/release/examples/prove_and_verify 100000 succinct \
    2>&1 | tee /tmp/recursion_kickoff_h3_wg${WG}.log
done
# Per R3-02 + R1-13: expect 0-0.3% delta between WG=1024 and others. WG=1024
# is the predicted optimum (recursion uses -cl-opt-disable, register
# pressure is low). Confirms knob is live and chooses a winner.
```
**Acceptance:** 3 cold runs each (drop first as warm-up); pick the
fastest WG; if no clear winner, default stays at 1024. Verify byte-identical
seals across all three (the knob must not change math).

### Hour 4 — Run R2-17's par-safe audit instrumentation
This is the **single most important measurement** of the next 48 hours.
Patch `ffi.cpp` to count par-safe cycles at preflight time, rebuild, run.

Open `/home/user/risc0-intel/risc0/risc0/circuit/recursion-sys/kernels/cxx/ffi.cpp`
around the parStepExec dispatch loop (line 60-91 region) and add the
instrumentation block from R2-17 (lines 251-263 of that report):
```cpp
// In MachineContext::parStepExec or the dispatch top:
static thread_local size_t par_safe_cycles = 0, total_cycles_seen = 0;
total_cycles_seen++;
if (preflight->cycles[cycle].isParSafe) par_safe_cycles++;
// emit at end-of-run via a fprintf hook tied to RISC0_RECURSION_PAR_AUDIT
```
Or, less invasively, walk `preflight->cycles` from the FFI entry point
before dispatch and emit one summary line. Gate behind
`RISC0_RECURSION_PAR_AUDIT=1`.

```bash
cargo clean -p risc0-circuit-recursion-sys
LD_LIBRARY_PATH=/home/user/igc-rebuild/build/IGC/Release \
  cargo build --release -p risc0-circuit-recursion-sys

RISC0_RECURSION_PAR_AUDIT=1 RISC0_VERBOSE=1 \
  LD_LIBRARY_PATH=/home/user/igc-rebuild/build/IGC/Release \
  target/release/examples/prove_and_verify 100000 succinct \
  2>&1 | tee /tmp/recursion_kickoff_h4_parsafe_audit.log

grep "par_safe=" /tmp/recursion_kickoff_h4_parsafe_audit.log
```
**Acceptance:** one number — % cycles that are par-safe. Per R2-17, if
≥70% are par-safe, Phase B+C is the right plan. If ≤30% par-safe,
**the GPU port ceiling is ~10% E2E, not the plan's 20-35%** — and the
recommendation flips toward Alternative 6 (ESIMD on the par-safe slice
specifically) rather than vanilla SYCL on the whole loop.

### Hours 5-6 — Read R1+R2 reports in this order (~90 min)
Required reading before any harness/kernel work:
1. `/tmp/recursion_round2_agent_16_SCORECARD.md` — synthesis (you already
   read this; re-read the Phased Recommendation in §5).
2. `/tmp/recursion_round1_agent_17.md` — harness design (3 layers).
3. `/tmp/recursion_round1_agent_03.md` — accum-first PoC argument.
4. `/tmp/recursion_round1_agent_07.md` — six bit-exactness landmines.
5. `/tmp/recursion_round2_agent_17.md` — alternatives ranking,
   ESIMD-hybrid (Alt 6).
6. `/tmp/recursion_round3_agent_05.md` — R3-05 verdict: drop "skip
   intermediate verify" from quick-wins (already bypassed).

Skim others as needed: R1-06 (chain-bounded parallelism), R1-14 (oneDPL
sort), R1-16 (effort re-estimate), R2-04 (poly_fp split if
RISC0_RECURSION_OPTIMIZE is back on the table).

### Hour 7 — Create the harness skeleton (R1-17 Layer-1 scaffold, no GPU yet)
```bash
mkdir -p /home/user/risc0-intel/risc0/risc0/circuit/recursion/src/prove/hal
# Create testutil.rs — port verbatim from rv32im
cp /home/user/risc0-intel/risc0/risc0/circuit/rv32im/src/prove/hal/intel.rs \
   /tmp/rv32im_intel_test_module_reference.rs
# (Use this as a copy-source for EnvGuard, assert_check_eq, golden helpers.)
```
Then create `recursion/src/prove/hal/testutil.rs` with:
- `EnvGuard` struct (verbatim from rv32im, lines 393-425 of
  `rv32im/src/prove/hal/intel.rs`)
- `assert_buffer_eq(lhs, rhs, label)` helper (rename of `assert_check_eq`)
- `ENV_LOCK` global mutex
- Wire `mod testutil;` into `prove/hal/mod.rs`

**Acceptance:** `cargo build --release -p risc0-circuit-recursion` compiles
cleanly with the new module. No tests yet; just the scaffold.

### Hour 8 — Add env-var rerun-if-changed hooks to `recursion-sys/build.rs`
Mirror the existing `RISC0_RECURSION_OPTIMIZE` pattern (already at
`build.rs:425`). Add:
```rust
println!("cargo:rerun-if-env-changed=RISC0_RECURSION_GPU_WITGEN_VERIFY");
println!("cargo:rerun-if-env-changed=RISC0_RECURSION_GPU_WITGEN_VERIFY_SOFT");
println!("cargo:rerun-if-env-changed=RISC0_DISABLE_INTEL_RECURSION_WITGEN");
println!("cargo:rerun-if-env-changed=RISC0_RECURSION_PAR_AUDIT");
```
**Acceptance:** the rebuild stamp at `build.rs:234-236` honors these so a
later flag flip rebuilds.

---

## Day 2 (tomorrow, 2026-05-21)

### Hour 9 — Layer-2 harness: shadow CPU witgen probe (no GPU change)
Add to `recursion/src/prove/witgen.rs` (or the equivalent FFI-entry function)
the R1-17 §Layer-2 probe (lines 124-144 of `recursion_round1_agent_17.md`).
Gate behind `RISC0_RECURSION_GPU_WITGEN_VERIFY`. **Critically — at this
point the "GPU" path is still the CPU FFI**, so the probe compares CPU
witgen with itself. That's the green-baseline self-test:
- when probe is on, two CPU FFI runs produce identical buffers → harness
  silently passes;
- when a future kernel introduces a bug, the same probe goes red with
  cycle-level diagnostics.

This is the "harness goes online before any kernel code lands" check.

### Hour 10 — Validate the Layer-2 probe is wired
```bash
LD_LIBRARY_PATH=/home/user/igc-rebuild/build/IGC/Release \
  RISC0_RECURSION_GPU_WITGEN_VERIFY=1 RISC0_VERBOSE=1 \
  target/release/examples/prove_and_verify 100 succinct \
  2>&1 | tee /tmp/recursion_kickoff_h10_harness_selftest.log
grep -i "shadow\|witgen.*verify\|panic\|mismatch" \
  /tmp/recursion_kickoff_h10_harness_selftest.log
```
**Acceptance:** the probe runs (look for an emitted log line), the proof
completes, no mismatch is reported. Wall time ~2× baseline for the
shadowed phase (acceptable — gated on env var, off in production).

### Hour 11 — Pin one fixture preflight per R1-17 §"Bringing the harness online"
Add `RISC0_DUMP_RECURSION_PREFLIGHT=path/to.bin` hook that serializes a
`RawPreflightTrace` to disk. Run it once for the easiest fixture:
```bash
RISC0_DUMP_RECURSION_PREFLIGHT=/tmp/recursion_preflight_lift_po2_18.bin \
  target/release/examples/prove_and_verify 100 succinct
```
**Acceptance:** binary blob ~few MB on disk. Will be the fixture for
Layer-1 unit tests in Phase B. Don't commit yet — pinning the fixture
format is a Phase-B decision.

### Hour 12 — Quick win 2 verification (R3-05 verdict): NO-OP, but document
Per R3-05 (`recursion_round3_agent_05.md`), "default-on skip_verify for
intermediate lift+join" is already done by the pipelined
`composite_to_succinct` override at `prover_impl.rs:1159`. **Don't
re-apply R2-02's patch**; instead, document in your kickoff memo that this
lever is retired and the slot is freed for another (eval_check WG sweep,
Poseidon2 OpenCL default, or Tier-2 sweeps from
`project_optimization_landscape.md`).

### Hour 13 — Read & decide on Alt 6 / ESIMD-hybrid (R2-17)
Re-read `/tmp/recursion_round2_agent_17.md` §Alternative 6 with the par-safe
fraction from hour 4 in hand:
- ≥70% par-safe → Plan B (vanilla SYCL witgen first, ESIMD layered later);
- 30-70% par-safe → Plan B still goes, but allocate ESIMD time in Phase D;
- ≤30% par-safe → **flip immediately** to ESIMD-on-sub-kernels (WOM sort,
  accum, FpExt mul/add) and **skip the vanilla-SYCL witgen port entirely**.

This single hour is where the par-safe number turns into a roadmap
decision.

### Hour 14 — Confirm patched IGC is still loadable (sanity)
The witgen/accum rebuild needs the patched IGC
(memory: `project_igc_patch_required.md`):
```bash
ldd /home/user/igc-rebuild/build/IGC/Release/libigc.so.2 | head -5
ls -la /home/user/igc-rebuild/build/IGC/Release/libigc.so.2
# expect: present, recent mtime
```
Also smoke-test that a witgen-touching kernel can still build under
LD_LIBRARY_PATH override:
```bash
cargo clean -p risc0-circuit-recursion-sys
LD_LIBRARY_PATH=/home/user/igc-rebuild/build/IGC/Release \
  cargo build --release -p risc0-circuit-recursion-sys 2>&1 | tail -20
```
**Acceptance:** no PreCompiledFuncImport crash, build completes.

### Hour 15 — Draft Phase-B GO/NO-GO checklist for the kickoff memo
Per R2-16 §5 conditional-GO criteria, write a one-page checklist that the
human reviewer uses to greenlight Phase B (accum-first PoC):
- Par-safe fraction ≥ X% (X from hour 4)
- WG sweep showed no regression / confirms 1024 is fine
- Harness self-test passes (hour 10)
- Patched IGC builds clean (hour 14)
- Engineer-week budget approved: 5-7 wk optimistic, 10-14 wk pessimistic
  for full port; 1-2 wk for accum-only PoC

### Hour 16 — Compose the deliverable memo
File: `/tmp/recursion_phase_a_kickoff_memo.md` (handwritten, ~2 pages).
Sections:
1. **Baseline** — wall-clock numbers from hour 2 and 3 logs.
2. **Par-safe measurement** — the % from hour 4.
3. **Plan adjustment** — which of R2-16's phases is greenlit and why.
4. **Harness state** — what's on disk now (testutil.rs, build.rs hooks,
   layer-2 probe) and what's NOT (Layer 1 unit tests, fixture pinning —
   deferred to Phase B week 1).
5. **Next decision point** — Phase-B GO/NO-GO criteria (the checklist
   from hour 15). Hand to project lead.

---

## End of 48 hours: deliverable

**A single human-readable Phase A go/no-go memo** containing:

1. **Confirmed Intel-Battlemage baseline** for Succinct E2E with the
   R3-02 WG knob applied at default. Files:
   - `/tmp/recursion_kickoff_h2_baseline_wg1024.log` (control)
   - `/tmp/recursion_kickoff_h3_wg{256,512,1024}.log` (sweep)
   Headline: lift+join wall-clock ~99-103 s, Succinct total ~244 s,
   knob causes zero seal drift, no clear winner across WG values
   (default 1024 stays).

2. **A par-safe fraction measurement** for a representative
   `prove_and_verify 100000 succinct` run, gating the entire R3-A06 effort:
   - ≥70% → continue with the R2-16 phased plan as-is (Phase B accum-first).
   - 30-70% → continue with phased plan but pre-budget Alt 6 ESIMD work
     in Phase D.
   - ≤30% → **pivot to ESIMD-on-par-safe-sub-kernels**, skip vanilla
     SYCL witgen port.
   File: `/tmp/recursion_kickoff_h4_parsafe_audit.log`.

3. **A landed validation harness skeleton** on disk:
   - `risc0/risc0/circuit/recursion/src/prove/hal/testutil.rs` (new file,
     EnvGuard + assert_buffer_eq, mirrors rv32im)
   - `risc0/risc0/circuit/recursion/src/prove/witgen.rs` (patched with
     RISC0_RECURSION_GPU_WITGEN_VERIFY Layer-2 probe; currently no-op
     because GPU path is still CPU FFI)
   - `risc0/risc0/circuit/recursion-sys/build.rs` (env-var rerun-if-changed
     hooks added)
   - Proof that the Layer-2 probe runs: green CPU-vs-CPU self-test logged
     at `/tmp/recursion_kickoff_h10_harness_selftest.log`.

4. **A signed Phase-B GO/NO-GO checklist** with the four gates of R2-16 §5
   each marked PASS/FAIL/UNKNOWN. Greenlights (or doesn't) the 1-2 week
   accum-first PoC starting Day 3.

What is **NOT** in the deliverable:
- Zero new kernel code. (R2-16: harness lands before kernel.)
- Zero edits to `step_compute_accum.cpp` / `step_exec.cpp`.
- No `poly_fp.cpp` split (R2-04's plan is a Phase D follow-up, not
  Phase A — IGC works at -cl-opt-disable today).
- No CUDA wrapper / multi-GPU / cache experiments (R2-17 ranked these
  dead-ends or future-only).

## Why this 48-hour ordering is opinionated

- **R3-02 WG knob first** because it's already applied and a one-rebuild
  sweep gives the cleanest baseline to compare against — and R3-A06's
  whole framing assumes we know the recursion E2E to 1% precision.
- **Par-safe audit second** because R2-17 elevated it to a gate on the
  whole multi-week port. Cheap (1 day) to know whether the GPU port is
  even worth the engineering investment.
- **Harness skeleton last on Day 1** because R1-17 + R2-16 are
  unanimous: harness lands before kernel work. Building it against the
  current CPU-only path proves the plumbing; when a real kernel lands
  in Phase B, the harness is already producing first-mismatch diagnostics
  for free.
- **Phase B PoC is NOT started in the first 48 hours.** It's a 1-2 week
  job (accum-first, 15.6K LOC, 1 extern, embarrassingly parallel — R1-03)
  and starting it before the par-safe number is in hand risks
  committing to a port whose ceiling we don't know.

## Files touched / committed by hour 48

Modified:
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion-sys/build.rs` (+4 env rerun lines)
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion-sys/kernels/cxx/ffi.cpp` (par-safe audit gated on `RISC0_RECURSION_PAR_AUDIT`)
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion/src/prove/witgen.rs` (Layer-2 shadow probe gated on `RISC0_RECURSION_GPU_WITGEN_VERIFY`)
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion/src/prove/hal/mod.rs` (add `mod testutil;`)

Added:
- `/home/user/risc0-intel/risc0/risc0/circuit/recursion/src/prove/hal/testutil.rs` (new, from rv32im pattern)
- `/tmp/recursion_phase_a_kickoff_memo.md` (the deliverable)

Logs (working artifacts, not committed):
- `/tmp/recursion_kickoff_h1_build.log`
- `/tmp/recursion_kickoff_h2_baseline_wg1024.log`
- `/tmp/recursion_kickoff_h3_wg{256,512,1024}.log`
- `/tmp/recursion_kickoff_h4_parsafe_audit.log`
- `/tmp/recursion_kickoff_h10_harness_selftest.log`

## What changes for hour 49

Either:
- **Greenlight** → start Phase B (accum-first PoC, R1-03), 1-2 weeks.
  First task: port `step_compute_accum` body into
  `risc0/risc0/circuit/recursion-sys/kernels/intel/step_compute_accum_kernel.cpp`,
  with the harness from Day 1 already shadowing every run.
- **Pivot to Alt 6** (par-safe was low) → start ESIMD on WOM sort / accum
  sub-kernels using the templates from
  `/home/user/risc0-intel/benchmarks/esimd-bench/kernels/`.
- **No-go** → harness + WG knob still net-positive freebies; document
  that recursion port is parked, redirect to Poseidon2-OpenCL-default or
  Tier-2 WG sweeps per `project_optimization_landscape.md`.
