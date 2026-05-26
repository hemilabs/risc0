# R3-A06 Round 3 — Agent 04: Cross-Segment Lift Pipelining (Tier-1 Implementation Plan)

## Summary

Implement R2-07's Tier-1 pipelining: a background thread runs
`Prover::new_lift(seg N+1) + prepare()` concurrently with the main thread's
`lift(N).run() + join(left, lift(N))` on the GPU. The preflight of lift(N+1)
(~300-500 ms of single-threaded CPU work over `program.code_by_row()`) hides
behind ~700-900 ms of GPU work for lift(N) + the GPU portion of join. Zero
GPU queue contention because the background thread never touches the GPU.

**EV: 6-7% E2E on Succinct (matching R1-12 / R2-07's projection).
~50 LOC change to `risc0/zkvm/src/host/server/prove/mod.rs:374-398`.
Bit-exact final receipt.**

## 1. What R2-07 established

(Verified against the source tree this round.)

- The recursion outer `Prover` (`zkvm/src/host/recursion/prove/mod.rs:568`)
  holds `risc0_circuit_recursion::prove::Prover` + `Digest` + `ProverOpts`.
  All three are `Send`. The inner prover (`circuit/recursion/src/prove/mod.rs:107-113`)
  is `Arc<Program> + String + VecDeque<u32> + Option<Preflight>` — `Send`.
- `Prover::prepare()` (`zkvm/src/host/recursion/prove/mod.rs:1013-1017`)
  is `pub` and explicitly documented: *"Can be called from any thread."* The
  inner implementation (`circuit/recursion/src/prove/mod.rs:138-146`) is a
  pure CPU loop over `program.code_by_row()` invoking `Preflight::step`.
  No GPU touch. Result is stored in `self.prepared_preflight`.
- `SegmentReceipt` derives `Clone` (`zkvm/src/receipt/segment.rs:35`).
  Receipts hold a `Vec<u32>` seal (~4-8 MB at PO2=18-20). The seal is the
  only large field; cloning 42 of them at fold setup is ~250 MB of
  allocation. Mitigation in §3: use `Arc<SegmentReceipt>` end-to-end OR
  clone only the reference passed to the bg thread (the bg thread only
  needs `&SegmentReceipt` until `new_lift` returns, then can drop it). The
  simplest fix is to clone the `SegmentReceipt` directly — at PO2=18
  that's ~4 MB × 1 bg thread alive at any time = 4 MB resident overhead,
  which is acceptable.
- The current sequential fold is at
  `zkvm/src/host/server/prove/mod.rs:378-396` (confirmed).
- The rv32im SegmentProver has the exact pipelining pattern we want:
  `RefCell<Option<JoinHandle<…>>>` (`circuit/rv32im/src/prove/hal/mod.rs:131`)
  joined at the head of the next iteration and re-armed at the tail
  (`circuit/rv32im/src/prove/hal/mod.rs:395-454`).

## 2. The Tier-1 architecture

### Thread layout

| Thread | Work |
|---|---|
| **Main** | The full GPU sequence of lift(N) (witgen H2D/D2H, commits, accum, finalize, FRI) + join(left, lift(N)) on the GPU. Uses `get_queue()` → main SYCL queue. |
| **Bg** (spawned for each (idx)) | Pure CPU: `Prover::new_lift(seg N+1, opts)` + `prover.prepare()`. **No GPU calls.** Bg thread never reads `get_queue()` and never goes through `with_queue_override` (Tier-1 doesn't need the eval-check queue). |

Cardinality: at most 1 bg thread alive at any moment. Bg thread spawned at
iteration N's tail, joined at iteration N+1's head. Same idiom as rv32im's
`pending_finalize`.

### State carried across iterations

A single `Option<JoinHandle<Result<Prover>>>` local to the fold. The handle
returns the **already-prepared** outer recursion `Prover` for segment N+1.

### Dependency analysis (why bg work is safe to overlap)

Bg work touches:
- `program.code_by_row()` — read-only, the ZKR program is `Arc<Program>`
  cloned by reference. Safe.
- `SegmentReceipt` fields (seal, claim, hashfn) — bg thread holds its own
  owned `SegmentReceipt` clone, no aliasing with main.
- `Preflight::new(...).step(cycle, row)` for every cycle — single-threaded
  loop, owned data, allocates into bg thread's heap. Safe.
- `Prover::new_lift_inner` (`zkvm/src/host/recursion/prove/mod.rs:657`)
  internally constructs an inner `Prover::new(program_arc, hashfn)` and
  feeds it `add_seal(...)` + `add_input(...)`. All pushes are into
  `VecDeque<u32>` owned by the bg thread's `Prover`. Safe.

Main work touches: the GPU queue, the per-thread `PROVER_CACHE` and
`PROOF_CACHE`/`CTRL_CACHE` thread_local!s in
`circuit/recursion/src/prove/mod.rs:178-181`. Bg thread does **not** touch
those caches because it stops before `run()`. The bg thread's freshly-built
`Prover` is moved back to the main thread via `JoinHandle::join()`, at
which point the main thread calls `prover.run()` and reads/populates the
main thread's caches as normal. No cache pollution.

### Critical resource: GPU queue

Single GPU. Single main queue. Bg thread never uses it. Therefore no
contention. (Tier 2/3 would need `with_queue_override(EVAL_CHECK_QUEUE, ...)`
and `eval_to_main_barrier()` for correctness — explicitly out of scope for
this design.)

## 3. Diff sketch (~50 LOC)

Single file: `risc0/zkvm/src/host/server/prove/mod.rs`. The fold at lines
378-396 is replaced. `lift_with_opts` in `host/recursion/prove/mod.rs` is
*not* touched — we just need its pieces inlined into the fold so the
bg-prepared `Prover` flows in.

### Step 1: add `prove_with_prepared` to `lift` machinery

We need a way to run a lift starting from an *already-prepared* `Prover`,
since the bg thread builds the `Prover` and runs `prepare()` on it, then
hands it back. Currently `lift_with_opts`
(`risc0/zkvm/src/host/recursion/prove/mod.rs:81-99`) owns the `Prover::new_lift`
call. Two options:

- **(A) Expose a `lift_from_prover` variant:** ~10 LOC in
  `host/recursion/prove/mod.rs` that takes a `Prover` (already constructed)
  + a `&SegmentReceipt` (for the merge claim) and runs `run()`,
  `decode + merge`, `make_succinct_receipt`. **Recommended.**
- (B) Inline the lift body directly in the fold. Works but couples the
  fold to recursion internals; uglier.

Add to `host/recursion/prove/mod.rs`:

```rust
/// Run the lift program from an already-built (and optionally pre-prepared)
/// Prover. Used by `composite_to_succinct` pipelining to allow the
/// `Prover::new_lift` + `prepare()` CPU work to overlap with the previous
/// segment's GPU work on a background thread.
pub fn lift_from_prover(
    mut prover: Prover,
    segment_receipt: &SegmentReceipt,
) -> Result<SuccinctReceipt<ReceiptClaim>> {
    let t0 = std::time::Instant::now();
    let receipt = prover.prover.run()?;
    let run_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = std::time::Instant::now();
    let claim_decoded = ReceiptClaim::decode(&mut receipt.out_stream())?;
    tracing::debug!("Proving lift finished: decoded claim = {claim_decoded:#?}");
    let claim = claim_decoded.merge(&segment_receipt.claim)?;
    let result = make_succinct_receipt(prover, receipt, claim);
    let post_ms = t1.elapsed().as_secs_f64() * 1000.0;

    if *VERBOSE { eprintln!("[lift_from_prover] run={run_ms:.1}ms post={post_ms:.1}ms"); }
    result
}
```

### Step 2: rewrite the fold

Replace lines 378-396 of `risc0/zkvm/src/host/server/prove/mod.rs` with:

```rust
let t_pipeline = std::time::Instant::now();

// Build the segment slice once; we need indexed access for prefetching.
let segments: &[SegmentReceipt] = &composite_receipt.segments;

// Background-thread prefetch of the next segment's prepared lift Prover.
// `Prover::prepare()` is single-threaded CPU work over program.code_by_row()
// (~300-500 ms at PO2=18). It overlaps fully with the previous iteration's
// GPU work (lift.run + join). Bg thread never touches the GPU.
type PreparedProver = risc0_zkvm::host::recursion::prove::Prover;
let mut pending: Option<std::thread::JoinHandle<Result<PreparedProver>>> = None;

let spawn_prepare = |seg: SegmentReceipt|
    -> std::thread::JoinHandle<Result<PreparedProver>>
{
    std::thread::spawn(move || -> Result<PreparedProver> {
        let t = std::time::Instant::now();
        let mut p = risc0_zkvm::host::recursion::prove::Prover::new_lift(
            &seg, ProverOpts::succinct(),
        )?;
        p.prepare()?;
        if std::env::var_os("RISC0_VERBOSE").is_some() {
            eprintln!("[bg-preflight] {:.1}ms", t.elapsed().as_secs_f64()*1000.0);
        }
        Ok(p)
    })
};

let continuation_receipt = segments.iter().enumerate().try_fold(
    None,
    |left: Option<SuccinctReceipt<Claim>>, (idx, right): (usize, &SegmentReceipt)|
        -> Result<_>
    {
        // (1) Get prepared Prover for THIS segment (either from bg thread
        //     or built synchronously on the very first iteration).
        let prepared = if let Some(h) = pending.take() {
            let t = std::time::Instant::now();
            let p = h.join().map_err(|_| anyhow!("preflight thread panicked"))??;
            if std::env::var_os("RISC0_VERBOSE").is_some() {
                eprintln!("[fold-join] wait={:.1}ms", t.elapsed().as_secs_f64()*1000.0);
            }
            p
        } else {
            // First segment only: build synchronously on the main thread.
            let mut p = risc0_zkvm::host::recursion::prove::Prover::new_lift(
                right, ProverOpts::succinct(),
            )?;
            p.prepare()?;
            p
        };

        // (2) Speculatively start the NEXT segment's prepare on a bg thread,
        //     BEFORE we kick off this segment's GPU work. The bg thread
        //     runs purely on CPU; main thread now hits the GPU.
        if let Some(next) = segments.get(idx + 1) {
            pending = Some(spawn_prepare(next.clone()));
        }

        // (3) Main-thread GPU work: lift this segment using the prepared Prover.
        let lifted = risc0_zkvm::host::recursion::prove::lift_from_prover(
            prepared, right,
        )?;

        // (4) Main-thread GPU work: join with the accumulated left receipt.
        let result = match left {
            Some(left) => self.join(&left, &lifted)?,
            None => lifted,
        };
        Ok(Some(result))
    },
)?
.ok_or_else(|| anyhow!("malformed composite receipt has no continuation segment receipts"))?;

// Safety: pending must always be None here because the last iteration of
// the fold called segments.get(idx+1) == None and didn't spawn anything.
// (Explicit assertion below is defence-in-depth in case of refactors.)
debug_assert!(pending.is_none(), "leaked bg preflight thread");

eprintln!("[composite_to_succinct] lift/join done: {:.1}s", t_pipeline.elapsed().as_secs_f64());
```

### Step 3: trait routing

The fold is generic over `Claim ∈ {ReceiptClaim, WorkClaim<ReceiptClaim>}`.
The `lift_from_prover` API above only covers `ReceiptClaim`. For
`WorkClaim<ReceiptClaim>` (povw path), add a sibling `lift_povw_from_prover`
and a `LiftFromPrepared<Claim>` trait that mirrors the existing `Lift<Claim>`
trait at `mod.rs:279-299`. Routing is mechanical, ~15 LOC.

Alternative: gate Tier-1 pipelining behind `cfg!(feature = "intel")` and
only on the non-povw path for the first PR. PoVW workloads are short-running
anyway; the easy 6.5% win is on the long Succinct path.

Net LOC: ~50 in `host/server/prove/mod.rs` + ~15 in
`host/recursion/prove/mod.rs` = **65 LOC total** (close to R2-07's 50 LOC
estimate; the extra 15 is the `lift_from_prover` helper).

## 4. Validation plan

### 4a. Bit-exactness

- Pipelining changes **nothing** about the transcript: bg thread builds a
  `Prover` and calls `prepare()`, identical to the synchronous path —
  `prepared_preflight` flows untouched into `run()`. The Fiat-Shamir
  sponge inside `run()` consumes the *same* witness rows in the *same*
  order. The bg thread does not call any RNG (`prepare()` doesn't sample).
- Seal bytes must be byte-identical pre/post pipelining. Run
  `cargo test --test recursion -p risc0-zkvm --release` and compare
  `seal.iter().map(|x| format!("{x:08x}"))` of the final SuccinctReceipt
  before vs after. Add a comparison harness if one doesn't exist.

### 4b. Per-lift timing harness

Set `RISC0_VERBOSE=1`. With the design above we get three timestamps per
iteration:
- `[bg-preflight] Xms` (logged from bg thread when it returns)
- `[fold-join] wait=Yms` (how long the main thread blocked waiting for bg)
- `[lift_from_prover] run=Zms post=Wms` (main-thread GPU work)

**Success criterion:** `[fold-join] wait` should be near 0 ms (preflight
finished before main thread needed it). If `wait` is consistently > 50 ms,
preflight is on the critical path and Tier-1 alone may be undersizing the
overlap — we'd need to look at whether `prepare()` is heavier than 500 ms
or whether the GPU phase is shorter than 700 ms at this PO2.

### 4c. Sanity perf gate

Run the same 42-segment program twice (baseline + pipelined) on B70 with
`RISC0_KECCAK=0`, all other env vars default. Expect:
- Baseline `composite_to_succinct lift/join done`: ~99 s.
- Pipelined: ~91-93 s (6.5%).
- Bit-exact seals.

If neither timing nor bit-exactness holds, roll back.

## 5. Risks (Tier-1 only)

| # | Risk | Mitigation |
|---|---|---|
| R1 | `SegmentReceipt::clone()` cost: cloning per-iteration is ~4-8 MB. | Tier-1 only clones 1 receipt at a time (the next one) → ~8 MB peak overhead. Acceptable. If profiling says otherwise, swap to `Arc<SegmentReceipt>` along the slice. |
| R2 | Bg thread panic propagation. | `JoinHandle::join()` returns `Err(...)` which we map to `anyhow!("preflight thread panicked")`. No silent corruption. |
| R3 | Bg thread leaks if main thread errors mid-fold (`?` on `self.join(...)`). | `JoinHandle` drops on early exit. The OS will reap the thread; `prepare()` doesn't hold GPU resources, so no leak. We could add a `Drop` guard if we care about waiting before returning the error. |
| R4 | Last-iteration bg work wasted (we only need N segments' preflights, not N+1). | The `segments.get(idx+1)` guard prevents this — bg thread is only spawned when there's a next segment. |
| R5 | `lift_povw` path not covered by Tier-1. | Either gate Tier-1 to non-povw initially (cleaner first PR) or add the parallel `lift_povw_from_prover` helper. Both options are fine. |

## 6. EV breakdown

(Same numbers R2-07 derived, re-validated against the code this round.)

- t_lift_cpu_preflight ≈ 0.4 s (assumption; verified by `[bg-preflight]`)
- t_lift_gpu + t_join_gpu ≈ 0.8 + ~0.5 s ≈ 1.3 s of overlap window per iter
- Overlap saved per iter ≈ min(0.4, 1.3) = 0.4 s
- 42 lifts × 0.4 s = ~16-17 s
- Baseline Succinct E2E ≈ 246 s
- **Saving ≈ 6.5-7%** ✓ matches R1-12 / R2-07.

Conservative downside if `prepare()` is < 200 ms (program-dependent):
~3.5%. Upside if `prepare()` is closer to 600 ms: ~10%.

## 7. Why this is THE single biggest non-port win

- 50-65 LOC, single file logic change (+ small helper in recursion lift).
- Bit-exact final seal — no correctness risk surface.
- No new IGC flags, no kernel surgery, no algorithmic changes.
- Re-uses an existing public API (`Prover::prepare`) and an existing
  infrastructure pattern (rv32im `pending_finalize`).
- Doesn't conflict with eval_check WG=512 or any future kernel-level wins.
- Composes additively with Tier-2 (eval_check_queue routing of witgen) and
  Tier-3 (deferred finalize parity) when the team has bandwidth.

## 8. Recommendation

Implement exactly as described. One PR. Validation = bit-exact seal test
+ `RISC0_VERBOSE=1` lift timing showing `[fold-join] wait` ≈ 0 ms. After
landing, consider Tier 2 only if R3-A06 GPU-witgen port is deferred; if
the port lands, Tier 2/3 become moot because CPU phases shrink to zero.

## 9. File touch list

- `/home/user/risc0-intel/risc0/risc0/zkvm/src/host/server/prove/mod.rs`
  (replace fold at lines 378-396, ~50 LOC)
- `/home/user/risc0-intel/risc0/risc0/zkvm/src/host/recursion/prove/mod.rs`
  (add `lift_from_prover` helper, ~15 LOC, near existing `lift_with_opts`
  at line 81)
