# R3-A05 Verdict: R2-13 is correct. Drop "skip intermediate verify" from quick-wins.

## TL;DR

**R2-13 wins.** The optimized pipelined `composite_to_succinct` at `prover_impl.rs:1159` already bypasses `self.lift()/self.join()` and therefore bypasses the per-call `verify_integrity`. The Intel default path runs zero intermediate verifies in the lift+join loop. R1-05's "1.5-6% Succinct E2E" estimate is wrong: the lever is already pulled.

**Recommendation: drop QW#2 from the roadmap's first PR.** Use the slot for a real lever (eval_check WG=512 is already merged; consider Poseidon2-OpenCL-by-default or Tier-2 WG sweeps instead).

## Dispatch trace (confirmed against source)

1. `Session::prove()` → `get_prover_server(opts)` at `mod.rs:440`.
2. `get_prover_server` (`mod.rs:446-453`) returns `Rc::new(ProverImpl::new(opts))` for every non-dev path. **No CUDA/ROCm/Intel branching here.** ProverImpl is the *only* non-dev prover for Intel.
3. `ProverImpl::prove_session` → calls `self.composite_to_succinct(&composite_receipt)` at `prover_impl.rs:896`.
4. `self.composite_to_succinct` resolves to the **inherent** `ProverImpl::composite_to_succinct` at `prover_impl.rs:1159`, NOT the blanket-impl `Compress::composite_to_succinct` at `mod.rs:374-398` (Rust prefers inherent methods over trait default methods on the same impl block).
5. The inherent impl at line 1159 calls `lift_with_opts(&segments[step_idx], …)` (line 1207) and `join_with_opts(&left, &lifted, …)` (line 1229) **directly** — bypassing `self.lift()/self.join()` and their `verify_integrity` wrappers (lines 1011-1029 and 1038-1060).
6. The wrapper `fn lift/join/resolve/union` at `prover_impl.rs:1011/1038/1078/1128` is therefore dead code in the lift+join hot loop. It only fires from:
   - `host/api/server.rs:539,588` (gRPC external API, per-call) — not the prove_session path.
   - The blanket `Compress` impl at `mod.rs:374` — overridden, not reachable from `ProverImpl::prove_session`.
   - The `resolve` path *is* still called via `self.resolve(...)` inside the inherent `composite_to_succinct` (line 1267) for assumptions. So `verify_integrity` *does* still run during resolve, but only when assumptions are present (rare in the default benchmark workload, e.g. `prove_and_verify` has 0 assumptions).

## Key code references

- `risc0/zkvm/src/host/server/prove/mod.rs:446-453` — `get_prover_server` always returns `ProverImpl` for non-dev mode (no Intel/CUDA/ROCm dispatch).
- `risc0/zkvm/src/host/server/prove/mod.rs:374-398` — blanket `Compress<Claim>` trait default impl using `self.lift()/self.join()` (which WOULD trigger verify). This is **NOT reached** by Intel because:
- `risc0/zkvm/src/host/server/prove/prover_impl.rs:1159-1291` — inherent `ProverImpl::composite_to_succinct` overrides the trait default. Uses `lift_with_opts`/`join_with_opts` (NOT `self.lift()/self.join()`).
- `risc0/zkvm/src/host/server/prove/prover_impl.rs:1011-1029` — `fn lift` wrapper with `verify_integrity` (`receipt.verify_integrity().context("verify lift")?` line 1023). Gated by `RISC0_SKIP_VERIFY` env var. **Bypassed.**
- `risc0/zkvm/src/host/server/prove/prover_impl.rs:1038-1060` — same for `fn join` (line 1054). **Bypassed.**
- `risc0/zkvm/src/host/server/prove/prover_impl.rs:1207, 1229` — the actual call sites used in the hot loop. No verify.
- `risc0/zkvm/src/host/recursion/prove/mod.rs` — `lift_with_opts`/`join_with_opts` definitions; pure prove, no internal verify_integrity.

## Why R1-05 was wrong

R1-05's QW#2 was built on the assumption that `prover.lift()` is called per-segment during composite_to_succinct. That's true only for the *generic* `Compress` blanket impl at `mod.rs:374-398`. ProverImpl overrides it with the pipelined inherent method. R1-05 didn't trace the dispatch.

Additionally, R1-05 estimated verify_integrity at 50-200 ms per call. R2-13's measurement on PO2=18 recursion seal shows ~10-13 ms. So even the per-call cost was overestimated by 4-15×.

Combined: claimed 4-17 s savings; real ceiling 0-1 s; effective today = **0 s** (bypass already in place).

## What's still salvageable from R1-05's QW#2

1. **`RISC0_SKIP_VERIFY=1` does still help in the segment phase.** The wrapper at `prover_impl.rs:711,770` (segment receipt verify) IS in the critical path, and `mod.rs:76` for `prove_segment`. R2-13 estimates ~50-100 ms × 42 segs ≈ ~3 s savings = ~1-1.5% Succinct E2E for that. This is a different lever from R1-05's claim, but it's real.
2. **The `host/api/server.rs:539,588` gRPC path** still calls `prover.lift()/join()` per RPC, hitting the wrapper. If anyone uses that path in production (Bonsai-style), default-on skip_verify gives ~12 ms × N savings. Not session-worthy.

## Verdict on the R3-A06 plan first PR

- **Drop "default-on skip_verify for intermediate lift+join" from QW menu.** It would change exactly 0 code paths in the default Intel benchmark workload (`prove_and_verify N succinct`). The pipelined override already short-circuits all four wrapper functions.
- **Optionally** add a separate quick win: "default-on `RISC0_SKIP_VERIFY=1` for *segment* verify in `prove_session`." That's real, ~1-1.5% E2E, but distinct from R1-05's framing and probably already captured by env-var docs anyway.

R2-13's audit is correct on every claim I verified by source inspection: dispatch path, wrapper bypass, file locations, line numbers. The empirical measurement (no `[lift] prove=…ms verify=…ms` log lines appearing in a real trace) is also consistent with the bypass.
