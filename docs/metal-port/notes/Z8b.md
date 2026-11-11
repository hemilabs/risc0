# Z8b: review round, Poseidon2 / SHA / mix / FRI kernels and the HAL runtime

Branch `metal/Z8b` (worktree `/Users/user/zkminer/deps/risc0-wt-Z8b`), based on `zkminer/metal` @ `9827b70c4`.
Plan: 4.1 row Z8 (family split B), 2.4-2.9, 5, 8.7, 9.
Scope (owned): `sys/kernels/zkp/metal/{pv,sha,mix,fri}.metal`, `sha256.h`,
`zkp/src/hal/metal/{poseidon2,runtime,buffer}.rs`, `zkp/src/hal/metal.rs`. Z8a owns ntt/eltwise/misc/mdiv/eval.
Built on: F0, Z3, Z4, Z6, Z7 (notes read). Severity: P0 wrong result / crash / silent corruption,
P1 resource / robustness, P2 hygiene.

## What was audited
- Every kernel against the source it mirrors: `pv.metal` against `supra/pv.cuh` (HIP C++ branch) and
  `supra/pv_small.cuh`; `sha.metal` + `sha256.h` against `cuda/sha.cu` + `cuda/sha256.h`; `mix.metal`
  against `cuda/kernels.cu` (`mix_poly_coeffs` 158-185, `mix_poly_coeffs_v2` 389-455); `fri.metal`
  against `cuda/kernels.cu:80`.
- The host glue: `MetalHashSha256`, `MetalHashPoseidon2` (crossovers, launch shapes, fold tree), the
  `Hal` methods in `metal.rs`, `BufferImpl` (sync on every host path), the runtime (batches, pool,
  budget, peak, binding dedupe, threadgroup asserts, error paths).
- The checklist: SIMD lane ranges, bounds under grid rounding, 32-bit index overflow at po2 22-24 x
  211+ columns, sync on host access, pool reuse and aliasing, budget accounting, threadgroup-memory
  asserts, `use_resource` coverage, Poseidon2 crossover boundaries, fault words.

## Findings

| # | Sev | Where | Finding | Status |
|---|---|---|---|---|
| F1 | **P1** | `runtime.rs` `dispatch` | Every zkp and circuit kernel indexes with a 32-bit `thread_position_in_grid`. A grid of 2^32 or more threads in one dimension was dispatched anyway, and Metal ran it **without any error**: a 2^32-thread `fill_u32` grid completed (regression tests, pre-fix run). The thread index wraps silently. Reachable through any 1-D Hal op over a whole LDE buffer at 2^24 rows (po2 22) x >= 256 columns, e.g. `eltwise_add_elem`, `eltwise_copy_elem`, `alloc_elem_init`, zeroize. That is silent corruption when hit; graded P1 because no current prover shape reaches 2^32 elements in one launch (211 columns gives 3.5e9). | **Fixed**: each grid dimension (groups x threads for `dispatch_thread_groups`) is asserted to be <= `u32::MAX` before encoding, with a "32-bit thread index ... split the dispatch" panic. |
| F2 | **P1** | `runtime.rs` `check_budget`, alloc retry | The budget is checked against the device-wide `currentAllocatedSize`, which includes buffers idling in **every** HAL's pool. On failure only the allocating HAL's pool was drained. In a prover process with several HALs (rv32im, recursion, keccak), one HAL's pool (capped at half the budget each) made another HAL's allocations fail, even though that memory was idle. This matters for the 16 GB target. The plan 2.9 drain-and-retry after a nil `newBuffer` had the same gap. | **Fixed**: a registry of every runtime's pool (`Weak`), and `drain_all_pools()` on budget failure (device scope) and on allocation failure. This mirrors CUDA's OOM hook draining its process-wide pool. Lock order: registry lock released before the pool locks are taken, and only one pool lock is held at a time. `trim_device_memory` still drains only its own HAL's pool. |
| F3 | **P1** (tests) | runtime budget, plan 8.7 | The device-wide budget makes `RISC0_METAL_MAX_WORKING_SET_BYTES` runs of any test suite depend on scheduling. Under the default parallel harness, tests count each other's allocations. Measured on the base: 4 GiB parallel runs failed `z6_poly_group_no_coeffs_spill` ("1073741824 bytes for nodes with 3888054272 already allocated") and `batch_expand_into_evaluate_ntt`, neither of which fails alone. This is the wave-3 "device-wide vs per-HAL" flakiness. | **Fixed**: new opt-in `RISC0_METAL_BUDGET_SCOPE=hal` (`BudgetScope::Hal`, `MetalHal::set_budget_scope`, `owned_allocated_size`). In this scope the budget counts only the HAL's own bytes: live and pooled shared buffers, plus private buffers until the HAL drops. Over-budget then drains only the HAL's own pool. `device` stays the default and is the production model. With `hal`, the 4/8 GiB zkp suites under the parallel harness give exactly the serial device-scope results. |
| F4 | P2 | `buffer.rs` `as_arg_with_offset`, `gpu_address_at` | No bounds check: an offset past the view bound an out-of-bounds pointer. | **Fixed** (assert `off <= size`; binding at the end is allowed). |
| F5 | P2 | `metal.rs` `hash_fold`, `hash_fold_tree` (SHA) | The CPU HAL asserts `io.size() >= 2 * input_size`. Metal had no check and read past the buffer. `hash_fold_tree` on SHA did not check `io.size() >= 2 << layers` (Poseidon2 did). | **Fixed** (same asserts in the `Hal` entry points, for both hashes). |
| F6 | P2 | `metal.rs` `gather_sample` (host; Z7 open issue) | No checks: `dst`/`src` ranges, `idx`/`stride` truncated to u32, and the kernel's 32-bit `gid * stride + idx` overflows past 2^32. Not on the Metal prove path (only CUDA/Intel `gather_samples` call it), so only latent. | **Fixed host side** (CPU-equivalent panics plus a 32-bit index check). The kernel itself (`misc.metal`) is Z8a's. |
| F7 | P2 | `metal.rs` `MetalHashSha256::hash_rows`, `fri_fold`, `sha_fold` | `as u32` truncation of row, column and count values with no check. SHA `hash_rows` with 0 rows divided by zero. | **Fixed** (`u32::try_from(..).expect`; 0 rows returns, as Poseidon2 already did). |
| F8 | P2 | `poseidon2.rs` `hash_fold_tree` | `n as u32 <= tail_max` truncates at layer 32 (`n = 2^32` became 0 and was taken as a tail). Unreachable by memory. | **Fixed** (compare in `usize`). |
| F9 | P2 (coverage) | `tests_z3.rs` | The dispatcher tests used CUDA's crossovers (511/512/513, 4095/4096/4097), not Metal's defaults `pvs_w <= 256 < pvs_q <= 2048 < pv`. So 256/257 and 2048/2049 were never exercised through `hash_rows`/`hash_fold`; `rows_each_kernel` has 255 and 2049 but not 256, 257 or 2048. | **Added** `poseidon2_crossover_boundaries`: rows and folds at {1, 255, 256, 257, 1024, 2047, 2048, 2049}, with 1/8/16/17/211/256/257 columns, for both the `pv` and `m32` forms, against the CPU HAL. It passed on the unfixed code, so there is no kernel defect at the boundaries. |
| F10 | P2 | `sha.metal` | Kernel parameters have no explicit `[[buffer(n)]]` (plan 2.4). The implicit order matches the host and the binding comment, so this is hygiene only. | Not fixed. A metallib change would cost a full acceptance re-run; left for Z9 or whoever next touches the file. |
| F11 | P2 | `runtime.rs` peak counter | `PEAK_ALLOCATED` is sampled only after zkp allocations, so growth from PSO or library creation (also in `currentAllocatedSize`) is missed until the next allocation. `reset_peak_allocated` races a concurrent `record_peak`, because the store can lose a larger value. Both are process-wide by design. | Documented. Tests that reset the peak take `gpu_test_lock`. |
| F12 | P2 | `runtime.rs` tg-length dedupe | The dedupe is correct: it resets on every new compute encoder, after `pre` callbacks and in `with_encoder` (Z6 open issue). The caveat F0 noted remains: a `[[threadgroup(i)]]` length set by an earlier dispatch persists in the encoder when a later caller does not pass `tg_bytes[i]`, and the 32 KB assert only sums the passed lengths. Every current caller passes all of its indices. | Documented (no change). |
| F13 | P2 | `runtime.rs` | The budget check is time-of-check/time-of-use: two threads may both pass the check and together exceed the budget. The pool admits small (< 64 KiB) buffers without a cap. Cross-HAL buffer use has no debug assertion (F0 gap). | Documented. |
| F14 | **P0?** (unexplained, seen once) | test process, CPU code | One baseline (pre-Z8b) run with `RISC0_METAL_MAX_WORKING_SET_BYTES=4294967296` under the parallel harness died with **SIGSEGV**. The faulting thread was `z6_combos_divide_po2_21` in pure CPU code: `testutil::divisible_poly` → `Vec::resize` → `realloc` → `_platform_memmove`, `KERN_INVALID_ADDRESS at 0x7d2d000000` (page-aligned), with no Metal frame. Report: `~/Library/Logs/DiagnosticReports/risc0_zkp-19c2dae00ddd4f22-2026-11-10-222154.ips`. It was not reproduced in 11 later parallel runs of the same binary (4 GiB device scope x3, 4 GiB hal scope x2, 8 GiB hal scope x1, validation layer x1, full suite x4) or in 4 serial budget runs. Either libmalloc's old block was unmapped under it (an over-released MTLBuffer whose VA malloc had reused would look exactly like this), or libmalloc deferred reclaim misbehaved under memory pressure. The machine was shared with other agents' GPU jobs. I audited every `MTLBuffer` ownership path (the `newBufferWithLength` +1 taken by `from_ptr`, `ManuallyDrop::take` once per `RawBuffer`, pool `Vec` ownership, `to_owned` on autoreleased command buffers and encoders): no over-release found. | **Open** (see below). |

No finding in the kernels themselves:
- **pv / pvs.** Shuffle sources are always in 0..31: `base + 3 <= 31`, `j + 20 <= 23`, `r + 1 <= 21`, xor masks <= 16. pvs_w exits early per whole SIMD group (`h` is uniform per `simdgroup_index`). pvs_q masks with `act`, so every lane reaches every shuffle. Grids are whole 128-thread groups. PSO max = 128 and SIMD width = 32 were measured on every pv kernel; the new `launch_shapes_fit_psos` test guards them. 32-bit indices: `pv_rows` advances a pointer per column, the pvs rows kernels are gated to `rows * cols < 2^32` on the host, and the fold kernels' 4n index fits for n < 2^30. `fold_tail` reaches the barrier with every thread in every iteration.
- **sha.** It is the CUDA code with `pad = false`. `i * stride` is 64-bit `size_t` in MSL, so it does not overflow at 2^24 x 256+ columns. The double `convertU32` of the init state matches CUDA's single one.
- **mix.** Every `count * id` / `count * i` product is 64-bit. The `mx_redc_pair` bound is 2p^2 + 2^32 p < 2^64, and the result is < 1.94p, so a single conditional subtract suffices. Threadgroup tables are 20 B x 1638 = 32760 and 16 B x 2048 = 32768 (<= 32768, asserted on the host). The barrier comes before the early return.
- **fri.** The indices are `size_t`. The mix value is copied from `constant` into a thread-local `FpExt`.
- **Fault words / use_resource.** None needed in these files: there are no untrusted indices except mix combo ids (host data from the circuit taps; see the issue for Z8a), and no pointer-bearing `set_bytes` structs (`MdJobs` is Z8a's).
- **Sync.** Every `BufferImpl` host path (`view`, `view_mut`, `get_at`, `to_vec`) goes through `sync_for_host`. The only unsynced paths are `as_ptr`/`as_mut_ptr` (documented; debug-asserted) and `copy_from`, which writes only new memory or memory from the pool whose batches have completed. A pooled buffer is reused only after `completed_seq >= ` the open sequence number at drop. A pending buffer drained from the pool is still retained by its command buffer (retained-reference command buffers).

## What changed
| File | Change |
|---|---|
| `zkp/src/hal/metal/runtime.rs` | F1 grid assert; F2 pool registry and `drain_all_pools`; F3 `BudgetScope`, `BUDGET_SCOPE_ENV`, owned-byte accounting (`Pool::owned_bytes`, `allocated`, `freed`, `budget_level`), and `free_size` per scope; docs. |
| `zkp/src/hal/metal/buffer.rs` | F4 asserts; bypass-pool drop reports `freed`. |
| `zkp/src/hal/metal.rs` | F5-F7; re-exports `BudgetScope`, `BUDGET_SCOPE_ENV`; new `budget_scope`, `set_budget_scope`, `owned_allocated_size`; `#[cfg(test)] gpu_test_lock()` (poison-tolerant); `mod tests_z8b`. |
| `zkp/src/hal/metal/poseidon2.rs` | F8. |
| `zkp/src/hal/metal/tests.rs` | `gpu_test_lock` taken by the four tests that assert process-wide state: `pool_reuse_across_batches`, `alloc_failure_drains_pool_and_retries`, `z6_working_set_budget_errors_early` (pool state, which other tests' failures may now drain) and `z6_peak_allocated_size` (resets the peak). |
| `zkp/src/hal/metal/tests_z8b.rs` (new) | The regression and coverage tests below. |

No `.metal` file changed, so the metallib and `KERNEL_NAMES` are unchanged.

## Regression tests (each fails on the pre-fix code)
Run on the base plus only the new test file (before any fix):
`cargo test -p risc0-zkp --features metal --lib -- hal::metal::tests_z8b` gave **10 failed, 3 passed**. Every
`should_panic` test failed with "did not panic as expected", and the two drain tests failed their assertion
(log: `scratchpad/w4/Z8b/prefix_regressions.log`). The 3 that passed are the coverage tests:
`grid_below_u32_runs`, `gather_sample_matches_cpu` and `poseidon2_crossover_boundaries`.

| Test | Finding |
|---|---|
| `grid_threads_beyond_u32_panics`, `grid_groups_beyond_u32_panics` | F1. Before the fix, Metal ran the 2^32-thread grids with no error. The tests use `fill_u32` with `count = 0`, so the pre-fix run wrote nothing. |
| `budget_failure_drains_every_pool`, `alloc_failure_drains_every_pool` | F2: HAL A's pool is empty after HAL B's over-budget or failed allocation. |
| `budget_scope_hal_counts_own_allocations` | F3. A HAL-scope budget ignores another HAL's 64 MiB; it counts live, pooled and bypassed buffers; an over-budget request drains the HAL's own pool and then fits; the error names the scope. This is a new API, so it does not compile before the fix. |
| `arg_offset_past_view_panics` | F4 |
| `hash_fold_small_io_panics_{sha,poseidon2}`, `hash_fold_tree_small_io_panics_sha` | F5 |
| `gather_sample_src_out_of_range_panics`, `gather_sample_dst_too_small_panics`, `gather_sample_matches_cpu` | F6 |
| `poseidon2_crossover_boundaries` | F9 (coverage) |
| `launch_shapes_fit_psos` | pv/pvs launch shapes against the PSO maximum and the SIMD width (audit guard) |

## Verified (in `/Users/user/zkminer/deps/risc0-wt-Z8b/risc0`, after `source /Users/user/zkminer/env.sh; export CARGO_BUILD_JOBS=6`; logs in `/private/tmp/claude-501/-Users-user-zkminer/a30bccea-eec8-43d4-ac51-c5ccb6e4b957/scratchpad/w4/Z8b/`)
- `cargo test -p risc0-zkp --features metal --lib` under the default parallel harness, final code:
  run 1 **183 passed, 0 failed, 8 ignored** (157 s); run 2 **183 passed, 0 failed, 8 ignored** (233 s)
  (`final_par1.log`, `final_par2.log`). Also twice with the intermediate code (F1/F2/F4-F9 only): 182 passed both times.
  The base: 168 passed.
- Validation layer: `RISC0_METAL_DEBUG=1 MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=nslog MTL_DEBUG_LAYER_WARNING_MODE=nslog <test binary> hal::metal`
  (parallel): **153 passed, 8 ignored**. The only Metal output was "Metal API Validation Enabled" (`debug_layer.log`).
- Budgets, plan 9.1 (`hal::metal`, 161 tests incl. 8 ignored):
  - `RISC0_METAL_BUDGET_SCOPE=hal RISC0_METAL_MAX_WORKING_SET_BYTES=4294967296`, parallel: **152 passed, 1 failed**.
    The failure is the known `tests_z2::ntt_commit_coeffs_lde_lg_10_20` at lg 20 x 211:
    "allocating 3539992576 bytes for evaluated with 1769996288 bytes already allocated by this HAL would exceed
    the limit of 4294967296". The numbers are identical to Z6/Z7's serial device-scope run (`b4g_hal_par2.log`).
  - `RISC0_METAL_BUDGET_SCOPE=hal RISC0_METAL_MAX_WORKING_SET_BYTES=8589934592`, parallel: **153 passed** (`b8g_hal_par1.log`).
  - Device scope (production model), `--test-threads=1`, split in two runs each:
    - 4 GiB: tests_z2 14 passed, 1 failed (same lg-20 case); the rest 138 passed.
    - 8 GiB: tests_z2 15 passed; the rest 138 passed.
  - Device scope, 4 GiB, parallel: once 2 failed (lg-20 plus `z6_poly_group_no_coeffs_spill` from interference),
    once only lg-20. This is the scheduling dependence F3 describes; use `hal` scope for parallel budget runs.
- `cargo test -p risc0-zkp --features prove --lib` (CPU HAL): 30 passed.
- `cargo check -p risc0-zkvm --features metal`: exit 0.
- `cargo clippy -p risc0-zkp --features metal --lib --tests --benches -- -A clippy::type_complexity`: no warnings
  (only the pre-existing `block v0.1.6` future-incompat note). `rustfmt --check zkp/src/hal/metal.rs` (incl.
  submodules): clean. `tests_z8b.rs` was formatted with rustfmt.

## Measurements
- **PSO limits** (M5 Max, all from `launch_shapes_fit_psos` and a one-off print):
  - every `pv*` kernel: `maxTotalThreadsPerThreadgroup` = 128 (its attribute), SIMD width 32;
  - mix v1/v2/v2_dcids: 256;
  - all with 0 static threadgroup memory.
- **Peak GPU memory.** Unchanged by this item (no allocation-path changes on the hot path). The 4 GiB boundary
  is lg 20 x 211 commit at 3.30 GiB evaluated + 1.65 GiB held by the HAL (HAL-scope numbers above);
  Z6's per-method table stands.
- **Timings.** None taken: indicative only and not affected (the checks are host-side and O(1) per dispatch).

## Assumed / not verified
- F1's "Metal runs it without error" was observed on the M5 Max (macOS 27.2); other GPUs might reject the
  dispatch instead. The assert makes both cases a clear panic.
- `hal` scope does not observe private buffers being freed: it counts them until the HAL drops. zkp tests
  allocate only small private buffers. keccak's per-call `alloc_private` scratch would accumulate under
  `hal` scope, so keccak budget runs should stay on `device` scope or trim per call.
- F2's production value (rv32im + recursion HALs in one worker under a 4 GiB budget) is argued from the
  accounting, not measured end to end here (M-items / M4).
- F14's cause is unknown (see above).

## Dispositions of open issues addressed to Z8 / Z8b
- **Z6 → Z8**, review of `check_budget` / `record_peak` / tg-length dedupe / `Bound` resets:
  - `check_budget`: fixed (F2, F3).
  - `record_peak`: P2 limits documented (F11).
  - tg-length dedupe and `Bound` resets: verified correct, with the persistence caveat (F12).
- **Z7 → Z8**, `gather_sample` bounds and 32-bit index: host side fixed (F6). The kernel's lack of a
  count argument and its 32-bit multiply are in `misc.metal`, which is Z8a's (issue below).
- **Z2 → Z8**, coalesced transpose relying on 8-lane groups inside one simdgroup: NTT family, Z8a's.
- **R8 / R11 → Z8**, rv32im GPU witgen nondeterminism: re-scoped to **R17** by PLAN 8.7. It is not in Z8b's
  files and, per the task, not touched here.
- **F0**:
  - The sha_rows params-buffer workaround is resolved: `sha.metal` takes `constant` scalars and the host
    passes `KernelArg::Integer` (`tests_z1::hash_scalar_bindings`).
  - The `poseidon2_fold` unused binding is gone (retired in Z7).
  - Cross-HAL buffer use still has no debug assertion (F13, P2).
- **Plan 7 "pvs not audited"**: audited (see above).
- **Plan 9.2 "fri.metal / mix.metal scalars should be constant"**: done (Z6, Z4); verified.

## Issues for other items
- **Z8a**:
  - (P2) `gather_sample` in `misc.metal` has no count or bounds check and computes `gid * stride + idx` in
    32 bits. The host now refuses shapes beyond 2^32 and out-of-range shapes.
  - (P2) `eval.rs` `mix_poly_coeffs` does not check the combo ids against `output.size() / count`. A bad
    combos table makes the kernel write out of bounds, where the CPU HAL panics.
  - (P2) Callers of threadgroup-memory kernels must pass every `[[threadgroup(i)]]` length, because lengths
    persist in the encoder (F12).
- **rv32im owner (R12/R16/R17, whoever owns `witgen_tests.rs`)**: `witgen_snapshot_over_budget_is_an_error`
  sets a device-wide budget of `allocated_size() + 2 MiB` and expects the 13.8 MB snapshot to fail. Since F2,
  an over-budget allocation first drains **every** HAL's pool in the process. With other tests' pooled
  buffers present, the snapshot may then fit and the test stop failing. Make it deterministic with
  `hal.set_budget_scope(BudgetScope::Hal)` and `set_working_set_limit(Some(hal.owned_allocated_size() + (2 << 20)))`.
- **All circuit units, M4**: budget runs can use `RISC0_METAL_BUDGET_SCOPE=hal` to be independent of
  parallel tests. `device` remains the production model and should still be run once serially (as here),
  because it also counts PSOs and libraries.
- **Plan owner**:
  - Add `RISC0_METAL_BUDGET_SCOPE` (`device` | `hal`) to the 2.8 env-knob table.
  - 2.8 still lists `RISC0_METAL_LEGACY`, which was retired in Z7.
- **Z9**:
  - Re-measure the Poseidon2 crossovers on an idle GPU and at small core counts. Boundary correctness is
    now covered at 256/257/2048/2049.
  - (F10) Add explicit `[[buffer(n)]]` to `sha.metal` when it is next touched.
  - (F13) Consider capping the small-buffer pool.
- **Plan owner / M4 / Z9 (F14)**: if the `realloc` SIGSEGV recurs, keep the `.ips` file and the test log. To
  confirm or rule out an `MTLBuffer` over-release, run the 4 GiB parallel suite under
  `MallocScribble=1 MallocGuardEdges=1` and with `OBJC_DEBUG_MISSING_POOLS=YES` / zombies
  (`NSZombieEnabled=YES`).
