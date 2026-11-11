# Z8a: review of the zkp Metal kernels (NTT, eltwise, misc, mdiv, eval)

Scope (PLAN 4.1 row Z8, family split A): `sys/kernels/zkp/metal/{ntt_x,eltwise,misc,mdiv,eval}.metal`
and `zkp/src/hal/metal/{ntt,small,mdiv,eval,kernels}.rs`. Z8b owns the rest (`metal.rs`,
`runtime.rs`, `buffer.rs`, `poseidon2.rs`, `tests*.rs`, `pv/sha/fri/mix.metal`).

Method: a line-by-line read of every kernel against its CUDA source (`sys/kernels/zkp/cuda/{eltwise,kernels,combos}.cu`,
`supra/{ntt_x.cuh,mdiv.cuh}`, ffi.cu launchers; the HIP branches only differ in ntt step splits and
pv, neither relevant here), the checklist below, plus a new tool: the whole `hal::metal` suite under
Metal GPU validation (`MTL_SHADER_VALIDATION=1`). Result: **no P0 and no P1 found**. Seven P2s
fixed, two P2s documented and not fixed (reasons below), one tooling finding passed to R17.

## Checklist results

| Check | Result |
|---|---|
| simd_shuffle / simd_ballot lanes out of range | No `simd_ballot` in these files. NTT: `simd_shuffle_xor` lane masks are `1 << (s-1)` with `s < min(iterations, 6)`, so at most 16, and always below the threadgroup size (block = `min(2^(lg-1), 2^(radix-1))`, radix >= 6). mdiv and eu2: `simd_shuffle_down/up` from lanes outside the 32 are discarded by `csel` (mdiv) or never reach lane 0 (eu2 tree reductions, which only use lane 0's result, as on CUDA). All of it assumes 32-wide SIMD groups made of consecutive thread indices. That was assumed silently before; it is now asserted (finding 6). |
| Bounds under grid rounding | Every `Grid::Groups` launch in these files is exact: NTT `groups * block * Z * 2 = n`, num_blocks is a power of two >= 8 when divided by Z_COUNT; bitrev tiles n/4096; mdiv (ntiles, njobs); eval one group per point / (S, nEvals). `Grid::Threads`/`Threads2D` use `dispatch_threads` (non-uniform, exact). Kernels that take a count still check it. The only rounded grid is mix (ceil(count/256) groups, kernels in Z8b's mix.metal check `idx < count` after the barrier). |
| 32-bit index overflow | Findings 1, 2, 3, 7. Everything else is in range: NTT column offsets are `ulong(blockIdx.y) * stride`, in-column indices < 2^27; mdiv offsets are asserted to fit u32 (plan); eval uses `size_t` for `which * deg`; small.rs asserts each 32-bit index on the host (`to_u32`). |
| Host access after GPU writes goes through sync | mdiv returns remainders through `to_vec` (syncs) and the host fallback through `view_mut`; the NTT self-check reads through `to_vec`; small.rs only writes fresh pooled buffers (not GPU-owned until the dispatch). No raw `contents()` reads in these files. |
| Threadgroup memory <= 32768 | NTT <= 8 * 4 * 512 = 16 KB (radix <= 10 for lg <= 27; checked: max step 10 at lg 17/18/26), bitrev 16 KB, Horner 4 KB, mdiv static 1040 B, eu2 static 128 B, mix asserted in eval.rs (and the runtime asserts static + dynamic against `maxThreadgroupMemoryLength` on every dispatch). |
| use_resource coverage | No pointer-carrying structs in these kernels: mdiv jobs hold element offsets into explicitly bound buffers. Nothing needs `use_resource`. |
| Diff vs CUDA/HIP | Matches line for line, except: Z2's skip-expand `s0` fix (`!skip` instead of `s0 == 1`, documented in the kernel), GS `nat` output not ported (no caller), `bit_rev` with nbits 0 made explicit (MSL shift by 32 is undefined), and the mdiv extensions documented in mdiv.metal (offset jobs, E = 16, K tiles per scan thread, scan padding). Findings 2 and 4 are the only diffs that mattered. |
| Error paths (fault words) | None of these kernels take a fault word. Indices from host data are validated on the host (small.rs, mdiv.rs). Device-resident index buffers (`which`, mix `combos`) are not (finding 8). |

## Findings

| # | Sev | Where | Finding | Status |
|---|---|---|---|---|
| 1 | P2 | misc.metal `gather_sample` (Z7 open issue) | `src[gid * stride + idx]` was computed in 32 bits, as on CUDA. It wraps once the matrix passes 2^32 elements (16 GiB), and Metal buffers can reach ~22 GiB. No bounds check, because there is no size argument. The host launches exactly `size` threads with `dispatch_threads`, so no thread runs past the end. No prover path calls it on Metal (merkle.rs uses `view` on unified memory). | Fixed: the index is now computed in 64 bits. The host side (`idx as u32`/`stride as u32` truncation, no dst/src size check in `metal.rs::gather_sample`) belongs to Z8b. |
| 2 | P2 | eltwise.metal `eltwise_sum_fpext` | Input index `count * i + gid` was 32-bit; CUDA's loop variable is `size_t`, so CUDA is 64-bit there. Diverges from CUDA once count * to_add >= 2^32. | Fixed: index `ulong(count) * i + gid`. small.rs asserts the output size (4 * count) fits u32, which the 32-bit output index needs. |
| 3 | P2 | small.rs `scatter_bits` | The kernel reads `data[gid * 3 + 2]` in 32 bits; only `n` was range-checked, not `3n`. | Fixed: host asserts `3n` fits u32. |
| 4 | P2 | ntt_x.metal `ct_x/gs_x` coalesced (Z2 open issue) | After the load transposes, the store transpose of a pass with `iterations <= 6` writes columns of the 8x8 exchange tile that other lanes of the same 8-lane group read as rows in the load transpose. Nothing ordered the two (`iterations > 6` gets the V3 threadgroup barrier). The `simd_shuffle`s in between are not memory barriers. CUDA relies on warp lockstep here, and so did the port. This is formally a race under the MSL memory model, though it works in practice. Also checked: the 8-lane groups never cross a SIMD group (`y = tx & ~7`; coalesced passes only run at lg >= 12, where the block is >= 32 threads), so `simdgroup_barrier` is the right scope. | Fixed: `simdgroup_barrier(mem_threadgroup)` before the store transpose when `iterations <= 6` (both kernels). Results unchanged (all NTT tests green). |
| 5 | P2 | mdiv.rs `mdiv_plan` | A combo index listed twice in `chunks` became two concurrent jobs on the same slice: a race with silently wrong output. The host fallback already panicked on this ("listed twice"). Prover callers never pass duplicates (prover.rs builds `chunks` with `i` = 0..combo_count, then combo_count). | Fixed: `mdiv_plan` panics with the host path's message. Regression tests `mdiv::tests_z8a::{plan,gpu_path,host_path}_rejects_duplicate_combo`. The first two **fail before the fix** (verified by reverting the check: "should panic" failures). |
| 6 | P2 | ntt.rs, mdiv.rs, eval.rs (Z5 assumption) | Every cross-lane step assumes 32-wide SIMD groups (`t & 31`, `t >> 5`, xor masks <= 16, 8-lane transposes). This was never checked. A PSO with another `threadExecutionWidth` would compute wrong results silently. | Fixed: `kernels::assert_simd_width` (panics, plan 2.9 programming error) before every ct/gs, mdiv and eu2 dispatch. Test `eval::tests_z8a::simd_width_is_32` covers every such PSO. |
| 7 | P2 | ntt.rs `x_step` | `in_stride as u32` / `out_stride as u32` truncated silently. These are unreachable today: strides are column lengths <= 2^27. | Fixed: `u32::try_from(..).expect(..)`. |
| 8 | P2 | eval.metal / eval.rs (Z4 open issue) | `which[e]` (eu2, Horner) and the mix combo ids are not range-checked. A bad value reads (eval) or writes (mix) outside the buffer, where the CPU HAL would panic. These indices come from the prover's tap tables, not from a trace, and CUDA behaves the same. Checking them needs a host read of a device buffer, which would sync between eval groups. | Not fixed: accepted as prover-internal. If wanted, the cheap form is a clamp using the `_pad` words of `EvalAnyParams`/`Eu2Params` (memory-safe, wrong value instead of a panic). |
| 9 | P2 (tooling) | eval.metal `eu2_main` under `MTL_SHADER_VALIDATION=1` | With Metal GPU validation on, `eu2_main` returns a few wrong partial sums per dispatch, nondeterministically. Without validation it is exact: 0 wrong in 200 dispatches, and every test is green. Under validation, tests that use eu2 fail intermittently (`tests_z4::eval_eu2_shapes`, `eval_forced_paths_agree`, `tests::batch_evaluate_any`, `z6_eval_any_shapes`, `z6_threadgroup_length_dedupe`, `z6_prove_verify_round_trip_sha256`); everything else passes (136/142). Details and variants below. | Not fixed in the kernel: the only forms that stay correct under validation are 1.5-15x slower (below) or recompute the powers. Ignored reproducer `eval::tests_z8a::eu2_main_repeat` (`Z8A_REPS=n`). Lead for R17. |

Not findings (checked and correct): the `bit_rev_z_batched_64` early `return`s and barriers are
uniform per threadgroup (64-thread groups, `group_idx == blockIdx.x`). `bit_rev` at nbits 0 (lg 12)
is handled. `apply_intermediate_twiddles` plus-one row `nbits - 10 <= 11` holds, because every
non-first pass has `pstage >= 6` for lg 11..27. The mdiv suffix scan's trailing barrier separates
the two scans in `mdiv_apply`, and its `has_next` branch (with barriers) is uniform per threadgroup.
The Horner `while (cur)` reduction needs a power-of-two block (always 256). `eltwise_copy_elem_slice`
overlapping rows race exactly as on CUDA, and no caller overlaps (Z1 note). The eu2 lane-0 tree
reductions are correct even though lanes >= 32-d read their own value.

### Finding 9 in detail (eu2_main under GPU validation)
Shape: deg 2^16, 223 polys, 865 evals, S = 2 (the testutil shape). Harness: `eu2_main` alone,
recompiled at runtime from variant sources (`new_library_with_source`), 50-100 dispatches, each
partial checked against a CPU Horner. Normal = no validation, SV = `MTL_SHADER_VALIDATION=1`.

| Variant of eu2_main | normal | SV |
|---|---|---|
| v0: as shipped (`FpExt xp[16]` power table, `uint32 c[16]` from uint4 loads, pair REDC) | 0/200 bad, 0.91 ms | 47-98 % of dispatches bad (1-55 wrong partials each), 1.32 ms |
| v1: plain `Fp` mul/add instead of 64-bit pair REDC | - | 48/50 bad |
| v2: threadgroup-memory reduction instead of shuffles | - | 44/50 bad |
| v3: scalar u32 loads instead of uint4 | - | 50/50 bad (9491 partials) |
| v5: no `max_total_threads_per_threadgroup` | - | 49/50 bad |
| v8: no `c[]` array, keep `xp[16]` | - | 100/100 bad, every partial wrong |
| v10: power table as `uint32 xr[4][16]` | 0/100 bad, 15 ms (spills) | 100/100 bad |
| v6: unrolling disabled on the inner loops | 0/100 | 85/100 bad |
| v9: `xp` indexed dynamically (`xp[j ^ (t >> 8)]`, forces stack) | 0/100 | **0/100** |
| v7: no power table (running `x^j` product), keep `c[]` | - | **0/100** |
| v4: no tables at all (per-coefficient FpExt * Fp) | - | **0/100** |

So the trigger is the per-thread power table (64 words, constant indices after unrolling) under the
validation-instrumented compile. Changing the arithmetic, the loads, the reduction or the launch
bound does not help. Wrong partials have all four limbs wrong and do not equal the true value minus
one thread's or one SIMD group's contribution, so the error is inside a thread's own accumulation.
No UB was found in the source: every array element is written before it is read, indices are
compile-time constants, the 64-bit REDC is in range, and `FpExt` default-constructs to zero.
Simple kernels stay exact under validation (a 2^24 copy, 20 reps: 0 bad), and so do all the NTT,
mdiv and small kernels in the suite. **Assumed, not proven:** this is a bug in how the driver compiles
the instrumented version of a register-heavy kernel (spills or register allocation), not a bug in
eu2_main.

## Verified (in /Users/user/zkminer/deps/risc0-wt-Z8a/risc0, after `source /Users/user/zkminer/env.sh; export CARGO_BUILD_JOBS=6`)
Logs: `/private/tmp/claude-501/-Users-user-zkminer/a30bccea-eec8-43d4-ac51-c5ccb6e4b957/scratchpad/w4/Z8a/`.
- Baseline before any change: `cargo test -p risc0-zkp --features metal --lib -- hal::metal`: 138 passed, 8 ignored.
- After the fixes, default parallel harness:
  - `cargo test -p risc0-zkp --features metal --lib -- hal::metal`: **142 passed**, 0 failed, 9 ignored, run three times (85-159 s; the machine was shared). Peak RSS 20.5 GB (`/usr/bin/time -l`).
  - `RISC0_METAL_DEBUG=1 MTL_DEBUG_LAYER=1` (same command): 142 passed, run twice. No `-[MTL` assertion in the log.
  - `RISC0_METAL_MAX_WORKING_SET_BYTES=8589934592`: 142 passed.
  - `RISC0_METAL_MAX_WORKING_SET_BYTES=4294967296`: 141 passed, **1 failed**: `tests_z2::ntt_commit_coeffs_lde_lg_10_20` at lg 20 x 211 columns, with the budget error (`allocating 3539992576 bytes for evaluated with 1771765760 bytes already allocated would exceed the limit of 4294967296`). This is the known shape Z6 recorded: a 211-column po2 20 group does not fit a 4 GB budget. It is a working-set-budget error, not a wrong result, and it is unchanged by this item.
  - `cargo test -p risc0-zkp --features prove --lib`: 30 passed.
  - `cargo check -p risc0-zkvm --features metal`: green.
  - `rustfmt --edition 2021 --check` on the five Rust files: clean.
- New tests: `eval::tests_z8a::simd_width_is_32`, `mdiv::tests_z8a::{plan_rejects_duplicate_combo, gpu_path_rejects_duplicate_combo, host_path_rejects_duplicate_combo}`, ignored reproducer `eval::tests_z8a::eu2_main_repeat`.
- Extra (not an acceptance gate): the whole suite under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 RISC0_METAL_DEBUG=1`: only eu2 users fail (finding 9). That was 3 tests on the baseline and 6 after the fixes; the extra failures are the same eu2 tests failing intermittently. No command-buffer errors from the NTT, mdiv, small or eltwise kernels. **Assumed:** GPU validation turns out-of-bounds device accesses into command-buffer errors here. No positive control (a deliberate out-of-bounds access) was run.
- `cargo clippy -p risc0-zkp --features metal --tests` fails on pre-existing `type_complexity` errors in `prove/fri.rs:133` and `prove/prover.rs:669-709` (not files of this item). There are no clippy findings in the five reviewed Rust files.

## Measurements
- eu2_main at the testutil shape: 0.91 ms median per dispatch (normal), 1.32 ms (GPU validation). The table-free forms that stay correct under validation cost 6.5-15 ms (v10, v9-class stack forms). Indicative only: the GPU was shared.
- Peak GPU memory: no allocation changed. The new host checks allocate nothing, and the mdiv dedupe uses a host `HashSet` of at most `combo_count + 1` entries.

## Open issues addressed to this item: dispositions
- **Z7: gather_sample bounds / 32-bit index**: kernel fixed (finding 1). The host truncation and missing size check in `metal.rs` go to Z8b. With `dispatch_threads` there is no grid-rounding overrun.
- **Z2: coalesced transpose relies on 8-lane groups inside one simdgroup**: verified true (finding 4 analysis), and the remaining lane-order assumption is now explicit (`simdgroup_barrier`) and asserted (finding 6).
- **Z2: self-check flag set before the check finishes**: accepted. The tables are deterministic, so a second HAL that runs NTTs during the first one's self-check only changes when a failure is reported, never a result.
- **Z4: untrusted `which`/`combos` indices**: finding 8, not fixed, with reasons.
- **Z4: Horner assumes 256-thread groups**: the host always launches 256, the threadgroup memory (4 KB) is sized for 256, and the kernel uses `threads_per_threadgroup`. Consistent; no change.
- **Z5: SIMD lanes are `t & 31` (not probed)**: now asserted per dispatch (finding 6).
- **Z6: review `check_budget` / `record_peak` / tg-length dedupe / `Bound` resets**: the code lives in `runtime.rs`/`metal.rs`, owned by Z8b. Forwarded there.
- **Z1: `eltwise_copy_elem_slice` overlapping target rows race**: same as CUDA, callers cannot overlap (re-checked). No change.
- **R8 / R11: rv32im GPU witgen nondeterminism (addressed to "Z8")**: moved to R17 by PLAN 8.7. Not worked on here. Finding 9 is passed to R17 as a lead.

## Issues for later items
- **Z8b**: `metal.rs::gather_sample` should range-check `idx`, `stride` and `size` against `src`/`dst` and use `u32::try_from`, not `as u32`. `metal.rs` 1-D `launch` passes `count` as u64, but kernel `gid` is 32-bit, so a launch above 2^32 threads would wrap (unreachable below 16 GiB of Fp). Consider asserting it in `launch`.
- **R17 (GPU witgen correctness)**: finding 9 shows the Metal toolchain/driver computing a register-heavy kernel wrongly and nondeterministically when its codegen changes (here: GPU validation instrumentation), while the normal compile is exact. The failing form keeps a 64-word constant-indexed per-thread table live across a loop. Making it dynamically indexed (stack) or removing it fixes it, and arithmetic changes do not. Suggested experiments for the rv32im witgen kernel: (a) run R8's `witgen_segment_file` under `MTL_SHADER_VALIDATION=1` and compare failure rates; (b) look for large constant-indexed thread arrays or very long live ranges in the generated TU (cf. `Fp x33[1007]` / `FpExt x34[144]` in plan 2.1), and try forcing them to device scratch or to dynamic indexing; (c) runtime-compile variants with `new_library_with_source` as done here (harness pattern in this note) to bisect quickly. The `eu2_main` v0/v8 pair (100 % wrong under validation, exact without) is a small kernel that could serve as the core of an Apple Feedback report if R17 confirms the same root cause.
- **Z9 (optimization)**: if eu2_main is rewritten for speed, re-run `eval::tests_z8a::eu2_main_repeat` under `MTL_SHADER_VALIDATION=1`. A form that is exact under validation would make the whole zkp suite validation-clean, which is useful as a review tool.
- **Owner of `zkp/src/prove/{fri,prover}.rs` / merge agent**: `cargo clippy -p risc0-zkp --features metal --tests` fails on `clippy::type_complexity` (7 errors, pre-existing).
