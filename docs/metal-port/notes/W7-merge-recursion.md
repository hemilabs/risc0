# W7 merge record: unit recursion (2026-11-15)

Merged into `zkminer/metal` with `--no-ff`:

- `metal/W7-C1` (e8a874ecf, a single commit, no WIP; based on the pre-zkp-merge `zkminer/metal`). It fixes the
  p254 prewarm gate race, turns C14 staging on by default (`RISC0_METAL_REC_STAGE=0` reverts), cuts the
  recursion suite's memory and adds hygiene fixes. Merge commit 964769e98.

The item touches only `risc0/circuit/recursion{,-sys}` and its own note. The zkp and rv32im unit merges did not
touch those files, so there were no conflicts and no interaction fixes. No Metal kernel source changed. The
item's open issues are listed in `W7-C1.md`.

## Gate (main checkout after the merge, logs in `scratch/w7/merge-recursion/`)

`RISC0_METAL_ARCHIVE_GC=0` was set on every GPU test run (shared machine, W7-Z1 note).

| command | result |
|---|---|
| `risc0-circuit-recursion --features metal`, default parallel harness, twice | 81 passed / 0 failed / 21 ignored (+1 doctest), both runs (`suite1.log`, `suite2.log`) |
| `c11_captures_seal_match_cpu` (release, `C11_CAPTURES=fixtures/w5/C11/caps`, `RISC0_METAL_REC_WITGEN_CHECK=1`) | 11 / 11 seals == CPU, 11 / 11 witgens == CPU (`seal_c11.log`) |
| zkvm `host::recursion::tests`, debug lib (`--test-threads=4`, `--huge`) | **14 passed / 2 failed**, pre-existing (see below) (`zk_rec.log`, `zk_rec_run2.log`) |
| same, release lib | 16 passed / 0 failed (`zk_rec_release.log`) |
| worker `BENCH_ONLY=memory-merkle --benchmark`, 4 GiB budget (`gpu-run --budget 4G`, `RISC0_METAL_MAX_WORKING_SET_BYTES=4294967296`), worker built against this checkout | 8,388,608 cycles proved; lift/join 6.02 s; Groth16 wrap 1.98 s; `verified` (`worker4g.log`). This is a gate run, not a timing claim |
| `cargo check -p risc0-zkvm --features metal` | ok |
| `risc0-zkp --features metal --lib hal::metal`, default parallel harness, twice | 201 passed / 0 failed / 48 ignored, both runs |
| `risc0-circuit-rv32im-sys --features metal` (R17 guards) | 10 + 2 passed |
| `emit_ir.sh` + `ir_scan.py` | 175 TUs, tier A = S = 0 (B = 28, C = 29, the same as the rv32im merge) |

### Pre-existing failure: Z14 prefill cross-thread `debug_assert` (zkp, not this unit)

`test_recursion_host_chains_check` and `test_recursion_lift_resolve_e2e` fail in debug builds on
`risc0/zkp/src/hal/metal/runtime.rs:2498` with "Metal prefill: a buffer marked filled on another thread was
taken". This is W7-A1's single-thread check in `Runtime::prefill_take`, which came in with the zkp unit merge.
The failure is not caused by W7-C1:

- It reproduces on the pre-merge head 1f8d3aa8f. In a scratch worktree, the same two tests fail with the same
  panic: 14 passed / 2 failed (`zk_rec_base.log`), and the test also fails when run alone (`zk_rec_base_solo.log`).
- After the merge it fails when run alone too, with `--test-threads=1` (`zk_rec_solo.log`). So the trigger is
  not tests sharing the global pool.
- Both tests prove two or more receipts in a row. Each proof's segments run on a new `segment-stream` thread. The
  panic is on that thread, in rv32im segment proving. Recursion code never calls `prefill_*`.
- Release builds keep only the one-time error log, and the release run passes 16 / 16.

The likely cause is a cached HAL that a later proof's new thread reuses. Its set-aside buffer was marked and
encoded by an earlier, finished thread, so the check fires although the fill completed long ago. That would make
it a false positive, but this merge did not prove it. The fix belongs to the zkp owner: for example, clear
`marked_by` once the marking batch completes, or compare batch sequence numbers instead of threads. The zkp and
rv32im merge gates did not run these zkvm tests, so neither caught it.
