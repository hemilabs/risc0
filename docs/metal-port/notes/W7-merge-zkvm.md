# W7 merge record: unit zkvm (2026-11-15)

Merged into `zkminer/metal` with `--no-ff`:

- `metal/W7-V1` (edb7cb373, one commit, no WIP): memory-safe default zkvm test harness (`heavy_test_lock`),
  budget-sized keccak/sha tests, Z17/gdb skips. Merge commit ed4ea463b.
- `metal/W7-V2` (9994dc266, one commit, no WIP): keccak at small Metal budgets (headroom cap, HAL buffer
  release, auto rebuild). Merge commit fde9b8830.

Both branches are based on b5fa9a97a (the wave-6 baseline), not on the current `zkminer/metal`. They touch
disjoint files, and the zkp, rv32im, recursion and groth16 unit merges did not touch those files either. So
there were no conflicts and no interaction fixes. V2's question about its comment-only `exec_stream.rs` change
and the `mod.rs` fmt reorder is settled: V1 changes neither file. No Metal kernel source changed. The items'
open issues are in `W7-V1.md` and `W7-V2.md`.

## Gate (main checkout after both merges, logs in `scratch/w7/merge-zkvm/`)

`RISC0_METAL_ARCHIVE_GC=0` was set on every GPU run except the two debug zkvm suite runs.

| command | result |
|---|---|
| `cargo check -p risc0-zkvm --features metal` | ok |
| zkvm `--lib -- host::server::prove`, debug, default parallel harness, `--huge`, twice | **104 / 103 passed, 5 / 6 failed**, 7 ignored. Pre-existing failure, see below (`prove1.log`, `prove2.log`) |
| same, on the pre-merge head 4217b43d2 (scratch worktree) | 99 passed, **the same 6 failed** with the same panic (`prove_base.log`) |
| same, release lib | **109 passed / 0 failed**, 7 ignored (`prove_release.log`) |
| zkvm `--lib -- host::recursion::tests`, debug | 11 passed / 2 failed (`test_recursion_host_chains_check`, `test_recursion_lift_resolve_e2e`), the pre-existing failure from `W7-merge-recursion.md` (`rec.log`) |
| same, release lib | 13 passed / 0 failed, 3 ignored (`rec_release.log`) |
| worker `--benchmark`, all six programs, `--no-budget` (zk-miner `metal` built against this checkout) | every proof and Groth16 wrap `verified`; sha256-chain 923K, ecdsa-verify 925K c/s; lifetime peak 14019 MiB (`workerdef.log`). This is a gate run, not a timing claim |
| worker `BENCH_ONLY=memory-merkle --benchmark`, `--budget 4G` | po2 18, STARK 13.76 s, wrap 1.63 s, `verified`, peak 4892 MiB (`worker4G.log`) |
| same, `--budget 8G` | po2 19, STARK 10.10 s, wrap 1.65 s, `verified`, peak 6772 MiB (`worker8G.log`) |
| `k5_keccak_metal_e2e --ignored`, `K5_E2E_PO2S=14,17,18`, release, 4 GiB / 8 GiB | 4 GiB: 14 and 17 ok (18 skipped by the cap, as V2 documents); 8 GiB: 14, 17 and 18 ok (`k5_4G.log`, `k5_8G.log`) |
| `risc0-zkp --features metal --lib hal::metal`, default parallel harness, twice | 201 passed / 0 failed / 48 ignored, both runs (`zkp1.log`, `zkp2.log`) |
| `risc0-circuit-rv32im-sys --features metal` (R17 guards) | 10 + 2 passed (`r17.log`) |
| `emit_ir.sh` + `ir_scan.py` | 175 TUs, tier A = S = 0 (B = 28, C = 29, the same as the earlier W7 merges) (`irscan.log`) |

### Pre-existing failure: Z14 prefill cross-thread `debug_assert` (zkp, W7-A1)

In debug builds, `sha_basics`, `sys_verify::sys_verify_1`, `sys_verify::sys_verify_integrity_halt_1` and
`tests_z17::z17_stream_{matches_loop,error_then_prove,succinct_twice}` panic at
`risc0/zkp/src/hal/metal/runtime.rs:2498` ("Metal prefill: a buffer marked filled on another thread was
taken"). The panic is on the `segment-stream` thread or the test thread. This is W7-A1's single-thread check
in `Runtime::prefill_take`, the same failure that `W7-merge-recursion.md` records for the two recursion tests.
W7-V1 and W7-V2 did not cause it:

- The pre-merge head fails the same six tests with the same panic.
- Release builds keep only the one-time error log, and release runs pass both modules completely.
- `z17_stream_succinct_twice` failed in one of the two merged runs and on the base, so it is timing-dependent.

The zkp owner should fix it, for example by clearing `marked_by` once the marking batch completes, or by
comparing batch sequence numbers instead of threads. Until then the debug zkvm `host::server::prove` and
`host::recursion::tests` gates cannot pass on any tree that contains W7-A1.
