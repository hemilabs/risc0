# B9-SEC: 256-bit OS-seeded ZK noise for the recursion seal check

Branch `metal/B9-SEC`, base `zkminer/metal-3.1` 84df3e8c0. Release blocker (security), fixes the
P31-SEC finding in BASELINES.md section '3.1' ("Security review finding").

## Finding

`RISC0_METAL_REC_SEAL_CHECK=1` (`recursion/src/prove/hal/metal/mod.rs`, `prove_seal_checked`) is
read in release builds. It proves each recursion program on Metal and on the CPU under one noise
seed, compares the seals, and **returns the Metal receipt**. The seed was `rand::random::<u64>()`,
and each noise stream was `StdRng::seed_from_u64(seed ^ stream * K)`. The ZK noise of a returned
receipt (the data and accum tails, `ZK_CYCLES` rows) was therefore a function of 64 bits. Stock 3.1
(`risc0-circuit-recursion` 4.1.0 `prove/witgen.rs`) draws every cell from `rand::rng()`, a ChaCha12
seeded from the OS. With the knob on, the receipt's zero knowledge rested on 64 bits of entropy.
The default path was not affected: it never sets a seed.

## Fix (root cause: the seed type)

`prove/witgen.rs`:

- The seed is now `NoiseSeed = [u8; 32]`. The thread-local holds `Option<NoiseSeed>`, set by
  `with_noise_seed32(seed, f)`.
- `os_noise_seed()`: 32 bytes from the OS (`rand::rngs::OsRng`, i.e. `getrandom`). Errors are
  propagated, never replaced by a fallback.
- Per-stream key: `noise_stream_key(seed, stream) = SHA-256("risc0-circuit-recursion/zk-noise/stream/v1" || seed || stream_le)`.
  `seeded_noise_rng` is `StdRng::from_seed(key)` (ChaCha12 in rand 0.9). Every seed bit and the
  stream index go into the key.
- `with_noise_seed(u64, f)` is now `#[cfg(test)]`. It expands the u64 with
  `SHA-256("risc0-circuit-recursion/zk-noise/test-u64-seed/v1" || seed_le)`, a separate domain, so
  no non-test code path can key the noise from 64 bits. This is enforced by the compiler: the
  function does not exist outside tests.
- `draw_noise(None, ..)` (the default path) is unchanged: `rand::rng()`, one draw per cell.

`prove/hal/metal/mod.rs`:

- `prove_seal_checked` draws `os_noise_seed()?` per proof and runs the Metal and CPU proofs under
  `with_noise_seed32(seed, ..)`. The seal comparison is still bit-for-bit under the same seed. The
  seed is never logged or returned. Under `cfg(test)` it is recorded in `LAST_SEAL_CHECK_SEED` for
  the B9-SEC test.
- The module and `SEAL_CHECK_ENV` docs now describe the knob as a **validation knob**. It doubles
  proof cost and is not a production setting. They also record that its receipts stay zero
  knowledge (256-bit OS seed).

The returned receipt's noise is now ChaCha12 keyed by a 256-bit secret derived from 32 OS bytes.
This is the same strength class as stock's `rand::rng()`, which is ChaCha12 keyed from the OS.

Options not taken:

- Returning an unseeded re-proof: this would add a third proof per recursion step and return a seal
  that was never compared.
- Refusing the knob outside debug builds: P31-10 and P31-14 run it on release builds (zkvm e2e,
  producers), so it is kept and documented instead.

## Proving-path RNG audit (whole risc0-3.1 tree)

Grep: `seed_from_u64|from_seed|StdRng|SmallRng|ChaCha|rand::random|rand::rng()|OsRng|getrandom|with_noise|noise_seed`
over `risc0/**/*.rs`. Each hit was classified by whether it sits after the file's first
`#[cfg(test)]`, in a `#[cfg(test)]` / `feature = "test"` module, or on a proving path.

| hit | where | verdict |
|---|---|---|
| `recursion/src/prove/witgen.rs` `draw_noise`, `with_noise_seed*`, `seeded_noise_rng` | recursion ZK noise | **fixed** (above). Default `rand::rng()` = stock |
| `recursion/src/prove/hal/metal/mod.rs:963` `rand::random::<u64>()` | seal check | **fixed** (`os_noise_seed`) |
| `recursion/src/prove/hal/metal/{tests,c11,c13,z10}_tests.rs` `with_noise_seed(n)` | tests | test-only (`with_noise_seed` is `cfg(test)`) |
| `recursion/src/prove/hal/metal/accum.rs:472`, `witgen/c6_tests.rs`, `witgen/c14_tests.rs`, `z10_tests.rs:69` `seed_from_u64` | test inputs | test-only (`accum.rs` after `#[cfg(test)]` at 345) |
| `recursion/src/lib.rs:103` `rand::rng()` | `testutil` (`#[cfg(feature = "test")]`), eval-check inputs | not proving |
| `rv32im/src/prove/mod.rs:35` `preflight_rand_z` (`rand::rng()`, or `ONE` with `witgen_debug`) | rv32im preflight | same as stock 4.1.0 (`prove/hal/mod.rs:133/147`). `witgen_debug` is not enabled by zk-miner or the tree's defaults. rv32im has no ZK noise of its own: zero knowledge comes from the recursion lift |
| `rv32im/src/prove/hal/{intel,cuda,hip}.rs`, `metal/eval_check.rs` `StdRng::seed_from_u64` / `rand::rng()` | `EvalCheckParams` test inputs | test-only (after `#[cfg(test)]`; cuda/hip not built on macOS) |
| `keccak/src/prove/testutil.rs:50` | keccak eval-check inputs | test-only (`#[cfg(test)] pub mod testutil`). keccak has no ZK noise: its receipt is lifted by recursion |
| `zkp/src/{core/ntt,hal/cpu,hal/mod,hal/intel,prove/merkle}.rs`, `zkp/src/hal/metal/tests_*.rs`, `eval.rs:390` | zkp | all inside `#[cfg(test)]` modules. The zkp prover has no RNG: `iop().random_*` is the Fiat-Shamir transcript |
| `groth16-sys/src/metal/prover.rs` `random_fr` (`/dev/urandom`, 64 bytes mod r, nonzero) | Groth16 blinding r, s | OS randomness, per proof. OK |
| `groth16-sys/src/metal/{small,prover}.rs` ChaCha fuzz (all-zero key) | Groth16 MSM fuzz | the CUDA prover's public constant fuzz, removed exactly. It has no privacy role (zero knowledge comes from r, s; prover.rs module doc) |
| `groth16-sys/src/metal/{ntt,compact,srs,coeffs,tests_tierz}.rs` `ChaCha8Rng::seed_from_u64` | tests | test-only (after `#[cfg(test)]`) |
| `core/src/field/{mod,baby_bear}.rs` | field tests | test-only |
| `zkvm/src/claim/{merge,work}.rs` `rand::random`, `host/server/testutils.rs` `OsRng`, `binfmt/src/povw.rs` | tests | test-only |
| `zkvm/methods/**`, `zkvm/platform/src/getrandom.rs`, `build/src/lib.rs` | guest-side getrandom | not prover noise (stock) |

No other seal-check or seeded-noise knob exists in the tree (`SEAL_CHECK|ZCHECK` grep).
`RISC0_HIP_REC_ZCHECK` compares witnesses and does not seed. zk-miner-3.1 does not set
`RISC0_METAL_REC_SEAL_CHECK` and cannot reach `with_noise_seed*` (`pub(crate)`).

## Profile / performance

The default path is untouched. Full diff: `scratch/w9/B9-SEC/B9-SEC.diff`. The only change it
sees is the thread-local `Cell<Option<u64>>` -> `Cell<Option<[u8; 32]>>`, read once per fill. The
`None` branch of `draw_noise` is byte-identical. The seal-check path adds two SHA-256 over 82 bytes
and one 32-byte `getrandom` per proof. That is microseconds against a CPU re-proof of 3-21 s.

Interleaved A/B, `c11_bench_captures` (4 po2-18 lift captures `rows134164`, `C11_REPS=3`; joins
`rows2596*`), A = base 84df3e8c0 test binary, B = B9-SEC. Order A B A B, 8 pairs, one process per
arm, `scratch/w9/B9-SEC/ab/ab.sh` and `sum.py`. **The machine was loaded by other work: load
average 9-30**.

| captures | budget | A ms/lift | B ms/lift | B-A per pair [min, max] | A-vs-A |
|---|---|---|---|---|---|
| lifts | default (16 GiB) | 108.30 [106.31, 111.28] | 107.06 [106.07, 110.84] | -0.15 [-4.15, +1.22] (5/8 faster) | +0.14 [-2.19, +4.66] |
| lifts | 8 GiB | 111.01 [106.61, 111.31] | 110.67 [106.38, 111.66] | -0.10 [-0.51, +0.59] (5/8) | -0.02 [-0.34, +4.45] |
| lifts | 4 GiB | 106.39 [106.10, 110.84] | 106.31 [106.13, 115.24] | +0.01 [-0.33, +4.40] (3/8) | -0.03 [-4.48, +0.33] |
| joins/resolves | default | 127.58 [126.57, 128.76] | 127.63 [126.10, 128.93] | -0.11 [-1.26, +0.54] (4/8) | -0.01 [-1.16, +0.57] |

Every difference is inside the A-vs-A spread, so the result is **no change**. The GPU and host
memory of the default path is unchanged (no new buffers). The seal-check path holds 32 bytes
more. The 3.1 baseline (BASELINES '3.1') still applies. The full `--benchmark` was not re-run,
because the default path code is unchanged.

No Metal kernel changed (Rust host code only), so R17's `ir_scan.py` / `run_stress.sh` do not apply.

## Correctness

Logs are in `scratch/w9/B9-SEC/logs/`. Every GPU run was under `gpu-run` heavy (the zkvm e2e
under `--huge`).

| check | result |
|---|---|
| New `b9_sec_noise_key_uses_every_seed_bit` (host): flipping any one of the 256 seed bits changes the stream key and the first RNG output, for both streams; stream 1 != stream 2; two OS seeds differ; the u64 test expansion is domain-separated | pass |
| New `b9_sec_seal_checked_receipt_noise_is_256_bit` (Metal + CPU, po2 16): two `prove_seal_checked` runs (Metal vs CPU, the knob's code path) use different OS seeds and give different seals. The CPU re-proof under the recorded 256-bit seed reproduces the returned Metal seal. Flipping seed bit 64, 128 or 255 changes the seal. The seed's low 64 bits used as a `u64` test seed do not reproduce it. The seed has nonzero bytes above 64 bits | pass |
| **Negative control**: the key derivation temporarily truncated to `seed[..8]` (64 bits, not committed) | both new tests **fail** (`negative_control.log`), so they detect a 64-bit key |
| `p31_09_*` (7 tests; `seeded_stream` now recomputes the SHA-256/ChaCha stream independently of `witgen`), `noise_seed_controls_the_seal`, `seal_check_detects_a_difference`, `seal_matches_cpu_*`, `c11_*seal_matches_cpu*`, `c13_fresh_prover_seal_matches_cpu_*`, `z10 p254_seal_matches_cpu_po2_{16,18}` | 23/23 (`targeted_run1.log`) |
| `risc0-circuit-recursion --features metal --release --lib`, default parallel harness, x2 | **95 passed, 0 failed, 23 ignored, both runs** (`suite_run{1,2}.log`; the 2 new tests included) |
| `risc0-circuit-recursion --release --lib` (CPU only, no metal) | 6 passed |
| `RISC0_METAL_REC_SEAL_CHECK=1` zkvm `--features metal --release`: `test_recursion_lift_join_identity_p254_e2e`, `test_recursion_lift_resolve_e2e` | **2 passed**. 11 po2-18 seals == CPU (lifts 134515/121777 rows, joins 259620, resolves 259676, identity_p254 129813) (`zkvm_e2e_sealcheck.log`) |
| `tools/stock_verify31/run_matrix.sh` with `RISC0_METAL_REC_SEAL_CHECK=1`, `CASES="fibonacci compose" VERSIONS=31 BUDGETS=""` (lift, join and resolve receipts made with the knob on: 8 seal checks in the producers) | composite, succinct and Groth16 for both cases: **STOCK 3.1 VERIFY: OK** (crates.io 3.1.0 verifier), exit 0 (`sv31_matrix.log`, `sv31/logs/`) |
| `cargo clippy -p risc0-circuit-recursion --features metal --release --tests` and `--features prove --tests` | clean apart from the 2 pre-existing `useless_vec` errors in unchanged `witgen.rs` lines (P31-09 note) |

Test isolation (PLAN 8.7): the new GPU test takes `heavy_test_lock` and `global_state_test_lock`
(it reads the process-wide `LAST_SEAL_CHECK_SEED`). The host test touches no shared state.

## Knobs

- `RISC0_METAL_REC_SEAL_CHECK=1`: validation knob (C9), off by default. It proves every recursion
  program on Metal and the CPU under one fresh 256-bit OS seed and requires identical seals. It
  doubles cost. Its receipts are now zero knowledge, with 256-bit OS-seeded noise. Not for
  production use.
- No revert knob restores the 64-bit seed. There is no performance optimization to revert, and the
  old behaviour was the defect.

## Open issues

- BASELINES.md '3.1' still describes the finding as open. The merge agent should mark it fixed by
  B9-SEC (that file is not owned here).
- Not run: the zk-miner worker `--benchmark` (the default path code is unchanged). Budgeted matrix
  rows (`BUDGETS`) with the knob on were also not run; the budget does not interact with the seed.
