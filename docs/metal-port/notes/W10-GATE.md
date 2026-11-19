# W10-GATE: gate the fork's non-Metal changes to Metal builds

Branch `metal/W10-GATE`, base `zkminer/metal-3.1` 28b4bef33 (W10-R0 merged: upstream release-3.1
5beab26b5 plus the audit). Wave 10 (merge wave), user decision 2026-11-18 "Gate to Metal": once
zk-miner main pins this fork for every platform, Linux/Windows CUDA/ROCm/Intel/CPU builds must behave
like upstream hemilabs/risc0 release-3.1 by default. Scratch and logs: `scratch/w10/W10-GATE/`
(`logs/`, `abd/` and `ab4/` worker A/Bs, `sv31/` stock-verifier matrix, `scripts/`).

## Result

- Every fork change that W10-R0 found active off Metal is now **off by default unless the build is a
  Metal build**, `cfg!(all(feature = "metal", any(target_os = "macos", target_os = "ios")))`. In these
  crates `metal` only builds on macOS/iOS (the sys crates' and zkvm's `compile_error!`), so this is the
  crates' existing Metal gate.
- Each env knob keeps its meaning and is an explicit opt-in elsewhere: unset means the build's default,
  the knob's existing "off" spelling turns the change off, and any other value turns it on, on every
  build. The off spellings are unchanged: `0` for most, `0`/`off`/`false` for the zkr cache.
- `risc0_circuit_recursion::prove::zkr::get_zkr` is upstream's again on every platform:
  `fn(&str, usize) -> Result<Program>`, with the same body. The Metal path uses the new
  `get_zkr_shared -> Result<Arc<Program>>`.
- Metal behaviour is unchanged, by construction and by measurement:
  - Every default on Metal is the old one, and each knob parses the same way as before.
  - The five metallibs are byte-identical to W10-R0's.
  - Every Metal suite passed twice.
  - The stock 3.1 verifier accepts 30/30 receipts at the default, 8 and 4 GiB budgets, and 3.0 receipts
    are rejected 60/60.
  - The worker A/B (8 interleaved rounds at the default budget, 8 at 4 GiB) is inside A-vs-A for STARK,
    wrap and footprint. Device peaks are identical.
- No performance work, no kernel change, no zk-miner change. Peak memory is unchanged (measured, below).

## The changes (one row per change)

The gate is `METAL_BUILD` / `*_DEFAULT` = `cfg!(all(feature = "metal", any(target_os = "macos",
target_os = "ios")))`. "CPU test" means a test that runs in the CPU-only (no `metal`) suite on this Mac.
It compiles on every platform and asserts the upstream default there. "Metal test" means the same test
in the `--features metal` suite, asserting the Metal default.

| # | change (W10-R0 issue) | gate (file:line) | off Metal by default (= upstream) | knob | how checked |
|---|---|---|---|---|---|
| 1 | K10 keccak right-sizing at the executor | `zkvm/src/host/server/exec/syscall/keccak.rs:35` (`KECCAK_RIGHTSIZE_DEFAULT = METAL_BUILD`), `:60` `rightsize_enabled` | requests carry the session's keccak max po2. Inputs are cloned and cleared after the hand-off, exactly as upstream (Metal still moves them, `:159`) | `RISC0_KECCAK_RIGHTSIZE_PO2`: `0` off, other values on | CPU tests: `syscall::keccak::tests::{rightsize_default_is_metal_only, default_request_po2_is_upstream_off_metal}`, `exec::tests::keccak_update2` (po2 **15** = upstream's literal assertion), `api::tests::keccak_po2::keccak_max_po2` (the coprocessor gets po2 15). Metal tests: the same tests see 14 (right-sized) |
| 2a | per-process zkr program cache | `recursion/src/prove/zkr.rs:56` (`ZKR_CACHE_DEFAULT`), `:61` | `get_zkr_shared` = `Arc::new(get_zkr(..)?)`, so a new program is built per call (upstream's `get_zkr(..).into()`), and the cache is never filled | `RISC0_METAL_REC_ZKR_CACHE`: `0`/`off`/`false` off, other values on | CPU tests: `zkr::tests::get_zkr_shared_default_off_metal_is_upstream` (two calls give distinct `Arc`s equal to `get_zkr`, and the cache is empty) and `get_zkr_signature_and_cache_default`. The P31-10 cache tests force it on and pass on both builds |
| 2b | public `get_zkr` signature | `recursion/src/prove/zkr.rs:25` | `pub fn get_zkr(name, po2) -> Result<Program>`, the same body as 5beab26b5 | none | `get_zkr_signature_and_cache_default` binds `let _: fn(&str, usize) -> Result<Program> = get_zkr` (compile-time check on every target). Callers moved to `get_zkr_shared`: zkvm `host/recursion/prove/zkr.rs:35`, `ctrl_persist.rs:500`, the Metal tests and zkvm `test_zkr` |
| 3 | recursion ZK noise on a helper thread | `recursion/src/prove/witgen.rs:326` `NOISE_OVERLAP` (`noise_overlap_from`: unset gives `METAL_BUILD`) | drawn on the proving thread where it is used (upstream's order). Same RNG (`rand::rng()`) and same per-cell draws | `RISC0_RECURSION_NOISE_OVERLAP`: `0` off, other values on | CPU test `witgen::w10_gate_tests::noise_overlap_default_is_metal_only` (also checks the static itself when the knob is unset) |
| 4a | rv32im early data-witness drop | `rv32im/src/prove/witgen/mod.rs:43` (`DROP_DATA_EARLY_DEFAULT`), `:56` `drops_data_after_accum` | only at po2 >= 22 on every HAL (upstream). `CpuHal` (unified memory) no longer drops at every po2 | `RISC0_METAL_DROP_DATA_EARLY`: `0` off, other values on | CPU test `witgen::w10_gate_tests::drop_data_early_default_is_metal_only`: the threshold table for po2 14-24 with the knob off, and the CpuHal case at po2 20 with the default |
| 4b | CPU recursion witgen on a copy of the trace's cycles | `recursion/src/prove/hal/cpu.rs:71` | runs on the caller's trace (upstream's call, unchanged arguments) | none (no knob before either) | CPU suites: zkvm `host::recursion` 12/12 twice, recursion lib 10/10 twice. Metal builds keep the copy, so the Metal suites cover the copy |
| 5 | Intel Groth16 G13 memoized graph build + overlapped digest (`fast.rs` compiles for `intel`) | `groth16/src/prove/fast.rs:133` (`MEMO_DEFAULT = METAL_BUILD`), `:121`; the `RISC0_VERBOSE` eprintln at `:330` is Metal-only too | pre-G13 build: one inverse per Div node, serial translate, digest then parse | `RISC0_G16_GRAPH_MEMO`: `0` off, other values on | `fast::w10_gate_tests::memo_default_is_metal_only`, which runs in the Metal suite (Metal default). Off Metal `fast.rs` only compiles with `intel`, so `cargo check --target x86_64-unknown-linux-gnu --tests --features intel` type-checks it there (below). Not run on Intel hardware |
| 6 | `RISC0_REC_HOST_CHAIN_MIN_LEN` overriding CUDA/ROCm | `recursion/src/prove/witgen.rs:198-214` (`host_chain_min_len_from(METAL_BUILD, ..)`) | the variable is not read off Metal: CUDA/ROCm use the constant 16, as upstream | Metal only | CPU test `witgen::w10_gate_tests::host_chain_min_len_env_is_metal_only` (with the env ignored off Metal) |
| 7 | Metal tests that ran on every build | `build_kernel/tests/f1_metal_cache_key.rs:19` (`target_os = macos/ios`, like its `f1_metal_*` siblings: build_kernel has no `metal` feature); `keccak-sys/tests/k3_eval_translate.rs:25`, `k6_witgen_translate.rs:26`, `recursion-sys/tests/p31_07_step_exec_split.rs:33`, `rv32im-sys/tests/r1_layout.rs:23` (`all(feature = "metal", macos/ios)`); `zkvm/tests/k5_keccak_metal_e2e.rs:27`, `k10_keccak_jobs.rs:36` (`all(prove, metal)`) | not compiled: 0 tests in the CPU binaries (`logs/cpu_gate_tests.log`) | n/a | CPU: each binary runs 0 tests. Metal: k3 4+1i, k6 4+1i, p31_07 2, r1 2, f1 9, k5 1+1i, k10 0+2i, all passing twice (`logs/metal_sys_{1,2}.log`, `logs/metal_zk_it_{1,2}.log`) |
| 7b | zkvm `request_po2` unit test | kept ungated | n/a | n/a | A pure function test of code that ships on every build (an opt-in there). It needs no GPU or toolchain and asserts nothing Metal-specific |

### (8) Re-classification of the rest of `git diff origin/release-3.1...HEAD`

The 95 non-Metal-path files W10-R0 classified were re-read, plus every new `env::var` in a non-Metal
file (the grep in the `Method` section reproduces the list). Nothing else is active off Metal by
default:

- **Env reads.** All the other new knobs are read only under `cfg(feature = "metal")` or behind
  `unified_memory_gpu`/`has_unified_memory() && gpu_total_memory() != MAX`, which is false for
  CUDA/HIP/Intel/CPU:
  - zkp: `RISC0_METAL_DROP_COEFFS_AFTER_MIX`, `RISC0_METAL_SYNC_TRIM` (`cfg!(metal) && ..`);
  - rv32im: `RISC0_METAL_ACCUM_IN_PLACE`;
  - groth16: `RISC0_GROTH16_FAST` (the Metal-only `fast_enabled`);
  - zkvm: `RISC0_KECCAK_*` (`two_phase = cfg!(metal) && ..`), `RISC0_METAL_WRAP_EARLY_RELEASE`.
  - `RISC0_EVAL_U_EARLY` is upstream's ROCm knob, unchanged.
- **Structural refactors that run everywhere, with the same steps:**
  - zkvm `prove_keccak_with` → `prove_seal`/`verify_seal`/`lift_seal`, with the same order under the
    same `exclusive_gpu`;
  - `prove_session_tail` (the segment loop's tail moved into a function);
  - recursion `commit_ctrl` (C13 persistence needs `ctrl_persist::mode() != Off`, which is `Off` unless
    `cfg!(metal)`);
  - `host_chains_backend()` (true on cuda/rocm, false on CPU/Intel, exactly the old cfg).

  The CPU suites below exercise them.
- **Additive public items on every build:**
  - `risc0_circuit_keccak::prove::verify_seal`;
  - `risc0_circuit_recursion::prove::{prewarm_p254_ctrl, P254Prewarm}` (it returns `Skipped` off Metal);
  - the new `zkr::get_zkr_shared`;
  - new `Hal` trait methods with defaults that reproduce the old sequence.

  None changes an existing signature. `release_hal_buffers` is `pub`, but only in a private module; it
  is re-exported only on Metal.
- **Test-only infrastructure active on every `prove` test build:**
  - zkvm `heavy_test_lock` (a file lock that serializes heavy tests);
  - `segment_po2_for_budget` (`None` off Metal);
  - the always-on `p31_14_release_3_1_constants`, `segment_split_po2_14_15_16` and
    `heavy_test_lock_excludes_threads`.

  They check upstream behaviour or serialize tests. No library behaviour depends on them.
- **macOS-only differences, not Linux/Windows:**
  - `risc0_zkp::hal::metal` (and the zkp metallib build and the `metal` link) is now behind
    `feature = "metal"`. Upstream exposed it on any aarch64 macOS build.
  - zkvm's `metal` feature now refuses non-macOS targets (`compile_error!`). Upstream's was a deprecated
    no-op alias of `prove` on any target. zk-miner main enables no `metal` feature in any Cargo.toml
    (`git grep metal origin/main -- '*Cargo.toml'`: empty).
- **Minor text differences off Metal** (logs and errors, no behaviour):
  - groth16 `build_program`'s `tracing::debug!` line has more fields;
  - a missing graph file now says "failed to open the witness graph" (upstream: "failed to read");
  - the program slot is a `Mutex<Option<Arc<_>>>` that remembers a first failure, like upstream's
    `OnceLock`.

## Correctness and equality evidence

### CPU-only (no `metal` feature) on this Mac (`CARGO_TARGET_DIR=scratch/.../target`, dev profile)

| suite | result (run 1 / run 2) |
|---|---|
| `risc0-circuit-recursion --lib` | 10 passed, 1 ignored / same (`logs/cpu_suites_a.log`) |
| `risc0-circuit-rv32im --lib` | 90 / 90 passed |
| `risc0-groth16 --lib`, `tests/verify.rs` | 2 + 1 / same |
| `risc0-circuit-{rv32im,recursion,keccak}-sys`, `risc0-build-kernel` lib + gated tests | 0 tests each (the gated Metal files compile to nothing) |
| zkvm `--lib`, all but `host::server::prove`/`host::recursion` (default parallel harness) | 157 passed, 3 ignored / same (`logs/cpu_zkvm_misc_r{1,2}.log`). Includes `keccak_update2` and `keccak_max_po2` at the upstream po2 15 |
| zkvm `--lib host::recursion` (CPU recursion proofs) | 12 passed, 3 ignored / same (`logs/cpu_zkvm_rec_r{1,2}.log`) |
| zkvm `--lib host::server::prove::tests::riscv` | 46 / 46 |
| zkvm `--lib host::server::prove` (rest) | 39 passed, 1 ignored (`cpu_zkvm_prove_r1.log`) / run 2 in two shards: 19 + 20 passed, 1 ignored (`cpu_zkvm_prove_r2{a,b}.log`) |
| zkvm `tests/keccak.rs` (upstream's keccak e2e, po2 16 on CPU) | 3 passed, 1 ignored |
| `cargo clippy --lib` (zkvm prove, recursion, rv32im, groth16), CPU and metal, `-D warnings` | clean, apart from two lints that already fail on 28b4bef33 in untouched lines, allowed for the run: `clippy::useless_vec` at `recursion/.../witgen.rs:671,707` (upstream code, new in clippy 1.97) and `clippy::type_complexity` at zkvm `recursion/prove/mod.rs:656` |
| `rustfmt --check` on the changed files | clean |

The zkvm `examples/zeroize_overrun.rs` does not build without `cuda`/`rocm`. This is upstream's
example and is unchanged, so the CPU builds use `--lib --tests`.

### Cross-target type checks (`x86_64-unknown-linux-gnu`, W10-R0's stand-in tools)

`scratch/w10/W10-R0/{fakecc,fakecuda,fakerocm,fakeintel}`, plus `CUDA_LIBRARY_PATH=fakecuda` for
`cust_raw`. Build scripts run, but no C/C++/CUDA/HIP/SYCL source is compiled and nothing links.

| `cargo check --target x86_64-unknown-linux-gnu --tests ...` | result |
|---|---|
| `-p risc0-zkvm --features prove` + recursion, rv32im, groth16, build-kernel, the 3 circuit sys crates | ok (`logs/xcheck_linux_cpu_tests.log`) |
| `-p risc0-zkvm --features cuda` + recursion, rv32im, groth16 | ok (`logs/xcheck_linux_cuda.log`) |
| same, `--features rocm` | ok (`logs/xcheck_linux_rocm.log`) |
| same, `--features intel` (compiles `fast.rs` with `MEMO_DEFAULT = false`) | ok (`logs/xcheck_linux_intel.log`) |
| Windows | not checked: the target is not installed. This item adds no platform-specific code |

### Metal (unchanged path)

| check | result |
|---|---|
| metallibs (this branch's `--features metal` builds, `logs/metallib_sha.log`) | zkp `209f0c11…`, rv32im `416179b8…`, recursion `de21a58b…`, keccak `7311962f…`, groth16 `07eafee3…`, the same sha256 and sizes as W10-R0's table. No kernel, `.metal`, generator or build.rs changed, so R17 `ir_scan.py`/`run_stress.sh` do not apply |
| `risc0-circuit-recursion --features metal --lib` (heavy lock) | 99 passed, 23 ignored / same (`logs/metal_rec_{1,2}.log`) |
| `risc0-circuit-rv32im --features metal --lib` | 152 passed, 18 ignored / same |
| `risc0-groth16 --features metal --lib` | 30 passed, 5 ignored / same |
| keccak lib, keccak-sys / recursion-sys / rv32im-sys tests incl. the gated files, groth16 `metal_srs`, build-kernel `f1_metal_cache_key` | all pass, twice (`logs/metal_sys_{1,2}.log`) |
| zkvm `--release --features metal --lib`, 4 shards (`--huge`) | misc 158/3i, `host::recursion` 13/3i, `prove::tests::riscv` 46, `host::server::prove` rest 64/7i; each twice (`logs/metal_zk_*`). The Metal suite asserts the Metal defaults: `keccak_update2` and the coprocessor see po2 14, and the gate tests see `true` |
| zkvm `tests/{keccak,k5_keccak_metal_e2e,k10_keccak_jobs}` (release, metal) | 3/1i, 1/1i, 0/2i, twice |
| stock crates.io 3.1.0 verifier, `run_matrix.sh` (`VERSIONS=31`, default + 8 + 4 GiB budgets) | **30/30 STOCK 3.1 VERIFY: OK**: fibonacci, merkle, compose and keccak unbudgeted; merkle, keccak and fibonacci-after-worker-warmup at 8 and 4 GiB; composite, succinct and Groth16 (`sv31/logs/matrix.txt`). 4 GiB wrap peaks 3944 / 3928 / 3944 MiB, the same as B9-4G. 3.0 receipts: **60/60 REJECTED** (30 plain + 30 relabelled). These are the 3.0 producer's receipts from B9-M26's matrix, copied: the 3.0 producer and the frozen tree are untouched and the verifier is this matrix's |

### Interleaved worker A/B (Metal-reachable Rust changed: the keccak syscall, the `get_zkr_shared` call path)

- **Arms:**
  - A1 and A2 = `bin/W_A`: zk-miner-3.1 6e7a1c1 (RC2) on risc0 28b4bef33, from a detached scratch
    worktree;
  - B = `bin/W_B`: the same zk-miner on this branch.

  Both were built the same way (`make -f Makefile.mac RISC0_SRC=local RISC0_DIR=<tree>
  mac-build-worker`, one target dir) in a throw-away zk-miner worktree, now removed.
- **Runs:** `BENCH_ONLY=memory-merkle --benchmark`, `/usr/bin/time -l`, 8 rounds per budget. Each round
  runs A1/B/A2 in a rotated order under one `gpu-run` lock (`scripts/round.sh`, `ab.sh`, `ana.py` from
  B9-G1). B is compared with its round's mean of A1 and A2. Each cell gives the median [min, max].

| default budget (po2 20) | A1 median | B - A | A2 - A1 |
|---|---|---|---|
| STARK | 8.435 s | +0.005 [-0.060, +0.035] s, 4/8 > 0 | -0.010 [-0.120, +0.180] |
| segments / recursion | 6.940 / 1.500 s | +0.005 / +0.002 | -0.015 / 0.000 |
| Groth16 wrap | 1.610 s | +0.003 [-0.020, +0.015] | -0.005 [-0.040, +0.010] |
| peak footprint (`time -l`) | 12.296 GiB | +0.000 [-0.002, +0.021] | -0.001 |
| Metal peak (segments / recursion / wrap) | 9437 / 9437 / 5635 MiB | identical in every run | |

Load 2.8-6.9 (other agents' work). B: 8.440 s STARK, 1.610 s wrap.

| 4 GiB (`RISC0_METAL_MAX_WORKING_SET_BYTES=4294967296`, po2 18) | A1 median | B - A | A2 - A1 |
|---|---|---|---|
| STARK | 14.065 s | -0.015 [-0.080, +0.035] s, 2/8 > 0 | +0.010 [-0.120, +0.070] |
| segments / recursion | 7.860 / 6.210 s | -0.010 / -0.020 | +0.020 / -0.010 |
| Groth16 wrap | 1.740 s | 0.000 [0.000, +0.005] | 0.000 [-0.010, 0.000] |
| peak footprint | 4.605 GiB | -0.007 [-0.029, +0.046] | -0.006 |
| Metal peak | 2404 / 2199 / 3944 MiB | identical in every run | |

Load 1.4-5.8. Every run of every arm proved 7,340,032 cycles and verified its receipt. **B is "no change"
at both budgets.** The differences are inside the A-vs-A spread. The default-budget STARK of 8.44 s is in
line with RC2's memory-merkle results.

## Knobs

No new knob; the existing ones changed default off Metal only:

| knob | Metal default | elsewhere |
|---|---|---|
| `RISC0_KECCAK_RIGHTSIZE_PO2` | on | off (opt-in: any value but `0`) |
| `RISC0_METAL_REC_ZKR_CACHE` | on | off (opt-in: any value but `0`/`off`/`false`) |
| `RISC0_RECURSION_NOISE_OVERLAP` | on | off (opt-in: any value but `0`) |
| `RISC0_METAL_DROP_DATA_EARLY` | on | off, po2 >= 22 only (opt-in: any value but `0`) |
| `RISC0_G16_GRAPH_MEMO` | on | off (Intel; opt-in: any value but `0`) |
| `RISC0_REC_HOST_CHAIN_MIN_LEN` | read (default 16) | not read (16) |

`RISC0_G16_GRAPH_REVERIFY` is unchanged. Its file-identity skip only acts after a release, and only Metal
releases.

## Method (to reproduce the classification)

- `git diff origin/release-3.1...HEAD --name-only` without the Metal-only paths (W10-R0's list), then
  every hunk of the rest.
- `git diff -U0 origin/release-3.1...HEAD -- <non-metal .rs> | grep '^+' | grep -o 'env::var[_os]*("…")'`
  lists every new env read. Each one was traced to its gate.
- New test files were found with `git diff --diff-filter=A` and their `#![cfg]` lines were checked.

## Open issues

- Not checkable on this Mac: Windows targets, real CUDA/HIP/SYCL compiles and links, and runs on
  non-Metal GPUs. Intel's `fast.rs` default (`MEMO_DEFAULT = false`) is type-checked for Linux but not
  run. A Linux CI run of the fork is still needed before zk-miner main relies on it there (W10-R0's open
  issue stands).
- macOS non-Metal builds differ from upstream in two places (see (8)): `risc0_zkp::hal::metal` is no
  longer exposed without the `metal` feature, and zkvm's `metal` feature is macOS-only. Neither affects
  Linux/Windows or zk-miner main.
- zk-miner is re-pinned by W10-PIN. After the merge, run the mirror fetch from W10-R0's note and move the
  8 `[patch]` revs to the new `zkminer/metal-3.1` head.
