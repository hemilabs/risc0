## Completeness critique of `docs/metal-port/PLAN.md`

All facts below were read from code under `/Users/user/zkminer/deps/risc0/risc0/` and `/Users/user/zkminer/zk-miner/` or compiled with `xcrun metal` (Apple metal 32023.922, macOS 27.2). Nothing was edited.

### A. Gaps and corrections, ordered by severity

**1. (P0, section 2.4 + 2.3 + 2.5) The kernel-signature template is rejected by the MSL compiler.**
`[[max_total_threads_per_threadgroup(N)]]` placed after the parameter list (as 2.4 shows) fails: `error: 'max_total_threads_per_threadgroup' attribute cannot be applied to types`. It compiles only before `kernel void`. Every translator (R2, C2, K3) and hand port would emit the broken form.
Proposed text (2.4 code block):
```metal
[[max_total_threads_per_threadgroup(256)]]            // replaces __launch_bounds__; MUST precede `kernel void`
kernel void foo(device Fp* out [[buffer(0)]], ...)
```
and in 2.3 "Not aliasable": "`__launch_bounds__(N)` → `[[max_total_threads_per_threadgroup(N)]]` emitted **before** the `kernel` keyword (verified: the post-signature position is rejected)."

**2. (P0, sections 2.0 / 2.8 / F2 / R7 / K1 / G1) The feature wiring is not complete end to end; several crates have no `metal` feature and no metal-rs dependency at all.**
Verified today: `risc0-circuit-rv32im`, `risc0-circuit-keccak`, `risc0-circuit-keccak-sys` and `risc0-groth16` have **no** `metal` feature (`rv32im/Cargo.toml` has only cuda/rocm/intel/prove/witgen_debug; `keccak/Cargo.toml:46-69` cuda/rocm/intel/prove; `keccak-sys/Cargo.toml:25-27` cuda/rocm/intel; `groth16/Cargo.toml:48-90` cuda/rocm/intel/prove/std). The `metal = "0.29"` crate is a dependency only of `risc0-zkp` and `risc0-circuit-recursion` (target-gated, non-optional); `rv32im`, `keccak`, `groth16`, `groth16-sys` cannot `use metal::` today. `groth16-sys` pulls `risc0-sys` only through `dep:risc0-sys` on cuda/rocm. zk-miner's `zkminer-prove-risc0/Cargo.toml` depends directly on `risc0-circuit-keccak` (line 52) in addition to `risc0-zkvm`.
Proposed text for F2 (replace "`metal` features in zkvm, zkp, every circuit, groth16, every `-sys`, `risc0-sys`"):
"Add, verbatim:
- `zkvm/Cargo.toml`: `metal = ["prove", "risc0-circuit-keccak/metal", "risc0-circuit-recursion/metal", "risc0-circuit-rv32im/metal", "risc0-groth16/metal", "risc0-zkp/metal"]`
- `zkp`: `metal = ["prove", "dep:metal", "risc0-sys/metal"]` and make the macOS/iOS `metal` dependency `optional = true`.
- `circuit/rv32im`: new `[target.'cfg(any(target_os="macos", target_os="ios"))'.dependencies] metal = { workspace = true, optional = true }` and `metal = ["prove", "dep:metal", "risc0-circuit-rv32im-sys/metal", "risc0-zkp/metal"]`; same shape for `circuit/keccak` (`risc0-circuit-keccak-sys/metal`) and `groth16` (`dep:risc0-groth16-sys`, `dep:rayon`, `dep:circom-witnesscalc`, `dep:wtns-file`, `prove`, `risc0-groth16-sys/metal`, `risc0-zkp/metal` — mirror the `intel` list).
- `circuit/recursion`: `metal = ["prove", "dep:metal", "risc0-circuit-recursion-sys/metal", "risc0-zkp/metal", "std"]`.
- `-sys` crates: `rv32im-sys`/`recursion-sys`: `metal = ["risc0-sys/metal"]`; `keccak-sys`: add `metal = ["risc0-sys/metal"]`; `groth16-sys`: `metal = ["dep:risc0-sys", "risc0-sys/metal", "dep:metal"]` plus an optional `metal` crate dependency (its `src/metal/*.rs` is Rust over metal-rs).
- zk-miner `zkminer-prove-risc0/Cargo.toml`: `metal = ["risc0-zkvm/metal", "risc0-groth16/metal", "risc0-circuit-keccak/metal"]` and `[[bin]] name = "zkminer-prove-risc0-metal", required-features = ["metal"]` (M1)."

**3. (P0, section 4.4 K1/K3/K6 and 4.3 C2/C3 and 4.2 R13–R15: parallel items share files.)**
- Wave 1 runs K1 and K3 in parallel; both own `keccak-sys/build.rs` (K1: `build_metal_kernels()` body; K3: `write_metal_eval_check` in the same file). Wave 2 runs K2 and K6 in parallel; both own `keccak/src/prove/hal/metal/witgen.rs` and K6 also edits `keccak-sys/build.rs`.
- Wave 2 runs C2 and C3 in parallel; both own `recursion-sys/kernels/cuda/cuda2metal.py` ("same script (eval section)").
- Wave 5 runs R13 (`misc.metal`, `hal/metal/accum.rs`), R14 (`hal/metal/{mod,eval_check}.rs`) and R15 ("per sub-track": always_inline vs noinline accum, EF_* knobs in eval_check, single-buffer witgen) in parallel; R15 necessarily touches the same accum/eval_check/witgen files.
Proposed text: "K3's translator lives in `keccak-sys/build_metal_eval.rs` (`#[path]` module of build.rs); K1 adds the one `mod` line. K6 starts after K2 merges (move K6 to wave 3). C2 owns `cuda2metal_common.py` + `cuda2metal_steps.py`; C3 owns `cuda2metal_eval.py` and depends on C2. R15 sub-tracks start after R13 and R14 merge, or each sub-track is restricted to a file none of R13/R14 lists."

**4. (P1, section 4.3 Kernels + C2) Recursion translation input list omits `step_verify_mem.cu`.**
`step_verify_mem` is a `__device__` function in its own TU, called from the VerifyWom kernels at `ffi.cu:759,772,784`; `step_verify_bytes.cu` and `sppark.cu` contain no `__global__`. The plan's "compile the four files" in C2 never names them.
Proposed text (4.3 table, new row): "`step_verify_mem` (step_verify_mem.cu, called by parStepVerifyWom/fwdStepVerifyWom) | translate in steps mode alongside step_exec; `step_verify_bytes.cu` is empty and skipped." And in C2: "translated TUs: `step_exec.cu`, `step_compute_accum.cu`, `step_verify_accum.cu`, `step_verify_mem.cu`."

**5. (P1, sections 4.2/4.3/4.4) Trait methods with no Metal story.**
- `CircuitHal::accumulate` (zkp `hal/mod.rs:410`) is implemented by every circuit's CUDA HAL (`rv32im cuda.rs:196`, `recursion cuda.rs`, `keccak cuda.rs:153`) but is never mentioned for rv32im, recursion or keccak. Add to R9, C7, K2: "implement `CircuitHal::accumulate` exactly as `cuda.rs` does (delegating to `step_accum`/`accumulate` or the CPU path)".
- recursion `CircuitWitnessGenerator::upload_host_witness` (`hal/mod.rs:60`, default `bail!`) — not mentioned. Add to C6: "`upload_host_witness` = memcpy into the shared `data`/`global` buffers (needed by any pre-witgen/shadow check path that calls it; callers unverified)". Also `eval_check_dep` for recursion and keccak (no-op) should be listed as rv32im's is.
- keccak `CircuitWitnessGenerator` has an **associated type** `PreferredPreflightOrder` (`hal/mod.rs:60`); CUDA declares its own `CudaPreflightOrder` with a custom `key_for_cycle` (`cuda.rs:73-76`). Add to K2: "phase 1 declares `type PreferredPreflightOrder = ForwardPreflightOrder`"; to K6: "phase 2 declares a `MetalPreflightOrder` keyed by (cycle_type, sub_type) for per-bucket dispatch". Also `impl Drop for CudaCircuitHal` (`cuda.rs:62`) — state whether the Metal HAL needs a Drop.
- zkp `Hal::trim_device_memory`, `prefill_elem`, `batch_get_digest_at`, `batch_interpolate_ntt_zk_shift_from` are not named in 4.1. Add one line to F0/Z1/Z2: "`trim_device_memory` drains the pool; `prefill_elem` stays the default no-op; `batch_get_digest_at` stays the default (`get_at`); `batch_interpolate_ntt_zk_shift_from` is part of Z2's NTT entry points."

**6. (P1, section 4.1 kernel table and 4.5) Kernel inventory rows missing, so the "every `__global__` maps" rule cannot be audited.**
zkp (all verified in `sys/kernels/zkp/cuda`): `k_coset_tabs`, `k_line_tabs`, `k_pow_table`, `pass_h`, `pass_k` (nttq only, `supra/ntt_q.cuh`); `bit_rev_t2_batched` (HIP-only branch `ntt_x.cuh:650`), `bit_rev_t2_oop` (nttq expand path `ntt.cu:462`); `bit_rev_permutation` (sppark, the n < Z·Z branch `ntt_x.cuh:647`); `fold_tail_q4_k`; `combos_prepare` (`combos.cu`); host externs `sppark_init`, `sppark_poseidon2_init`, `risc0_zkp_cuda_warmup`, `risc0_zkp_cuda_set_oom_hook`, `risc0_zkp_cuda_sync_stream`.
Proposed rows: "`k_coset_tabs/k_line_tabs/k_pow_table/pass_h/pass_k` — nttq only, not ported | — ; `bit_rev_t2_batched/bit_rev_t2_oop` — HIP/nttq only, not ported ; `bit_rev_permutation` — replaced by the existing naive Metal bit-reverse for n < 4096 ; `fold_tail_q4_k` — with `fold_q4_k`, not ported ; `sppark_init`/`sppark_poseidon2_init` — CPU-built `NttTables` (Z2) and `constant` round-constant tables (Z3) ; `risc0_zkp_cuda_warmup` → `MetalHal::new`; `set_oom_hook` → 2.9 pool drain-and-retry; `sync_stream` → `sync()`."
groth16 (`groth16-sys/kernels/cuda`): `batch_diff`, `split_bit_sum`, `integrate_combine`, `integrate_split` are absent from the 4.5 table; host entry points `risc0_groth16_cuda_host_buffer`, `risc0_groth16_cuda_set_bit_signals`, `risc0_groth16_cuda_prestart`, `risc0_groth16_cuda_stager_init`, `risc0_groth16_cuda_pinned_release` (all declared in `groth16-sys/src/lib.rs:291-335`) are not mapped. Proposed: add rows "`batch_diff` — not ported (CUDA batch-addition path)"; "`split_bit_sum` — phase 3 with split_*"; "`integrate_combine/integrate_split` — folded into `msm_window_reduce`"; and a host-entry table: "`cuda_prove/preload/release` → `metal_prove/preload/release`; `prestart`/`stager_init`/`pinned_release`/`host_buffer`/`set_bit_signals` → no-ops on unified memory (the `witness_par` staging they serve is cuda-only, `prove/mod.rs:33`)".
recursion: host externs `risc0_circuit_recursion_cuda_eval_check`, `_warmup_eval_check`, `rec_eval_check_split_warmup` (`ffi.cu:128,1656`), `_host_alloc/_host_free` (pinned, `recursion-sys/src/lib.rs:183-188`) — add: "eval_check/warmup → C8/C5 `metal_warmup`; pinned alloc/free → none (the `pinned` module is `cfg(any(cuda, rocm))` at `prove/mod.rs:21`, and `witgen.rs:159` already has the non-pinned `pinned_ptr` stub the Metal build will use)".

**7. (P1, section 4.5 G1 and 6.2) groth16 cfg widening must be enumerated; the `--test verify` acceptance does not do what the plan says.**
`groth16/src/prove/mod.rs` has nine cfg_if/cfg sites that would silently stay Docker: `mod component` (`any(rocm, intel)`), `mod docker` (`not(any(cuda, rocm))`), `mod fast`/`mod intel`, `preload_graph` (45), `preload_graph_background` (61, cuda only), `prestart_prover` (73, intel only), `prestart_key`/`prestart_key_go` (97/103, cuda), `preload_srs` (110), `release_srs` (138), `invalidate_caches` (161), `shrink_wrap_with` (170, cuda only), `shrink_wrap` (182), and `docker.rs:41,88 fast_enabled` (intel). `groth16/tests/verify.rs` verifies one fixed JSON triple from `tests/data` (`test_verify`, lines 19-24); it does not iterate proofs.
Proposed text (G1): list the sites above and "each gains a `metal` arm mirroring `intel`". Proposed text (4.5 Acceptance): replace "`cargo test -p risc0-groth16 --test verify` over every Metal proof" with "new `src/metal/tests.rs::metal_proofs_verify`: for each Metal seal, build `proof.json`/`public.json` and call `risc0_groth16::Verifier` as `tests/verify.rs::test_verify` does".

**8. (P1, section 6.2 / M3) zkvm GPU-gated tests to widen are incomplete.**
`zkvm/src/host/server/prove/tests.rs:568,600,621,642` gate `compress`, `fake_compress`, `shrink_wrap`, `verify_in_guest` on `any(cuda, rocm[, docker])`, and `zkvm/src/host/recursion/tests.rs:339` gates `test_recursion_host_chains_check` on `any(cuda, rocm)`; the recursion acceptance ("`set_host_chains_check` hooks pass") depends on the latter. Proposed text: "M3/C9: add `feature = "metal"` to the five cfg lists above."

**9. (P2, section 2.10 / 6.3) Several acceptance commands are not runnable as written.**
- `cargo test -p zkminer-prover discovery:: dispatcher::` — cargo accepts one positional filter; write `cargo test -p zkminer-prover -- discovery:: dispatcher::`.
- "`cargo test -p zkminer-prove-risc0 --features metal`, `--features testing --test worker_lifecycle --test measure_only`" — the second command belongs to `zkminer-prover` (verified: `zk-miner/Makefile:80`, tests live in `crates/zkminer-prover/tests/`; `zkminer-prove-risc0` has no `tests/` dir, only 5 `#[cfg(test)]` blocks in `main.rs`). Write `cargo test -p zkminer-prover --features testing --test worker_lifecycle --test measure_only`.
- `cargo test -p risc0-build-kernel metal_probes` — the crate has no tests today; D0 must add `build_kernel/tests/metal_probes.rs` (a new file, so it does not collide with F1's ownership of `build_kernel/src/lib.rs`). State that.
- `cargo check --target x86_64-unknown-linux-gnu ...` needs that rustup target installed on the Mac (unverified whether it is).
- 6.3: the worker's `--benchmark` flag is **verified** (`zkminer-prove-risc0/src/main.rs:68`); remove the "unverified" caveat and the matching line in section 7.
- `cargo test -p risc0-circuit-rv32im ... prove::` "46 riscv cases": `prove/tests.rs` has 48 `#[test]`/`test_case!` lines including `basic` and `multi_read` — consistent; `fwd_rev_ab` tests exist as `witgen::tests::{fwd_rev_ab_basic, fwd_rev_ab_split}`.

**10. (P2, section 3 F2 vs G1 ownership ambiguity.)** F2 owns "`prove/mod.rs` of each circuit" and G1 owns `groth16/src/prove/{mod,metal}.rs` + `groth16/Cargo.toml`; F2 also says "`metal` features in ... groth16". State explicitly: "F2 does not touch `groth16/` or `groth16-sys/`; G1 owns all groth16 feature lines and cfg arms (G1 therefore moves to wave 0 or the zkvm `metal` feature's `risc0-groth16/metal` entry is added by G1, not F2)."

**11. (P2, section 2.3) `__ballot_sync` alias needs a cast.** `simd_ballot` returns `simd_vote`, not an integer (probe compiled only with `(uint64_t)b`). Proposed: "`__ballot_sync(m, p)` → `(uint)(simd_vote::vote_t)simd_ballot(p)`".

**12. (P2, section F2 / R3 / C4 / K1 / G1) Who removes the placeholder `kernels/metal/noop.metal`?** F2 creates it in four `-sys` crates; R1/C1/K1/G2 own those directories later. Add "the unit's first kernel item (R3, C4, K1, G1) deletes `noop.metal` and removes it from `METAL_KERNEL_NAMES`."

**13. (P2, section F1) `compile_metal` writes `.air` next to absolute sources.** `build_kernel/src/lib.rs:327`: `out_dir.join(src).with_extension("air")` — for an absolute `$OUT_DIR/metal/x.metal` (the plan's generated-output location) `Path::join` discards `out_dir`, so the `.air` lands beside the generated source. Harmless, but F1 should "strip the source's prefix before joining so all `.air` land under `<OUT_DIR>/<output>/`".

**14. (P2, section 1 status / 2.0) Minor factual tightening.** `KernelBuild::compile_metal` already passes `-Wno-unused-variable` and `-I <sys_inc_dir>` (where `METAL_INCS` = `fp.h`, `fpext.h`, `lib.rs:25-28`) — the 2.0 "Shared MSL headers" row should say `cuda2metal.h`/`sppark_compat.h` are appended to that `METAL_INCS` const. `DEP_RISC0_SYS_METAL_ROOT` is emitted (`sys/build.rs:59`) and has no consumer anywhere — consistent with the plan.

### B. Probe results (xcrun metal 32023.922, `-std=metal3.2`, `-I build_kernel/kernels/metal`; sources in `/private/tmp/claude-501/-Users-user-zkminer/a30bccea-eec8-43d4-ac51-c5ccb6e4b957/scratchpad/probes/`)

| Probe | Result |
|---|---|
| `mulhi(ulong, ulong)` and `ulong*ulong` | **ACCEPT**; lowers to `@air.mul_hi.u.i64` (32-bit to `@air.mul_hi.u.i32`). Backend cost on Apple GPUs not measurable here (unverified). |
| `threadgroup uint buf[9216]` (36 KB static) | **ACCEPT at compile and metallib link, no diagnostic.** The compiler does not enforce the 32 KB device limit; failure would be at PSO/dispatch (unverified at runtime). Confirms the plan's host-side assert is mandatory; D0 should add the runtime probe. 32 KB static also compiles. |
| `simd_shuffle_down/up/xor/simd_shuffle` on `uint`, `simd_prefix_exclusive_sum(uint)`, `simd_ballot` | **ACCEPT** (ballot needs the `simd_vote` → integer cast, see gap 11). |
| `template [[host_name("ct_x_8")]] kernel void ct_x<8>(...)` explicit instantiation | **ACCEPT**, and `metallib` links it with another `.air`. Runtime `get_function("ct_x_8")` not verified. |
| `[[max_total_threads_per_threadgroup(768)]]` (non-power-of-two) | **ACCEPT** when placed before `kernel void` (IR carries `"max-work-group-size"="768"`); **REJECT** in the position shown in plan section 2.4. |
| `assert()` | `<metal_assert>` defines `assert(condition) ((void)0)` on both branches (`metal_assert:18-21`), i.e. a no-op even without `NDEBUG`; no `trap` in IR at -O2 or -O0. Whether `<metal_stdlib>` alone pulls in `<metal_assert>` was not checked. |
| `atomic_ulong` `atomic_fetch_add_explicit` | **REJECT** (`_valid_fetch_add_sub_type` unsatisfied) — matches the plan. |
| `threadgroup Fp x[8]` with current `fp.h` | **REJECT** ("cannot be used to construct an object in address space 'threadgroup'") — confirms F1. |
| `-std=metal4.0` | **ACCEPT** on this toolchain (plan section 7 lists it as unverified). |

### C. Verified-correct claims worth keeping as "verified"
`RawPreflightCycle`/`RawMemoryTransaction` structs exist in `rv32im-sys/src/lib.rs:25,38` (sizes not re-measured); CUDA `prefix_products` is a CPU loop (`zkp/src/hal/cuda.rs:1814`); rv32im/recursion CUDA HALs use `Arc`, keccak uses `Rc` (`cuda.rs:51/53`); `MetalHal::new` stores `_lock` from `singleton().lock()`; `zkp/src/hal/mod.rs:25` gates `pub mod metal` on target; `recursion/src/prove/hal/metal.rs:32` reads `RECURSION_METAL_PATH`; stale recursion `.metal` files are 43/15625/8300 lines; `python_command()` exists (`build_kernel/src/lib.rs:1414`) and is used at `rv32im-sys/build.rs:1610..2066`, `recursion-sys/build.rs:769,842`; `write_intel_translation_unit` at `keccak-sys/build.rs:475`; `metal_implies_prove` at `zkvm/src/lib.rs:233`; zk-miner `GPU_TAGS` at `discovery.rs:28`, worker naming `zkminer-prove-{backend}-{gpu_tag}` at `discovery.rs:67-73`; `release_groth16_cache_if_tight` reads `ZKMINER_CUDA_VRAM_BYTES` (`main.rs:1176`).