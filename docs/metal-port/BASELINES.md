# Metal port baselines (PLAN M5 row, section 6.4)

Quiet-machine baseline after the wave-5 merges (Z9 Z10 C11 C12 R13 R14 R15e R15w G9 G10 K7).

| | |
|---|---|
| Date | 2026-11-12, 06:17-07:45 local time |
| Machine | Apple M5 Max, 40-core GPU, 36 GB unified memory (28.1 GiB recommended working set), macOS 27.2.1 (26B131) |
| risc0 | `zkminer/metal` @ `eed5019552e99adb36a44bdb95d250a7bf4c6375` (merge of metal/G10) |
| zk-miner | `metal` @ `bf4ec2eab8d3b014d86343d71a0936f1c277118f` (clean tree; path-patched to the risc0 checkout above) |
| Worker | `cargo build --release -p zkminer-prove-risc0 --features metal --bin zkminer-prove-risc0-metal`, installed to `~/.zkminer/provers/zkminer-prove-risc0-metal` (sha256 `b6d3422e…dcdad`) |
| Benchmark command | `target/release/zkminer-prove-risc0-metal --benchmark` (full suite; `BENCH_ONLY=<name>` for one program) under `/usr/bin/time -l` |
| Machine state | No other cargo, test or prover process ran (checked with `ps` before each step). The load averages are given per table. The only load spikes, to 5-6, came right after this run's own `RISC0_METAL_REC_P254=cpu` arm, which uses every CPU core. |
| Logs | `/private/tmp/claude-501/-Users-user-zkminer/a30bccea-eec8-43d4-ac51-c5ccb6e4b957/scratchpad/w5/baseline/` |

The pre-wave reference is the same command on 2026-11-11 (`scratchpad/bench/metal_full.log`; the machine was
not quiet). The RTX 4090 reference is the test table in `zk-miner/crates/zkminer-prover/src/benchmark.rs`.

## 1. Worker `--benchmark` (full suite, po2 20 segments)

Two full runs back to back (`full1.log`, `full2.log`). The load average was 2.97 falling to 0.66 for run 1,
and 0.61 to 0.85 for run 2. Both runs proved, wrapped and verified every program.

| program | cycles | pre-wave STARK (s) | run 1 (s) | run 2 (s) | change | RTX 4090 (s) | Metal / 4090 |
|---|---|---|---|---|---|---|---|
| fibonacci | 32,768 | 0.50 | 0.40 | 0.40 | -20 % | 0.48 | 0.83x |
| sha256-chain | 46,137,344 | 133.16 | 100.75 | 100.57 | -24.4 % | 35.82 | 2.81x |
| ecdsa-verify | 65,011,712 | 186.33 | 141.24 | 141.94 | -24.0 % | 47.65 | 2.97x |
| bigint-mul | 27,525,120 | 80.29 | 60.49 | 60.56 | -24.6 % | 19.77 | 3.06x |
| memory-merkle | 7,340,032 | 21.31 | 16.18 | 16.25 | -23.9 % | 5.70 | 2.85x |
| chacha-mix | 8,404,992 | 24.88 | 18.81 | 18.82 | -24.4 % | 6.67 | 2.82x |
| **all (sum of cycles / sum of STARK time)** | 154,451,968 | 446.47 s, **346K c/s** | 337.87 s, **457K c/s** | 338.54 s, **456K c/s** | **+32 % c/s** | 116.09 s, 1.33M c/s | 2.91x |

- The RTX 5090 does about 2.3M c/s overall, 5.0x this Mac.
- Before the wave the Mac was 3.85x slower than the 4090. It is now 2.91x slower.
- Per-program segment / recursion split (run 1): sha256-chain 89.34 + 11.41 s (44 segments), ecdsa-verify
  125.68 + 15.56 s (62), bigint-mul 53.47 + 7.02 s (27), memory-merkle 14.41 + 1.77 s (7), chacha-mix
  16.50 + 2.32 s (9).
- Before the wave, memory-merkle was 18.37 + 2.95 s: segments are 22 % faster and recursion 40 % faster.

**Groth16 wrap.** Before the wave a wrap took 16.1 s: identity_p254 13.4 s on the CPU plus shrink_wrap 2.7 s.
After the wave:

| wrap | identity_p254 | shrink_wrap | total |
|---|---|---|---|
| every wrap after the first in a process (10 samples) | 1.11-1.14 s | 1.65-1.69 s | **2.78-2.83 s** |
| first wrap of a process | 1.11 s | 5.2-5.3 s | 6.4 s |

The first shrink_wrap waits ~3.6 s for the witness-graph preload and key load (section 3, Z10). The RTX 4090
wraps in ~2.0 s.

**Memory.**

| | pre-wave | after the wave |
|---|---|---|
| process peak footprint, full suite (`time -l`) | 20.75 GB | 20.02 / 20.01 GB |
| `--benchmark` process lifetime peak | 19793 MiB | 19079 MiB (run 2) |
| CPU user time of the full suite | 1432 s | 29.8 s |

The CPU user time fell because identity_p254 left the CPU (Z10).

Peak Metal allocation per phase at po2 20, in MiB:

| program | segments | recursion | wrap |
|---|---|---|---|
| memory-merkle | 9402 | 11098 | 11098 |
| chacha-mix (pre-wave) | 9503 | 11402 | - |
| chacha-mix (now) | 9492 | 11391 | - |

**Cold start.** With `MTL_SHADER_CACHE_SIZE=0`, the worker builds its pipelines in 24.9-26.2 s. The groth16
PSOs bound that time, at ~25 s. rv32im+zkp take 9.8-11.4 s with the R14 archive hit, and recursion takes
9.7-11.3 s, both in parallel. With the OS shader cache warm, startup takes 0.0-0.2 s.

## 2. Per-phase breakdown (quiet machine)

### po2 20 rv32im segment
Command: `risc0-circuit-rv32im` release lib test `prove_po2_sweep` with `R11_PO2S=20 R11_CPU=0 R11_REPS=6`
(simple_loop segment; `phase_rv32im.log`). The load average was 1.5. The marks are walls between waits. Reps 1-5
were within ±5 ms; rep 0 is the cold first call.

| phase | pre-wave (ENV) | now: ms (reps 1-5) | share |
|---|---|---|---|
| preflight + alloc | - | 10-11 | 0.5 % |
| witgen | 60-100 | 8 | 0.4 % |
| data commit | ~300 | 221 | 10.8 % |
| accum | ~150 (accum + accum commit) | 9-10 | 0.5 % |
| accum commit | (see accum) | 114-115 | 5.6 % |
| eval_check (evalfast) | 2100-2300 | **1598-1607** | 78.8 % |
| FRI + tail | - | 66-69 | 3.3 % |
| **segment** | 2600-3000 | **2031-2040** | |

Notes:
- The worker's real segments come to 14.41 s / 7 = 2.06 s each (memory-merkle).
- The witgen above is simple_loop's. R15w's shift_loop witgen call is 18.2 ms (section 3).
- Peak GPU allocation is 9287 MiB.
- Per-kernel GPU time over a whole memory-merkle proof (`RISC0_METAL_PROFILE=1`, serialized, `prof_mm.log`;
  7 segments plus 13 recursion proofs):
  - the circuit kernels (not registered by name, mostly `ef_eval_split`) take 76 %;
  - `pv_rows_k_z9` (Poseidon2 commits) 11.9 %;
  - NTT 7.8 %;
  - `pv_fold_k_z9` 2.1 %;
  - eu2 0.8 %;
  - mdiv 0.4 %.

### Lift/join (recursion, po2 18 Poseidon2)
memory-merkle runs 7 lifts and 6 joins in 1.77-1.83 s wall (pipelined), about 138 ms per proof. Before the
wave this took 2.95 s.

The per-phase profile below comes from `RISC0_METAL_REC_PROFILE=1` (serialized), 13 proofs, ms:

| witgen (fill / host / exec / sort / verify) | pre_accum (data noise + commit + mix) | accum | pre_eval (accum commit) | eval_check (lz split) | remainder (check commit + FRI) |
|---|---|---|---|---|---|
| 9.5-22.8 (0.3 / 1-6 / 2.6-4.6 / 4.2-10 / 0.5-1.9) | 33.5-55 (first proof 55) | 6.2-10.6 | 6.6-9.2 | **47.6-49.2** | ~25-35 |

### identity_p254 + shrink_wrap
- **identity_p254: 1.11-1.15 s.** Profile, ms: witgen 17.9, pre_accum (Poseidon254 commit) 684, accum 5,
  pre_eval 123, eval_check 48. Metal GPU time is 1041 ms, of which `p254_rows_w64` takes 652 ms (63 %) and
  `p254_fold_w64` 307 ms (29 %).
- **shrink_wrap: 1.60-1.69 s with the key cached.** These numbers come from the `risc0-groth16`
  `metal_shrink_wrap` test (`RISC0_VERBOSE=1`, 3 verified wraps per process, 18 processes, `g16ab/`).

  | witness_poly + NTT | sort_h + MSM H | fuzz + sort_w + MSM A, B1, C | MSM B2 | GPU sum | upload / encode / host |
  |---|---|---|---|---|---|
  | 141-178 ms | 230-233 ms | 437 ms | **554-566 ms (40 %)** | **1364-1410 ms** | 2.6-8.5 / 0.1-0.5 / 0.9 ms |

  Device allocation of the cached prover is 3893 MiB. The test process's peak footprint is 6.06 GiB.

## 3. Wave-5 revert knobs: quiet interleaved A/B

Method: one process per run, A = defaults. Each knob arm sets its item's revert knob.

**Worker A/B.** Command: `BENCH_ONLY=memory-merkle --benchmark` (`ab/round.sh`, `ab/summary.txt`, `ab/analyze.py`).
- 8 rounds. Each round runs A1, nine knob arms, A2, with the start position rotated by 3 every round.
- Each arm is compared with the mean of its round's A1 and A2.
- The A2-vs-A1 spread was: segments -0.04 s [-0.18, 0.00] per 7 segments, recursion 0.00 [-0.03, +0.01] s,
  identity_p254 [-5, +4] ms, shrink_wrap [-32, +32] ms.
- Load averages were 0.75-2.9, with spikes to 5-6.4 in the run right after the CPU-p254 arm.

Every arm proved, wrapped and verified (rc 0, 96/96 runs).

| item | knob (off) | phase | knob on (default) median [min-max] | knob off median [min-max] | per-pair delta, off slower in | verdict |
|---|---|---|---|---|---|---|
| Z9 Poseidon2 form | `RISC0_METAL_POSEIDON2=pv` | 7 segments; recursion | 14.553 [14.535-14.615] s; 1.805 s | 14.870 [14.780-14.930] s; 1.890 s | +0.275 s (+39 ms/segment, 1.9 %), 8/8; rec +0.082 s, 8/8 | **holds** |
| Z9 mdiv crossovers | `RISC0_METAL_MDIV_GPU_MIN_PO2=0 RISC0_METAL_MDIV_E16_MIN_PO2=20` | segments | 14.553 s | 14.510 [14.47-14.62] s | -0.037 s, 1/8 | **no change** (inside the A-A spread; at po2 20 the crossovers are not in play) |
| Z10 Poseidon254 on Metal | `RISC0_METAL_REC_P254=cpu` | identity_p254; shrink_wrap; whole wrap | 1.152 s; 5.290 s; 6.44 s | 13.944 s; 1.658 s; 15.60 s | identity_p254 +12.79 s (12x), 8/8; first-wrap total +9.16 s | **holds** (see the note below the table) |
| C11 split eval_check | `RISC0_METAL_REC_KERNELS=monolith` | recursion (13 proofs); identity_p254 | 1.805 s; 1.152 s | 2.895 [2.89-2.94] s; 1.238 s | +1.090 s (+60 %), 8/8; p254 +86 ms, 8/8 | **holds** (larger than C11's claim of 26-30 % per proof; the monolith arm here also lacks C12's lz) |
| C12 lazy-Fp eval_check | `RISC0_METAL_REC_KERNELS=split` | recursion; identity_p254 | 1.805 s; 1.152 s | 1.940 [1.94-1.95] s; 1.163 s | +0.135 s (+7.5 %, ~10 ms per proof), 8/8; p254 +11 ms, 8/8 | **holds** (C12 predicted ~10 ms per lift/join) |
| R13 GPU accum scans | `RISC0_METAL_ACCUM_SCAN=cpu` | segments | 14.553 s | 14.555 [14.48-14.65] s | +0.007 s, 4/8 | **no change** at worker level; see the in-process A/B below |
| R15e evalfast const table | `RISC0_METAL_RV32IM_EVAL=evalfast_r12` | segments | 14.553 s | 18.055 [17.97-18.11] s | +3.49 s (+0.50 s/segment, +24 %), 8/8 | **holds** (the largest segment win of the wave) |
| R15w sorted + single-buffer witgen | `RISC0_METAL_WITGEN_SORT=0 RISC0_METAL_WITGEN_SINGLE_BUFFER=0` | segments | 14.553 s | 14.575 [14.53-14.61] s | +0.003 s, 4/8 | **no change** at worker level (below its resolution); in-process A/B below holds |
| R14 rv32im PSO archive | `RISC0_METAL_RV32IM_ARCHIVE=0` | warm: segments and startup | 14.553 s, startup 0.0 s | 14.535 s, startup 0.0 s | -0.015 s, 2/8 | warm: **no change** (expected); cold: holds (below) |

Note on Z10: with Metal p254, the first shrink_wrap of a process is 3.6 s longer (5.29 vs 1.66 s). The CPU
p254 used to hide the witness-graph preload. Z10 already raised this as an open issue: start
`prestart_prover()` earlier.

**In-process A/Bs (rv32im release lib tests).** The worker cannot resolve these sizes, so they ran in process.
Load 1.2-1.6.
- **R13 `accum_ab`**, `R13_AB_PAIRS=15`, po2 20 (`r13_accum_ab.log`):
  - per-pair step B-A median **-1.49 ms [-3.50, +2.04]**;
  - A'-A median +0.49 ms [-5.73, +4.22];
  - the scans alone take 0.81 ms on the CPU and 0.26 ms on the GPU.

  **Still no change by the protocol.** The direction is consistent with R13's estimate of about -2 ms, but the
  difference is inside the A-A spread even on the quiet machine. The code stays (it is bit-exact and not
  slower).
- **R15w `r15w_witgen_ab`**, `R15W_VARIANTS="r13;base;r13"`, 9 pairs, po2 20 shift_loop
  (`r15w_witgen_ab.log`). Every call matched the CPU witgen.

  | variant | kernels | call | minus A per pair |
  |---|---|---|---|
  | r13 (A) | 21.23 ms [18.95-22.01] | 29.83 ms | - |
  | base (R15w) | **14.84 ms** [13.14-16.54] | **18.17 ms** | **-6.34 ms [-8.19, -2.52]**, 9/9 |
  | r13 (A') | - | - | +0.04 [-0.68, +2.35] |

  **Holds.** On the quiet machine the call-level gain (-11.7 ms) is larger than the round-1 claim. The memory
  gain (witgen-phase peak -885 MB) is not visible in the process peak, which the eval phase sets.

**Cold start (R14, C11 PSO threads)** (`cold/summary.txt`). `MTL_SHADER_CACHE_SIZE=0`, `BENCH_ONLY=fibonacci`.
- **R14 (1 pair; each cold run takes minutes when the archive is off):**
  - archive on: rv32im+zkp 11.4 s, 13 rv32im PSOs in 0.02 s from the archive;
  - `RISC0_METAL_RV32IM_ARCHIVE=0`: rv32im+zkp 84.8 s, 13 PSOs in 74.9 s.

  **Holds.**
- **C11, `RISC0_METAL_REC_PSO_THREADS=1` vs default (5 pairs):**

  | | default | `RISC0_METAL_REC_PSO_THREADS=1` |
  |---|---|---|
  | recursion PSO build | 9.8-11.3 s | 9.7-9.8 s |
  | whole pipeline build | 25.0-26.2 s | 24.9-25.4 s |

  **No change.** The worker builds all units concurrently and the groth16 PSOs (~25 s) bound the build, so the
  parallel recursion build does not shorten worker startup. C11's split-kernel part of the cold win is
  unaffected.

**Groth16 (G9, G10)** (`g16ab/run.sh`, `g16ab/summary.txt`). `metal_shrink_wrap`, 9 rounds of A1, G9 off,
G10 off, A2. Every wrap verified. A2-A1 GPU-sum spread: +1.5 ms [-4.2, +6.4].

| item | knob | GPU sum per wrap | MSM B2 | wrap (key cached) | device alloc / process peak | per-pair delta | verdict |
|---|---|---|---|---|---|---|---|
| default | - | 1377 [1364-1410] ms | 558 ms | 1.605 [1.597-1.618] s | 3893 MiB / 6.06 GiB | - | - |
| G9 | `RISC0_G16_METAL_G9=0` | 2428 [2413-2458] ms | 1040 ms | 2.656 [2.646-2.672] s | 3901 MiB / 6.13 GiB | +1048 ms [+1043, +1058], 9/9 slower | **holds** (-43 % GPU, -1.05 s per wrap) |
| G10 | `RISC0_G16_METAL_G10=0` | 1355 [1344-1367] ms | 558 ms | 1.596 [1.581-1.625] s | **5627 MiB / 10.1 GiB** | -21.8 ms [-30.5, -19.4], 0/9 slower | **memory win holds** (-1734 MiB device, -4.3 GB process peak). It costs **~22 ms GPU per wrap (1.6 %)**: the mmap / compact layout, as G10 documented (+4-5 ms per G1 MSM, +11 ms H). |

**Keccak (K7)** (`k7ab/run.sh`, `k7ab/summary.txt`, `k7ab/p18_summary.txt`). Process-level A/B of
`k7_phase_breakdown` at po2 17: 9 rounds of A1, strict, chunk 32768, A2, with 3 proofs after a warm-up.
A2-A1: +0.3 ms [-5.9, +2.2].

| arm | eval_check po2 17 | prove po2 17 | process peak | per-round delta | verdict |
|---|---|---|---|---|---|
| default (lazy, chunk 16384) | 627.3 [625.1-631.8] ms | 829.8 ms | 5.26 GiB | - | - |
| `RISC0_KECCAK_METAL_EVAL_MACC=strict` | 1079.3 ms | 1290.9 ms | 7.42 GiB | +460.5 ms [+456.6, +462.5], 9/9 slower | **holds** (-36 % prove) |
| `RISC0_KECCAK_METAL_EVAL_CHUNK=32768` | **590.8 ms** | **793.6 ms** | 5.76 GiB | **-36.0 ms [-38.8, -33.6], 0/9 slower** | **does NOT hold as "same speed"** |

At po2 18 (7 pairs, A vs chunk 32768), chunk 32768 is again faster:

| | eval_check | prove | process peak |
|---|---|---|---|
| default (chunk 16384) | 1249 ms | 1654 ms | 8.73 GiB |
| chunk 32768 | 1169 ms | 1576 ms | 9.24 GiB |

So on a quiet GPU the 16384 chunk costs **4.4-4.7 % of a keccak proof** in exchange for the 532 MiB (K7's
claim was speed-neutral at po2 17; that measurement ran under load 40-72).

**Other baseline asks from the notes.**
- C12: `RISC0_REC_HOST_CHAIN_MIN_LEN=32` on the worker, 7 pairs (`hc/summary.txt`). Lift/join median 1.800 s
  vs 1.803 s (A), and identity_p254 is unchanged. **No change**, so the default of 16 stays right.
- Z9's `z9_ab_poseidon2` and `z9_eu2_s_sweep` micro-benches: **not run**. The worker-level Poseidon2 A/B above
  settles the e2e question, and the eu2 sweep is 0.8 % of GPU time.

## 4. Simulated budgets (`RISC0_METAL_MAX_WORKING_SET_BYTES`)

`BENCH_ONLY=fibonacci` and `BENCH_ONLY=memory-merkle`, each in a fresh worker process (`budget/`). The load
average was 1.5-2.3. **Everything proves, wraps (Groth16) and verifies at both budgets.** Before G10 the
4 GiB tier could not wrap.

| budget | program | segment po2 | STARK | Groth16 wrap | Metal peak MiB (segments / recursion / wrap) | process peak footprint |
|---|---|---|---|---|---|---|
| 4 GiB | fibonacci | 17 | 0.40 s | 6.33 s (identity_p254 1.11 + shrink_wrap 5.22) | 328 / 1731 / 3920 | 5.73 GB |
| 4 GiB | memory-merkle | 17 | 40.80 s (segments 20.62 + recursion 20.18), 10,289,152 cycles, 252K c/s | 6.34 s | 1913 / 3801 / 3920 | 8.96 GB |
| 8 GiB | fibonacci | 18 | 0.37 s | 6.34 s | 328 / 1731 / 3920 | 5.73 GB |
| 8 GiB | memory-merkle | 18 | 24.87 s (segments 16.63 + recursion 8.24), 8,388,608 cycles, 337K c/s | 6.34 s | 2477 / 4166 / 4166 | 9.30 GB |
| 28.1 GiB (default) | memory-merkle | 20 | 16.2 s, 452K c/s | 2.8 s (6.4 s first in process) | 9402 / 11098 / 11098 | 20.0 GB (suite) |

At 4 GiB the wrap peaks at 3920 MiB, 176 MiB under the budget. At po2 17 recursion is half of the STARK time:
memory-merkle needs about 4x as many lift/join proofs. Wrap times are first-in-process (they include the
~3.6 s preload).

## 5. Core-count dependence (PLAN 9.1)

Everything above ran on the 40-core M5 Max only. None of the wave-5 defaults hard-codes a core count:
launch widths come from PSO limits and occupancy, and PSO thread pools from `available_parallelism`. The
8-10 and 16-20-core parts remain unmeasured.

## 6. Open items from this baseline

- The **first wrap of each worker process** pays ~3.6 s of witness-graph preload and key load (shrink_wrap
  5.2 s vs 1.65 s). Z10's open issue (call `prestart_prover()` at worker start) would bring the first wrap
  from 6.4 s down to ~2.8 s.
- **K7 chunk default**: on a quiet GPU, chunk 32768 is 4.4-4.7 % faster at po2 17/18 for +532 MiB.
  - Choosing the chunk from the working-set budget (32768 when the budget allows) would recover the speed on
    large Macs and keep the 4 GiB fit.
  - Alternatively, accept the trade and update K7's note.
- **G10** costs ~22 ms GPU per wrap (1.6 %) for -1.7 GB. `RISC0_G16_METAL_G10_OFF=mmap` remains the
  documented speed option on large Macs.
- **R13**: no change, even quiet (-1.5 ms inside the A-A spread). **C11 PSO threads**: no change in the worker,
  because groth16 PSOs bound cold startup at ~25 s. Cold startup now waits on the groth16 PSOs; an archive for
  groth16 would cut it.
- Remaining gap to the RTX 4090: 2.9x STARK. eval_check is 79 % of a segment (1.60 s); then data commit (11 %),
  accum commit (6 %). In the wrap, MSM B2 is 40 % of shrink_wrap and the Poseidon254 row hash is 63 % of
  identity_p254.
- Not measured here: `comprehensive_zkvm_test`, CPU prover baselines, NTT GB/s, xcrun compile times
  (other parts of the PLAN 6.4 list). Their numbers are in the item notes.
