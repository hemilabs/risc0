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

# Wave 5b

Quiet-machine baseline after the wave-5b merges: M7 (zk-miner), K9, G11, G12, R18 and R19 (risc0).

| | |
|---|---|
| Date | 2026-11-12, 12:45-14:05 local time |
| Machine | Apple M5 Max, 40-core GPU, 36 GB unified memory (28.1 GiB recommended working set), macOS 27.2.1 (26B131) |
| risc0 | `zkminer/metal` @ `dab2ba1a6aea0eee85bb70e6a62c0ae34ba9ce2b` (merge of metal/G12) |
| zk-miner | `metal` @ `789fed55316b16022d43b895679990f36c744e8d` (clean tree; path-patched to the risc0 checkout above) |
| Worker | `cargo build --release -p zkminer-prove-risc0 --features metal --bin zkminer-prove-risc0-metal`, installed to `~/.zkminer/provers/zkminer-prove-risc0-metal` (sha256 `9fefcbd1…a48f92`). The rv32im and groth16 metallibs in the worker build and in the risc0 test builds are byte-identical. |
| Benchmark command | `target/release/zkminer-prove-risc0-metal --benchmark` (full suite; `BENCH_ONLY=<name>` for one program) under `/usr/bin/time -l` |
| Machine state | No other cargo, test or prover process ran (checked with `ps` before starting; the only other processes were idle shells, a `macmon` monitor and the agent CLIs). Load averages are given per table; they stayed between 0.5 and 2.9 for the whole session. |
| Logs | `/private/tmp/claude-501/-Users-user-zkminer/a30bccea-eec8-43d4-ac51-c5ccb6e4b957/scratchpad/w5b/baseline/` |

## 5b.1 Worker `--benchmark` (full suite, po2 20 segments)

Two full runs back to back (`full1.log`, `full2.log`). The load average was 3.40 falling to 0.99 for run 1, and
0.99 to 1.07 for run 2. Both runs proved, wrapped and verified every program.

| program | cycles | wave 5 (s) | run 1 (s) | run 2 (s) | change | RTX 4090 (s) | Metal / 4090 |
|---|---|---|---|---|---|---|---|
| fibonacci | 32,768 | 0.40 | 0.35 | 0.35 | -12 % | 0.48 | 0.73x |
| sha256-chain | 46,137,344 | 100.75 | 58.57 | 58.70 | -41.9 % | 35.82 | 1.64x |
| ecdsa-verify | 65,011,712 | 141.24 | 82.34 | 82.25 | -41.7 % | 47.65 | 1.73x |
| bigint-mul | 27,525,120 | 60.49 | 35.29 | 35.29 | -41.7 % | 19.77 | 1.79x |
| memory-merkle | 7,340,032 | 16.18 | 9.53 | 9.52 | -41.1 % | 5.70 | 1.67x |
| chacha-mix | 8,404,992 | 18.81 | 11.13 | 11.16 | -40.8 % | 6.67 | 1.67x |
| **all (sum of cycles / sum of STARK time)** | 154,451,968 | 337.87 s, 457K c/s | 197.21 s, **783K c/s** | 197.27 s, **783K c/s** | **+71 % c/s** | 116.09 s, 1.33M c/s | **1.70x** |

- The RTX 5090 (~2.3M c/s) is now 2.94x this Mac (5.0x at wave 5). The 4090 gap fell from 2.91x to 1.70x.
- Segment / recursion split (run 1): sha256-chain 47.32 + 11.25 s (44 segments), ecdsa-verify 66.49 + 15.85 s
  (62), bigint-mul 28.36 + 6.93 s (27), memory-merkle 7.75 + 1.78 s (7), chacha-mix 8.81 + 2.32 s (9).
  A worker segment is now 1.11 s (memory-merkle: 7.75 s / 7; wave 5: 2.06 s). Recursion is unchanged (no
  wave-5b item touched it). The whole STARK gain is R18's eval_check.

**Groth16 wrap** (first and warm reported separately):

| wrap | identity_p254 | shrink_wrap | total |
|---|---|---|---|
| warm: every wrap after the first in a process (10 samples, runs 1-2) | 1.125-1.150 s | 1.459-1.492 s | **2.59-2.64 s** (wave 5: 2.78-2.83) |
| first wrap of a `--benchmark` process (fibonacci: 0.35 s STARK, so the graph preload is not yet done) | 1.105-1.107 s | 4.63-4.68 s | **5.73-5.78 s** (wave 5: 6.4 s) |
| first wrap when >= ~4.7 s pass between worker start and the wrap (`BENCH_ONLY=memory-merkle`, 16 runs of A1/A2; also every cold-cache start) | 1.14-1.15 s | 1.45-1.46 s | **2.60-2.62 s** |

The M7 preload logs `Groth16 witness graph ready 4.70-4.72 s after the preload started`. The full suite runs
fibonacci first, so its first wrap still waits ~3.2 s for the graph; every production job whose STARK +
identity_p254 lasts longer than ~4.7 s after worker start gets a warm-speed first wrap. The RTX 4090 wraps in
~2.0 s.

**Memory.**

| | wave 5 | wave 5b |
|---|---|---|
| process peak footprint, full suite (`time -l`) | 20.02 / 20.01 GB | **18.30 / 18.30 GB** |
| `--benchmark` process lifetime peak | 19079 MiB | 17450 / 17451 MiB |
| CPU user time of the full suite | 29.8 s | 29.75 / 30.00 s |
| wall time of the full suite | - | 217.2 / 216.3 s |

Metal peaks per phase at po2 20 (MiB, segments / recursion / wrap) are unchanged within a few MiB:
memory-merkle 9400 / 11096 / 11096, chacha-mix 9492 / 11390 / 11390, bigint-mul 10201 / 12099 / 12099
(largest), fibonacci 328 / 1731 / 3926. The lower process peak comes from the shorter segment phase (fewer
host-side segments in flight); it is not attributed to one item.

**Cold start** (`cold/summary.txt`, `cold/g11_summary.txt`; `MTL_SHADER_CACHE_SIZE=0`, `BENCH_ONLY=fibonacci`).

| | pipeline build | of which groth16 | process wall (`time -p`) | first wrap |
|---|---|---|---|---|
| cold OS shader cache, default (12 runs) | **9.2-11.6 s** (median 9.3) | 0.0 s (archive hit) | 12.2-14.6 s | 2.56 s (the PSO build hides the graph preload) |
| warm OS shader cache (5 runs) | 0.0 s | 0.0 s | 6.23-6.26 s | 5.81-5.83 s |
| wave 5 cold | 24.9-26.2 s | ~25 s | - | - |

Cold startup is now set by rv32im+zkp (9.2-11.6 s) and recursion (9.2-11.6 s), built in parallel; keccak and
groth16 load from their archives in 0.0 s. The warm-cache process wall (6.2 s for a 0.35 s proof) is the
wait for the witness graph.

## 5b.2 Per-phase breakdown (quiet machine)

### po2 20 rv32im segment
`risc0-circuit-rv32im` release lib test `prove_po2_sweep`, `R11_PO2S=20 R11_CPU=0 R11_REPS=6`
(`phase_rv32im.log`, load 2.16 -> 1.90). Reps 1-5; rep 0 (cold first call) was 1329 ms. Every seal verifies.
Peak GPU allocation is 9287 MiB (unchanged).

| phase | wave 5 ms | wave 5b ms (reps 1-5) | share |
|---|---|---|---|
| preflight + alloc | 10-11 | 10-11 | 1.0 % |
| witgen | 8 | 7-8 | 0.7 % |
| data commit | 221 | **217** | 20.3 % |
| accum | 9-10 | 9 | 0.8 % |
| accum commit | 114-115 | **113-114** | 10.6 % |
| eval_check (evalfast) | 1598-1607 | **650-654** | 60.8 % |
| FRI + tail | 66-69 | 60-63 | 5.8 % |
| **segment** | 2031-2040 | **1068-1072** | |

The commits (data + accum, 331 ms) are now 31 % of a segment.

### identity_p254 + shrink_wrap
- **identity_p254: 1.11-1.15 s, unchanged** (no wave-5b item touched it). `RISC0_METAL_REC_PROFILE=1
  RISC0_METAL_PROFILE=1` (`prof_p254.log`), ms: witgen 18.4, pre_accum (Poseidon254 commit) 683.3, accum 5.2,
  pre_eval 122.7, eval_check 48.3. GPU 1040.6 ms: `p254_rows_w64` 651.7 ms (62.6 %), `p254_fold_w64`
  306.0 ms (29.4 %).
- **shrink_wrap, key cached** (`risc0-groth16` `metal_shrink_wrap`, `RISC0_VERBOSE=1`, 3 verified wraps per
  process, the A1 and A2 arms of `g16ab/`: 18 processes, 36 cached wraps; load 0.9-1.6):

  | witness_poly + NTT | sort_h + MSM H | fuzz + sort_w + MSM A, B1, C | MSM B2 | GPU sum | upload / encode / host | wrap |
  |---|---|---|---|---|---|---|
  | 145.5 [141.0-155.0] ms | 210.2 [209.8-211.2] ms | 382.1 [381.0-382.9] ms | **440.8 [437.5-443.2] ms (37 %)** | **1178 [1171-1189] ms** | 2.6 / 0.1 / 0.9 ms | 1.413 [1.409-1.420] s |
  | wave 5: 141-178 | 230-233 | 437 | 554-566 | 1364-1410 | | 1.60-1.69 s |

  Device allocation of the cached prover is 3899.6 MiB (+6.5 MiB, G12); the test process peak is 5.925 GiB.

## 5b.3 Wave-5b revert knobs: quiet interleaved A/B

Every arm proved, wrapped and verified (worker 48/48, groth16 36/36 processes with 108 verified wraps, keccak
45/45, rv32im 24/24 with every seal verified).

**Worker A/B** (`ab/round.sh`, `ab/summary.txt`, `ab/analyze.py`): `BENCH_ONLY=memory-merkle --benchmark`,
8 rounds of A1, M7, G11, A2, G12, R18 (start rotated by 2 each round); each arm against the mean of its
round's A1 and A2. A2-A1 spread: segments +0.005 s [-0.050, +0.020], recursion 0.00 [-0.01, +0.01] s,
identity_p254 -1.1 [-4.1, +1.8] ms, shrink_wrap -1.1 [-7.5, +5.8] ms, wrap 0.00 [-0.02, 0.00] s. Load
0.87-2.88.

| item | knob (off) | phase | knob on (default) median [min-max] | knob off median [min-max] | per-round delta (off - on), off slower in | verdict |
|---|---|---|---|---|---|---|
| M7 graph preload | `ZKMINER_G16_PRELOAD=0` | first wrap of the process | 2.61 [2.60-2.62] s (shrink_wrap 1.46 s) | 6.27 [6.19-6.29] s (shrink_wrap 5.12 s) | **+3.66 s [+3.59, +3.69]**, 8/8 | **holds**. Segments -0.023 s [-0.035, -0.005] with the knob off: inside the A-A spread, no change. Process peak footprint is **+2.14 GB** with the preload (16.63 vs 14.49 GB), the staged-`--benchmark` cost M7 documented; Metal peaks unchanged. |
| G11 groth16 PSO archive (warm) | `RISC0_G16_METAL_ARCHIVE=0` | shrink_wrap; startup | 1.461 s; 0.0 s | 1.459 s; 0.0 s | +0.2 ms [-6.8, +7.0], 4/8 | warm: **no change** (expected) |
| G12 MSM buckets / B2 | `RISC0_G16_METAL_G12=0` | shrink_wrap; wrap | 1.461 s; 2.61 s | 1.653 [1.648-1.657] s; 2.80 s | **+193 ms [+189, +199]**, 8/8; wrap +0.19 s | **holds** |
| R18 32-byte row loads | `RISC0_METAL_RV32IM_EVAL=evalfast_r15e` | 7 segments | 7.825 [7.80-7.85] s | 14.535 [14.49-14.62] s | **+6.70 s [+6.67, +6.80] (+0.96 s/segment)**, 8/8 | **holds**. The off arm's process peak is +0.93 GB (more segments in flight during the slower segment phase); Metal +16 MiB (the lazy r15e PSO). |

**Groth16 in process** (`g16ab/run.sh`, `g16ab/summary.txt`, `g16ab/analyze.py`): `metal_shrink_wrap`, 9 rounds
of A1, G12 off, G11 off, A2 (load 0.9-1.85); the mean of each process's two key-cached wraps.
A2-A1: GPU -0.4 ms [-4.3, +5.2], B2 +0.4 [-3.4, +2.1], wrap -0.002 s.

| arm | GPU sum | MSM B2 | sort_h + MSM H | fuzz + MSM A/B1/C | wrap (key cached) | first wrap (no preload) | alloc / process peak | per-round delta vs mean A | verdict |
|---|---|---|---|---|---|---|---|---|---|
| default | 1180.8 [1176.6-1182.5] ms | 440.7 ms | 210.2 | 382.1 | 1.413 s | 6.248 s | 3899.6 MiB / 5.925 GiB | - | - |
| G12 off `RISC0_G16_METAL_G12=0` | 1373.0 [1368.3-1376.4] ms | 558.4 ms | 231.0 | 436.4 | 1.604 s | 6.446 s | 3899.6 MiB / 5.919 GiB | GPU **+192.3 ms [+187.9, +197.9]**, B2 **+117.5 [+114.2, +121.1]**, wrap +0.190 s; 9/9 slower | **holds** (G12 claimed -193 ms GPU, -119 ms B2) |
| G11 off `RISC0_G16_METAL_ARCHIVE=0` | 1180.3 ms | 440.3 ms | - | - | 1.415 s | 6.250 s | 3899.6 MiB / 5.926 GiB | GPU -0.15 [-1.5, +2.4] ms, 4/9 | **no change** (kernel speed equal, as G11 claimed) |

**G11 cold start** (`cold/g11_summary.txt`): 7 interleaved pairs, `MTL_SHADER_CACHE_SIZE=0`,
`BENCH_ONLY=fibonacci`, load 1.24-2.87.

| | default (archive) | `RISC0_G16_METAL_ARCHIVE=0` |
|---|---|---|
| groth16 PSOs | 0.0 s | 31.2-32.1 s |
| pipeline build | 9.2-11.6 s | 31.2-32.1 s |
| process wall | 12.2-14.6 s | 34.3-35.2 s |

**Holds**: -22 s per cold start, 7/7 (G11 claimed 31.8 -> 10.0 s).

**R18 in process** (`r18ab/run.sh`, `r18ab/sum.py`): `prove_po2_sweep` po2 20, `R11_REPS=4` (reps 1-3), 8 rounds
of A, R18 off, A2 (load 0.92-1.70). A2-A: eval_check +0.5 ms [-2, +4].

| arm | eval_check | segment |
|---|---|---|
| default | 652 [649-656] ms | 1072.5 [1068-1075] ms |
| `RISC0_METAL_RV32IM_EVAL=evalfast_r15e` | 1596 [1584-1618] ms | 2030.5 [2023-2053] ms |
| per pair | **+943 ms [+932, +962]**, 8/8 | +957.5 ms |

**Holds** (-59 % eval_check; R18 claimed 1606 -> 699 ms with a shared GPU).

**R19** has no runtime knob (tests and bench tooling only). Its tests pass on the merged tree:
`ef_tests` 7/7 including `evalfast_matches_cpu_extreme_po2_12`, and the ignored `evalfast_matches_cpu_po2_20`
and `evalfast_seeds_match_baseline` (`r19_ef_tests.log`, `r19_ef_ignored.log`). R19's segment profile
(1.07 s, eval_check 652 ms) matches 5b.2.

**K9 keccak chunk** (`k9ab/run.sh`, `k9ab/summary.txt`, `k9ab/analyze.py`): `k7_phase_breakdown`, 3 proofs after
a warm-up, A1 / K9 off / A2.

| po2 | rounds | arm | eval_check | prove | process peak | per-round delta vs mean A | verdict |
|---|---|---|---|---|---|---|---|
| 17 | 8 | default (auto: 32768) | 591.5 [590.3-592.8] ms | 794.5 [793.2-795.7] ms | 5.78-5.87 GiB | A2-A1 eval +0.1 [-1.1, +2.5] | - |
| 17 | 8 | `RISC0_KECCAK_METAL_EVAL_CHUNK=16384` | 628.4 ms | 831.8 ms | 5.36 GiB | prove **+37.6 ms [+35.8, +40.8] (+4.7 %)**, 8/8; peak -0.52 GiB | **holds** |
| 18 | 7 | default | 1169.9 ms | 1577.9 ms | 9.28 GiB | A2-A1 prove -0.7 [-3.3, +1.2] | - |
| 18 | 7 | `RISC0_KECCAK_METAL_EVAL_CHUNK=16384` | 1249.4 ms | 1654.4 ms | 8.78 GiB | prove **+77.0 ms [+73.7, +78.3] (+4.9 %)**, 7/7; peak -0.50 GiB | **holds** |

Load 0.49-1.44. This recovers the 4.4-4.7 % the wave-5 baseline measured, for +0.5 GiB at po2 17-18 where
the budget allows.

## 5b.4 Simulated budgets (`RISC0_METAL_MAX_WORKING_SET_BYTES`)

`BENCH_ONLY=fibonacci` and `BENCH_ONLY=memory-merkle`, each in a fresh worker process (`budget/`; load
1.75-2.41). **Everything proves, wraps (Groth16) and verifies at both budgets.** Wrap times are the first in
the process; memory-merkle's STARK is long enough to hide the graph preload, fibonacci's is not.

| budget | program | segment po2 | STARK | Groth16 wrap | Metal peak MiB (segments / recursion / wrap) | process peak footprint (`time -l`) |
|---|---|---|---|---|---|---|
| 4 GiB | fibonacci | 17 | 0.35 s | 5.76 s (1.11 + 4.65) | 328 / 1732 / 3926 | 5.55 GB (wave 5: 5.73) |
| 4 GiB | memory-merkle | 17 | **31.11 s** (segments 11.09 + recursion 20.03), 10,289,152 cycles, **331K c/s** (wave 5: 40.80 s, 252K) | **2.60 s** (wave 5: 6.34) | 1912 / 3801 / 3926 | 9.08 GB (wave 5: 8.96) |
| 8 GiB | fibonacci | 18 | 0.33 s | 5.79 s | 328 / 1731 / 3926 | 5.54 GB |
| 8 GiB | memory-merkle | 18 | **17.05 s** (8.92 + 8.13), 8,388,608 cycles, **492K c/s** (wave 5: 24.87 s, 337K) | **2.60 s** (wave 5: 6.34) | 2476 / 4165 / 4165 | 9.67 GB (wave 5: 9.30) |

- The wrap peak at 4 GiB is 3926 MiB (G12 +6 MiB), 170 MiB under the budget.
- The process footprint rises by 0.1-0.4 GB on the staged memory-merkle runs (the M7 graph is resident during
  the STARK, as M7 documented for `--benchmark`); Metal peaks are unchanged.
- At po2 17 (4 GiB) recursion is now 64 % of the STARK time: 20.0 s of lift/join against 11.1 s of segments.

## 5b.5 Open items from this baseline

- **Recursion is the next STARK target at small budgets and a fifth of the STARK at po2 20.** It did not
  change in wave 5b (memory-merkle 1.78 s, 13 proofs). At 4 GiB it is 64 % of memory-merkle's STARK.
- **Segment (1.07 s at po2 20): eval_check 61 %, data commit 20 %, accum commit 11 %.** R19's P1 (zkp:
  `pv_rows_k_z9`, commit NTTs, `blit_fill`) is the next segment item; eval_check has no cheap win left
  (R19).
- **Groth16 wrap 2.6 s vs 2.0 s on the 4090.** identity_p254 (1.13 s, unchanged) is now 43 % of the wrap and
  `p254_rows_w64` is 63 % of its GPU time; MSM B2 (441 ms) is at G12's madd floor; the sorts (~60 ms) are
  G12's next MSM item.
- **First wrap of a short first job:** the graph takes 4.7 s on one core after worker start. A job that
  reaches the wrap sooner (the `--benchmark` fibonacci, 5.7 s first wrap) still waits. A faster graph build
  (parallel conversion, or a cached converted graph) would remove the rest.
- **M7 footprint:** +2.1 GB process footprint on staged `--benchmark` memory-merkle at the default budget
  (Metal unchanged). Watch it on 16 GB Macs if the worker is used in staged mode.
- **Cold start ~9.3 s** is now rv32im+zkp and recursion PSOs (G11 open issue: route the zkp HAL and
  recursion PSOs through the archive).
- Not measured here: 8-10 and 16-20-core parts (PLAN 9.1; K9's chunk crossover and R18's compiler regime
  should be re-checked there), `comprehensive_zkvm_test`, CPU prover baselines.

# Wave 5c

Quiet-machine baseline after the wave-5c merges: R20, R21 (probe), Z11, C13, Z12, G13 and G14 (risc0), and M8
(zk-miner).

| | |
|---|---|
| Date | 2026-11-13, 00:27-01:31 local time |
| Machine | Apple M5 Max, 40-core GPU, 36 GB unified memory (28.1 GiB recommended working set), macOS 27.2.1 (26B131) |
| risc0 | `zkminer/metal` @ `6234335ab008c00e2f883166fcec359df9903e6d` (merge of metal/G14) |
| zk-miner | `metal` @ `0ca7ec9f346703f976487b30e93c63a8c74b33f8` (merge of item/M8; clean tree, path-patched to the risc0 checkout above) |
| Worker | `cargo build --release -p zkminer-prove-risc0 --features metal --bin zkminer-prove-risc0-metal`, installed to `~/.zkminer/provers/zkminer-prove-risc0-metal` (sha256 `fa498152…6ec9`) |
| Benchmark command | `target/release/zkminer-prove-risc0-metal --benchmark` (full suite; `BENCH_ONLY=<name>` for one program) under `/usr/bin/time -l` |
| Machine state | No other cargo, test or prover process ran (checked with `ps` before starting and in each full run's `.load` file). Two browser renderer processes (not ours) used 100 % of one CPU core each for the whole session, so the load average never fell below ~2.3. Load averages were 2.3-4.3, with peaks to 5.2 during this run's own cold-shader-cache arms. |
| Logs | `/private/tmp/claude-501/-Users-user-zkminer/a30bccea-eec8-43d4-ac51-c5ccb6e4b957/scratchpad/w5c/baseline/` |

Before the full runs, one `RISC0_METAL_RV32IM_EVAL_FC=wait` memory-merkle run per budget (`warm/`) made sure the
R20 specialized-PSO archives for po2 17-20 existed. All of them were already archive hits.

## 5c.1 Worker `--benchmark` (full suite, po2 20 segments)

Two full default runs (`full1.log`, `full2.log`), interleaved with two `RISC0_METAL_REC_CTRL_PERSIST=0` runs
(`fullC13off1/2.log`, section 5c.3). Load 3.21 -> 3.65 (run 1), 2.53 -> 2.57 (run 2). Both runs proved, wrapped
and verified every program.

| program | cycles | wave 5b (s) | run 1 (s) | run 2 (s) | change | RTX 4090 (s) | Metal / 4090 |
|---|---|---|---|---|---|---|---|
| fibonacci | 32,768 | 0.35 | 0.37 | 0.36 | noise | 0.48 | 0.76x |
| sha256-chain | 46,137,344 | 58.57 | 56.37 | 56.09 | -3.8 % | 35.82 | 1.57x |
| ecdsa-verify | 65,011,712 | 82.34 | 78.81 | 78.63 | -4.3 % | 47.65 | 1.65x |
| bigint-mul | 27,525,120 | 35.29 | 34.08 | 33.83 | -3.4 % | 19.77 | 1.72x |
| memory-merkle | 7,340,032 | 9.53 | 9.18 | 9.12 | -3.7 % | 5.70 | 1.61x |
| chacha-mix | 8,404,992 | 11.13 | 10.68 | 10.68 | -4.0 % | 6.67 | 1.60x |
| **all (sum of cycles / sum of STARK time)** | 154,451,968 | 197.21 s, 783K c/s | 189.49 s, **815K c/s** | 188.71 s, **818K c/s** | **+4.1 % c/s** | 116.09 s, 1.33M c/s | **1.63x** |

- The RTX 5090 (~2.3M c/s) is 2.82x this Mac (wave 5b: 2.94x). The 4090 gap fell from 1.70x to 1.63x.
- Segment / recursion split (run 1): sha256-chain 45.39 + 10.98 s, ecdsa-verify 63.39 + 15.41 s, bigint-mul
  27.34 + 6.75 s, memory-merkle 7.44 + 1.75 s, chacha-mix 8.41 + 2.26 s. Segments 152.2 s in all (wave 5b
  ~158.7), recursion 37.3 s (38.1). A worker segment is 1.06 s (memory-merkle 7.44 s / 7; wave 5b 1.11 s).
- The STARK gain is R20 (~41 ms per po2 20 segment) plus Z11 (~8 ms per segment and ~1.5 ms per lift/join)
  plus C13 (b) (~3 ms per lift/join); see 5c.3.

**Groth16 wrap** (first and warm reported separately):

| wrap | identity_p254 | shrink_wrap | total |
|---|---|---|---|
| warm: every wrap after the first in a process (10 samples, runs 1-2) | **918-922 ms** | 689-701 ms | **1.61-1.62 s** (wave 5b: 2.59-2.64) |
| first wrap of a `--benchmark` process (fibonacci, 0.36 s STARK) | 1.106-1.114 s | 691-693 ms | **1.80-1.81 s** (wave 5b: 5.73-5.78) |
| first wrap of a fresh `BENCH_ONLY=fibonacci` worker (16 runs, A arms of 5c.3) | 1.106-1.118 s | 691-698 ms | **1.80-1.81 s** |
| first wrap of a fresh `BENCH_ONLY=memory-merkle` worker (16 runs) | 1.097-1.113 s | 688-695 ms | 1.79-1.81 s |

- The first wrap no longer waits for the witness graph: G13 builds it in 0.89-1.02 s after the preload starts
  (wave 5b: 4.7 s). The first wrap is now only ~0.19 s slower than a warm one: identity_p254 still commits its
  ctrl group on the first job of a process (C13 keeps it for later jobs).
- The RTX 4090 wraps in ~2.0 s. **The Mac's warm wrap (1.62 s) is now faster than the 4090's.**

**Memory.**

| | wave 5b | wave 5c |
|---|---|---|
| process peak footprint, full suite (`time -l`) | 18.30 / 18.30 GB | **15.67 / 15.67 GB** (C13 off: 15.57 / 15.56) |
| `--benchmark` process lifetime peak | 17450 / 17451 MiB | 14942 / 14942 MiB |
| CPU user time of the full suite | 29.75 / 30.00 s | 29.0 / 29.0 s |
| wall time of the full suite | 217.2 / 216.3 s | 199.6 / 198.7 s |

Metal peaks per phase at po2 20 (MiB, segments / recursion / wrap):

| program | wave 5b | wave 5c |
|---|---|---|
| memory-merkle | 9400 / 11096 / 11096 | 9416 / **9416** / **5347** |
| bigint-mul (largest) | 10201 / 12099 / 12099 | 10217 / 10217 / 5347 |
| chacha-mix | 9492 / 11390 / 11390 | 9508 / 9508 / 5347 |
| fibonacci | 328 / 1731 / 3926 | 344 / 1449 / 5347 |

- Segments +16 MiB: R20's cached specialized PSOs.
- Recursion no longer stacks on the segment pool: the worker now releases the segment prover before
  lift/join (M8 sets Z12's `RISC0_METAL_RELEASE_SEG_BEFORE_REC=1`). The recursion column is now the segment
  peak, carried over.
- The wrap peak is 5347 MiB at the default and 8 GiB budgets, because G13's early key load puts the Groth16
  key beside identity_p254. At 4 GiB the key is not loaded early and the wrap peak is 3942 MiB.
- The process footprint fell by 2.6 GB. Most of it is the release (-2.0 GB, 5c.3); the rest is G13 (-0.4 GB,
  no ref-program graph). C13 (a) adds ~0.1 GB of host memory (the kept ctrl group).

**Cold start** (`cold/summary.txt`; `BENCH_ONLY=fibonacci`; 7 interleaved cold / warm pairs, load 2.96-5.24).

| | pipeline build | of which groth16 | process wall (`time -p`) | first wrap |
|---|---|---|---|---|
| cold OS shader cache (`MTL_SHADER_CACHE_SIZE=0`, 7 runs) | **9.5-9.9 s** (median 9.6) | 0.0 s (archive hit) | 11.68-12.12 s (wave 5b: 12.2-14.6) | 1.81-1.82 s |
| warm OS shader cache (7 runs) | 0.0-0.1 s | 0.0 s | **2.29-2.32 s** (wave 5b: 6.23-6.26) | 1.81-1.83 s |

- Cold startup is still rv32im+zkp and recursion PSOs (9.5-9.9 s each, in parallel). Nothing in wave 5c
  touched it.
- Warm process wall fell by 3.9 s: the job no longer waits for the witness graph (G13).
- R20's prewarm loaded 8 specialized PSOs in 0.02 s each in every warm run. In the cold-cache runs it loaded
  none before the process ended, so a cold-cache first job runs the plain `ef_eval_split` (R20 open issue 3,
  as documented).

## 5c.2 Per-phase breakdown (quiet machine)

### po2 20 rv32im segment
`risc0-circuit-rv32im` release lib test `prove_po2_sweep`, `R11_PO2S=20 R11_CPU=0 R11_REPS=4`. These are the A and A2
arms of `phab/` (16 processes, the per-process median of reps 1-3; load 2.87-3.88). Rep 0 (cold first call) was
1282 ms. Every seal verifies (128/128 across `phab/`). Peak GPU allocation is 9303 MiB (+16 MiB, R20).

| phase | wave 5b ms | wave 5c ms | share |
|---|---|---|---|
| preflight + alloc | 10-11 | 11 | 1.1 % |
| witgen | 7-8 | 8 | 0.8 % |
| data commit | 217 | 217 | 21.3 % |
| accum | 9 | 9 | 0.9 % |
| accum commit | 113-114 | 112-113 | 11.0 % |
| eval_check (evalfast, R20 specialized) | 650-654 | **608 [604-617]** | 59.7 % |
| FRI + tail | 60-63 | **53-54** | 5.2 % |
| **segment** | 1068-1072 | **1020 [1016-1029]** | |

The commits (data + accum, 329 ms) are now 32 % of a segment, and eval_check 60 %.

### identity_p254 + shrink_wrap
- **identity_p254**:
  - **First job of a process** (worker `BENCH_ONLY=fibonacci`, `RISC0_METAL_REC_PROFILE=1 RISC0_METAL_PROFILE=1`,
    `prof_p254.log`), ms: witgen 20.2, pre_accum 701.4, accum 5.2, pre_eval 123.1, eval_check 45.0. GPU
    1047.5 ms: `p254_rows_w64` 653.3 ms (62.4 %, 7 calls), `p254_fold_w64` 313.9 ms (30.0 %, 116 calls).
  - **Later jobs** (C13's kept ctrl group restored): `c13_bench_jobs` with fresh provers per round
    (`C11_ONLY=poseidon_254`, `prof_p254_jobs.log`), ms: pre_accum 512-513, pre_eval 122.7-122.8, eval_check
    45.3-45.7, proof 917 ms. GPU 864.6 ms: `p254_rows_w64` 545.6 ms (63.1 %, 6 calls), `p254_fold_w64`
    238.7 ms (27.6 %, 96 calls). C13 reports `host_bytes=103169564`.
  - In the worker: 918-922 ms per warm wrap, 1.10-1.11 s for the first.
- **shrink_wrap, key cached** (`risc0-groth16` `metal_shrink_wrap`, `RISC0_VERBOSE=1`, 3 verified wraps per
  process, the A1 and A2 arms of `g16ab/`: 16 processes; each is the mean of its 2 key-cached wraps; load
  2.65-3.05):

  | witness_poly + NTT | sort_h + MSM H | sort_w + MSM A, B1, C | MSM B2 | GPU sum | wrap |
  |---|---|---|---|---|---|
  | 145.4 [143.9-146.0] ms | **210.5 [210.0-211.2] ms (34 %)** | 118.5 [117.7-121.3] ms | 150.9 [150.2-151.8] ms | **625.5 [623.3-627.2] ms** | 651.6 [649.6-653.1] ms |
  | wave 5b: 145.5 | 210.2 | 382.1 (with the fuzz) | 440.8 | 1178 | 1.413 s |

  - The first wrap of each process (key load) takes 1.66 s. Device allocation is 3899.7 MiB.
  - The test process peak is 6.28 GiB (wave 5b: 5.93 GiB; G13's `par` witness array is resident).
  - In the worker, shrink_wrap is 689-701 ms at the default budget. At the 4 and 8 GiB budgets it is
    824-834 ms: below a 20 GiB limit G13 uses the `fast` evaluator, not `par`.

## 5c.3 Wave-5c revert knobs: quiet interleaved A/B

Every arm proved, wrapped and verified:
- worker: 205/205 processes, every Groth16 receipt `verified`;
- groth16: 24/24 `metal_shrink_wrap` processes, 72 verified wraps;
- rv32im: 32/32 sweep processes, 128 seals verified;
- recursion: 1/1 `c13_bench_jobs`.

**Worker A/B, default budget** (`ab/round.sh`, `ab/analyze.py`, `ab/summary_def_memory-merkle.txt`):
`BENCH_ONLY=memory-merkle --benchmark`, 8 rounds of A1, R20, Z11, C13b, A2, REL0, G13, G14, with the start
rotated by 3 each round; each arm against the mean of its round's A1 and A2. Load 2.32-4.27.

A2-A1 spread:
- STARK 0.000 [-0.03, +0.10] s; segments 0.000 [-0.04, +0.09] s; recursion 0.000 [-0.01, +0.01] s;
- identity_p254 +0.2 [-13.2, +12.3] ms; shrink_wrap +0.1 [-3.4, +4.3] ms; wrap 0.00 [-0.01, +0.01] s;
- footprint +0.007 GB.

| item | knob (off) | phase | knob on (default) median [min-max] | knob off median [min-max] | per-round delta (off - on), off slower in | verdict |
|---|---|---|---|---|---|---|
| R20 specialized eval | `RISC0_METAL_RV32IM_EVAL_FC=0` | 7 segments | 7.490 [7.45-7.52] s | 7.780 [7.75-7.86] s | **+0.290 s [+0.255, +0.390] (+41 ms/segment, 3.9 %)**, 8/8 | **holds**. The off arm has a 16 MiB lower Metal peak and a 0.07 GB higher process footprint (more segments in flight). |
| Z11 parallel query reads | `RISC0_METAL_QUERY_PAR=0` | segments; recursion; STARK | 7.490 s; 1.740 s; 9.230 s | 7.540 s; 1.760 s; 9.300 s | seg +0.055 [+0.005, +0.090], 8/8; rec +0.020 [+0.010, +0.030], 8/8; STARK +0.077 [+0.020, +0.120], 8/8 | **holds** (8/8 in every column; the worker deltas are small next to the A-A extremes, but the in-process A/B below is 8x its spread) |
| C13 (b) uint4 mix load | `RISC0_METAL_REC_LZ_UINT4=0` | recursion (13 proofs) | 1.740 [1.73-1.75] s | 1.780 [1.77-1.78] s | **+0.040 s [+0.030, +0.045] (~3 ms per proof, 2.3 %)**, 8/8 | **holds** (C13 expected ~2 %) |
| Z12 / M8 release before recursion | `RISC0_METAL_RELEASE_SEG_BEFORE_REC=0` | recursion; identity_p254; footprint | 1.740 s; 1107 ms; 14.17 GB, Metal 9420 / 9420 / 5347 MiB | 1.760 s; 1154 ms; 16.16 GB, Metal 9420 / **11116** / 11116 MiB | rec +0.020 [+0.015, +0.030], 8/8; identity_p254 **+44.7 ms [+38.9, +51.0]**, 8/8; footprint **+1.98 GB**, 8/8; STARK +0.022 (no change) | **holds**. The release costs nothing; it saves 2.0 GB, and identity_p254 is ~45 ms faster with it (cause not investigated). |
| G13 graph memo + `par` witness + early key (all three) | `RISC0_G16_METAL_WITNESS=ref RISC0_G16_GRAPH_MEMO=0 RISC0_G16_METAL_EARLY_KEY=0` | warm-graph wrap | shrink_wrap 692.6 ms; wrap 1.80 s | 911.5 ms; 2.01 s | shrink_wrap **+220.4 ms [+216.0, +222.6]**, 8/8; identity_p254 -7.1 ms (off faster, 0/8) | **holds** (G13 expected -200 ms warm). Metal wrap peak 5347 with G13 vs 3942 MiB without (early key, +1405 MiB). Process footprint is 0.40 GB lower with G13. |
| G14 witness MSMs without the fuzz | `RISC0_G16_METAL_FUZZ=1` | shrink_wrap | 692.6 ms | 1244.9 [1243.0-1247.5] ms | **+552.6 ms [+550.3, +557.3]**, 8/8 | **holds** (G14 expected -540 to -555 ms). Peaks unchanged. |

**First wrap of a short first job** (`summary_def_fibonacci.txt`): `BENCH_ONLY=fibonacci` fresh workers, 8 rounds of A1, G13 off,
A2, G14 off, early key off. Load 2.84-3.24. A2-A1: wrap 0.00 [-0.01, +0.01] s, shrink_wrap +1.3 [-5.3, +3.4] ms.

| arm | first wrap | shrink_wrap | identity_p254 | process wall | process peak | per-round delta vs mean A | verdict |
|---|---|---|---|---|---|---|---|
| default | **1.81 [1.80-1.81] s** | 692.7 ms | 1113.7 ms | 2.25 s | 5.21 GB | - | - |
| G13 all off | 5.545 [5.54-5.55] s | 4437.4 ms | 1105.3 ms | 5.98 s | 5.66 GB | wrap **+3.737 s [+3.730, +3.750]**, 8/8 | **holds** (G13 expected 5.7 -> ~2.6 s; it is 1.81 s) |
| `RISC0_G16_METAL_EARLY_KEY=0` only | 1.88 [1.88-1.89] s | 775.2 ms | 1103.7 ms | 2.33 s | 5.32 GB | shrink_wrap +82.7 ms [+77.6, +85.2], 8/8; identity_p254 -10.1 ms; wrap +0.07 s, 8/8 | **holds** (net -70 ms per wrap, for +1405 MiB of Metal wrap peak at budgets >= 8 GiB) |
| `RISC0_G16_METAL_FUZZ=1` | 2.36 [2.35-2.37] s | 1246.4 ms | 1112.6 ms | 2.81 s | 5.21 GB | shrink_wrap **+553.6 ms [+551.4, +558.5]**, 8/8 | **holds** |

**C13 (a) kept ctrl group** needs a second job in one process, which `BENCH_ONLY` cannot run. It was therefore
measured on the full suite: A, `RISC0_METAL_REC_CTRL_PERSIST=0`, A, off (2 interleaved process pairs, 10 warm
wraps per arm).

| | identity_p254, warm wraps (10 each) | first wrap of the process | full-suite STARK | process peak footprint |
|---|---|---|---|---|
| default | **918.2-922.4 ms** (median 920.1) | 1105.7-1113.7 ms | 189.49 / 188.71 s | 15.67 / 15.67 GB |
| `RISC0_METAL_REC_CTRL_PERSIST=0` | 1077.3-1089.6 ms (median 1087.0) | 1103.9-1104.8 ms | 188.70 / 189.00 s | 15.57 / 15.56 GB |

**Holds**: -167 ms per warm wrap (-15 % of identity_p254; C13 measured -230 ms in process and expected ~-200 ms).
The ranges do not overlap. The cost is +0.10 GB of host footprint. `c13_bench_jobs` agrees: 1174 ms for the
first job (cold PSOs, committed ctrl), then 917 ms per later job (restored).

**In-process A/B for R20 and Z11** (`phab/run.sh`, `phab/ana.py`): `prove_po2_sweep` po2 20, `R11_REPS=4`
(reps 1-3), 8 rounds of A, R20 off, Z11 off, A2 (rotated). Load 2.80-3.89. A2-A: eval_check +1.0 [-4, +8] ms,
FRI + tail -1.0 [-1, +1] ms, segment +0.5 [-5, +7] ms.

| arm | eval_check | FRI + tail | segment | per-round delta vs mean A | verdict |
|---|---|---|---|---|---|
| default | 608 [604-616] ms | 54 [53-54] ms | 1019.5 ms | - | - |
| `RISC0_METAL_RV32IM_EVAL_FC=0` | 651 [647-654] ms | 54 ms | 1062.5 ms | eval_check **+42.2 ms [+36.0, +43.0]**, 8/8 | **holds** (R20: -41.5 ms kernel, -45 ms phase) |
| `RISC0_METAL_QUERY_PAR=0` | 611.5 ms | 62 [61-63] ms | 1031.5 ms | FRI + tail **+8.5 ms [+7.0, +9.5]**, 8/8 | **holds** (Z11: -11.5 ms in a contended session; 8.5 ms quiet) |

**Groth16 in process** (`g16ab/run.sh`, `g16ab/summary.txt`): `metal_shrink_wrap`, 8 rounds of A1, G14 off, A2.
A2-A1: GPU sum -0.2 [-2.7, +4.0] ms, wrap -0.3 [-2.4, +3.4] ms.

| arm | GPU sum | MSM B2 | sort_w + MSM A/B1/C | wrap (key cached) | first wrap | per-round delta vs mean A | verdict |
|---|---|---|---|---|---|---|---|
| default | 625.5 ms | 150.9 ms | 118.5 ms | 0.652 s | 1.661 s | - | - |
| `RISC0_G16_METAL_FUZZ=1` | 1178.8 [1177.2-1181.7] ms | 440.4 ms | 382.1 ms | 1.20 s | 2.214 s | GPU **+553.8 ms [+551.9, +556.2]**, B2 +289.5, A/B1/C +263.5; 8/8 | **holds** (-47 % shrink_wrap GPU) |

**Budgets (Z12 / M8)** (`summary_4g_memory-merkle.txt`, `summary_8g_memory-merkle.txt`): `BENCH_ONLY=memory-merkle`,
7 rounds each of A1, M8 table off, release auto, A2, R20 off. Load 2.38-4.05. A2-A1: STARK 0.00 [-0.05, +0.04] s
(4 GiB) and -0.01 [-0.07, +0.04] s (8 GiB).

| budget | arm | segment po2 | STARK median [min-max] | segments + recursion | Metal peaks MiB | process peak footprint | per-round delta vs mean A, slower in | verdict |
|---|---|---|---|---|---|---|---|---|
| 4 GiB | default (M8) | 18 | **16.80 [16.78-16.84] s** | 9.09 + 7.71 | 2492 / 1735 / 3942 | 6.76 GB | - | - |
| 4 GiB | `ZKMINER_METAL_PO2_TABLE=m1` | 17 | 29.61 [29.60-29.66] s | 10.67 + 18.95 | 1928 / 1934 / 3942 | 5.34 GB | **+12.84 s [+12.79, +12.84]**, 7/7; footprint -1.43 GB | **holds** (-43 %; M8 measured -44 %). M8's table costs +1.4 GB of footprint at 4 GiB with the release on in both arms. |
| 4 GiB | `RISC0_METAL_RELEASE_SEG_BEFORE_REC=auto` | 18 | 17.01 s | 9.10 + 7.91 | 2492 / **3337** / 3942 | 7.53 GB | rec **+0.200 s [+0.170, +0.205]**, 7/7; identity_p254 +9.7 ms; footprint **+0.76 GB**, 7/7 | **holds**: the worker's forced release is faster and 0.76 GB smaller |
| 4 GiB | `RISC0_METAL_RV32IM_EVAL_FC=0` | 18 | 17.13 s | 9.42 + 7.71 | 2476 / 1719 / 3926 | 6.84 GB | segments **+0.315 s [+0.285, +0.370]**, 7/7 | **holds** |
| 8 GiB | default (M8) | 19 | **12.03 [12.00-12.07] s** | 8.37 + 3.65 | 4784 / 3096 / 5347 | 8.74 GB | - | - |
| 8 GiB | `ZKMINER_METAL_PO2_TABLE=m1` | 18 | 16.18 [16.16-16.29] s | 8.47 + 7.71 | 2492 / 2492 / 5347 | 5.92 GB | **+4.155 s [+4.14, +4.27]**, 7/7; footprint -2.82 GB | **holds** (-26 %; M8 measured -26 %). Footprint +2.8 GB with the M8 table. |
| 8 GiB | `RISC0_METAL_RELEASE_SEG_BEFORE_REC=auto` | 19 | 12.10 s | 8.37 + 3.73 | 4784 / **4788** / 5347 | 8.80 GB | rec +0.080 s [+0.075, +0.090], 7/7; identity_p254 +16.8 ms, 7/7; footprint +0.05 GB | **holds** (small) |
| 8 GiB | `RISC0_METAL_RV32IM_EVAL_FC=0` | 19 | 12.33 s | 8.69 + 3.65 | 4768 / 3080 / 5331 | 8.79 GB | segments **+0.315 s [+0.300, +0.325]**, 7/7 | **holds** |

**R21** is a probe with no runtime knob. Its verdict: **NO-GO** for a code-size split of `ef_eval_split`.
- Duplicating the real kernel's code to 2x / 4x footprint costs only +4-5 % / +11-14 %, so a split could gain
  at most ~2-5 % of the kernel, before paying ~1.7 GB per cut of cut-crossing traffic at po2 20.
- Across 84 R18/R19 variants, time follows the per-thread **stack** size: 18-21 KB is fast (~160 ms at po2 18)
  and 32-40 KB is slow (~404 ms). It does not follow code size.
- The follow-ups R21 recommends: a stack-size regime guard with `tools/metal_audit/agx_size.py` (gate at
  ~24 KB), and a stack-slope probe before funding R19's P2.
- Nothing to A/B here. R21's tooling is off the production path.

**Not every sub-knob was A/B'd separately.** G13's `RISC0_G16_GRAPH_MEMO=0`, `RISC0_G16_METAL_WITNESS=fast|ref` and
`RISC0_G16_METAL_DIRECT=0` were only measured together (the all-off arm above). The early-key knob was measured
alone. C13's `RISC0_METAL_REC_CTRL_PERSIST=all` is not a default and was not measured.

## 5c.4 Simulated budgets (`RISC0_METAL_MAX_WORKING_SET_BYTES`)

`BENCH_ONLY=fibonacci` (4 fresh runs per budget) and `BENCH_ONLY=memory-merkle` (the 14 A runs per budget of
5c.3). **Everything proves, wraps (Groth16) and verifies at both budgets.** With G13, every first wrap is at
warm speed: there is no graph wait any more.

| budget | program | segment po2 | STARK | Groth16 wrap | Metal peak MiB (segments / recursion / wrap) | process peak footprint (`time -l`) |
|---|---|---|---|---|---|---|
| 4 GiB | fibonacci | 18 | 0.35-0.37 s | **1.93-1.94 s** (1.10 + 0.83; wave 5b 5.76 s) | 344 / 1449 / 3942 | 4.54-4.56 GB (wave 5b: 5.55) |
| 4 GiB | memory-merkle | 18 (wave 5b: 17) | **16.80 s** (9.09 + 7.71), 8,388,608 cycles, **499K c/s** (wave 5b: 31.11 s, 331K) | 1.93 s (wave 5b: 2.60) | 2492 / 1735 / 3942 | **6.76 GB** (wave 5b: 9.08) |
| 8 GiB | fibonacci | 19 | 0.35 s | 1.94-1.95 s (wave 5b: 5.79) | 344 / 1449 / 5347 | 4.62 GB (wave 5b: 5.54) |
| 8 GiB | memory-merkle | 19 (wave 5b: 18) | **12.03 s** (8.37 + 3.65), 7,864,320 cycles, **654K c/s** (wave 5b: 17.05 s, 492K) | 1.94 s (wave 5b: 2.60) | 4784 / 3096 / 5347 | 8.74 GB (wave 5b: 9.67) |
| 28.1 GiB (default) | memory-merkle | 20 | 9.23 s, 795K c/s | 1.80 s first / 1.62 s warm | 9420 / 9420 / 5347 | 14.17 GB (BENCH_ONLY) |

- At 4 GiB the wrap peaks at 3942 MiB, 154 MiB under the budget (the early key does not load there). At 8 GiB
  it peaks at 5347 MiB (early key), 2.8 GiB under.
- At 4 GiB recursion is now 46 % of memory-merkle's STARK (7.7 of 16.8 s; wave 5b 64 %).
- At both budgets the process footprint is lower than in wave 5b, although the segment po2 is one higher. This
  is the forced release (M8 / Z12) plus G13. M8's table alone raises it (+1.4 GB at 4 GiB, +2.8 GB at 8 GiB
  against po2 17 / 18 with the release on; 5c.3).
- Below a 20 GiB limit, shrink_wrap is 0.83 s, not 0.69 s: G13 picks the `fast` witness evaluator, not `par`.

## 5c.5 Open items from this baseline

- **No wave-5c knob fails to hold.** Every default measured faster than its revert arm beyond the A-A spread,
  or (Z12 / M8 release) is both faster and smaller.
- **Segment (1.02 s at po2 20): eval_check 60 %, data commit 21 %, accum commit 11 %.** R20 took eval_check
  from 652 to 608 ms. The commits (329 ms) are now the next segment target (R19 P1: `pv_rows_k_z9`, commit
  NTTs). Per R21, eval_check's lever is the stack-size regime, not code size. R21's regime guard (P1) is cheap
  and protects R18/R20.
- **Wrap 1.62 s warm, 1.80 s first.** identity_p254 (918 ms warm, 1.11 s on the first job) is now 57 % of the
  wrap. `p254_rows_w64` is 63 % of its GPU time. On the first job of a process identity_p254 still commits its ctrl
  (C13 open issue: a start-of-worker prewarm). In shrink_wrap, MSM H (210 ms, 34 %) is now the largest stage,
  ahead of B2 (151 ms) and witness_poly + NTT (145 ms).
- **Small Macs**:
  - shrink_wrap runs the `fast` witness evaluator below 20 GiB, which costs +135 ms per wrap against `par` at
    the default budget. A budget-aware `par` (its 580 MB array is the reason for the gate) would recover it.
  - M8's footprint rise at 8 GiB (8.74 GB at po2 19 against 5.92 GB at po2 18, both with the release) still
    needs confirming on a real 16 GB Mac (M8 open issue).
- **Cold start ~9.6 s** (rv32im+zkp and recursion PSOs; unchanged). With a cold OS cache the R20 prewarm does
  not finish before a short job ends, so that job runs the plain eval kernel (R20 open issue 3).
- **Recursion**: 37.3 s of the 189 s STARK at po2 20 (~135 ms per lift/join), 46 % of the STARK at 4 GiB.
- Not measured here: 8-10 and 16-20-core parts (PLAN 9.1; R18/R20's regime, R21's footprint sensitivity, K9's
  chunk crossover), a real 16 GB Mac, `comprehensive_zkvm_test`, CPU prover baselines. G13's separate
  sub-knobs were not measured one by one.

# Wave 6

Baseline after the wave-6 merges: Z14, Z15, Z16, Z17, C14, C15, C16, G15, R22, R23, K10 (risc0; Z13 was a
NO-GO with nothing merged). zk-miner is unchanged since wave 5c: M9 (the worker prewarm that would call C16's
`prewarm_p254_ctrl` and Z16's `metal_prewarm_segment_po2`) is **not** merged, so neither prewarm runs here.

| | |
|---|---|
| Date | 2026-11-14, 02:44-03:33 local time |
| Machine | Apple M5 Max, 40-core GPU, 36 GB unified memory (28.1 GiB recommended working set), macOS 27.2.1 (26B131) |
| risc0 | `zkminer/metal` @ `ef90a0d1e13fc66cd80bd25d32bf10f58abed8cd` (Z17 merge record) |
| zk-miner | `metal` @ `0ca7ec9f346703f976487b30e93c63a8c74b33f8` (as wave 5c; clean tree, path-patched to the risc0 checkout above) |
| Worker | `cargo build --release -p zkminer-prove-risc0 --features metal --bin zkminer-prove-risc0-metal`, installed to `~/.zkminer/provers/zkminer-prove-risc0-metal` (sha256 `b27efa9e…a322`; the wave-5c binary is kept beside it as `.w5c-bak`) |
| Benchmark command | `target/release/zkminer-prove-risc0-metal --benchmark` (full suite; `BENCH_ONLY=<name>` for one program) under `/usr/bin/time -l`, `RUST_LOG=info` |
| Machine state | Every GPU command ran alone under `bin/gpu-run` (heavy lock). Other agents' short probes ran between some of this run's commands; no other prover ran during a timed command. Load averages 1.5-4.9 (each log records its start / end load). |
| Logs | `/Users/user/zkminer/baseline-w6/` (`full/`, `cold/`, `phab/`, `wab_def/`, `bud_4g/`, `bud_8g/`, `bud/`, `prof/`; drivers and parsers in `scripts/`) |

Before the timed runs, one `RISC0_METAL_RV32IM_EVAL_FC=wait` memory-merkle run per budget (`warm/`) checked that
the specialized eval PSOs were in the archive (all hits), and one run per revert arm warmed that arm's PSOs
(R22's nested `ef_eval_split_fc_nest` for domain 2^22 took 12.3 s once) and G15's version-1 coefficient cache.

## 6.1 Worker `--benchmark` (full suite, po2 20 segments)

Three full default runs (`full1-3.log`), interleaved with two `RISC0_METAL_EXEC_STREAM=0` runs (6.3). Load
2.4 -> 2.1, 2.1 -> 1.8, 1.9 -> 1.9. Every run proved, wrapped and verified every program.

| program | cycles | wave 5c (s) | wave 6 runs 1 / 2 / 3 (s) | change | RTX 4090 (s) | Metal / 4090 |
|---|---|---|---|---|---|---|
| fibonacci | 32,768 | 0.37 / 0.36 | 0.31 / 0.31 / 0.31 | -0.05 s | 0.48 | 0.65x |
| sha256-chain | 46,137,344 | 56.37 / 56.09 | 50.38 / 49.88 / 50.26 | -10.7 % | 35.82 | 1.40x |
| ecdsa-verify | 65,011,712 | 78.81 / 78.63 | 69.95 / 70.31 / 70.15 | -10.9 % | 47.65 | 1.47x |
| bigint-mul | 27,525,120 | 34.08 / 33.83 | 30.20 / 30.26 / 30.34 | -10.9 % | 19.77 | 1.53x |
| memory-merkle | 7,340,032 | 9.18 / 9.12 | 8.22 / 8.22 / 8.22 | -10.2 % | 5.70 | 1.44x |
| chacha-mix | 8,404,992 | 10.68 / 10.68 | 9.57 / 9.58 / 9.55 | -10.4 % | 6.67 | 1.43x |
| **all (sum of cycles / sum of STARK time)** | 154,451,968 | 189.49 / 188.71 s, 818K c/s | 168.63 / 168.56 / 168.83 s, **916K / 916K / 915K c/s** | **+12.0 % c/s** | 116.09 s, 1.33M c/s | **1.45x** |

- The RTX 5090 (~2.3M c/s) is **2.51x** this Mac (wave 5c: 2.82x). The 4090 gap fell from 1.63x to 1.45x.
- Segments / recursion (run 1): 139.2 + 29.5 s (wave 5c 152.2 + 37.3 s): segments -8.6 %, recursion -21 %.
  A worker segment is 0.974 s (memory-merkle 6.82 s / 7; wave 5c 1.06 s). In a `RISC0_VERBOSE=1` memory-merkle
  run (`prof/mm_verbose.log`) the steady segments take 931-935 ms; the first takes 1187 ms.
- Recursion per proof (same log): steady lift 93.5 ms, join 99.8-100.9 ms (wave 5c ~135 ms per lift/join
  averaged); memory-merkle's 13 proofs take 1.40 s (wave 5c 1.75 s).

**Groth16 wrap** (first and warm reported separately):

| wrap | identity_p254 | shrink_wrap | total |
|---|---|---|---|
| warm: every wrap after the first in a process (15 samples, runs 1-3) | **732-753 ms** (median 735; wave 5c 918-922) | 611-623 ms (wave 5c 689-701) | **1.34-1.38 s** (median 1.35; wave 5c 1.61-1.62) |
| first wrap of a `--benchmark` process (fibonacci) | 894.6-898.0 ms | 612-617 ms | **1.51 s** (wave 5c 1.80-1.81) |
| first wrap of a fresh `BENCH_ONLY=fibonacci` worker (13 runs, `cold/warm*`, `cold/coldos*`) | 895-909 ms | 615-623 ms | **1.52-1.53 s** (wave 5c 1.80-1.81) |
| first wrap of a fresh `BENCH_ONLY=memory-merkle` worker (14 A runs of 6.3) | 890-895 ms | 610-617 ms | 1.50-1.51 s |

- **The warm wrap (1.35 s) is now 0.65 s faster than the RTX 4090's (~2.0 s); the first wrap (1.51 s) is 0.5 s
  faster.**
- The first wrap is still ~160 ms slower than a warm one: identity_p254 commits its ctrl group on the first job
  (`ctrl=157.2ms(Committed)`, `prof/g16.log`). C16's `prewarm_p254_ctrl` would remove it but needs zk-miner M9.

**First job of a fresh worker** (`BENCH_ONLY=fibonacci`, process wall from `time -l`): **1.92-1.99 s** (13 fresh
runs; wave 5c 2.29-2.32 s).

**Memory.**

| | wave 5c | wave 6 |
|---|---|---|
| process peak footprint, full suite (`time -l`) | 15.67 / 15.67 GB | **15.12 / 15.13 / 15.14 GB** |
| `--benchmark` process lifetime peak | 14942 / 14942 MiB | 14422 / 14430 / 14438 MiB |
| CPU user time of the full suite | 29.0 / 29.0 s | 28.0 / 28.4 / 28.3 s |
| wall time of the full suite | 199.6 / 198.7 s | **177.1 / 177.1 / 177.3 s** |

Metal peaks per phase at po2 20 (MiB, segments / recursion / wrap): memory-merkle 9416 / 9416 / 5377, bigint-mul
(largest) 10217 / 10217 / 5377, chacha-mix 9508 / 9508 / 5377, fibonacci 344 / 1477 / 5377. These are the wave-5c
peaks plus 30 MiB in the wrap and fibonacci's recursion column (5347 -> 5377, 1449 -> 1477).

## 6.2 Cold start

`BENCH_ONLY=fibonacci`; `cold/`. Load 1.5-4.8.

| | pipeline build | process wall (`time -l`) | first wrap |
|---|---|---|---|
| warm (8 runs, interleaved with the next two rows) | 0.1 s | **1.92-1.99 s** | 1.52 s |
| cold OS shader cache (`MTL_SHADER_CACHE_SIZE=0`), archives present (5 runs; the wave-5c method) | **0.1 s** (all archive hits) | **1.93-1.99 s** (wave 5c 11.68-12.12 s) | 1.52-1.53 s |
| same, `RISC0_METAL_ARCHIVE_ZKP=0 RISC0_METAL_ARCHIVE_REC=0` (Z16 off, 3 runs) | 7.9-9.2 s (rv32im+zkp 7.9-9.2, recursion 7.8-9.1) | 9.71-11.04 s | 1.51 s |
| first install: `MTL_SHADER_CACHE_SIZE=0` and an empty `RISC0_METAL_ARCHIVE_DIR` (3 runs) | **65.5-84.8 s** (rv32im+zkp 65.5-84.8: zkp 72 PSOs 14.3 s, rv32im 13 PSOs 60.3 s; recursion, keccak, groth16 27-32 s each in parallel) | 67.3-86.6 s | 1.51-1.52 s |

- **Z16 holds: a cold OS shader cache now costs nothing** (pipelines 0.1 s; wave 5c 9.5-9.9 s). Z16 off gives the
  wave-5c behaviour back (7.9-9.2 s).
- A first install (no archive anywhere) takes 65-85 s, dominated by rv32im's 13 PSOs (~60 s). Shipping Z16's
  offline archives (`RISC0_METAL_AOT_DIR`) is the remaining fix (Z16 packaging open issue).

## 6.3 Per-phase breakdown (quiet machine)

### po2 20 rv32im segment
`risc0-circuit-rv32im` release lib test `prove_po2_sweep`, `R11_PO2S=20 R11_CPU=0 R11_REPS=4`: the A and A2 arms
of `phab/` (14 processes, per-process median of reps 1-3; load 2.2-4.4). Every seal verifies (168 / 168 across
`phab/`). Peak GPU allocation 9303 MiB (unchanged).

| phase | wave 5c ms | wave 6 ms | share |
|---|---|---|---|
| preflight + alloc | 11 | 11 | 1.2 % |
| witgen | 8 | 6 | 0.6 % |
| data commit | 217 | **210 [209-210]** | 22.5 % |
| accum | 9 | 8 | 0.9 % |
| accum commit | 112-113 | **109** | 11.7 % |
| eval_check (flat + hoisted, specialized) | 608 [604-617] | **540 [533-544]** | 57.9 % |
| FRI + tail | 53-54 | **50 [49-50]** | 5.4 % |
| **segment** | 1020 [1016-1029] | **933 [926-937]** (-8.5 %) | |

### identity_p254 + shrink_wrap (worker, `BENCH_ONLY=fibonacci`, `RISC0_VERBOSE=1 RISC0_G16_METAL_PROFILE=1`, `prof/g16.log`)
- **identity_p254, first job** (880 ms in the prover; 896 ms in the wrap): preflight 11.0, witgen 24.1,
  ctrl 157.2 (committed), data commit 424.5, accum 5.1, accum commit 95.3, check commit 122.1, FRI 31.8 ms.
  GPU (`prof/p254.log`, `RISC0_METAL_REC_PROFILE=1 RISC0_METAL_PROFILE=1`): 827.6 ms, `p254_rows_w64` 548.0 ms
  (66 %, 5 calls), `p254_fold_w64` 189.3 ms (23 %, 31 calls), C16's `p254_fold4_w64` 23.5 ms (85 calls) and
  `p254_rows4_w64` 3.2 ms (2 calls). Wave 5c first job: GPU 1047.5 ms (rows 653.3, fold 313.9).
- **identity_p254, later jobs**: 732-753 ms in the worker (C13's kept ctrl group; ~157 ms less than the first).
- **shrink_wrap** (one wrap, key early-loaded, `par` witness): GPU ms zero + witness_poly 39.9, NTT 82.8,
  sort_h 12.6, MSM H 174.9, sort_w 6.2, MSM A / B1 / C 36.4 / 36.6 / 36.3, MSM B2 150.4, **sum 576.0**
  (wave 5c 625.5); wrap 611 ms (wave 5c 692). MSM H (188 ms with its sort) and B2 (150 ms) are now 59 % of the GPU sum.
  Device allocation 3941.7 MiB.

## 6.4 Wave-6 revert knobs: quiet interleaved A/B

Every arm proved, wrapped and verified: worker 56 / 56 (default) + 40 / 40 (budgets) processes with every Groth16
receipt `verified`; rv32im 42 / 42 sweep processes, 168 seals verified; 5 / 5 full suites.

**In-process phase A/B** (`phab/`, `scripts/phab.sh`, `scripts/phana.py`): `prove_po2_sweep` po2 20, 7 rounds of A,
R22, R23, Z14, Z15, A2 (start rotated by 2 each round), each arm against the mean of its round's A and A2. Load
2.2-4.4. A2-A: eval_check 0.0 [-10, +1] ms, data commit 0 [-1, +1], FRI + tail 0 [-1, +1], segment 0.0 [-10, +1] ms.

| item | knob (off) | phase | on (default) | off | per-round delta (off - on), off slower in | verdict |
|---|---|---|---|---|---|---|
| R22 + R23 (flat chain, hoist) | `RISC0_METAL_RV32IM_EVAL_FLAT=0` (nested chain; hoist exists only in the flat form) | eval_check | 540 ms | 601 [600-607] ms | **+61.5 ms [+61.0, +69.0]**, 7/7 | **holds** (the nested arm equals wave 5c's 608 ms) |
| R23 hoist | `RISC0_METAL_RV32IM_EVAL_HOIST=0` (R22 flat) | eval_check | 540 ms | 561 [558-572] ms | **+23.0 ms [+18.0, +32.5]**, 7/7 | **holds** (R23 claimed -17 to -20 ms). R22 alone is therefore ~-38 ms (R22 claimed ~-38). |
| Z14 commit fusion | `RISC0_METAL_COMMIT_FUSE=0` | data commit; accum commit; segment | 210; 109; 933 ms | 216; 112; 947 ms | data +6.0 [+6, +7], accum commit +3.0, witgen +2.0, accum +1.0; **segment +16.0 ms [+12.0, +19.5]**, 7/7 | **holds** (Z14 claimed -16 ms) |
| Z15 tail (lazy eu, fused mix, sync trim) | `RISC0_METAL_EU_LAZY=0 RISC0_METAL_MIX_FUSED=0 RISC0_METAL_SYNC_TRIM=0` | FRI + tail; segment | 50; 933 ms | 53.5; 938 ms | FRI + tail **+3.5 ms [+3.0, +4.5]**, 7/7; segment +4.0 [+1.0, +13.0], 7/7 | **holds** (Z15 expected ~-5 ms; measured -4) |

**Worker A/B, default budget** (`wab_def/`, `scripts/wab.sh`, `scripts/wana.py`): `BENCH_ONLY=memory-merkle
--benchmark`, 7 rounds of A1, C15, C14, C16, G15, Z17, RV, A2 (start rotated by 3), each arm against the mean of its
round's A1 and A2. Load 1.6-2.6. A2-A1: STARK +0.04 [-0.06, +0.08] s, recursion 0.00 [-0.01, 0.00] s,
identity_p254 -1.3 [-3.5, +1.9] ms, shrink_wrap -0.7 [-6.5, +3.1] ms, footprint +0.003 GB.

| item | knob (off) | phase | on (default) median | off median | per-round delta (off - on), off slower in | verdict |
|---|---|---|---|---|---|---|
| C15 recursion eval regroup | `RISC0_METAL_REC_EC_REGROUP=0` | recursion (13 proofs); identity_p254 | 1.41 s; 892 ms | 1.63 s; 910 ms | rec **+0.220 s [+0.220, +0.235] (~17 ms per proof)**, 7/7; identity_p254 +17.3 ms, 7/7 | **holds** (C15 claimed -17 ms per proof) |
| C14 GPU WOM sort | `RISC0_METAL_REC_GPU_SORT=0` | recursion | 1.41 s | 1.49 s | rec **+0.080 s [+0.070, +0.090] (~6 ms per proof)**, 7/7; identity_p254 +4.2 ms, 7/7 | **holds** (C14 expected -4 / -6 ms per lift / join; it did not resolve on a shared machine) |
| C16 compact + 4-lane p254 kernels | `RISC0_METAL_P254_KERNELS=w64u RISC0_METAL_P254_LANES=0` | identity_p254 (first job) | 892.6 ms | 1083.1 ms | **+190.7 ms [+189.8, +192.0]**, 7/7; wrap +0.19 s | **holds** (C16 expected ~-160 ms warm). Warm identity_p254 is 735 ms (C16 expected ~760). Metal peaks unchanged. |
| G15 shrink_wrap kernels + coefficient order | `RISC0_G16_METAL_G15=0 RISC0_G16_METAL_COEFF_ORDER=natural` | shrink_wrap | 613 ms | 690 ms | **+76.8 ms [+72.9, +80.8]**, 7/7 | **holds** (G15 expected ~-75 ms) |
| wave-6 rv32im items together (R22, R23, Z14, Z15) | the four knob sets of the phase A/B | segments (7); footprint | 6.83 s; 12.95 GB | 7.39 s; 14.18 GB | segments **+0.555 s [+0.515, +0.585] (+79 ms per segment)**, 7/7; recursion +0.03 s; **footprint +1.23 GB**, 7/7 | **holds**; the phase sum is +81.5 ms. The 1.2 GB footprint saving was not attributed to one item. Metal peak 16 MiB lower with the knobs off. |
| Z17 executor stream | `RISC0_METAL_EXEC_STREAM=0` | segments | 6.83 s | 6.87 s | +0.025 s [-0.025, +0.070], 5/7 | **not resolved on memory-merkle** (its executor is 42 ms; A2-A1 spread is ±0.08 s). See the full-suite A/B below. |

**Z17 on the full suite** (`full/`): `z17off1`, `full3`, `z17off2` run back to back after `full1-2`.
STARK 169.63 / 169.03 s off against 168.63 / 168.56 / 168.83 s on; segments 140.16 / 139.51 against 139.16 / 139.06
/ 139.38 s. Off is +0.6 s (+0.35 %) by median, slower than each neighbouring A run; Z17 estimated ~-0.85 s.
**Holds, weakly** (2 off runs; the margin is about twice the A-A spread of the three A runs, 0.27 s).

**Not measured separately:** K10 (keccak; 0 % of this benchmark, no keccak requests), C14's opt-in staging
(`RISC0_METAL_REC_STAGE=1`, default off), Z14's `RISC0_METAL_PREFILL=0` alone, Z15's three knobs one by one,
G15's single `RISC0_G16_METAL_G15_OFF=` switches, C16's `RISC0_METAL_RELEASE_SEG_BEFORE_REC` default change,
R22's regime guard (no speed effect by design; the log shows `ef_eval_split: native stack 3984 B ... fast regime`).

## 6.5 Simulated budgets (`RISC0_METAL_MAX_WORKING_SET_BYTES`)

`BENCH_ONLY=fibonacci` (3 fresh runs per budget, `bud/`) and `BENCH_ONLY=memory-merkle` (5 rounds of A1, G15, RV,
A2 per budget, `bud_4g/`, `bud_8g/`; load 1.6-2.5). **Everything proves, wraps and verifies at both budgets.**

| budget | program | segment po2 | STARK | Groth16 wrap (identity_p254 + shrink_wrap) | Metal peak MiB (segments / recursion / wrap) | process peak footprint |
|---|---|---|---|---|---|---|
| 4 GiB | fibonacci | 18 | 0.31 s | **1.64 s** (0.885 + 0.750; wave 5c 1.93) | 344 / 1477 / 3942 | 4.56-4.58 GB (wave 5c 4.54-4.56) |
| 4 GiB | memory-merkle | 18 | **14.45 [14.43-14.48] s** (8.29 + 6.16), 8,388,608 cycles, **581K c/s** (wave 5c 16.80 s, 499K) | 1.63-1.64 s | 2492 / 1828 / 3942 | **5.50 GB** (wave 5c 6.76) |
| 8 GiB | fibonacci | 19 | 0.31 s | 1.65 s (0.890-0.895 + 0.749-0.755; wave 5c 1.94-1.95) | 344 / 1477 / 5377 | 4.62-4.63 GB (wave 5c 4.62) |
| 8 GiB | memory-merkle | 19 | **10.58 [10.53-10.61] s** (7.65 + 2.93), 7,864,320 cycles, **743K c/s** (wave 5c 12.03 s, 654K) | 1.65-1.66 s | 4784 / 3096 / 5377 | **7.58 GB** (wave 5c 8.74) |
| 28.1 GiB (default) | memory-merkle | 20 | 8.24 s, 891K c/s | 1.51 s first / 1.35 s warm | 9420 / 9420 / 5377 | 12.95 GB (BENCH_ONLY; wave 5c 14.17) |

Budget A/B (per-round delta against the mean of A1 and A2; A2-A1 STARK +0.02 [-0.06, +0.07] s at 4 GiB, -0.01
[-0.05, +0.02] s at 8 GiB):

| budget | arm | delta, off slower in | verdict |
|---|---|---|---|
| 4 GiB | G15 off | shrink_wrap **+76.8 ms [+74.2, +80.1]**, 5/5 | holds (with the `fast` evaluator) |
| 4 GiB | rv32im items off | segments **+0.640 s [+0.595, +0.715]**, recursion +0.140 s, STARK +0.775 s, 5/5; footprint **+1.32 GB** | holds |
| 8 GiB | G15 off | shrink_wrap **+75.5 ms [+74.7, +81.4]**, 5/5 | holds |
| 8 GiB | rv32im items off | segments **+0.635 s [+0.570, +0.665]**, recursion +0.070 s, STARK +0.705 s, 5/5; footprint **+1.16 GB** | holds |

- Below 20 GiB, shrink_wrap is 0.75 s against 0.61 s at the default budget (`fast` evaluator, as in wave 5c).
- Recursion is 43 % of memory-merkle's STARK at 4 GiB (6.16 of 14.45 s; wave 5c 46 %).

## 6.6 Open items from this baseline

- **No wave-6 knob fails to hold.** Z17 is the weakest (+0.35 % of the full-suite STARK with 2 off runs; not
  resolved on memory-merkle). Everything else beats its A-A spread in every round.
- **STARK 916K c/s, 1.45x the RTX 4090's time, 2.51x below the RTX 5090.** A po2 20 segment is 933 ms: eval_check
  58 %, data commit 22.5 %, accum commit 11.7 %. The commits (319 ms) are the next segment target after Z13's NO-GO.
- **First-job wrap**: still +160 ms over warm (identity_p254's ctrl commit). zk-miner M9 (calling
  `prewarm_p254_ctrl` and `metal_prewarm_segment_po2` at worker start) would remove it and the first segment's
  ~250 ms excess (1187 against 932 ms steady).
- **First install** takes 65-85 s of PSO compiles (rv32im ~60 s). A cold OS cache with archives present is now free.
  Shipping offline archives (Z16 packaging) is the fix.
- The ~1.2 GB footprint saving of the rv32im wave-6 items (6.4) is unattributed; a per-knob footprint A/B
  would place it (Z14's prefill is the likely candidate).
- Not measured here: K10 (no keccak in the benchmark), 8-10 and 16-20-core parts, a real 16 GB Mac,
  `comprehensive_zkvm_test`, CPU prover baselines.

# Wave 7

Baseline after the wave-7 merges. risc0: W7-A1, W7-Z1 (zkp), W7-R2, W7-R1 (rv32im), W7-C1 (recursion),
W7-V1, W7-V2 (zkvm). **Not merged**: W7-T1 (memguard / NTT split check) and W7-G1 (Groth16 graph release),
both held back for fix rounds (`notes/W7-merge-zkp.md`, `notes/W7-merge-groth16.md`), so G1's wrap-phase
memory saving is not in these numbers. zk-miner: M9 (first-job prewarm, QoS, recycle policy), W7-M10
(owner-active tier, off by default), W7-M11 (power state, tier table `w7`, packaged AOT, image-id pass-through).

| | |
|---|---|
| Date | 2026-11-15, 14:21-15:14 local time |
| Machine | Apple M5 Max, 40-core GPU, 36 GB unified memory (28.1 GiB recommended working set), macOS 27.2.1 (26B131) |
| risc0 | `zkminer/metal` @ `e841e5263261347a0669f45b1a66cb6dc660076c` (zkvm unit merge record) |
| zk-miner | `metal` @ `2bf2b24712f58c90a1329ac4e4f199e22e15bb24` (W7-M11 merge; clean tree, path-patched to the risc0 checkout above) |
| Worker | `cargo build --release -p zkminer-prove-risc0 --features metal --bin zkminer-prove-risc0-metal`, installed to `~/.zkminer/provers/zkminer-prove-risc0-metal` (sha256 `731bcd24…d1d1`; the wave-6 binary is kept beside it as `.w6-bak`). Installed without `AOT_SRC`, so no `metal-aot/` directory (the default install) |
| Method | as wave 6: `--benchmark` under `/usr/bin/time -l`, `RUST_LOG=info`, every GPU command alone under `bin/gpu-run` (heavy lock, `--no-budget`; budgets set with `RISC0_METAL_MAX_WORKING_SET_BYTES` inside the command). New for this wave: `RISC0_METAL_ARCHIVE_GC=0` on every run (W7-Z1: a shared machine must not GC other agents' archives) |
| Machine state | No other cargo / test / prover process ran during this baseline (checked with `ps` before starting); load averages 1.6-4.3 (each log records start / end load). No memguard stop or kill in any log |
| Logs | `/Users/user/zkminer/scratch/w7/baseline/` (`full/`, `cold/`, `phab_def/`, `phab_8g/`, `phab_8g_aip/`, `phab_4g/`, `wab_def/`, `wab_4g/`, `wab_8g/`, `bud/`, `sim/`, `r2tab/`, `keccak/`, `prof/`, `warm/`; drivers and parsers in `scripts/`, adapted from `baseline-w6/scripts`) |

Warm-up before the timed runs: one `BENCH_ONLY=fibonacci` run and one `RISC0_METAL_RV32IM_EVAL_FC=wait`
memory-merkle run at the default, 4 GiB and 8 GiB budgets (`warm/`; every specialized eval PSO was an archive hit).

## 7.1 Worker `--benchmark` (full suite, po2 20 segments)

Three full default runs back to back (`full/full1-3.log`). Load 2.6 -> 2.4, 2.3 -> 1.9, 1.9 -> 2.2. Every run
proved, wrapped and verified every program (18 / 18 `verified`).

| program | cycles | wave 5c (s) | wave 6 runs 1 / 2 / 3 (s) | wave 7 runs 1 / 2 / 3 (s) | RTX 4090 (s) | Metal / 4090 |
|---|---|---|---|---|---|---|
| fibonacci | 32,768 | 0.37 / 0.36 | 0.31 / 0.31 / 0.31 | 0.23 / 0.24 / 0.24 | 0.48 | 0.50x |
| sha256-chain | 46,137,344 | 56.37 / 56.09 | 50.38 / 49.88 / 50.26 | 50.03 / 49.89 / 49.96 | 35.82 | 1.39x |
| ecdsa-verify | 65,011,712 | 78.81 / 78.63 | 69.95 / 70.31 / 70.15 | 70.16 / 70.06 / 70.15 | 47.65 | 1.47x |
| bigint-mul | 27,525,120 | 34.08 / 33.83 | 30.20 / 30.26 / 30.34 | 30.10 / 30.16 / 30.13 | 19.77 | 1.52x |
| memory-merkle | 7,340,032 | 9.18 / 9.12 | 8.22 / 8.22 / 8.22 | 8.19 / 8.17 / 8.18 | 5.70 | 1.43x |
| chacha-mix | 8,404,992 | 10.68 / 10.68 | 9.57 / 9.58 / 9.55 | 9.51 / 9.52 / 9.52 | 6.67 | 1.43x |
| **all (sum of cycles / sum of STARK time)** | 154,451,968 | 189.49 / 188.71 s, 818K c/s | 168.63 / 168.56 / 168.83 s, 916K c/s | **168.22 / 168.04 / 168.18 s, 918K / 919K / 918K c/s** | 116.09 s, 1.33M c/s | **1.45x** |

- **STARK is unchanged from wave 6** (median 168.18 against 168.63 s, -0.27 %; separate sessions, not an
  interleaved A/B, so this is "no change"). Wave 7 was a memory / robustness wave: no item had a default-budget
  STARK claim. fibonacci's -0.07 s is M9's first-job prewarm (7.4).
- **Goal gap**: the program goal is >= 1.197M c/s, i.e. a full-suite STARK <= 129.0 s. This is 918K c/s:
  **39.2 s (23.3 %) of STARK time still to remove**. RTX 4090 1.45x this Mac's time; RTX 5090 (~2.3M c/s) 2.50x.
- Segments / recursion: 139.38 / 139.14 / 139.33 s + 28.82 / 28.87 / 28.85 s (wave 6 run 1: 139.2 + 29.5 s).

**Groth16 wrap** (first and warm reported separately):

| wrap | identity_p254 | shrink_wrap | total |
|---|---|---|---|
| warm: every wrap after the first in a process (15 samples, runs 1-3) | **734-743 ms** (median 739.5; wave 6 732-753) | 612-627 ms (median 620) | **1.35-1.37 s** (wave 6 1.34-1.38) |
| first wrap of a `--benchmark` process (fibonacci) | 744.4 / 747.7 / 751.1 ms (wave 6 894.6-898.0) | 611-619 ms | **1.36 s** (wave 6 1.51) |
| first wrap of a fresh `BENCH_ONLY=fibonacci` worker (10 runs, `cold/warm*`, `cold/coldos*`) | 745-750 ms | 616-618 ms | **1.36-1.37 s** (wave 6 1.52-1.53) |

- **The first wrap now equals a warm one** (M9's prewarm calls C16's `prewarm_p254_ctrl`: "identity_p254 ctrl
  kept (98 MiB) in 216 ms" at start). Warm wrap 1.36 s against the 4090's ~2.0 s.

**First job of a fresh worker** (`BENCH_ONLY=fibonacci`, `time -l` wall, warm OS cache): **1.93-1.95 s** (5 runs;
wave 6 1.92-1.99 s). It includes M9's synchronous first-job prewarm (0.23 s: po2 20 eval PSO 0.04 s, p254 ctrl 0.22 s).

**Memory.**

| | wave 6 | wave 7 |
|---|---|---|
| process peak footprint, full suite (`time -l`) | 15.12 / 15.13 / 15.14 GB | **14.69 / 14.69 / 14.69 GB** (-0.44 GB) |
| `--benchmark` process lifetime peak | 14422-14438 MiB | 14007-14013 MiB |
| CPU user time of the full suite | 28.0-28.4 s | 24.8-24.9 s |
| wall time of the full suite | 177.1-177.3 s | 176.6-176.8 s |

Metal peaks per phase at po2 20 (MiB, segments / recursion / wrap): memory-merkle 9405 / 9405 / 5376, bigint-mul
(largest segments) 9580 / 8958 / 5376 (wave 6 10217 / 10217), chacha-mix 9370-9386 / 8249 / 5376, ecdsa-verify
9466 / 9452 / 5376, fibonacci 339 / 1476 / 5376. Recursion peaks are now below the segment peaks for bigint and
chacha (the W7-A1 process-wide pool cap evicts across pools; 7.4).

## 7.2 Cold start

`BENCH_ONLY=fibonacci`; `cold/`. 5 interleaved pairs warm / cold OS cache, then the other rows. Load 1.9-4.1.

| | pipelines | first-job prewarm | process wall (`time -l`) | first wrap |
|---|---|---|---|---|
| warm (5 runs) | 0.1 s | 0.23 s | **1.93-1.95 s** | 1.36-1.37 s |
| cold OS shader cache (`MTL_SHADER_CACHE_SIZE=0`), archives present (5 runs, the wave-5c / 6 method) | 0.1 s (all hits) | **6.70 s** (po2 20 `ef_eval_split_fc`, "1 hits, 0 misses" but 6.7 s) | **7.95-8.42 s** (wave 6 1.93-1.99 s) | 1.36-1.37 s |
| same, `ZKMINER_METAL_PREWARM=0` (2 runs) | 0.1 s | none | 1.95-1.96 s | 1.53 s (identity_p254 904-907 ms) |
| first install with an offline set: `MTL_SHADER_CACHE_SIZE=0`, empty `RISC0_METAL_ARCHIVE_DIR`, `RISC0_METAL_AOT_DIR` = this Mac's set from `W7-Z1/aot` installed by zk-miner's `install-metal-aot.sh` (11 archives, 165 MiB; 3 runs) | 0.1 s | 5.65-6.19 s | **7.40-7.95 s** | 1.36-1.37 s |
| first install, no offline set (the default install; 2 runs) | **71.6-81.1 s** (rv32im+zkp 71.6-81.1; recursion 25-31, keccak 16-33, groth16 23-31 in parallel) | 18.8 s | **92.0-101.5 s** (wave 6 67.3-86.6 s) | 1.36-1.37 s |

- **Regression in `--benchmark` with a cold OS shader cache: +6 s of wall** (1.95 -> 8.0-8.4 s). M9's first-job
  prewarm runs synchronously in `--benchmark` and waits for the po2 20 specialized eval PSO, which takes 6.7 s
  when the OS cache is cold even though the runtime archive hits. Wave 6 ran the plain kernel for that short job
  instead. STARK and wrap are unaffected. In the IPC worker the prewarm runs in the background (M9), so this
  costs a fresh worker's first job only if the job arrives within ~6.7 s. Not a default-path cost on a warm
  machine; worth an M9 follow-up (wait only for archive-warm work in `--benchmark`, or measure what the IPC path
  pays).
- **W7-Z1 + W7-M11's offline set cuts a first install from 92-102 s to 7.4-8.0 s**, but only when the worker is
  installed with `AOT_SRC`; the default `mac-install-worker` installs none, so a default first install is still
  ~90-100 s.

## 7.3 Per-phase breakdown (quiet machine)

### po2 20 rv32im segment
`risc0-circuit-rv32im` release lib test `prove_po2_sweep`, `R11_PO2S=20 R11_CPU=0 R11_REPS=4`: A and A2 arms of
`phab_def/` (14 processes, per-process median of reps 1-3; load 1.9-3.0). Every seal verifies.
Peak GPU allocation 9303 MiB (unchanged).

| phase | wave 6 ms | wave 7 ms (A / A2 medians) | share |
|---|---|---|---|
| preflight + alloc | 11 | 12 / 12 | 1.3 % |
| witgen | 6 | 6 / 6 | 0.6 % |
| data commit | 210 | **210 / 210** | 22.5 % |
| accum | 8 | 8 / 8 | 0.9 % |
| accum commit | 109 | **109 / 109** | 11.7 % |
| eval_check (flat + hoisted, specialized) | 540 | **542 [538-552] / 539 [535-544]** | 57.8 % |
| FRI + tail | 50 | 50 / 50 | 5.3 % |
| **segment** | 933 | **936 [932-950] / 933 [930-938]** (unchanged) | |

In the worker (`prof/mm_verbose.log`, `RISC0_VERBOSE=1`) the steady segments take 933.6-936.7 ms and the first
1124 ms (wave 6 1187 ms; M9's prewarm). At the worker's tiers (7.5): po2 20 at 8 GiB 1010-1012 ms (accum
commit 132, data commit 227, FRI + tail 57-58 ms), po2 19 at 4 GiB 506-507 ms (peak 4080 MiB).

### Recursion per proof (same verbose log)
Steady lift 91.8-92.6 ms, join 96.9-97.5 ms (wave 6 93.5 / 99.8-100.9); first pair lift0+lift1 265 ms;
memory-merkle's 13 proofs 1.37 s (wave 6 1.40 s).

### identity_p254 + shrink_wrap (worker, `BENCH_ONLY=fibonacci`, `RISC0_VERBOSE=1 RISC0_G16_METAL_PROFILE=1`, `prof/g16.log`)
- **identity_p254, first job** (744 ms in the prover; 749 ms in the wrap): preflight 10.9, witgen 26.1,
  **ctrl 5.8 (Restored**; wave 6 157.2 Committed), data commit 438.0, accum 5.1, accum commit 95.7, check commit
  122.0, FRI 31.6 ms. The data commit (438 ms, 59 %) is now the whole of what is left.
- **shrink_wrap**: GPU ms zero + witness_poly 43.5, NTT 82.8, sort_h 12.6, MSM H 174.6, sort_w 6.0, MSM A / B1 / C
  36.8 / 37.9 / 37.4, MSM B2 150.6, **sum 582.1** (wave 6 576.0); wrap 619 ms. Device allocation 3941.5 MiB.

## 7.4 Wave-7 revert knobs: quiet interleaved A/B

**Worker A/B** (`scripts/wab.sh`, `scripts/wana.py`): `BENCH_ONLY=memory-merkle --benchmark`, rounds of A1, C1,
A1X, M9, R2, A2 (start rotated by 3 each round), each arm against the mean of its round's A1 and A2. Every one of
the 102 worker runs proved, wrapped and `verified`.

| arm | knob (off) |
|---|---|
| C1 (C14 staging default) | `RISC0_METAL_REC_STAGE=0` |
| A1X (W7-A1 pool cap, pressure drain, system check) | `RISC0_METAL_POOL_GLOBAL=0 RISC0_METAL_PRESSURE_DRAIN=0 RISC0_METAL_SYSTEM_CHECK=0` |
| M9 (first-job prewarm) | `ZKMINER_METAL_PREWARM=0` |
| R2 (in-place accum, early data drop, post-mix coeff drop, prefill cap) | `RISC0_METAL_ACCUM_IN_PLACE=0 RISC0_METAL_DROP_COEFFS_AFTER_MIX=0 RISC0_METAL_DROP_DATA_EARLY=0 RISC0_METAL_PREFILL_CAP=limit8` |

Default budget: 7 rounds, load 1.7-3.3. A2-A1 STARK +0.010 [-0.040, +0.060] s, recursion 0.000 [-0.010, +0.010],
wrap 0.000, footprint -0.000 GB. 4 GiB (po2 18): 5 rounds, load 1.6-3.5, A2-A1 STARK 0.000 [-0.020, +0.010].
8 GiB (po2 19): 5 rounds, load 1.6-4.3, A2-A1 STARK 0.000 [-0.040, +0.020]. Deltas are off - on, median [min, max],
"n/N" = rounds in which off is larger.

| item | budget | on (default) median | delta (off - on) | verdict |
|---|---|---|---|---|
| W7-C1 staging | default | rec 1.39 s | rec **+0.030 [+0.025, +0.035] 7/7** (~2.3 ms per proof); STARK +0.025, 6/7 | **holds** (C1 claimed -29 ms per worker lift/join set on memory-merkle: matches) |
| W7-C1 staging | 4 GiB | rec 6.02 s | rec **+0.135 [+0.125, +0.140] 5/5**; footprint +0.023 GB | **holds** (C1: -138 ms) |
| W7-C1 staging | 8 GiB | rec 2.87 s | rec **+0.060 [+0.055, +0.070] 5/5** | **holds** (C1: -63 ms) |
| W7-A1 (all three) | default | STARK 8.16 s; 13.07 GB | STARK +0.000 [-0.035, +0.030] 3/7; footprint +0.011 GB 7/7; Metal peak +11 MiB | no change at the default budget |
| W7-A1 (all three) | 4 GiB | STARK 13.72 s; 5.165 GB | **STARK +0.550 [+0.535, +0.580] 5/5** (all in segments); **footprint +0.434 GB 5/5**; Metal peak segments 2384 -> 2488, recursion 2183 -> 1910 MiB | **holds** (A1 claimed STARK -0.525 s, footprint -0.32 GB at 4 GiB) |
| W7-A1 (all three) | 8 GiB | STARK 10.05 s; 7.246 GB | **STARK +0.390 [+0.345, +0.420] 5/5**; **footprint +0.462 GB 5/5**; Metal peak 4714 / 4321 -> 4780 / 3092 MiB | **holds** (A1 claimed -0.435 s, -0.51 GB) |
| M9 prewarm | default | wrap 1.37 s; STARK 8.16 s | **identity_p254 +150.3 ms [+146.8, +151.6] 7/7** (wrap +0.150 s); **STARK +0.085 [+0.010, +0.110] 7/7** (segments: first-segment PSO); footprint **-0.127 GB** (the kept ctrl group and the specialized PSO cost 0.13 GB) | **holds** (M9: first wrap -150 ms) |
| M9 prewarm | 4 GiB | | identity_p254 +154.9 ms 5/5, STARK +0.085 5/5, footprint -0.033 GB | holds |
| M9 prewarm | 8 GiB | | identity_p254 +150.8 ms 5/5, STARK +0.070 5/5, footprint -0.131 GB | holds |
| W7-R2 (all four) | default / 4 GiB / 8 GiB | | STARK +0.005 / -0.005 / +0.010 s, footprint 0.000 GB at all three | no change at the worker's tiers (po2 20 / 18 / 19): R2 changes only po2 19 at 4 GiB and po2 20 at 8 GiB, which the default tier table `w7` does not select |

**In-process phase A/B for W7-R2 at its own shapes** (`scripts/phab.sh`, `prove_po2_sweep`, 7 rounds rotated;
load 1.9-3.8):

| shape | arm | result |
|---|---|---|
| po2 20, default budget (`phab_def/`) | R2 all four off | segment -1.0 [-6.5, +1.5] ms, 1/7 (A2-A -5.0 [-15, +4]): no change, as R2 claims |
| po2 20, 8 GiB (`phab_8g/`) | R2 all four off | **fails, 7/7** ("allocating 1728053248 bytes for evaluated with 6872825856 already allocated would exceed the limit"); A proves in 1012 ms [1004-1020], peak 8129 MiB, seals verify. **Fit claim holds** |
| po2 20, 8 GiB (`phab_8g_aip/`) | `RISC0_METAL_ACCUM_IN_PLACE=0` only | proves; accum commit **+10.5 ms [+9.5, +11.0] 7/7**, FRI + tail +3.0 7/7, **segment +15.0 [+8.5, +20.0] 7/7** (A2-A 0.0 [-6, +7]); peak 8014 vs 8129 MiB. **Holds** (R2: accum commit 134 vs 145 ms) |
| po2 19, 4 GiB (`phab_4g/`) | R2 all four off | **fails, 7/7** (budget exceeded allocating `evaluated`); A 506-507 ms, peak 4080 MiB. **Fit claim holds** |
| po2 19, 4 GiB | `RISC0_METAL_ACCUM_IN_PLACE=0` only | accum commit +5.5 [+5.0, +6.0] 7/7, segment +5.5 [+4.5, +7.0] 7/7 (A2-A +1.0 [-2, +3]); peak 4022 vs 4080 MiB. Holds |

**What R2 buys when the tier table uses it** (`r2tab/`, memory-merkle, 3 interleaved pairs per budget, A = default
table `w7`, R = `ZKMINER_METAL_TIER_TABLE=r2`; load 1.8-3.2):

| budget | A (w7) | R (r2) | change |
|---|---|---|---|
| 4 GiB | po2 18, STARK 13.73-13.74 s (8,388,608 cycles, 611K c/s), footprint 5.16 GB, Metal 2384 / 2183 / 3942 MiB | po2 19, STARK **10.49-10.52 s** (7,864,320 cycles, **750K c/s**), footprint **6.99 GB**, Metal 4076 / 2240 / 3942 MiB | **+23 % c/s, +1.83 GB footprint** |
| 8 GiB | po2 19, STARK 10.05-10.07 s (783K c/s), footprint 7.25 GB | po2 20, STARK **8.60-8.61 s** (7,340,032 cycles, **853K c/s**), footprint **11.77 GB** | +9 % c/s, **+4.52 GB footprint** (does not fit a 16 GB Mac's ~11.2 GiB allowance for the full suite; M11 rejected it) |

**Other items** (no worker A/B here):
- **W7-V2** (keccak): `k5_keccak_metal_e2e`, 4 GiB, `K5_E2E_PO2S=14,17,14,17`, 5 interleaved pairs A vs
  `RISC0_KECCAK_SMALL_BUDGET_RELEASE=0` (`keccak/ab_*`, load 1.9-2.1): both arms prove every request; po2 17
  1.76-1.77 / 1.69-1.71 s in both arms, footprint 5.98 GB in both. **No change on this harness**: V2's release
  fires before a standalone po2 17 proof in the cached-prover path (`keccak_standalone_repeat`), which this e2e
  test does not exercise; that test was not re-run here. Headroom cap: po2 18 is skipped at 4 GiB by design (7.5).
- **W7-Z1 / W7-M11 AOT**: 7.2 (first install 92-102 s -> 7.4-8.0 s with a set). Warm start: no change claimed;
  not re-measured.
- **W7-M10** (enforcement, owner tier: both off by default; respawn-early is IPC-only), **M9 recycle / growth
  net, QoS** (normal launch no change), **W7-M11 image-id pass-through** (after the STARK), **W7-R1 / W7-V1 /
  W7-A1's test-suite footprint** (tests only): not measurable with `--benchmark`; not run.
- **W7-T1, W7-G1**: not merged; nothing to A/B on this tree.

## 7.5 Simulated budgets and hosts

`BENCH_ONLY=fibonacci` (3 fresh runs per budget, `bud/`), memory-merkle (the A1 / A2 arms of 7.4), keccak
(`keccak/k5_*`, 2 runs per budget). **Everything proves, wraps and verifies at both budgets.**

| budget | program | segment po2 | STARK | Groth16 wrap (identity_p254 + shrink_wrap) | Metal peak MiB (segments / recursion / wrap) | process peak footprint |
|---|---|---|---|---|---|---|
| 4 GiB | fibonacci | 18 | 0.24 s | **1.49 s** (0.734-0.736 + 0.749-0.753; wave 6 1.64) | 339 / 1477 / 3942 | 4.53-4.54 GB (wave 6 4.56-4.58) |
| 4 GiB | memory-merkle | 18 | **13.72 [13.71-13.74] s** (7.70 + 6.02), 8,388,608 cycles, **611K c/s** (wave 6 14.45 s, 581K) | 1.49-1.50 s | 2384 / 2183 / 3942 | **5.16-5.17 GB** (wave 6 5.50) |
| 4 GiB | keccak e2e 14,17,18 | 14, 17 (18 skipped below 7084 MiB, V2 cap) | po2 14 0.78-0.83 s, po2 17 1.77 s (prove + lift + verify; no wrap in this test) | - | - | 5.81-5.82 GB |
| 8 GiB | fibonacci | 19 | 0.24 s | 1.50-1.51 s (0.747-0.748 + 0.751-0.760; wave 6 1.65) | 339 / 1477 / 5377 | 4.62-4.63 GB |
| 8 GiB | memory-merkle | 19 | **10.05 [10.03-10.09] s** (7.19 + 2.87), 7,864,320 cycles, **783K c/s** (wave 6 10.58 s, 743K) | 1.50-1.51 s | 4714 / 4321 / 5376 | **7.25 GB** (wave 6 7.58) |
| 8 GiB | keccak e2e 14,17,18 | 14, 17, 18 | 0.77-0.78, 1.76-1.77, 2.93-2.94 s | - | - | 10.22-10.24 GB |
| 28.1 GiB (default) | memory-merkle | 20 | 8.16 s, 899K c/s | 1.36 s first / 1.37 s warm | 9405 / 9405 / 5376 | 13.07 GB (`BENCH_ONLY`; wave 6 12.95) |

- STARK at 4 GiB -5.0 % and at 8 GiB -5.0 % against wave 6, footprint -0.34 / -0.33 GB: W7-A1 (7.4) and W7-C1.
- Wrap at both budgets -0.15 s (M9's kept ctrl group). shrink_wrap is still 0.75-0.76 s below 20 GiB (`fast`
  evaluator) against 0.62 s at the default budget; G1 (not merged) was the item for the wrap-phase memory.
- Recursion is 44 % of memory-merkle's STARK at 4 GiB (6.02 of 13.72 s).

**Tier the worker selects on a simulated host** (`sim/`, memory-merkle, `ZKMINER_METAL_HOST_RAM_BYTES` = host
RAM, `RISC0_METAL_MAX_WORKING_SET_BYTES` = that Mac's HAL limit, `ZKMINER_CUDA_VRAM_BYTES` = the dispatcher's
figure, RAM - 8 GiB as M10 does; tier table `w7`; one run each):

| host | HAL limit / planned figure | po2 chosen | STARK | wrap | Metal peaks MiB | footprint |
|---|---|---|---|---|---|---|
| 16 GB | 10.67 / 8 GiB | 19 | 9.96 s | 1.51 s | 4774 / 4774 / 5376 | 7.32 GB |
| 18 GB | 12 / 10 GiB | 19 (po2 20's full-suite footprint model does not fit 18 GB) | 9.96 s | 1.51 s | 4774 / 4774 / 5376 | 7.30 GB |
| 24 GB | 16 / 16 GiB | 20 | 8.25 s | 1.51 s | 9373 / 8598 / 5376 | 12.29 GB |

Without the dispatcher figure (`--benchmark` plans with the HAL limit) the 18 GB case picks po2 20 at 12 GiB
(8.44 s, 12.29 GB, `sim/h18_mm.log`): the footprint gate only removes rows above M8's, and M8 gives po2 20 from
12 GiB. Not measured: the full suite at 16/18 GB footprints, a real 16 GB Mac.

## 7.6 Open items from this baseline

- **STARK 918K c/s, unchanged from wave 6; 23.3 % (39.2 s of 168.2 s) short of the 1.197M c/s goal.** A po2 20
  segment is 934 ms: eval_check 58 %, data commit 22.5 %, accum commit 11.7 %. Recursion 28.8 s (17 % of STARK).
- **M9 cold-OS-cache regression in `--benchmark`**: +6 s wall (7.2). Measure the IPC path (background prewarm)
  with a cold OS cache and a job that arrives right after start before deciding.
- **The default install ships no AOT set** (7.2): a first install still costs 92-102 s. `mac-install-worker`
  needs an `AOT_SRC` in the release flow.
- **Small Macs**: W7-R2's fits (po2 19 at 4 GiB, po2 20 at 8 GiB) hold and give +23 % / +9 % c/s on memory-merkle,
  but the default tier table does not use them; the 4 GiB po2 19 tier costs +1.83 GB footprint (6.99 GB), which
  W7-G1's graph release (not merged) was meant to offset. Merging G1 and re-deciding the `r2` table is the next
  small-Mac lever.
- W7-A1 carries the budget tiers this wave (-0.39 / -0.55 s STARK and -0.46 / -0.43 GB footprint at 8 / 4 GiB,
  all in segments, as its note claimed); at the default budget it is a no-op for speed.
- Not measured here: W7-V2's standalone-release path (`keccak_standalone_repeat`), IPC-only paths (M9 recycle,
  M10 respawn-early), 8-10 and 16-20-core parts, a real 16 GB Mac, CPU prover baselines.

# 3.1

Baseline of the risc0 3.1 port (PORT_PLAN P31-17) after the phase-A/B/C merges. risc0 v3.1.0 changes the
circuits and the GPU kernels for security (rv32im `RV32IM:v2rev3___`, 484 poly-mix powers, `totalCycles`;
recursion BIT_AND_ELEM canonical decomposition and independent ZK noise; Poseidon254 S-box x^8 -> x^7; Groth16
ceremony v0.2.0, selector `e71e6620`). Wave 7 (3.0) is the reference. Every 3.1-vs-3.0 delta below comes from
an **interleaved A/B in this session** against the frozen wave-7 3.0 builds (the wave-7 worker kept as
`.w7-bak`, and the 3.0 rv32im test binary in the read-only `deps/risc0` target). It is attributed per phase.

| | |
|---|---|
| Date | 2026-11-16, 18:48-19:50 local time |
| Machine | Apple M5 Max, 40-core GPU, 36 GB unified memory (28.1 GiB recommended working set), macOS 27.2.1 (26B131) |
| risc0 | `zkminer/metal-3.1` @ `a0ff0fb751ccf3079549e942b45d0914755ae464` (P31-01..16 + T1 merged) |
| zk-miner | `metal-3.1` @ `0e57ab4d2198b6b15c4e1927c9bd26eea08eee7e` (P31-19 + P31-M9b merged; clean tree, path-patched to the checkout above) |
| Worker | `make -f Makefile.mac mac-install-worker WARM=0` (`cargo build --locked --release -p zkminer-prove-risc0 --features metal --bin zkminer-prove-risc0-metal`, then the default P31-M9b offline archive set: 11 archives, 165 MiB, built in 112 s). Installed to `~/.zkminer/provers/zkminer-prove-risc0-metal` (sha256 `9161605e…6ed0`, identical to `target/release`). The wave-7 binary is kept as `.w7-bak` (sha256 `731bcd24…`) |
| Method | as wave 7: `--benchmark` of the **installed** worker under `/usr/bin/time -l`, `RUST_LOG=info RISC0_METAL_ARCHIVE_GC=0`, every GPU command alone under `bin/gpu-run` (heavy, `--no-budget`; budgets via `RISC0_METAL_MAX_WORKING_SET_BYTES` inside the command). `ZKMINER_BENCH_RECEIPT_DIR` dumped the Groth16 receipts for the stock verifier |
| Machine state | No other cargo / test / prover process ran (checked with `ps`). Load averages 1.4-5.1 over the session, 1.4-2.9 during the full-suite runs (each log records start / end load). No memguard stop or kill. After cold-OS-cache runs, `scripts/quiet.sh` waited until `MTLCompilerService` was idle before the next timed run |
| Logs | `/Users/user/zkminer/scratch/w8/baseline/` (`full/`, `cold/`, `phx_def/`, `phx_bud/`, `wabx_def/`, `wabx_4g/`, `wabx_8g/`, `kab_def/`, `fit4g/`, `bud/`, `keccak/`, `sim/`, `prof/`, `warm/`, `prod/`, `receipts/`, `logs/`; drivers and parsers in `scripts/`, adapted from the wave-7 ones) |

Warm-up: one `BENCH_ONLY=fibonacci` run and one `RISC0_METAL_RV32IM_EVAL_FC=wait` memory-merkle run at the
default, 4 GiB and 8 GiB budgets (`warm/`). Every pipeline was warm ("built the Metal pipelines in 0.1 s (warm)").

## 3.1.1 Worker `--benchmark` (full suite, po2 20 segments)

Four 3.1 runs (`full/full1-4.log`) and two frozen 3.0 runs interleaved with the last of them (`full30_1`
between 3.1 runs 3 and 4, `full30_2` after run 4). Load 1.4-2.9. Every run proved, wrapped and verified every program.

| program | cycles | wave 7 (3.0) runs 1 / 2 / 3 (s) | 3.0 this session (s) | **3.1 runs 1 / 2 / 3 / 4 (s)** | RTX 4090 (s) | Metal / 4090 |
|---|---|---|---|---|---|---|
| fibonacci | 32,768 | 0.23 / 0.24 / 0.24 | 0.24 / 0.24 | 0.28 / 0.28 / 0.28 / 0.27 | 0.48 | 0.58x |
| sha256-chain | 46,137,344 | 50.03 / 49.89 / 49.96 | 49.99 / 49.85 | 50.47 / 50.57 / 50.42 / 50.36 | 35.82 | 1.41x |
| ecdsa-verify | 65,011,712 | 70.16 / 70.06 / 70.15 | 70.11 / 69.90 | 70.57 / 70.62 / 70.54 / 70.66 | 47.65 | 1.48x |
| bigint-mul | 27,525,120 | 30.10 / 30.16 / 30.13 | 30.10 / 30.04 | 30.36 / 30.44 / 30.43 / 30.37 | 19.77 | 1.54x |
| memory-merkle | 7,340,032 | 8.19 / 8.17 / 8.18 | 8.21 / 8.17 | 8.25 / 8.25 / 8.26 / 8.24 | 5.70 | 1.45x |
| chacha-mix | 8,404,992 | 9.51 / 9.52 / 9.52 | 9.57 / 9.51 | 9.65 / 9.63 / 9.59 / 9.64 | 6.67 | 1.44x |
| **all (sum of cycles / sum of STARK)** | 154,451,968 | 168.22 / 168.04 / 168.18 s, 918K c/s | 168.22 / 167.71 s, 918K / 921K | **169.58 / 169.79 / 169.52 / 169.54 s; 910.8K / 909.7K / 911.1K / 911.0K c/s (median 169.56 s, 910.9K c/s)** | 116.09 s, 1.33M c/s | **1.46x** |

- **3.1 costs +1.3 to +1.8 s of STARK (+0.8 to +1.1 %)** against the 3.0 worker in the same session. Segments
  140.44-140.75 s against 138.80-139.35 s (+1.1 to +1.9 s); recursion 29.03-29.11 against 28.86-28.90 s (+0.15 to
  +0.25 s). The per-phase A/B (3.1.3) explains the segments: +13 ms per po2 20 segment, mostly eval_check (the 26
  extra poly-mix powers and the new constraints of the 3.1 rv32im circuit). Recursion: 3.1.3 and 3.1.4.
- **Goal gap**: the goal is >= 1.197M c/s, i.e. a full-suite STARK <= 129.0 s. 3.1 gives 910.9K c/s:
  **40.5 s (23.9 %) of STARK still to remove** (wave 7: 39.2 s, 23.3 %). The RTX 4090 takes 1.46x less time than
  this Mac (wave 7: 1.45x). The RTX 5090 (~2.3M c/s, 67.2 s) takes 2.53x less time than this Mac.

**Groth16 wrap** (first and warm separately):

| wrap | identity_p254 | shrink_wrap | total |
|---|---|---|---|
| warm, 3.1 (every wrap after the first, 20 samples) | **900.6-907.6 ms** (median 903.8) | 670.5-679.1 ms (median 675.1) | **1.57-1.59 s** |
| warm, 3.0 this session (10 samples) | 738.2-745.4 ms (median 740.5) | 618.3-625.1 (median 620.5) | 1.36-1.37 s |
| first wrap of a `--benchmark` process, 3.1 | 911.5 / 913.7 / 912.9 / 925.0 ms | 672.6-675.5 ms | **1.58-1.60 s** |
| first wrap of a fresh `BENCH_ONLY=fibonacci` worker, 3.1 (10 runs, `cold/`) | 911.7-920.0 ms | | 1.59 s |

- The **wrap is +0.22 s (+16 %)** on 3.1: identity_p254 +163 ms and shrink_wrap +55 ms (A/B and profile in 3.1.4).
  The first wrap is still a warm one (+9 ms; the M9 / P31-M9b ctrl prewarm: "identity_p254 ctrl kept (98 MiB)
  in 251-253 ms"). Wrap 1.58 s is still below the RTX 4090's ~2.0 s.

**First job of a fresh worker** (`BENCH_ONLY=fibonacci`, `time -l` wall, warm OS cache): **2.25-2.32 s** (5 runs;
wave 7 1.93-1.95 s). The +0.35 s is the 3.1 wrap (+0.22 s) and the slower 3.1 recursion of a fresh process (3.1.4).

**Memory.**

| | wave 7 (3.0) | 3.1 |
|---|---|---|
| process peak footprint, full suite (`time -l`) | 14.69 / 14.69 / 14.69 GB | **14.69 / 14.70 / 14.69 / 14.69 GB** (unchanged) |
| `--benchmark` process lifetime peak | 14007-14013 MiB | 14009-14016 MiB |
| CPU user time of the full suite | 24.8-24.9 s | 26.4 s |
| wall time of the full suite | 176.6-176.8 s (176.3-176.8 this session) | 179.5-179.7 s (+2.9 s: STARK +1.5, six wraps +1.3) |

Metal peaks per phase at po2 20 (MiB, segments / recursion / wrap): memory-merkle 9437 / 9437 / **5635**,
bigint-mul 9612 / 8994 / 5635, chacha-mix 9402-9419 / 8281 / 5635, ecdsa-verify 9498 / 9484-9485 / 5635,
sha256-chain 9408 / 9408 / 5635, fibonacci 356 / 1493 / 5635. Segments are +32 MiB over wave 7. The wrap
peak is **+259 MiB** (5376 -> 5635): the v0.2.0 Groth16 circuit is larger (6,257,165 witness elements against
5,635,930). The device allocation of shrink_wrap is 4200.0 MiB (3.0 3957.5 MiB). The process footprint does
not change because it peaks in the segments.

## 3.1.2 Cold start and first install

`BENCH_ONLY=fibonacci`, `cold/`. The worker is the default install, which now ships the offline (AOT) set (P31-M9b).

| | pipelines | process wall (`time -l`) | first wrap |
|---|---|---|---|
| warm (5 runs, interleaved with the next row) | 0.1 s | **2.25-2.32 s** (wave 7 1.93-1.95) | 1.59 s |
| cold OS shader cache (`MTL_SHADER_CACHE_SIZE=0`), runtime archives present (5 runs) | 0.1 s | **2.29-2.32 s** (wave 7 **7.95-8.42 s**) | 1.59 s |
| first install with the default offline set: `MTL_SHADER_CACHE_SIZE=0`, empty `RISC0_METAL_ARCHIVE_DIR`, worker's `metal-aot/` (3 runs) | 0.1 s ("AOT_DIR=~/.zkminer/provers/metal-aot (packaged beside the worker)") | **2.31 / 2.31 / 2.31 s** (wave 7 with a hand-installed set 7.40-7.95 s; wave 7 default install 92.0-101.5 s) | 1.59 s |
| first install without the set (`ZKMINER_METAL_AOT=0`, otherwise as above; 2 runs) | 57.0-58.4 s (rv32im+zkp 57-58, recursion 25-27, keccak 17-32, groth16 22-27 in parallel) | 59.2-60.6 s | 1.59 s |

- **Both wave-7 cold-start open items are closed by P31-M9b.** The cold-OS-cache regression in `--benchmark`
  is gone: 8.0-8.4 s -> 2.3 s, because fibonacci "starts at once (first segment po2 15, probed in 0.005 s)"
  instead of waiting for the po2 20 specialization. The default install ships the AOT set, so a first start
  takes 2.3 s (wave 7: 92-102 s).
- `make mac-install-worker` took 142.6 s, including the AOT set build of 112 s. The risc0 dump tests had been
  compiled into `target/aot-dump` beforehand, outside the GPU lock. The 4 dump tests then ran under
  `gpu-run --light`.
- A cold run leaves `MTLCompilerService` at ~300 % CPU for several seconds after the worker exits. That is the
  background po2 20 specialization, which keeps running after exit. A timed run right after a cold one can
  therefore see extra load.

## 3.1.3 Per-phase breakdown: po2 20 rv32im segment

`risc0-circuit-rv32im` release lib test `prove_po2_sweep`, `R11_PO2S=20 R11_CPU=0 R11_REPS=4`, 7 rounds of
A (3.1), T (frozen 3.0 binary `risc0_circuit_rv32im-5d2fa880e1a95e2f`), A2 (3.1), rotated (`phx_def/`,
`scripts/phx.sh`, `phana.py`; per-process median of reps 1-3; load 1.9-3.3). Every seal verifies (3.1: seal v3).
The delta is T minus the mean of A and A2 in its round, so a negative number means 3.1 is slower.

| phase | wave 7 ms | 3.0 this session (T) ms | **3.1 (A / A2) ms** | 3.0 - 3.1 per round | share (3.1) |
|---|---|---|---|---|---|
| preflight + alloc | 12 | 12 | 12 / 12 | 0.0 | 1.3 % |
| witgen | 6 | 6 | 6 / 6 | 0.0 | 0.6 % |
| data commit | 210 | 210 | **210 / 210** | 0.0 [-1.0, 0.0] | 22.2 % |
| accum | 8 | 8 | 10 / 10 | **-2.0 [-2.0, -2.0] 7/7** | 1.1 % |
| accum commit | 109 | 109 | **109 / 109** | 0.0 | 11.5 % |
| eval_check | 542 / 539 | 541 [539-544] | **550 [549-551] / 550 [543-553]** | **-9.5 [-13.0, -2.5] 7/7** | 58.0 % |
| FRI + tail | 50 | 50 | 52 / 52 | **-2.0 [-3.0, -1.0] 7/7** | 5.5 % |
| **segment** | 936 / 933 | **935 [933-938]** | **948 [946-950] / 948 [941-951]** | **-13.0 [-15.5, -6.5] 7/7** (A2-A 0.0 [-7, +4]) | |

- **3.1 makes a po2 20 segment 13 ms slower (+1.4 %).** eval_check is +9.5 ms (+1.8 %): 484 poly-mix powers
  instead of 458 and the new constraints of the 3.1 rv32im circuit (P31-03). accum is +2 ms: the 3.1 accum
  layout (P31-05, `argU16` 4 -> 10). FRI + tail is +2 ms: the one extra global (`totalCycles`) and the regenerated
  taps. The two commits and witgen do not change. Peak GPU allocation is 9319 MiB in both builds.
- In the worker (`prof/mm_verbose*.log`, `RISC0_VERBOSE=1`, memory-merkle) the steady segments take 942.6-946.9
  ms on 3.1 and 933.2-937.5 ms on 3.0. The first segment takes 1156.5 ms (3.0: 1125.5 ms).
- At the worker's budget tiers (one process each, 3.1 / 3.0, `phx_bud/`): po2 20 at 8 GiB takes 1022 / 1013 ms,
  with eval_check 545 / 537 and peak 8129 / 8145 MiB. po2 19 at 4 GiB takes 518 / 511 ms, with peak 4080 /
  4021 MiB. A single process at each tier is a record, not a claim.

### Recursion per proof (`prof/mm_verbose.log` vs `mm_verbose_30.log`)

| | 3.0 | 3.1 |
|---|---|---|
| steady lift | 91.8-92.2 ms | **91.7-92.5 ms** (unchanged) |
| steady join | 96.9-98.0 ms | **97.2-97.9 ms** (unchanged) |
| first pair lift0+lift1 | 263.8 ms | **306.9 ms (+43)** |
| step 1 next_lift | 109.1 ms | **177.2 ms (+68)** |
| memory-merkle lift/join total (13 proofs) | 1.368 s | 1.479 s |

The 3.1 BIT_AND_ELEM constraints and the independent noise cells cost nothing measurable per proof. The +0.11 s
appears once per process, on the first lift and join. 3.1 ships the zkr programs xz-compressed, and
`get_zkr` decodes each one on its first use: 52 ms for `lift_rv32im_v2_20`, 79 ms for `join` (P31-10's
profile). P31-10's cache prevents a second decode. A full suite pays the decode once per program, which is why
its recursion grows by only +0.15 to +0.25 s.

### identity_p254 + shrink_wrap (`BENCH_ONLY=fibonacci`, `RISC0_VERBOSE=1 RISC0_G16_METAL_PROFILE=1`, `prof/g16*.log`)

| identity_p254 phase (ms) | 3.0 | 3.1 | |
|---|---|---|---|
| preflight / witgen / ctrl | 11.1 / 26.6 / 5.4 (Restored) | 11.0 / 26.4 / 6.6 (Restored) | |
| data commit | 436.1 | **544.3** | **+108 (+25 %)** |
| accum / accum commit | 5.1 / 95.6 | 5.1 / **120.0** | +24 |
| check commit | 122.2 | **146.9** | +25 |
| FRI | 31.7 (161.4 finalize) | 39.0 (193.5 finalize) | +7 |
| **identity_p254** | 742.7 | **908.3** | **+166** |

Every identity_p254 delta is in the Poseidon254 merkle hashing. The x^7 S-box costs +25 % per hash on Metal, as
P31-11 measured (+26.5 % on the hash shapes). The prewarmed ctrl commit is 187.8 ms on 3.1 against 151.1 ms on
3.0. It runs at start-up, off the job.

| shrink_wrap GPU ms | zero+witness_poly | NTT | sort_h | MSM H | sort_w | MSM A / B1 / C | MSM B2 | sum | allocated |
|---|---|---|---|---|---|---|---|---|---|
| 3.0 (v0.1.0) | 42.1 | 82.8 | 12.5 | 175.0 | 5.9 | 36.4 / 37.7 / 37.3 | 151.0 | 580.8 | 3957.5 MiB |
| 3.1 (v0.2.0) | 45.3 | 82.8 | 12.5 | 174.5 | 6.9 | 44.4 / 45.8 / 43.7 | 177.3 | **633.2** | **4200.0 MiB** |

The +52 ms is all in the witness-sized MSMs (A, B1, C, B2) and in witness_poly: v0.2.0 has 11 % more witness
elements. The domain is unchanged (2^23), so the NTT and the H MSM cost the same.

## 3.1.4 3.1 vs 3.0 worker A/B, and 3.1 knobs

**3.1 vs 3.0** (`scripts/wabx.sh`, `BENCH_ONLY=memory-merkle`, rounds of A1 (3.1), T (3.0 `.w7-bak`), A2 (3.1),
rotated; delta = T minus the mean of A1 and A2, so a negative number means 3.1 is slower). Every run proved,
wrapped and verified.

| budget | rounds, load | A2-A1 STARK | 3.1 STARK (seg + rec) | 3.0 STARK | 3.0 - 3.1 STARK | seg | rec | wrap (p254 / shrink) | footprint 3.0 - 3.1 | Metal peaks 3.1 (seg / rec / wrap) |
|---|---|---|---|---|---|---|---|---|---|---|
| default (po2 20) | 7, 1.8-2.9 | +0.000 [-0.020, +0.030] | 8.35 (6.85 + 1.49) | 8.19 | **-0.170 [-0.195, -0.140] 7/7** | -0.065 | -0.105 | **-0.220** (-166.0 / -55.4 ms) | **-0.130 GB** (13.20 vs 13.07) | 9437 / 9437 / 5635 (3.0 9421 / 9421 / 5392) |
| 8 GiB (po2 19) | 5, 1.6-3.1 | +0.010 [-0.010, +0.030] | 10.22 (7.24 + 2.97) | 10.04 | **-0.175 [-0.205, -0.145] 5/5** | -0.080 | -0.110 | **-0.230** (-162.9 / -67.0) | -0.059 GB (7.33 vs 7.27) | 4738 / 4337 / 5635 |
| 4 GiB (po2 18) | 5, 1.7-3.1 | +0.010 [-0.020, +0.040] | 13.92 (7.79 + 6.13) | 13.74 | **-0.185 [-0.200, -0.160] 5/5** | -0.075 | -0.110 | **-0.230** (-163.5 / -71.6) | **+0.147 GB** (5.02 vs 5.17: 3.1 is smaller) | 2388 / 2183 / 4054 (3.0 2400 / 2199 / 3958) |

- On a 7-segment job, 3.1 costs **+2.1 % of STARK**. About 40 % of that is the segments (~10 ms per segment) and
  about 60 % is the per-process first-use zkr decode in recursion (3.1.3). The wrap costs +0.22-0.23 s at every
  budget.
- At 4 GiB, 3.1 uses 0.15 GB **less** memory: P31-19 defers the keccak pipelines when the Groth16 key would not
  fit, and P31-12's split H shrinks the wrap. At the default budget it uses 0.13 GB more (the wrap
  allocation and P31-10's zkr cache).

**Knobs of this wave** (default budget, `kab_def/`, 7 rounds A1 / ZK / BL / A2 rotated, load 1.6-4.1;
A2-A1 STARK +0.020 [-0.050, +0.030], rec 0.000 [-0.010, +0.010]):

| item | knob (off) | delta (off - on) | verdict |
|---|---|---|---|
| P31-10 zkr program cache | `RISC0_METAL_REC_ZKR_CACHE=0` | **rec +0.025 [+0.010, +0.030] 7/7** (~2 ms per proof over 13 proofs); STARK +0.020 7/7; footprint +0.013 GB 7/7 | **holds** (P31-10: lift -1.2, join -2.0 ms per proof) |
| P31-M9b background prewarm | `ZKMINER_METAL_PREWARM_BLOCKING=1` (M9's blocking prewarm) | STARK -0.010 [-0.060, +0.025] 2/7, wrap -0.005, wall -0.025 | no change on a warm machine, as M9b claims. Its gain is the cold-OS-cache wall in 3.1.2 (8.0 -> 2.3 s) |
| P31-19 keccak prewarm deferral | `ZKMINER_METAL_PREWARM_KECCAK=on`, 4 GiB fibonacci (`fit4g/KC.log`) | **Groth16 wrap refused**: "allocating 4235118872 bytes for the Groth16 prover ... with ... already allocated would exceed the limit" | **fit claim holds**: the knob is what makes the 4 GiB wrap fit |
| P31-12 split H | `RISC0_G16_METAL_SPLIT_H=off`, 4 GiB fibonacci (`fit4g/SH.log`) | **Groth16 wrap refused** (plans 4371531224 B) | **fit claim holds** |
| P31-11 x^7, P31-09 noise | none (security fixes, no way back) | priced in 3.1.3 | - |

## 3.1.5 Budgets and simulated hosts

`BENCH_ONLY=fibonacci` (3 fresh runs per budget, `bud/`), memory-merkle (the A1 / A2 arms of 3.1.4), keccak
`k5_keccak_metal_e2e` (`K5_E2E_PO2S=14,17,18`, `gpu-run --budget`, 2 runs per budget, `keccak/`). **Everything
proves and verifies at both budgets. Every worker job wraps at both budgets.**

| budget | program | po2 | STARK | Groth16 wrap (p254 + shrink_wrap) | Metal peak MiB (seg / rec / wrap) | process peak footprint |
|---|---|---|---|---|---|---|
| 4 GiB | fibonacci | 18 | 0.27 s | **1.72 s** (0.898-0.899 + 0.816-0.824; wave 7 1.49) | 340 / 1477 / 4054 | 4.46-4.47 GB (wave 7 4.53-4.54) |
| 4 GiB | memory-merkle | 18 | **13.92 [13.91-13.95] s** (7.79 + 6.13), 8,388,608 cycles, **603K c/s** (wave 7 13.72 s, 611K) | 1.72-1.73 s | 2388 / 2183 / 4054 | **5.02 GB** (wave 7 5.16-5.17) |
| 4 GiB | keccak e2e 14,17,18 | 14, 17 (18 skipped below 7084 MiB) | po2 14 0.79-0.82 s, po2 17 1.78 s (prove + lift + verify; no wrap in this test) | - | - | 5.81-5.82 GB (= wave 7) |
| 8 GiB | fibonacci | 19 | 0.27 s | **1.73-1.74 s** (0.911-0.913 + 0.819-0.823; wave 7 1.50-1.51) | 356 / 1493 / 5635 | 4.75 GB (wave 7 4.62-4.63) |
| 8 GiB | memory-merkle | 19 | **10.22 [10.21-10.23] s** (7.24 + 2.97), 7,864,320 cycles, **770K c/s** (wave 7 10.05 s, 783K) | 1.74 s | 4738 / 4337 / 5635 | **7.33 GB** (wave 7 7.25) |
| 8 GiB | keccak e2e 14,17,18 | 14, 17, 18 | 0.78, 1.77-1.78, 2.93-2.94 s (= wave 7) | - | - | 10.24 GB (= wave 7) |
| 28.1 GiB (default) | memory-merkle | 20 | 8.35 s, 879K c/s | 1.59 s | 9437 / 9437 / 5635 | 13.20 GB (wave 7 13.07) |

- On 3.1, memory-merkle STARK is +1.3 to +1.7 % at the budget tiers (3.1.4). keccak is unchanged: its circuit
  and translators are byte-identical to 3.0 (P31-13).
- shrink_wrap below 20 GiB takes 0.82 s (wave 7 0.75 s; the `fast` evaluator), so the budget wraps are 1.72-1.74 s.
- **Not every receipt can be made at 4 GiB** (P31-14 / P31-19 open issue, reproduced here): after a keccak proof
  in the same process, the Groth16 key load is refused ("allocating 4235118872 bytes ... with 64946176 bytes
  already allocated would exceed the limit of 4294967296", `prod/b4_keccak.log`). The composite and succinct
  receipts are made and verify. Worker jobs without keccak wrap at 4 GiB (13 of 13 here; P31-M9b saw 1 refusal
  in 56). The remaining Groth16 headroom at 4 GiB is ~57 MiB.

**Tier the worker selects on a simulated host** (`sim/`, memory-merkle, `ZKMINER_METAL_HOST_RAM_BYTES` = host
RAM, `RISC0_METAL_MAX_WORKING_SET_BYTES` = that Mac's HAL limit, `ZKMINER_CUDA_VRAM_BYTES` = RAM - 8 GiB as the
dispatcher passes it; tier table `w7`; one run each):

| host | HAL limit / planned figure | po2 chosen | STARK | wrap | Metal peaks MiB | footprint |
|---|---|---|---|---|---|---|
| 16 GB | 10.67 / 8 GiB | 19 | 10.19 s (wave 7 9.96) | 1.74 s | 4798 / 4798 / 5635 | 7.38 GB (wave 7 7.32) |
| 18 GB | 12 / 10 GiB | 19 | 10.17 s (wave 7 9.96) | 1.74 s | 4798 / 4798 / 5635 | 7.38 GB |
| 24 GB | 16 / 16 GiB | 20 | 8.40 s (wave 7 8.25) | 1.74 s | 9405 / 8614 / 5635 | 12.39 GB (wave 7 12.29) |

Without the dispatcher figure the 18 GB case again picks po2 20 at 12 GiB (8.63 s, 12.39 GB, `sim/h18_mm.log`),
as in wave 7. 16 GB without the figure: po2 19, 10.16 s; 24 GB: po2 20, 8.39 s.

## 3.1.6 Correctness

- **Stock verifier**: `tools/stock_verify31` (crates.io `risc0-zkvm 3.1.0`, build.rs checks the registry
  sources), built `--locked` into scratch. **34 / 34 receipts made here verify ("STOCK 3.1 VERIFY: OK")**:
  - 18 Groth16 receipts of full runs 1-3 (6 programs each);
  - memory-merkle Groth16 at 4 and 8 GiB, plus the simulated 16 / 18 / 24 GB hosts (5);
  - the P31-14 producer's composite, succinct and Groth16 receipts for keccak (default and 8 GiB) and compose
    (default) (9);
  - keccak composite and succinct at 4 GiB (2). Its Groth16 is the known refusal above.
  
  Every Groth16 receipt reports verifier parameters `e71e6620a834f4d8...` (selector `e71e6620`), i.e. the v0.2.0
  setup's VK in the stock verifier. Every worker run also printed `verified` (the worker's own check).
- **3.0 receipts are rejected** by the same verifier: 6 / 6 of P31-14's 3.0 fibonacci / merkle composite,
  succinct and Groth16 receipts (`logs/rej30/`). Composite: "invalid receipt format". Succinct: parameters
  digest `ece5e9b8...`. Groth16: `73c457ba...`.
- Segment seals verify in every `prove_po2_sweep` rep (seal v3, 3.1 verifier). k5 keccak prove + lift + verify
  passed in 4 / 4 runs.

## 3.1.7 Open items from this baseline

- **STARK 910.9K c/s, 23.9 % (40.5 s of 169.6 s) short of the 1.197M c/s goal.** 3.1 added +0.8-1.1 % to the
  full-suite STARK: +13 ms per po2 20 segment (eval_check +9.5, accum +2, FRI + tail +2). A po2 20 segment takes
  948 ms: eval_check 58 %, data commit 22 %, accum commit 11.5 %. Recursion takes 29.0 s (17 % of the STARK).
- **The wrap is +0.22 s on 3.1** (1.36 -> 1.58 s warm), almost all from Poseidon254 x^7 in identity_p254
  (+166 ms, every merkle commit +25 %) and the larger v0.2.0 MSMs (+52 ms). It is still below the 4090's
  ~2.0 s. The p254 merkle hash is the lever if wrap time matters, for example latency on small jobs.
- **First-use zkr decode** (~0.1 s per fresh process: `lift_rv32im_v2_20` 52 ms + `join` 79 ms xz decode) lands
  on the first job's recursion. M9b's background prewarm could call `get_zkr` for the tier's lift and join
  programs at start-up to take it off the first job. Not measured.
- **4 GiB Groth16 after keccak** (P31-14 open issue 1 / P31-19 section 6): the wrap is still refused; ~57 MiB
  of headroom. Needs >= 5 MiB more room on the risc0 Groth16 side.
- `--benchmark` exits 0 when a program's Groth16 wrap fails (`fit4g/*.log`: "Groth16 wrap failed" and rc=0).
  A script that checks only the exit status misses the failure.
- **Security review finding (FIXED by B9-SEC, merge abea7c97c; see `notes/B9-SEC.md`):** the opt-in validation knob
  `RISC0_METAL_REC_SEAL_CHECK=1` (`recursion/src/prove/hal/metal/mod.rs`, `prove_seal_checked`, reachable in
  release builds) returns the Metal recursion receipt, and that receipt's ZK noise comes from a single 64-bit
  `rand::random()` seed expanded by `StdRng::seed_from_u64`. Stock 3.1 draws each cell from the OS-seeded
  ChaCha thread RNG. In that path the zero-knowledge of the returned receipt therefore rests on 64 bits. Default
  and row-pass paths are unaffected (`p31_09_noise_rows_independent_*`). Fix: a 256-bit seed with domain-separated
  per-stream derivation, or return an unseeded re-proof, or refuse the knob outside test / debug builds; document
  it as a validation-only knob. Owner: the recursion item (P31-09 follow-up).
  Resolution (B9-SEC): the seal check now draws a 32-byte OS seed per proof (`os_noise_seed`), each noise
  stream is ChaCha12 keyed by domain-separated SHA-256(domain || seed || stream), the u64 seed helper is
  `cfg(test)` only, and the knob is documented as validation-only. Default path unchanged; performance no change
  (interleaved A/B in the note). Receipts made with the knob on verify with the stock 3.1.0 verifier.
- Not measured here: the IPC worker path, 8-10 and 16-20-core parts, a real 16 GB Mac, CPU prover baselines,
  W7-T1 / W7-G1 (not merged into 3.0 or 3.1).
