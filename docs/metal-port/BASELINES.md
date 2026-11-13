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
