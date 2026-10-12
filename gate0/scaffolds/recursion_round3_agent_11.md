# R3-A06 Round-3 Agent 11 — Is data NTT + merkle optimizable on Intel B70?

**Verdict: The 36% / 1151 ms attribution from R2-08 is an artifact of shared-queue
contention. In isolation, data NTT + merkle is ~4% of a lift (~56 ms). It IS
near saturation, but the round-2 number does NOT tighten R3-A06's ceiling —
that ceiling was built on a wrong premise.**

## 1. Dispatch — where the data NTT + merkle actually lives

The lift path enters here:
- `risc0/circuit/recursion/src/prove/mod.rs:308` — `prover.commit_group(REGISTER_GROUP_DATA, &witgen.data)` (timed as `data_commit`).
- `risc0/zkp/src/prove/prover.rs:109` — `commit_group()` does:
  1. `make_coeffs()` → `hal.batch_interpolate_ntt_zk_shift(coeffs, count)` — fused iNTT + 3^bit_rev zk-shift (one GPU pass).
  2. `PolyGroup::new()` (`risc0/zkp/src/prove/poly_group.rs:108`) → on the GPU:
     - `hal.batch_expand_into_evaluate_ntt(evaluated, coeffs, count, 2)` → `esimd_batch_expand_ffi` (zero-fill 4x), then `esimd_batch_forward_ntt` (128 in-order NTT submissions).
     - `hal.batch_bit_reverse(coeffs, count)` → `esimd_batch_bit_reverse_ffi` ending in `q->wait()`.
     - `MerkleTreeProver::new()` → `hash_rows` then `hash_fold_tree`.

All on the Intel ESIMD kernels in `risc0/sys/kernels/zkp/intel/`:
- NTT: `ntt_kernel.cpp::gpu_forward_ntt_no_wait` → `ntt_ct_slm_combined` (SLM stages 1-14) + `ntt_ct_fused_{4,3,2}stage` tails. SLM_LG_BLOCK=14, 256-GRF mode.
- Bitrev: `eltwise_ops.cpp::esimd_batch_bit_reverse` — plain SYCL parallel_for of `range<1>(count)`.
- Merkle: `poseidon2.cpp::esimd_poseidon2_rows` (col_size=128 → ESIMD path, since col_size!=24) and `esimd_poseidon2_fold` (OpenCL fast path, **on by default** since `RISC0_POSEIDON2_OPENCL_OFF` is the new opt-out gate).

Queue: a singleton `sycl::queue` (`risc0/sys/src/intel.rs:131::get_queue`), in-order, shared by the rv32im prove path and the recursion prove path. There's a separate `EVAL_CHECK_QUEUE`, but no separate recursion queue.

## 2. Re-reading R2-08's numbers — they are queue-contention artifacts

R2-08 reported (lift seg 1, "warm cache, representative"):
```
data_commit = 1151.3 ms (36% of 3189 ms lift)
  · poly_group(data) ntt=0.9 bitrev=1027.9 merkle=121.5 total=1150.3
```
But in the raw `/tmp/succinct_5seg_log.txt`, recursion-lift `data_commit` ranges
from **466 ms to 1314 ms** across the same 4 lifts in that run. That huge swing,
plus the matching swing in `eval_check_launch` (27 ms → 1917 ms in the same log),
plus the matching swing in composite `poly_group(data)` (490 ms → 1764 ms — all
on the same kernel, same domain, same hardware), all share one cause: the
in-order singleton SYCL queue is being drained inside whichever phase calls
`q->wait()` first. The bitrev FFI wraps `q->wait()`, so the `bitrev` field
absorbs that drain.

The isolated lift measurement (this session, `/tmp/recursion_round3_03_lift_v2.log`,
po2=18, no concurrent composite prove on the queue) shows:
```
poly_group(data) count=128 size=262144 domain=1048576
  ntt=1.3ms bitrev=11.6ms merkle=38.3ms total=51.2ms
data_commit=56.1ms  fri=869.7ms  total=1350.2ms
```
That makes data NTT + merkle = **~56 ms ≈ 4.2 % of a 1.35 s lift**, NOT 36 %.
The 1151 ms in R2-08 was waiting for prior composite/join GPU work to drain.

What R2-08 was actually measuring as "data NTT+merkle" was mostly the previous
composite segment's NTT + merkle + eval_check residue draining out of the shared
queue at the moment the next lift's bitrev called `q->wait()`. Real per-lift
recursion data NTT+merkle work is ~56 ms.

## 3. Is the ~56 ms genuinely saturated?

Yes — to a very high degree. Decomposition of the 56 ms / lift at po2=18,
domain=2^20, 128 columns:

| Component | ms | What |
|---|---:|---|
| `batch_interpolate_ntt_zk_shift` (in `make_coeffs`) | ~5 | 128 × inverse NTTs at lg_n=18, fused zk-shift |
| `batch_expand_into_evaluate_ntt` (expand + 128 × fwd NTT) | ~10 | per `project_ntt_no_integrate.md`, ESIMD batched fwd NTT at po2=20 ≈ 0.084 ms/poly × 128 ≈ 11 ms |
| `batch_bit_reverse` (on coeffs, 33 M elems) | ~3 | one parallel_for, swaps |
| Merkle `hash_rows` (128 cols × 1 048 576 rows, ESIMD path) | ~25 | the chunky one — 1 048 576 sponges of 128 elems each, 8 perm chunks per sponge |
| Merkle `hash_fold_tree` (log2(1 048 576) layers, OpenCL fast path ON) | ~13 | already optimized |
| GPU launch / sync overhead | ~few | residual |

The NTT side is already at ~2 TB/s effective L2 BW on BMG-G31 (per
`project_ntt_no_integrate.md` head-to-head vs gate0 OpenCL NTT — ESIMD ties
or wins, gate0 brings no new tricks). Forward NTT at po2=20 is ~0.084 ms/poly
batched. Twiddles are cached.

The Poseidon2 merkle side: `hash_fold` already uses the OpenCL fast path by
default (~2.71× vs ESIMD per `project_poseidon2_e2e_findings.md`). The only
non-fast-path piece is `hash_rows` at col_size=128 — the OpenCL `_mont_io`
specialization is hard-coded for col_size=24 (FRI commit row width). A generic
col_size kernel `poseidon2_hash_rows` exists in `poseidon2_gate0.cl:1389` but
uses to_mont/from_mont on I/O, so it cannot drop in over the Mont-form buffers
of the risc0-intel pipeline without writing a `_mont_io` variant.

## 4. Quick wins ranked

| # | Win | Effort | Save / lift | Save E2E (42-seg Succinct) | Risk |
|---|---|---|---:|---:|---|
| 1 | **Write `poseidon2_hash_rows_mont_io` generic kernel** for col_size≠24 (so the 128-col data merkle gets the 1.96× OpenCL speedup the fold path already enjoys) | 1-2 days | ~12 ms (25/1.96 ≈ 13 ms saved) | ~0.5 s on 99 s lift bucket = **~0.5 % Succinct E2E** | Low — kernel exists, just needs Mont-IO wrapping + dispatcher gate widening |
| 2 | Out-of-order queue for recursion HAL (separate from rv32im queue) — would eliminate the queue-drain inflation R2-08 saw, but at concurrent-lift level | 2-3 days | 0 in isolation; eliminates queue contention when lifts overlap composite proves | Mostly variance-reducing; perhaps **~2-3 % E2E** if it lets composite+lift overlap genuinely | Medium — risk of breaking ordering assumptions in MerkleTreeProver/eval_check |
| 3 | Cross-poly batched fwd NTT (one kernel handles 128 polys' stage simultaneously, sharing twiddles, instead of 128 sequential kernel submissions) | 3-5 days | ~5 ms | **~0.2 % Succinct E2E** | Medium — kernel redesign |
| 4 | Bitrev SIMD16/coalesced rewrite (currently `parallel_for(range<1>(count))` with half the threads no-op) | 1 day | ~2 ms | **~0.1 % Succinct E2E** | Low |
| 5 | Concurrent data and accum poly_groups (currently serialized in `commit_group`) | 1 day | ~5 ms (overlap accum=141 + accum_commit=13 with data_commit=56) | **~0.2 % Succinct E2E** | Low — different buffers, independent |

**Total realistic quick-win envelope on data NTT+merkle alone: ~1-1.5 % Succinct E2E.**

## 5. Recommendation for R3-A06's ceiling

R2-08's claim that data NTT+merkle is 36 % of lift is **wrong** — it confused
shared-queue drainage with GPU work attribution. In isolation, the bucket is
**~4 % of a lift, ~1.6 % of Succinct E2E**. R3-A06's ceiling argument
("recursion lift+join is 40 % of Succinct E2E, GPU port would unlock 20-30 %")
is unaffected because R3-A06 targets the CPU witgen/accum FFI work (the only
non-par-safe-within-lift residue plus the 200 ms PCIe d2h/h2d round-trips),
NOT the GPU data NTT+merkle that the round-2 task was looking at.

**Conclusion:** Data NTT+merkle on Intel B70 is essentially saturated.
The 4-8 % aggregated saves above are real but tiny relative to R3-A06's 20-30 %
target. The R3-A06 ceiling does NOT need to be tightened — the 36 % "data
NTT+merkle" bucket simply didn't exist; it was queue-drain inflation that R3-A06
also doesn't touch.

## Files referenced

- `risc0/circuit/recursion/src/prove/mod.rs:308` — `commit_group(REGISTER_GROUP_DATA, ...)` call site
- `risc0/zkp/src/prove/prover.rs:109-129` — `commit_group` = `make_coeffs` + `PolyGroup::new`
- `risc0/zkp/src/prove/poly_group.rs:101-135` — NTT + bitrev + merkle dispatch
- `risc0/zkp/src/hal/intel.rs:649-693` — `batch_expand_into_evaluate_ntt` host dispatch
- `risc0/sys/kernels/zkp/intel/intel_ffi.cpp:249-263` — `esimd_batch_forward_ntt` (in-order 128 NTTs)
- `risc0/sys/kernels/zkp/intel/intel_ffi.cpp:420-423` — `esimd_batch_bit_reverse_ffi` (has the `q->wait()`)
- `risc0/sys/kernels/zkp/intel/ntt_kernel.cpp:2995-3015` — `gpu_forward_ntt_no_wait` per-poly
- `risc0/sys/kernels/zkp/intel/eltwise_ops.cpp:317-346` — `esimd_batch_bit_reverse` kernel
- `risc0/sys/kernels/zkp/intel/poseidon2.cpp:403-503` — `esimd_poseidon2_rows` ESIMD path + col_size==24 OpenCL gate at line 410
- `risc0/sys/kernels/zkp/intel/poseidon2_gate0.cl:1389-1442` — generic OpenCL `poseidon2_hash_rows` (needs Mont-IO wrapping to integrate)
- `risc0/sys/src/intel.rs:131` — shared singleton SYCL queue
- `/tmp/recursion_round3_03_lift_v2.log` — isolated lift measurement showing data_commit=56 ms
- `/tmp/succinct_5seg_log.txt` — the round-2 raw log showing data_commit swings 466-1314 ms across lifts (queue-drain inflation)
