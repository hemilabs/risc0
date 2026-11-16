# Z13's reported shape: guarded loop update of a private array (NOT reproduced)

Z13 (notes/Z13.md, "R17-family defect found in the prototype") reported that an NTT prototype applied a
coset twist as `if (MODE == 2 && bz != 0) { for m: v[m] = mul(v[m], t); t = mul(t, st); }` on a private
`uint v[16]` in fully inlined, unrolled code, and that the `bz == 0` threadgroups (the ones that skip
the branch) came out wrong in 7/20 to 19/20 repetitions. The prototype (`nttq_r11_proto.patch`,
`proto/ntt_q.v1.metal`, `z13_debug_stress`) no longer exists.

`z13.metal` rebuilds the shape from that description; `risc0/zkp/src/hal/metal/tests_r17.rs`
(`R17_Z13`, `r17_shape_probe_z13`) compiles the same kernel at run time. `harness.m` runs it and its
variants (a `-D` define) against a C reference:

```
clang -fobjc-arc -O1 -framework Metal -framework Foundation -o harness harness.m
gpu-run --light -- ./harness z13.metal <reps> <cols> <col blocks> <threads> [VARIANT]   # DISP=k: k dispatches per command buffer
```

Each thread loads a private `uint v[16]`, applies a Montgomery coset twist to every element (only
where the guard holds), runs radix-16 butterflies, exchanges through threadgroup memory, butterflies
again and stores. Every helper is `always_inline` and every loop `#pragma unroll`; there is no call.
The `bz == 0` blocks need no twist, so skipping the branch is correct by construction; `CONTROL` applies
the twist everywhere, as an identity for `bz == 0` (t = st = Montgomery one).

## Fix round 1: the first rebuild's "reproduction" was a reference bug

The first W7-T1 version reported "20/20 wrong, exactly the bz == 0 blocks, deterministic". Its CONTROL
and both references (C and Rust) set `t = ONE` for `bz == 0` but kept stepping `t = mul(t, st)` with
`st != ONE`, i.e. they multiplied `v[m]` of the skipped blocks by `st^m`, which the guarded kernel never
does. The two computed different functions, so every skipped output differed (an independent
verifier found this). Mutation check (fix round 1): putting the old `if (bz == 0) t = ONE;` back into the
Rust probe's CONTROL makes the CONTROL variant 5/5 wrong with 100 % of the wrong outputs in the bz == 0
blocks, and the guarded loop 0/5: the old pattern, now on the other side.

## Results with the kernel's own semantics (macOS 27.2, M5 Max 32-core GPU)

20 reps each (10 at the large size); "wrong" = differs from the CPU reference; "nondet" = differs from rep 0.

| variant | twist | 65536 cols x 4 blocks, T 256 | T 1024 | T 64 | 262144 x 8 blocks | 8 dispatches / command buffer |
|---|---|---|---|---|---|---|
| (default) | `if (bz != 0) { for m (unrolled): v[m] = mul(v[m], t); t = mul(t, st); }` | 0/20, nondet 0/20 | 0/20 | 0/20 | 0/10 | 0/20 |
| `CONTROL` | unconditional, identity for bz == 0 | 0/20 | 0/20 | 0/20 | 0/10 | 0/20 |
| `NOUNROLL` | as default, `#pragma unroll 1` | 0/20 | 0/20 | 0/20 | 0/10 | 0/20 |
| `TRIP` | `for (m < (bz != 0 ? 16 : 0))`, no `if` | 0/20 | 0/20 | 0/20 | 0/10 | 0/20 |
| `GUARD_DIV` | `if ((col & 1) != 0)` (divergent guard) | 0/20 | 0/20 | 0/20 | 0/10 | 0/20 |
| `NODYN` | default without the dynamically indexed read | 0/20 | 0/20 | 0/20 | 0/10 | 0/20 |
| `SINGLE` | `if (bz != 0) v[0] = mul(v[0], t);` | 0/20 | 0/20 | 0/20 | 0/10 | 0/20 |
| `TABLE` | coset factors from device memory, `tw[bz*n+col]`, `stw[bz]` (identity at bz 0) | 0/20 | 0/20 | 0/20 | 0/10 | 0/20 |
| `SPILL` | default plus a second live private array `u[16]` updated in the same guarded loop | 0/20 | 0/20 | 0/20 | 0/10 | 0/20 |

**No variant fails, and none is nondeterministic** (logs: `scratch/w7/W7-T1/r1/z13/sweep_default.log`,
`sweep_geo.log`). The Rust probe under `run_stress.sh`'s zkp step: guarded loop 0/10, control 0/10.
Z13's failure is therefore not reproduced; whatever made the prototype fail (its reference, a race in
its threadgroup exchange, or a real compiler defect in a shape this rebuild misses) cannot be told
apart without the prototype. The probe stays in the stress corpus as a regression case: if it ever
fails, that is a reproduction and a P0.

`tools/metal_audit/ir_scan.py` tier Z matches the default, NOUNROLL, TRIP, GUARD_DIV, NODYN, TABLE and
SPILL IR (SPILL twice) and not CONTROL or SINGLE: it detects the shape, but the shape has no
demonstrated GPU failure, so tier Z is an unvalidated heuristic (reported, not gated).
