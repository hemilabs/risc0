#!/usr/bin/env python3
"""Synthetic instruction-fetch probe for Apple GPUs (R21, LEADS #4). Not a production kernel.

Emits one Metal TU with one kernel per code size. Every kernel does the SAME arithmetic per row
(thread): TOTAL blocks of an eval-like integer mix (32x32->64 multiply-adds into 64-bit lazy
accumulators, a fold every second block), in the SAME number of out-of-line calls.

The code is organised like ef_eval_split: chunks of CHUNK blocks live in noinline functions that
read the row state from a private slab at entry and write it back at exit. A kernel of size class
B owns B / CHUNK distinct chunk functions (distinct random constants, so nothing is shared or
folded) and calls them in sequence inside a loop whose trip count R = TOTAL / B comes from a
buffer at run time. Small B: the loop's code fits in the instruction caches and is re-executed.
Large B: the thread streams B blocks of distinct code once per row (R = 1 at B = TOTAL).
If the GPU is instruction-fetch bound at the eval's code size, ns/row rises with B.

Why chunks: the AGX back end's translation time is superlinear in the size of one function body
(a 1 MB single-function kernel took 240 s, 4 MB did not finish in 10 min); the real eval is ~25
noinline functions for the same reason.

R17 note: the slab is written element by element only at fixed indices, in straight-line code,
never in a loop; this probe is timing-only, and the bench checks run-to-run determinism.

usage: gen.py OUT.metal --total N --blocks B1,B2,... [--chunk 52 | --fns F] [--accs 8] [--seed 1] [--mix eval|cheap] [--slab WORDS [--slab-dyn]]
"""
import argparse, random

P = 0x78000001  # BabyBear


MIX = "eval"
SLAB_DYN = False  # slab indices move with the loop counter (same slab working set for every B)
SLAB = 0  # words of per-thread private slab (stack traffic like ef_eval_split's), 0 = none


def block(rng, accs, bi):
    if MIX == "cheap":
        # Worst case for instruction fetch: full-rate 32-bit ops only (no multiplies), so the
        # bytes of code consumed per ALU cycle are as high as this generator can make them.
        lines = []
        for a in range(accs):
            c = rng.randrange(1, 1 << 32)
            src = (a + bi) % accs
            lines.append(f"  x{a} = (x{a} + (x{src} ^ {c}u)) ^ (x{src} >> {1 + (a + bi) % 13});")
        return "\n".join(lines)
    lines = []
    if SLAB:
        w, r = rng.randrange(SLAB), rng.randrange(SLAB)
        if SLAB_DYN:
            m = SLAB - 1
            lines.append(f"  sl[({w}u + rk) & {m}u] = x{bi % accs}; x{(bi + 1) % accs} ^= sl[({r}u + rk) & {m}u];")
        else:
            lines.append(f"  sl[{w}] = x{bi % accs}; x{(bi + 1) % accs} ^= sl[{r}];")
    for a in range(accs):
        c = rng.randrange(1, P)
        src = (a + bi) % accs
        lines.append(f"  a{a} += ulong(x{src}) * {c}u;")
    if bi % 2 == 1:
        for a in range(accs):
            lines.append(f"  a{a} = (a{a} & 0xffffffffu) + (a{a} >> 32) * 0x07ffffffu;")
    for a in range(accs):
        k = rng.randrange(1, 1 << 32)
        lines.append(f"  x{a} = uint(a{a}) ^ {k}u;")
    return "\n".join(lines)


def chunk_fn(name, chunk, accs, rng):
    slab = (", thread uint* sl" + (", uint rk" if SLAB_DYN else "")) if SLAB else ""
    out = [f"static __attribute__((noinline)) void {name}(thread ulong* st{slab}) {{"]
    for a in range(accs):
        out.append(f"  ulong a{a} = st[{a}]; uint x{a} = uint(st[{accs + a}]);")
    for b in range(chunk):
        out.append(block(rng, accs, b))
    for a in range(accs):
        out.append(f"  st[{a}] = a{a}; st[{accs + a}] = x{a};")
    out.append("}")
    return "\n".join(out)


def kernel(name, nblocks, chunk, accs, rng):
    fns = [f"{name}_c{j}" for j in range(nblocks // chunk)]
    out = [chunk_fn(f, chunk, accs, rng) for f in fns]
    k = [f"kernel void {name}(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],",
         "    constant uint& reps [[buffer(2)]], threadgroup uint* tgm [[threadgroup(0)]],",
         "    uint gid [[thread_position_in_grid]]) {",
         f"  ulong st[{2 * accs}];"]
    if SLAB:
        # Straight-line init would be SLAB statements; a loop with no call in it is not the R17
        # shape (that needs a call in the loop body).
        k.append(f"  uint sl[{SLAB}];")
        k.append(f"  for (uint i = 0; i < {SLAB}u; i++) sl[i] = gid + i;")
    for a in range(accs):
        k.append(f"  st[{a}] = {a}; st[{accs + a}] = in[gid * {accs} + {a}];")
    k.append("  for (uint r = 0; r < reps; r++) {")
    if SLAB_DYN:
        k.append("    uint rk = r * 0x9e37u;")
    args = (", sl, rk" if SLAB_DYN else ", sl") if SLAB else ""
    k += [f"    {f}(st{args});" for f in fns]
    k.append("  }")
    acc = " ^ ".join(f"uint(st[{a}]) ^ uint(st[{a}] >> 32) ^ uint(st[{accs + a}])" for a in range(accs))
    if SLAB:
        acc += f" ^ sl[gid % {SLAB}u]"
    k.append(f"  uint res = {acc};")
    # Keeps the threadgroup allocation (the bench's occupancy limiter) bound; never taken.
    k.append("  if (reps == 0xffffffffu) { tgm[0] = res; res = tgm[1]; }")
    k.append("  out[gid] = res;")
    k.append("}")
    out.append("\n".join(k))
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--total", type=int, required=True, help="blocks per row (same for every kernel)")
    ap.add_argument("--blocks", required=True, help="comma-separated distinct blocks per kernel")
    ap.add_argument("--chunk", type=int, default=52, help="blocks per noinline function")
    ap.add_argument("--fns", type=int, default=0,
                    help="instead of --chunk: every kernel has exactly this many chunk functions "
                         "(chunk = B / fns), so the number of distinct callees does not vary with B")
    ap.add_argument("--accs", type=int, default=8)
    ap.add_argument("--slab", type=int, default=0,
                    help="per-thread private slab words; each block stores and loads one word")
    ap.add_argument("--slab-dyn", action="store_true",
                    help="slab indices move with the loop counter, so every kernel touches the whole "
                         "slab per row (otherwise small-B kernels touch only ~2B words); SLAB must "
                         "be a power of two")
    ap.add_argument("--mix", choices=["eval", "cheap"], default="eval",
                    help="eval: 32x32->64 multiply-add links with folds; cheap: 32-bit add/xor/shift only")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    global MIX
    MIX = a.mix
    global SLAB
    SLAB = a.slab
    global SLAB_DYN
    SLAB_DYN = a.slab_dyn
    if SLAB_DYN and (SLAB & (SLAB - 1)):
        raise SystemExit("--slab-dyn needs a power-of-two --slab")
    rng = random.Random(a.seed)
    parts = ["#include <metal_stdlib>", "using namespace metal;", ""]
    for b in [int(x) for x in a.blocks.split(",")]:
        chunk = b // a.fns if a.fns else a.chunk
        if a.total % b or b % chunk or (a.fns and b % a.fns):
            raise SystemExit(f"{b} must divide {a.total} and be a multiple of the chunk ({chunk})")
        parts.append(f"// cs_{b}: {b // chunk} chunk functions of {chunk} blocks, reps = {a.total // b}")
        parts.append(kernel(f"cs_{b}", b, chunk, a.accs, rng))
        parts.append("")
    open(a.out, "w").write("\n".join(parts))


if __name__ == "__main__":
    main()
