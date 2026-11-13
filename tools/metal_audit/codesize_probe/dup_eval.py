#!/usr/bin/env python3
"""dup_eval.py IN.metal OUT.metal N [--ph main|logup|both]: code-footprint probe of the real
ef_eval_split (R21). Not a production transform.

Copies the generated evalfast TU's `namespace rv32im_metal_ef { ... }` N times (identical code
under the names rv32im_metal_ef, rv32im_metal_ef_c1, ...) and makes the split kernel pick copy
(rb % N) for row block rb. Every row runs exactly the code it runs today, with the same per-thread
work, stack and call path, so the results stay bit-exact; only the GPU-wide code footprint grows
N-fold (threadgroups resident on one core execute different copies). If ef_eval_split were
limited by instruction fetch from the cache hierarchy, the N-copy kernel would be slower. Check
the stack with agx_size.py: a copy that changes the regime is not a valid probe.
"""
import argparse, re

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("n", type=int)
ap.add_argument("--ph", choices=["main", "logup", "both"], default="both",
                help="which phase's calls pick a copy (the other phase uses copy 0)")
a = ap.parse_args()
src = open(a.inp).read()
start = src.index("namespace rv32im_metal_ef {")
end_tok = "} // namespace rv32im_metal_ef\n"
end = src.index(end_tok) + len(end_tok)
body = src[start:end]
copies = [body.replace("namespace rv32im_metal_ef {", f"namespace rv32im_metal_ef_c{i} {{")
          .replace(end_tok, f"}} // namespace rv32im_metal_ef_c{i}\n") for i in range(1, a.n)]
src = src[:end] + "".join(copies) + src[end:]


OLD_SEL = """  FpExt t;
  device Fp* dst;
  if (ph == 0) {
    t = rv32im_metal_ef::poly_fp(cycle, domain, ctx);
    dst = check;
  } else {
    t = rv32im_metal_ef::rk_accblk(cycle, domain, ctx);
    dst = scratch;
  }
"""
if OLD_SEL not in src:
    raise SystemExit("kernel phase selection not found (generator changed?)")


def chain(fn, on):
    # Nested ?: of calls: only the selected copy is called, and `t` is copy-initialised from the
    # call (no default-constructed FpExt written after a call: R17 tier A, which an if/else
    # assignment chain produced).
    expr = f"rv32im_metal_ef::{fn}(cycle, domain, ctx)"
    if on:
        for i in range(1, a.n):
            expr = f"(rb % {a.n}u == {i}u ? rv32im_metal_ef_c{i}::{fn}(cycle, domain, ctx) : {expr})"
    return expr


main = chain("poly_fp", a.ph in ("main", "both"))
logup = chain("rk_accblk", a.ph in ("logup", "both"))
src = src.replace(OLD_SEL, f"""  const FpExt t = ph == 0 ? {main} : {logup};
  device Fp* dst = ph == 0 ? check : scratch;
""")
open(a.out, "w").write(src)
