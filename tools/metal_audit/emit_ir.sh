#!/usr/bin/env bash
# R17 audit: compile every Metal translation unit the workspace builds to optimized LLVM IR (AIR, text)
# with the same flags the build scripts use (-std=metal3.2 -O2 + includes), for ir_scan.py.
#
# usage: emit_ir.sh <repo-root> <target-dir e.g. target/debug> <out-dir> [extra metal flags...]
# Needs `source ~/zkminer/env.sh` (Metal toolchain on PATH) and a prior metal build of rv32im,
# recursion and keccak (their generated sources live in OUT_DIR).
# EMIT_IR_UNITS (P31-T1; default "rv32im recursion keccak zkp groth16") selects the units; only the
# OUT_DIRs of the selected generated units are required. zkp and groth16 need no OUT_DIR. The shared
# Metal headers come from rv32im's OUT_DIR (`_sys_`) when it exists, else from build_kernel (the
# same files: build_kernel writes them there verbatim).
set -euo pipefail
ROOT=$(cd "$1" && pwd); TGT="$ROOT/$2"; OUT="$3"; shift 3; EXTRA=("$@")
mkdir -p "$OUT"; rm -f "$OUT"/.failed.*
newest() { { ls -td "$TGT"/build/$1-*/out 2>/dev/null || true; } | while read d; do if [ -d "$d/$2" ]; then echo "$d"; break; fi; done; }
UNITS=" ${EMIT_IR_UNITS:-rv32im recursion keccak zkp groth16} "
want() { case "$UNITS" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
for u in $UNITS; do
  case $u in rv32im|recursion|keccak|zkp|groth16) ;;
    *) echo "emit_ir.sh: ERROR: unknown unit '$u' in EMIT_IR_UNITS" >&2; exit 2 ;; esac
done
RV=$(newest risc0-circuit-rv32im-sys metal); RC=$(newest risc0-circuit-recursion-sys metal)
KC=$(newest risc0-circuit-keccak-sys metal)
# W7-T1: name every missing crate and fail (run_stress.sh used to hide this message).
missing=()
want rv32im && { [ -n "$RV" ] || missing+=(risc0-circuit-rv32im); }
want recursion && { [ -n "$RC" ] || missing+=(risc0-circuit-recursion); }
want keccak && { [ -n "$KC" ] || missing+=(risc0-circuit-keccak); }
if [ ${#missing[@]} -gt 0 ]; then
  echo "emit_ir.sh: ERROR: no metal OUT_DIR under $TGT/build for: ${missing[*]}" >&2
  echo "emit_ir.sh: build them first, e.g. cargo test --no-run --features metal --lib $(printf -- '-p %s ' "${missing[@]}")" >&2
  echo "emit_ir.sh: or restrict EMIT_IR_UNITS (now:$UNITS)" >&2
  exit 2
fi
if [ -n "$RV" ] && [ -d "$RV/_sys_" ]; then SYS="$RV/_sys_"; else SYS="$ROOT/risc0/build_kernel/kernels/metal"; fi
echo "units:$UNITS; shared headers $SYS" >&2
want rv32im && echo "rv32im OUT_DIR $RV" >&2
want recursion && echo "recursion OUT_DIR $RC" >&2
want keccak && echo "keccak OUT_DIR $KC" >&2
FLAGS=(-std=metal3.2 -O2 -Wno-unused-variable ${EXTRA[@]+"${EXTRA[@]}"})
jobs=()
emit() { # unit src includes...
  local unit=$1 src=$2; shift 2
  local inc=(); for d in "$@"; do inc+=(-I "$d"); done
  local o="$OUT/$unit/$(basename "${src%.metal}").ll"; mkdir -p "$OUT/$unit"
  echo "$src" > "${o%.ll}.src"
  xcrun -sdk macosx metal "${FLAGS[@]}" -S -emit-llvm -c "$src" -o "$o" -I "$SYS" "${inc[@]}" 2> "${o%.ll}.err" ||
    { echo "FAIL $src (see ${o%.ll}.err)" >&2; : > "$OUT/.failed.$(basename "$o")"; }
}
P=$ROOT/risc0
if want rv32im; then
  for f in "$P/circuit/rv32im-sys/kernels/metal/misc.metal" "$RV"/metal/*.metal; do
    emit rv32im "$f" "$P/circuit/rv32im-sys/kernels/metal" "$RV/metal" & done
fi
if want recursion; then
  for f in "$P"/circuit/recursion-sys/kernels/metal/*.metal "$RC"/metal/*.metal; do
    [ -e "$f" ] || continue
    emit recursion "$f" "$P/circuit/recursion-sys/kernels/metal" & done
fi
wait
n=0
if want keccak; then
  for f in "$P"/circuit/keccak-sys/kernels/metal/*.metal "$KC"/metal/*.metal; do
    [ -e "$f" ] || continue
    emit keccak "$f" "$P/circuit/keccak-sys/kernels/metal" & n=$((n+1)); [ $((n % 8)) = 0 ] && wait; done
fi
want zkp && for f in "$P"/sys/kernels/zkp/metal/*.metal; do emit zkp "$f" "$P/sys/kernels/zkp/metal" & done
want groth16 && for f in "$P"/groth16-sys/kernels/metal/*.metal; do emit groth16 "$f" "$P/groth16-sys/kernels/metal" & done
wait
nll=$(find "$OUT" -name '*.ll' | wc -l | tr -d ' '); echo "emit_ir.sh: $nll translation units" >&2
nf=$(find "$OUT" -name '.failed.*' | wc -l | tr -d ' ')
[ "$nf" = 0 ] || { echo "emit_ir.sh: ERROR: $nf translation unit(s) failed to compile" >&2; exit 3; }
