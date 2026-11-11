#!/usr/bin/env bash
# R17 audit: compile every Metal translation unit the workspace builds to optimized LLVM IR (AIR, text)
# with the same flags the build scripts use (-std=metal3.2 -O2 + includes), for ir_scan.py.
#
# usage: emit_ir.sh <repo-root> <target-dir e.g. target/debug> <out-dir> [extra metal flags...]
# Needs `source ~/zkminer/env.sh` (Metal toolchain on PATH) and a prior metal build of rv32im,
# recursion and keccak (their generated sources live in OUT_DIR).
set -euo pipefail
ROOT=$(cd "$1" && pwd); TGT="$ROOT/$2"; OUT="$3"; shift 3; EXTRA=("$@")
mkdir -p "$OUT"
newest() { ls -td "$TGT"/build/$1-*/out 2>/dev/null | while read d; do [ -d "$d/$2" ] && { echo "$d"; break; }; done; }
RV=$(newest risc0-circuit-rv32im-sys metal); RC=$(newest risc0-circuit-recursion-sys metal)
KC=$(newest risc0-circuit-keccak-sys metal); SYS="$RV/_sys_"
[ -n "$RV" ] && [ -n "$RC" ] && [ -n "$KC" ] || { echo "missing OUT_DIR (build with --features metal first)" >&2; exit 2; }
echo "rv32im OUT_DIR $RV" >&2; echo "recursion OUT_DIR $RC" >&2; echo "keccak OUT_DIR $KC" >&2
FLAGS=(-std=metal3.2 -O2 -Wno-unused-variable ${EXTRA[@]+"${EXTRA[@]}"})
jobs=()
emit() { # unit src includes...
  local unit=$1 src=$2; shift 2
  local inc=(); for d in "$@"; do inc+=(-I "$d"); done
  local o="$OUT/$unit/$(basename "${src%.metal}").ll"; mkdir -p "$OUT/$unit"
  echo "$src" > "${o%.ll}.src"
  xcrun -sdk macosx metal "${FLAGS[@]}" -S -emit-llvm -c "$src" -o "$o" -I "$SYS" "${inc[@]}" 2> "${o%.ll}.err" || echo "FAIL $src" >&2
}
P=$ROOT/risc0
for f in "$P/circuit/rv32im-sys/kernels/metal/misc.metal" "$RV"/metal/*.metal; do
  emit rv32im "$f" "$P/circuit/rv32im-sys/kernels/metal" "$RV/metal" & done
for f in "$P/circuit/recursion-sys/kernels/metal/rec_kernels.metal" "$RC"/metal/*.metal; do
  emit recursion "$f" "$P/circuit/recursion-sys/kernels/metal" & done
wait
n=0
for f in "$P"/circuit/keccak-sys/kernels/metal/*.metal "$KC"/metal/*.metal; do
  emit keccak "$f" "$P/circuit/keccak-sys/kernels/metal" & n=$((n+1)); [ $((n % 8)) = 0 ] && wait; done
for f in "$P"/sys/kernels/zkp/metal/*.metal; do emit zkp "$f" "$P/sys/kernels/zkp/metal" & done
for f in "$P"/groth16-sys/kernels/metal/*.metal; do emit groth16 "$f" "$P/groth16-sys/kernels/metal" & done
wait
find "$OUT" -name '*.ll' | wc -l >&2
