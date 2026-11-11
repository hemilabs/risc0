#!/usr/bin/env bash
# R17 audit: build an rv32im metallib outside cargo with a perturbed codegen for one TU, for
# RISC0_METAL_RV32IM_METALLIB=<out>/rv32im.metallib (test/debug builds only).
#
# usage: build_rv32im_variant.sh <rv32im-sys OUT_DIR> <out dir> [TU=path-or-- TU_FLAGS...]...
#   TU is one of misc rv32im_incs accum_tu witgen_tu eval_check eval_check_ef. Env:
#   SRC_<TU>=<file>   compile this source instead of the generated one
#   FLAGS_<TU>="..."  extra metal flags for that TU (e.g. FLAGS_accum_tu="-Os")
# Every TU gets -std=metal3.2 -O2 first (as the build script), so a later -O flag wins.
set -euo pipefail
OUT=$1; DST=$2; mkdir -p "$DST"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SYS=$ROOT/risc0/circuit/rv32im-sys/kernels/metal
GEN=$OUT/metal
common=(-std=metal3.2 -O2 -I "$OUT/_sys_" -Wno-unused-variable -I "$SYS" -I "$GEN")
airs=()
for tu in misc rv32im_incs accum_tu witgen_tu eval_check eval_check_ef; do
  src_var="SRC_$tu"; flags_var="FLAGS_$tu"
  src=${!src_var:-}
  if [ -z "$src" ]; then
    if [ $tu = misc ]; then src=$SYS/misc.metal; else src=$GEN/$tu.metal; fi
  fi
  extra=(${!flags_var:-})
  xcrun -sdk macosx metal -c "$src" "${common[@]}" ${extra[@]+"${extra[@]}"} -o "$DST/$tu.air" &
  airs+=("$DST/$tu.air")
done
wait
xcrun -sdk macosx metallib "${airs[@]}" -o "$DST/rv32im.metallib"
ls -la "$DST/rv32im.metallib"
