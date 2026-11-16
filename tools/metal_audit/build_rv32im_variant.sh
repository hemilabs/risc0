#!/usr/bin/env bash
# R17 audit: build an rv32im metallib outside cargo with a perturbed codegen for one TU, for
# RISC0_METAL_RV32IM_METALLIB=<out>/rv32im.metallib (test/debug builds only).
#
# usage: build_rv32im_variant.sh <rv32im-sys OUT_DIR> <out dir>
#   The TUs are misc (kernels/metal/misc.metal) and every generated <OUT_DIR>/metal/*.metal, in
#   the build script's order (misc, rv32im_incs, accum_tu, witgen_tu, eval_check, eval_check_ef,
#   then the other eval_check_ef_* forms); P31-16: on 3.1 the build has 13 TUs (the evalfast
#   revert forms r12/r15e/fc/nest/fc_nest/nohoist/fc_nohoist), and a metallib without one of them
#   fails to load, so the list is no longer fixed. Env, per TU name (file name without .metal):
#   SRC_<TU>=<file>   compile this source instead of the generated one (a quoted #include of a
#                     sibling, e.g. eval_check_ef_nest.metal's "eval_check_ef.metal", resolves
#                     next to <file> first)
#   FLAGS_<TU>="..."  extra metal flags for that TU (e.g. FLAGS_witgen_tu="-Os")
#   INCS_<TU>="dir..." include dirs searched BEFORE the build's (P31-16: a generator variant's
#                     witgen_fpext.h / witgen_fp.h are included from witgen.h, so a SRC_ next to
#                     them is not enough: without this the build's copies in OUT_DIR/metal win)
# Every TU gets -std=metal3.2 -O2 first (as the build script), so a later -O flag wins.
set -euo pipefail
OUT=$1; DST=$2; mkdir -p "$DST"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SYS=$ROOT/risc0/circuit/rv32im-sys/kernels/metal
GEN=$OUT/metal
[ -d "$GEN" ] || { echo "build_rv32im_variant.sh: no $GEN (build rv32im-sys with --features metal)" >&2; exit 2; }
common=(-std=metal3.2 -O2 -I "$OUT/_sys_" -Wno-unused-variable -I "$SYS" -I "$GEN")
tus=(misc)
for tu in rv32im_incs accum_tu witgen_tu eval_check eval_check_ef; do
  [ -e "$GEN/$tu.metal" ] && tus+=("$tu")
done
for f in "$GEN"/*.metal; do
  tu=$(basename "${f%.metal}")
  case " ${tus[*]} " in *" $tu "*) ;; *) tus+=("$tu") ;; esac
done
airs=()
for tu in "${tus[@]}"; do
  src_var="SRC_$tu"; flags_var="FLAGS_$tu"
  src=${!src_var:-}
  if [ -z "$src" ]; then
    if [ $tu = misc ]; then src=$SYS/misc.metal; else src=$GEN/$tu.metal; fi
  fi
  extra=(${!flags_var:-})
  incs_var="INCS_$tu"; pre=()
  for d in ${!incs_var:-}; do pre+=(-I "$d"); done
  rm -f "$DST/$tu.air"
  xcrun -sdk macosx metal -c "$src" ${pre[@]+"${pre[@]}"} "${common[@]}" ${extra[@]+"${extra[@]}"} -o "$DST/$tu.air" 2> "$DST/$tu.err" &
  airs+=("$DST/$tu.air")
done
wait
for a in "${airs[@]}"; do
  [ -s "$a" ] || { echo "build_rv32im_variant.sh: compile failed: $a" >&2; cat "${a%.air}.err" >&2; exit 3; }
done
xcrun -sdk macosx metallib "${airs[@]}" -o "$DST/rv32im.metallib"
echo "build_rv32im_variant.sh: ${#tus[@]} TUs (${tus[*]})" >&2
ls -la "$DST/rv32im.metallib"
