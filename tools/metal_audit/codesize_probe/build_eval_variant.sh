#!/usr/bin/env bash
# build_eval_variant.sh OUT_DIR NAME SRC.metal [metal flags...] -> OUT_DIR/NAME.metallib
# Compiles an evalfast TU variant like rv32im-sys's build (-std=metal3.2 -O2, the build's include
# path). Needs a built risc0-circuit-rv32im-sys (its OUT_DIR provides the generated headers).
set -euo pipefail
WT=$(cd "$(dirname "$0")/../../.." && pwd)
SYSOUT=${SYSOUT:-$(ls -td "$WT"/target/debug/build/risc0-circuit-rv32im-sys-*/out | head -1)}
dst=$1; name=$2; src=$3; shift 3
mkdir -p "$dst"
t0=$(date +%s)
xcrun -sdk macosx metal -std=metal3.2 -O2 -Wno-unused-variable -I "$SYSOUT/_sys_" \
  -I "$WT/risc0/circuit/rv32im-sys/kernels/metal" -I "$SYSOUT/metal" "$@" -c "$src" -o "$dst/$name.air"
xcrun -sdk macosx metallib "$dst/$name.air" -o "$dst/$name.metallib"
echo "$name: $(( $(date +%s) - t0 )) s, $(stat -f %z "$dst/$name.metallib") B"
