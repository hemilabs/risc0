#!/usr/bin/env bash
# N1 step 2: the chain-shape A/B (chain.metal) with the host from build.sh, 9 interleaved rounds per
# process; ch_scalar runs twice (A, A2) and every tensor form is compared word for word with it.
#   run_chain.sh <build dir> [rounds]
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); O=${1:?build dir}; R=${2:-9}
OB=$((8192 * 128 * 16)); sp() { echo "$1:128:8192:150:$OB:16777216${2:+:$2}"; }
for bl in 8 16 32; do [ -f "$O/tab_L32_BL$bl.bin" ] || python3 "$D/tables.py" "$O/tab_L32_BL$bl.bin" 32 $bl 150; done
uptime
for lib in chain chain_noload; do
  for bl in 8 16 32; do
    extra=""; [ $bl = 16 ] && extra=$(sp ch_mm4_16 x)
    echo "== $lib BL=$bl"
    N1_INB=$O/tab_L32_BL$bl.bin "$O/host" "$O/$lib.metallib" "$R" $(sp ch_scalar x) $(sp ch_mm_$bl x) $extra \
      $(sp ch_none) $(sp ch_scalar x) | grep -E "TIME|CHK"
  done
done
uptime
