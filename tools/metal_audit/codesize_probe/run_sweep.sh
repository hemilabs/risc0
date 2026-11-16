#!/usr/bin/env bash
# run_sweep.sh LIB TOTAL ROWS TGMEM SIZES SESSIONS ROUNDS LOG [MAXWAIT]
# Runs SESSIONS short interleaved bench sessions; before each, waits up to MAXWAIT s (default 120)
# for an idle GPU and records whether it got one. Pool them with agg.py [--idle-only] LOG.
set -u
D=$(cd "$(dirname "$0")" && pwd)
LIB=$1; TOTAL=$2; ROWS=$3; TG=$4; SIZES=$5; SESS=$6; ROUNDS=$7; LOG=$8; MAXW=${9:-120}
BENCH=${BENCH:-$(dirname "$LIB")/bench}
: > "$LOG"
for s in $(seq 1 "$SESS"); do
  w=$(python3 "$D/wait_idle.py" "$MAXW")
  echo "R21CS session $s $(date +%H:%M:%S) load $(sysctl -n vm.loadavg) wait: $w" >> "$LOG"
  "$BENCH" "$LIB" "$TOTAL" "$ROWS" "$ROUNDS" "$TG" "$SIZES" >> "$LOG" 2>&1
done
