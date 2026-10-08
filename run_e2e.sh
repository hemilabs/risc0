#!/bin/bash
# Intel Arc Pro B70 end-to-end prover validation suite.
#
# Runs prove_e2e in each receipt kind that the Intel build is expected to
# support after Blocker 1 (hip gating — already resolved) and Blocker 2
# (prove_e2e example — now in place).
#
#   composite — per-segment STARK proving only                 (baseline)
#   succinct  — composite + lift + join over all segments      (lift+join path)
#   groth16   — succinct + Docker shrink_wrap to BN254         (Docker-backed)
#
# Usage: ./run_e2e.sh [iters] [po2] [modes...]
#   iters   default 2_900_000 (gives ~3 segments at po2=20)
#   po2     default 20
#   modes   space-separated subset of {composite,succinct,groth16}
#           default "composite succinct groth16"
#
# Exits non-zero if any mode fails. Each mode's output is captured to a
# per-mode log under $RISC0_E2E_LOG_DIR (default: ${TMPDIR:-/tmp}/risc0-e2e).
set -o pipefail

ITERS=${1:-2900000}
PO2=${2:-20}
shift $(( $# > 2 ? 2 : $# ))
MODES=${*:-composite succinct groth16}

cd "$(dirname "$0")"
# Intel's setvars.sh is not clean under `set -u`, so source it before enabling.
# setvars.sh returns 3 when the environment is already set up.
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
set -u

# Glob the risc0-sys build out dir so the hash isn't hardcoded.
RISC0_SYS_OUT=$(ls -td "$(pwd)"/target/release/build/risc0-sys-*/out 2>/dev/null | paste -sd ":" -)
export LD_LIBRARY_PATH="$(pwd)/target/release/intel_recursion_cache:$(pwd)/target/release/intel_rv32im_cache_default:$(pwd)/target/intel-circuit-cache:${RISC0_SYS_OUT}:${LD_LIBRARY_PATH:-}"
export NEO_CACHE_PERSISTENT=1
export SYCL_CACHE_PERSISTENT=1
export IGC_TotalGRFNum=256

BIN=./target/release/examples/prove_e2e
if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found. Build first:"
    echo "  source /opt/intel/oneapi/setvars.sh"
    echo "  cargo build --release -p risc0-zkvm --features intel --example prove_e2e"
    exit 10
fi

OUTDIR=${RISC0_E2E_LOG_DIR:-${TMPDIR:-/tmp}/risc0-e2e}
mkdir -p "$OUTDIR"

summary=()
overall=0
for mode in $MODES; do
    log="$OUTDIR/${mode}.log"
    echo "=== e2e: ${mode} (iters=${ITERS}, po2=${PO2}) ==="
    t0=$(date +%s.%N)
    if "$BIN" "$ITERS" "$mode" "$PO2" 2>&1 | tee "$log"; then
        rc=0
    else
        rc=${PIPESTATUS[0]}
    fi
    t1=$(date +%s.%N)
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')
    if [ "$rc" -eq 0 ]; then
        summary+=("PASS  ${mode}  wall=${wall}s")
    else
        summary+=("FAIL  ${mode}  wall=${wall}s  rc=${rc}  log=${log}")
        overall=1
    fi
    echo
done

echo "=============================="
echo "e2e test suite summary:"
for line in "${summary[@]}"; do
    echo "  $line"
done
echo "=============================="
exit "$overall"
