#!/bin/bash
# Repeatable Intel B70 busy-loop bench wrapper.
set -e
ITERS=${1:-2900000}
PO2=${2:-20}
# Consume the two positional args before iterating any remaining KEY=VAL.
if [ $# -ge 2 ]; then shift 2; elif [ $# -ge 1 ]; then shift; fi
# setvars.sh returns 3 when the environment is already set up.
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
# Glob the risc0-sys build out dir(s) rather than hardcoding build-hash paths —
# the hash changes on any build.rs edit, which silently broke the wrapper.
RISC0_SYS_OUT=$(ls -td "$(pwd)"/target/release/build/risc0-sys-*/out 2>/dev/null | paste -sd ":" -)
export LD_LIBRARY_PATH="$(pwd)/target/release/intel_recursion_cache:$(pwd)/target/release/intel_rv32im_cache_default:$(pwd)/target/intel-circuit-cache:${RISC0_SYS_OUT}:$LD_LIBRARY_PATH"
export NEO_CACHE_PERSISTENT=1
export SYCL_CACHE_PERSISTENT=1
export IGC_TotalGRFNum=256
for kv in "$@"; do export "$kv"; done
./target/release/examples/prove_busy "$ITERS" "$PO2" 2>&1
