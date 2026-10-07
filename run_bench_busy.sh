#!/bin/bash
# Repeatable Intel B70 busy-loop bench wrapper.
set -e
ITERS=${1:-2900000}
shift || true
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
export LD_LIBRARY_PATH="$(pwd)/target/release/intel_recursion_cache:$(pwd)/target/release/intel_rv32im_cache_default:$(pwd)/target/intel-circuit-cache:$(pwd)/target/release/build/risc0-sys-c50db953e109914f/out:$(pwd)/target/release/build/risc0-sys-cba09c05130c0c4b/out:$LD_LIBRARY_PATH"
export NEO_CACHE_PERSISTENT=1
export SYCL_CACHE_PERSISTENT=1
export IGC_TotalGRFNum=256
for kv in "$@"; do export "$kv"; done
./target/release/examples/prove_busy "$ITERS" 2>&1
