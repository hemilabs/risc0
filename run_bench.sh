#!/bin/bash
# Repeatable Intel B70 prove_bench wrapper.
# Usage: ./run_bench.sh [iters] [extra_env...]
#   iters defaults to 6000 (3 segments)
set -e
ITERS=${1:-6000}
shift || true

source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

# Patched IGC (rebuilt locally, contains PreCompiledFuncImport fix +
# Solinas/mulpair-fusion perf patches). Required for compile-time AOT of
# witgen and for any kernel that uses 256-GRF.
PATCHED_IGC="/tmp/igc_ws/build/IGC/Release"
SPIRV_TRANS="/tmp/igc_ws/SPIRV-LLVM-Translator/build/lib/SPIRV"
LLVM16_LIB="/tmp/llvm16_install/usr/lib/x86_64-linux-gnu"
LLVM16_CORE="/tmp/llvm16_install/usr/lib/llvm-16/lib"

# Pick the most-recently-built risc0-sys cache dir so we link against
# whatever cargo most recently produced (multiple sibling dirs from
# different feature combos are common).
RISC0_SYS_OUT=$(ls -dt $(pwd)/target/release/build/risc0-sys-*/out 2>/dev/null | head -1)

export LD_LIBRARY_PATH="$PATCHED_IGC:$SPIRV_TRANS:$LLVM16_LIB:$LLVM16_CORE:$(pwd)/target/release/intel_recursion_cache:$(pwd)/target/release/intel_rv32im_cache_default:$(pwd)/target/release/intel_rv32im_cache:${RISC0_SYS_OUT}:${LD_LIBRARY_PATH:-}"
export NEO_CACHE_PERSISTENT=1
export SYCL_CACHE_PERSISTENT=1
export IGC_TotalGRFNum=256
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

# Allow caller to override / add env vars by passing KEY=VAL pairs
for kv in "$@"; do
  export "$kv"
done

./target/release/examples/prove_bench "$ITERS" 2>&1
