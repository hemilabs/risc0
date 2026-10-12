#!/bin/bash
# Fast eval_check thread count sweep.
# Usage: ./test_eval_check.sh <THREADS>
# Example: ./test_eval_check.sh 128
set -e

THREADS=${1:-256}
echo "=== Testing eval_check with ${THREADS} threads/block ==="

export PATH="/usr/local/cuda-13.1/bin:/home/max/.cargo/bin:/usr/bin:/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH"

SRCDIR="/home/max/poseidon/risc0-v5/risc0/circuit/rv32im-sys/kernels/cuda"
OUTDIR="/home/max/poseidon/risc0-v5/target/release/build/risc0-circuit-rv32im-sys-4631fae89546b619/out"
CUDA_ROOT="/home/max/poseidon/risc0-v5/risc0/sys/kernels/zkp/cuda"
CXX_ROOT="/home/max/poseidon/risc0-v5/risc0/sys/cxx"
SPPARK_ROOT="/home/max/risc0-sppark-upstream"
OBJ="${OUTDIR}/eval_check_combined_standalone.o"
ARCHIVE="${OUTDIR}/librisc0_rv32im_cuda.a"

echo "Compiling eval_check_combined.cu with EVAL_CHECK_THREADS=${THREADS}..."
time nvcc \
  -ccbin=c++ -std=c++17 \
  -Xcompiler "-O3,-ffunction-sections,-fdata-sections,-fPIC" \
  -Xcompiler "-Wno-unused-function,-Wno-unused-parameter" \
  -m64 -Xptxas -O3 -Xptxas=-v \
  -diag-suppress=177 -diag-suppress=550 -diag-suppress=2922 \
  -I "${CUDA_ROOT}" -I "${CXX_ROOT}" -I "${SPPARK_ROOT}" \
  -arch=native \
  -DEVAL_CHECK_THREADS=${THREADS} \
  -c "${SRCDIR}/eval_check_combined.cu" \
  -o "${OBJ}"

echo "Updating archive..."
ar rcs "${ARCHIVE}" "${OBJ}"

echo "Relinking benchmark..."
# Touch a source file to force cargo to relink (but not recompile CUDA)
touch /home/max/poseidon/risc0-v5/risc0/zkvm/benches/fib.rs

export SPPARK_ROOT="/home/max/risc0-sppark-upstream"
cargo bench --manifest-path /home/max/poseidon/risc0-v5/risc0/zkvm/Cargo.toml \
  --bench fib --features cuda,prove -- prove 2>&1 | tail -100

echo "=== Done: ${THREADS} threads ==="
