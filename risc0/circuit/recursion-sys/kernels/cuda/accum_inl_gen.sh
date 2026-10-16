#!/bin/sh
# Regenerate step_{compute,verify}_accum_inl.cuh (included by accum_inl.cu, CUDA only) after
# regenerating step_compute_accum.cu / step_verify_accum.cu.
cd "$(dirname "$0")"
sed 's/^__device__ void step_compute_accum(/static __device__ __forceinline__ void step_compute_accum_inl(/' step_compute_accum.cu > step_compute_accum_inl.cuh
sed 's/^__device__ void step_verify_accum(/static __device__ __forceinline__ void step_verify_accum_inl(/' step_verify_accum.cu > step_verify_accum_inl.cuh
