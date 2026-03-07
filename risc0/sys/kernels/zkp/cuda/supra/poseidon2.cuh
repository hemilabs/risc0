#include "poseidon2_constants.cuh"

#define CELLS 24
#define ROUNDS_FULL 8
#define ROUNDS_HALF_FULL (ROUNDS_FULL / 2)
#define ROUNDS_PARTIAL 21
#define ROW_SIZE (CELLS + ROUNDS_PARTIAL)
#define CELLS_RATE 16
#define CELLS_OUT 8

struct __align__(CELLS_OUT % 4 == 0 ? 16 : 4) poseidon_out_t {
  fr_t data[CELLS_OUT];
};

struct __align__(CELLS_RATE % 4 == 0 ? 16 : 4) poseidon_in_t {
  fr_t data[CELLS_RATE];
};

namespace poseidon2 {

__device__ __forceinline__ void pow7_cells_full(fr_t cells[CELLS]) {
#pragma unroll
  for (uint32_t i = 0; i < CELLS; i++) {
    cells[i] ^= 7;
  }
}

__device__ __forceinline__ void multiply_by_m_int(fr_t cells[CELLS]) {
  fr_t sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < CELLS; i++) {
    sum += cells[i];
  }

#pragma unroll
  for (uint32_t i = 0; i < CELLS; i++) {
    cells[i] = sum + M_INT_DIAG_HZN[i] * cells[i];
  }
}

__device__ __forceinline__ void multiply_by_4x4_circulant(fr_t x[4]) {
  // See appendix B of Poseidon2 paper.
  fr_t t0 = x[0] + x[1];
  fr_t t1 = x[2] + x[3];
  fr_t t2 = x[1] + x[1] + t1;
  fr_t t3 = x[3] + x[3] + t0;
  // Replace fr_t(4)*x with (x+x)+(x+x): 3 full-rate adds vs 3 quarter-rate muls
  fr_t t1_2 = t1 + t1;
  fr_t t0_2 = t0 + t0;
  fr_t t4 = t1_2 + t1_2 + t3;
  fr_t t5 = t0_2 + t0_2 + t2;
  fr_t t6 = t3 + t5;
  fr_t t7 = t2 + t4;
  x[0] = t6;
  x[1] = t5;
  x[2] = t7;
  x[3] = t4;
}

#ifdef __HIPCC__
// ROCm 7.2 clang-22 gfx1201 miscompilation workaround:
// The circulant LOOP miscompiles, but this fully-unrolled, copy-based
// approach with no loop is safe to forceinline (no loop = no miscompile).
// forceinline eliminates scratch save/restore overhead (~9 noinline calls
// per poseidon2_mix, each saving/restoring ~26 VGPRs to scratch memory).
__device__ __forceinline__ void multiply_by_m_ext(fr_t cells[CELLS]) {
  fr_t s0{0u}, s1{0u}, s2{0u}, s3{0u}; // tmp_sums

  // Macro: apply 4x4 circulant on group G, accumulate sums
  #define DO_GROUP(G) do { \
    fr_t a = cells[(G)*4+0], b = cells[(G)*4+1]; \
    fr_t c = cells[(G)*4+2], d = cells[(G)*4+3]; \
    fr_t t0 = a + b, t1 = c + d; \
    fr_t t2 = b + b + t1, t3 = d + d + t0; \
    fr_t t1_2 = t1 + t1, t0_2 = t0 + t0; \
    fr_t t4 = t1_2 + t1_2 + t3, t5 = t0_2 + t0_2 + t2; \
    cells[(G)*4+0] = t3 + t5; cells[(G)*4+1] = t5; \
    cells[(G)*4+2] = t2 + t4; cells[(G)*4+3] = t4; \
    s0 += cells[(G)*4+0]; s1 += cells[(G)*4+1]; \
    s2 += cells[(G)*4+2]; s3 += cells[(G)*4+3]; \
  } while(0)

  DO_GROUP(0); DO_GROUP(1); DO_GROUP(2);
  DO_GROUP(3); DO_GROUP(4); DO_GROUP(5);
  #undef DO_GROUP

  // Add accumulated sums (unrolled to avoid any loop miscompilation risk)
#pragma unroll
  for (uint32_t i = 0; i < CELLS; i += 4) {
    cells[i+0] += s0; cells[i+1] += s1;
    cells[i+2] += s2; cells[i+3] += s3;
  }
}
#else
__device__ __forceinline__ void multiply_by_m_ext(fr_t cells[CELLS]) {
  // Optimized method for multiplication by M_EXT.
  // See appendix B of Poseidon2 paper for additional details.
  fr_t tmp_sums[4] = {0};

  for (uint32_t i = 0; i < CELLS / 4; i++) {
    fr_t* tmp = cells + i * 4;
    multiply_by_4x4_circulant(tmp);

    for (uint32_t j = 0; j < 4; j++) {
      tmp_sums[j] += tmp[j];
    }
  }

  for (uint32_t i = 0; i < CELLS; i++) {
    cells[i] += tmp_sums[i % 4];
  }
}
#endif

__device__ __forceinline__ void full_round(fr_t cells[CELLS], uint32_t round_constants_off) {
#pragma unroll
  for (uint32_t i = 0; i < CELLS; i++) {
    cells[i] += ROUND_CONSTANTS[round_constants_off + i];
  }
  pow7_cells_full(cells);
  multiply_by_m_ext(cells);
}

__device__ __forceinline__ void partial_round(fr_t cells[CELLS], uint32_t round_constants_off) {
  cells[0] += ROUND_CONSTANTS[round_constants_off];
  cells[0] ^= 7;
  multiply_by_m_int(cells);
}

#ifdef __HIPCC__
// ROCm 7.2 clang-22 gfx1201: forceinline eliminates scratch save/restore
// overhead from noinline function calls. With multiply_by_m_ext already
// forceinline, poseidon2_mix inlining avoids one more scratch round-trip.
__device__ __forceinline__ void poseidon2_mix(fr_t cells[CELLS]) {
#else
__device__ __forceinline__ void poseidon2_mix(fr_t cells[CELLS]) {
#endif
  uint32_t round_constants_off = 0;

  // First linear layer.
  multiply_by_m_ext(cells);

  // First half full rounds
#pragma unroll 1
  for (uint32_t i = 0; i < ROUNDS_HALF_FULL; i++) {
    full_round(cells, round_constants_off);
    round_constants_off += CELLS;
  }

  // Partial rounds
#pragma unroll 1
  for (uint32_t i = 0; i < ROUNDS_PARTIAL; i++) {
    partial_round(cells, round_constants_off);
    round_constants_off++;
  }

  // Second half full rounds
#pragma unroll 1
  for (uint32_t i = 0; i < ROUNDS_HALF_FULL; i++) {
    full_round(cells, round_constants_off);
    round_constants_off += CELLS;
  }
}

} // namespace poseidon2

#ifdef __HIPCC__
// MI300X (CDNA3, gfx942): 512 VGPRs/SIMD, ~50 VGPRs/thread for Poseidon2.
// 512 threads = 8 wavefronts/block × 2 blocks/CU = 16 wavefronts/CU (40% occ)
// vs 256 threads × 3 = 12 wavefronts/CU (30% occ).
__launch_bounds__(512, 2)
#else
__launch_bounds__(256, 3)
#endif
__global__
    void _poseidon2_fold(poseidon_out_t* output, const poseidon_in_t* input, uint32_t output_size) {
  uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

  fr_t cells[CELLS];
#pragma unroll
  for (uint32_t i = 0; i < CELLS; i++) {
    cells[i] = 0;
  }

  poseidon_in_t in = input[gid];
#pragma unroll
  for (size_t i = 0; i < CELLS_RATE; i++) {
    cells[i] = in.data[i];
  }

  poseidon2::poseidon2_mix(cells);
  poseidon_out_t tmp;

#pragma unroll
  for (uint32_t i = 0; i < CELLS_OUT; i++) {
    tmp.data[i] = cells[i];
  }
  output[gid] = tmp;
}

#ifdef __HIPCC__
__launch_bounds__(512, 2)
#else
__launch_bounds__(256, 3)
#endif
__global__
    void _poseidon2_rows(poseidon_out_t* out, const fr_t* matrix, uint32_t dim_x, uint32_t dim_y) {
  uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid >= dim_x)
    return;

  fr_t cells[CELLS];
#pragma unroll
  for (uint32_t k = 0; k < CELLS; k++)
    cells[k] = 0;

  matrix += gid;
  uint32_t i = 0;
  fr_t zero(0u);

  do {
    for (uint32_t j = 0; j < CELLS_RATE; j++, i++)
      cells[j] = i < dim_y ? matrix[i * dim_x] : zero;

    poseidon2::poseidon2_mix(cells);
  } while (i < dim_y);

  poseidon_out_t tmp;
#pragma unroll
  for (uint32_t i = 0; i < CELLS_OUT; i++) {
    tmp.data[i] = cells[i];
  }

  out[gid] = tmp;
}
