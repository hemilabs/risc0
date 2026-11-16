// Copyright 2022 Risc0, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// mix_poly_coeffs for Metal. Ports of sys/kernels/zkp/cuda/kernels.cu: mix_poly_coeffs (v1 with
// a shared mix-power table, lines 158-185) and mix_poly_coeffs_v2 (lines 389-455). The ROCm-only
// v6 is not ported. Host selection: zkp/src/hal/metal/eval.rs (ffi.cu
// risc0_zkp_cuda_mix_poly_coeffs, adjusted for the 32 KB threadgroup memory limit):
//   inputSize <= 1638          mix_poly_coeffs_v2        (20 B/input: powers + combo ids)
//   1638 < inputSize <= 2048   mix_poly_coeffs_v2_dcids  (16 B/input: powers in TG, combo ids
//                                                         read from device; measured ~1.6x faster
//                                                         than mix_poly_coeffs_v1, which is kept
//                                                         as the plan's v1-with-table alternative)
//   inputSize > 2048           mix_poly_coeffs           (no table; per-thread running power)
// All variants are exact field arithmetic with canonical outputs, so bit-identical to each other
// and to the CPU HAL.
//
// Bindings (all kernels): 0 out (FpExt[count * nCombos]), 1 in (Fp[count * inputSize]),
//   2 combos (u32[inputSize]), 3 MixParams. Table kernels: threadgroup(0) = the table
//   (v2: 20 B * inputSize, v1/v2_dcids: 16 B * inputSize); grid = ceil(count/256) groups x 256.
//   mix_poly_coeffs: dispatch_threads(count).

// R17b (docs/metal-port/notes/R17.md, "R17b hardening"): FpExt * FpExt is always_inline in this
// TU. Declared before fpext.h so that the friend definition there inherits the attribute. At -O2
// the compiler otherwise calls it out of line inside this file's FpExt loops (ir_scan.py tier A:
// a default-constructed private FpExt / FpExt array written after the call and read after the
// loop, the Apple GPU lost-store shape). Same arithmetic, bit-exact.
#include "fp.h"
struct FpExt;
__attribute__((always_inline)) constexpr FpExt operator*(FpExt lhs, FpExt rhs);
#include "fp.h"
#include "fpext.h"

// R17b: FpExt pow for this TU: fpext.h's pow, with `tot *= x; x *= x` spelled through the
// always_inline FpExt * FpExt above (fpext.h's FpExt::operator*=(FpExt) stays an out-of-line
// call at -O2 here, inside the caller's loops). Same square-and-multiply, bit-exact.
static inline __attribute__((always_inline)) FpExt pow_r17b(FpExt x, size_t n) {
  FpExt tot(1);
  while (n != 0) {
    if (n % 2 == 1) {
      tot = tot * x;
    }
    n = n / 2;
    x = x * x;
  }
  return tot;
}

using namespace metal;

struct MixParams {
  uint32_t mixStart[4];  // FpExt, raw Montgomery words
  uint32_t mix[4];       // FpExt, raw Montgomery words
  uint32_t inputSize;
  uint32_t count;
  uint32_t _pad[2];
};

inline FpExt mx_load_ext(constant uint32_t* w) {
  return FpExt(Fp::fromRaw(w[0]), Fp::fromRaw(w[1]), Fp::fromRaw(w[2]), Fp::fromRaw(w[3]));
}

// ---- table-less (inputSize > 2048) ----
kernel void mix_poly_coeffs(device FpExt* out [[buffer(0)]],
                            const device Fp* in [[buffer(1)]],
                            const device uint32_t* combos [[buffer(2)]],
                            constant MixParams& p [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
  const uint32_t count = p.count;
  if (gid >= count) {
    return;
  }
  const uint32_t inputSize = p.inputSize;
  FpExt cur = mx_load_ext(p.mixStart);
  FpExt mix = mx_load_ext(p.mix);
  for (uint32_t i = 0; i < inputSize; i++) {
    size_t id = combos[i];
    out[count * id + gid] += cur * Fp(in[(size_t)count * i + gid]);
    cur = cur * mix;
  }
}

// ---- v1 with table (CUDA mix_poly_coeffs) ----
// Thread 0 computes all mix powers once into threadgroup memory.
[[max_total_threads_per_threadgroup(256)]]
kernel void mix_poly_coeffs_v1(device FpExt* out [[buffer(0)]],
                               const device Fp* in [[buffer(1)]],
                               const device uint32_t* combos [[buffer(2)]],
                               constant MixParams& p [[buffer(3)]],
                               threadgroup uint32_t* mixPowsRaw [[threadgroup(0)]],
                               uint3 blockIdx [[threadgroup_position_in_grid]],
                               uint3 threadIdx [[thread_position_in_threadgroup]],
                               uint3 blockDim [[threads_per_threadgroup]]) {
  const uint32_t inputSize = p.inputSize;
  const uint32_t count = p.count;
  if (threadIdx.x == 0) {
    FpExt cur = mx_load_ext(p.mixStart);
    FpExt m = mx_load_ext(p.mix);
    for (uint32_t i = 0; i < inputSize; i++) {
      for (int k = 0; k < 4; k++) {
        mixPowsRaw[4 * i + k] = cur.elems[k].asRaw();
      }
      cur = cur * m;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    for (size_t i = 0; i < inputSize; i++) {
      size_t id = combos[i];
      FpExt m(Fp::fromRaw(mixPowsRaw[4 * i + 0]),
              Fp::fromRaw(mixPowsRaw[4 * i + 1]),
              Fp::fromRaw(mixPowsRaw[4 * i + 2]),
              Fp::fromRaw(mixPowsRaw[4 * i + 3]));
      out[count * id + idx] += m * Fp(in[count * i + idx]);
    }
  }
}

// ---- v2 ----
// One RMW of `out` per run of equal combo ids (instead of one per input), lazy pair REDC for the
// FpExt x Fp products, parallel mix-power table in threadgroup memory. TG_CIDS: combo ids are
// copied to threadgroup memory too (CUDA v2); otherwise read from device (open question 4).

inline uint32_t mx_addmod(uint32_t a, uint32_t b) {
  uint32_t r = a + b;
  return r >= Fp::P ? r - Fp::P : r;
}

// (a0*b0 + a1*b1) * 2^-32 mod p for inputs < p (same REDC as Fp::mul).
inline uint32_t mx_redc_pair(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1) {
  uint64_t o64 = (uint64_t)a0 * b0 + (uint64_t)a1 * b1;
  uint32_t low = -uint32_t(o64);
  uint32_t red = Fp::M * low;
  o64 += uint64_t(red) * uint64_t(Fp::P);
  uint32_t ret = o64 >> 32;
  return ret >= Fp::P ? ret - Fp::P : ret;
}

template <bool TG_CIDS>
[[max_total_threads_per_threadgroup(256)]]
kernel void mix_poly_coeffs_v2_t(device FpExt* out [[buffer(0)]],
                                 const device Fp* in [[buffer(1)]],
                                 const device uint32_t* combos [[buffer(2)]],
                                 constant MixParams& p [[buffer(3)]],
                                 threadgroup uint32_t* mx_smem [[threadgroup(0)]],
                                 uint3 blockIdx [[threadgroup_position_in_grid]],
                                 uint3 threadIdx [[thread_position_in_threadgroup]],
                                 uint3 blockDim [[threads_per_threadgroup]]) {
  const uint32_t inputSize = p.inputSize;
  const uint32_t count = p.count;
  threadgroup uint32_t* mixPows = mx_smem;
  threadgroup uint32_t* tg_cids = mx_smem + 4 * inputSize;
  {
    FpExt ms = mx_load_ext(p.mixStart);
    FpExt m = mx_load_ext(p.mix);
    for (uint32_t i = threadIdx.x; i < inputSize; i += blockDim.x) {
      FpExt v = ms * pow_r17b(m, (size_t)i);
      for (int k = 0; k < 4; k++) {
        mixPows[4 * i + k] = v.elems[k].asRaw();
      }
      if (TG_CIDS) {
        tg_cids[i] = combos[i];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  const device uint32_t* in32 = reinterpret_cast<const device uint32_t*>(in);
  uint32_t acc[4] = {0, 0, 0, 0};
  uint32_t i = 0;
  while (i < inputSize) {
    uint32_t id = TG_CIDS ? tg_cids[i] : combos[i];
    bool pair = (i + 1 < inputSize) && ((TG_CIDS ? tg_cids[i + 1] : combos[i + 1]) == id);
    uint32_t c0 = in32[(size_t)count * i + idx];
    uint32_t c1 = pair ? in32[(size_t)count * (i + 1) + idx] : 0;
    uint32_t i1 = pair ? i + 1 : i;
    for (int k = 0; k < 4; k++) {
      acc[k] = mx_addmod(acc[k],
                         mx_redc_pair(c0, mixPows[4 * i + k], c1, mixPows[4 * i1 + k]));
    }
    i += pair ? 2 : 1;
    if (i >= inputSize || (TG_CIDS ? tg_cids[i] : combos[i]) != id) {
      device FpExt* o = out + (size_t)count * id + idx;
      FpExt v = *o;
      for (int k = 0; k < 4; k++) {
        v.elems[k] = v.elems[k] + Fp::fromRaw(acc[k]);
        acc[k] = 0;
      }
      *o = v;
    }
  }
}

template [[host_name("mix_poly_coeffs_v2")]] kernel void
mix_poly_coeffs_v2_t<true>(device FpExt*,
                           const device Fp*,
                           const device uint32_t*,
                           constant MixParams&,
                           threadgroup uint32_t*,
                           uint3,
                           uint3,
                           uint3);

template [[host_name("mix_poly_coeffs_v2_dcids")]] kernel void
mix_poly_coeffs_v2_t<false>(device FpExt*,
                            const device Fp*,
                            const device uint32_t*,
                            constant MixParams&,
                            threadgroup uint32_t*,
                            uint3,
                            uint3,
                            uint3);

// ---- Z15 fused, write-once mix (LEADS #22 segtail-mix-wide) ----
// One dispatch mixes every input of every group (accum, code, data, check: up to 4 coefficient
// buffers) into the combos buffer, which it writes exactly once: no zero fill, no
// read-modify-write. The host sorts the inputs by combo id; entry e of combo c (entries
// [comboBegin[c], comboBegin[c+1])) names its source (buffer << 28 | poly) and its mix-power
// index. Each thread owns 4 consecutive rows (one 16-byte load per input), accumulates
// sum_e mixPow[e] * in_e[row] per row with one REDC per two products (as v2), and stores the 4
// FpExt of each combo. Exact field arithmetic with canonical outputs: bit-identical to zeroed
// combos + mix_poly_coeffs per group (the sum order does not matter).
//
// Bindings: 0 out (FpExt[count * nCombos], as uint4), 1-4 in0..in3 (Fp[count * polys], as uint4;
// unused slots bind any buffer), 5 pows (uint4 raw FpExt per power), 6 entries (uint2: src,
// pow), 7 comboBegin (u32[nCombos + 1]), 8 MixFusedParams. Grid: dispatch_threads(count / 4).
// R17: no calls (every helper is always_inline) and the accumulators are scalars indexed only
// by fully unrolled constant loops.

struct MixFusedParams {
  uint32_t count;   // rows per polynomial (a multiple of 4)
  uint32_t nCombos; // combo slots in out
  uint32_t _pad[2];
};

static inline __attribute__((always_inline)) uint32_t mf_addmod(uint32_t a, uint32_t b) {
  uint32_t r = a + b;
  return r >= Fp::P ? r - Fp::P : r;
}

static inline __attribute__((always_inline)) uint32_t
mf_redc_pair(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1) {
  uint64_t o64 = (uint64_t)a0 * b0 + (uint64_t)a1 * b1;
  uint32_t low = -uint32_t(o64);
  uint32_t red = Fp::M * low;
  o64 += uint64_t(red) * uint64_t(Fp::P);
  uint32_t ret = o64 >> 32;
  return ret >= Fp::P ? ret - Fp::P : ret;
}

#define MF_ACC_STEP(W, V0, V1, M0, M1)                                                         \
  a##W##0 = mf_addmod(a##W##0, mf_redc_pair(V0, M0.x, V1, M1.x));                              \
  a##W##1 = mf_addmod(a##W##1, mf_redc_pair(V0, M0.y, V1, M1.y));                              \
  a##W##2 = mf_addmod(a##W##2, mf_redc_pair(V0, M0.z, V1, M1.z));                              \
  a##W##3 = mf_addmod(a##W##3, mf_redc_pair(V0, M0.w, V1, M1.w));

static inline __attribute__((always_inline)) const device uint4*
mf_src(uint32_t src,
       uint32_t count4,
       const device uint4* in0,
       const device uint4* in1,
       const device uint4* in2,
       const device uint4* in3) {
  uint32_t b = src >> 28;
  const device uint4* base = b == 0 ? in0 : b == 1 ? in1 : b == 2 ? in2 : in3;
  return base + (size_t)(src & 0x0fffffffu) * count4;
}

kernel void mix_poly_coeffs_fused(device uint4* out [[buffer(0)]],
                                  const device uint4* in0 [[buffer(1)]],
                                  const device uint4* in1 [[buffer(2)]],
                                  const device uint4* in2 [[buffer(3)]],
                                  const device uint4* in3 [[buffer(4)]],
                                  constant uint4* pows [[buffer(5)]],
                                  constant uint2* entries [[buffer(6)]],
                                  constant uint32_t* comboBegin [[buffer(7)]],
                                  constant MixFusedParams& p [[buffer(8)]],
                                  uint gid [[thread_position_in_grid]]) {
  const uint32_t count4 = p.count / 4;
  if (gid >= count4) {
    return;
  }
  for (uint32_t c = 0; c < p.nCombos; c++) {
    uint32_t a00 = 0, a01 = 0, a02 = 0, a03 = 0;
    uint32_t a10 = 0, a11 = 0, a12 = 0, a13 = 0;
    uint32_t a20 = 0, a21 = 0, a22 = 0, a23 = 0;
    uint32_t a30 = 0, a31 = 0, a32 = 0, a33 = 0;
    uint32_t e = comboBegin[c];
    const uint32_t end = comboBegin[c + 1];
    for (; e + 1 < end; e += 2) {
      uint2 e0 = entries[e];
      uint2 e1 = entries[e + 1];
      uint4 v0 = mf_src(e0.x, count4, in0, in1, in2, in3)[gid];
      uint4 v1 = mf_src(e1.x, count4, in0, in1, in2, in3)[gid];
      uint4 m0 = pows[e0.y];
      uint4 m1 = pows[e1.y];
      MF_ACC_STEP(0, v0.x, v1.x, m0, m1)
      MF_ACC_STEP(1, v0.y, v1.y, m0, m1)
      MF_ACC_STEP(2, v0.z, v1.z, m0, m1)
      MF_ACC_STEP(3, v0.w, v1.w, m0, m1)
    }
    if (e < end) {
      uint2 e0 = entries[e];
      uint4 v0 = mf_src(e0.x, count4, in0, in1, in2, in3)[gid];
      uint4 m0 = pows[e0.y];
      uint4 z = uint4(0);
      MF_ACC_STEP(0, v0.x, 0u, m0, z)
      MF_ACC_STEP(1, v0.y, 0u, m0, z)
      MF_ACC_STEP(2, v0.z, 0u, m0, z)
      MF_ACC_STEP(3, v0.w, 0u, m0, z)
    }
    device uint4* o = out + (size_t)c * p.count + (size_t)gid * 4;
    o[0] = uint4(a00, a01, a02, a03);
    o[1] = uint4(a10, a11, a12, a13);
    o[2] = uint4(a20, a21, a22, a23);
    o[3] = uint4(a30, a31, a32, a33);
  }
}
