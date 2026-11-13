// 16 BabyBear elements (risc0's Montgomery form, canonical in [0, P)) operated on together:
// AVX-512 as one zmm, AVX2 as two ymm. Bit-identical to risc0::Fp / risc0::FpExt.
#pragma once

#include "fp.h"
#include "fpext.h"

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define SIMD_INLINE __forceinline
#else
#define SIMD_INLINE inline __attribute__((always_inline))
#endif

namespace risc0::simd {
inline namespace RISC0_SIMD_NS {

#if defined(__AVX512F__)

struct V {
  __m512i v;
};
SIMD_INLINE V vset1(uint32_t x) { return {_mm512_set1_epi32(int(x))}; }
SIMD_INLINE V vload(const uint32_t* p) { return {_mm512_loadu_si512(p)}; }
SIMD_INLINE void vstore(uint32_t* p, V a) { _mm512_storeu_si512(p, a.v); }
SIMD_INLINE V vadd(V a, V b) {
  const __m512i p = _mm512_set1_epi32(int(Fp::P));
  __m512i r = _mm512_add_epi32(a.v, b.v);
  return {_mm512_min_epu32(r, _mm512_sub_epi32(r, p))};
}
SIMD_INLINE V vsub(V a, V b) {
  const __m512i p = _mm512_set1_epi32(int(Fp::P));
  __m512i r = _mm512_sub_epi32(a.v, b.v);
  return {_mm512_min_epu32(r, _mm512_add_epi32(r, p))};
}
// Montgomery: (a*b - m*P) / 2^32 with m = lo(a*b) * P^-1, in (-P, P); min() maps into [0, P).
SIMD_INLINE V vmul(V a, V b) {
  const __m512i p = _mm512_set1_epi32(int(Fp::P));
  const __m512i m = _mm512_set1_epi32(int(Fp::M));
  __m512i ao = _mm512_srli_epi64(a.v, 32);
  __m512i bo = _mm512_srli_epi64(b.v, 32);
  __m512i te = _mm512_mul_epu32(a.v, b.v);
  __m512i to = _mm512_mul_epu32(ao, bo);
  __m512i qe = _mm512_mul_epu32(_mm512_mul_epu32(te, m), p);
  __m512i qo = _mm512_mul_epu32(_mm512_mul_epu32(to, m), p);
  // High halves: t_hi - q_hi (the low halves are equal, so no borrow).
  __m512i de = _mm512_srli_epi64(_mm512_sub_epi64(te, qe), 32);
  __m512i dn = _mm512_sub_epi64(to, qo);
  __m512i r = _mm512_mask_blend_epi32(__mmask16(0xAAAA), de, dn);
  return {_mm512_min_epu32(r, _mm512_add_epi32(r, p))};
}

#elif defined(__AVX2__)

struct V {
  __m256i lo, hi;
};
SIMD_INLINE V vset1(uint32_t x) {
  __m256i a = _mm256_set1_epi32(int(x));
  return {a, a};
}
SIMD_INLINE V vload(const uint32_t* p) {
  return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)),
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 8))};
}
SIMD_INLINE void vstore(uint32_t* p, V a) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), a.lo);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + 8), a.hi);
}
SIMD_INLINE __m256i add8(__m256i a, __m256i b) {
  const __m256i p = _mm256_set1_epi32(int(Fp::P));
  __m256i r = _mm256_add_epi32(a, b);
  return _mm256_min_epu32(r, _mm256_sub_epi32(r, p));
}
SIMD_INLINE __m256i sub8(__m256i a, __m256i b) {
  const __m256i p = _mm256_set1_epi32(int(Fp::P));
  __m256i r = _mm256_sub_epi32(a, b);
  return _mm256_min_epu32(r, _mm256_add_epi32(r, p));
}
SIMD_INLINE __m256i mul8(__m256i a, __m256i b) {
  const __m256i p = _mm256_set1_epi32(int(Fp::P));
  const __m256i m = _mm256_set1_epi32(int(Fp::M));
  __m256i ao = _mm256_srli_epi64(a, 32);
  __m256i bo = _mm256_srli_epi64(b, 32);
  __m256i te = _mm256_mul_epu32(a, b);
  __m256i to = _mm256_mul_epu32(ao, bo);
  __m256i qe = _mm256_mul_epu32(_mm256_mul_epu32(te, m), p);
  __m256i qo = _mm256_mul_epu32(_mm256_mul_epu32(to, m), p);
  __m256i de = _mm256_srli_epi64(_mm256_sub_epi64(te, qe), 32);
  __m256i dn = _mm256_sub_epi64(to, qo);
  __m256i r = _mm256_blend_epi32(de, dn, 0xAA);
  return _mm256_min_epu32(r, _mm256_add_epi32(r, p));
}
SIMD_INLINE V vadd(V a, V b) { return {add8(a.lo, b.lo), add8(a.hi, b.hi)}; }
SIMD_INLINE V vsub(V a, V b) { return {sub8(a.lo, b.lo), sub8(a.hi, b.hi)}; }
SIMD_INLINE V vmul(V a, V b) { return {mul8(a.lo, b.lo), mul8(a.hi, b.hi)}; }

#else
#error "fp_simd.h needs AVX2 or AVX-512"
#endif

constexpr size_t kLanes = 16;

struct FpV {
  V v;
  SIMD_INLINE FpV() : v(vset1(0)) {}
  SIMD_INLINE FpV(V v) : v(v) {}
  SIMD_INLINE FpV(Fp x) : v(vset1(x.asRaw())) {}
};

SIMD_INLINE FpV operator+(FpV a, FpV b) { return vadd(a.v, b.v); }
SIMD_INLINE FpV operator-(FpV a, FpV b) { return vsub(a.v, b.v); }
SIMD_INLINE FpV operator*(FpV a, FpV b) { return vmul(a.v, b.v); }
SIMD_INLINE FpV operator-(FpV a) { return vsub(vset1(0), a.v); }

// The 16 cycles starting at the block's first: `idx` is the trace index the scalar code computes
// for that first cycle. A block whose rows wrap around the end of the column is gathered.
SIMD_INLINE FpV ld(const Fp* base, size_t idx, size_t mask) {
  const uint32_t* p = reinterpret_cast<const uint32_t*>(base);
  size_t off = idx & mask;
  if (off + (kLanes - 1) <= mask) {
    return vload(p + idx);
  }
  alignas(64) uint32_t tmp[kLanes];
  size_t col = idx & ~mask;
  for (size_t j = 0; j < kLanes; j++) {
    tmp[j] = p[col + ((off + j) & mask)];
  }
  return vload(tmp);
}

struct FpExtV {
  FpV e[4];
  SIMD_INLINE FpExtV() {}
  SIMD_INLINE FpExtV(FpV a, FpV b, FpV c, FpV d) : e{a, b, c, d} {}
  SIMD_INLINE FpExtV(const FpExt& x) : e{x.elems[0], x.elems[1], x.elems[2], x.elems[3]} {}
};

SIMD_INLINE FpExtV operator+(const FpExtV& a, const FpExtV& b) {
  return {a.e[0] + b.e[0], a.e[1] + b.e[1], a.e[2] + b.e[2], a.e[3] + b.e[3]};
}
SIMD_INLINE FpExtV operator-(const FpExtV& a, const FpExtV& b) {
  return {a.e[0] - b.e[0], a.e[1] - b.e[1], a.e[2] - b.e[2], a.e[3] - b.e[3]};
}
SIMD_INLINE FpExtV operator-(const FpExtV& a) { return {-a.e[0], -a.e[1], -a.e[2], -a.e[3]}; }
SIMD_INLINE FpExtV operator+(const FpExtV& a, FpV b) { return {a.e[0] + b, a.e[1], a.e[2], a.e[3]}; }
SIMD_INLINE FpExtV operator+(FpV a, const FpExtV& b) { return b + a; }
SIMD_INLINE FpExtV operator-(const FpExtV& a, FpV b) { return {a.e[0] - b, a.e[1], a.e[2], a.e[3]}; }
SIMD_INLINE FpExtV operator-(FpV a, const FpExtV& b) { return {a - b.e[0], -b.e[1], -b.e[2], -b.e[3]}; }
SIMD_INLINE FpExtV operator*(const FpExtV& a, FpV b) {
  return {a.e[0] * b, a.e[1] * b, a.e[2] * b, a.e[3] * b};
}
SIMD_INLINE FpExtV operator*(FpV a, const FpExtV& b) { return b * a; }
// A lane value times a constant of the extension (the constraint mix powers): four products.
SIMD_INLINE FpExtV operator*(FpV a, const FpExt& b) {
  return {a * FpV(b.elems[0]), a * FpV(b.elems[1]), a * FpV(b.elems[2]), a * FpV(b.elems[3])};
}
SIMD_INLINE FpExtV operator*(const FpExt& b, FpV a) { return a * b; }
SIMD_INLINE FpExtV operator*(const FpExtV& a, const FpExtV& b) {
  const FpV nbeta = Fp(Fp::P - 11);
  FpV b1n = b.e[1] * nbeta, b2n = b.e[2] * nbeta, b3n = b.e[3] * nbeta;
  return {a.e[0] * b.e[0] + a.e[1] * b3n + a.e[2] * b2n + a.e[3] * b1n,
          a.e[0] * b.e[1] + a.e[1] * b.e[0] + a.e[2] * b3n + a.e[3] * b2n,
          a.e[0] * b.e[2] + a.e[1] * b.e[1] + a.e[2] * b.e[0] + a.e[3] * b3n,
          a.e[0] * b.e[3] + a.e[1] * b.e[2] + a.e[2] * b.e[1] + a.e[3] * b.e[0]};
}
SIMD_INLINE FpExtV operator*(const FpExtV& a, const FpExt& b) { return a * FpExtV(b); }
SIMD_INLINE FpExtV operator*(const FpExt& a, const FpExtV& b) { return FpExtV(a) * b; }

} // inline namespace RISC0_SIMD_NS
} // namespace risc0::simd
