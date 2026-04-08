#pragma once

#include <cstdint>

// BabyBear NTT twiddle factor tables.
// All values in Montgomery form (sppark convention, non-canonical).
// Extension field polynomial: X^4 + 11 (so X^4 = -11 = NBETA).
//
// forward_roots[k] is a primitive 2^k-th root of unity.
// inverse_roots[k] is the inverse of forward_roots[k].
// domain_inv[k] = Montgomery(1 / 2^k) = inverse of domain size.
//
// From risc0-sppark/ntt/parameters/baby_bear.h (non-canonical, group_gen=3).

namespace ntt {

static constexpr uint32_t MAX_LG_DOMAIN = 27;

// Forward roots of unity in Montgomery form.
// forward_roots[k]^(2^k) = 1 (mod P), forward_roots[k]^(2^(k-1)) != 1.
static constexpr uint32_t forward_roots[MAX_LG_DOMAIN + 1] = {
    0x0ffffffeu, 0x68000003u, 0x5bc72af0u, 0x02ec07f3u,
    0x67e027cau, 0x19e5f901u, 0x3b27e54au, 0x20d1773eu,
    0x771ea53au, 0x0fb182adu, 0x146d1455u, 0x3e7d65f0u,
    0x327884f2u, 0x53fc8703u, 0x20742dd1u, 0x31062edau,
    0x642b70abu, 0x1ccd534bu, 0x03cc9bf7u, 0x6686182fu,
    0x2e2516d3u, 0x5701b5c8u, 0x193a6352u, 0x112fc5b9u,
    0x63ec6b91u, 0x5b34b3ffu, 0x3fff6398u, 0x1ffffedcu
};

// Inverse roots of unity in Montgomery form.
static constexpr uint32_t inverse_roots[MAX_LG_DOMAIN + 1] = {
    0x0ffffffeu, 0x68000003u, 0x1c38d511u, 0x3d85298fu,
    0x5f06e481u, 0x38a3c615u, 0x4ed6e525u, 0x55372b64u,
    0x4d88ae94u, 0x5806fd5eu, 0x2ced6d6au, 0x1851eacdu,
    0x2fa36b4du, 0x0a556a3bu, 0x18ae7209u, 0x742ba568u,
    0x3f462cbau, 0x50b5c3b2u, 0x0dfdfca6u, 0x3821b546u,
    0x45e4cd80u, 0x3e6793bdu, 0x5bdeafa3u, 0x2e01d37au,
    0x2da9f4f0u, 0x1db7e183u, 0x167ca34bu, 0x50b3630au
};

// Domain size inverse: domain_inv[k] = Montgomery(1 / 2^k).
// Used to normalize inverse NTT output.
static constexpr uint32_t domain_inv[MAX_LG_DOMAIN + 1] = {
    0x0ffffffeu, 0x07ffffffu, 0x40000000u, 0x20000000u,
    0x10000000u, 0x08000000u, 0x04000000u, 0x02000000u,
    0x01000000u, 0x00800000u, 0x00400000u, 0x00200000u,
    0x00100000u, 0x00080000u, 0x00040000u, 0x00020000u,
    0x00010000u, 0x00008000u, 0x00004000u, 0x00002000u,
    0x00001000u, 0x00000800u, 0x00000400u, 0x00000200u,
    0x00000100u, 0x00000080u, 0x00000040u, 0x00000020u
};

// Compute bit-reversal of n-bit integer (scalar version).
static inline uint32_t bit_rev(uint32_t val, uint32_t nbits) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < nbits; i++) {
        result |= ((val >> i) & 1) << (nbits - 1 - i);
    }
    return result;
}

// Vectorized bit-reversal is defined in ntt_kernel.cpp (needs ESIMD types)

} // namespace ntt
