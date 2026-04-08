#pragma once

#include "bn254_poseidon.hpp"

// Poseidon-254 hash kernels for Intel ESIMD (SIMD16 lane-per-hash).
// hash_fold:  digest pairs → digests (Merkle tree levels)
// hash_rows:  BabyBear matrix → digests (Merkle tree leaves)
//
// Digest = 8 × u32, canonical Fr little-endian representation.
// Matrix = column-major: matrix[col * row_size + row].

namespace bn254 {

static constexpr uint32_t BB31_P = 0x78000001u;

// ============================================================================
// bb31_absorb_8: Pack 8 BabyBear u32 values into one Fr (Montgomery form).
// Computes val[0] + val[1]*p + val[2]*p^2 + ... + val[7]*p^7 (mod r).
// Evaluated via Horner in normal form, then one to_mont at the end.
// ============================================================================
ESIMD_INLINE Fp bb31_absorb_8(Vec16 vals[8]) {
    Fp acc = Fp::zero();
    acc.v[0] = vals[7];

    #pragma unroll
    for (int k = 6; k >= 0; --k) {
        // acc *= p (single-limb multiply across 8 limbs)
        Vec16 carry = Vec16(0u);
        #pragma unroll
        for (int j = 0; j < N_LIMBS; ++j) {
            Vec16 lo;
            Vec16 hi = esimd_exp::imul(lo, acc.v[j], Vec16(BB31_P));
            Vec16 c;
            lo = esimd::addc(c, lo, carry);
            carry = hi + c;
            acc.v[j] = lo;
        }
        // acc += vals[k]
        Vec16 c;
        acc.v[0] = esimd::addc(c, acc.v[0], vals[k]);
        #pragma unroll
        for (int j = 1; j < N_LIMBS; ++j) {
            Vec16 c2;
            acc.v[j] = esimd::addc(c2, acc.v[j], c);
            c = c2;
        }
    }
    return to_mont(acc);
}

// ============================================================================
// hash_fold kernel: output[i] = hash_pair(input[2*i], input[2*i+1])
//
// Digests stored AoS: digest[i] at ptr[i*8 .. i*8+7].
// hash_pair(a,b) = fr_to_digest(poseidon_mix([0, digest_to_fr(a), digest_to_fr(b)])[0])
// ============================================================================
static void launch_hash_fold(sycl::queue& q,
                             uint32_t* d_out,
                             const uint32_t* d_in,
                             uint32_t count) {
    if (count == 0) return;
    uint32_t num_threads = (count + 15) / 16;

    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;
            if (base >= count) return;

            // Lane offsets [0..15]
            esimd::simd<uint32_t, 16> lanes;
            #pragma unroll
            for (int l = 0; l < 16; ++l) lanes[l] = l;

            // Byte offsets for left/right digests.
            // Left: d_in[((base+lane)*2) * 8 + j]
            // Right: d_in[((base+lane)*2 + 1) * 8 + j]
            esimd::simd<uint32_t, 16> left_base  = (2u * (base + lanes)) * 8u;
            esimd::simd<uint32_t, 16> right_base = left_base + 8u;

            Fp a_raw, b_raw;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                a_raw.v[j] = esimd::gather<uint32_t, 16>(d_in, (left_base + j) * 4u);
                b_raw.v[j] = esimd::gather<uint32_t, 16>(d_in, (right_base + j) * 4u);
            }

            // digest_to_fr = to_mont (canonical bytes → Montgomery)
            Fp a_mont = to_mont(a_raw);
            Fp b_mont = to_mont(b_raw);

            // Poseidon-254 permutation on [0, a, b]
            Fp cells[3];
            cells[0] = Fp::zero();
            cells[1] = a_mont;
            cells[2] = b_mont;
            poseidon_mix(cells);

            // fr_to_digest = from_mont (Montgomery → canonical)
            Fp result = from_mont(cells[0]);

            // Scatter output digest
            esimd::simd<uint32_t, 16> out_base = (base + lanes) * 8u;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                esimd::scatter<uint32_t, 16>(d_out, (out_base + j) * 4u, result.v[j]);
            }
        }).wait();
}

// ============================================================================
// hash_rows kernel: digest[row] = hash_elem_slice(matrix_row[0..col_size])
//
// Matrix is column-major: matrix[col * row_size + row].
// BabyBear elements packed 8-at-a-time into Fr rate cells via bb31_absorb_8.
// Rate = 2 (cells[1] and cells[2]). Permute every 16 BabyBear elements.
// ============================================================================
static void launch_hash_rows(sycl::queue& q,
                             uint32_t* d_out,
                             const uint32_t* d_matrix,
                             uint32_t row_size,
                             uint32_t col_size) {
    if (row_size == 0) return;
    uint32_t num_threads = (row_size + 15) / 16;

    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;
            if (base >= row_size) return;

            Fp cells[3];
            cells[0] = Fp::zero();
            cells[1] = Fp::zero();
            cells[2] = Fp::zero();

            uint32_t cell_idx = 1;  // CELLS_OUT = 1

            // Process columns in groups of 8 (one Fr absorption per group)
            for (uint32_t col = 0; col < col_size; col += 8) {
                uint32_t n = col_size - col;
                if (n > 8) n = 8;

                // Load up to 8 column values per lane via block_load
                Vec16 vals[8];
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    if (col + k < col_size) {
                        vals[k] = esimd::block_load<uint32_t, 16>(
                            d_matrix + (uint64_t)(col + k) * row_size + base);
                    } else {
                        vals[k] = Vec16(0u);
                    }
                }

                // Pack 8 BabyBear values into one Fr element
                cells[cell_idx] = bb31_absorb_8(vals);
                cell_idx++;

                if (cell_idx == P254_CELLS) {
                    // Both rate cells filled — permute and reset
                    poseidon_mix(cells);
                    cell_idx = 1;
                    cells[1] = Fp::zero();
                    cells[2] = Fp::zero();
                }
            }

            // Final permute if any partial data in rate cells
            if (cell_idx != 1) {
                poseidon_mix(cells);
            }

            // fr_to_digest: convert cells[0] from Montgomery to canonical
            Fp result = from_mont(cells[0]);

            // Store output digest (AoS layout)
            esimd::simd<uint32_t, 16> lanes;
            #pragma unroll
            for (int l = 0; l < 16; ++l) lanes[l] = l;
            esimd::simd<uint32_t, 16> out_base = (base + lanes) * 8u;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                esimd::scatter<uint32_t, 16>(d_out, (out_base + j) * 4u, result.v[j]);
            }
        }).wait();
}

} // namespace bn254
