// Poseidon-254 ESIMD kernel entry points with device-memory constants.
// Uses plain-MDS variant (NOT sparse). Sparse was benchmarked and found slower
// on Intel GPU due to L1 cache thrashing from 168 unique sparse matrix entries
// vs 9 reused MDS entries that stay warm in L1.

#include "bn254_poseidon_kernels.hpp"

// ============================================================================
// Device constants management (uploaded once, cached globally)
// ============================================================================

static uint32_t* g_d_p254_rc = nullptr;   // 150 x 8 = 1200 uint32_t
static uint32_t* g_d_p254_mds = nullptr;  // 9 x 8 = 72 uint32_t

static void ensure_poseidon254_device_constants(sycl::queue& q) {
    if (g_d_p254_rc) return;

    g_d_p254_rc = sycl::malloc_device<uint32_t>(150 * 8, q);
    g_d_p254_mds = sycl::malloc_device<uint32_t>(9 * 8, q);

    q.memcpy(g_d_p254_rc, &POSEIDON254_RC[0][0], 150 * 8 * sizeof(uint32_t));
    q.memcpy(g_d_p254_mds, &POSEIDON254_MDS[0][0], 9 * 8 * sizeof(uint32_t));
    q.wait();
}

// ============================================================================
// Kernel launchers using device-memory constants (plain MDS)
// ============================================================================

static void poseidon254_fold_impl(sycl::queue& q, uint32_t* d_out,
                                   const uint32_t* d_in, uint32_t count) {
    if (count == 0) return;
    ensure_poseidon254_device_constants(q);
    uint32_t num_threads = (count + 15) / 16;

    const uint32_t* prc = g_d_p254_rc;
    const uint32_t* pmds = g_d_p254_mds;

    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;
            if (base >= count) return;

            esimd::simd<uint32_t, 16> lanes;
            #pragma unroll
            for (int l = 0; l < 16; ++l) lanes[l] = l;

            esimd::simd<uint32_t, 16> left_base  = (2u * (base + lanes)) * 8u;
            esimd::simd<uint32_t, 16> right_base = left_base + 8u;

            bn254::Fp a_raw, b_raw;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                a_raw.v[j] = esimd::gather<uint32_t, 16>(d_in, (left_base + j) * 4u);
                b_raw.v[j] = esimd::gather<uint32_t, 16>(d_in, (right_base + j) * 4u);
            }

            bn254::Fp a_mont = bn254::to_mont(a_raw);
            bn254::Fp b_mont = bn254::to_mont(b_raw);

            bn254::Fp cells[3];
            cells[0] = bn254::Fp::zero();
            cells[1] = a_mont;
            cells[2] = b_mont;
            bn254::poseidon_mix_dev(cells, prc, pmds);

            bn254::Fp result = bn254::from_mont(cells[0]);

            esimd::simd<uint32_t, 16> out_base = (base + lanes) * 8u;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                esimd::scatter<uint32_t, 16>(d_out, (out_base + j) * 4u, result.v[j]);
            }
        }).wait();
}

static void poseidon254_rows_impl(sycl::queue& q, uint32_t* d_out,
                                   const uint32_t* d_matrix,
                                   uint32_t row_size, uint32_t col_size) {
    if (row_size == 0) return;
    ensure_poseidon254_device_constants(q);
    uint32_t num_threads = (row_size + 15) / 16;

    const uint32_t* prc = g_d_p254_rc;
    const uint32_t* pmds = g_d_p254_mds;

    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;
            if (base >= row_size) return;

            bn254::Fp cells[3];
            cells[0] = bn254::Fp::zero();
            cells[1] = bn254::Fp::zero();
            cells[2] = bn254::Fp::zero();

            uint32_t cell_idx = 1;

            for (uint32_t col = 0; col < col_size; col += 8) {
                bn254::Vec16 vals[8];
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    if (col + k < col_size) {
                        vals[k] = esimd::block_load<uint32_t, 16>(
                            d_matrix + (uint64_t)(col + k) * row_size + base);
                    } else {
                        vals[k] = bn254::Vec16(0u);
                    }
                }

                cells[cell_idx] = bn254::bb31_absorb_8(vals);
                cell_idx++;

                if (cell_idx == bn254::P254_CELLS) {
                    bn254::poseidon_mix_dev(cells, prc, pmds);
                    cell_idx = 1;
                    cells[1] = bn254::Fp::zero();
                    cells[2] = bn254::Fp::zero();
                }
            }

            if (cell_idx != 1) {
                bn254::poseidon_mix_dev(cells, prc, pmds);
            }

            bn254::Fp result = bn254::from_mont(cells[0]);

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

// ============================================================================
// extern "C" FFI entry points
// ============================================================================

extern "C" {

void esimd_poseidon254_fold(sycl::queue& q, uint32_t* d_out,
                            const uint32_t* d_in, uint32_t count) {
    poseidon254_fold_impl(q, d_out, d_in, count);
}

void esimd_poseidon254_rows(sycl::queue& q, uint32_t* d_out,
                            const uint32_t* d_in, uint32_t row_size, uint32_t col_size) {
    poseidon254_rows_impl(q, d_out, d_in, row_size, col_size);
}

} // extern "C"
