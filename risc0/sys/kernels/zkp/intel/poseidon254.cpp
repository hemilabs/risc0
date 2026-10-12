// Poseidon-254 ESIMD kernel entry points with device-memory constants.
//
// Uses the optimized arithmetic of bn254_poseidon_opt.hpp: lazy Montgomery
// range [0, 2r), product-scanning multiplies with fused dot products and
// multiply-adds, and the sparse partial rounds (constants in
// bn254_poseidon_sparse_constants.hpp, same order as the CUDA kernels). Only
// cell 0 of the state is computed in the last full round; both kernels read
// only cell 0 after a permutation (rows re-absorbs into cells 1-2 or ends,
// fold outputs cell 0). Output is bit-identical to the CPU Poseidon254 hash
// suite (poseidon254_hash_ops_match_cpu); ~3.2x (fold) and ~3.8-4.8x (rows)
// faster than the previous plain-MDS kernels.

#include "bn254_poseidon_kernels.hpp"
#include "bn254_poseidon_opt.hpp"
#include "bn254_poseidon_sparse_constants.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>

#include <cstring>
#include <vector>

namespace syclex_p254 = sycl::ext::intel::experimental;
namespace syclex_oneapi_p254 = sycl::ext::oneapi::experimental;

// ============================================================================
// Device constants management (uploaded once, cached globally)
// ============================================================================

// Sparse layout: RC[68] | MDS[9] | PRE_SPARSE[9] | SPARSE[42][4] | M00[1], each 8 words.
static uint32_t* g_d_p254_sparse = nullptr;

static void ensure_poseidon254_device_constants(sycl::queue& q) {
    // Thread-safe one-time init (C++11 magic static).
    static const bool once = [&] {
        std::vector<uint32_t> h(255 * 8);
        std::memcpy(h.data(), &P254_SPARSE_RC[0][0], 68 * 32);
        std::memcpy(h.data() + p254o::SP_MDS * 8, &P254_SPARSE_MDS[0][0], 9 * 32);
        std::memcpy(h.data() + p254o::SP_PRE * 8, &P254_PRE_SPARSE[0][0], 9 * 32);
        std::memcpy(h.data() + p254o::SP_SPARSE * 8, &P254_SPARSE_MATRICES[0][0][0], 42 * 4 * 32);
        std::memcpy(h.data() + p254o::SP_M00 * 8, &P254_M_0_0[0], 32);
        // 64-byte aligned: the constants are read with 8-dword block loads.
        g_d_p254_sparse = static_cast<uint32_t*>(sycl::aligned_alloc_device(64, h.size() * 4, q));
        q.memcpy(g_d_p254_sparse, h.data(), h.size() * 4).wait();
        return true;
    }();
    (void)once;
}

// ============================================================================
// Kernel launchers
// ============================================================================

static void poseidon254_fold_impl(sycl::queue& q, uint32_t* d_out,
                                   const uint32_t* d_in, uint32_t count) {
    if (count == 0) return;
    ensure_poseidon254_device_constants(q);
    uint32_t num_threads = (count + 15) / 16;
    const uint32_t* pc = g_d_p254_sparse;

    q.parallel_for(sycl::range<1>(num_threads),
        syclex_oneapi_p254::properties{syclex_p254::grf_size<128>},
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            using namespace p254o;
            uint32_t base = idx[0] * 16;
            if (base >= count) return;

            esimd::simd<uint32_t, 16> lanes;
            #pragma unroll
            for (int l = 0; l < 16; ++l) lanes[l] = l;

            // count need not be a multiple of 16 (tail layers of a small tree).
            esimd::simd_mask<16> live = (base + lanes) < count;
            esimd::simd<uint32_t, 16> left_base  = (2u * (base + lanes)) * 8u;
            esimd::simd<uint32_t, 16> right_base = left_base + 8u;

            Fp a, b;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                a.v[j] = esimd::gather<uint32_t, 16>(d_in, (left_base + j) * 4u, live);
                b.v[j] = esimd::gather<uint32_t, 16>(d_in, (right_base + j) * 4u, live);
            }

            Fp s[3];
            #pragma unroll
            for (int j = 0; j < 8; ++j) s[0].v[j] = 0;
            s[1] = to_mont(a);
            s[2] = to_mont(b);
            mix_sparse(s, pc);
            Fp r = from_mont(s[0]);

            esimd::simd<uint32_t, 16> out_base = (base + lanes) * 8u;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                esimd::scatter<uint32_t, 16>(d_out, (out_base + j) * 4u, r.v[j], live);
            }
        }).wait();
}

static void poseidon254_rows_impl(sycl::queue& q, uint32_t* d_out,
                                   const uint32_t* d_matrix,
                                   uint32_t row_size, uint32_t col_size) {
    if (row_size == 0) return;
    ensure_poseidon254_device_constants(q);
    uint32_t num_threads = (row_size + 15) / 16;
    const uint32_t* pc = g_d_p254_sparse;

    q.parallel_for(sycl::range<1>(num_threads),
        syclex_oneapi_p254::properties{syclex_p254::grf_size<128>},
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;
            if (base >= row_size) return;

            esimd::simd<uint32_t, 16> lanes;
            #pragma unroll
            for (int l = 0; l < 16; ++l) lanes[l] = l;
            // row_size need not be a multiple of 16: mask the tail group.
            const bool full = base + 16 <= row_size;
            esimd::simd_mask<16> live = (base + lanes) < row_size;

            p254o::Fp s[3];
            #pragma unroll
            for (int i = 0; i < 3; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) s[i].v[j] = 0;

            uint32_t cell_idx = 1;
            for (uint32_t col = 0; col < col_size; col += 8) {
                bn254::Vec16 vals[8];
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    if (col + k < col_size && full) {
                        vals[k] = esimd::block_load<uint32_t, 16>(
                            d_matrix + (uint64_t)(col + k) * row_size + base);
                    } else if (col + k < col_size) {
                        vals[k] = esimd::gather<uint32_t, 16>(
                            d_matrix + (uint64_t)(col + k) * row_size + base,
                            lanes * 4u, live);
                    } else {
                        vals[k] = bn254::Vec16(0u);
                    }
                }
                // 8 BabyBear elements -> one Fr cell (Montgomery form).
                bn254::Fp a = bn254::bb31_absorb_8(vals);
                p254o::Fp& dst = cell_idx == 1 ? s[1] : s[2];
                #pragma unroll
                for (int j = 0; j < 8; ++j) dst.v[j] = a.v[j];
                cell_idx++;
                if (cell_idx == 3) {
                    p254o::mix_sparse(s, pc);
                    cell_idx = 1;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) { s[1].v[j] = 0; s[2].v[j] = 0; }
                }
            }
            if (cell_idx != 1) p254o::mix_sparse(s, pc);
            p254o::Fp r = p254o::from_mont(s[0]);

            esimd::simd<uint32_t, 16> out_base = (base + lanes) * 8u;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                esimd::scatter<uint32_t, 16>(d_out, (out_base + j) * 4u, r.v[j], live);
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
