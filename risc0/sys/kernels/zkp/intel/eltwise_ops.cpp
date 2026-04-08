// Element-wise HAL operations for Intel GPU STARK prover.
// ESIMD implementations matching risc0's CUDA HAL kernels.
// All kernels operate on device-resident data (no host-device transfer).

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include "bb31_field.hpp"

namespace esimd = sycl::ext::intel::esimd;

// Cache hints for streaming access (data used once, no L1 reuse)
static constexpr auto ELT_LOAD = esimd::properties{
    esimd::cache_hint_L1<esimd::cache_hint::streaming>,
    esimd::cache_hint_L2<esimd::cache_hint::cached>};
static constexpr auto ELT_STORE = esimd::properties{
    esimd::cache_hint_L1<esimd::cache_hint::streaming>,
    esimd::cache_hint_L2<esimd::cache_hint::write_back>};

// Shared queue creation (same as ntt_kernel.cpp)
static sycl::queue create_eltwise_queue() {
    auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    for (auto& d : gpu_devices) {
        auto name = d.get_info<sycl::info::device::name>();
        if (name.find("Intel") != std::string::npos ||
            name.find("0xe2") != std::string::npos) {
            return sycl::queue(d, sycl::property_list{
                sycl::property::queue::in_order{}});
        }
    }
    throw std::runtime_error("Intel GPU not found");
}

extern "C" {

// ============================================================================
// Element-wise field operations
// ============================================================================

// out[i] = x[i] + y[i] for i in 0..count
// Z=4: 64 elements per thread, 8 loads upfront for maximum MLP
void esimd_eltwise_add_fp(sycl::queue& q, uint32_t* out,
                           const uint32_t* x, const uint32_t* y,
                           uint32_t count) {
    auto* po = out;
    auto* px = x;
    auto* py = y;

    if (count >= 64) {
        uint32_t bulk = count & ~63u; // round down to multiple of 64
        uint32_t num_threads = bulk / 64;
        q.parallel_for(sycl::range<1>(num_threads),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t base = idx[0] * 64;
                // Phase 1: Issue all 8 loads for maximum MLP
                auto a0 = esimd::block_load<uint32_t, 16>(px + base, ELT_LOAD);
                auto a1 = esimd::block_load<uint32_t, 16>(px + base + 16, ELT_LOAD);
                auto a2 = esimd::block_load<uint32_t, 16>(px + base + 32, ELT_LOAD);
                auto a3 = esimd::block_load<uint32_t, 16>(px + base + 48, ELT_LOAD);
                auto b0 = esimd::block_load<uint32_t, 16>(py + base, ELT_LOAD);
                auto b1 = esimd::block_load<uint32_t, 16>(py + base + 16, ELT_LOAD);
                auto b2 = esimd::block_load<uint32_t, 16>(py + base + 32, ELT_LOAD);
                auto b3 = esimd::block_load<uint32_t, 16>(py + base + 48, ELT_LOAD);
                // Phase 2: 4 independent adds
                auto r0 = bb31::field_add(a0, b0);
                auto r1 = bb31::field_add(a1, b1);
                auto r2 = bb31::field_add(a2, b2);
                auto r3 = bb31::field_add(a3, b3);
                // Phase 3: Store results
                esimd::block_store(po + base, r0, ELT_STORE);
                esimd::block_store(po + base + 16, r1, ELT_STORE);
                esimd::block_store(po + base + 32, r2, ELT_STORE);
                esimd::block_store(po + base + 48, r3, ELT_STORE);
            });
        // Handle remaining 0..63 elements
        if (bulk < count) {
            uint32_t rem = count - bulk;
            uint32_t tail_threads = (rem + 15) / 16;
            q.parallel_for(sycl::range<1>(tail_threads),
                [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                    uint32_t base = bulk + idx[0] * 16;
                    if (base + 16 <= count) {
                        auto a = esimd::block_load<uint32_t, 16>(px + base, ELT_LOAD);
                        auto b = esimd::block_load<uint32_t, 16>(py + base, ELT_LOAD);
                        esimd::block_store(po + base, bb31::field_add(a, b), ELT_STORE);
                    } else {
                        for (uint32_t i = base; i < count; i++) {
                            uint32_t av = *(px + i), bv = *(py + i);
                            uint32_t r = av + bv;
                            *(po + i) = (r >= bb31::MOD) ? r - bb31::MOD : r;
                        }
                    }
                });
        }
    } else {
        // Small count: Z=1 path
        uint32_t num_threads = (count + 15) / 16;
        q.parallel_for(sycl::range<1>(num_threads),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t base = idx[0] * 16;
                if (base + 16 <= count) {
                    auto a = esimd::block_load<uint32_t, 16>(px + base, ELT_LOAD);
                    auto b = esimd::block_load<uint32_t, 16>(py + base, ELT_LOAD);
                    esimd::block_store(po + base, bb31::field_add(a, b), ELT_STORE);
                } else {
                    for (uint32_t i = base; i < count; i++) {
                        uint32_t av = *(px + i), bv = *(py + i);
                        uint32_t r = av + bv;
                        *(po + i) = (r >= bb31::MOD) ? r - bb31::MOD : r;
                    }
                }
            });
    }
}

// out[i] = in[i] for i in 0..count
// Uses streaming hints since data is not reused.
void esimd_eltwise_copy_fp(sycl::queue& q, uint32_t* out,
                            const uint32_t* in, uint32_t count) {
    auto* po = out;
    auto* pi = in;
    uint32_t num_threads = (count + 15) / 16;

    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;
            if (base + 16 <= count) {
                auto v = esimd::block_load<uint32_t, 16>(pi + base, ELT_LOAD);
                esimd::block_store(po + base, v, ELT_STORE);
            } else {
                for (uint32_t i = base; i < count; i++)
                    *(po + i) = *(pi + i);
            }
        });
}

// Copy a sub-region: for each row in 0..fromRows, copy fromCols elements
// from[fromOffset + row*fromStride + col] -> into[intoOffset + row*intoStride + col]
void esimd_eltwise_copy_fp_region(sycl::queue& q,
                                   uint32_t* into, const uint32_t* from,
                                   uint32_t fromRows, uint32_t fromCols,
                                   uint32_t fromOffset, uint32_t fromStride,
                                   uint32_t intoOffset, uint32_t intoStride) {
    auto* pi = into;
    auto* pf = from;
    // One thread per row (each row copies fromCols elements)
    q.parallel_for(sycl::range<1>(fromRows),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t row = idx[0];
            uint32_t src_base = fromOffset + row * fromStride;
            uint32_t dst_base = intoOffset + row * intoStride;
            uint32_t col = 0;
            // SIMD16 bulk copy
            for (; col + 16 <= fromCols; col += 16) {
                auto v = esimd::block_load<uint32_t, 16>(pf + src_base + col);
                esimd::block_store(pi + dst_base + col, v);
            }
            // Scalar tail
            for (; col < fromCols; col++)
                *(pi + dst_base + col) = *(pf + src_base + col);
        });
}

// Sum FpExt elements: for each i in 0..count, sum to_add FpExt values
// and scatter the 4 components to separate output arrays.
// Input layout: FpExt* in → in[4*(count*k + i) + c] = component c of k-th FpExt for index i
// Output layout: Fp* out → out[i + c*count] = component c of result for i
// SIMD16: each thread processes 16 consecutive output indices
void esimd_eltwise_sum_fpext(sycl::queue& q,
                              uint32_t* out, const uint32_t* in,
                              uint32_t to_add, uint32_t count) {
    auto* po = out;
    auto* pi = in;
    uint32_t num_threads = (count + 15) / 16;

    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;
            if (base + 16 <= count) {
                // Accumulate 4 component sums across to_add FpExt values
                bb31::Vec16 s0(0u), s1(0u), s2(0u), s3(0u);
                for (uint32_t k = 0; k < to_add; k++) {
                    // Input: 16 consecutive FpExt structs at offset 4*(count*k + base)
                    // Each FpExt is 4 u32, stride-4 between elements
                    // Gather component c of 16 consecutive FpExts:
                    // offsets = 4*(count*k + base + lane)*4 + c*4 bytes
                    esimd::simd<uint32_t, 16> lane(0u, 1u);
                    auto fpext_base = (count * k + base + lane) * 4u; // u32 index of each FpExt
                    auto byte_base = fpext_base * (uint32_t)sizeof(uint32_t);
                    // Gather each component (stride-4 pattern)
                    auto e0 = esimd::gather<uint32_t, 16>(pi, byte_base);
                    auto e1 = esimd::gather<uint32_t, 16>(pi, byte_base + 4u);
                    auto e2 = esimd::gather<uint32_t, 16>(pi, byte_base + 8u);
                    auto e3 = esimd::gather<uint32_t, 16>(pi, byte_base + 12u);
                    s0 = bb31::field_add(s0, e0);
                    s1 = bb31::field_add(s1, e1);
                    s2 = bb31::field_add(s2, e2);
                    s3 = bb31::field_add(s3, e3);
                }
                // Store to 4 separate output arrays (contiguous within each)
                esimd::block_store(po + base + 0 * count, s0, ELT_STORE);
                esimd::block_store(po + base + 1 * count, s1, ELT_STORE);
                esimd::block_store(po + base + 2 * count, s2, ELT_STORE);
                esimd::block_store(po + base + 3 * count, s3, ELT_STORE);
            } else {
                // Scalar tail
                for (uint32_t i = base; i < count; i++) {
                    uint32_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                    for (uint32_t k = 0; k < to_add; k++) {
                        uint32_t off = 4 * (count * k + i);
                        sum0 += *(pi+off+0); if (sum0>=bb31::MOD) sum0-=bb31::MOD;
                        sum1 += *(pi+off+1); if (sum1>=bb31::MOD) sum1-=bb31::MOD;
                        sum2 += *(pi+off+2); if (sum2>=bb31::MOD) sum2-=bb31::MOD;
                        sum3 += *(pi+off+3); if (sum3>=bb31::MOD) sum3-=bb31::MOD;
                    }
                    *(po+i+0*count)=sum0; *(po+i+1*count)=sum1;
                    *(po+i+2*count)=sum2; *(po+i+3*count)=sum3;
                }
            }
        });
}

// Replace INVALID (0xffffffff) sentinel values with 0
void esimd_eltwise_zeroize_fp(sycl::queue& q, uint32_t* elems, uint32_t count) {
    auto* pe = elems;
    uint32_t num_threads = (count + 15) / 16;

    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;
            if (base + 16 <= count) {
                auto v = esimd::block_load<uint32_t, 16>(pe + base, ELT_LOAD);
                // Replace 0xffffffff with 0
                esimd::simd<uint32_t, 16> invalid(0xffffffffu);
                esimd::simd_mask<16> mask = (v == invalid);
                v.merge(esimd::simd<uint32_t, 16>(0u), mask);
                esimd::block_store(pe + base, v, ELT_STORE);
            } else {
                for (uint32_t i = base; i < count; i++) {
                    if (*(pe + i) == 0xffffffffu)
                        *(pe + i) = 0;
                }
            }
        });
}

// Replace INVALID sentinel in FpExt array (4 components per element)
// Treats FpExt as flat u32 array of 4*count elements
void esimd_eltwise_zeroize_fpext(sycl::queue& q, uint32_t* elems, uint32_t count) {
    // Each FpExt is 4 u32. Zeroize on the flat array of 4*count u32s.
    esimd_eltwise_zeroize_fp(q, elems, count * 4);
}

// ============================================================================
// Gather / Scatter
// ============================================================================

// dst[gid] = src[gid * stride + idx] for gid in 0..size
// SIMD16 optimized: block_load fast path for stride=1, gather for stride>1
void esimd_gather_sample(sycl::queue& q,
                          uint32_t* dst, const uint32_t* src,
                          uint32_t idx, uint32_t size, uint32_t stride) {
    auto* pd = dst;
    auto* ps = src;
    uint32_t num_threads = (size + 15) / 16;

    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> tid) [[intel::sycl_explicit_simd]] {
            uint32_t base = tid[0] * 16;
            if (base + 16 <= size) {
                if (stride == 1) {
                    // Fast path: contiguous read at src[base + idx]
                    auto data = esimd::block_load<uint32_t, 16>(ps + base + idx, ELT_LOAD);
                    esimd::block_store(pd + base, data, ELT_STORE);
                } else {
                    // Strided path: gather with computed offsets
                    esimd::simd<uint32_t, 16> lane_idx(0u, 1u);
                    auto src_idx = (lane_idx + base) * stride + idx;
                    auto byte_off = src_idx * (uint32_t)sizeof(uint32_t);
                    auto data = esimd::gather<uint32_t, 16>(ps, byte_off);
                    esimd::block_store(pd + base, data, ELT_STORE);
                }
            } else {
                for (uint32_t i = base; i < size; i++)
                    *(pd + i) = *(ps + i * stride + idx);
            }
        });
}

// Indexed scatter: for each gid in 0..count,
// for idx in index[gid]..index[gid+1]: into[offsets[idx]] = values[idx]
void esimd_scatter(sycl::queue& q,
                    uint32_t* into, const uint32_t* index,
                    const uint32_t* offsets, const uint32_t* values,
                    uint32_t count) {
    auto* pi = into;
    auto* pidx = index;
    auto* poff = offsets;
    auto* pval = values;

    q.parallel_for(sycl::range<1>(count),
        [=](sycl::id<1> gid) [[intel::sycl_explicit_simd]] {
            uint32_t g = gid[0];
            uint32_t start = *(pidx + g);
            uint32_t end = *(pidx + g + 1);
            for (uint32_t i = start; i < end; i++) {
                *(pi + *(poff + i)) = *(pval + i);
            }
        });
}

// ============================================================================
// Batch bit-reverse permutation
// ============================================================================

// For count elements viewed as batches of (1 << nBits) elements,
// swap element [batch*rowSize + idx] with [batch*rowSize + bit_rev(idx)]
void esimd_batch_bit_reverse(sycl::queue& q, uint32_t* io,
                              uint32_t nBits, uint32_t count) {
    auto* pd = io;
    q.parallel_for(sycl::range<1>(count),
        [=](sycl::id<1> gid) [[intel::sycl_explicit_simd]] {
            uint32_t totIdx = gid[0];
            uint32_t rowSize = 1u << nBits;
            uint32_t idx = totIdx & (rowSize - 1);
            uint32_t s = totIdx >> nBits;

            // Bit-reverse idx within nBits bits
            uint32_t ridx = 0;
            uint32_t tmp = idx;
            for (uint32_t b = 0; b < nBits; b++) {
                ridx = (ridx << 1) | (tmp & 1);
                tmp >>= 1;
            }

            if (idx < ridx) {
                uint32_t idx1 = s * rowSize + idx;
                uint32_t idx2 = s * rowSize + ridx;
                uint32_t tmp_val = *(pd + idx1);
                *(pd + idx1) = *(pd + idx2);
                *(pd + idx2) = tmp_val;
            }
        });
}

// ============================================================================
// Validation helpers
// ============================================================================

int32_t validate_eltwise_ops() {
    try {
    auto q = create_eltwise_queue();
    constexpr uint32_t N = 1024;
    constexpr uint32_t BIG = 4096; // for larger tests

    auto* a = sycl::malloc_device<uint32_t>(BIG, q);
    auto* b = sycl::malloc_device<uint32_t>(BIG, q);
    auto* c = sycl::malloc_device<uint32_t>(BIG, q);
    auto* h = sycl::malloc_host<uint32_t>(BIG, q);
    auto* h2 = sycl::malloc_host<uint32_t>(BIG, q);
    auto* h3 = sycl::malloc_host<uint32_t>(BIG, q);
    int errs = 0;

    // ========== eltwise_add_fp ==========
    // Test 1: basic addition
    for (uint32_t i = 0; i < N; i++) { h[i] = (i+1) % bb31::MOD; h2[i] = (i+2) % bb31::MOD; }
    q.memcpy(a, h, N*4); q.memcpy(b, h2, N*4); q.wait();
    esimd_eltwise_add_fp(q, c, a, b, N); q.wait();
    q.memcpy(h3, c, N*4); q.wait();
    for (uint32_t i = 0; i < N; i++) {
        uint32_t exp = ((i+1)+(i+2)) % bb31::MOD;
        if (h3[i] != exp) { errs++; break; }
    }
    // Test 2: near-MOD values (exercises reduction branch)
    for (uint32_t i = 0; i < N; i++) { h[i] = bb31::MOD - 1; h2[i] = bb31::MOD - 1; }
    q.memcpy(a, h, N*4); q.memcpy(b, h2, N*4); q.wait();
    esimd_eltwise_add_fp(q, c, a, b, N); q.wait();
    q.memcpy(h3, c, N*4); q.wait();
    for (uint32_t i = 0; i < N; i++) {
        if (h3[i] != bb31::MOD - 2) { errs++; break; } // (P-1)+(P-1) = 2P-2, mod P = P-2
    }
    // Test 3: exact wraparound MOD-1 + 1 = 0
    for (uint32_t i = 0; i < N; i++) { h[i] = bb31::MOD - 1; h2[i] = 1; }
    q.memcpy(a, h, N*4); q.memcpy(b, h2, N*4); q.wait();
    esimd_eltwise_add_fp(q, c, a, b, N); q.wait();
    q.memcpy(h3, c, N*4); q.wait();
    for (uint32_t i = 0; i < N; i++) {
        if (h3[i] != 0) { errs++; break; }
    }
    // Test 4: non-multiple-of-16 count (tail path)
    esimd_eltwise_add_fp(q, c, a, b, 17); q.wait();
    q.memcpy(h3, c, 17*4); q.wait();
    if (h3[16] != 0) errs++; // element 16 is in the scalar tail
    fprintf(stderr, "  eltwise_add_fp: %s\n", errs == 0 ? "PASS" : "FAIL"); fflush(stderr);
    if (errs) return -1;

    // ========== eltwise_copy_fp ==========
    for (uint32_t i = 0; i < N; i++) h[i] = i * 7 + 3;
    q.memcpy(a, h, N*4); q.wait();
    esimd_eltwise_copy_fp(q, c, a, N); q.wait();
    q.memcpy(h2, c, N*4); q.wait();
    for (uint32_t i = 0; i < N; i++) { if (h2[i] != h[i]) { errs++; break; } }
    // Also test count=17 (tail)
    esimd_eltwise_copy_fp(q, c, a, 17); q.wait();
    q.memcpy(h2, c, 17*4); q.wait();
    for (uint32_t i = 0; i < 17; i++) { if (h2[i] != h[i]) { errs++; break; } }
    fprintf(stderr, "  eltwise_copy_fp: %s\n", errs == 0 ? "PASS" : "FAIL"); fflush(stderr);
    if (errs) return -2;

    // ========== eltwise_copy_fp_region ==========
    // Set up a 4x8 source matrix at offset 10 with stride 12
    // Copy to offset 5 with stride 10
    constexpr uint32_t ROWS = 4, COLS = 8, SRC_STRIDE = 12, DST_STRIDE = 10;
    constexpr uint32_t SRC_OFF = 10, DST_OFF = 5;
    for (uint32_t i = 0; i < BIG; i++) h[i] = i + 100;
    for (uint32_t i = 0; i < BIG; i++) h2[i] = 0;
    q.memcpy(a, h, BIG*4); q.memcpy(b, h2, BIG*4); q.wait();
    esimd_eltwise_copy_fp_region(q, b, a, ROWS, COLS, SRC_OFF, SRC_STRIDE, DST_OFF, DST_STRIDE);
    q.wait();
    q.memcpy(h2, b, BIG*4); q.wait();
    for (uint32_t r = 0; r < ROWS; r++) {
        for (uint32_t c2 = 0; c2 < COLS; c2++) {
            uint32_t src_val = h[SRC_OFF + r*SRC_STRIDE + c2];
            uint32_t dst_val = h2[DST_OFF + r*DST_STRIDE + c2];
            if (src_val != dst_val) { errs++; break; }
        }
        if (errs) break;
    }
    // Verify untouched areas remain 0
    if (h2[0] != 0 || h2[DST_OFF - 1] != 0) errs++;
    fprintf(stderr, "  eltwise_copy_fp_region: %s\n", errs == 0 ? "PASS" : "FAIL"); fflush(stderr);
    if (errs) return -3;

    // ========== eltwise_sum_fpext ==========
    // Sum 3 FpExt values for 4 output elements
    // Input: 3 groups of 4 FpExt (each FpExt = 4 u32)
    constexpr uint32_t SUM_COUNT = 4, SUM_TO_ADD = 3;
    // in[4*(count*k + i) + c] = component c of k-th FpExt for index i
    uint32_t sum_in[SUM_TO_ADD * SUM_COUNT * 4];
    for (uint32_t k = 0; k < SUM_TO_ADD; k++)
        for (uint32_t i = 0; i < SUM_COUNT; i++)
            for (uint32_t c2 = 0; c2 < 4; c2++)
                sum_in[4*(SUM_COUNT*k + i) + c2] = (k * 100 + i * 10 + c2 + 1) % bb31::MOD;

    q.memcpy(a, sum_in, sizeof(sum_in)); q.wait();
    esimd_eltwise_sum_fpext(q, b, a, SUM_TO_ADD, SUM_COUNT); q.wait();
    q.memcpy(h, b, SUM_COUNT * 4 * 4); q.wait();

    for (uint32_t i = 0; i < SUM_COUNT; i++) {
        for (uint32_t c2 = 0; c2 < 4; c2++) {
            uint32_t expected = 0;
            for (uint32_t k = 0; k < SUM_TO_ADD; k++) {
                expected += sum_in[4*(SUM_COUNT*k + i) + c2];
                if (expected >= bb31::MOD) expected -= bb31::MOD;
            }
            uint32_t got = h[i + c2 * SUM_COUNT];
            if (got != expected) {
                fprintf(stderr, "  sum_fpext fail: i=%u c=%u got=%u exp=%u\n", i, c2, got, expected);
                errs++;
            }
        }
    }
    fprintf(stderr, "  eltwise_sum_fpext: %s\n", errs == 0 ? "PASS" : "FAIL"); fflush(stderr);
    if (errs) return -4;

    // ========== eltwise_zeroize_fp ==========
    for (uint32_t i = 0; i < N; i++) h[i] = (i % 3 == 0) ? 0xffffffffu : (i * 7 + 1);
    q.memcpy(a, h, N*4); q.wait();
    esimd_eltwise_zeroize_fp(q, a, N); q.wait();
    q.memcpy(h2, a, N*4); q.wait();
    for (uint32_t i = 0; i < N; i++) {
        uint32_t expected = (i % 3 == 0) ? 0 : (i * 7 + 1);
        if (h2[i] != expected) { errs++; break; }
    }
    // Test with tail (count=19)
    for (uint32_t i = 0; i < 19; i++) h[i] = (i == 17) ? 0xffffffffu : 42;
    q.memcpy(a, h, 19*4); q.wait();
    esimd_eltwise_zeroize_fp(q, a, 19); q.wait();
    q.memcpy(h2, a, 19*4); q.wait();
    if (h2[17] != 0 || h2[0] != 42 || h2[18] != 42) errs++;
    fprintf(stderr, "  eltwise_zeroize_fp: %s\n", errs == 0 ? "PASS" : "FAIL"); fflush(stderr);
    if (errs) return -5;

    // ========== gather_sample ==========
    // Test stride=3, idx=1
    for (uint32_t i = 0; i < N; i++) h[i] = i * 10;
    q.memcpy(a, h, N*4); q.wait();
    uint32_t gsize = N / 3;
    esimd_gather_sample(q, b, a, 1, gsize, 3); q.wait();
    q.memcpy(h2, b, gsize*4); q.wait();
    for (uint32_t i = 0; i < gsize; i++) {
        if (h2[i] != (i*3+1)*10) { errs++; break; }
    }
    // Test stride=1, idx=0 (contiguous copy)
    esimd_gather_sample(q, b, a, 0, 100, 1); q.wait();
    q.memcpy(h2, b, 100*4); q.wait();
    for (uint32_t i = 0; i < 100; i++) {
        if (h2[i] != i*10) { errs++; break; }
    }
    // Test stride=1, idx=5 (offset contiguous)
    esimd_gather_sample(q, b, a, 5, 50, 1); q.wait();
    q.memcpy(h2, b, 50*4); q.wait();
    for (uint32_t i = 0; i < 50; i++) {
        if (h2[i] != (i+5)*10) { errs++; break; }
    }
    fprintf(stderr, "  gather_sample: %s\n", errs == 0 ? "PASS" : "FAIL"); fflush(stderr);
    if (errs) return -6;

    // ========== scatter ==========
    // CSR-style scatter: 3 groups writing 2, 1, 3 values respectively
    uint32_t sc_index[4] = {0, 2, 3, 6};  // group 0: [0,2), group 1: [2,3), group 2: [3,6)
    uint32_t sc_offsets[6] = {5, 10, 3, 0, 7, 15}; // target positions
    uint32_t sc_values[6] = {100, 200, 300, 400, 500, 600};
    for (uint32_t i = 0; i < 20; i++) h[i] = 0;
    q.memcpy(a, h, 20*4); q.wait(); // into (zeros)
    auto* d_idx = sycl::malloc_device<uint32_t>(4, q);
    auto* d_off = sycl::malloc_device<uint32_t>(6, q);
    auto* d_val = sycl::malloc_device<uint32_t>(6, q);
    q.memcpy(d_idx, sc_index, 4*4);
    q.memcpy(d_off, sc_offsets, 6*4);
    q.memcpy(d_val, sc_values, 6*4); q.wait();
    esimd_scatter(q, a, d_idx, d_off, d_val, 3); q.wait();
    q.memcpy(h, a, 20*4); q.wait();
    if (h[5] != 100 || h[10] != 200 || h[3] != 300 ||
        h[0] != 400 || h[7] != 500 || h[15] != 600) errs++;
    // Check untouched positions remain 0
    if (h[1] != 0 || h[2] != 0 || h[4] != 0 || h[6] != 0) errs++;
    sycl::free(d_idx, q); sycl::free(d_off, q); sycl::free(d_val, q);
    fprintf(stderr, "  scatter: %s\n", errs == 0 ? "PASS" : "FAIL"); fflush(stderr);
    if (errs) return -7;

    // ========== batch_bit_reverse ==========
    // Test 1: nBits=3, single batch
    uint32_t br8[8] = {0,1,2,3,4,5,6,7};
    uint32_t br8_exp[8] = {0,4,2,6,1,5,3,7};
    q.memcpy(a, br8, 8*4); q.wait();
    esimd_batch_bit_reverse(q, a, 3, 8); q.wait();
    q.memcpy(h, a, 8*4); q.wait();
    for (int i = 0; i < 8; i++) { if (h[i] != br8_exp[i]) { errs++; break; } }

    // Test 2: nBits=3, 2 batches (count=16)
    uint32_t br16[16];
    for (int i = 0; i < 16; i++) br16[i] = i + 100;
    q.memcpy(a, br16, 16*4); q.wait();
    esimd_batch_bit_reverse(q, a, 3, 16); q.wait();
    q.memcpy(h, a, 16*4); q.wait();
    // Batch 0: positions 0-7 bit-reversed. Batch 1: positions 8-15 bit-reversed.
    uint32_t br16_exp[16] = {100,104,102,106,101,105,103,107, 108,112,110,114,109,113,111,115};
    for (int i = 0; i < 16; i++) { if (h[i] != br16_exp[i]) { errs++; break; } }

    // Test 3: nBits=1 (trivial: 0↔0, 1↔1, no swaps)
    uint32_t br2[4] = {10, 20, 30, 40}; // 2 batches of 2
    q.memcpy(a, br2, 4*4); q.wait();
    esimd_batch_bit_reverse(q, a, 1, 4); q.wait();
    q.memcpy(h, a, 4*4); q.wait();
    if (h[0] != 10 || h[1] != 20 || h[2] != 30 || h[3] != 40) errs++;

    fprintf(stderr, "  batch_bit_reverse: %s\n", errs == 0 ? "PASS" : "FAIL"); fflush(stderr);
    if (errs) return -8;

    sycl::free(a, q); sycl::free(b, q); sycl::free(c, q);
    sycl::free(h, q); sycl::free(h2, q); sycl::free(h3, q);

    return 0;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what()); fflush(stderr);
        return -99;
    }
}

} // extern "C"
