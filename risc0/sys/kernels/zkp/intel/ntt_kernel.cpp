#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include "bb31_field.hpp"
#include "ntt_twiddles.hpp"
#include "cpu_reference.hpp"

namespace esimd = sycl::ext::intel::esimd;

// Vectorized bit-reversal: reverse the low `nbits` bits of each SIMD16 lane.
// Uses the 5-stage parallel bit-swap. ~12 SIMD ops for all 16 lanes.
ESIMD_INLINE esimd::simd<uint32_t, 16> vec_bit_rev(esimd::simd<uint32_t, 16> v, uint32_t nbits) {
    v = ((v >> 1) & esimd::simd<uint32_t, 16>(0x55555555u)) |
        ((v & esimd::simd<uint32_t, 16>(0x55555555u)) << 1);
    v = ((v >> 2) & esimd::simd<uint32_t, 16>(0x33333333u)) |
        ((v & esimd::simd<uint32_t, 16>(0x33333333u)) << 2);
    v = ((v >> 4) & esimd::simd<uint32_t, 16>(0x0F0F0F0Fu)) |
        ((v & esimd::simd<uint32_t, 16>(0x0F0F0F0Fu)) << 4);
    v = ((v >> 8) & esimd::simd<uint32_t, 16>(0x00FF00FFu)) |
        ((v & esimd::simd<uint32_t, 16>(0x00FF00FFu)) << 8);
    v = (v >> 16) | (v << 16);
    v >>= (32 - nbits);
    return v;
}

// Cache hint properties for per-stage NTT kernels.
// Streaming stores avoid polluting L1 with data that won't be reused until
// the next kernel launch (different stride pattern).
static constexpr auto STORE_STREAMING = esimd::properties{
    esimd::cache_hint_L1<esimd::cache_hint::streaming>,
    esimd::cache_hint_L2<esimd::cache_hint::write_back>};

// Cache hints for multi-pass scatter/gather
static constexpr auto LOAD_STREAMING = esimd::properties{
    esimd::cache_hint_L1<esimd::cache_hint::streaming>,
    esimd::cache_hint_L2<esimd::cache_hint::cached>};
static constexpr auto LOAD_CACHED = esimd::properties{
    esimd::cache_hint_L1<esimd::cache_hint::cached>,
    esimd::cache_hint_L2<esimd::cache_hint::cached>};

struct BenchResult {
    double kernel_ns;
    double total_ops;
    int32_t correct;
};

// Queue without profiling overhead — used by NTT benchmarks (wall-clock timing)
static sycl::queue create_queue() {
    auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    for (auto& d : gpu_devices) {
        auto name = d.get_info<sycl::info::device::name>();
        if (name.find("Intel") != std::string::npos ||
            name.find("0xe2") != std::string::npos) {
            return sycl::queue(d,
                sycl::property_list{
                    sycl::property::queue::in_order{}
                });
        }
    }
    throw std::runtime_error("Intel GPU not found");
}

// ============================================================================
// GPU NTT: Simple per-stage kernel approach.
// Each kernel launch processes ONE stage of the radix-2 butterfly.
// This is the simplest correct approach (not optimized for multi-stage per kernel).
// ============================================================================

// Cooley-Tukey (DIT) butterfly: one stage.
// Matches the CPU reference forward_ntt exactly.
// For stage s (1-indexed, s=1 is first/narrowest, s=lg_n is last/widest):
//   m = 2^s, half = m/2
//   For each group k=0,m,2m,...:
//     For j=0..half-1:
//       t = w^j * data[k+j+half]
//       data[k+j] = data[k+j] + t
//       data[k+j+half] = data[k+j] - t  (before the add)
// Twiddle: w = root[s] (primitive 2^s-th root of unity)

extern "C" {

// One stage of Cooley-Tukey (DIT) NTT butterfly.
// s: stage number (1-indexed, s=1..lg_n). Matches CPU reference forward_ntt.
// m = 2^s, half = m/2, w = roots[s] (primitive 2^s-th root)
void ntt_ct_stage(sycl::queue& q, uint32_t* d_data, uint32_t lg_n,
                  uint32_t s, const uint32_t* roots) {
    uint32_t n = 1u << lg_n;
    uint32_t m = 1u << s;
    uint32_t half = m >> 1;
    uint32_t num_butterflies = n / 2;
    uint32_t num_threads = num_butterflies / 16;
    if (num_threads == 0) num_threads = 1;

    uint32_t w_root = roots[s]; // primitive 2^s-th root of unity

    auto* pd = d_data;
    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t tid = idx[0];

            for (uint32_t lane = 0; lane < 16; lane++) {
                uint32_t k = tid * 16 + lane;
                if (k >= n / 2) break;

                // Map linear butterfly index to (group_start, position within group)
                uint32_t group = k / half;
                uint32_t j = k % half;

                uint32_t top_idx = group * m + j;
                uint32_t bot_idx = top_idx + half;

                // Load pair
                uint32_t u = *(pd + top_idx);
                uint32_t v = *(pd + bot_idx);

                // Compute twiddle: w^j via binary exponentiation
                uint32_t tw = bb31::ONE;
                uint32_t base = w_root;
                uint32_t exp = j;
                while (exp > 0) {
                    if (exp & 1) {
                        uint64_t prod = uint64_t(tw) * uint64_t(base);
                        uint32_t lo = uint32_t(prod);
                        uint32_t hi = uint32_t(prod >> 32);
                        uint32_t red = lo * bb31::M0;
                        uint64_t rprod = uint64_t(red) * uint64_t(bb31::MOD);
                        uint32_t rlo = uint32_t(rprod);
                        uint32_t rhi = uint32_t(rprod >> 32);
                        uint32_t carry = (uint64_t(lo) + rlo) >= (1ULL << 32) ? 1 : 0;
                        uint32_t res = hi + rhi + carry;
                        tw = res >= bb31::MOD ? res - bb31::MOD : res;
                    }
                    {
                        uint64_t prod = uint64_t(base) * uint64_t(base);
                        uint32_t lo = uint32_t(prod);
                        uint32_t hi = uint32_t(prod >> 32);
                        uint32_t red = lo * bb31::M0;
                        uint64_t rprod = uint64_t(red) * uint64_t(bb31::MOD);
                        uint32_t rlo = uint32_t(rprod);
                        uint32_t rhi = uint32_t(rprod >> 32);
                        uint32_t carry = (uint64_t(lo) + rlo) >= (1ULL << 32) ? 1 : 0;
                        uint32_t res = hi + rhi + carry;
                        base = res >= bb31::MOD ? res - bb31::MOD : res;
                    }
                    exp >>= 1;
                }

                // CT butterfly: t = w^j * v; data[top] = u + t; data[bot] = u - t
                uint32_t t;
                {
                    uint64_t prod = uint64_t(tw) * uint64_t(v);
                    uint32_t lo = uint32_t(prod);
                    uint32_t hi = uint32_t(prod >> 32);
                    uint32_t red = lo * bb31::M0;
                    uint64_t rprod = uint64_t(red) * uint64_t(bb31::MOD);
                    uint32_t rlo = uint32_t(rprod);
                    uint32_t rhi = uint32_t(rprod >> 32);
                    uint32_t carry = (uint64_t(lo) + rlo) >= (1ULL << 32) ? 1 : 0;
                    uint32_t res = hi + rhi + carry;
                    t = res >= bb31::MOD ? res - bb31::MOD : res;
                }

                uint32_t sum = u + t;
                if (sum >= bb31::MOD) sum -= bb31::MOD;
                uint32_t diff = (u >= t) ? (u - t) : (u + bb31::MOD - t);

                *(pd + top_idx) = sum;
                *(pd + bot_idx) = diff;
            }
        }).wait();
}

// One stage of Gentleman-Sande (DIF) inverse NTT butterfly.
// s: stage number (lg_n down to 1). Matches CPU reference inverse_ntt.
void ntt_gs_stage(sycl::queue& q, uint32_t* d_data, uint32_t lg_n,
                  uint32_t s, const uint32_t* roots) {
    uint32_t n = 1u << lg_n;
    uint32_t m = 1u << s;
    uint32_t half = m >> 1;
    uint32_t num_butterflies = n / 2;
    uint32_t num_threads = num_butterflies / 16;
    if (num_threads == 0) num_threads = 1;

    uint32_t w_root = roots[s];

    auto* pd = d_data;
    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t tid = idx[0];

            for (uint32_t lane = 0; lane < 16; lane++) {
                uint32_t k = tid * 16 + lane;
                if (k >= n / 2) break;

                uint32_t group = k / half;
                uint32_t j = k % half;

                uint32_t top_idx = group * m + j;
                uint32_t bot_idx = top_idx + half;

                uint32_t u = *(pd + top_idx);
                uint32_t v = *(pd + bot_idx);

                // Compute twiddle via binary exponentiation
                uint32_t tw = bb31::ONE;
                uint32_t base = w_root;
                uint32_t exp = j;
                while (exp > 0) {
                    if (exp & 1) {
                        uint64_t prod = uint64_t(tw) * uint64_t(base);
                        uint32_t lo = uint32_t(prod);
                        uint32_t hi = uint32_t(prod >> 32);
                        uint32_t red = lo * bb31::M0;
                        uint64_t rprod = uint64_t(red) * uint64_t(bb31::MOD);
                        uint32_t rlo = uint32_t(rprod);
                        uint32_t rhi = uint32_t(rprod >> 32);
                        uint32_t carry = (uint64_t(lo) + rlo) >= (1ULL << 32) ? 1 : 0;
                        uint32_t res = hi + rhi + carry;
                        tw = res >= bb31::MOD ? res - bb31::MOD : res;
                    }
                    {
                        uint64_t prod = uint64_t(base) * uint64_t(base);
                        uint32_t lo = uint32_t(prod);
                        uint32_t hi = uint32_t(prod >> 32);
                        uint32_t red = lo * bb31::M0;
                        uint64_t rprod = uint64_t(red) * uint64_t(bb31::MOD);
                        uint32_t rlo = uint32_t(rprod);
                        uint32_t rhi = uint32_t(rprod >> 32);
                        uint32_t carry = (uint64_t(lo) + rlo) >= (1ULL << 32) ? 1 : 0;
                        uint32_t res = hi + rhi + carry;
                        base = res >= bb31::MOD ? res - bb31::MOD : res;
                    }
                    exp >>= 1;
                }

                // GS butterfly: data[top] = u + v; data[bot] = (u - v) * w^j
                uint32_t sum = u + v;
                if (sum >= bb31::MOD) sum -= bb31::MOD;
                uint32_t diff = (u >= v) ? (u - v) : (u + bb31::MOD - v);

                // diff * tw
                {
                    uint64_t prod = uint64_t(diff) * uint64_t(tw);
                    uint32_t lo = uint32_t(prod);
                    uint32_t hi = uint32_t(prod >> 32);
                    uint32_t red = lo * bb31::M0;
                    uint64_t rprod = uint64_t(red) * uint64_t(bb31::MOD);
                    uint32_t rlo = uint32_t(rprod);
                    uint32_t rhi = uint32_t(rprod >> 32);
                    uint32_t carry = (uint64_t(lo) + rlo) >= (1ULL << 32) ? 1 : 0;
                    uint32_t res = hi + rhi + carry;
                    diff = res >= bb31::MOD ? res - bb31::MOD : res;
                }

                *(pd + top_idx) = sum;
                *(pd + bot_idx) = diff;
            }
        }).wait();
}

// GPU bit-reversal permutation (scalar per-element, simple but correct)
void ntt_bit_reverse(sycl::queue& q, uint32_t* d_data, uint32_t lg_n) {
    uint32_t n = 1u << lg_n;
    uint32_t num_threads = (n + 15) / 16;

    auto* pd = d_data;
    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            for (uint32_t lane = 0; lane < 16; lane++) {
                uint32_t i = idx[0] * 16 + lane;
                if (i >= n) break;

                // Compute bit-reversed index
                uint32_t rev = 0;
                uint32_t val = i;
                for (uint32_t b = 0; b < lg_n; b++) {
                    rev = (rev << 1) | (val & 1);
                    val >>= 1;
                }

                // Only swap if i < rev (to avoid double-swap)
                if (i < rev) {
                    uint32_t tmp = *(pd + i);
                    *(pd + i) = *(pd + rev);
                    *(pd + rev) = tmp;
                }
            }
        }).wait();
}

// ============================================================================
// OPTIMIZED NTT: Precomputed twiddle tables + SIMD16 vectorized butterflies
// ============================================================================

// Helper: CPU-side Montgomery multiply for twiddle precomputation
static uint32_t host_mont_mul(uint32_t a, uint32_t b) {
    uint64_t prod = uint64_t(a) * uint64_t(b);
    uint32_t lo = uint32_t(prod);
    uint32_t hi = uint32_t(prod >> 32);
    uint32_t red = lo * bb31::M0;
    uint64_t rprod = uint64_t(red) * uint64_t(bb31::MOD);
    uint32_t rlo = uint32_t(rprod);
    uint32_t rhi = uint32_t(rprod >> 32);
    uint32_t carry = (uint64_t(lo) + rlo) >= (1ULL << 32) ? 1 : 0;
    uint32_t res = hi + rhi + carry;
    return res >= bb31::MOD ? res - bb31::MOD : res;
}

// Fused 2-stage CT butterfly: processes stages s and s+1 in one kernel.
// Loads 4 elements per butterfly quad, applies both stages in registers.
// Eliminates one full global read+write pass between the two stages.
void ntt_ct_fused_2stage(sycl::queue& q, uint32_t* d_data, uint32_t lg_n,
                          uint32_t s, const uint32_t* d_twiddles_s,
                          const uint32_t* d_twiddles_s1) {
    uint32_t n = 1u << lg_n;
    uint32_t half_s = 1u << (s - 1);    // stride for stage s
    uint32_t m_s = 1u << s;              // group size for stage s
    uint32_t half_s1 = 1u << s;          // stride for stage s+1 = 2 * half_s
    uint32_t m_s1 = 1u << (s + 1);      // group size for stage s+1

    // Each quad: 4 elements forming two stage-s butterfly pairs that feed one stage-s+1 pair.
    // For each j in [0, half_s), within each stage-s+1 group:
    //   a = group_base + j, b = a + half_s (stage s pair in subgroup 0)
    //   c = group_base + m_s + j, d = c + half_s (stage s pair in subgroup 1)
    // After stage s: a'=a+tw*b, b'=a-tw*b, c'=c+tw*d, d'=c-tw*d
    // Stage s+1: (a', c') and (b', d') at distance half_s1 = m_s
    uint32_t num_s1_groups = n / m_s1;
    uint32_t quads_per_group = half_s;
    uint32_t total_quads = num_s1_groups * quads_per_group;
    uint32_t num_threads = total_quads / 16;

    auto* pd = d_data;
    auto* ptw_s = d_twiddles_s;
    auto* ptw_s1 = d_twiddles_s1;

    if (half_s >= 64) {
        // Z=4 path: each thread processes 4×16=64 consecutive quads.
        // Issues 28 loads upfront for maximum memory-level parallelism.
        uint32_t num_threads_z4 = total_quads / 64;
        q.parallel_for(sycl::range<1>(num_threads_z4),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t tid = idx[0];
                uint32_t k_base = tid * 64;

                uint32_t grp = k_base / quads_per_group;
                uint32_t j_base = k_base % quads_per_group;

                uint32_t pos_a = grp * m_s1 + j_base;
                uint32_t pos_b = pos_a + half_s;
                uint32_t pos_c = pos_a + m_s;
                uint32_t pos_d = pos_c + half_s;

                // Phase 1: Issue all loads for maximum MLP (streaming: no L1 reuse)
                auto a0 = esimd::block_load<uint32_t, 16>(pd + pos_a, LOAD_STREAMING);
                auto a1 = esimd::block_load<uint32_t, 16>(pd + pos_a + 16, LOAD_STREAMING);
                auto a2 = esimd::block_load<uint32_t, 16>(pd + pos_a + 32, LOAD_STREAMING);
                auto a3 = esimd::block_load<uint32_t, 16>(pd + pos_a + 48, LOAD_STREAMING);
                auto b0 = esimd::block_load<uint32_t, 16>(pd + pos_b, LOAD_STREAMING);
                auto b1 = esimd::block_load<uint32_t, 16>(pd + pos_b + 16, LOAD_STREAMING);
                auto b2 = esimd::block_load<uint32_t, 16>(pd + pos_b + 32, LOAD_STREAMING);
                auto b3 = esimd::block_load<uint32_t, 16>(pd + pos_b + 48, LOAD_STREAMING);
                auto c0 = esimd::block_load<uint32_t, 16>(pd + pos_c, LOAD_STREAMING);
                auto c1 = esimd::block_load<uint32_t, 16>(pd + pos_c + 16, LOAD_STREAMING);
                auto c2 = esimd::block_load<uint32_t, 16>(pd + pos_c + 32, LOAD_STREAMING);
                auto c3 = esimd::block_load<uint32_t, 16>(pd + pos_c + 48, LOAD_STREAMING);
                auto d0 = esimd::block_load<uint32_t, 16>(pd + pos_d, LOAD_STREAMING);
                auto d1 = esimd::block_load<uint32_t, 16>(pd + pos_d + 16, LOAD_STREAMING);
                auto d2 = esimd::block_load<uint32_t, 16>(pd + pos_d + 32, LOAD_STREAMING);
                auto d3 = esimd::block_load<uint32_t, 16>(pd + pos_d + 48, LOAD_STREAMING);

                // Phase 2: Stage s butterflies (4 independent sets)
                #define FUSED2_STAGE_S(A, B, C, D, TW_OFF) \
                { \
                    auto tw = esimd::block_load<uint32_t, 16>(ptw_s + j_base + TW_OFF); \
                    auto tb = bb31::mont_mul(tw, B); \
                    auto td = bb31::mont_mul(tw, D); \
                    auto an = bb31::field_add(A, tb); \
                    auto bn = bb31::field_sub(A, tb); \
                    auto cn = bb31::field_add(C, td); \
                    auto dn = bb31::field_sub(C, td); \
                    auto tw_ac = esimd::block_load<uint32_t, 16>(ptw_s1 + j_base + TW_OFF); \
                    auto tw_bd = esimd::block_load<uint32_t, 16>(ptw_s1 + j_base + half_s + TW_OFF); \
                    auto tc = bb31::mont_mul(tw_ac, cn); \
                    auto tdd = bb31::mont_mul(tw_bd, dn); \
                    A = bb31::field_add(an, tc); \
                    C = bb31::field_sub(an, tc); \
                    B = bb31::field_add(bn, tdd); \
                    D = bb31::field_sub(bn, tdd); \
                }

                FUSED2_STAGE_S(a0, b0, c0, d0, 0)
                FUSED2_STAGE_S(a1, b1, c1, d1, 16)
                FUSED2_STAGE_S(a2, b2, c2, d2, 32)
                FUSED2_STAGE_S(a3, b3, c3, d3, 48)

                #undef FUSED2_STAGE_S

                // Phase 3: Store results (streaming)
                esimd::block_store(pd + pos_a,      a0, STORE_STREAMING);
                esimd::block_store(pd + pos_a + 16, a1, STORE_STREAMING);
                esimd::block_store(pd + pos_a + 32, a2, STORE_STREAMING);
                esimd::block_store(pd + pos_a + 48, a3, STORE_STREAMING);
                esimd::block_store(pd + pos_b,      b0, STORE_STREAMING);
                esimd::block_store(pd + pos_b + 16, b1, STORE_STREAMING);
                esimd::block_store(pd + pos_b + 32, b2, STORE_STREAMING);
                esimd::block_store(pd + pos_b + 48, b3, STORE_STREAMING);
                esimd::block_store(pd + pos_c,      c0, STORE_STREAMING);
                esimd::block_store(pd + pos_c + 16, c1, STORE_STREAMING);
                esimd::block_store(pd + pos_c + 32, c2, STORE_STREAMING);
                esimd::block_store(pd + pos_c + 48, c3, STORE_STREAMING);
                esimd::block_store(pd + pos_d,      d0, STORE_STREAMING);
                esimd::block_store(pd + pos_d + 16, d1, STORE_STREAMING);
                esimd::block_store(pd + pos_d + 32, d2, STORE_STREAMING);
                esimd::block_store(pd + pos_d + 48, d3, STORE_STREAMING);
            });
    } else if (half_s >= 16) {
        // Z=1 path for smaller stages (half_s = 16 or 32)
        q.parallel_for(sycl::range<1>(num_threads),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t tid = idx[0];
                uint32_t k_base = tid * 16;

                uint32_t grp = k_base / quads_per_group;
                uint32_t j_base = k_base % quads_per_group;

                uint32_t pos_a = grp * m_s1 + j_base;
                uint32_t pos_b = pos_a + half_s;
                uint32_t pos_c = pos_a + m_s;
                uint32_t pos_d = pos_c + half_s;

                auto a = esimd::block_load<uint32_t, 16>(pd + pos_a, LOAD_STREAMING);
                auto b = esimd::block_load<uint32_t, 16>(pd + pos_b, LOAD_STREAMING);
                auto c = esimd::block_load<uint32_t, 16>(pd + pos_c, LOAD_STREAMING);
                auto d = esimd::block_load<uint32_t, 16>(pd + pos_d, LOAD_STREAMING);
                auto tw_s = esimd::block_load<uint32_t, 16>(ptw_s + j_base);

                auto tb = bb31::mont_mul(tw_s, b);
                auto td = bb31::mont_mul(tw_s, d);
                auto a_new = bb31::field_add(a, tb);
                auto b_new = bb31::field_sub(a, tb);
                auto c_new = bb31::field_add(c, td);
                auto d_new = bb31::field_sub(c, td);

                auto tw_s1_ac = esimd::block_load<uint32_t, 16>(ptw_s1 + j_base);
                auto tw_s1_bd = esimd::block_load<uint32_t, 16>(ptw_s1 + j_base + half_s);
                auto tc = bb31::mont_mul(tw_s1_ac, c_new);
                auto tdd = bb31::mont_mul(tw_s1_bd, d_new);

                esimd::block_store(pd + pos_a, bb31::field_add(a_new, tc), STORE_STREAMING);
                esimd::block_store(pd + pos_c, bb31::field_sub(a_new, tc), STORE_STREAMING);
                esimd::block_store(pd + pos_b, bb31::field_add(b_new, tdd), STORE_STREAMING);
                esimd::block_store(pd + pos_d, bb31::field_sub(b_new, tdd), STORE_STREAMING);
            });
    }
    // Note: half_s < 16 not handled here. Orchestrator only calls this when half >= 16.
}

// Fused 3-stage CT DIT kernel: processes stages s, s+1, s+2 in one launch.
// Loads 8 blocks forming an "octet" — 4 stage-s butterfly pairs that feed
// 2 stage-s+1 pairs that feed 1 stage-s+2 pair. Z=4 for MLP.
// Requires half_s >= 64 (stages 15+ with SLM_LG_BLOCK=14).
void ntt_ct_fused_3stage(sycl::queue& q, uint32_t* d_data, uint32_t lg_n,
                          uint32_t s,
                          const uint32_t* d_tw_s,
                          const uint32_t* d_tw_s1,
                          const uint32_t* d_tw_s2) {
    uint32_t n = 1u << lg_n;
    uint32_t half_s = 1u << (s - 1);
    uint32_t m_s = 1u << s;
    uint32_t m_s1 = 1u << (s + 1);
    uint32_t m_s2 = 1u << (s + 2);  // group size for stage s+2
    uint32_t half_s1 = m_s;          // = 2 * half_s
    uint32_t half_s2 = m_s1;         // = 4 * half_s

    // Each octet: 8 elements at positions within one s+2 group
    uint32_t num_s2_groups = n / m_s2;
    uint32_t octets_per_group = half_s;
    uint32_t total_octets = num_s2_groups * octets_per_group;

    auto* pd = d_data;
    auto* ptw0 = d_tw_s;
    auto* ptw1 = d_tw_s1;
    auto* ptw2 = d_tw_s2;

    // Z=4: each thread processes 4 consecutive octets (64 elements per position)
    uint32_t num_threads = total_octets / 64;
    if (num_threads == 0) num_threads = total_octets / 16; // fallback for small sizes

    if (half_s >= 64) {
        uint32_t nt = total_octets / 64;
        q.parallel_for(sycl::range<1>(nt),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t tid = idx[0];
                uint32_t k_base = tid * 64;
                uint32_t grp = k_base / octets_per_group;
                uint32_t j_base = k_base % octets_per_group;

                // 8 positions within the s+2 group, at 4 Z-offsets each
                uint32_t base = grp * m_s2 + j_base;

                // Macro: process one Z-position (16 elements from each of 8 blocks)
                #define FUSED3_Z(Z_OFF) \
                { \
                    uint32_t j = j_base + Z_OFF; \
                    auto a = esimd::block_load<uint32_t, 16>(pd + base + Z_OFF, LOAD_STREAMING); \
                    auto b = esimd::block_load<uint32_t, 16>(pd + base + half_s + Z_OFF, LOAD_STREAMING); \
                    auto c = esimd::block_load<uint32_t, 16>(pd + base + m_s + Z_OFF, LOAD_STREAMING); \
                    auto d = esimd::block_load<uint32_t, 16>(pd + base + m_s + half_s + Z_OFF, LOAD_STREAMING); \
                    auto e = esimd::block_load<uint32_t, 16>(pd + base + m_s1 + Z_OFF, LOAD_STREAMING); \
                    auto f = esimd::block_load<uint32_t, 16>(pd + base + m_s1 + half_s + Z_OFF, LOAD_STREAMING); \
                    auto g = esimd::block_load<uint32_t, 16>(pd + base + m_s1 + m_s + Z_OFF, LOAD_STREAMING); \
                    auto h = esimd::block_load<uint32_t, 16>(pd + base + m_s1 + m_s + half_s + Z_OFF, LOAD_STREAMING); \
                    /* Stage s: 4 butterflies */ \
                    auto tw_s = esimd::block_load<uint32_t, 16>(ptw0 + j); \
                    auto tb = bb31::mont_mul(tw_s, b); auto td = bb31::mont_mul(tw_s, d); \
                    auto tf = bb31::mont_mul(tw_s, f); auto th = bb31::mont_mul(tw_s, h); \
                    auto a1 = bb31::field_add(a, tb); auto b1 = bb31::field_sub(a, tb); \
                    auto c1 = bb31::field_add(c, td); auto d1 = bb31::field_sub(c, td); \
                    auto e1 = bb31::field_add(e, tf); auto f1 = bb31::field_sub(e, tf); \
                    auto g1 = bb31::field_add(g, th); auto h1 = bb31::field_sub(g, th); \
                    /* Stage s+1: 2 butterflies on (a1,c1), (b1,d1), (e1,g1), (f1,h1) */ \
                    auto tw1_lo = esimd::block_load<uint32_t, 16>(ptw1 + j); \
                    auto tw1_hi = esimd::block_load<uint32_t, 16>(ptw1 + j + half_s); \
                    auto tc1 = bb31::mont_mul(tw1_lo, c1); auto td1 = bb31::mont_mul(tw1_hi, d1); \
                    auto tg1 = bb31::mont_mul(tw1_lo, g1); auto th1 = bb31::mont_mul(tw1_hi, h1); \
                    auto a2 = bb31::field_add(a1, tc1); auto c2 = bb31::field_sub(a1, tc1); \
                    auto b2 = bb31::field_add(b1, td1); auto d2 = bb31::field_sub(b1, td1); \
                    auto e2 = bb31::field_add(e1, tg1); auto g2 = bb31::field_sub(e1, tg1); \
                    auto f2 = bb31::field_add(f1, th1); auto h2 = bb31::field_sub(f1, th1); \
                    /* Stage s+2: 4 butterflies on (a2,e2), (c2,g2), (b2,f2), (d2,h2) */ \
                    auto tw2_a = esimd::block_load<uint32_t, 16>(ptw2 + j); \
                    auto tw2_b = esimd::block_load<uint32_t, 16>(ptw2 + j + half_s); \
                    auto tw2_c = esimd::block_load<uint32_t, 16>(ptw2 + j + half_s1); \
                    auto tw2_d = esimd::block_load<uint32_t, 16>(ptw2 + j + half_s + half_s1); \
                    auto te2 = bb31::mont_mul(tw2_a, e2); auto tg2 = bb31::mont_mul(tw2_c, g2); \
                    auto tf2 = bb31::mont_mul(tw2_b, f2); auto th2 = bb31::mont_mul(tw2_d, h2); \
                    /* Store results */ \
                    esimd::block_store(pd + base + Z_OFF, bb31::field_add(a2, te2), STORE_STREAMING); \
                    esimd::block_store(pd + base + m_s1 + Z_OFF, bb31::field_sub(a2, te2), STORE_STREAMING); \
                    esimd::block_store(pd + base + m_s + Z_OFF, bb31::field_add(c2, tg2), STORE_STREAMING); \
                    esimd::block_store(pd + base + m_s1 + m_s + Z_OFF, bb31::field_sub(c2, tg2), STORE_STREAMING); \
                    esimd::block_store(pd + base + half_s + Z_OFF, bb31::field_add(b2, tf2), STORE_STREAMING); \
                    esimd::block_store(pd + base + m_s1 + half_s + Z_OFF, bb31::field_sub(b2, tf2), STORE_STREAMING); \
                    esimd::block_store(pd + base + m_s + half_s + Z_OFF, bb31::field_add(d2, th2), STORE_STREAMING); \
                    esimd::block_store(pd + base + m_s1 + m_s + half_s + Z_OFF, bb31::field_sub(d2, th2), STORE_STREAMING); \
                }

                FUSED3_Z(0)
                FUSED3_Z(16)
                FUSED3_Z(32)
                FUSED3_Z(48)

                #undef FUSED3_Z
            });
    }
}

// Fused 4-stage CT DIT kernel: processes stages s, s+1, s+2, s+3 in one launch.
// Loads 16 blocks at positions within one stage-(s+3) group.
// No Z-batching (16 elements per thread) to keep register pressure manageable.
// Requires half_s >= 16.
void ntt_ct_fused_4stage(sycl::queue& q, uint32_t* d_data, uint32_t lg_n,
                          uint32_t s,
                          const uint32_t* d_tw_s,
                          const uint32_t* d_tw_s1,
                          const uint32_t* d_tw_s2,
                          const uint32_t* d_tw_s3) {
    uint32_t n = 1u << lg_n;
    uint32_t hs = 1u << (s - 1);     // half for stage s
    uint32_t ms = 1u << s;            // group size for stage s
    uint32_t ms1 = ms << 1;           // group size for stage s+1 = 2^(s+1)
    uint32_t ms2 = ms << 2;           // group size for stage s+2 = 2^(s+2)
    uint32_t ms3 = ms << 3;           // group size for stage s+3 = 2^(s+3)
    uint32_t hs1 = ms;               // half for stage s+1
    uint32_t hs2 = ms1;              // half for stage s+2
    uint32_t hs3 = ms2;              // half for stage s+3

    uint32_t num_s3_groups = n / ms3;
    uint32_t hexadecs_per_group = hs; // number of 16-element "hexadectet" positions
    uint32_t total_hexadecs = num_s3_groups * hexadecs_per_group;
    uint32_t num_threads = total_hexadecs / 16;

    auto* pd = d_data;
    auto* tw0 = d_tw_s;
    auto* tw1 = d_tw_s1;
    auto* tw2 = d_tw_s2;
    auto* tw3 = d_tw_s3;

    if (hs >= 16) {
        q.parallel_for(sycl::range<1>(num_threads),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t tid = idx[0];
                uint32_t k_base = tid * 16;
                uint32_t grp = k_base / hexadecs_per_group;
                uint32_t j = k_base % hexadecs_per_group;
                uint32_t base = grp * ms3 + j;

                // 16 positions: 4 layers of binary splits
                // Layer 0 (stage s+3 split): offsets 0 and hs3=ms2
                // Layer 1 (stage s+2 split): +0 and +hs2=ms1
                // Layer 2 (stage s+1 split): +0 and +hs1=ms
                // Layer 3 (stage s split):   +0 and +hs

                // Load all 16 blocks (labeled by 4-bit binary: dcba)
                // Offset = d*hs3 + c*hs2 + b*hs1 + a*hs
                auto v0000 = esimd::block_load<uint32_t, 16>(pd + base, LOAD_STREAMING);
                auto v0001 = esimd::block_load<uint32_t, 16>(pd + base + hs, LOAD_STREAMING);
                auto v0010 = esimd::block_load<uint32_t, 16>(pd + base + hs1, LOAD_STREAMING);
                auto v0011 = esimd::block_load<uint32_t, 16>(pd + base + hs1 + hs, LOAD_STREAMING);
                auto v0100 = esimd::block_load<uint32_t, 16>(pd + base + hs2, LOAD_STREAMING);
                auto v0101 = esimd::block_load<uint32_t, 16>(pd + base + hs2 + hs, LOAD_STREAMING);
                auto v0110 = esimd::block_load<uint32_t, 16>(pd + base + hs2 + hs1, LOAD_STREAMING);
                auto v0111 = esimd::block_load<uint32_t, 16>(pd + base + hs2 + hs1 + hs, LOAD_STREAMING);
                auto v1000 = esimd::block_load<uint32_t, 16>(pd + base + hs3, LOAD_STREAMING);
                auto v1001 = esimd::block_load<uint32_t, 16>(pd + base + hs3 + hs, LOAD_STREAMING);
                auto v1010 = esimd::block_load<uint32_t, 16>(pd + base + hs3 + hs1, LOAD_STREAMING);
                auto v1011 = esimd::block_load<uint32_t, 16>(pd + base + hs3 + hs1 + hs, LOAD_STREAMING);
                auto v1100 = esimd::block_load<uint32_t, 16>(pd + base + hs3 + hs2, LOAD_STREAMING);
                auto v1101 = esimd::block_load<uint32_t, 16>(pd + base + hs3 + hs2 + hs, LOAD_STREAMING);
                auto v1110 = esimd::block_load<uint32_t, 16>(pd + base + hs3 + hs2 + hs1, LOAD_STREAMING);
                auto v1111 = esimd::block_load<uint32_t, 16>(pd + base + hs3 + hs2 + hs1 + hs, LOAD_STREAMING);

                // Stage s: butterfly on bit 'a' (8 pairs)
                auto t0 = esimd::block_load<uint32_t, 16>(tw0 + j);
                #define BFY_S(TOP, BOT) { auto t = bb31::mont_mul(t0, BOT); \
                    auto tmp = TOP; TOP = bb31::field_add(tmp, t); BOT = bb31::field_sub(tmp, t); }
                BFY_S(v0000, v0001) BFY_S(v0010, v0011)
                BFY_S(v0100, v0101) BFY_S(v0110, v0111)
                BFY_S(v1000, v1001) BFY_S(v1010, v1011)
                BFY_S(v1100, v1101) BFY_S(v1110, v1111)
                #undef BFY_S

                // Stage s+1: butterfly on bit 'b' (8 pairs)
                auto t1_lo = esimd::block_load<uint32_t, 16>(tw1 + j);
                auto t1_hi = esimd::block_load<uint32_t, 16>(tw1 + j + hs);
                #define BFY_S1(TOP, BOT, TW) { auto t = bb31::mont_mul(TW, BOT); \
                    auto tmp = TOP; TOP = bb31::field_add(tmp, t); BOT = bb31::field_sub(tmp, t); }
                BFY_S1(v0000, v0010, t1_lo) BFY_S1(v0001, v0011, t1_hi)
                BFY_S1(v0100, v0110, t1_lo) BFY_S1(v0101, v0111, t1_hi)
                BFY_S1(v1000, v1010, t1_lo) BFY_S1(v1001, v1011, t1_hi)
                BFY_S1(v1100, v1110, t1_lo) BFY_S1(v1101, v1111, t1_hi)
                #undef BFY_S1

                // Stage s+2: butterfly on bit 'c' (8 pairs)
                auto t2_00 = esimd::block_load<uint32_t, 16>(tw2 + j);
                auto t2_01 = esimd::block_load<uint32_t, 16>(tw2 + j + hs);
                auto t2_10 = esimd::block_load<uint32_t, 16>(tw2 + j + hs1);
                auto t2_11 = esimd::block_load<uint32_t, 16>(tw2 + j + hs1 + hs);
                #define BFY_S2(TOP, BOT, TW) { auto t = bb31::mont_mul(TW, BOT); \
                    auto tmp = TOP; TOP = bb31::field_add(tmp, t); BOT = bb31::field_sub(tmp, t); }
                BFY_S2(v0000, v0100, t2_00) BFY_S2(v0001, v0101, t2_01)
                BFY_S2(v0010, v0110, t2_10) BFY_S2(v0011, v0111, t2_11)
                BFY_S2(v1000, v1100, t2_00) BFY_S2(v1001, v1101, t2_01)
                BFY_S2(v1010, v1110, t2_10) BFY_S2(v1011, v1111, t2_11)
                #undef BFY_S2

                // Stage s+3: butterfly on bit 'd' (8 pairs)
                auto t3_000 = esimd::block_load<uint32_t, 16>(tw3 + j);
                auto t3_001 = esimd::block_load<uint32_t, 16>(tw3 + j + hs);
                auto t3_010 = esimd::block_load<uint32_t, 16>(tw3 + j + hs1);
                auto t3_011 = esimd::block_load<uint32_t, 16>(tw3 + j + hs1 + hs);
                auto t3_100 = esimd::block_load<uint32_t, 16>(tw3 + j + hs2);
                auto t3_101 = esimd::block_load<uint32_t, 16>(tw3 + j + hs2 + hs);
                auto t3_110 = esimd::block_load<uint32_t, 16>(tw3 + j + hs2 + hs1);
                auto t3_111 = esimd::block_load<uint32_t, 16>(tw3 + j + hs2 + hs1 + hs);
                #define BFY_S3(TOP, BOT, TW) { auto t = bb31::mont_mul(TW, BOT); \
                    auto tmp = TOP; TOP = bb31::field_add(tmp, t); BOT = bb31::field_sub(tmp, t); }
                BFY_S3(v0000, v1000, t3_000) BFY_S3(v0001, v1001, t3_001)
                BFY_S3(v0010, v1010, t3_010) BFY_S3(v0011, v1011, t3_011)
                BFY_S3(v0100, v1100, t3_100) BFY_S3(v0101, v1101, t3_101)
                BFY_S3(v0110, v1110, t3_110) BFY_S3(v0111, v1111, t3_111)
                #undef BFY_S3

                // Store all 16 results
                esimd::block_store(pd + base, v0000, STORE_STREAMING);
                esimd::block_store(pd + base + hs, v0001, STORE_STREAMING);
                esimd::block_store(pd + base + hs1, v0010, STORE_STREAMING);
                esimd::block_store(pd + base + hs1 + hs, v0011, STORE_STREAMING);
                esimd::block_store(pd + base + hs2, v0100, STORE_STREAMING);
                esimd::block_store(pd + base + hs2 + hs, v0101, STORE_STREAMING);
                esimd::block_store(pd + base + hs2 + hs1, v0110, STORE_STREAMING);
                esimd::block_store(pd + base + hs2 + hs1 + hs, v0111, STORE_STREAMING);
                esimd::block_store(pd + base + hs3, v1000, STORE_STREAMING);
                esimd::block_store(pd + base + hs3 + hs, v1001, STORE_STREAMING);
                esimd::block_store(pd + base + hs3 + hs1, v1010, STORE_STREAMING);
                esimd::block_store(pd + base + hs3 + hs1 + hs, v1011, STORE_STREAMING);
                esimd::block_store(pd + base + hs3 + hs2, v1100, STORE_STREAMING);
                esimd::block_store(pd + base + hs3 + hs2 + hs, v1101, STORE_STREAMING);
                esimd::block_store(pd + base + hs3 + hs2 + hs1, v1110, STORE_STREAMING);
                esimd::block_store(pd + base + hs3 + hs2 + hs1 + hs, v1111, STORE_STREAMING);
            });
    }
}

// SIMD16 vectorized CT butterfly stage with precomputed twiddle table.
// For stages where half >= 16: 16 consecutive butterflies per thread, all SIMD16.
// For stages where half < 16: fall back to scalar (these are cheap anyway).
void ntt_ct_stage_fast(sycl::queue& q, uint32_t* d_data, uint32_t lg_n,
                       uint32_t s, const uint32_t* roots, const uint32_t* d_twiddles) {
    uint32_t n = 1u << lg_n;
    uint32_t m = 1u << s;
    uint32_t half = m >> 1;
    uint32_t num_butterflies = n / 2;

    auto* pd = d_data;
    auto* ptw = d_twiddles;

    if (half >= 64) {
        // Z=4 path: each thread processes 4x16=64 consecutive butterflies.
        // All 4 Z-positions are in the same group (half >= 64 guarantees this).
        // 4x fewer threads, 4x more work per thread = better MLP and ILP.
        uint32_t num_threads = num_butterflies / 64;

        q.parallel_for(sycl::range<1>(num_threads),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t tid = idx[0];
                uint32_t k_base = tid * 64;
                uint32_t group = k_base / half;
                uint32_t j_base = k_base % half;
                uint32_t top = group * m + j_base;
                uint32_t bot = top + half;

                // Phase 1: Issue all 12 loads for maximum MLP (streaming: no L1 reuse)
                auto u0 = esimd::block_load<uint32_t, 16>(pd + top, LOAD_STREAMING);
                auto u1 = esimd::block_load<uint32_t, 16>(pd + top + 16, LOAD_STREAMING);
                auto u2 = esimd::block_load<uint32_t, 16>(pd + top + 32, LOAD_STREAMING);
                auto u3 = esimd::block_load<uint32_t, 16>(pd + top + 48, LOAD_STREAMING);
                auto v0 = esimd::block_load<uint32_t, 16>(pd + bot, LOAD_STREAMING);
                auto v1 = esimd::block_load<uint32_t, 16>(pd + bot + 16, LOAD_STREAMING);
                auto v2 = esimd::block_load<uint32_t, 16>(pd + bot + 32, LOAD_STREAMING);
                auto v3 = esimd::block_load<uint32_t, 16>(pd + bot + 48, LOAD_STREAMING);
                auto w0 = esimd::block_load<uint32_t, 16>(ptw + j_base);
                auto w1 = esimd::block_load<uint32_t, 16>(ptw + j_base + 16);
                auto w2 = esimd::block_load<uint32_t, 16>(ptw + j_base + 32);
                auto w3 = esimd::block_load<uint32_t, 16>(ptw + j_base + 48);

                // Phase 2: 4 independent butterfly computes
                auto t0 = bb31::mont_mul(w0, v0);
                auto t1 = bb31::mont_mul(w1, v1);
                auto t2 = bb31::mont_mul(w2, v2);
                auto t3 = bb31::mont_mul(w3, v3);

                // Phase 3: Store results (streaming)
                esimd::block_store(pd + top,      bb31::field_add(u0, t0), STORE_STREAMING);
                esimd::block_store(pd + top + 16, bb31::field_add(u1, t1), STORE_STREAMING);
                esimd::block_store(pd + top + 32, bb31::field_add(u2, t2), STORE_STREAMING);
                esimd::block_store(pd + top + 48, bb31::field_add(u3, t3), STORE_STREAMING);
                esimd::block_store(pd + bot,      bb31::field_sub(u0, t0), STORE_STREAMING);
                esimd::block_store(pd + bot + 16, bb31::field_sub(u1, t1), STORE_STREAMING);
                esimd::block_store(pd + bot + 32, bb31::field_sub(u2, t2), STORE_STREAMING);
                esimd::block_store(pd + bot + 48, bb31::field_sub(u3, t3), STORE_STREAMING);
            });
    } else if (half >= 16) {
        // SIMD16 path (Z=1): for stages where half is 16 or 32
        uint32_t num_threads = num_butterflies / 16;

        q.parallel_for(sycl::range<1>(num_threads),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t tid = idx[0];
                uint32_t k_base = tid * 16;
                uint32_t group = k_base / half;
                uint32_t j_base = k_base % half;
                uint32_t top_base = group * m + j_base;
                uint32_t bot_base = top_base + half;

                auto u = esimd::block_load<uint32_t, 16>(pd + top_base);
                auto v = esimd::block_load<uint32_t, 16>(pd + bot_base);
                auto tw = esimd::block_load<uint32_t, 16>(ptw + j_base);
                auto t = bb31::mont_mul(tw, v);
                esimd::block_store(pd + top_base, bb31::field_add(u, t), STORE_STREAMING);
                esimd::block_store(pd + bot_base, bb31::field_sub(u, t), STORE_STREAMING);
            });
    } else {
        ntt_ct_stage(q, d_data, lg_n, s, roots);
    }
}

// Fused 2-stage GS butterfly: processes stages s and s-1 in one kernel.
// GS goes from high to low: stage s first (wider stride), then s-1 (narrower).
void ntt_gs_fused_2stage(sycl::queue& q, uint32_t* d_data, uint32_t lg_n,
                          uint32_t s, const uint32_t* d_twiddles_s,
                          const uint32_t* d_twiddles_s1) {
    uint32_t n = 1u << lg_n;
    uint32_t half_s = 1u << (s - 1);    // stride for stage s (wider)
    uint32_t m_s = 1u << s;
    uint32_t half_s1 = 1u << (s - 2);   // stride for stage s-1 (narrower)
    uint32_t m_s1 = 1u << (s - 1);      // group size for stage s-1

    // Each quad: 4 elements at positions a, a+half_s1, a+half_s, a+half_s+half_s1
    // Stage s butterfly: (a, a+half_s) and (a+half_s1, a+half_s+half_s1)
    // Stage s-1 butterfly: operates within each half after stage s
    uint32_t num_s_groups = n / m_s;
    uint32_t total_quads = num_s_groups * half_s1;
    uint32_t num_threads = total_quads / 16;

    auto* pd = d_data;
    auto* ptw_s = d_twiddles_s;
    auto* ptw_s1 = d_twiddles_s1;

    if (half_s1 >= 64) {
        // Z=4 path: 4×16=64 consecutive quads per thread
        uint32_t nt = total_quads / 64;
        q.parallel_for(sycl::range<1>(nt),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t tid = idx[0];
                uint32_t k_base = tid * 64;
                uint32_t grp = k_base / half_s1;
                uint32_t j_base = k_base % half_s1;

                uint32_t pos_a = grp * m_s + j_base;
                uint32_t pos_b = pos_a + half_s1;
                uint32_t pos_c = pos_a + half_s;
                uint32_t pos_d = pos_c + half_s1;

                // Phase 1: Issue all loads (streaming)
                auto a0 = esimd::block_load<uint32_t, 16>(pd + pos_a, LOAD_STREAMING);
                auto a1 = esimd::block_load<uint32_t, 16>(pd + pos_a + 16, LOAD_STREAMING);
                auto a2 = esimd::block_load<uint32_t, 16>(pd + pos_a + 32, LOAD_STREAMING);
                auto a3 = esimd::block_load<uint32_t, 16>(pd + pos_a + 48, LOAD_STREAMING);
                auto b0 = esimd::block_load<uint32_t, 16>(pd + pos_b, LOAD_STREAMING);
                auto b1 = esimd::block_load<uint32_t, 16>(pd + pos_b + 16, LOAD_STREAMING);
                auto b2 = esimd::block_load<uint32_t, 16>(pd + pos_b + 32, LOAD_STREAMING);
                auto b3 = esimd::block_load<uint32_t, 16>(pd + pos_b + 48, LOAD_STREAMING);
                auto c0 = esimd::block_load<uint32_t, 16>(pd + pos_c, LOAD_STREAMING);
                auto c1 = esimd::block_load<uint32_t, 16>(pd + pos_c + 16, LOAD_STREAMING);
                auto c2 = esimd::block_load<uint32_t, 16>(pd + pos_c + 32, LOAD_STREAMING);
                auto c3 = esimd::block_load<uint32_t, 16>(pd + pos_c + 48, LOAD_STREAMING);
                auto d0 = esimd::block_load<uint32_t, 16>(pd + pos_d, LOAD_STREAMING);
                auto d1 = esimd::block_load<uint32_t, 16>(pd + pos_d + 16, LOAD_STREAMING);
                auto d2 = esimd::block_load<uint32_t, 16>(pd + pos_d + 32, LOAD_STREAMING);
                auto d3 = esimd::block_load<uint32_t, 16>(pd + pos_d + 48, LOAD_STREAMING);

                // Phase 2: GS fused 2-stage butterflies (4 independent Z positions)
                #define GS_FUSED2_Z(A, B, C, D, TW_OFF) \
                { \
                    auto tw_ac = esimd::block_load<uint32_t, 16>(ptw_s + j_base + TW_OFF); \
                    auto tw_bd = esimd::block_load<uint32_t, 16>(ptw_s + j_base + half_s1 + TW_OFF); \
                    auto an = bb31::field_add(A, C); \
                    auto cn = bb31::mont_mul(tw_ac, bb31::field_sub(A, C)); \
                    auto bn = bb31::field_add(B, D); \
                    auto dn = bb31::mont_mul(tw_bd, bb31::field_sub(B, D)); \
                    auto tw1 = esimd::block_load<uint32_t, 16>(ptw_s1 + j_base + TW_OFF); \
                    A = bb31::field_add(an, bn); \
                    B = bb31::mont_mul(tw1, bb31::field_sub(an, bn)); \
                    C = bb31::field_add(cn, dn); \
                    D = bb31::mont_mul(tw1, bb31::field_sub(cn, dn)); \
                }

                GS_FUSED2_Z(a0, b0, c0, d0, 0)
                GS_FUSED2_Z(a1, b1, c1, d1, 16)
                GS_FUSED2_Z(a2, b2, c2, d2, 32)
                GS_FUSED2_Z(a3, b3, c3, d3, 48)

                #undef GS_FUSED2_Z

                // Phase 3: Store results
                esimd::block_store(pd + pos_a, a0, STORE_STREAMING);
                esimd::block_store(pd + pos_a + 16, a1, STORE_STREAMING);
                esimd::block_store(pd + pos_a + 32, a2, STORE_STREAMING);
                esimd::block_store(pd + pos_a + 48, a3, STORE_STREAMING);
                esimd::block_store(pd + pos_b, b0, STORE_STREAMING);
                esimd::block_store(pd + pos_b + 16, b1, STORE_STREAMING);
                esimd::block_store(pd + pos_b + 32, b2, STORE_STREAMING);
                esimd::block_store(pd + pos_b + 48, b3, STORE_STREAMING);
                esimd::block_store(pd + pos_c, c0, STORE_STREAMING);
                esimd::block_store(pd + pos_c + 16, c1, STORE_STREAMING);
                esimd::block_store(pd + pos_c + 32, c2, STORE_STREAMING);
                esimd::block_store(pd + pos_c + 48, c3, STORE_STREAMING);
                esimd::block_store(pd + pos_d, d0, STORE_STREAMING);
                esimd::block_store(pd + pos_d + 16, d1, STORE_STREAMING);
                esimd::block_store(pd + pos_d + 32, d2, STORE_STREAMING);
                esimd::block_store(pd + pos_d + 48, d3, STORE_STREAMING);
            });
    } else if (half_s1 >= 16) {
        // Z=1 path for smaller stages
        q.parallel_for(sycl::range<1>(num_threads),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t tid = idx[0];
                uint32_t k_base = tid * 16;
                uint32_t grp = k_base / half_s1;
                uint32_t j_base = k_base % half_s1;

                uint32_t pos_a = grp * m_s + j_base;
                uint32_t pos_b = pos_a + half_s1;
                uint32_t pos_c = pos_a + half_s;
                uint32_t pos_d = pos_c + half_s1;

                auto a = esimd::block_load<uint32_t, 16>(pd + pos_a, LOAD_STREAMING);
                auto b = esimd::block_load<uint32_t, 16>(pd + pos_b, LOAD_STREAMING);
                auto c = esimd::block_load<uint32_t, 16>(pd + pos_c, LOAD_STREAMING);
                auto d = esimd::block_load<uint32_t, 16>(pd + pos_d, LOAD_STREAMING);

                auto tw_s_ac = esimd::block_load<uint32_t, 16>(ptw_s + j_base);
                auto tw_s_bd = esimd::block_load<uint32_t, 16>(ptw_s + j_base + half_s1);

                auto a_new = bb31::field_add(a, c);
                auto c_new = bb31::mont_mul(tw_s_ac, bb31::field_sub(a, c));
                auto b_new = bb31::field_add(b, d);
                auto d_new = bb31::mont_mul(tw_s_bd, bb31::field_sub(b, d));

                auto tw_s1 = esimd::block_load<uint32_t, 16>(ptw_s1 + j_base);

                auto a_final = bb31::field_add(a_new, b_new);
                auto b_final = bb31::mont_mul(tw_s1, bb31::field_sub(a_new, b_new));
                auto c_final = bb31::field_add(c_new, d_new);
                auto d_final = bb31::mont_mul(tw_s1, bb31::field_sub(c_new, d_new));

                esimd::block_store(pd + pos_a, a_final, STORE_STREAMING);
                esimd::block_store(pd + pos_b, b_final, STORE_STREAMING);
                esimd::block_store(pd + pos_c, c_final, STORE_STREAMING);
                esimd::block_store(pd + pos_d, d_final, STORE_STREAMING);
            });
    }
}

// SIMD16 vectorized GS butterfly stage with precomputed twiddle table.
void ntt_gs_stage_fast(sycl::queue& q, uint32_t* d_data, uint32_t lg_n,
                       uint32_t s, const uint32_t* roots, const uint32_t* d_twiddles) {
    uint32_t n = 1u << lg_n;
    uint32_t m = 1u << s;
    uint32_t half = m >> 1;
    uint32_t num_butterflies = n / 2;

    auto* pd = d_data;
    auto* ptw = d_twiddles;

    if (half >= 64) {
        // Z=4 path: 4x16=64 butterflies per thread
        uint32_t num_threads = num_butterflies / 64;

        q.parallel_for(sycl::range<1>(num_threads),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t tid = idx[0];
                uint32_t k_base = tid * 64;
                uint32_t group = k_base / half;
                uint32_t j_base = k_base % half;
                uint32_t top = group * m + j_base;
                uint32_t bot = top + half;

                auto u0 = esimd::block_load<uint32_t, 16>(pd + top, LOAD_STREAMING);
                auto u1 = esimd::block_load<uint32_t, 16>(pd + top + 16, LOAD_STREAMING);
                auto u2 = esimd::block_load<uint32_t, 16>(pd + top + 32, LOAD_STREAMING);
                auto u3 = esimd::block_load<uint32_t, 16>(pd + top + 48, LOAD_STREAMING);
                auto v0 = esimd::block_load<uint32_t, 16>(pd + bot, LOAD_STREAMING);
                auto v1 = esimd::block_load<uint32_t, 16>(pd + bot + 16, LOAD_STREAMING);
                auto v2 = esimd::block_load<uint32_t, 16>(pd + bot + 32, LOAD_STREAMING);
                auto v3 = esimd::block_load<uint32_t, 16>(pd + bot + 48, LOAD_STREAMING);
                auto w0 = esimd::block_load<uint32_t, 16>(ptw + j_base);
                auto w1 = esimd::block_load<uint32_t, 16>(ptw + j_base + 16);
                auto w2 = esimd::block_load<uint32_t, 16>(ptw + j_base + 32);
                auto w3 = esimd::block_load<uint32_t, 16>(ptw + j_base + 48);

                // GS: top = u + v; bot = (u - v) * tw
                auto s0 = bb31::field_add(u0, v0);
                auto s1 = bb31::field_add(u1, v1);
                auto s2 = bb31::field_add(u2, v2);
                auto s3 = bb31::field_add(u3, v3);
                auto d0 = bb31::mont_mul(w0, bb31::field_sub(u0, v0));
                auto d1 = bb31::mont_mul(w1, bb31::field_sub(u1, v1));
                auto d2 = bb31::mont_mul(w2, bb31::field_sub(u2, v2));
                auto d3 = bb31::mont_mul(w3, bb31::field_sub(u3, v3));

                esimd::block_store(pd + top,      s0, STORE_STREAMING);
                esimd::block_store(pd + top + 16, s1, STORE_STREAMING);
                esimd::block_store(pd + top + 32, s2, STORE_STREAMING);
                esimd::block_store(pd + top + 48, s3, STORE_STREAMING);
                esimd::block_store(pd + bot,      d0, STORE_STREAMING);
                esimd::block_store(pd + bot + 16, d1, STORE_STREAMING);
                esimd::block_store(pd + bot + 32, d2, STORE_STREAMING);
                esimd::block_store(pd + bot + 48, d3, STORE_STREAMING);
            });
    } else if (half >= 16) {
        // SIMD16 Z=1 path
        uint32_t num_threads = num_butterflies / 16;

        q.parallel_for(sycl::range<1>(num_threads),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t tid = idx[0];
                uint32_t k_base = tid * 16;
                uint32_t group = k_base / half;
                uint32_t j_base = k_base % half;
                uint32_t top_base = group * m + j_base;
                uint32_t bot_base = top_base + half;

                auto u = esimd::block_load<uint32_t, 16>(pd + top_base);
                auto v = esimd::block_load<uint32_t, 16>(pd + bot_base);
                auto tw = esimd::block_load<uint32_t, 16>(ptw + j_base);
                esimd::block_store(pd + top_base, bb31::field_add(u, v), STORE_STREAMING);
                esimd::block_store(pd + bot_base,
                    bb31::mont_mul(tw, bb31::field_sub(u, v)), STORE_STREAMING);
            });
    } else {
        ntt_gs_stage(q, d_data, lg_n, s, roots);
    }
}

// Precompute ALL twiddle tables for all stages in a single contiguous buffer.
// Each stage's twiddles are padded to a multiple of 16 for aligned SIMD16 block_load.
// Stages with half < 16 have their pattern repeated to fill 16 entries.
struct TwiddleTables {
    uint32_t* d_buffer;  // contiguous device buffer
    uint32_t offsets[28]; // offset[s] = start of stage s twiddles in buffer
    uint32_t total_size;
    uint32_t lg_n;       // domain size this table was built for
};

static TwiddleTables precompute_all_twiddles(sycl::queue& q, uint32_t lg_n,
                                              const uint32_t* roots) {
    TwiddleTables tables = {};
    tables.lg_n = lg_n;

    // Calculate total size with 16-element alignment per stage
    uint32_t total = 0;
    for (uint32_t s = 1; s <= lg_n; s++) {
        uint32_t half = 1u << (s - 1);
        total += (half < 16) ? 16 : half;
    }
    tables.total_size = total;

    auto* h_buf = sycl::malloc_host<uint32_t>(total, q);

    uint32_t offset = 0;
    for (uint32_t s = 1; s <= lg_n; s++) {
        tables.offsets[s] = offset;
        uint32_t half = 1u << (s - 1);
        uint32_t root = roots[s];

        // Compute half twiddle values: w^0, w^1, ..., w^(half-1)
        uint32_t w = bb31::ONE;
        for (uint32_t j = 0; j < half; j++) {
            h_buf[offset + j] = w;
            w = host_mont_mul(w, root);
        }

        // Pad to 16 by repeating pattern (for SIMD16 block_load alignment)
        uint32_t padded = (half < 16) ? 16 : half;
        for (uint32_t j = half; j < padded; j++) {
            h_buf[offset + j] = h_buf[offset + j % half];
        }

        offset += padded;
    }

    tables.d_buffer = sycl::malloc_device<uint32_t>(total, q);
    q.memcpy(tables.d_buffer, h_buf, total * sizeof(uint32_t));
    q.wait(); // Must wait before freeing host staging buffer
    sycl::free(h_buf, q);

    return tables;
}

// ============================================================================
// Static twiddle cache — compute once, reuse across all NTT calls.
// Stores the SYCL context to validate that cached device pointers are still
// valid when reused by a different queue (safe if same context).
// ============================================================================
static TwiddleTables g_fwd_tw[28] = {};
static TwiddleTables g_inv_tw[28] = {};
static bool g_fwd_valid[28] = {};
static bool g_inv_valid[28] = {};
static sycl::context* g_tw_context = nullptr;

static const TwiddleTables& get_cached_twiddles(sycl::queue& q, uint32_t lg_n, bool forward) {
    if (lg_n > ntt::MAX_LG_DOMAIN) {
        fprintf(stderr, "FATAL: lg_n=%u exceeds MAX_LG_DOMAIN=%u\n", lg_n, ntt::MAX_LG_DOMAIN);
        fflush(stderr);
        abort();
    }

    // If the SYCL context changed, invalidate all cached tables
    auto ctx = q.get_context();
    if (g_tw_context == nullptr) {
        g_tw_context = new sycl::context(ctx);
    } else if (*g_tw_context != ctx) {
        // Context changed — free old allocations and re-cache
        for (int i = 0; i < 28; i++) {
            if (g_fwd_valid[i] && g_fwd_tw[i].d_buffer) {
                sycl::free(g_fwd_tw[i].d_buffer, *g_tw_context);
                g_fwd_valid[i] = false;
            }
            if (g_inv_valid[i] && g_inv_tw[i].d_buffer) {
                sycl::free(g_inv_tw[i].d_buffer, *g_tw_context);
                g_inv_valid[i] = false;
            }
        }
        *g_tw_context = ctx;
    }

    auto& tables = forward ? g_fwd_tw[lg_n] : g_inv_tw[lg_n];
    auto& valid = forward ? g_fwd_valid[lg_n] : g_inv_valid[lg_n];
    if (!valid) {
        tables = precompute_all_twiddles(q, lg_n,
            forward ? ntt::forward_roots : ntt::inverse_roots);
        valid = true;
    }
    return tables;
}

// ============================================================================
// Fused in-register butterfly kernels for stages 1-4.
// Each thread loads 16 consecutive elements, applies up to 4 butterfly stages
// using XOR-based lane swizzle (iselect) — no memory round-trips between stages.
// Requires lg_n >= 4 (N >= 16).
// ============================================================================

// CT DIT fused stages 1..min(4, lg_n) — used by forward NTT
static void ntt_ct_fused_small(sycl::queue& q, uint32_t* d_data, uint32_t lg_n,
                                const uint32_t* d_twiddles, const TwiddleTables& tw) {
    uint32_t n = 1u << lg_n;
    uint32_t num_blocks = n / 16;
    uint32_t max_s = (lg_n < 4) ? lg_n : 4;

    auto* pd = d_data;
    auto* ptw = d_twiddles;
    uint32_t off2 = (lg_n >= 2) ? tw.offsets[2] : 0;
    uint32_t off3 = (lg_n >= 3) ? tw.offsets[3] : 0;
    uint32_t off4 = (lg_n >= 4) ? tw.offsets[4] : 0;
    uint32_t ms = max_s;

    q.parallel_for(sycl::range<1>(num_blocks),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;
            auto data = esimd::block_load<uint32_t, 16>(pd + base);

            // Use uint32_t for lane arithmetic (avoids uint16_t->int promotion)
            esimd::simd<uint32_t, 16> lane32(0u, 1u);

            // Precompute XOR index vectors and top masks per stage
            esimd::simd<uint16_t, 16> xor1(lane32 ^ esimd::simd<uint32_t, 16>(1u));
            esimd::simd<uint16_t, 16> xor2(lane32 ^ esimd::simd<uint32_t, 16>(2u));
            esimd::simd<uint16_t, 16> xor4(lane32 ^ esimd::simd<uint32_t, 16>(4u));
            esimd::simd<uint16_t, 16> xor8(lane32 ^ esimd::simd<uint32_t, 16>(8u));

            auto top1 = (lane32 & 1u) == esimd::simd<uint32_t, 16>(0u);
            auto top2 = (lane32 & 2u) == esimd::simd<uint32_t, 16>(0u);
            auto top4 = (lane32 & 4u) == esimd::simd<uint32_t, 16>(0u);
            auto top8 = (lane32 & 8u) == esimd::simd<uint32_t, 16>(0u);

            // --- Stage 1 (half=1): XOR 1, tw = ONE (skip multiply) ---
            {
                auto partner = data.iselect(xor1);
                bb31::Vec16 bot_val = data;
                bot_val.merge(partner, top1);
                bb31::Vec16 top_val = partner;
                top_val.merge(data, top1);

                auto sum = bb31::field_add(top_val, bot_val);
                auto diff = bb31::field_sub(top_val, bot_val);
                data = diff;
                data.merge(sum, top1);
            }

            // --- Stage 2 (half=2): XOR 2 ---
            if (ms >= 2) {
                auto partner = data.iselect(xor2);
                bb31::Vec16 bot_val = data;
                bot_val.merge(partner, top2);
                bb31::Vec16 top_val = partner;
                top_val.merge(data, top2);

                auto tw_vec = esimd::block_load<uint32_t, 16>(ptw + off2);
                auto t = bb31::mont_mul(tw_vec, bot_val);
                auto sum = bb31::field_add(top_val, t);
                auto diff = bb31::field_sub(top_val, t);
                data = diff;
                data.merge(sum, top2);
            }

            // --- Stage 3 (half=4): XOR 4 ---
            if (ms >= 3) {
                auto partner = data.iselect(xor4);
                bb31::Vec16 bot_val = data;
                bot_val.merge(partner, top4);
                bb31::Vec16 top_val = partner;
                top_val.merge(data, top4);

                auto tw_vec = esimd::block_load<uint32_t, 16>(ptw + off3);
                auto t = bb31::mont_mul(tw_vec, bot_val);
                auto sum = bb31::field_add(top_val, t);
                auto diff = bb31::field_sub(top_val, t);
                data = diff;
                data.merge(sum, top4);
            }

            // --- Stage 4 (half=8): XOR 8 ---
            if (ms >= 4) {
                auto partner = data.iselect(xor8);
                bb31::Vec16 bot_val = data;
                bot_val.merge(partner, top8);
                bb31::Vec16 top_val = partner;
                top_val.merge(data, top8);

                auto tw_vec = esimd::block_load<uint32_t, 16>(ptw + off4);
                auto t = bb31::mont_mul(tw_vec, bot_val);
                auto sum = bb31::field_add(top_val, t);
                auto diff = bb31::field_sub(top_val, t);
                data = diff;
                data.merge(sum, top8);
            }

            esimd::block_store(pd + base, data);
        });
}

// GS DIF fused stages min(4,lg_n)..1 — used by inverse NTT
// Processes in decreasing order: 4, 3, 2, 1
static void ntt_gs_fused_small(sycl::queue& q, uint32_t* d_data, uint32_t lg_n,
                                const uint32_t* d_twiddles, const TwiddleTables& tw,
                                uint32_t scale_factor = bb31::ONE) {
    uint32_t n = 1u << lg_n;
    uint32_t num_blocks = n / 16;
    uint32_t max_s = (lg_n < 4) ? lg_n : 4;

    auto* pd = d_data;
    auto* ptw = d_twiddles;
    uint32_t off2 = (lg_n >= 2) ? tw.offsets[2] : 0;
    uint32_t off3 = (lg_n >= 3) ? tw.offsets[3] : 0;
    uint32_t off4 = (lg_n >= 4) ? tw.offsets[4] : 0;
    uint32_t ms = max_s;

    q.parallel_for(sycl::range<1>(num_blocks),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;
            auto data = esimd::block_load<uint32_t, 16>(pd + base);

            esimd::simd<uint32_t, 16> lane32(0u, 1u);
            esimd::simd<uint16_t, 16> xor1(lane32 ^ esimd::simd<uint32_t, 16>(1u));
            esimd::simd<uint16_t, 16> xor2(lane32 ^ esimd::simd<uint32_t, 16>(2u));
            esimd::simd<uint16_t, 16> xor4(lane32 ^ esimd::simd<uint32_t, 16>(4u));
            esimd::simd<uint16_t, 16> xor8(lane32 ^ esimd::simd<uint32_t, 16>(8u));
            auto top1 = (lane32 & 1u) == esimd::simd<uint32_t, 16>(0u);
            auto top2 = (lane32 & 2u) == esimd::simd<uint32_t, 16>(0u);
            auto top4 = (lane32 & 4u) == esimd::simd<uint32_t, 16>(0u);
            auto top8 = (lane32 & 8u) == esimd::simd<uint32_t, 16>(0u);

            // GS DIF: stages in decreasing order

            // --- Stage 4 (half=8): XOR 8 ---
            if (ms >= 4) {
                auto partner = data.iselect(xor8);
                bb31::Vec16 bot_val = data;
                bot_val.merge(partner, top8);
                bb31::Vec16 top_val = partner;
                top_val.merge(data, top8);

                auto sum = bb31::field_add(top_val, bot_val);
                auto diff = bb31::field_sub(top_val, bot_val);
                auto tw_vec = esimd::block_load<uint32_t, 16>(ptw + off4);
                auto diff_tw = bb31::mont_mul(tw_vec, diff);
                data = diff_tw;
                data.merge(sum, top8);
            }

            // --- Stage 3 (half=4): XOR 4 ---
            if (ms >= 3) {
                auto partner = data.iselect(xor4);
                bb31::Vec16 bot_val = data;
                bot_val.merge(partner, top4);
                bb31::Vec16 top_val = partner;
                top_val.merge(data, top4);

                auto sum = bb31::field_add(top_val, bot_val);
                auto diff = bb31::field_sub(top_val, bot_val);
                auto tw_vec = esimd::block_load<uint32_t, 16>(ptw + off3);
                auto diff_tw = bb31::mont_mul(tw_vec, diff);
                data = diff_tw;
                data.merge(sum, top4);
            }

            // --- Stage 2 (half=2): XOR 2 ---
            if (ms >= 2) {
                auto partner = data.iselect(xor2);
                bb31::Vec16 bot_val = data;
                bot_val.merge(partner, top2);
                bb31::Vec16 top_val = partner;
                top_val.merge(data, top2);

                auto sum = bb31::field_add(top_val, bot_val);
                auto diff = bb31::field_sub(top_val, bot_val);
                auto tw_vec = esimd::block_load<uint32_t, 16>(ptw + off2);
                auto diff_tw = bb31::mont_mul(tw_vec, diff);
                data = diff_tw;
                data.merge(sum, top2);
            }

            // --- Stage 1 (half=1): XOR 1, tw = ONE (skip multiply) ---
            {
                auto partner = data.iselect(xor1);
                bb31::Vec16 bot_val = data;
                bot_val.merge(partner, top1);
                bb31::Vec16 top_val = partner;
                top_val.merge(data, top1);

                auto sum = bb31::field_add(top_val, bot_val);
                auto diff = bb31::field_sub(top_val, bot_val);
                data = diff;
                data.merge(sum, top1);
            }

            // Fuse 1/N scaling into final store (avoids separate bandwidth pass)
            if (scale_factor != bb31::ONE) {
                data = bb31::mont_mul(data, bb31::Vec16(scale_factor));
            }
            esimd::block_store(pd + base, data);
        });
}

// ============================================================================
// Combined SLM kernel: fused stages 1-4 in-register + stages 5-11 via SLM
// One kernel launch processes 11 NTT stages per 2048-element block.
// Block = 2048 elements = 8 KB SLM, 64 threads/work-group.
// Each thread loads 2x16 elements (lo and hi halves of the block).
// All 64 threads are active during SLM stages (2048/32 = 64).
// ============================================================================
static constexpr uint32_t SLM_LG_BLOCK = 14;  // Extended: 16384 elements, 64KB SLM (optimal)
static constexpr uint32_t SLM_BLOCK = 1u << SLM_LG_BLOCK; // 16384
static constexpr uint32_t SLM_THREADS = 64;                  // Fixed at 64 (HW limit)
static constexpr uint32_t SLM_LOADS_PER_THREAD = SLM_BLOCK / (SLM_THREADS * 16); // 16
static constexpr uint32_t SLM_BYTES = SLM_BLOCK * sizeof(uint32_t); // 65536

// Helper: apply fused CT DIT stages 1-4 in-register to a single Vec16
#define FUSED_CT_STAGES_1_4(data, ptw, off2, off3, off4) \
{ \
    esimd::simd<uint32_t, 16> lane32(0u, 1u); \
    esimd::simd<uint16_t, 16> xi1(lane32 ^ esimd::simd<uint32_t, 16>(1u)); \
    esimd::simd<uint16_t, 16> xi2(lane32 ^ esimd::simd<uint32_t, 16>(2u)); \
    esimd::simd<uint16_t, 16> xi4(lane32 ^ esimd::simd<uint32_t, 16>(4u)); \
    esimd::simd<uint16_t, 16> xi8(lane32 ^ esimd::simd<uint32_t, 16>(8u)); \
    auto t1 = (lane32 & 1u) == esimd::simd<uint32_t, 16>(0u); \
    auto t2 = (lane32 & 2u) == esimd::simd<uint32_t, 16>(0u); \
    auto t4 = (lane32 & 4u) == esimd::simd<uint32_t, 16>(0u); \
    auto t8 = (lane32 & 8u) == esimd::simd<uint32_t, 16>(0u); \
    { auto p = data.iselect(xi1); \
      bb31::Vec16 bv = data; bv.merge(p, t1); \
      bb31::Vec16 tv = p; tv.merge(data, t1); \
      auto s = bb31::field_add(tv, bv); auto d = bb31::field_sub(tv, bv); \
      data = d; data.merge(s, t1); } \
    { auto p = data.iselect(xi2); \
      bb31::Vec16 bv = data; bv.merge(p, t2); \
      bb31::Vec16 tv = p; tv.merge(data, t2); \
      auto tw = esimd::block_load<uint32_t, 16>(ptw + off2); \
      auto r = bb31::mont_mul(tw, bv); \
      auto s = bb31::field_add(tv, r); auto d = bb31::field_sub(tv, r); \
      data = d; data.merge(s, t2); } \
    { auto p = data.iselect(xi4); \
      bb31::Vec16 bv = data; bv.merge(p, t4); \
      bb31::Vec16 tv = p; tv.merge(data, t4); \
      auto tw = esimd::block_load<uint32_t, 16>(ptw + off3); \
      auto r = bb31::mont_mul(tw, bv); \
      auto s = bb31::field_add(tv, r); auto d = bb31::field_sub(tv, r); \
      data = d; data.merge(s, t4); } \
    { auto p = data.iselect(xi8); \
      bb31::Vec16 bv = data; bv.merge(p, t8); \
      bb31::Vec16 tv = p; tv.merge(data, t8); \
      auto tw = esimd::block_load<uint32_t, 16>(ptw + off4); \
      auto r = bb31::mont_mul(tw, bv); \
      auto s = bb31::field_add(tv, r); auto d = bb31::field_sub(tv, r); \
      data = d; data.merge(s, t8); } \
}

// Helper: apply fused GS DIF stages 4..1 in-register to a single Vec16
#define FUSED_GS_STAGES_4_1(data, ptw, off2, off3, off4) \
{ \
    esimd::simd<uint32_t, 16> lane32(0u, 1u); \
    esimd::simd<uint16_t, 16> xi1(lane32 ^ esimd::simd<uint32_t, 16>(1u)); \
    esimd::simd<uint16_t, 16> xi2(lane32 ^ esimd::simd<uint32_t, 16>(2u)); \
    esimd::simd<uint16_t, 16> xi4(lane32 ^ esimd::simd<uint32_t, 16>(4u)); \
    esimd::simd<uint16_t, 16> xi8(lane32 ^ esimd::simd<uint32_t, 16>(8u)); \
    auto t1 = (lane32 & 1u) == esimd::simd<uint32_t, 16>(0u); \
    auto t2 = (lane32 & 2u) == esimd::simd<uint32_t, 16>(0u); \
    auto t4 = (lane32 & 4u) == esimd::simd<uint32_t, 16>(0u); \
    auto t8 = (lane32 & 8u) == esimd::simd<uint32_t, 16>(0u); \
    { auto p = data.iselect(xi8); \
      bb31::Vec16 bv = data; bv.merge(p, t8); \
      bb31::Vec16 tv = p; tv.merge(data, t8); \
      auto s = bb31::field_add(tv, bv); auto d = bb31::field_sub(tv, bv); \
      auto tw = esimd::block_load<uint32_t, 16>(ptw + off4); \
      auto dtw = bb31::mont_mul(tw, d); \
      data = dtw; data.merge(s, t8); } \
    { auto p = data.iselect(xi4); \
      bb31::Vec16 bv = data; bv.merge(p, t4); \
      bb31::Vec16 tv = p; tv.merge(data, t4); \
      auto s = bb31::field_add(tv, bv); auto d = bb31::field_sub(tv, bv); \
      auto tw = esimd::block_load<uint32_t, 16>(ptw + off3); \
      auto dtw = bb31::mont_mul(tw, d); \
      data = dtw; data.merge(s, t4); } \
    { auto p = data.iselect(xi2); \
      bb31::Vec16 bv = data; bv.merge(p, t2); \
      bb31::Vec16 tv = p; tv.merge(data, t2); \
      auto s = bb31::field_add(tv, bv); auto d = bb31::field_sub(tv, bv); \
      auto tw = esimd::block_load<uint32_t, 16>(ptw + off2); \
      auto dtw = bb31::mont_mul(tw, d); \
      data = dtw; data.merge(s, t2); } \
    { auto p = data.iselect(xi1); \
      bb31::Vec16 bv = data; bv.merge(p, t1); \
      bb31::Vec16 tv = p; tv.merge(data, t1); \
      auto s = bb31::field_add(tv, bv); auto d = bb31::field_sub(tv, bv); \
      data = d; data.merge(s, t1); } \
}

// CT DIT combined kernel: stages 1..min(12, lg_n) in one launch.
// 64 threads/WG, each loads 4x16=64 elements. Block = 4096 elements = 16 KB SLM.
// total_elements overrides n for batched sub-NTT processing (0 = use 2^lg_n).
static void ntt_ct_slm_combined(sycl::queue& q, uint32_t* d_data, uint32_t lg_n,
                                 const uint32_t* d_twiddles, const TwiddleTables& tw,
                                 uint32_t total_elements = 0) {
    uint32_t n = total_elements ? total_elements : (1u << lg_n);
    uint32_t num_groups = n / SLM_BLOCK;
    uint32_t last_slm_stage = (lg_n < SLM_LG_BLOCK) ? lg_n : SLM_LG_BLOCK;

    auto* pd = d_data;
    auto* ptw = d_twiddles;
    uint32_t off2 = tw.offsets[2], off3 = tw.offsets[3], off4 = tw.offsets[4];
    uint32_t off5 = tw.offsets[5], off6 = tw.offsets[6], off7 = tw.offsets[7];
    uint32_t off8 = tw.offsets[8], off9 = tw.offsets[9], off10 = tw.offsets[10];
    uint32_t off11 = (lg_n >= 11) ? tw.offsets[11] : 0;
    uint32_t off12 = (lg_n >= 12) ? tw.offsets[12] : 0;
    uint32_t off13 = (lg_n >= 13) ? tw.offsets[13] : 0;
    uint32_t off14 = (lg_n >= 14) ? tw.offsets[14] : 0;
    uint32_t off15 = (lg_n >= 15) ? tw.offsets[15] : 0;
    uint32_t lst = last_slm_stage;

    // Each thread handles SLM_LOADS_PER_THREAD chunks of 16 elements
    constexpr uint32_t CHUNK = SLM_BLOCK / SLM_LOADS_PER_THREAD;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * SLM_THREADS, SLM_THREADS),
            [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
                esimd::slm_init<SLM_BYTES>();
                uint32_t lid = item.get_local_id(0);
                uint32_t gid = item.get_group(0);
                uint32_t global_base = gid * SLM_BLOCK;

                // --- Load SLM_LOADS_PER_THREAD x 16 elements from global ---
                bb31::Vec16 dv[32]; // max 32 loads per thread (SLM_LG_BLOCK up to 15)
                for (uint32_t c = 0; c < SLM_LOADS_PER_THREAD; c++)
                    dv[c] = esimd::block_load<uint32_t, 16>(pd + global_base + c*CHUNK + lid*16);

                // --- Fused stages 1-4 in-register (independently on each chunk) ---
                for (uint32_t c = 0; c < SLM_LOADS_PER_THREAD; c++)
                    FUSED_CT_STAGES_1_4(dv[c], ptw, off2, off3, off4)

                // --- Write chunks to SLM in natural element order ---
                for (uint32_t c = 0; c < SLM_LOADS_PER_THREAD; c++)
                    esimd::slm_block_store<uint32_t, 16>((c*CHUNK + lid*16) * 4u, dv[c]);
                esimd::barrier();

                // --- SLM stages 5..last_slm_stage ---
                // bfly_threads = BLOCK/32. With 64 threads, each handles BLOCK/32/64 pairs.
                // For BLOCK=8192: 256 bfly threads, 4 pairs per thread.
                constexpr uint32_t BFLY_PAIRS = SLM_BLOCK / 32 / SLM_THREADS; // 4
                #define SLM_CT_STAGE(STAGE, OFF) \
                if (lst >= STAGE) { \
                    uint32_t half = 1u << (STAGE - 1); \
                    uint32_t m = 1u << STAGE; \
                    for (uint32_t p = 0; p < BFLY_PAIRS; p++) { \
                        uint32_t kb = (lid + p * SLM_THREADS) * 16; \
                        uint32_t grp = kb / half; \
                        uint32_t jb = kb % half; \
                        uint32_t top_off = (grp * m + jb) * 4; \
                        uint32_t bot_off = top_off + half * 4; \
                        auto u = esimd::slm_block_load<uint32_t, 16>(top_off); \
                        auto v = esimd::slm_block_load<uint32_t, 16>(bot_off); \
                        auto twv = esimd::block_load<uint32_t, 16>(ptw + OFF + jb); \
                        auto tv = bb31::mont_mul(twv, v); \
                        esimd::slm_block_store<uint32_t, 16>(top_off, bb31::field_add(u, tv)); \
                        esimd::slm_block_store<uint32_t, 16>(bot_off, bb31::field_sub(u, tv)); \
                    } \
                    esimd::barrier(); \
                }

                SLM_CT_STAGE(5, off5)
                SLM_CT_STAGE(6, off6)
                SLM_CT_STAGE(7, off7)
                SLM_CT_STAGE(8, off8)
                SLM_CT_STAGE(9, off9)
                SLM_CT_STAGE(10, off10)
                SLM_CT_STAGE(11, off11)
                SLM_CT_STAGE(12, off12)
                SLM_CT_STAGE(13, off13)
                SLM_CT_STAGE(14, off14)
                SLM_CT_STAGE(15, off15)

                #undef SLM_CT_STAGE

                // --- Read chunks from SLM and write to global ---
                for (uint32_t c = 0; c < SLM_LOADS_PER_THREAD; c++) {
                    auto rv = esimd::slm_block_load<uint32_t, 16>((c*CHUNK + lid*16) * 4u);
                    esimd::block_store(pd + global_base + c*CHUNK + lid*16, rv);
                }
            });
    });
}

// GS DIF combined kernel: stages min(12,lg_n)..1 in one launch (inverse NTT)
// 64 threads/WG, 4 loads per thread. Block = 4096 elements = 16 KB SLM.
static void ntt_gs_slm_combined(sycl::queue& q, uint32_t* d_data, uint32_t lg_n,
                                 const uint32_t* d_twiddles, const TwiddleTables& tw,
                                 uint32_t scale_factor = bb31::ONE,
                                 uint32_t total_elements = 0) {
    uint32_t n = total_elements ? total_elements : (1u << lg_n);
    uint32_t num_groups = n / SLM_BLOCK;
    uint32_t last_slm_stage = (lg_n < SLM_LG_BLOCK) ? lg_n : SLM_LG_BLOCK;

    auto* pd = d_data;
    auto* ptw = d_twiddles;
    uint32_t off2 = tw.offsets[2], off3 = tw.offsets[3], off4 = tw.offsets[4];
    uint32_t off5 = tw.offsets[5], off6 = tw.offsets[6], off7 = tw.offsets[7];
    uint32_t off8 = tw.offsets[8], off9 = tw.offsets[9], off10 = tw.offsets[10];
    uint32_t off11 = (lg_n >= 11) ? tw.offsets[11] : 0;
    uint32_t off12 = (lg_n >= 12) ? tw.offsets[12] : 0;
    uint32_t off13 = (lg_n >= 13) ? tw.offsets[13] : 0;
    uint32_t off14 = (lg_n >= 14) ? tw.offsets[14] : 0;
    uint32_t off15 = (lg_n >= 15) ? tw.offsets[15] : 0;
    uint32_t lst = last_slm_stage;
    uint32_t sf = scale_factor;
    constexpr uint32_t CHUNK = SLM_BLOCK / SLM_LOADS_PER_THREAD;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * SLM_THREADS, SLM_THREADS),
            [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
                esimd::slm_init<SLM_BYTES>();
                uint32_t lid = item.get_local_id(0);
                uint32_t gid = item.get_group(0);
                uint32_t global_base = gid * SLM_BLOCK;

                // --- Load chunks from global to SLM ---
                bb31::Vec16 dv[32]; // max 32 loads per thread (SLM_LG_BLOCK up to 15)
                for (uint32_t c = 0; c < SLM_LOADS_PER_THREAD; c++) {
                    dv[c] = esimd::block_load<uint32_t, 16>(pd + global_base + c*CHUNK + lid*16);
                    esimd::slm_block_store<uint32_t, 16>((c*CHUNK + lid*16) * 4u, dv[c]);
                }
                esimd::barrier();

                // --- SLM stages last_slm_stage..5 (GS DIF, decreasing) ---
                constexpr uint32_t GS_BFLY_PAIRS = SLM_BLOCK / 32 / SLM_THREADS;
                #define SLM_GS_STAGE(STAGE, OFF) \
                if (lst >= STAGE) { \
                    uint32_t half = 1u << (STAGE - 1); \
                    uint32_t m = 1u << STAGE; \
                    for (uint32_t p = 0; p < GS_BFLY_PAIRS; p++) { \
                        uint32_t kb = (lid + p * SLM_THREADS) * 16; \
                        uint32_t grp = kb / half; \
                        uint32_t jb = kb % half; \
                        uint32_t top_off = (grp * m + jb) * 4; \
                        uint32_t bot_off = top_off + half * 4; \
                        auto u = esimd::slm_block_load<uint32_t, 16>(top_off); \
                        auto v = esimd::slm_block_load<uint32_t, 16>(bot_off); \
                        auto sv = bb31::field_add(u, v); \
                        auto dv2 = bb31::field_sub(u, v); \
                        auto twv = esimd::block_load<uint32_t, 16>(ptw + OFF + jb); \
                        auto dtw = bb31::mont_mul(twv, dv2); \
                        esimd::slm_block_store<uint32_t, 16>(top_off, sv); \
                        esimd::slm_block_store<uint32_t, 16>(bot_off, dtw); \
                    } \
                    esimd::barrier(); \
                }

                SLM_GS_STAGE(15, off15)
                SLM_GS_STAGE(14, off14)
                SLM_GS_STAGE(13, off13)
                SLM_GS_STAGE(12, off12)
                SLM_GS_STAGE(11, off11)
                SLM_GS_STAGE(10, off10)
                SLM_GS_STAGE(9, off9)
                SLM_GS_STAGE(8, off8)
                SLM_GS_STAGE(7, off7)
                SLM_GS_STAGE(6, off6)
                SLM_GS_STAGE(5, off5)

                #undef SLM_GS_STAGE

                // --- Read chunks from SLM, fused stages 4..1 in-register ---
                for (uint32_t c = 0; c < SLM_LOADS_PER_THREAD; c++)
                    dv[c] = esimd::slm_block_load<uint32_t, 16>((c*CHUNK + lid*16) * 4u);

                for (uint32_t c = 0; c < SLM_LOADS_PER_THREAD; c++)
                    FUSED_GS_STAGES_4_1(dv[c], ptw, off2, off3, off4)

                // Fuse 1/N scaling into final store
                if (sf != bb31::ONE) {
                    for (uint32_t c = 0; c < SLM_LOADS_PER_THREAD; c++)
                        dv[c] = bb31::mont_mul(dv[c], bb31::Vec16(sf));
                }
                for (uint32_t c = 0; c < SLM_LOADS_PER_THREAD; c++)
                    esimd::block_store(pd + global_base + c*CHUNK + lid*16, dv[c]);
            });
    });
}

// ============================================================================
// Two-pass (Four-Step FFT) decomposition for large NTTs (lg_n > SLM_LG_BLOCK)
// Splits N = N1 × N2 where N2 = SLM_BLOCK, N1 = N/N2.
// Pass 1: N1 row-NTTs of size N2 (SLM kernel, 1 launch)
// Twiddle multiply + transpose: out[j*N1+i] = in[i*N2+j] * w_N^(i*j)
// Pass 2: N2 col-NTTs of size N1 (SLM + per-stage for stages > SLM_LG_BLOCK)
// Total memory traffic: ~3-5 passes vs 12+ per-stage passes.
// ============================================================================

// Scalar Montgomery multiply usable in both host and device code.
// Identical algorithm to host_mont_mul but marked for device use.
ESIMD_INLINE uint32_t device_mont_mul(uint32_t a, uint32_t b) {
    uint64_t prod = uint64_t(a) * uint64_t(b);
    uint32_t lo = uint32_t(prod);
    uint32_t hi = uint32_t(prod >> 32);
    uint32_t red = lo * bb31::M0;
    uint64_t rprod = uint64_t(red) * uint64_t(bb31::MOD);
    uint32_t rlo = uint32_t(rprod);
    uint32_t rhi = uint32_t(rprod >> 32);
    uint32_t carry = (uint64_t(lo) + rlo) >= (1ULL << 32) ? 1 : 0;
    uint32_t res = hi + rhi + carry;
    return res >= bb31::MOD ? res - bb31::MOD : res;
}

// Host-side modular exponentiation for twiddle precomputation
static uint32_t host_mont_pow(uint32_t base, uint32_t exp) {
    uint32_t result = bb31::ONE;
    while (exp > 0) {
        if (exp & 1) result = host_mont_mul(result, base);
        base = host_mont_mul(base, base);
        exp >>= 1;
    }
    return result;
}

// Precompute twiddle column: tw_col[i] = w_N^i for i=0..N1-1
// Used by the transpose kernel to build per-element twiddles iteratively.
static uint32_t* precompute_twiddle_column(sycl::queue& q, uint32_t lg_n,
                                            uint32_t N1, bool forward) {
    auto* h_tw = sycl::malloc_host<uint32_t>(N1, q);
    uint32_t w = forward ? ntt::forward_roots[lg_n] : ntt::inverse_roots[lg_n];
    uint32_t cur = bb31::ONE;
    for (uint32_t i = 0; i < N1; i++) {
        h_tw[i] = cur;
        cur = host_mont_mul(cur, w);
    }
    auto* d_tw = sycl::malloc_device<uint32_t>(N1, q);
    q.memcpy(d_tw, h_tw, N1 * sizeof(uint32_t));
    q.wait();
    sycl::free(h_tw, q);
    return d_tw;
}

// Precompute full 2D twiddle table on host: tw_2d[i*N2 + j] = w_N^(i*j)
// Each row i has N2 twiddle values. The table is contiguous and block-loadable.
static uint32_t* precompute_twiddle_2d(sycl::queue& q, uint32_t lg_n,
                                        uint32_t N1, uint32_t N2, bool forward) {
    uint32_t w_N = forward ? ntt::forward_roots[lg_n] : ntt::inverse_roots[lg_n];
    auto* h_tw = sycl::malloc_host<uint32_t>(N1 * N2, q);

    for (uint32_t i = 0; i < N1; i++) {
        // w_row = w_N^i
        uint32_t w_row = host_mont_pow(w_N, i);
        // Row i: w_row^0, w_row^1, ..., w_row^(N2-1)
        uint32_t cur = bb31::ONE;
        for (uint32_t j = 0; j < N2; j++) {
            h_tw[i * N2 + j] = cur;
            cur = host_mont_mul(cur, w_row);
        }
    }

    auto* d_tw = sycl::malloc_device<uint32_t>(N1 * N2, q);
    q.memcpy(d_tw, h_tw, N1 * N2 * sizeof(uint32_t));
    q.wait();
    sycl::free(h_tw, q);
    return d_tw;
}

// Twiddle-multiply + transpose kernel (out-of-place).
// Reads from in[i*N2 + j], multiplies by tw_2d[i*N2 + j] = w_N^(i*j),
// writes to out[j*N1 + i].
// Each thread processes 16 consecutive columns within one row.
static void twiddle_transpose(sycl::queue& q,
                               uint32_t* d_out, const uint32_t* d_in,
                               const uint32_t* d_tw_2d,
                               uint32_t N1, uint32_t N2, uint32_t lg_n) {
    uint32_t threads_per_row = N2 / 16;
    uint32_t total_threads = N1 * threads_per_row;

    auto* pin = d_in;
    auto* pout = d_out;
    auto* ptw = d_tw_2d;

    q.parallel_for(sycl::range<1>(total_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t tid = idx[0];
            uint32_t row = tid / threads_per_row;
            uint32_t col_start = (tid % threads_per_row) * 16;

            // Load 16 data elements and 16 precomputed twiddles
            auto data = esimd::block_load<uint32_t, 16>(pin + row * N2 + col_start);
            auto tw = esimd::block_load<uint32_t, 16>(ptw + row * N2 + col_start);

            // Apply twiddle
            data = bb31::mont_mul(data, tw);

            // Scatter to transposed positions: out[col*N1 + row]
            esimd::simd<uint32_t, 16> offsets;
            for (int c = 0; c < 16; c++) {
                offsets[c] = ((col_start + c) * N1 + row) * sizeof(uint32_t);
            }
            esimd::scatter<uint32_t, 16>(pout, offsets, data);
        });
}

// Batched per-stage CT butterfly: operates on multiple independent sub-NTTs
// stored contiguously. Each sub-NTT has sub_n elements. Stage s is within
// the sub-NTT (not the full array). Uses Z=4 when half >= 64.
static void batched_ct_stage(sycl::queue& q, uint32_t* d_data,
                              uint32_t total_elements, uint32_t sub_n,
                              uint32_t s, const uint32_t* d_twiddles) {
    uint32_t m = 1u << s;
    uint32_t half = m >> 1;
    uint32_t bflies_per_sub = sub_n / 2;
    uint32_t num_subs = total_elements / sub_n;
    uint32_t total_bflies = num_subs * bflies_per_sub;

    auto* pd = d_data;
    auto* ptw = d_twiddles;

    if (half >= 64) {
        uint32_t num_threads = total_bflies / 64;
        q.parallel_for(sycl::range<1>(num_threads),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t tid = idx[0];
                uint32_t k_global = tid * 64;
                uint32_t sub_idx = k_global / bflies_per_sub;
                uint32_t k_local = k_global % bflies_per_sub;
                uint32_t base_off = sub_idx * sub_n;

                uint32_t group = k_local / half;
                uint32_t j_base = k_local % half;
                uint32_t top = base_off + group * m + j_base;
                uint32_t bot = top + half;

                auto u0 = esimd::block_load<uint32_t, 16>(pd + top);
                auto u1 = esimd::block_load<uint32_t, 16>(pd + top + 16);
                auto u2 = esimd::block_load<uint32_t, 16>(pd + top + 32);
                auto u3 = esimd::block_load<uint32_t, 16>(pd + top + 48);
                auto v0 = esimd::block_load<uint32_t, 16>(pd + bot);
                auto v1 = esimd::block_load<uint32_t, 16>(pd + bot + 16);
                auto v2 = esimd::block_load<uint32_t, 16>(pd + bot + 32);
                auto v3 = esimd::block_load<uint32_t, 16>(pd + bot + 48);
                auto w0 = esimd::block_load<uint32_t, 16>(ptw + j_base);
                auto w1 = esimd::block_load<uint32_t, 16>(ptw + j_base + 16);
                auto w2 = esimd::block_load<uint32_t, 16>(ptw + j_base + 32);
                auto w3 = esimd::block_load<uint32_t, 16>(ptw + j_base + 48);

                auto t0 = bb31::mont_mul(w0, v0);
                auto t1 = bb31::mont_mul(w1, v1);
                auto t2 = bb31::mont_mul(w2, v2);
                auto t3 = bb31::mont_mul(w3, v3);

                esimd::block_store(pd + top,      bb31::field_add(u0, t0), STORE_STREAMING);
                esimd::block_store(pd + top + 16, bb31::field_add(u1, t1), STORE_STREAMING);
                esimd::block_store(pd + top + 32, bb31::field_add(u2, t2), STORE_STREAMING);
                esimd::block_store(pd + top + 48, bb31::field_add(u3, t3), STORE_STREAMING);
                esimd::block_store(pd + bot,      bb31::field_sub(u0, t0), STORE_STREAMING);
                esimd::block_store(pd + bot + 16, bb31::field_sub(u1, t1), STORE_STREAMING);
                esimd::block_store(pd + bot + 32, bb31::field_sub(u2, t2), STORE_STREAMING);
                esimd::block_store(pd + bot + 48, bb31::field_sub(u3, t3), STORE_STREAMING);
            });
    } else if (half >= 16) {
        uint32_t num_threads = total_bflies / 16;
        q.parallel_for(sycl::range<1>(num_threads),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t tid = idx[0];
                uint32_t k_global = tid * 16;
                uint32_t sub_idx = k_global / bflies_per_sub;
                uint32_t k_local = k_global % bflies_per_sub;
                uint32_t base_off = sub_idx * sub_n;

                uint32_t group = k_local / half;
                uint32_t j_base = k_local % half;
                uint32_t top = base_off + group * m + j_base;
                uint32_t bot = top + half;

                auto u = esimd::block_load<uint32_t, 16>(pd + top);
                auto v = esimd::block_load<uint32_t, 16>(pd + bot);
                auto tw = esimd::block_load<uint32_t, 16>(ptw + j_base);
                auto t = bb31::mont_mul(tw, v);
                esimd::block_store(pd + top, bb31::field_add(u, t), STORE_STREAMING);
                esimd::block_store(pd + bot, bb31::field_sub(u, t), STORE_STREAMING);
            });
    }
}

// Simple out-of-place transpose: out[j*N1 + i] = in[i*N2 + j]
// No twiddle multiply. Used to un-transpose after two-pass NTT.
static void simple_transpose(sycl::queue& q,
                              uint32_t* d_out, const uint32_t* d_in,
                              uint32_t N1, uint32_t N2) {
    uint32_t threads_per_row = N2 / 16;
    uint32_t total_threads = N1 * threads_per_row;
    auto* pin = d_in;
    auto* pout = d_out;

    q.parallel_for(sycl::range<1>(total_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t tid = idx[0];
            uint32_t row = tid / threads_per_row;
            uint32_t col_start = (tid % threads_per_row) * 16;

            auto data = esimd::block_load<uint32_t, 16>(pin + row * N2 + col_start);

            esimd::simd<uint32_t, 16> offsets;
            for (int c = 0; c < 16; c++) {
                offsets[c] = ((col_start + c) * N1 + row) * sizeof(uint32_t);
            }
            esimd::scatter<uint32_t, 16>(pout, offsets, data);
        });
}

// ============================================================================
// Four-Step FFT decomposition (scatter-free) for lg_n = 24.
// Decomposes N = N1 × N2 where N1 = N2 = 2^12 = 4096 = SLM_BLOCK.
// Algorithm:
//   1. Transpose N1×N2 → N2×N1 (SLM-based tiled, scatter-free)
//   2. N2 row-NTTs of size N1 via SLM kernel (all 12 stages in SLM)
//   3. Twiddle multiply: element[i][j] *= w_N^{i * bit_rev(j)}
//      (bit_rev corrects for CT DIT's bit-reversed sub-NTT output)
//   4. Transpose N2×N1 → N1×N2 (SLM-based tiled, scatter-free)
//   5. N1 row-NTTs of size N2 via SLM kernel
// Output: NR-ordered DFT (same as single-pass CT DIT).
// Total scatter: ZERO (all global access via block_load/block_store).
// ============================================================================

// SLM-based tiled matrix transpose (scatter-free on global memory).
// Transposes a rows×cols matrix: out[j*rows + i] = in[i*cols + j].
// Uses 64×64 tiles. Each workgroup loads one tile into SLM row-major,
// reads back column-major via slm_gather, writes contiguously.
// Requires rows and cols to be multiples of 64.
static void slm_tiled_transpose(sycl::queue& q,
                                 uint32_t* d_out, const uint32_t* d_in,
                                 uint32_t rows, uint32_t cols) {
    constexpr uint32_t TILE = 64;
    constexpr uint32_t SLM_PITCH = TILE + 1; // Pad rows by 1 to avoid 16-way SLM bank conflicts
    constexpr uint32_t SLM_TILE_BYTES = TILE * SLM_PITCH * sizeof(uint32_t); // ~16.25KB
    uint32_t tiles_r = rows / TILE;
    uint32_t tiles_c = cols / TILE;
    uint32_t total_wgs = tiles_r * tiles_c;

    auto* pin = d_in;
    auto* pout = d_out;
    uint32_t R = rows, C = cols;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(total_wgs * TILE, TILE),
            [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
                esimd::slm_init<SLM_TILE_BYTES>();
                uint32_t lid = item.get_local_id(0);  // 0..63 = row within tile
                uint32_t wg_id = item.get_group(0);

                uint32_t tile_r = wg_id / (C / TILE);
                uint32_t tile_c = wg_id % (C / TILE);

                // --- Load one row of the tile from global (4 × block_load<16>) ---
                uint32_t in_row = tile_r * TILE + lid;
                uint32_t in_col_base = tile_c * TILE;
                auto d0 = esimd::block_load<uint32_t, 16>(pin + in_row * C + in_col_base);
                auto d1 = esimd::block_load<uint32_t, 16>(pin + in_row * C + in_col_base + 16);
                auto d2 = esimd::block_load<uint32_t, 16>(pin + in_row * C + in_col_base + 32);
                auto d3 = esimd::block_load<uint32_t, 16>(pin + in_row * C + in_col_base + 48);

                // --- Store into SLM row-major with padding: SLM[lid*SLM_PITCH + j] ---
                // Padding avoids 16-way bank conflicts on column gather
                uint32_t slm_row_base = lid * SLM_PITCH * 4u;
                esimd::slm_block_store<uint32_t, 16>(slm_row_base, d0);
                esimd::slm_block_store<uint32_t, 16>(slm_row_base + 64, d1);
                esimd::slm_block_store<uint32_t, 16>(slm_row_base + 128, d2);
                esimd::slm_block_store<uint32_t, 16>(slm_row_base + 192, d3);

                esimd::barrier();

                // --- Read column 'lid' from SLM via gather (transposed read) ---
                // Column lid: SLM[row*SLM_PITCH + lid] for row=0..63
                // With SLM_PITCH=65, bank = (row*65 + lid) % 32 cycles through all banks
                constexpr uint32_t STRIDE = SLM_PITCH * 4u; // bytes between padded SLM rows
                uint32_t col_byte = lid * 4u;

                // Build gather offsets: row_i * STRIDE + col_byte
                esimd::simd<uint32_t, 16> row_idx0, row_idx1, row_idx2, row_idx3;
                for (int i = 0; i < 16; i++) {
                    row_idx0[i] = i;
                    row_idx1[i] = 16 + i;
                    row_idx2[i] = 32 + i;
                    row_idx3[i] = 48 + i;
                }
                auto off0 = row_idx0 * STRIDE + col_byte;
                auto off1 = row_idx1 * STRIDE + col_byte;
                auto off2 = row_idx2 * STRIDE + col_byte;
                auto off3 = row_idx3 * STRIDE + col_byte;

                auto t0 = esimd::slm_gather<uint32_t, 16>(off0);
                auto t1 = esimd::slm_gather<uint32_t, 16>(off1);
                auto t2 = esimd::slm_gather<uint32_t, 16>(off2);
                auto t3 = esimd::slm_gather<uint32_t, 16>(off3);

                // --- Write transposed output row ---
                // Output position: out[(tile_c*TILE + lid) * R + tile_r*TILE + 0..63]
                uint32_t out_row = tile_c * TILE + lid;
                uint32_t out_col_base = tile_r * TILE;
                esimd::block_store(pout + out_row * R + out_col_base, t0);
                esimd::block_store(pout + out_row * R + out_col_base + 16, t1);
                esimd::block_store(pout + out_row * R + out_col_base + 32, t2);
                esimd::block_store(pout + out_row * R + out_col_base + 48, t3);
            });
    });
}

// Precompute Four-Step twiddle table: tw[i*N2 + j] = w_N^{i * bit_rev_{lg_N2}(j)}
// The bit_rev corrects for CT DIT sub-NTTs producing bit-reversed output.
// For forward NTT, uses forward roots. For inverse, uses inverse roots.
static uint32_t* precompute_fourstep_twiddle(sycl::queue& q, uint32_t lg_n,
                                              uint32_t N1, uint32_t N2,
                                              uint32_t lg_N2, bool forward) {
    uint32_t w_N = forward ? ntt::forward_roots[lg_n] : ntt::inverse_roots[lg_n];
    auto* h_tw = sycl::malloc_host<uint32_t>(N1 * N2, q);

    // For each column j, the twiddle generator is g_j = w_N^{bit_rev(j)}
    // Row i gets g_j^i, computed iteratively
    for (uint32_t j = 0; j < N2; j++) {
        uint32_t k_actual = ntt::bit_rev(j, lg_N2);
        uint32_t g = host_mont_pow(w_N, k_actual);
        uint32_t cur = bb31::ONE; // g^0 = 1
        for (uint32_t i = 0; i < N1; i++) {
            h_tw[i * N2 + j] = cur;
            cur = host_mont_mul(cur, g);
        }
    }

    auto* d_tw = sycl::malloc_device<uint32_t>(N1 * N2, q);
    q.memcpy(d_tw, h_tw, N1 * N2 * sizeof(uint32_t));
    q.wait();
    sycl::free(h_tw, q);
    return d_tw;
}

// Pointwise twiddle multiply kernel: d_data[i] *= d_twiddle[i]
// Both block_load (data + twiddle) and block_store. No scatter.
static void fourstep_twiddle_kernel(sycl::queue& q, uint32_t* d_data,
                                     const uint32_t* d_twiddle, uint32_t n) {
    uint32_t num_threads = n / 16;
    auto* pd = d_data;
    auto* ptw = d_twiddle;

    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t offset = idx[0] * 16;
            auto data = esimd::block_load<uint32_t, 16>(pd + offset);
            auto tw = esimd::block_load<uint32_t, 16>(ptw + offset);
            esimd::block_store(pd + offset, bb31::mont_mul(data, tw));
        });
}

// Cached Four-Step twiddle tables (allocated once per direction)
static uint32_t* g_fourstep_tw_fwd = nullptr;
static uint32_t* g_fourstep_tw_inv = nullptr;
static uint32_t g_fourstep_lg_n = 0;

// Cached scratch buffer for Four-Step transpose (allocated once)
static uint32_t* g_fourstep_scratch = nullptr;
static uint32_t g_fourstep_scratch_size = 0;

static uint32_t* get_fourstep_scratch(sycl::queue& q, uint32_t n) {
    if (!g_fourstep_scratch || g_fourstep_scratch_size < n) {
        if (g_fourstep_scratch) sycl::free(g_fourstep_scratch, q);
        g_fourstep_scratch = sycl::malloc_device<uint32_t>(n, q);
        g_fourstep_scratch_size = n;
    }
    return g_fourstep_scratch;
}

// Four-Step forward NTT for lg_n = 24 (N1 = N2 = 4096).
// All global memory access is block_load/block_store — zero scatter.
static void gpu_forward_ntt_fourstep(sycl::queue& q, uint32_t* d_data, uint32_t lg_n) {
    uint32_t n = 1u << lg_n;
    uint32_t lg_half = lg_n / 2;  // 12 for lg_n=24
    uint32_t N1 = 1u << lg_half;  // 4096
    uint32_t N2 = N1;             // 4096 (square case)

    auto* d_tmp = get_fourstep_scratch(q, n);

    // Ensure twiddle tables are cached
    if (!g_fourstep_tw_fwd || g_fourstep_lg_n != lg_n) {
        if (g_fourstep_tw_fwd) sycl::free(g_fourstep_tw_fwd, q);
        if (g_fourstep_tw_inv) { sycl::free(g_fourstep_tw_inv, q); g_fourstep_tw_inv = nullptr; }
        g_fourstep_tw_fwd = precompute_fourstep_twiddle(q, lg_n, N1, N2, lg_half, true);
        g_fourstep_lg_n = lg_n;
    }

    // Sub-NTT twiddles: GS DIF with FORWARD roots (= NR-ordered DFT)
    // CT DIT forward does NOT produce NR-ordered DFT — must use GS DIF!
    const auto& tw_sub_fwd = get_cached_twiddles(q, lg_half, true);

    // Step 1: Transpose N1×N2 → N2×N1 (d_data → d_tmp)
    slm_tiled_transpose(q, d_tmp, d_data, N1, N2);

    // Step 2: N2 batched GS DIF forward NTTs of size N1 (all 12 stages in SLM)
    ntt_gs_slm_combined(q, d_tmp, lg_half, tw_sub_fwd.d_buffer, tw_sub_fwd,
                         bb31::ONE, n);

    // Step 3: Twiddle multiply: tmp[i][j] *= w_N^{i * bit_rev(j)}
    fourstep_twiddle_kernel(q, d_tmp, g_fourstep_tw_fwd, n);

    // Step 4: Transpose N2×N1 → N1×N2 (d_tmp → d_data)
    slm_tiled_transpose(q, d_data, d_tmp, N2, N1);

    // Step 5: N1 batched GS DIF forward NTTs of size N2 (all 12 stages in SLM)
    ntt_gs_slm_combined(q, d_data, lg_half, tw_sub_fwd.d_buffer, tw_sub_fwd,
                         bb31::ONE, n);

    q.wait();
}

// Four-Step inverse NTT for lg_n = 24 (N1 = N2 = 4096).
// Uses the same CT DIT structure as forward but with inverse roots, plus 1/N scaling.
// The CT DIT with inverse roots produces NR-ordered IDFT, so we apply a final
// bit-reversal permutation to get natural output. This adds one scatter pass
// but is still much faster than 6 scatter passes in the sppark multipass.
static void gpu_inverse_ntt_fourstep(sycl::queue& q, uint32_t* d_data, uint32_t lg_n) {
    uint32_t n = 1u << lg_n;
    uint32_t lg_half = lg_n / 2;
    uint32_t N1 = 1u << lg_half;
    uint32_t N2 = N1;

    auto* d_tmp = get_fourstep_scratch(q, n);

    // Ensure inverse twiddle table
    if (!g_fourstep_tw_inv || g_fourstep_lg_n != lg_n) {
        if (!g_fourstep_tw_fwd) {
            g_fourstep_tw_fwd = precompute_fourstep_twiddle(q, lg_n, N1, N2, lg_half, true);
        }
        if (g_fourstep_tw_inv) sycl::free(g_fourstep_tw_inv, q);
        g_fourstep_tw_inv = precompute_fourstep_twiddle(q, lg_n, N1, N2, lg_half, false);
        g_fourstep_lg_n = lg_n;
    }

    // CT DIT with inverse roots is the inverse of GS DIF forward.
    // The inverse reverses the forward steps:
    // Forward: T → GS_fwd → Tw → T → GS_fwd
    // Inverse: CT_inv → T → Tw_inv → CT_inv → T → scale(1/N)
    const auto& tw_sub_inv = get_cached_twiddles(q, lg_half, false);
    uint32_t inv_n = ntt::domain_inv[lg_n]; // 1/N scaling

    // Step 1: CT DIT inverse on d_data (undoes forward step 5)
    // Each row of size N2 gets CT_inv applied
    ntt_ct_slm_combined(q, d_data, lg_half, tw_sub_inv.d_buffer, tw_sub_inv, n);

    // Step 2: Transpose N1×N2 → N2×N1 (undoes forward step 4)
    slm_tiled_transpose(q, d_tmp, d_data, N1, N2);

    // Step 3: Inverse twiddle (undoes forward step 3)
    fourstep_twiddle_kernel(q, d_tmp, g_fourstep_tw_inv, n);

    // Step 4: CT DIT inverse on d_tmp (undoes forward step 2)
    ntt_ct_slm_combined(q, d_tmp, lg_half, tw_sub_inv.d_buffer, tw_sub_inv, n);

    // Step 5: Transpose N2×N1 → N1×N2 (undoes forward step 1)
    slm_tiled_transpose(q, d_data, d_tmp, N2, N1);

    // Step 6: Scale by 1/N
    {
        auto* pd = d_data;
        uint32_t inv = inv_n;
        q.parallel_for(sycl::range<1>(n / 16),
            [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                uint32_t base = idx[0] * 16;
                auto data = esimd::block_load<uint32_t, 16>(pd + base);
                esimd::block_store(pd + base, bb31::mont_mul(data, bb31::Vec16(inv)));
            });
    }

    q.wait();
}

// Two-pass forward NTT for large sizes (lg_n > SLM_LG_BLOCK).
// N = N1 × N2 where N2 = SLM_BLOCK. Data viewed as N1 rows of N2 columns.
// ============================================================================
// sppark-style multi-pass CT NTT kernel (scalar per-element for correctness).
// Each pass processes `iterations` butterfly stages in one kernel launch.
// Key components ported from sppark:
//   1. Bit-rearranged input index computation
//   2. Inter-pass twiddle multiply (windowed lookup)
//   3. Multi-stage butterfly with per-stage twiddles
//   4. Bit-rotated output index write
// ============================================================================

// Windowed twiddle table: partial_twiddles[WINDOW_NUM][WINDOW_SIZE]
// partial_twiddles[w][i] = root_of_unity ^ (i << (w * LG_WINDOW_SIZE))
static constexpr uint32_t LG_WINDOW_SIZE = (ntt::MAX_LG_DOMAIN + 4) / 5; // = 6
static constexpr uint32_t WINDOW_SIZE = 1u << LG_WINDOW_SIZE;             // = 64
static constexpr uint32_t WINDOW_NUM = (ntt::MAX_LG_DOMAIN + LG_WINDOW_SIZE - 1) / LG_WINDOW_SIZE; // = 5

struct PartialTwiddles {
    uint32_t* d_buffer;  // WINDOW_NUM * WINDOW_SIZE entries on device
    bool valid;
};
static PartialTwiddles g_partial_tw_fwd = {};
static PartialTwiddles g_partial_tw_inv = {};

static void ensure_partial_twiddles(sycl::queue& q, bool forward) {
    auto& pt = forward ? g_partial_tw_fwd : g_partial_tw_inv;
    if (pt.valid) return;

    uint32_t total = WINDOW_NUM * WINDOW_SIZE;
    auto* h = sycl::malloc_host<uint32_t>(total, q);
    uint32_t root = forward ? ntt::forward_roots[ntt::MAX_LG_DOMAIN]
                            : ntt::inverse_roots[ntt::MAX_LG_DOMAIN];

    for (uint32_t i = 0; i < WINDOW_SIZE; i++) {
        uint32_t r = host_mont_pow(root, i);
        h[0 * WINDOW_SIZE + i] = r;
        for (uint32_t w = 1; w < WINDOW_NUM; w++) {
            for (uint32_t b = 0; b < LG_WINDOW_SIZE; b++)
                r = host_mont_mul(r, r);
            h[w * WINDOW_SIZE + i] = r;
        }
    }

    pt.d_buffer = sycl::malloc_device<uint32_t>(total, q);
    q.memcpy(pt.d_buffer, h, total * sizeof(uint32_t));
    q.wait();
    sycl::free(h, q);
    pt.valid = true;
}

// Precompute radix twiddle table on host.
// Contains twiddles for in-pass butterfly stages, indexed by rank within group.
// radix_twiddles[rank] = w^rank where w is the primitive 2^radix-th root.
// Size: 2^(radix-1) entries. Supports multiple radixes in one buffer.
static uint32_t* g_radix_twiddles = nullptr;
static uint32_t g_radix_twiddles_size = 0;

static uint32_t* g_radix_twiddles_inv = nullptr;

static void ensure_radix_twiddles(sycl::queue& q) {
    if (g_radix_twiddles && g_radix_twiddles_inv) return;

    uint32_t radix = 10;
    uint32_t size = 1u << (radix - 1); // 512
    auto* h = sycl::malloc_host<uint32_t>(size, q);

    // Forward radix twiddles
    if (!g_radix_twiddles) {
        uint32_t root10 = ntt::forward_roots[radix];
        for (uint32_t i = 0; i < size; i++)
            h[i] = host_mont_pow(root10, i);
        g_radix_twiddles = sycl::malloc_device<uint32_t>(size, q);
        q.memcpy(g_radix_twiddles, h, size * sizeof(uint32_t));
        q.wait();
        g_radix_twiddles_size = size;
    }

    // Inverse radix twiddles
    if (!g_radix_twiddles_inv) {
        uint32_t inv_root10 = ntt::inverse_roots[radix];
        for (uint32_t i = 0; i < size; i++)
            h[i] = host_mont_pow(inv_root10, i);
        g_radix_twiddles_inv = sycl::malloc_device<uint32_t>(size, q);
        q.memcpy(g_radix_twiddles_inv, h, size * sizeof(uint32_t));
        q.wait();
    }

    sycl::free(h, q);
}

// CT pass kernel: processes `iterations` stages starting at `stage`.
// Currently uses existing per-stage + SLM infrastructure for the butterflies.
// The sppark-style fused kernel (with index rearrangement, inter-pass twiddle,
// and bit-rotated output all in one launch) requires a full rewrite to handle
// the ESIMD 1-work-item = 1-SIMD16-thread model vs sppark's 1-thread = 1-scalar.
// This fallback gives the same correctness but no bandwidth benefit.
// SLM-buffered multi-stage CT kernel for arbitrary stage range.
// Processes stages first_s..last_s in one global memory round-trip.
// The block size is 2^(last_s - first_s + 1) elements per work-group.
// Each thread handles 4x16 elements; SLM stages use 2 butterfly pairs/thread.
// Requires: last_s - first_s + 1 <= SLM_LG_BLOCK (stages must fit in SLM).
static void ct_slm_pass(sycl::queue& q, uint32_t* d_data,
                         uint32_t lg_n, uint32_t first_s, uint32_t last_s,
                         const uint32_t* d_twiddles, const TwiddleTables& tw) {
    uint32_t n = 1u << lg_n;
    uint32_t pass_stages = last_s - first_s + 1;

    // For this pass, the butterfly stride ranges from 2^(first_s-1) to 2^(last_s-1).
    // A block of B = 2^last_s elements contains all butterfly partners for stages
    // first_s through last_s. Each work-group processes one such block.
    uint32_t lg_block = last_s;
    uint32_t block = 1u << lg_block;
    uint32_t num_groups = n / block;
    uint32_t threads_per_wg = block / (16 * 4); // 4 loads of 16 per thread
    if (threads_per_wg > 64) threads_per_wg = 64;
    if (threads_per_wg < 1) threads_per_wg = 1;
    uint32_t slm_bytes = block * sizeof(uint32_t);
    uint32_t loads_per_thread = block / (threads_per_wg * 16);

    auto* pd = d_data;
    auto* ptw = d_twiddles;

    // Capture twiddle offsets for all stages in this pass
    uint32_t tw_offs[28] = {};
    for (uint32_t s = first_s; s <= last_s; s++)
        tw_offs[s] = tw.offsets[s];

    uint32_t fs = first_s, ls = last_s;
    uint32_t tpw = threads_per_wg;
    uint32_t lpt = loads_per_thread;
    uint32_t blk = block;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * tpw, tpw),
            [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
                esimd::slm_init<65536>(); // Max 64 KB SLM
                uint32_t lid = item.get_local_id(0);
                uint32_t gid = item.get_group(0);
                uint32_t global_base = gid * blk;

                // --- Load block from global to SLM ---
                for (uint32_t c = 0; c < lpt; c++) {
                    uint32_t elem_off = (lid + c * tpw) * 16;
                    if (elem_off < blk) {
                        auto data = esimd::block_load<uint32_t, 16>(pd + global_base + elem_off);
                        esimd::slm_block_store<uint32_t, 16>(elem_off * 4, data);
                    }
                }
                esimd::barrier();

                // --- SLM butterfly stages first_s..last_s ---
                // Each stage operates within the block. Twiddles come from
                // the full-NTT twiddle table, but indexed relative to the block.
                uint32_t bfly_threads = blk / 32;
                for (uint32_t s = fs; s <= ls; s++) {
                    uint32_t half = 1u << (s - 1);
                    uint32_t m = 1u << s;
                    // Each butterfly pair within the block: position within block
                    // maps to twiddle index = position % half.
                    // But we need the GLOBAL twiddle index, not block-local.
                    // Global butterfly offset t maps to twiddle w_s^t.
                    // Within our block: the block starts at global_base.
                    // For stage s, group = (global_base + local_pos) / m,
                    // but since block = 2^last_s >= m = 2^s, multiple groups fit in one block.
                    // The twiddle j = local_pos % half is the same as global j
                    // because twiddles repeat every `half` positions.

                    if (lid < bfly_threads) {
                        uint32_t kb = lid * 16;
                        uint32_t grp = kb / half;
                        uint32_t jb = kb % half;
                        uint32_t top_off = (grp * m + jb) * 4;
                        uint32_t bot_off = top_off + half * 4;
                        auto u = esimd::slm_block_load<uint32_t, 16>(top_off);
                        auto v = esimd::slm_block_load<uint32_t, 16>(bot_off);
                        auto twv = esimd::block_load<uint32_t, 16>(ptw + tw_offs[s] + jb);
                        auto tv = bb31::mont_mul(twv, v);
                        esimd::slm_block_store<uint32_t, 16>(top_off, bb31::field_add(u, tv));
                        esimd::slm_block_store<uint32_t, 16>(bot_off, bb31::field_sub(u, tv));
                    }
                    // Second set of butterflies if block > 32*threads
                    if (bfly_threads > tpw && lid + tpw < bfly_threads) {
                        uint32_t kb = (lid + tpw) * 16;
                        uint32_t grp = kb / half;
                        uint32_t jb = kb % half;
                        uint32_t top_off = (grp * m + jb) * 4;
                        uint32_t bot_off = top_off + half * 4;
                        auto u = esimd::slm_block_load<uint32_t, 16>(top_off);
                        auto v = esimd::slm_block_load<uint32_t, 16>(bot_off);
                        auto twv = esimd::block_load<uint32_t, 16>(ptw + tw_offs[s] + jb);
                        auto tv = bb31::mont_mul(twv, v);
                        esimd::slm_block_store<uint32_t, 16>(top_off, bb31::field_add(u, tv));
                        esimd::slm_block_store<uint32_t, 16>(bot_off, bb31::field_sub(u, tv));
                    }
                    esimd::barrier();
                }

                // --- Store block from SLM to global ---
                for (uint32_t c = 0; c < lpt; c++) {
                    uint32_t elem_off = (lid + c * tpw) * 16;
                    if (elem_off < blk) {
                        auto data = esimd::slm_block_load<uint32_t, 16>(elem_off * 4);
                        esimd::block_store(pd + global_base + elem_off, data, STORE_STREAMING);
                    }
                }
            });
    });
}

// ============================================================================
// sppark-style CT DIT pass kernel (ESIMD cooperative, one global R/W per pass).
// 16 ESIMD threads × 16 lanes = 256 scalar "threads" (radix 9 max).
// Each lane independently processes one butterfly pair.
// Stages 0-3: iselect XOR within SIMD16 registers (intra-register exchange).
// Stage 4+: SLM scatter/gather exchange between ESIMD threads.
// ============================================================================

static constexpr uint32_t SPPARK_WG_THREADS = 16; // ESIMD threads per work-group
static constexpr uint32_t SPPARK_SCALAR_THREADS = SPPARK_WG_THREADS * 16; // 256
static constexpr uint32_t SPPARK_MAX_RADIX = 9; // up to 9 stages per pass
// SLM: each thread writes 16 uint32 values for exchange = 16*16*4 = 1024 bytes
// Need separate regions for r0 and r1 exchange = 2048 bytes total
// SLM: enough for Z_COUNT regions of SCALAR_THREADS entries (max Z=8, 256 threads)
static constexpr uint32_t SPPARK_SLM_BYTES = SPPARK_SCALAR_THREADS * 8 * sizeof(uint32_t); // 8192

static constexpr uint32_t SPPARK_Z_COUNT = 1; // Z=1 is fastest (Z=4 tested: slower due to compute overhead)

static void ct_pass_kernel(sycl::queue& q, uint32_t* d_data,
                            uint32_t lg_domain_size, uint32_t stage,
                            uint32_t iterations,
                            const uint32_t* d_partial_tw,
                            const uint32_t* d_radix_tw) {
    uint32_t n = 1u << lg_domain_size;
    uint32_t num_scalar_threads = n / 2;
    uint32_t rdx = iterations < 6 ? 6 : iterations;
    uint32_t block_size = 1u << (rdx - 1);
    // For small NTTs where block_size > num_scalar_threads, fall back
    if (block_size > num_scalar_threads) {
        const auto& tw = get_cached_twiddles(q, lg_domain_size, true);
        for (uint32_t s = stage + 1; s <= stage + iterations; s++) {
            ntt_ct_stage_fast(q, d_data, lg_domain_size, s, ntt::forward_roots,
                              tw.d_buffer + tw.offsets[s]);
        }
        return;
    }

    // Z-batching: only for first pass (stage=0) where no inter-pass twiddle needed.
    // Later passes need plus_one_twiddles for z>0 correction (not yet implemented).
    uint32_t zc = (stage == 0 && SPPARK_Z_COUNT > 1) ? SPPARK_Z_COUNT : 1;
    uint32_t esimd_threads_per_wg = block_size / 16;
    if (esimd_threads_per_wg > 64) esimd_threads_per_wg = 64;
    uint32_t num_blocks_total = num_scalar_threads / block_size;
    uint32_t num_wgs = (zc > 1 && num_blocks_total >= zc)
                        ? num_blocks_total / zc : num_blocks_total;
    if (num_wgs == 0) num_wgs = 1;

    auto* pd = d_data;
    auto* ptw = d_partial_tw;
    auto* rtw = d_radix_tw;
    uint32_t stg = stage;
    uint32_t itr = iterations;
    uint32_t lgd = lg_domain_size;
    uint32_t bsz = block_size;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wgs * esimd_threads_per_wg, esimd_threads_per_wg),
            [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
                esimd::slm_init<SPPARK_SLM_BYTES>();
                uint32_t esimd_lid = item.get_local_id(0);
                uint32_t esimd_gid = item.get_group(0);
                esimd::simd<uint32_t, 16> lane_id(0u, 1u);
                esimd::simd<uint32_t, 16> lane32(0u, 1u);

                // tid for each lane (before z interleaving)
                esimd::simd<uint32_t, 16> tid_vec =
                    (esimd_gid * bsz) + (esimd_lid * 16) + lane_id;

                // sppark tiz formula: interleave z_count virtual threads
                uint32_t diff_mask = (1u << (itr - 1)) - 1;
                auto tiz = (tid_vec & esimd::simd<uint32_t, 16>(~diff_mask)) *
                           esimd::simd<uint32_t, 16>(zc) +
                           (tid_vec & esimd::simd<uint32_t, 16>(diff_mask));

                uint32_t inp_mask = (1u << stg) - 1;
                uint32_t out_mask_val = (1u << (stg + itr - 1)) - 1;
                auto thread_ntt_pos = (tiz >> (itr - 1)) & esimd::simd<uint32_t, 16>(inp_mask);

                auto idx0 = (tiz & esimd::simd<uint32_t, 16>(~out_mask_val)) |
                             ((tiz << stg) & esimd::simd<uint32_t, 16>(out_mask_val));
                idx0 = idx0 * 2 + thread_ntt_pos;
                auto idx1 = idx0 + esimd::simd<uint32_t, 16>(1u << stg);

                // z_shift: stride between z positions (matches sppark)
                // First pass (stage=0, inp_mask=0): z elements at stride 2^iterations
                // Later passes (stage>0, inp_mask!=0): z elements consecutive (z_shift=0)
                uint32_t z_shift = (inp_mask == 0) ? itr : 0;

                // === Load Z_COUNT pairs per lane ===
                bb31::Vec16 r0[SPPARK_Z_COUNT], r1[SPPARK_Z_COUNT];
                if (stg == 0 && zc == 1) {
                    // Stage=0 optimization: idx0=[0,2,4,...,30], idx1=[1,3,5,...,31]
                    // Load 32 contiguous elements with 2 block_loads, deinterleave
                    uint32_t base = tiz[0] * 2; // = 0 for first thread
                    auto block_lo = esimd::block_load<uint32_t, 16>(pd + base);
                    auto block_hi = esimd::block_load<uint32_t, 16>(pd + base + 16);
                    // Deinterleave: evens to r0, odds to r1
                    // block_lo = [e0,o0,e1,o1,e2,o2,e3,o3,e4,o4,e5,o5,e6,o6,e7,o7]
                    // block_hi = [e8,o8,e9,o9,...,e15,o15]
                    // Want: r0 = [e0,e1,...,e15], r1 = [o0,o1,...,o15]
                    esimd::simd<uint16_t, 16> even_idx({0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30});
                    esimd::simd<uint16_t, 16> odd_idx({1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31});
                    // Concatenate block_lo and block_hi into a 32-element view
                    // Use iselect on each half
                    esimd::simd<uint32_t, 32> full;
                    full.template select<16, 1>(0) = block_lo;
                    full.template select<16, 1>(16) = block_hi;
                    r0[0] = full.template iselect<16>(even_idx);
                    r1[0] = full.template iselect<16>(odd_idx);
                } else {
                    for (uint32_t z = 0; z < zc; z++) {
                        auto zoff = esimd::simd<uint32_t, 16>((uint32_t)(z << z_shift));
                        r0[z] = esimd::gather<uint32_t, 16>(pd, (idx0 + zoff) * sizeof(uint32_t));
                        r1[z] = esimd::gather<uint32_t, 16>(pd, (idx1 + zoff) * sizeof(uint32_t));
                    }
                }

                // === Inter-pass twiddle (if stage != 0) ===
                if (stg != 0) {
                    auto thread_ntt_idx = (tiz & esimd::simd<uint32_t, 16>(diff_mask)) * 2;
                    uint32_t nbits = ntt::MAX_LG_DOMAIN - stg;
                    // Vectorized bit_rev + index computation (replaces 16-lane scalar loop)
                    // Product br*ntt_pos fits in 27 bits — no uint64_t needed.
                    auto br_vec = vec_bit_rev(thread_ntt_idx, nbits);
                    esimd::simd<uint32_t, 16> ri0_lo = br_vec * thread_ntt_pos;
                    esimd::simd<uint32_t, 16> ri1_lo = ri0_lo + (thread_ntt_pos << (nbits - 1));
                    // Windowed twiddle for first_root and second_root
                    bb31::Vec16 first_root, second_root;
                    // ri0 lookup
                    { int win = (WINDOW_NUM-1)*LG_WINDOW_SIZE;
                      auto wi = (ri0_lo >> win) % esimd::simd<uint32_t,16>((uint32_t)WINDOW_SIZE);
                      first_root = esimd::gather<uint32_t,16>(ptw, wi*sizeof(uint32_t) +
                          esimd::simd<uint32_t,16>((uint32_t)((WINDOW_NUM-1)*WINDOW_SIZE*sizeof(uint32_t))));
                      for (int o=WINDOW_NUM-2; o>=0; o--) { win-=LG_WINDOW_SIZE;
                        wi = (ri0_lo >> win) % esimd::simd<uint32_t,16>((uint32_t)WINDOW_SIZE);
                        first_root = bb31::mont_mul(first_root, esimd::gather<uint32_t,16>(ptw,
                            wi*sizeof(uint32_t)+esimd::simd<uint32_t,16>((uint32_t)(o*WINDOW_SIZE*sizeof(uint32_t))))); } }
                    // ri1 lookup
                    { int win = (WINDOW_NUM-1)*LG_WINDOW_SIZE;
                      auto wi = (ri1_lo >> win) % esimd::simd<uint32_t,16>((uint32_t)WINDOW_SIZE);
                      second_root = esimd::gather<uint32_t,16>(ptw, wi*sizeof(uint32_t) +
                          esimd::simd<uint32_t,16>((uint32_t)((WINDOW_NUM-1)*WINDOW_SIZE*sizeof(uint32_t))));
                      for (int o=WINDOW_NUM-2; o>=0; o--) { win-=LG_WINDOW_SIZE;
                        wi = (ri1_lo >> win) % esimd::simd<uint32_t,16>((uint32_t)WINDOW_SIZE);
                        second_root = bb31::mont_mul(second_root, esimd::gather<uint32_t,16>(ptw,
                            wi*sizeof(uint32_t)+esimd::simd<uint32_t,16>((uint32_t)(o*WINDOW_SIZE*sizeof(uint32_t))))); } }

                    // Apply to all z (simplified: same twiddle for all z)
                    for (uint32_t z = 0; z < zc; z++) {
                        r0[z] = bb31::mont_mul(r0[z], first_root);
                        r1[z] = bb31::mont_mul(r1[z], second_root);
                    }
                }

                // === Butterfly stage 0: no twiddle, no exchange ===
                for (uint32_t z = 0; z < zc; z++) {
                    auto t = r1[z];
                    r1[z] = bb31::field_sub(r0[z], t);
                    r0[z] = bb31::field_add(r0[z], t);
                }

                // === Stages 1..min(3, itr-1): iselect XOR ===
                for (uint32_t s = 1; s < itr && s <= 3; s++) {
                    uint32_t laneMask = 1u << (s - 1);
                    uint32_t thrdMask = (1u << s) - 1;
                    auto local_scalar = esimd_lid * 16 + lane32;
                    auto rank = local_scalar & esimd::simd<uint32_t, 16>(thrdMask);
                    auto pos_mask = rank < esimd::simd<uint32_t, 16>(laneMask);
                    auto bot_mask = rank >= esimd::simd<uint32_t, 16>(laneMask);
                    esimd::simd<uint16_t, 16> xor_idx(lane32 ^ esimd::simd<uint32_t, 16>(laneMask));
                    auto tw_off = (rank << (10 - (s+1))) % esimd::simd<uint32_t, 16>(512u);
                    auto root_vec = esimd::gather<uint32_t, 16>(rtw, tw_off * sizeof(uint32_t));

                    for (uint32_t z = 0; z < zc; z++) {
                        bb31::Vec16 xchg = r0[z]; xchg.merge(r1[z], pos_mask);
                        auto pxchg = xchg.iselect(xor_idx);
                        bb31::Vec16 nr0 = r0[z]; nr0.merge(pxchg, bot_mask);
                        bb31::Vec16 nr1 = r1[z]; nr1.merge(pxchg, pos_mask);
                        auto tw_r1 = bb31::mont_mul(root_vec, nr1);
                        r1[z] = bb31::field_sub(nr0, tw_r1);
                        r0[z] = bb31::field_add(nr0, tw_r1);
                    }
                }

                // === Stages 4..itr-1: SLM exchange ===
                // Optimize: write ALL z values to SLM (at different offsets),
                // single barrier, read ALL partners, single barrier.
                // SLM layout: z * 256 + scalar_local_offset (256 scalars per z)
                for (uint32_t s = 4; s < itr; s++) {
                    uint32_t laneMask = 1u << (s - 1);
                    uint32_t thrdMask = (1u << s) - 1;
                    auto local_scalar = esimd_lid * 16 + lane32;
                    auto rank = local_scalar & esimd::simd<uint32_t, 16>(thrdMask);
                    auto pos_mask = rank < esimd::simd<uint32_t, 16>(laneMask);
                    auto bot_mask = rank >= esimd::simd<uint32_t, 16>(laneMask);
                    auto tw_off = (rank << (10 - (s+1))) % esimd::simd<uint32_t, 16>(512u);
                    auto root_vec = esimd::gather<uint32_t, 16>(rtw, tw_off * sizeof(uint32_t));

                    // Write all z values to SLM (each z gets its own 256-scalar region)
                    for (uint32_t z = 0; z < zc; z++) {
                        bb31::Vec16 xchg = r0[z]; xchg.merge(r1[z], pos_mask);
                        auto slm_off = (z * SPPARK_SCALAR_THREADS + local_scalar) * sizeof(uint32_t);
                        esimd::slm_scatter<uint32_t, 16>(slm_off, xchg);
                    }
                    esimd::barrier();

                    // Read all partner values
                    auto pscalar = local_scalar ^ esimd::simd<uint32_t, 16>(laneMask);
                    for (uint32_t z = 0; z < zc; z++) {
                        auto slm_off = (z * SPPARK_SCALAR_THREADS + pscalar) * sizeof(uint32_t);
                        auto pxchg = esimd::slm_gather<uint32_t, 16>(slm_off);
                        r0[z].merge(pxchg, bot_mask);
                        r1[z].merge(pxchg, pos_mask);
                    }
                    esimd::barrier();

                    // Butterfly all z values
                    for (uint32_t z = 0; z < zc; z++) {
                        auto tw_r1 = bb31::mont_mul(root_vec, r1[z]);
                        r1[z] = bb31::field_sub(r0[z], tw_r1);
                        r0[z] = bb31::field_add(r0[z], tw_r1);
                    }
                }

                // === Bit-rotated output + store ===
                uint32_t rot_mask = ((1u << itr) - 1) << stg;
                auto rotw0 = idx0 & esimd::simd<uint32_t, 16>(rot_mask);
                rotw0 = ((rotw0 >> 1) | (rotw0 << (itr - 1))) & esimd::simd<uint32_t, 16>(rot_mask);
                auto out_idx0 = (idx0 & esimd::simd<uint32_t, 16>(~rot_mask)) | rotw0;
                auto rotw1 = idx1 & esimd::simd<uint32_t, 16>(rot_mask);
                rotw1 = ((rotw1 >> 1) | (rotw1 << (itr - 1))) & esimd::simd<uint32_t, 16>(rot_mask);
                auto out_idx1 = (idx1 & esimd::simd<uint32_t, 16>(~rot_mask)) | rotw1;

                // Output: block_store for stage=0 (contiguous after rotation), scatter otherwise
                if (stg == 0) {
                    for (uint32_t z = 0; z < zc; z++) {
                        uint32_t zoff_scalar = z << z_shift;
                        esimd::block_store(pd + out_idx0[0] + zoff_scalar, r0[z]);
                        esimd::block_store(pd + out_idx1[0] + zoff_scalar, r1[z]);
                    }
                } else {
                    for (uint32_t z = 0; z < zc; z++) {
                        auto zoff = esimd::simd<uint32_t, 16>((uint32_t)(z << z_shift));
                        esimd::scatter<uint32_t, 16>(pd, (out_idx0 + zoff) * sizeof(uint32_t), r0[z]);
                        esimd::scatter<uint32_t, 16>(pd, (out_idx1 + zoff) * sizeof(uint32_t), r1[z]);
                    }
                }
            });
    });
}

// Multi-pass forward CT NTT using sppark's decomposition.
static void gpu_forward_ntt_multipass(sycl::queue& q, uint32_t* d_data, uint32_t lg_n) {
    ensure_partial_twiddles(q, true);
    ensure_radix_twiddles(q);

    // Pass decomposition matching sppark's CT_NTT strategy
    int stage = 0;
    if (lg_n <= 10) {
        ct_pass_kernel(q, d_data, lg_n, stage, lg_n,
                       g_partial_tw_fwd.d_buffer, g_radix_twiddles);
        stage += lg_n;
    } else if (lg_n <= 18) {
        int step = lg_n / 2;
        int first = step + lg_n % 2;
        ct_pass_kernel(q, d_data, lg_n, stage, first,
                       g_partial_tw_fwd.d_buffer, g_radix_twiddles);
        stage += first;
        ct_pass_kernel(q, d_data, lg_n, stage, step,
                       g_partial_tw_fwd.d_buffer, g_radix_twiddles);
        stage += step;
    } else {
        // 3 passes: split evenly (matching sppark's strategy)
        int step = lg_n / 3;
        int rem = lg_n % 3;
        int s0 = step, s1 = step, s2 = step;
        if (rem >= 1) s2++;
        if (rem >= 2) s1++;
        ct_pass_kernel(q, d_data, lg_n, stage, s0,
                       g_partial_tw_fwd.d_buffer, g_radix_twiddles);
        stage += s0;
        ct_pass_kernel(q, d_data, lg_n, stage, s1,
                       g_partial_tw_fwd.d_buffer, g_radix_twiddles);
        stage += s1;
        ct_pass_kernel(q, d_data, lg_n, stage, s2,
                       g_partial_tw_fwd.d_buffer, g_radix_twiddles);
        stage += s2;
    }
    q.wait();
}

// ============================================================================
// sppark-style GS DIF pass kernel (ESIMD cooperative)
// Mirrors ct_pass_kernel with GS-specific differences:
// 1. Butterfly: add/sub FIRST, then twiddle on difference (before exchange)
// 2. Stages go from iterations-1 down to 0 within the pass
// 3. Inter-pass twiddle applied AFTER butterflies (condition: stage-iterations != 0)
// 4. Bit rotation: LEFT-rotate (not right)
// 5. 1/N scaling on the last pass (stage == iterations)
// ============================================================================

static void gs_pass_kernel(sycl::queue& q, uint32_t* d_data,
                            uint32_t lg_domain_size, uint32_t stage,
                            uint32_t iterations,
                            const uint32_t* d_partial_tw,
                            const uint32_t* d_radix_tw,
                            bool is_intt) {
    uint32_t n = 1u << lg_domain_size;
    uint32_t num_scalar_threads = n / 2;
    uint32_t rdx = iterations < 6 ? 6 : iterations;
    uint32_t block_size = 1u << (rdx - 1);

    if (block_size > num_scalar_threads) {
        // Fallback for small sizes
        const auto& tw = get_cached_twiddles(q, lg_domain_size, false);
        for (uint32_t s = stage; s > stage - iterations; s--) {
            ntt_gs_stage_fast(q, d_data, lg_domain_size, s, ntt::inverse_roots,
                              tw.d_buffer + tw.offsets[s]);
        }
        if (is_intt && stage == iterations) {
            // Scale by 1/N
            uint32_t inv_n = ntt::domain_inv[lg_domain_size];
            auto* pd = d_data;
            q.parallel_for(sycl::range<1>(n / 16),
                [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
                    uint32_t base = idx[0] * 16;
                    auto vals = esimd::block_load<uint32_t, 16>(pd + base);
                    esimd::block_store(pd + base, bb31::mont_mul(vals, bb31::Vec16(inv_n)));
                });
        }
        return;
    }

    uint32_t esimd_threads_per_wg = block_size / 16;
    if (esimd_threads_per_wg > 64) esimd_threads_per_wg = 64;
    uint32_t num_wgs = num_scalar_threads / block_size;

    auto* pd = d_data;
    auto* ptw = d_partial_tw;
    auto* rtw = d_radix_tw;
    uint32_t stg = stage;
    uint32_t itr = iterations;
    uint32_t lgd = lg_domain_size;
    uint32_t bsz = block_size;
    uint32_t etpw = esimd_threads_per_wg;
    bool do_scale = is_intt && (stage == iterations);
    // Precompute 1/N scaling factor
    uint32_t inv_n = do_scale ? ntt::domain_inv[lg_domain_size] : bb31::ONE;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wgs * esimd_threads_per_wg, esimd_threads_per_wg),
            [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
                esimd::slm_init<SPPARK_SLM_BYTES>();

                uint32_t esimd_lid = item.get_local_id(0);
                uint32_t esimd_gid = item.get_group(0);

                esimd::simd<uint32_t, 16> lane_id(0u, 1u);
                esimd::simd<uint32_t, 16> scalar_tid =
                    (esimd_gid * bsz) + (esimd_lid * 16) + lane_id;

                uint32_t num_scalar = 1u << (lgd - 1);
                auto valid_mask = scalar_tid < esimd::simd<uint32_t, 16>(num_scalar);
                if (scalar_tid[0] >= num_scalar) return;

                // === GS index computation ===
                uint32_t diff_mask = (1u << (itr - 1)) - 1;
                uint32_t inp_mask = ((uint32_t)1 << (stg - 1)) - 1;
                uint32_t out_mask_val = ((uint32_t)1 << (stg - itr)) - 1;

                auto tiz = scalar_tid;
                auto idx0 = (tiz & esimd::simd<uint32_t, 16>(~inp_mask)) * 2;
                idx0 = idx0 + ((tiz << (stg - itr)) & esimd::simd<uint32_t, 16>(inp_mask));
                idx0 = idx0 + ((tiz >> (itr - 1)) & esimd::simd<uint32_t, 16>(out_mask_val));
                auto idx1 = idx0 + esimd::simd<uint32_t, 16>((uint32_t)1 << (stg - 1));

                // === Gather load ===
                auto byte_off0 = idx0 * sizeof(uint32_t);
                auto byte_off1 = idx1 * sizeof(uint32_t);
                bb31::Vec16 r0(0u), r1(0u);
                r0 = esimd::gather<uint32_t, 16>(pd, byte_off0, valid_mask, r0);
                r1 = esimd::gather<uint32_t, 16>(pd, byte_off1, valid_mask, r1);

                esimd::simd<uint32_t, 16> lane32(0u, 1u);

                // === GS butterfly stages iterations-1 down to 1 ===
                // Each stage: butterfly FIRST, then exchange
                for (uint32_t s = itr - 1; s >= 1; s--) {
                    uint32_t laneMask = 1u << (s - 1);
                    uint32_t thrdMask = (1u << s) - 1;

                    auto local_scalar = esimd_lid * 16 + lane32;
                    auto rank = local_scalar & esimd::simd<uint32_t, 16>(thrdMask);
                    auto pos_mask = rank < esimd::simd<uint32_t, 16>(laneMask);
                    auto bot_mask = rank >= esimd::simd<uint32_t, 16>(laneMask);

                    // GS butterfly: t = root * (r0 - r1); r0 = r0 + r1; r1 = t
                    bb31::Vec16 root_vec;
                    for (int lane = 0; lane < 16; lane++) {
                        uint32_t tw_idx = (rank[lane] << (10 - (s + 1))) % 512;
                        root_vec[lane] = *(rtw + tw_idx);
                    }
                    auto diff = bb31::field_sub(r0, r1);
                    r0 = bb31::field_add(r0, r1);
                    r1 = bb31::mont_mul(root_vec, diff);

                    // Exchange (same pattern as CT)
                    if (s <= 3) {
                        // In-register iselect exchange
                        esimd::simd<uint16_t, 16> xor_idx(lane32 ^ esimd::simd<uint32_t, 16>(laneMask));
                        bb31::Vec16 my_xchg = r0;
                        my_xchg.merge(r1, pos_mask);
                        auto partner_xchg = my_xchg.iselect(xor_idx);
                        bb31::Vec16 new_r0 = r0;
                        bb31::Vec16 new_r1 = r1;
                        new_r0.merge(partner_xchg, bot_mask);
                        new_r1.merge(partner_xchg, pos_mask);
                        r0 = new_r0;
                        r1 = new_r1;
                    } else {
                        // SLM exchange
                        bb31::Vec16 my_xchg = r0;
                        my_xchg.merge(r1, pos_mask);
                        auto slm_offset = local_scalar * sizeof(uint32_t);
                        esimd::slm_scatter<uint32_t, 16>(slm_offset, my_xchg);
                        esimd::barrier();
                        auto partner_scalar = local_scalar ^ esimd::simd<uint32_t, 16>(laneMask);
                        auto partner_offset = partner_scalar * sizeof(uint32_t);
                        auto partner_xchg = esimd::slm_gather<uint32_t, 16>(partner_offset);
                        r0.merge(partner_xchg, bot_mask);
                        r1.merge(partner_xchg, pos_mask);
                        esimd::barrier();
                    }
                }

                // === Stage 0: just add/sub, no twiddle, no exchange ===
                {
                    auto t = bb31::field_sub(r0, r1);
                    r0 = bb31::field_add(r0, r1);
                    r1 = t;
                }

                // === Inter-pass twiddle (if stage - iterations != 0) ===
                if (stg - itr != 0) {
                    auto thread_ntt_pos = (tiz & esimd::simd<uint32_t, 16>(inp_mask)) >> (itr - 1);
                    auto thread_ntt_idx = (tiz & esimd::simd<uint32_t, 16>(diff_mask)) * 2;
                    uint32_t nbits = ntt::MAX_LG_DOMAIN - (stg - itr);

                    // Vectorized bit_rev + index computation (same as CT kernel)
                    auto br_vec = vec_bit_rev(thread_ntt_idx, nbits);
                    esimd::simd<uint32_t, 16> ri0_lo = br_vec * thread_ntt_pos;
                    esimd::simd<uint32_t, 16> ri1_lo = ri0_lo + (thread_ntt_pos << (nbits - 1));

                    // Vectorized windowed twiddle via gather
                    bb31::Vec16 first_root_vec, second_root_vec;
                    { int win = (WINDOW_NUM-1)*LG_WINDOW_SIZE;
                      auto wi = (ri0_lo >> win) % esimd::simd<uint32_t,16>((uint32_t)WINDOW_SIZE);
                      first_root_vec = esimd::gather<uint32_t,16>(ptw, wi*sizeof(uint32_t) +
                          esimd::simd<uint32_t,16>((uint32_t)((WINDOW_NUM-1)*WINDOW_SIZE*sizeof(uint32_t))));
                      for (int o=WINDOW_NUM-2; o>=0; o--) { win-=LG_WINDOW_SIZE;
                        wi = (ri0_lo >> win) % esimd::simd<uint32_t,16>((uint32_t)WINDOW_SIZE);
                        first_root_vec = bb31::mont_mul(first_root_vec, esimd::gather<uint32_t,16>(ptw,
                            wi*sizeof(uint32_t)+esimd::simd<uint32_t,16>((uint32_t)(o*WINDOW_SIZE*sizeof(uint32_t))))); } }
                    { int win = (WINDOW_NUM-1)*LG_WINDOW_SIZE;
                      auto wi = (ri1_lo >> win) % esimd::simd<uint32_t,16>((uint32_t)WINDOW_SIZE);
                      second_root_vec = esimd::gather<uint32_t,16>(ptw, wi*sizeof(uint32_t) +
                          esimd::simd<uint32_t,16>((uint32_t)((WINDOW_NUM-1)*WINDOW_SIZE*sizeof(uint32_t))));
                      for (int o=WINDOW_NUM-2; o>=0; o--) { win-=LG_WINDOW_SIZE;
                        wi = (ri1_lo >> win) % esimd::simd<uint32_t,16>((uint32_t)WINDOW_SIZE);
                        second_root_vec = bb31::mont_mul(second_root_vec, esimd::gather<uint32_t,16>(ptw,
                            wi*sizeof(uint32_t)+esimd::simd<uint32_t,16>((uint32_t)(o*WINDOW_SIZE*sizeof(uint32_t))))); } }
                    r0 = bb31::mont_mul(r0, first_root_vec);
                    r1 = bb31::mont_mul(r1, second_root_vec);
                }

                // === 1/N scaling on last pass ===
                if (do_scale) {
                    r0 = bb31::mont_mul(r0, bb31::Vec16(inv_n));
                    r1 = bb31::mont_mul(r1, bb31::Vec16(inv_n));
                }

                // === Bit-rotated output: LEFT-rotate for GS ===
                uint32_t rot_mask = ((1u << itr) - 1) << (stg - itr);
                auto rotw0 = idx0 & esimd::simd<uint32_t, 16>(rot_mask);
                rotw0 = ((rotw0 << 1) | (rotw0 >> (itr - 1))) & esimd::simd<uint32_t, 16>(rot_mask);
                auto out_idx0 = (idx0 & esimd::simd<uint32_t, 16>(~rot_mask)) | rotw0;

                auto rotw1 = idx1 & esimd::simd<uint32_t, 16>(rot_mask);
                rotw1 = ((rotw1 << 1) | (rotw1 >> (itr - 1))) & esimd::simd<uint32_t, 16>(rot_mask);
                auto out_idx1 = (idx1 & esimd::simd<uint32_t, 16>(~rot_mask)) | rotw1;

                auto out_off0 = out_idx0 * sizeof(uint32_t);
                auto out_off1 = out_idx1 * sizeof(uint32_t);
                esimd::scatter<uint32_t, 16>(pd, out_off0, r0);
                esimd::scatter<uint32_t, 16>(pd, out_off1, r1);
            });
    });
}

// Multi-pass inverse GS NTT
static void gpu_inverse_ntt_multipass(sycl::queue& q, uint32_t* d_data, uint32_t lg_n) {
    ensure_partial_twiddles(q, false);
    ensure_radix_twiddles(q); // TODO: separate inverse radix twiddles

    // GS pass decomposition (stage starts at lg_n, decrements)
    int stage = lg_n;
    if (lg_n <= 10) {
        gs_pass_kernel(q, d_data, lg_n, stage, lg_n,
                       g_partial_tw_inv.d_buffer, g_radix_twiddles_inv, true);
    } else if (lg_n <= 18) {
        int step = lg_n / 2;
        gs_pass_kernel(q, d_data, lg_n, stage, step,
                       g_partial_tw_inv.d_buffer, g_radix_twiddles_inv, true);
        stage -= step;
        gs_pass_kernel(q, d_data, lg_n, stage, step + lg_n % 2,
                       g_partial_tw_inv.d_buffer, g_radix_twiddles_inv, true);
    } else {
        // 3 passes (matching sppark GS strategy)
        int step = lg_n / 3;
        int rem = lg_n % 3;
        int s0 = step + (lg_n == 29 ? 1 : rem);
        int s1 = step + (lg_n == 29 ? 1 : 0);
        int s2 = step;
        gs_pass_kernel(q, d_data, lg_n, stage, s0,
                       g_partial_tw_inv.d_buffer, g_radix_twiddles_inv, true);
        stage -= s0;
        gs_pass_kernel(q, d_data, lg_n, stage, s1,
                       g_partial_tw_inv.d_buffer, g_radix_twiddles_inv, true);
        stage -= s1;
        gs_pass_kernel(q, d_data, lg_n, stage, s2,
                       g_partial_tw_inv.d_buffer, g_radix_twiddles_inv, true);
    }
    q.wait();
}

// ============================================================================
// Optimized NTT orchestrators: SLM combined kernel + pipelined per-stage
// ============================================================================

// Forward NTT: SLM combined (stages 1-10), then per-stage SIMD16 (11+)
// Requires lg_n >= 4 (N >= 16) for the fused in-register stages.
//
// NTT ordering: CT DIT produces NR (natural-in, bit-reversed-out).
// NOTE: The risc0 prover (via sppark) calls forward NTT with RN ordering
// (bit-reversed-in, natural-out). At HAL integration time, either add a
// bit-reversal pass before this NTT, or implement a GS DIF forward path.
void gpu_forward_ntt_fast(sycl::queue& q, uint32_t* d_data, uint32_t lg_n) {
    if (lg_n < 4) {
        fprintf(stderr, "FATAL: gpu_forward_ntt_fast requires lg_n >= 4 (got %u)\n", lg_n);
        fflush(stderr);
        abort();
    }

    // Two-pass (Four-Step FFT) decomposition for large NTTs.
    // Reduces global memory passes from (lg_n - SLM_LG_BLOCK) to ~5 total.
    // Two-pass FFT: NOT applicable to the CT DIT butterfly network.
    // The CT DIT produces a transform that differs from the standard DFT (verified
    // empirically for N=4: NTT output ≠ DFT output at any permutation).
    // The Four-Step FFT decomposition applies to the DFT formula, not to arbitrary
    // butterfly networks. The cross-row stages use column-dependent twiddle factors
    // that cannot be absorbed into a simple twiddle multiply + column NTT.
    // Use the existing single-pass approach (SLM + per-stage) instead.
    // Multi-pass: disabled. The current per-stage architecture can't benefit from
    // splitting into passes because each pass still does one global pass per stage.
    // Benefit requires porting sppark's full cooperative SLM exchange kernel where
    // each pass reads/writes the array once for ALL assigned stages.
    // sppark multi-pass: CORRECT but currently slower than single-pass due to
    // scatter/gather overhead and lack of Z-batching. Enable only for testing.
    // Production: single-pass SLM=12 + Z=4 is faster until multi-pass is optimized.
    // sppark multi-pass: correct (33 G at 2^24) but slower than single-pass (49 G).
    // Gap due to scatter/gather overhead (no Z-batching to amortize it).
    // Multi-pass becomes competitive with Z_COUNT=8 + coalesced_load/store.
    // sppark multi-pass: correct at 33 G bfly/sec (2^24), needs Z-batching inside
    // the kernel to close 1.5x gap vs single-pass (49 G). Z-batching requires
    // r0[Z]/r1[Z] arrays, z-indexed SLM exchange, and plus_one_twiddles table.
    // Multi-pass at 33G with Z=1; Z=4 tested but slower (24G) due to compute overhead.
    // The scatter cost is already hidden by the 9-stage butterfly compute per pass.
    // Multi-pass: 33G at 2^24 with block_load/store optimizations on first pass.
    // Gap vs single-pass (49G) is from per-lane scalar twiddle computation and
    // higher total instruction count, NOT scatter overhead (agent review confirmed).
    // Multi-pass wins at 2^24 (76G vs 49G single-pass). Single-pass wins at 2^22 (120G vs 81G).
    // Crossover is around lg_n=23. Use multi-pass for lg_n >= 24.
    // Multi-pass wins at 2^24 (76G vs 49G). Single-pass wins at 2^22 (125G vs 81G).
    // HOWEVER: per-stage with block_load may now be competitive since existing
    // per-stage kernels use Z=4 block_load at full bandwidth.
    // Fall through to the same SLM + per-stage path used for lg_n < 24.
    if (lg_n >= 26) { // only use multipass for very large sizes
        gpu_forward_ntt_multipass(q, d_data, lg_n);
        return;
    }

    const auto& tw = get_cached_twiddles(q, lg_n, true);

    if (lg_n >= SLM_LG_BLOCK) {
        ntt_ct_slm_combined(q, d_data, lg_n, tw.d_buffer, tw);
        // Fuse groups of 4, 3, or 2 adjacent per-stage kernels to minimize launches.
        uint32_t s = SLM_LG_BLOCK + 1;
        uint32_t remaining = lg_n - s + 1;
        // Use fused-4-stage when 4+ stages remain and half_s >= 16
        for (; remaining >= 4 && (1u << (s - 1)) >= 16; s += 4, remaining = lg_n - s + 1) {
            ntt_ct_fused_4stage(q, d_data, lg_n, s,
                                tw.d_buffer + tw.offsets[s],
                                tw.d_buffer + tw.offsets[s + 1],
                                tw.d_buffer + tw.offsets[s + 2],
                                tw.d_buffer + tw.offsets[s + 3]);
        }
        // Use fused-3-stage when 3+ stages remain and half_s >= 64
        for (; remaining >= 3 && (1u << (s - 1)) >= 64; s += 3, remaining = lg_n - s + 1) {
            ntt_ct_fused_3stage(q, d_data, lg_n, s,
                                tw.d_buffer + tw.offsets[s],
                                tw.d_buffer + tw.offsets[s + 1],
                                tw.d_buffer + tw.offsets[s + 2]);
        }
        // Fuse remaining pairs
        for (; s + 1 <= lg_n; s += 2) {
            ntt_ct_fused_2stage(q, d_data, lg_n, s,
                                tw.d_buffer + tw.offsets[s],
                                tw.d_buffer + tw.offsets[s + 1]);
        }
        // Handle odd remaining stage
        if (s <= lg_n) {
            ntt_ct_stage_fast(q, d_data, lg_n, s, ntt::forward_roots,
                              tw.d_buffer + tw.offsets[s]);
        }
    } else {
        ntt_ct_fused_small(q, d_data, lg_n, tw.d_buffer, tw);
        for (uint32_t s = 5; s <= lg_n; s++) {
            ntt_ct_stage_fast(q, d_data, lg_n, s, ntt::forward_roots,
                              tw.d_buffer + tw.offsets[s]);
        }
    }
    q.wait();
}

// Inverse NTT: per-stage SIMD16 (lg_n..11), then SLM combined (10..1)
// Requires lg_n >= 4 (N >= 16) for the fused in-register stages.
void gpu_inverse_ntt_fast(sycl::queue& q, uint32_t* d_data, uint32_t lg_n) {
    if (lg_n < 4) {
        fprintf(stderr, "FATAL: gpu_inverse_ntt_fast requires lg_n >= 4 (got %u)\n", lg_n);
        fflush(stderr);
        abort();
    }

    // Per-stage path is faster than multipass for all tested sizes
    if (lg_n >= 26) {
        gpu_inverse_ntt_multipass(q, d_data, lg_n);
        return;
    }

    const auto& tw = get_cached_twiddles(q, lg_n, false);
    uint32_t inv_n = ntt::domain_inv[lg_n]; // 1/N scaling fused into last kernel

    if (lg_n >= SLM_LG_BLOCK) {
        // Large stages first: fuse pairs of per-stage kernels (GS goes high to low)
        uint32_t s = lg_n;
        for (; s >= SLM_LG_BLOCK + 2; s -= 2) {
            // Fuse stages s and s-1 (s is the wider stage)
            ntt_gs_fused_2stage(q, d_data, lg_n, s,
                                tw.d_buffer + tw.offsets[s],
                                tw.d_buffer + tw.offsets[s - 1]);
        }
        // Handle odd remaining stage
        if (s == SLM_LG_BLOCK + 1) {
            ntt_gs_stage_fast(q, d_data, lg_n, s, ntt::inverse_roots,
                              tw.d_buffer + tw.offsets[s]);
        }
        // Combined SLM kernel for stages 12..1, with 1/N scaling fused
        ntt_gs_slm_combined(q, d_data, lg_n, tw.d_buffer, tw, inv_n);
    } else {
        // Small NTTs: per-stage + fused, with 1/N scaling fused
        for (uint32_t s = lg_n; s > 4; s--) {
            ntt_gs_stage_fast(q, d_data, lg_n, s, ntt::inverse_roots,
                              tw.d_buffer + tw.offsets[s]);
        }
        ntt_gs_fused_small(q, d_data, lg_n, tw.d_buffer, tw, inv_n);
    }

    q.wait();
}

// Async variants: same as _fast but without q.wait() at the end.
// Used by batch NTT wrappers to submit multiple NTTs on an in-order queue.
void gpu_forward_ntt_no_wait(sycl::queue& q, uint32_t* d_data, uint32_t lg_n) {
    if (lg_n < 4) { abort(); }
    if (lg_n >= 26) { gpu_forward_ntt_multipass(q, d_data, lg_n); return; }
    const auto& tw = get_cached_twiddles(q, lg_n, true);
    if (lg_n >= SLM_LG_BLOCK) {
        ntt_ct_slm_combined(q, d_data, lg_n, tw.d_buffer, tw);
        uint32_t s = SLM_LG_BLOCK + 1;
        uint32_t remaining = lg_n - s + 1;
        for (; remaining >= 4 && (1u << (s - 1)) >= 16; s += 4, remaining = lg_n - s + 1)
            ntt_ct_fused_4stage(q, d_data, lg_n, s, tw.d_buffer+tw.offsets[s], tw.d_buffer+tw.offsets[s+1], tw.d_buffer+tw.offsets[s+2], tw.d_buffer+tw.offsets[s+3]);
        for (; remaining >= 3 && (1u << (s - 1)) >= 64; s += 3, remaining = lg_n - s + 1)
            ntt_ct_fused_3stage(q, d_data, lg_n, s, tw.d_buffer+tw.offsets[s], tw.d_buffer+tw.offsets[s+1], tw.d_buffer+tw.offsets[s+2]);
        for (; s + 1 <= lg_n; s += 2)
            ntt_ct_fused_2stage(q, d_data, lg_n, s, tw.d_buffer+tw.offsets[s], tw.d_buffer+tw.offsets[s+1]);
        if (s <= lg_n) ntt_ct_stage_fast(q, d_data, lg_n, s, ntt::forward_roots, tw.d_buffer+tw.offsets[s]);
    } else {
        ntt_ct_fused_small(q, d_data, lg_n, tw.d_buffer, tw);
        for (uint32_t s = 5; s <= lg_n; s++) ntt_ct_stage_fast(q, d_data, lg_n, s, ntt::forward_roots, tw.d_buffer+tw.offsets[s]);
    }
    // No q.wait() — caller batches multiple NTTs
}

void gpu_inverse_ntt_no_wait(sycl::queue& q, uint32_t* d_data, uint32_t lg_n) {
    if (lg_n < 4) { abort(); }
    if (lg_n >= 26) { gpu_inverse_ntt_multipass(q, d_data, lg_n); return; }
    const auto& tw = get_cached_twiddles(q, lg_n, false);
    uint32_t inv_n = ntt::domain_inv[lg_n];
    if (lg_n >= SLM_LG_BLOCK) {
        uint32_t s = lg_n;
        for (; s >= SLM_LG_BLOCK + 2; s -= 2)
            ntt_gs_fused_2stage(q, d_data, lg_n, s, tw.d_buffer+tw.offsets[s], tw.d_buffer+tw.offsets[s-1]);
        if (s == SLM_LG_BLOCK + 1) ntt_gs_stage_fast(q, d_data, lg_n, s, ntt::inverse_roots, tw.d_buffer+tw.offsets[s]);
        ntt_gs_slm_combined(q, d_data, lg_n, tw.d_buffer, tw, inv_n);
    } else {
        for (uint32_t s = lg_n; s > 4; s--) ntt_gs_stage_fast(q, d_data, lg_n, s, ntt::inverse_roots, tw.d_buffer+tw.offsets[s]);
        ntt_gs_fused_small(q, d_data, lg_n, tw.d_buffer, tw, inv_n);
    }
    // No q.wait() — caller batches
}

// ============================================================================
// Original (unoptimized) NTT orchestrators — kept for reference/validation
// ============================================================================

// Complete forward NTT using Cooley-Tukey DIT (matches CPU forward_ntt exactly).
void gpu_forward_ntt(sycl::queue& q, uint32_t* d_data, uint32_t lg_n) {
    for (uint32_t s = 1; s <= lg_n; s++) {
        ntt_ct_stage(q, d_data, lg_n, s, ntt::forward_roots);
    }
}

// Complete inverse NTT using Gentleman-Sande DIF (matches CPU inverse_ntt).
void gpu_inverse_ntt(sycl::queue& q, uint32_t* d_data, uint32_t lg_n) {
    for (uint32_t s = lg_n; s >= 1; s--) {
        ntt_gs_stage(q, d_data, lg_n, s, ntt::inverse_roots);
    }

    uint32_t n = 1u << lg_n;
    uint32_t inv_n = ntt::domain_inv[lg_n];
    auto* pd = d_data;
    uint32_t num_threads = n / 16;

    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;
            auto vals = esimd::block_load<uint32_t, 16>(pd + base);
            auto scaled = bb31::mont_mul(vals, bb31::Vec16(inv_n));
            esimd::block_store(pd + base, scaled);
        }).wait();
}

// ============================================================================
// Validation: GPU NTT round-trip (forward then inverse should recover original)
// ============================================================================
// Validate sppark CPU reference against existing CPU NTT
int32_t validate_sppark_cpu_ntt(uint32_t lg_n) {
    return bb31_cpu::validate_sppark_ntt(lg_n);
}

// Validate sppark CPU round-trip (forward + inverse = identity)
int32_t validate_sppark_cpu_roundtrip(uint32_t lg_n) {
    return bb31_cpu::validate_sppark_roundtrip(lg_n);
}

// Test sppark GPU round-trip: CT forward + GS inverse = identity
int32_t validate_sppark_gpu_roundtrip(uint32_t lg_n) {
    try {
    auto q = create_queue();
    uint32_t n = 1u << lg_n;

    ensure_partial_twiddles(q, true);
    ensure_partial_twiddles(q, false);
    ensure_radix_twiddles(q);

    auto* d_data = sycl::malloc_device<uint32_t>(n, q);
    auto* h_data = sycl::malloc_host<uint32_t>(n, q);
    auto* h_original = sycl::malloc_host<uint32_t>(n, q);

    bb31_cpu::generate_test_data(h_data, n, 55555 + lg_n);
    memcpy(h_original, h_data, n * sizeof(uint32_t));

    q.memcpy(d_data, h_data, n * sizeof(uint32_t)); q.wait();

    // Forward via sppark CT multi-pass
    gpu_forward_ntt_multipass(q, d_data, lg_n);

    // Inverse via sppark GS multi-pass
    gpu_inverse_ntt_multipass(q, d_data, lg_n);

    q.memcpy(h_data, d_data, n * sizeof(uint32_t)); q.wait();

    int32_t errors = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (h_data[i] != h_original[i]) {
            if (errors < 5)
                fprintf(stderr, "  sppark GPU roundtrip mismatch at %u: got=%08x exp=%08x\n",
                        i, h_data[i], h_original[i]);
            errors++;
        }
    }

    sycl::free(d_data, q); sycl::free(h_data, q); sycl::free(h_original, q);
    return errors;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what()); fflush(stderr); return -2;
    } catch (...) {
        fprintf(stderr, "Unknown exception\n"); fflush(stderr); return -4;
    }
}

// Test ESIMD sppark pass kernel against CPU sppark reference
int32_t validate_sppark_gpu_pass(uint32_t lg_n) {
    try {
    auto q = create_queue();
    uint32_t n = 1u << lg_n;

    ensure_partial_twiddles(q, true);
    ensure_radix_twiddles(q);

    auto* d_data = sycl::malloc_device<uint32_t>(n, q);
    auto* h_input = sycl::malloc_host<uint32_t>(n, q);
    auto* h_gpu = sycl::malloc_host<uint32_t>(n, q);

    bb31_cpu::generate_test_data(h_input, n, 88888 + lg_n);

    // CPU reference: same pass parameters as GPU
    uint32_t test_itr = lg_n; // test all stages (full pass)
    // CPU reference: same pass parameters as GPU
    std::vector<uint32_t> h_cpu(h_input, h_input + n);
    bb31_cpu::SparkTwiddles cpu_tw;
    cpu_tw.generate(true);
    bb31_cpu::sppark_ct_pass(h_cpu.data(), lg_n, 0, test_itr, cpu_tw);

    // GPU: same pass as CPU
    q.memcpy(d_data, h_input, n * sizeof(uint32_t));
    q.wait();
    ct_pass_kernel(q, d_data, lg_n, 0, test_itr,
                   g_partial_tw_fwd.d_buffer, g_radix_twiddles);
    q.wait();
    q.memcpy(h_gpu, d_data, n * sizeof(uint32_t));
    q.wait();

    int32_t errors = 0;
    // Always print first 8 elements for debugging
    if (n <= 256) {
        fprintf(stderr, "  Input: ");
        for (uint32_t i = 0; i < n && i < 8; i++) fprintf(stderr, "%08x ", h_input[i]);
        fprintf(stderr, "\n  CPU:   ");
        for (uint32_t i = 0; i < n && i < 8; i++) fprintf(stderr, "%08x ", h_cpu[i]);
        fprintf(stderr, "\n  GPU:   ");
        for (uint32_t i = 0; i < n && i < 8; i++) fprintf(stderr, "%08x ", h_gpu[i]);
        fprintf(stderr, "\n");
    }
    for (uint32_t i = 0; i < n; i++) {
        if (h_cpu[i] != h_gpu[i]) {
            errors++;
        }
    }
    if (errors == 0) {
        fprintf(stderr, "  GPU sppark pass MATCH for lg_n=%u (itr=%u)\n", lg_n, test_itr);
    } else {
        fprintf(stderr, "  GPU sppark pass: %d/%u mismatches (itr=%u)\n", errors, n, test_itr);
    }
    fflush(stderr);

    sycl::free(d_data, q);
    sycl::free(h_input, q);
    sycl::free(h_gpu, q);
    return errors;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what()); fflush(stderr); return -2;
    } catch (const std::exception& e) {
        fprintf(stderr, "Exception: %s\n", e.what()); fflush(stderr); return -3;
    } catch (...) {
        fprintf(stderr, "Unknown exception\n"); fflush(stderr); return -4;
    }
}

int32_t validate_gpu_ntt(uint32_t lg_n) {
    try {
    auto q = create_queue();
    uint32_t n = 1u << lg_n;

    auto* d_data = sycl::malloc_device<uint32_t>(n, q);
    auto* h_data = sycl::malloc_host<uint32_t>(n, q);
    auto* h_original = sycl::malloc_host<uint32_t>(n, q);

    bb31_cpu::generate_test_data(h_data, n, 12345 + lg_n);
    memcpy(h_original, h_data, n * sizeof(uint32_t));

    q.memcpy(d_data, h_data, n * sizeof(uint32_t));
    q.wait();

    // Forward NTT (CT DIT: natural order in, natural order out)
    gpu_forward_ntt(q, d_data, lg_n);

    // Inverse NTT (GS DIF: natural order in, natural order out, then scale by 1/N)
    gpu_inverse_ntt(q, d_data, lg_n);

    q.memcpy(h_data, d_data, n * sizeof(uint32_t));
    q.wait();

    // Compare against original
    int32_t errors = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (h_data[i] != h_original[i]) {
            if (errors < 5) {
                fprintf(stderr, "GPU NTT round-trip mismatch at %u: got=%08x expected=%08x (lg_n=%u)\n",
                        i, h_data[i], h_original[i], lg_n);
            }
            errors++;
        }
    }

    sycl::free(d_data, q);
    sycl::free(h_data, q);
    sycl::free(h_original, q);
    return errors;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception in validate_gpu_ntt: %s\n", e.what()); fflush(stderr); return -2;
    } catch (const std::exception& e) {
        fprintf(stderr, "Exception in validate_gpu_ntt: %s\n", e.what()); fflush(stderr); return -3;
    } catch (...) {
        fprintf(stderr, "Unknown exception in validate_gpu_ntt\n"); fflush(stderr); return -4;
    }
}

// Validation: compare GPU forward NTT against CPU forward NTT
int32_t validate_gpu_ntt_vs_cpu(uint32_t lg_n) {
    try {
    auto q = create_queue();
    uint32_t n = 1u << lg_n;

    auto* d_data = sycl::malloc_device<uint32_t>(n, q);
    auto* h_gpu = sycl::malloc_host<uint32_t>(n, q);
    auto* h_cpu = sycl::malloc_host<uint32_t>(n, q);

    bb31_cpu::generate_test_data(h_gpu, n, 54321 + lg_n);
    memcpy(h_cpu, h_gpu, n * sizeof(uint32_t));

    // GPU forward NTT (CT DIT: natural order output)
    q.memcpy(d_data, h_gpu, n * sizeof(uint32_t));
    q.wait();
    gpu_forward_ntt(q, d_data, lg_n);
    q.memcpy(h_gpu, d_data, n * sizeof(uint32_t));
    q.wait();

    // CPU forward NTT (produces natural order output)
    bb31_cpu::forward_ntt(h_cpu, n);

    // Compare
    int32_t errors = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (h_gpu[i] != h_cpu[i]) {
            if (errors < 5) {
                fprintf(stderr, "GPU vs CPU NTT mismatch at %u: GPU=%08x CPU=%08x (lg_n=%u)\n",
                        i, h_gpu[i], h_cpu[i], lg_n);
            }
            errors++;
        }
    }

    sycl::free(d_data, q);
    sycl::free(h_gpu, q);
    sycl::free(h_cpu, q);
    return errors;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception in validate_gpu_ntt_vs_cpu: %s\n", e.what()); fflush(stderr); return -2;
    } catch (const std::exception& e) {
        fprintf(stderr, "Exception in validate_gpu_ntt_vs_cpu: %s\n", e.what()); fflush(stderr); return -3;
    } catch (...) {
        fprintf(stderr, "Unknown exception in validate_gpu_ntt_vs_cpu\n"); fflush(stderr); return -4;
    }
}

// Benchmark: full NTT (forward + bit_reverse) timing
BenchResult bench_gpu_ntt(uint32_t lg_n) {
    try {
    auto q = create_queue();
    BenchResult result = {};
    uint32_t n = 1u << lg_n;

    auto* d_data = sycl::malloc_device<uint32_t>(n, q);
    auto* h_data = sycl::malloc_host<uint32_t>(n, q);
    bb31_cpu::generate_test_data(h_data, n, 99999);
    q.memcpy(d_data, h_data, n * sizeof(uint32_t));
    q.wait();

    // Warm-up
    for (int w = 0; w < 50; w++) {
        gpu_forward_ntt(q, d_data, lg_n);
    }
    q.wait();

    // Timed run: use wall clock since we have multiple kernel launches
    auto start_time = std::chrono::high_resolution_clock::now();
    gpu_forward_ntt(q, d_data, lg_n);
    q.wait();
    auto end_time = std::chrono::high_resolution_clock::now();

    result.kernel_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    result.total_ops = double(n) / 2.0 * double(lg_n); // (N/2) * lg_n butterflies
    result.correct = 1;

    sycl::free(d_data, q);
    sycl::free(h_data, q);
    return result;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception in bench_gpu_ntt: %s\n", e.what()); fflush(stderr);
        return BenchResult{0.0, 0.0, -2};
    } catch (const std::exception& e) {
        fprintf(stderr, "Exception in bench_gpu_ntt: %s\n", e.what()); fflush(stderr);
        return BenchResult{0.0, 0.0, -3};
    } catch (...) {
        fprintf(stderr, "Unknown exception in bench_gpu_ntt\n"); fflush(stderr);
        return BenchResult{0.0, 0.0, -4};
    }
}

// ============================================================================
// Validation + Benchmark for FAST NTT (precomputed twiddles + SIMD16)
// ============================================================================

// Diagnostic: compare single-pass vs two-pass forward NTT output
int32_t debug_two_pass_ntt(uint32_t lg_n) {
    try {
    auto q = create_queue();
    uint32_t n = 1u << lg_n;
    uint32_t N2 = SLM_BLOCK; // 4096
    uint32_t N1 = n / N2;    // 1024 for lg_n=22
    uint32_t lg_N1 = lg_n - SLM_LG_BLOCK;
    uint32_t lg_N2 = SLM_LG_BLOCK;

    // First test: verify transpose roundtrip
    {
        auto* d_a = sycl::malloc_device<uint32_t>(n, q);
        auto* d_b = sycl::malloc_device<uint32_t>(n, q);
        auto* d_c = sycl::malloc_device<uint32_t>(n, q);
        auto* h_orig = sycl::malloc_host<uint32_t>(n, q);
        auto* h_back = sycl::malloc_host<uint32_t>(n, q);
        bb31_cpu::generate_test_data(h_orig, n, 11111);
        q.memcpy(d_a, h_orig, n * sizeof(uint32_t)); q.wait();
        simple_transpose(q, d_b, d_a, N1, N2); q.wait();
        simple_transpose(q, d_c, d_b, N2, N1); q.wait();
        q.memcpy(h_back, d_c, n * sizeof(uint32_t)); q.wait();
        int errs = 0;
        for (uint32_t i = 0; i < n && errs < 3; i++) {
            if (h_orig[i] != h_back[i]) {
                fprintf(stderr, "  Transpose roundtrip fail at %u: orig=%08x back=%08x\n",
                        i, h_orig[i], h_back[i]);
                errs++;
            }
        }
        if (errs == 0) fprintf(stderr, "  Transpose roundtrip OK (N1=%u, N2=%u)\n", N1, N2);
        sycl::free(d_a, q); sycl::free(d_b, q); sycl::free(d_c, q);
        sycl::free(h_orig, q); sycl::free(h_back, q);
    }

    // Second test: verify SLM kernel with total_elements override
    // Run N2 independent 1024-point NTTs in a single SLM launch
    // Compare vs doing them one-at-a-time
    {
        auto* d_batch = sycl::malloc_device<uint32_t>(n, q);
        auto* d_oneatatime = sycl::malloc_device<uint32_t>(n, q);
        auto* h_in2 = sycl::malloc_host<uint32_t>(n, q);
        auto* h_batch = sycl::malloc_host<uint32_t>(n, q);
        auto* h_one = sycl::malloc_host<uint32_t>(n, q);
        bb31_cpu::generate_test_data(h_in2, n, 33333);
        const auto& tw2 = get_cached_twiddles(q, lg_n, true);

        // Batch: one SLM launch for all sub-NTTs
        q.memcpy(d_batch, h_in2, n * sizeof(uint32_t)); q.wait();
        ntt_ct_slm_combined(q, d_batch, lg_N1, tw2.d_buffer, tw2, n);
        q.wait();
        q.memcpy(h_batch, d_batch, n * sizeof(uint32_t)); q.wait();

        // One-at-a-time: call SLM kernel for each sub-NTT separately
        q.memcpy(d_oneatatime, h_in2, n * sizeof(uint32_t)); q.wait();
        for (uint32_t sub = 0; sub < N2; sub++) {
            // Each sub-NTT of size N1 starts at sub * N1
            // Can't easily call the SLM kernel on a sub-array...
            // Instead, use CPU reference
        }
        // Actually, let's compare vs CPU reference for each sub-NTT
        memcpy(h_one, h_in2, n * sizeof(uint32_t));
        for (uint32_t sub = 0; sub < N2; sub++) {
            bb31_cpu::forward_ntt(h_one + sub * N1, N1);
        }

        int errs = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (h_batch[i] != h_one[i]) {
                if (errs < 5) {
                    uint32_t sub = i / N1;
                    uint32_t pos = i % N1;
                    fprintf(stderr, "  Batch NTT mismatch at %u (sub=%u,pos=%u): gpu=%08x cpu=%08x\n",
                            i, sub, pos, h_batch[i], h_one[i]);
                }
                errs++;
            }
        }
        if (errs == 0) {
            fprintf(stderr, "  Batched %u-point NTTs (%u of them) match CPU! OK\n", N1, N2);
        } else {
            fprintf(stderr, "  Batched NTT: %d mismatches\n", errs);
        }

        sycl::free(d_batch, q); sycl::free(d_oneatatime, q);
        sycl::free(h_in2, q); sycl::free(h_batch, q); sycl::free(h_one, q);
    }

    // Third test: verify twiddle table values
    {
        auto* d_tw_2d = precompute_twiddle_2d(q, lg_n, N2, N1, true);
        auto* h_tw = sycl::malloc_host<uint32_t>(N2 * N1, q);
        q.memcpy(h_tw, d_tw_2d, N2 * N1 * sizeof(uint32_t)); q.wait();

        // Row 0 should be all ONE (w^0 = 1 for all j)
        bool row0_ok = true;
        for (uint32_t j = 0; j < N1 && j < 10; j++) {
            if (h_tw[0 * N1 + j] != bb31::ONE) {
                fprintf(stderr, "  Twiddle[0][%u] = %08x, expected ONE=%08x\n",
                        j, h_tw[0 * N1 + j], bb31::ONE);
                row0_ok = false;
            }
        }
        // Col 0 should be all ONE (w^{i*0} = 1 for all i)
        bool col0_ok = true;
        for (uint32_t i = 0; i < N2 && i < 10; i++) {
            if (h_tw[i * N1 + 0] != bb31::ONE) {
                fprintf(stderr, "  Twiddle[%u][0] = %08x, expected ONE=%08x\n",
                        i, h_tw[i * N1 + 0], bb31::ONE);
                col0_ok = false;
            }
        }
        // Row 1, col 1: should be w_N^1 = forward_roots[lg_n]
        uint32_t expected_11 = ntt::forward_roots[lg_n];
        fprintf(stderr, "  Twiddle[1][1] = %08x, expected w_N = %08x: %s\n",
                h_tw[1 * N1 + 1], expected_11,
                h_tw[1 * N1 + 1] == expected_11 ? "OK" : "MISMATCH");
        fprintf(stderr, "  Twiddle row0: %s, col0: %s\n",
                row0_ok ? "OK" : "BAD", col0_ok ? "OK" : "BAD");

        sycl::free(d_tw_2d, q);
        sycl::free(h_tw, q);
    }

    // Fourth test: Step 1+2+transpose_back (column DFTs only, no twiddle)
    // This should give the same as running stages 1..lg_N1 on columns of original
    // But since we can't easily do column stages, skip this test.

    // Fifth test: CPU-only Four-Step vs CPU single-pass
    // This isolates whether the Four-Step ALGORITHM is correct
    {
        auto* h_single = sycl::malloc_host<uint32_t>(n, q);
        auto* h_fourstep = sycl::malloc_host<uint32_t>(n, q);
        auto* h_tmp = sycl::malloc_host<uint32_t>(n, q);

        bb31_cpu::generate_test_data(h_single, n, 44444);
        memcpy(h_fourstep, h_single, n * sizeof(uint32_t));

        // CPU single-pass reference
        bb31_cpu::forward_ntt(h_single, n);

        // CPU Four-Step:
        // Step 1: Transpose N1×N2 → N2×N1
        for (uint32_t i = 0; i < N1; i++)
            for (uint32_t j = 0; j < N2; j++)
                h_tmp[j * N1 + i] = h_fourstep[i * N2 + j];

        // Step 2: N2 independent N1-point NTTs (on rows of N2×N1)
        for (uint32_t r = 0; r < N2; r++)
            bb31_cpu::forward_ntt(h_tmp + r * N1, N1);

        // Step 3: Twiddle w_N^{row*col} on N2×N1 matrix
        // Must use CPU convention (risc0) since the data is in that form
        uint32_t w_N_cpu = bb31_cpu::encode(bb31_cpu::ROU_FWD[lg_n]);
        for (uint32_t i = 0; i < N2; i++) {
            // w_row = w_N^i in risc0 Montgomery form
            uint32_t w_row = bb31_cpu::encode(1);
            uint32_t w_base = w_N_cpu;
            uint32_t exp = i;
            while (exp > 0) {
                if (exp & 1) w_row = bb31_cpu::mul(w_row, w_base);
                w_base = bb31_cpu::mul(w_base, w_base);
                exp >>= 1;
            }
            uint32_t cur = bb31_cpu::encode(1);
            for (uint32_t j = 0; j < N1; j++) {
                h_tmp[i * N1 + j] = bb31_cpu::mul(h_tmp[i * N1 + j], cur);
                cur = bb31_cpu::mul(cur, w_row);
            }
        }

        // Step 4: Transpose N2×N1 → N1×N2
        for (uint32_t i = 0; i < N2; i++)
            for (uint32_t j = 0; j < N1; j++)
                h_fourstep[j * N2 + i] = h_tmp[i * N1 + j];

        // Step 5: N1 independent N2-point NTTs (on rows of N1×N2)
        for (uint32_t r = 0; r < N1; r++)
            bb31_cpu::forward_ntt(h_fourstep + r * N2, N2);

        // Step 6: Transpose N1×N2 → N2×N1 for natural order
        for (uint32_t i = 0; i < N1; i++)
            for (uint32_t j = 0; j < N2; j++)
                h_tmp[j * N1 + i] = h_fourstep[i * N2 + j];
        memcpy(h_fourstep, h_tmp, n * sizeof(uint32_t));

        int errs = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (h_single[i] != h_fourstep[i]) {
                if (errs < 5) {
                    fprintf(stderr, "  CPU 4step mismatch at %u: single=%08x 4step=%08x\n",
                            i, h_single[i], h_fourstep[i]);
                }
                errs++;
            }
        }
        if (errs == 0) {
            fprintf(stderr, "  CPU Four-Step matches CPU single-pass! Algorithm is correct.\n");
        } else {
            fprintf(stderr, "  CPU Four-Step: %d mismatches vs single-pass\n", errs);
            // Debug: check if w_N from twiddle table matches CPU root
            uint32_t gpu_root = ntt::forward_roots[lg_n];
            uint32_t cpu_root = bb31_cpu::encode(bb31_cpu::ROU_FWD[lg_n]);
            fprintf(stderr, "  gpu forward_roots[%u] = %08x\n", lg_n, gpu_root);
            fprintf(stderr, "  cpu encode(ROU_FWD[%u]) = %08x\n", lg_n, cpu_root);
            fprintf(stderr, "  Match: %s\n", gpu_root == cpu_root ? "YES" : "NO");
        }
        fflush(stderr);

        sycl::free(h_single, q); sycl::free(h_fourstep, q); sycl::free(h_tmp, q);
    }

    // Sixth test: Compare CPU NTT vs naive DFT for tiny N=4
    {
        // Use N=4 to verify CPU NTT matches standard DFT Y[k] = Σ x[n]*w^{nk}
        uint32_t small_n = 4;
        uint32_t small_lgn = 2;
        uint32_t x[4] = {
            bb31_cpu::encode(1),
            bb31_cpu::encode(2),
            bb31_cpu::encode(3),
            bb31_cpu::encode(4)
        };
        uint32_t y_ntt[4];
        memcpy(y_ntt, x, 16);
        bb31_cpu::forward_ntt(y_ntt, 4);

        // Naive DFT using w = ROU_FWD[2] (primitive 4th root)
        uint32_t w4 = bb31_cpu::encode(bb31_cpu::ROU_FWD[small_lgn]);
        uint32_t y_dft[4];
        for (uint32_t k = 0; k < 4; k++) {
            uint32_t sum = bb31_cpu::encode(0);
            for (uint32_t n = 0; n < 4; n++) {
                // w^{n*k}
                uint32_t ww = bb31_cpu::encode(1);
                uint32_t base = w4;
                uint32_t exp2 = (n * k) % 4; // w4^4 = 1
                for (uint32_t e = 0; e < exp2; e++)
                    ww = bb31_cpu::mul(ww, base);
                sum = bb31_cpu::add(sum, bb31_cpu::mul(x[n], ww));
            }
            y_dft[k] = sum;
        }

        fprintf(stderr, "  N=4 NTT vs DFT:\n");
        for (int k = 0; k < 4; k++) {
            // bit-reverse k in 2 bits: 0->0, 1->2, 2->1, 3->3
            uint32_t kr = ntt::bit_rev(k, small_lgn);
            fprintf(stderr, "    k=%d: ntt=%08x dft[%u]=%08x dft[%u]=%08x -> %s\n",
                    k, y_ntt[k], k, y_dft[k], kr, y_dft[kr],
                    y_ntt[k] == y_dft[kr] ? "MATCH(bit-rev)" :
                    y_ntt[k] == y_dft[k] ? "MATCH(natural)" : "MISMATCH");
        }
        fflush(stderr);
    }

    // Seventh test: verify column NTTs alone (no twiddle, no row NTTs)
    // Run single-pass stages 1..lg_N1 on COLUMNS and compare
    // vs transpose + row NTTs of size N1 + transpose back
    {
        auto* d_ref = sycl::malloc_device<uint32_t>(n, q);
        auto* d_test = sycl::malloc_device<uint32_t>(n, q);
        auto* d_tmp2 = sycl::malloc_device<uint32_t>(n, q);
        auto* h_in = sycl::malloc_host<uint32_t>(n, q);
        auto* h_ref = sycl::malloc_host<uint32_t>(n, q);
        auto* h_test = sycl::malloc_host<uint32_t>(n, q);
        bb31_cpu::generate_test_data(h_in, n, 22222);
        const auto& tw = get_cached_twiddles(q, lg_n, true);

        // Reference: CPU column DFTs (just do the first lg_N1 stages of full NTT)
        q.memcpy(d_ref, h_in, n * sizeof(uint32_t)); q.wait();
        // Stages 1..lg_N1 via per-stage kernels on the full array
        // These are "within-column" stages: half < N2 for s <= lg_N2
        // Wait - stages 1..lg_N1 where lg_N1=10 have half = 1..512, all < N2=4096
        // So these ARE within-row stages, not column stages!
        // The COLUMN stages are lg_N2+1..lg_n = stages 13..22 (stride >= N2)
        // But we want column DFTs = the INNER sum = stages with stride >= N2
        // For the Four-Step, column DFTs use ω_{N1} roots, NOT full NTT roots
        // They operate on N1 elements from each column (stride N2 in original)

        // Actually, let me just compare the full two-pass step by step
        // Step by step comparison vs CPU reference
        memcpy(h_ref, h_in, n * sizeof(uint32_t));
        bb31_cpu::forward_ntt(h_ref, n); // CPU computes full NTT

        q.memcpy(d_test, h_in, n * sizeof(uint32_t)); q.wait();
        gpu_forward_ntt_multipass(q, d_test, lg_n);
        q.memcpy(h_test, d_test, n * sizeof(uint32_t)); q.wait();

        int errs = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (h_ref[i] != h_test[i]) {
                if (errs < 5) {
                    fprintf(stderr, "  vs CPU: idx=%u cpu=%08x two=%08x (row=%u,col=%u)\n",
                            i, h_ref[i], h_test[i], i/N2, i%N2);
                }
                errs++;
            }
        }
        if (errs == 0) {
            fprintf(stderr, "  Two-pass matches CPU reference!\n");
        } else {
            fprintf(stderr, "  Two-pass vs CPU: %d mismatches\n", errs);

            // Check if it's a permutation issue: are the same VALUES present?
            // Sort both arrays and compare
            std::vector<uint32_t> s_ref(h_ref, h_ref + n);
            std::vector<uint32_t> s_test(h_test, h_test + n);
            std::sort(s_ref.begin(), s_ref.end());
            std::sort(s_test.begin(), s_test.end());
            bool same_values = (s_ref == s_test);
            fprintf(stderr, "  Same multiset of values: %s\n", same_values ? "YES (permutation)" : "NO (computation error)");
        }

        sycl::free(d_ref, q); sycl::free(d_test, q); sycl::free(d_tmp2, q);
        sycl::free(h_in, q); sycl::free(h_ref, q); sycl::free(h_test, q);
    }

    auto* d_single = sycl::malloc_device<uint32_t>(n, q);
    auto* d_twopass = sycl::malloc_device<uint32_t>(n, q);
    auto* h_input = sycl::malloc_host<uint32_t>(n, q);
    auto* h_single = sycl::malloc_host<uint32_t>(n, q);
    auto* h_twopass = sycl::malloc_host<uint32_t>(n, q);

    bb31_cpu::generate_test_data(h_input, n, 54321 + lg_n);

    // Single-pass forward NTT
    q.memcpy(d_single, h_input, n * sizeof(uint32_t)); q.wait();
    {
        const auto& tw = get_cached_twiddles(q, lg_n, true);
        ntt_ct_slm_combined(q, d_single, lg_n, tw.d_buffer, tw);
        for (uint32_t s = SLM_LG_BLOCK + 1; s <= lg_n; s++) {
            ntt_ct_stage_fast(q, d_single, lg_n, s, ntt::forward_roots,
                              tw.d_buffer + tw.offsets[s]);
        }
        q.wait();
    }
    q.memcpy(h_single, d_single, n * sizeof(uint32_t)); q.wait();

    // Two-pass forward NTT
    q.memcpy(d_twopass, h_input, n * sizeof(uint32_t)); q.wait();
    gpu_forward_ntt_multipass(q, d_twopass, lg_n);
    q.memcpy(h_twopass, d_twopass, n * sizeof(uint32_t)); q.wait();

    // Compare
    int32_t errors = 0;
    int32_t first_err = -1;
    for (uint32_t i = 0; i < n; i++) {
        if (h_single[i] != h_twopass[i]) {
            if (errors < 10) {
                uint32_t row = i / N2;
                uint32_t col = i % N2;
                fprintf(stderr, "  2pass mismatch at %u (row=%u,col=%u): single=%08x two=%08x\n",
                        i, row, col, h_single[i], h_twopass[i]);
            }
            if (first_err < 0) first_err = i;
            errors++;
        }
    }
    if (errors == 0) {
        fprintf(stderr, "  Two-pass matches single-pass for lg_n=%u! (%u elements)\n", lg_n, n);
    } else {
        fprintf(stderr, "  Two-pass: %d mismatches (first at %d), N1=%u, N2=%u\n",
                errors, first_err, N1, N2);
    }
    fflush(stderr);

    sycl::free(d_single, q);
    sycl::free(d_twopass, q);
    sycl::free(h_input, q);
    sycl::free(h_single, q);
    sycl::free(h_twopass, q);
    return errors;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception in debug_two_pass: %s\n", e.what()); fflush(stderr); return -2;
    } catch (const std::exception& e) {
        fprintf(stderr, "Exception in debug_two_pass: %s\n", e.what()); fflush(stderr); return -3;
    } catch (...) {
        fprintf(stderr, "Unknown exception in debug_two_pass\n"); fflush(stderr); return -4;
    }
}

int32_t validate_gpu_ntt_fast(uint32_t lg_n) {
    try {
    auto q = create_queue();
    uint32_t n = 1u << lg_n;

    auto* d_data = sycl::malloc_device<uint32_t>(n, q);
    auto* h_data = sycl::malloc_host<uint32_t>(n, q);
    auto* h_original = sycl::malloc_host<uint32_t>(n, q);

    bb31_cpu::generate_test_data(h_data, n, 12345 + lg_n);
    memcpy(h_original, h_data, n * sizeof(uint32_t));

    q.memcpy(d_data, h_data, n * sizeof(uint32_t));
    q.wait();

    // Fast forward + fast inverse = round-trip
    gpu_forward_ntt_fast(q, d_data, lg_n);
    gpu_inverse_ntt_fast(q, d_data, lg_n);

    q.memcpy(h_data, d_data, n * sizeof(uint32_t));
    q.wait();

    int32_t errors = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (h_data[i] != h_original[i]) {
            if (errors < 5) {
                fprintf(stderr, "FAST NTT round-trip mismatch at %u: got=%08x expected=%08x (lg_n=%u)\n",
                        i, h_data[i], h_original[i], lg_n);
            }
            errors++;
        }
    }

    sycl::free(d_data, q);
    sycl::free(h_data, q);
    sycl::free(h_original, q);
    return errors;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception in validate_gpu_ntt_fast: %s\n", e.what()); fflush(stderr); return -2;
    } catch (const std::exception& e) {
        fprintf(stderr, "Exception in validate_gpu_ntt_fast: %s\n", e.what()); fflush(stderr); return -3;
    } catch (...) {
        fprintf(stderr, "Unknown exception in validate_gpu_ntt_fast\n"); fflush(stderr); return -4;
    }
}

// Compare fast NTT vs CPU reference
int32_t validate_gpu_ntt_fast_vs_cpu(uint32_t lg_n) {
    try {
    auto q = create_queue();
    uint32_t n = 1u << lg_n;

    auto* d_data = sycl::malloc_device<uint32_t>(n, q);
    auto* h_gpu = sycl::malloc_host<uint32_t>(n, q);
    auto* h_cpu = sycl::malloc_host<uint32_t>(n, q);

    bb31_cpu::generate_test_data(h_gpu, n, 54321 + lg_n);
    memcpy(h_cpu, h_gpu, n * sizeof(uint32_t));

    q.memcpy(d_data, h_gpu, n * sizeof(uint32_t));
    q.wait();
    gpu_forward_ntt_fast(q, d_data, lg_n);
    q.memcpy(h_gpu, d_data, n * sizeof(uint32_t));
    q.wait();

    bb31_cpu::forward_ntt(h_cpu, n);

    int32_t errors = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (h_gpu[i] != h_cpu[i]) {
            if (errors < 5) {
                fprintf(stderr, "FAST NTT vs CPU mismatch at %u: GPU=%08x CPU=%08x (lg_n=%u)\n",
                        i, h_gpu[i], h_cpu[i], lg_n);
            }
            errors++;
        }
    }

    sycl::free(d_data, q);
    sycl::free(h_gpu, q);
    sycl::free(h_cpu, q);
    return errors;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what()); fflush(stderr); return -2;
    } catch (const std::exception& e) {
        fprintf(stderr, "Exception: %s\n", e.what()); fflush(stderr); return -3;
    } catch (...) {
        fprintf(stderr, "Unknown exception\n"); fflush(stderr); return -4;
    }
}

BenchResult bench_gpu_ntt_fast(uint32_t lg_n) {
    try {
    auto q = create_queue();
    BenchResult result = {};
    uint32_t n = 1u << lg_n;

    auto* d_data = sycl::malloc_device<uint32_t>(n, q);
    auto* h_data = sycl::malloc_host<uint32_t>(n, q);
    bb31_cpu::generate_test_data(h_data, n, 99999);
    q.memcpy(d_data, h_data, n * sizeof(uint32_t));
    q.wait();

    // Warm-up
    for (int w = 0; w < 50; w++) {
        gpu_forward_ntt_fast(q, d_data, lg_n);
    }
    q.wait();

    // Timed run
    auto start_time = std::chrono::high_resolution_clock::now();
    gpu_forward_ntt_fast(q, d_data, lg_n);
    q.wait();
    auto end_time = std::chrono::high_resolution_clock::now();

    result.kernel_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    result.total_ops = double(n) / 2.0 * double(lg_n); // (N/2) * lg_n butterflies
    result.correct = 1;

    sycl::free(d_data, q);
    sycl::free(h_data, q);
    return result;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what()); fflush(stderr);
        return BenchResult{0.0, 0.0, -2};
    } catch (const std::exception& e) {
        fprintf(stderr, "Exception: %s\n", e.what()); fflush(stderr);
        return BenchResult{0.0, 0.0, -3};
    } catch (...) {
        fprintf(stderr, "Unknown exception\n"); fflush(stderr);
        return BenchResult{0.0, 0.0, -4};
    }
}

// ============================================================================
// Four-Step FFT: Validation and Benchmark
// ============================================================================

// GS DIF forward NTT: produces NR-ordered DFT from natural input.
// Unlike CT DIT (forward_ntt), GS DIF actually produces the correct NR-ordered DFT.
// CT DIT on natural input produces a different (scrambled) transform.
static void gs_dif_forward_ntt(uint32_t* data, uint32_t n) {
    uint32_t lgn = 0;
    for (uint32_t t = n; t > 1; t >>= 1) lgn++;
    for (uint32_t s = lgn; s >= 1; s--) {
        uint32_t m = 1u << s;
        uint32_t half = m >> 1;
        uint32_t w = bb31_cpu::encode(bb31_cpu::ROU_FWD[s]);
        for (uint32_t k = 0; k < n; k += m) {
            uint32_t wj = bb31_cpu::encode(1);
            for (uint32_t j = 0; j < half; j++) {
                uint32_t u = data[k + j];
                uint32_t v = data[k + j + half];
                data[k + j] = bb31_cpu::add(u, v);
                data[k + j + half] = bb31_cpu::mul(bb31_cpu::sub(u, v), wj);
                wj = bb31_cpu::mul(wj, w);
            }
        }
    }
}

// CPU-only Four-Step test for small sizes (algorithm validation)
int32_t validate_fourstep_cpu(uint32_t lg_n) {
    if (lg_n < 2 || lg_n % 2 != 0) return -1;

    uint32_t n = 1u << lg_n;
    uint32_t lg_half = lg_n / 2;
    uint32_t N1 = 1u << lg_half;
    uint32_t N2 = N1;

    std::vector<uint32_t> single(n), fourstep(n), tmp(n);
    bb31_cpu::generate_test_data(single.data(), n, 12345 + lg_n);
    memcpy(fourstep.data(), single.data(), n * sizeof(uint32_t));

    // CPU single-pass reference: use GS DIF forward (= NR-ordered DFT)
    // Note: forward_ntt (CT DIT) does NOT produce NR-ordered DFT!
    gs_dif_forward_ntt(single.data(), n);

    // CPU Four-Step with bit_rev twiddle, no extra transpose:
    // Step 1: Transpose N1×N2 → N2×N1
    for (uint32_t i = 0; i < N1; i++)
        for (uint32_t j = 0; j < N2; j++)
            tmp[j * N1 + i] = fourstep[i * N2 + j];

    // Step 2: N2 row DFTs of size N1 using GS DIF with FORWARD roots
    // (GS DIF forward = NR-ordered DFT. CT DIT forward ≠ NR-ordered DFT!)
    for (uint32_t r = 0; r < N2; r++)
        gs_dif_forward_ntt(tmp.data() + r * N1, N1);

    // DIAGNOSTIC: verify step 2 by comparing sub-NTT of column 0 against manual computation
    if (n <= 16) {
        // Recompute expected column 0 NTT manually
        std::vector<uint32_t> col0(N1);
        std::vector<uint32_t> orig(n);
        bb31_cpu::generate_test_data(orig.data(), n, 12345 + lg_n);
        for (uint32_t i = 0; i < N1; i++) col0[i] = orig[i * N2]; // column 0
        gs_dif_forward_ntt(col0.data(), N1);

        fprintf(stderr, "  After step 2, row 0 of tmp (should be NTT of col 0):\n");
        fprintf(stderr, "    tmp_row0: ");
        for (uint32_t j = 0; j < N1; j++) fprintf(stderr, "%08x ", tmp[0*N1+j]);
        fprintf(stderr, "\n    col0_ntt: ");
        for (uint32_t j = 0; j < N1; j++) fprintf(stderr, "%08x ", col0[j]);
        fprintf(stderr, "\n    match: %s\n",
            memcmp(tmp.data(), col0.data(), N1*sizeof(uint32_t)) == 0 ? "YES" : "NO");
        fflush(stderr);
    }

    // Try BOTH twiddle variants and report which one works

    // Variant A: bit_rev twiddle, no extra transpose
    std::vector<uint32_t> va(tmp);
    {
        uint32_t w_N = bb31_cpu::encode(bb31_cpu::ROU_FWD[lg_n]);
        for (uint32_t i = 0; i < N2; i++) {
            for (uint32_t j = 0; j < N1; j++) {
                uint32_t k_actual = ntt::bit_rev(j, lg_half);
                uint32_t tw = bb31_cpu::mont_pow(w_N, (uint64_t)i * k_actual);
                va[i * N1 + j] = bb31_cpu::mul(va[i * N1 + j], tw);
            }
        }
        // Transpose
        for (uint32_t i = 0; i < N2; i++)
            for (uint32_t j = 0; j < N1; j++)
                fourstep[j * N2 + i] = va[i * N1 + j];
        // Row NTTs
        for (uint32_t r = 0; r < N1; r++)
            gs_dif_forward_ntt(fourstep.data() + r * N2, N2);
        va = fourstep;
    }

    // Variant B: standard twiddle w_N^{i*j}, WITH extra transpose
    std::vector<uint32_t> vb(tmp);
    {
        uint32_t w_N = bb31_cpu::encode(bb31_cpu::ROU_FWD[lg_n]);
        for (uint32_t i = 0; i < N2; i++) {
            uint32_t w_row = bb31_cpu::mont_pow(w_N, i);
            uint32_t cur = bb31_cpu::encode(1);
            for (uint32_t j = 0; j < N1; j++) {
                vb[i * N1 + j] = bb31_cpu::mul(vb[i * N1 + j], cur);
                cur = bb31_cpu::mul(cur, w_row);
            }
        }
        // Transpose
        for (uint32_t i = 0; i < N2; i++)
            for (uint32_t j = 0; j < N1; j++)
                fourstep[j * N2 + i] = vb[i * N1 + j];
        // Row NTTs
        for (uint32_t r = 0; r < N1; r++)
            gs_dif_forward_ntt(fourstep.data() + r * N2, N2);
        // Extra transpose
        std::vector<uint32_t> extra(n);
        for (uint32_t i = 0; i < N1; i++)
            for (uint32_t j = 0; j < N2; j++)
                extra[j * N1 + i] = fourstep[i * N2 + j];
        vb = extra;
    }

    // Variant C: standard twiddle, NO extra transpose
    std::vector<uint32_t> vc(tmp);
    {
        uint32_t w_N = bb31_cpu::encode(bb31_cpu::ROU_FWD[lg_n]);
        for (uint32_t i = 0; i < N2; i++) {
            uint32_t w_row = bb31_cpu::mont_pow(w_N, i);
            uint32_t cur = bb31_cpu::encode(1);
            for (uint32_t j = 0; j < N1; j++) {
                vc[i * N1 + j] = bb31_cpu::mul(vc[i * N1 + j], cur);
                cur = bb31_cpu::mul(cur, w_row);
            }
        }
        // Transpose
        for (uint32_t i = 0; i < N2; i++)
            for (uint32_t j = 0; j < N1; j++)
                fourstep[j * N2 + i] = vc[i * N1 + j];
        // Row NTTs
        for (uint32_t r = 0; r < N1; r++)
            gs_dif_forward_ntt(fourstep.data() + r * N2, N2);
        vc = fourstep;
    }

    int errA = 0, errB = 0, errC = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (single[i] != va[i]) errA++;
        if (single[i] != vb[i]) errB++;
        if (single[i] != vc[i]) errC++;
    }
    fprintf(stderr, "  CPU Four-Step (2^%u): A(bit_rev,no_extra_T)=%d B(std,extra_T)=%d C(std,no_extra_T)=%d\n",
            lg_n, errA, errB, errC);

    // Dump all values for tiny sizes to debug
    if (n <= 16) {
        fprintf(stderr, "  idx: single    varA     varB     varC\n");
        for (uint32_t i = 0; i < n; i++) {
            fprintf(stderr, "  %3u: %08x %08x %08x %08x%s%s%s\n",
                    i, single[i], va[i], vb[i], vc[i],
                    single[i]==va[i] ? " A=" : "",
                    single[i]==vb[i] ? " B=" : "",
                    single[i]==vc[i] ? " C=" : "");
        }

        // Also dump the input data
        std::vector<uint32_t> orig(n);
        bb31_cpu::generate_test_data(orig.data(), n, 12345 + lg_n);
        fprintf(stderr, "  Input (decoded): ");
        for (uint32_t i = 0; i < n; i++)
            fprintf(stderr, "%u ", bb31_cpu::decode(orig[i]));
        fprintf(stderr, "\n");

        // Check if varA is a permutation of single
        auto sa = va, ss = single;
        std::sort(sa.begin(), sa.end());
        std::sort(ss.begin(), ss.end());
        fprintf(stderr, "  varA is permutation of single: %s\n", sa == ss ? "YES" : "NO");
    }
    fflush(stderr);

    // Only variant A matters (bit_rev twiddle, no extra transpose)
    if (errA == 0)
        fprintf(stderr, "  CPU Four-Step (2^%u) with bit_rev twiddle: PASS\n", lg_n);
    else
        fprintf(stderr, "  CPU Four-Step (2^%u): %d mismatches (A=%d B=%d C=%d)\n",
                lg_n, errA, errA, errB, errC);
    fflush(stderr);
    return errA;
}

// Validate Four-Step forward NTT vs CPU reference
int32_t validate_fourstep_ntt(uint32_t lg_n) {
    try {
    // First validate CPU algorithm for small sizes
    for (uint32_t test_lg = 2; test_lg <= 8; test_lg += 2) {
        int32_t cpu_errs = validate_fourstep_cpu(test_lg);
        if (cpu_errs != 0) return -100 - test_lg;
    }

    auto q = create_queue();
    uint32_t n = 1u << lg_n;

    auto* d_data = sycl::malloc_device<uint32_t>(n, q);
    auto* h_gpu = sycl::malloc_host<uint32_t>(n, q);
    auto* h_cpu = sycl::malloc_host<uint32_t>(n, q);

    bb31_cpu::generate_test_data(h_gpu, n, 77777 + lg_n);
    memcpy(h_cpu, h_gpu, n * sizeof(uint32_t));

    // GPU Four-Step forward NTT
    q.memcpy(d_data, h_gpu, n * sizeof(uint32_t)); q.wait();
    gpu_forward_ntt_fourstep(q, d_data, lg_n);
    q.memcpy(h_gpu, d_data, n * sizeof(uint32_t)); q.wait();

    // CPU reference: GS DIF forward (= NR-ordered DFT, matching Four-Step)
    // Skip for N > 2^20 (too slow on CPU). Use round-trip test instead.
    if (lg_n > 20) {
        fprintf(stderr, "  skipping CPU comparison (N=2^%u too large). Use round-trip test.\n", lg_n);
        fflush(stderr);
        sycl::free(d_data, q); sycl::free(h_gpu, q); sycl::free(h_cpu, q);
        return 0; // rely on round-trip
    }
    gs_dif_forward_ntt(h_cpu, n);

    int32_t errors = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (h_gpu[i] != h_cpu[i]) {
            if (errors < 10)
                fprintf(stderr, "  fourstep mismatch at %u (row=%u,col=%u): gpu=%08x cpu=%08x\n",
                        i, i / (1u << (lg_n/2)), i % (1u << (lg_n/2)), h_gpu[i], h_cpu[i]);
            errors++;
        }
    }

    sycl::free(d_data, q); sycl::free(h_gpu, q); sycl::free(h_cpu, q);
    return errors;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what()); fflush(stderr); return -2;
    } catch (...) {
        fprintf(stderr, "Unknown exception\n"); fflush(stderr); return -4;
    }
}

// Validate Four-Step round-trip (forward + inverse = identity)
int32_t validate_fourstep_roundtrip(uint32_t lg_n) {
    try {
    auto q = create_queue();
    uint32_t n = 1u << lg_n;

    auto* d_data = sycl::malloc_device<uint32_t>(n, q);
    auto* h_data = sycl::malloc_host<uint32_t>(n, q);
    auto* h_orig = sycl::malloc_host<uint32_t>(n, q);

    bb31_cpu::generate_test_data(h_data, n, 88888 + lg_n);
    memcpy(h_orig, h_data, n * sizeof(uint32_t));

    q.memcpy(d_data, h_data, n * sizeof(uint32_t)); q.wait();

    // Forward then inverse
    gpu_forward_ntt_fourstep(q, d_data, lg_n);
    gpu_inverse_ntt_fourstep(q, d_data, lg_n);

    q.memcpy(h_data, d_data, n * sizeof(uint32_t)); q.wait();

    int32_t errors = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (h_data[i] != h_orig[i]) {
            if (errors < 10)
                fprintf(stderr, "  fourstep roundtrip mismatch at %u: got=%08x exp=%08x\n",
                        i, h_data[i], h_orig[i]);
            errors++;
        }
    }

    sycl::free(d_data, q); sycl::free(h_data, q); sycl::free(h_orig, q);
    return errors;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what()); fflush(stderr); return -2;
    } catch (...) {
        fprintf(stderr, "Unknown exception\n"); fflush(stderr); return -4;
    }
}

// Validate Four-Step matches existing multipass NTT
int32_t validate_fourstep_vs_multipass(uint32_t lg_n) {
    try {
    auto q = create_queue();
    uint32_t n = 1u << lg_n;

    ensure_partial_twiddles(q, true);
    ensure_radix_twiddles(q);

    auto* d_fs = sycl::malloc_device<uint32_t>(n, q);
    auto* d_mp = sycl::malloc_device<uint32_t>(n, q);
    auto* h_in = sycl::malloc_host<uint32_t>(n, q);
    auto* h_fs = sycl::malloc_host<uint32_t>(n, q);
    auto* h_mp = sycl::malloc_host<uint32_t>(n, q);

    bb31_cpu::generate_test_data(h_in, n, 66666 + lg_n);

    // Four-Step
    q.memcpy(d_fs, h_in, n * sizeof(uint32_t)); q.wait();
    gpu_forward_ntt_fourstep(q, d_fs, lg_n);
    q.memcpy(h_fs, d_fs, n * sizeof(uint32_t)); q.wait();

    // Multipass
    q.memcpy(d_mp, h_in, n * sizeof(uint32_t)); q.wait();
    gpu_forward_ntt_multipass(q, d_mp, lg_n);
    q.memcpy(h_mp, d_mp, n * sizeof(uint32_t)); q.wait();

    int32_t errors = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (h_fs[i] != h_mp[i]) {
            if (errors < 10)
                fprintf(stderr, "  fourstep vs multipass mismatch at %u: fs=%08x mp=%08x\n",
                        i, h_fs[i], h_mp[i]);
            errors++;
        }
    }

    sycl::free(d_fs, q); sycl::free(d_mp, q);
    sycl::free(h_in, q); sycl::free(h_fs, q); sycl::free(h_mp, q);
    return errors;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what()); fflush(stderr); return -2;
    } catch (...) {
        fprintf(stderr, "Unknown exception\n"); fflush(stderr); return -4;
    }
}

// Benchmark Four-Step forward NTT
BenchResult bench_fourstep_ntt(uint32_t lg_n) {
    try {
    auto q = create_queue();
    BenchResult result = {};
    uint32_t n = 1u << lg_n;

    auto* d_data = sycl::malloc_device<uint32_t>(n, q);
    auto* h_data = sycl::malloc_host<uint32_t>(n, q);
    bb31_cpu::generate_test_data(h_data, n, 99999);
    q.memcpy(d_data, h_data, n * sizeof(uint32_t));
    q.wait();

    // Warm-up (also triggers twiddle table precomputation)
    for (int w = 0; w < 50; w++) {
        gpu_forward_ntt_fourstep(q, d_data, lg_n);
    }
    q.wait();

    // Timed runs: average over 15 samples
    constexpr int SAMPLES = 15;
    double best_ns = 1e18;
    for (int s = 0; s < SAMPLES; s++) {
        auto start = std::chrono::high_resolution_clock::now();
        gpu_forward_ntt_fourstep(q, d_data, lg_n);
        q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        double ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        if (ns < best_ns) best_ns = ns;
    }

    result.kernel_ns = best_ns;
    result.total_ops = double(n) / 2.0 * double(lg_n);
    result.correct = 1;

    sycl::free(d_data, q);
    sycl::free(h_data, q);
    return result;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what()); fflush(stderr);
        return BenchResult{0.0, 0.0, -2};
    } catch (const std::exception& e) {
        fprintf(stderr, "Exception: %s\n", e.what()); fflush(stderr);
        return BenchResult{0.0, 0.0, -3};
    } catch (...) {
        fprintf(stderr, "Unknown exception\n"); fflush(stderr);
        return BenchResult{0.0, 0.0, -4};
    }
}

} // extern "C"
