// FRI and polynomial HAL kernels for Intel GPU STARK prover.
// Implements: fri_fold, mix_poly_coeffs, batch_evaluate_any, combos_prepare, fri_prove.
// Uses the "4 × Vec16" FpExt layout (each lane = independent value).

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include "bb31_field.hpp"
#include "bb31_fpext4.hpp"

namespace esimd = sycl::ext::intel::esimd;
using namespace fpext4;

static constexpr uint32_t kFriFold = 16;
static constexpr uint32_t kFriFoldBits = 4; // log2(kFriFold)

// Bit-reverse a value within nbits bits (scalar)
static inline uint32_t bit_rev_scalar(uint32_t val, uint32_t nbits) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < nbits; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

static sycl::queue create_fri_queue() {
    auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    for (auto& d : gpu_devices) {
        auto name = d.get_info<sycl::info::device::name>();
        if (name.find("Intel") != std::string::npos ||
            name.find("0xe2") != std::string::npos)
            return sycl::queue(d, sycl::property_list{sycl::property::queue::in_order{}});
    }
    throw std::runtime_error("Intel GPU not found");
}

extern "C" {

// Forward declarations for scalar FpExt helpers (defined later, used by fri_fold precomputation)
static inline void scalar_ext_mul(uint32_t* r, const uint32_t* a, const uint32_t* b);

// ============================================================================
// fri_fold: FRI folding — accumulate kFriFold=16 FpExt values with mix powers
// Input: planar FpExt layout: in[comp * count * kFriFold + rev_i * count + idx]
// Output: planar FpExt layout: out[comp * count + idx]
// Each thread processes one output index.
// ============================================================================
void esimd_fri_fold(sycl::queue& q,
                     uint32_t* d_out, const uint32_t* d_in,
                     const uint32_t* d_mix, // 4 u32 values = 1 FpExt (Montgomery)
                     uint32_t count) {
    // Precompute mix^0..mix^15 on HOST to break the serial dependency chain
    // in the kernel's inner loop. This eliminates 16 fpext_mul(curMix, mix) calls.
    uint32_t h_mix[4];
    auto* h_tmp = sycl::malloc_host<uint32_t>(4, q);
    q.memcpy(h_tmp, d_mix, 16); q.wait();
    for (int c = 0; c < 4; c++) h_mix[c] = h_tmp[c];
    sycl::free(h_tmp, q);

    uint32_t h_powers[kFriFold * 4]; // 16 FpExt values = 256 bytes
    // mix^0 = [ONE, 0, 0, 0]
    h_powers[0] = bb31::ONE; h_powers[1] = 0; h_powers[2] = 0; h_powers[3] = 0;
    // mix^1..mix^15 via scalar multiply
    for (uint32_t i = 1; i < kFriFold; i++) {
        uint32_t prev[4] = {h_powers[(i-1)*4], h_powers[(i-1)*4+1], h_powers[(i-1)*4+2], h_powers[(i-1)*4+3]};
        scalar_ext_mul(h_powers + i*4, prev, h_mix);
    }

    auto* d_powers = sycl::malloc_device<uint32_t>(kFriFold * 4, q);
    q.memcpy(d_powers, h_powers, kFriFold * 4 * 4); q.wait();

    auto* pout = d_out;
    auto* pin = d_in;
    auto* ppow = d_powers;

    // SIMD16: each thread processes 16 consecutive output indices.
    // Mix powers precomputed — no serial dependency in inner loop.
    uint32_t num_threads = (count + 15) / 16;

    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;

            FpExt4 tot;     // zero-initialized

            for (uint32_t fi = 0; fi < kFriFold; fi++) {
                uint32_t rev_fi = bit_rev_scalar(fi, kFriFoldBits);
                uint32_t rev_base = rev_fi * count + base;

                // Load precomputed mix^fi (broadcast to all 16 lanes)
                FpExt4 curMix(bb31::Vec16(*(ppow + fi*4 + 0)),
                              bb31::Vec16(*(ppow + fi*4 + 1)),
                              bb31::Vec16(*(ppow + fi*4 + 2)),
                              bb31::Vec16(*(ppow + fi*4 + 3)));

                FpExt4 factor;
                if (base + 16 <= count) {
                    factor.c[0] = esimd::block_load<uint32_t, 16>(pin + 0*count*kFriFold + rev_base);
                    factor.c[1] = esimd::block_load<uint32_t, 16>(pin + 1*count*kFriFold + rev_base);
                    factor.c[2] = esimd::block_load<uint32_t, 16>(pin + 2*count*kFriFold + rev_base);
                    factor.c[3] = esimd::block_load<uint32_t, 16>(pin + 3*count*kFriFold + rev_base);
                } else {
                    factor = FpExt4();
                    for (uint32_t lane = 0; lane < 16 && base + lane < count; lane++) {
                        uint32_t ri = rev_fi * count + base + lane;
                        factor.c[0][lane] = *(pin + 0*count*kFriFold + ri);
                        factor.c[1][lane] = *(pin + 1*count*kFriFold + ri);
                        factor.c[2][lane] = *(pin + 2*count*kFriFold + ri);
                        factor.c[3][lane] = *(pin + 3*count*kFriFold + ri);
                    }
                }

                // No serial curMix *= mix needed — mix^fi is precomputed!
                tot = fpext_add(tot, fpext_mul(curMix, factor));
            }

            // Store results: block_store for 16 consecutive planar values
            if (base + 16 <= count) {
                esimd::block_store(pout + 0*count + base, tot.c[0]);
                esimd::block_store(pout + 1*count + base, tot.c[1]);
                esimd::block_store(pout + 2*count + base, tot.c[2]);
                esimd::block_store(pout + 3*count + base, tot.c[3]);
            } else {
                for (uint32_t lane = 0; lane < 16 && base + lane < count; lane++) {
                    *(pout + 0*count + base + lane) = tot.c[0][lane];
                    *(pout + 1*count + base + lane) = tot.c[1][lane];
                    *(pout + 2*count + base + lane) = tot.c[2][lane];
                    *(pout + 3*count + base + lane) = tot.c[3][lane];
                }
            }
        });
    q.wait();
    sycl::free(d_powers, q);
}

// ============================================================================
// mix_poly_coeffs: Mix polynomial coefficients into combo groups
// For each coefficient idx, iterate over inputSize polynomials,
// accumulating cur * in[count*i + idx] into out[count*combos[i] + idx]
// ============================================================================
void esimd_mix_poly_coeffs(sycl::queue& q,
                            uint32_t* d_out,       // FpExt output (4 u32 per element)
                            const uint32_t* d_in,  // Fp input coefficients
                            const uint32_t* d_combos, // combo index per polynomial
                            const uint32_t* d_mix_start, // 4 u32 = initial FpExt mix power
                            const uint32_t* d_mix,       // 4 u32 = FpExt mix base
                            uint32_t inputSize,
                            uint32_t count) {
    auto* pout = d_out;
    auto* pin = d_in;
    auto* pcombos = d_combos;
    auto* pms = d_mix_start;
    auto* pm = d_mix;

    // SIMD16: each thread processes 16 consecutive coefficient indices
    uint32_t num_threads = (count + 15) / 16;

    q.parallel_for(sycl::range<1>(num_threads),
        [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
            uint32_t base = idx[0] * 16;

            // Load mix parameters (broadcast — same for all 16 lanes)
            FpExt4 cur(bb31::Vec16(*(pms+0)), bb31::Vec16(*(pms+1)),
                       bb31::Vec16(*(pms+2)), bb31::Vec16(*(pms+3)));
            FpExt4 mix(bb31::Vec16(*(pm+0)), bb31::Vec16(*(pm+1)),
                       bb31::Vec16(*(pm+2)), bb31::Vec16(*(pm+3)));

            for (uint32_t p = 0; p < inputSize; p++) {
                uint32_t combo_id = *(pcombos + p);

                // Load 16 consecutive Fp coefficients via block_load
                bb31::Vec16 coeff;
                if (base + 16 <= count) {
                    coeff = esimd::block_load<uint32_t, 16>(pin + count * p + base);
                } else {
                    coeff = bb31::Vec16(0u);
                    for (uint32_t lane = 0; lane < 16 && base + lane < count; lane++)
                        coeff[lane] = *(pin + count * p + base + lane);
                }

                // cur * coeff (FpExt * Fp) — all 16 lanes compute independently
                FpExt4 term = fpext_mul_fp(cur, coeff);

                // Accumulate into out — AoS layout, scatter for non-contiguous FpExt elements
                // Each lane writes to (count * combo_id + base + lane) * 4 + c
                if (base + 16 <= count) {
                    for (uint32_t c = 0; c < 4; c++) {
                        // Gather old values
                        esimd::simd<uint32_t, 16> lane(0u, 1u);
                        auto byte_off = ((count * combo_id + base + lane) * 4 + c) * (uint32_t)sizeof(uint32_t);
                        auto old = esimd::gather<uint32_t, 16>(pout, byte_off);
                        auto sum = bb31::field_add(old, term.c[c]);
                        esimd::scatter<uint32_t, 16>(pout, byte_off, sum);
                    }
                } else {
                    for (uint32_t lane = 0; lane < 16 && base + lane < count; lane++) {
                        uint32_t out_base = (count * combo_id + base + lane) * 4;
                        for (uint32_t c = 0; c < 4; c++) {
                            uint32_t old = *(pout + out_base + c);
                            uint32_t s = old + term.c[c][lane];
                            *(pout + out_base + c) = s >= bb31::MOD ? s - bb31::MOD : s;
                        }
                    }
                }

                cur = fpext_mul(cur, mix);
            }
        });
    q.wait();
}

// ============================================================================
// combos_prepare: Single-threaded combo polynomial subtraction
// Subtracts U polynomial contributions from combo polynomial coefficients.
// ============================================================================
// combos_prepare: runs on HOST (single-threaded, same as CUDA <<<1,1>>>)
// Uses USM shared memory so host can directly access device data
void esimd_combos_prepare(sycl::queue& q,
                           uint32_t* d_combos,
                           const uint32_t* d_coeffU,
                           uint32_t comboCount,
                           uint32_t cycles,
                           uint32_t regsCount,
                           const uint32_t* d_regSizes,
                           const uint32_t* d_regComboIds,
                           uint32_t checkSize,
                           const uint32_t* d_mix) {
    // Copy small control data to host for scalar processing
    auto* h_regSizes = sycl::malloc_host<uint32_t>(regsCount, q);
    auto* h_regComboIds = sycl::malloc_host<uint32_t>(regsCount, q);
    uint32_t h_mix[4];
    q.memcpy(h_regSizes, d_regSizes, regsCount * 4);
    q.memcpy(h_regComboIds, d_regComboIds, regsCount * 4);
    q.memcpy(h_mix, d_mix, 16); q.wait();

    // Compute total coeffU size
    uint32_t total_u = 0;
    for (uint32_t i = 0; i < regsCount; i++) total_u += h_regSizes[i];
    total_u += checkSize;

    // Copy combo and coeffU data to host
    uint32_t combo_total = (comboCount + 1) * cycles * 4;
    auto* h_combos = sycl::malloc_host<uint32_t>(combo_total, q);
    auto* h_coeffU = sycl::malloc_host<uint32_t>(total_u * 4, q);
    q.memcpy(h_combos, d_combos, combo_total * 4);
    q.memcpy(h_coeffU, d_coeffU, total_u * 4 * 4); q.wait();

    // Scalar Montgomery helpers
    auto smul = [](uint32_t a, uint32_t b) -> uint32_t {
        uint64_t prod = (uint64_t)a * b;
        uint32_t lo = (uint32_t)prod, hi = (uint32_t)(prod >> 32);
        uint32_t red = lo * 0x77ffffffu;
        uint64_t rprod = (uint64_t)red * bb31::MOD;
        uint32_t rlo = (uint32_t)rprod, rhi = (uint32_t)(rprod >> 32);
        uint32_t carry = ((uint64_t)lo + rlo) >= (1ULL << 32) ? 1 : 0;
        uint32_t res = hi + rhi + carry;
        return res >= bb31::MOD ? res - bb31::MOD : res;
    };
    auto sadd = [](uint32_t a, uint32_t b) -> uint32_t {
        uint32_t r = a + b; return r >= bb31::MOD ? r - bb31::MOD : r;
    };
    auto ssub = [](uint32_t a, uint32_t b) -> uint32_t {
        return a >= b ? a - b : a + bb31::MOD - b;
    };

    uint32_t cur[4] = {bb31::ONE, 0, 0, 0};
    uint32_t pos = 0;

    auto ext_mul_mix = [&]() {
        uint32_t nb = bb31::NBETA;
        uint32_t r0 = sadd(smul(cur[0],h_mix[0]), smul(nb, sadd(sadd(smul(cur[1],h_mix[3]),smul(cur[2],h_mix[2])),smul(cur[3],h_mix[1]))));
        uint32_t r1 = sadd(sadd(smul(cur[0],h_mix[1]),smul(cur[1],h_mix[0])), smul(nb, sadd(smul(cur[2],h_mix[3]),smul(cur[3],h_mix[2]))));
        uint32_t r2 = sadd(sadd(sadd(smul(cur[0],h_mix[2]),smul(cur[1],h_mix[1])),smul(cur[2],h_mix[0])), smul(nb, smul(cur[3],h_mix[3])));
        uint32_t r3 = sadd(sadd(sadd(smul(cur[0],h_mix[3]),smul(cur[1],h_mix[2])),smul(cur[2],h_mix[1])), smul(cur[3],h_mix[0]));
        cur[0]=r0; cur[1]=r1; cur[2]=r2; cur[3]=r3;
    };

    auto process_coeff = [&](uint32_t combo_idx, uint32_t j) {
        uint32_t u[4];
        for (int c = 0; c < 4; c++) u[c] = h_coeffU[(pos+j)*4+c];
        uint32_t nb = bb31::NBETA;
        uint32_t p[4];
        p[0] = sadd(smul(cur[0],u[0]), smul(nb, sadd(sadd(smul(cur[1],u[3]),smul(cur[2],u[2])),smul(cur[3],u[1]))));
        p[1] = sadd(sadd(smul(cur[0],u[1]),smul(cur[1],u[0])), smul(nb, sadd(smul(cur[2],u[3]),smul(cur[3],u[2]))));
        p[2] = sadd(sadd(sadd(smul(cur[0],u[2]),smul(cur[1],u[1])),smul(cur[2],u[0])), smul(nb, smul(cur[3],u[3])));
        p[3] = sadd(sadd(sadd(smul(cur[0],u[3]),smul(cur[1],u[2])),smul(cur[2],u[1])), smul(cur[3],u[0]));
        uint32_t base = (cycles * combo_idx + j) * 4;
        for (int c = 0; c < 4; c++) h_combos[base+c] = ssub(h_combos[base+c], p[c]);
    };

    for (uint32_t rg = 0; rg < regsCount; rg++) {
        uint32_t regSize = h_regSizes[rg];
        uint32_t regComboId = h_regComboIds[rg];
        for (uint32_t j = 0; j < regSize; j++) process_coeff(regComboId, j);
        ext_mul_mix();
        pos += regSize;
    }
    for (uint32_t ci = 0; ci < checkSize; ci++) {
        process_coeff(comboCount, 0); // always j=0: CUDA subtracts from combos[cycles*comboCount] each time
        pos++;
        ext_mul_mix();
    }

    // Copy result back to device
    q.memcpy(d_combos, h_combos, combo_total * 4); q.wait();
    sycl::free(h_regSizes, q); sycl::free(h_regComboIds, q);
    sycl::free(h_combos, q); sycl::free(h_coeffU, q);
}

// ============================================================================
// Scalar FpExt helpers for single-thread kernels (batch_evaluate_any, combos_prepare)
// ============================================================================
static inline uint32_t scalar_mont_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * b;
    uint32_t lo = (uint32_t)prod, hi = (uint32_t)(prod >> 32);
    uint32_t red = lo * 0x77ffffffu;
    uint64_t rprod = (uint64_t)red * bb31::MOD;
    uint32_t rlo = (uint32_t)rprod, rhi = (uint32_t)(rprod >> 32);
    uint32_t carry = ((uint64_t)lo + rlo) >= (1ULL << 32) ? 1 : 0;
    uint32_t res = hi + rhi + carry;
    return res >= bb31::MOD ? res - bb31::MOD : res;
}
static inline uint32_t scalar_field_add(uint32_t a, uint32_t b) {
    uint32_t r = a + b; return r >= bb31::MOD ? r - bb31::MOD : r;
}
static inline void scalar_ext_mul(uint32_t* r, const uint32_t* a, const uint32_t* b) {
    uint32_t nb = bb31::NBETA;
    r[0] = scalar_field_add(scalar_mont_mul(a[0],b[0]), scalar_mont_mul(nb, scalar_field_add(scalar_field_add(scalar_mont_mul(a[1],b[3]),scalar_mont_mul(a[2],b[2])),scalar_mont_mul(a[3],b[1]))));
    r[1] = scalar_field_add(scalar_field_add(scalar_mont_mul(a[0],b[1]),scalar_mont_mul(a[1],b[0])), scalar_mont_mul(nb, scalar_field_add(scalar_mont_mul(a[2],b[3]),scalar_mont_mul(a[3],b[2]))));
    r[2] = scalar_field_add(scalar_field_add(scalar_field_add(scalar_mont_mul(a[0],b[2]),scalar_mont_mul(a[1],b[1])),scalar_mont_mul(a[2],b[0])), scalar_mont_mul(nb, scalar_mont_mul(a[3],b[3])));
    r[3] = scalar_field_add(scalar_field_add(scalar_field_add(scalar_mont_mul(a[0],b[3]),scalar_mont_mul(a[1],b[2])),scalar_mont_mul(a[2],b[1])), scalar_mont_mul(a[3],b[0]));
}
static inline void scalar_ext_pow(uint32_t* result, const uint32_t* base_in, uint32_t n) {
    uint32_t b[4], tmp[4];
    for (int c=0;c<4;c++) { result[c]=(c==0?bb31::ONE:0); b[c]=base_in[c]; }
    while (n > 0) {
        if (n & 1) { scalar_ext_mul(tmp, result, b); for (int c=0;c<4;c++) result[c]=tmp[c]; }
        scalar_ext_mul(tmp, b, b); for (int c=0;c<4;c++) b[c]=tmp[c];
        n >>= 1;
    }
}

// ============================================================================
// batch_evaluate_any: Evaluate polynomials at arbitrary FpExt points
// SIMD16: each workgroup evaluates 16 polynomials simultaneously (one per lane).
// 64 threads per WG, each thread strides through coefficients.
// SLM tree reduction across 64 threads, with FpExt4 (Vec16) values.
// ============================================================================
void esimd_batch_evaluate_any(sycl::queue& q,
                               uint32_t* d_out,       // FpExt output (4 u32 per eval, AoS)
                               const uint32_t* d_coeffs, // Fp coefficients (flat)
                               const uint32_t* d_which,  // polynomial index per eval
                               const uint32_t* d_xs,     // FpExt eval points (4 u32 per point, AoS)
                               uint32_t count,            // number of evaluations
                               uint32_t deg) {            // degree (coefficients per poly)
    if (count == 0) return;

    constexpr uint32_t WG_SIZE = 64;
    // SLM: 64 threads × 4 FpExt components × 16 lanes × 4 bytes = 16KB
    constexpr uint32_t SLM_SIZE = WG_SIZE * 4 * 16 * sizeof(uint32_t);

    // Each workgroup handles 16 evaluations. Total WGs = ceil(count/16).
    uint32_t num_groups = (count + 15) / 16;

    auto* pout = d_out;
    auto* pcoeffs = d_coeffs;
    auto* pwhich = d_which;
    auto* pxs = d_xs;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
                esimd::slm_init<SLM_SIZE>();
                uint32_t lid = item.get_local_id(0);
                uint32_t grp = item.get_group(0);
                uint32_t eval_base = grp * 16; // first of 16 evaluations for this WG

                // Load which[] and xs[] for 16 evaluations (one per lane)
                // Gather: each lane reads from a different evaluation's data
                esimd::simd<uint32_t, 16> lane(0u, 1u); // {0,1,...,15}
                esimd::simd<uint32_t, 16> eval_idx = eval_base + lane;

                // Build per-lane polynomial base offset: which[eval_idx] * deg
                // Guard: lanes beyond count get poly_idx=0 (harmless, result discarded)
                esimd::simd<uint32_t, 16> poly_off;
                {
                    auto which_byte = eval_idx * (uint32_t)sizeof(uint32_t);
                    auto valid = eval_idx < count;
                    auto which_vals = esimd::gather<uint32_t, 16>(pwhich, which_byte, valid,
                                                                   esimd::simd<uint32_t, 16>(0u));
                    poly_off = which_vals * deg;
                }

                // Load 16 FpExt evaluation points (AoS: 4 contiguous u32 per eval)
                FpExt4 x_val;
                {
                    auto xs_byte = eval_idx * 4 * (uint32_t)sizeof(uint32_t);
                    auto valid = eval_idx < count;
                    x_val.c[0] = esimd::gather<uint32_t, 16>(pxs, xs_byte, valid, esimd::simd<uint32_t, 16>(0u));
                    x_val.c[1] = esimd::gather<uint32_t, 16>(pxs, xs_byte + 4, valid, esimd::simd<uint32_t, 16>(0u));
                    x_val.c[2] = esimd::gather<uint32_t, 16>(pxs, xs_byte + 8, valid, esimd::simd<uint32_t, 16>(0u));
                    x_val.c[3] = esimd::gather<uint32_t, 16>(pxs, xs_byte + 12, valid, esimd::simd<uint32_t, 16>(0u));
                }

                // Compute stepx = x^WG_SIZE and powx = x^lid (per-lane, independent)
                FpExt4 stepx = fpext_pow(x_val, WG_SIZE);
                FpExt4 powx = fpext_pow(x_val, lid);

                // Strided polynomial evaluation: each thread processes coeff[lid, lid+64, lid+128, ...]
                FpExt4 tot; // zero-initialized
                for (uint32_t i = lid; i < deg; i += WG_SIZE) {
                    // Gather coefficient from 16 different polynomials
                    // coeff[lane] = pcoeffs[poly_off[lane] + i]
                    auto coeff_byte = (poly_off + i) * (uint32_t)sizeof(uint32_t);
                    auto coeff = esimd::gather<uint32_t, 16>(pcoeffs, coeff_byte);

                    // FpExt * Fp: powx * coeff (component-wise)
                    FpExt4 term = fpext_mul_fp(powx, coeff);
                    tot = fpext_add(tot, term);

                    // Advance power: powx *= stepx
                    powx = fpext_mul(powx, stepx);
                }

                // Store FpExt4 to SLM: each thread writes 4 Vec16 values
                // Layout: SLM[lid * 4 * 16 * 4 + c * 16 * 4 + lane * 4]
                // Simplified: thread lid, component c → SLM offset (lid * 64 + c * 16) * 4 bytes
                uint32_t slm_base = lid * 64; // in u32 units: lid * 4 components * 16 lanes
                for (int c = 0; c < 4; c++)
                    esimd::slm_block_store<uint32_t, 16>((slm_base + c * 16) * 4, tot.c[c]);
                esimd::barrier();

                // Tree reduction in SLM (64 threads → 1)
                for (uint32_t stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        uint32_t my_base = lid * 64;
                        uint32_t partner_base = (lid + stride) * 64;
                        for (int c = 0; c < 4; c++) {
                            auto a = esimd::slm_block_load<uint32_t, 16>((my_base + c * 16) * 4);
                            auto b = esimd::slm_block_load<uint32_t, 16>((partner_base + c * 16) * 4);
                            esimd::slm_block_store<uint32_t, 16>((my_base + c * 16) * 4,
                                                                  bb31::field_add(a, b));
                        }
                    }
                    esimd::barrier();
                }

                // Thread 0 writes 16 evaluation results (AoS: 4 contiguous u32 per eval)
                if (lid == 0) {
                    FpExt4 result;
                    for (int c = 0; c < 4; c++)
                        result.c[c] = esimd::slm_block_load<uint32_t, 16>(c * 16 * 4);

                    // Scatter to AoS output: out[eval_idx * 4 + c]
                    auto valid = eval_idx < count;
                    for (int c = 0; c < 4; c++) {
                        auto out_byte = (eval_idx * 4 + c) * (uint32_t)sizeof(uint32_t);
                        esimd::scatter<uint32_t, 16>(pout, out_byte, result.c[c], valid);
                    }
                }
            });
    });
    q.wait();
}

// ============================================================================
// Validation
// ============================================================================
int32_t validate_fri_poly_ops() {
    try {
    auto q = create_fri_queue();

    // Test fri_fold with simple data
    constexpr uint32_t COUNT = 4;
    // Input: 4 * kFriFold * COUNT Fp values (planar FpExt)
    constexpr uint32_t IN_SIZE = 4 * kFriFold * COUNT;
    auto* d_in = sycl::malloc_device<uint32_t>(IN_SIZE, q);
    auto* d_out = sycl::malloc_device<uint32_t>(4 * COUNT, q);
    auto* d_mix = sycl::malloc_device<uint32_t>(4, q);
    auto* h_in = sycl::malloc_host<uint32_t>(IN_SIZE, q);
    auto* h_out = sycl::malloc_host<uint32_t>(4 * COUNT, q);

    // Initialize input: FpExt = [ONE, 0, 0, 0] in planar layout
    // Planar: comp 0 = ONE for all, comp 1-3 = 0
    for (uint32_t i = 0; i < kFriFold * COUNT; i++) h_in[0 * kFriFold * COUNT + i] = bb31::ONE;
    for (uint32_t i = 0; i < kFriFold * COUNT; i++) h_in[1 * kFriFold * COUNT + i] = 0;
    for (uint32_t i = 0; i < kFriFold * COUNT; i++) h_in[2 * kFriFold * COUNT + i] = 0;
    for (uint32_t i = 0; i < kFriFold * COUNT; i++) h_in[3 * kFriFold * COUNT + i] = 0;
    // Mix = [ONE, 0, 0, 0] (= scalar 1 in extension field)
    uint32_t h_mix[4] = {bb31::ONE, 0, 0, 0};
    q.memcpy(d_in, h_in, IN_SIZE * 4);
    q.memcpy(d_mix, h_mix, 4 * 4); q.wait();

    esimd_fri_fold(q, d_out, d_in, d_mix, COUNT);
    q.memcpy(h_out, d_out, 4 * COUNT * 4); q.wait();

    // With mix=[1,0,0,0] and all inputs=[1,0,0,0], each output should be
    // sum of 16 * [1,0,0,0] = [16,0,0,0] in extension field
    // 16 in Montgomery = host_mont_encode(16)
    uint32_t sixteen_mont = (uint32_t)(((uint64_t)16 << 32) % bb31::MOD);
    bool fold_ok = true;
    for (uint32_t i = 0; i < COUNT; i++) {
        if (h_out[0 * COUNT + i] != sixteen_mont || h_out[1 * COUNT + i] != 0 ||
            h_out[2 * COUNT + i] != 0 || h_out[3 * COUNT + i] != 0) {
            fprintf(stderr, "  fri_fold mismatch at %u: [%08x,%08x,%08x,%08x] exp [%08x,0,0,0]\n",
                    i, h_out[i], h_out[COUNT+i], h_out[2*COUNT+i], h_out[3*COUNT+i], sixteen_mont);
            fold_ok = false; break;
        }
    }
    fprintf(stderr, "  fri_fold: %s\n", fold_ok ? "PASS" : "FAIL");

    sycl::free(d_in, q); sycl::free(d_out, q); sycl::free(d_mix, q);
    sycl::free(h_in, q); sycl::free(h_out, q);

    return fold_ok ? 0 : -1;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what()); fflush(stderr);
        return -99;
    }
}

} // extern "C"
