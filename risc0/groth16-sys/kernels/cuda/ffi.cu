#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>

#define FEATURE_BN254

#include "ff/alt_bn128-fp2.hpp"

#include "ec/affine_t.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbitwise-instead-of-logical"
#include "ec/jacobian_t.hpp"
#pragma clang diagnostic pop

#include "ec/xyzz_t.hpp"

typedef Affine_t<fp_t> affine_t;
typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;

typedef Affine_t<fp2_t> affine_fp2_t;
typedef jacobian_t<fp2_t> point_fp2_t;
typedef xyzz_t<fp2_t> bucket_fp2_t;

typedef fr_t scalar_t;

#ifndef __HIPCC__
#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#endif
#include "msm/pippenger.cuh"

#ifdef __HIPCC__
// G2 MSM kernel explicit instantiations (G1 handled by pippenger.cuh defaults).
// All four template arguments are specified explicitly so that the mangled names
// match the call sites in msm_t<bucket_fp2_t, point_fp2_t, affine_fp2_t, fr_t>.
template __global__
void accumulate<bucket_fp2_t, affine_fp2_t::mem_t,
                bucket_fp2_t::mem_t, affine_fp2_t>(
    bucket_fp2_t::mem_t buckets_[],
    uint32_t nwins, uint32_t wbits,
    /*const*/ affine_fp2_t::mem_t points_[],
    const vec2d_t<uint32_t> digits,
    const vec2d_t<uint32_t> histogram,
    uint32_t* counter);
template __global__
void batch_addition<bucket_fp2_t, affine_fp2_t::mem_t,
                    bucket_fp2_t::mem_t, affine_fp2_t>(
    bucket_fp2_t::mem_t buckets[],
    const affine_fp2_t::mem_t points[], size_t npoints,
    const uint32_t digits[], const uint32_t& ndigits);
template __global__
void integrate<bucket_fp2_t, bucket_fp2_t::mem_t>(
    bucket_fp2_t::mem_t buckets_[], uint32_t nwins,
    uint32_t wbits, uint32_t nbits);
template __global__
void reduce_rows<bucket_fp2_t, bucket_fp2_t::mem_t>(
    bucket_fp2_t::mem_t buckets_[], uint32_t nwins,
    uint32_t wbits, uint32_t nbits,
    uint32_t thr_per_sub);
#endif

namespace sppark::bn254 {
#include "ntt/ntt.cuh"
} // namespace sppark::bn254
using namespace sppark::bn254;

#include "util.cuh"

#include "groth16_coeffs.cuh"
#include "groth16_srs.cuh"
#include "groth16_prover.cuh"

#ifdef __HIPCC__
// Explicit template instantiations for HIP device pass.
// The groth16_prover host code is hidden from the device pass (it uses
// blst-specific methods), so template __global__ functions called from it
// must be explicitly instantiated here.

// NTT kernels (inside the sppark::bn254 namespace where ntt.cuh was included)
namespace sppark::bn254 {
template __global__
void _GS_NTT<0, fr_t>(const unsigned int, const unsigned int,
    const unsigned int, const unsigned int,
    fr_t*, const fr_t (*)[WINDOW_SIZE],
    const fr_t*, const fr_t*, const fr_t*,
    const unsigned int, const bool, const fr_t,
    const unsigned int);
template __global__
void _GS_NTT<1, fr_t>(const unsigned int, const unsigned int,
    const unsigned int, const unsigned int,
    fr_t*, const fr_t (*)[WINDOW_SIZE],
    const fr_t*, const fr_t*, const fr_t*,
    const unsigned int, const bool, const fr_t,
    const unsigned int);
template __global__
void _GS_NTT<2, fr_t>(const unsigned int, const unsigned int,
    const unsigned int, const unsigned int,
    fr_t*, const fr_t (*)[WINDOW_SIZE],
    const fr_t*, const fr_t*, const fr_t*,
    const unsigned int, const bool, const fr_t,
    const unsigned int);
template __global__
void _CT_NTT<0, fr_t>(const unsigned int, const unsigned int,
    const unsigned int, const unsigned int,
    fr_t*, const fr_t (*)[WINDOW_SIZE],
    const fr_t*, const fr_t*, const fr_t*,
    const unsigned int, const bool, const fr_t,
    const unsigned int);
template __global__
void _CT_NTT<1, fr_t>(const unsigned int, const unsigned int,
    const unsigned int, const unsigned int,
    fr_t*, const fr_t (*)[WINDOW_SIZE],
    const fr_t*, const fr_t*, const fr_t*,
    const unsigned int, const bool, const fr_t,
    const unsigned int);
template __global__
void _CT_NTT<2, fr_t>(const unsigned int, const unsigned int,
    const unsigned int, const unsigned int,
    fr_t*, const fr_t (*)[WINDOW_SIZE],
    const fr_t*, const fr_t*, const fr_t*,
    const unsigned int, const bool, const fr_t,
    const unsigned int);
template __global__
void bit_rev_permutation<fr_t>(fr_t*, const fr_t*, uint32_t);
template __global__
void LDE_distribute_powers<fr_t>(fr_t*, uint32_t, uint32_t, bool,
    const fr_t (*)[WINDOW_SIZE], const unsigned int);
template __global__
void generate_partial_twiddles<fr_t>(fr_t (*)[WINDOW_SIZE], const fr_t);
template __global__
void generate_all_twiddles<fr_t>(fr_t*, const fr_t);
template __global__
void generate_radixX_twiddles_X<fr_t>(fr_t*, int, const fr_t);
} // namespace sppark::bn254

// Utility kernels
template __global__
void chacha_generate_random_scalars<8, fr_t>(fr_t*, const chacha_state, size_t);
#endif // __HIPCC__

#if !defined(__HIP_DEVICE_COMPILE__)
struct SetupParams {
  const char* pcoeffs_path;
  const char* fres_path;
  const char* srs_path;
};

struct ProveParams {
  const char* public_path;
  const char* proof_path;
  const fr_t* witness;
};

// Cached SRS + prover to avoid reloading ~200MB of point data and
// reinitializing GPU memory on every Groth16 prove call.
// Preloaded on a background thread via risc0_groth16_cuda_preload().
static std::unique_ptr<SRS> cached_srs;
static std::unique_ptr<groth16_prover> cached_prover;
static std::mutex cached_mutex;

extern "C" const char* risc0_groth16_cuda_preload(SetupParams* setup_params) {
  try {
    std::lock_guard<std::mutex> lock(cached_mutex);
    if (!cached_srs) {
      cached_srs = std::make_unique<SRS>(0, setup_params->srs_path);
    }
    if (!cached_prover) {
      cached_prover = std::make_unique<groth16_prover>(
          *cached_srs, setup_params->pcoeffs_path, setup_params->fres_path);
    }
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

extern "C" const char* risc0_groth16_cuda_prove(SetupParams* setup_params,
                                                ProveParams* prover_params) {

  try {
    // Release cached async memory pool allocations from prior GPU work (e.g.
    // STARK proving) so that the Groth16 prover has enough VRAM.
    // Only needed on first call (before SRS/prover are cached).
    {
      std::lock_guard<std::mutex> lock(cached_mutex);
      if (!cached_prover) {
        cudaMemPool_t pool;
        if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess)
          cudaMemPoolTrimTo(pool, 0);
      }

      if (!cached_srs) {
        cached_srs = std::make_unique<SRS>(0, setup_params->srs_path);
      }
      if (!cached_prover) {
        cached_prover = std::make_unique<groth16_prover>(
            *cached_srs, setup_params->pcoeffs_path, setup_params->fres_path);
      }
    }

    groth16_proof proof = cached_prover->prove(
        prover_params->public_path, prover_params->witness);
    write_proof_file(prover_params->proof_path, proof);
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

#ifdef SRS_READ_COEFFS

extern "C" const char* risc0_groth16_cuda_setup(SetupParams* params) {
  try {
    SRS sw(0, params->srs_path);
    groth16_prover prover(sw);
    prover.write_precomputations_to_file(params->fres_path, params->pcoeffs_path);
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

#endif // SRS_READ_COEFFS
#endif // !__HIP_DEVICE_COMPILE__
