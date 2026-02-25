#include <cstring>
#include <iostream>

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

#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#include "msm/pippenger.cuh"

namespace sppark::bn254 {
#include "ntt/ntt.cuh"
} // namespace sppark::bn254
using namespace sppark::bn254;

#include "util.cuh"

#include "groth16_coeffs.cuh"
#include "groth16_prover.cuh"
#include "groth16_srs.cuh"

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

static SRS* g_cached_srs = nullptr;
static groth16_prover* g_cached_prover = nullptr;
static std::string g_cached_srs_path;

static void ensure_initialized(SetupParams* setup_params) {
  std::string path(setup_params->srs_path);
  if (g_cached_srs && g_cached_srs_path == path) return;
  delete g_cached_prover; g_cached_prover = nullptr;
  delete g_cached_srs; g_cached_srs = nullptr;
  g_cached_srs = new SRS(0, setup_params->srs_path);
  g_cached_srs_path = std::move(path);
  g_cached_prover = new groth16_prover(*g_cached_srs,
      setup_params->pcoeffs_path, setup_params->fres_path);
}

extern "C" const char* risc0_groth16_cuda_init(SetupParams* setup_params) {
  try {
    ensure_initialized(setup_params);
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// Raw proof output: 8 field elements (a.x, a.y, c.x, c.y, b[0..3]) in
// non-Montgomery little-endian form. Layout matches groth16_proof struct.
struct RawProofOutput {
  uint8_t data[256]; // 8 × 32 bytes
};

extern "C" const char* risc0_groth16_cuda_prove(SetupParams* setup_params,
                                                ProveParams* prover_params) {

  try {
    ensure_initialized(setup_params);
    groth16_proof proof = g_cached_prover->prove(prover_params->public_path, prover_params->witness);
    write_proof_file(prover_params->proof_path, proof);
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

extern "C" const char* risc0_groth16_cuda_prove_raw(SetupParams* setup_params,
                                                    ProveParams* prover_params,
                                                    RawProofOutput* raw_out) {
  try {
    ensure_initialized(setup_params);
    groth16_proof proof = g_cached_prover->prove(prover_params->public_path, prover_params->witness);

    // Convert from Montgomery form to standard and copy raw bytes.
    // Use the same named union as write_proof_file to access individual fp_t values.
    union proof_and_fp_t {
      groth16_proof proof;
      struct { fp_t a[2], c[2], b[4]; };
    };
    proof_and_fp_t u = proof_and_fp_t{proof};

    // Convert all 8 field elements from Montgomery to standard form
    for (int i = 0; i < 2; i++) { u.a[i].to(); }
    for (int i = 0; i < 2; i++) { u.c[i].to(); }
    for (int i = 0; i < 4; i++) { u.b[i].to(); }

    memcpy(raw_out->data, &u, 256);
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
