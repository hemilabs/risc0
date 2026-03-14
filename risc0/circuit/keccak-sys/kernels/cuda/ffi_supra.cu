// Copyright 2026 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT

#include "eval_check.cuh"

#include "cuda.h"
#include "supra/fp.h"

#include <exception>

namespace risc0::circuit::keccak::cuda {

__global__ __launch_bounds__(256, 1) void eval_check(Fp* check,
                           const Fp* ctrl,
                           const Fp* data,
                           const Fp* accum,
                           const Fp* mix,
                           const Fp* out,
                           const Fp rou,
                           uint32_t po2,
                           uint32_t domain,
                           const FpExt* poly_mix) {
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x; cycle < domain; cycle += stride) {
    FpExt tot = poly_fp(cycle, domain, ctrl, out, data, mix, accum, poly_mix);
    Fp x = pow(rou, cycle);
    Fp y = pow(Fp(3) * x, 1 << po2);
    FpExt ret = tot * inv(y - Fp(1));
    check[domain * 0 + cycle] = ret[0];
    check[domain * 1 + cycle] = ret[1];
    check[domain * 2 + cycle] = ret[2];
    check[domain * 3 + cycle] = ret[3];
  }
}

} // namespace risc0::circuit::keccak::cuda

extern "C" {

using namespace risc0::circuit::keccak::cuda;

const char* risc0_circuit_keccak_cuda_eval_check(Fp* check,
                                                 const Fp* ctrl,
                                                 const Fp* data,
                                                 const Fp* accum,
                                                 const Fp* mix,
                                                 const Fp* out,
                                                 const Fp& rou,
                                                 uint32_t po2,
                                                 uint32_t domain,
                                                 const uint32_t* poly_mix_pows) {
  try {
    // Use grid-stride loop with capped grid to limit local memory allocation.
    // eval_check has ~30KB stack per thread; launching all domain threads would
    // exceed VRAM at po2=18 (1M threads * 30KB = 30GB). Cap to SM count.
    int smCount = 0;
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    CUDA_OK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    int grid = smCount;  // 1 block per SM (launch_bounds(256,1))
    eval_check<<<grid, 256>>>(
        check, ctrl, data, accum, mix, out, rou, po2, domain, (const FpExt*)poly_mix_pows);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

} // extern "C"
