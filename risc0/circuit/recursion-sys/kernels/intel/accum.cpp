// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// GPU recursion accum: step_compute_accum -> inclusive prefix product ->
// step_verify_accum, on the shared in-order queue. Same results as the CPU
// risc0_circuit_recursion_cpu_accum (FpExt multiplication is exact, so the
// blocked scan gives the same bits as the sequential inclusive scan).
//
// build.rs splices the generated kernels/cxx/step_compute_accum.cpp and
// step_verify_accum.cpp (minus their extern.h/ffi.h includes) in at the
// marker below.
#define NDEBUG 1
#include <sycl/sycl.hpp>

#include <cstring>
#include <mutex>

#include "fp.h"
#include "fpext.h"

using namespace risc0;

namespace risc0::circuit::recursion {
struct DevAccumCtx {
  FpExt* accum;
};
inline void extern_plonkWriteAccum_wom(void* ctx, size_t cycle, const char*, std::array<Fp, 4> a) {
  static_cast<DevAccumCtx*>(ctx)->accum[cycle] = FpExt(a[0], a[1], a[2], a[3]);
}
inline std::array<Fp, 4> extern_plonkReadAccum_wom(void* ctx, size_t cycle, const char*, std::array<Fp, 0>) {
  const FpExt& v = static_cast<DevAccumCtx*>(ctx)->accum[cycle];
  return {v.elems[0], v.elems[1], v.elems[2], v.elems[3]};
}
} // namespace risc0::circuit::recursion

// @RECURSION_ACCUM_STEPS@

using namespace risc0::circuit::recursion;

namespace {
constexpr uint32_t kBlk = 512;
struct Scratch {
  FpExt* tmp = nullptr;
  FpExt* blk = nullptr;
  DevAccumCtx* dctx = nullptr;
  uint32_t cap = 0;
};
Scratch g;
std::mutex g_mu;
} // namespace

extern "C" const char* risc0_circuit_recursion_intel_accum(void* queue_ptr,
                                                           void* ctrl_,
                                                           void* global_,
                                                           void* data_,
                                                           void* mix_,
                                                           void* accum_,
                                                           uint32_t work,
                                                           uint32_t total) {
  try {
    std::lock_guard<std::mutex> lk(g_mu);
    auto& q = *static_cast<sycl::queue*>(queue_ptr);
    Fp* ctrl = static_cast<Fp*>(ctrl_);
    Fp* global = static_cast<Fp*>(global_);
    Fp* data = static_cast<Fp*>(data_);
    Fp* mix = static_cast<Fp*>(mix_);
    Fp* accum = static_cast<Fp*>(accum_);
    if (work == 0) {
      return nullptr;
    }
    if (work > total) {
      throw std::runtime_error("recursion accum: work_cycles > total_cycles");
    }
    // Scratch is sized for the full domain (work <= total) so that it is
    // reallocated only when the po2 grows. Before freeing, drain the queue:
    // sycl::free does not wait for kernels that may still use the buffers.
    if (total > g.cap) {
      q.wait();
      if (g.tmp) {
        sycl::free(g.tmp, q);
        sycl::free(g.blk, q);
      }
      g.tmp = sycl::malloc_device<FpExt>(total, q);
      g.blk = sycl::malloc_device<FpExt>(total / kBlk + 2, q);
      if (!g.dctx) {
        g.dctx = sycl::malloc_device<DevAccumCtx>(1, q);
      }
      if (!g.tmp || !g.blk || !g.dctx) {
        throw std::runtime_error("recursion accum: device allocation failed");
      }
      DevAccumCtx h{g.tmp};
      q.memcpy(g.dctx, &h, sizeof(h)).wait();
      g.cap = total;
    }
    FpExt* tmp = g.tmp;
    FpExt* blk = g.blk;
    DevAccumCtx* dctx = g.dctx;
    uint32_t nblk = (work + kBlk - 1) / kBlk;

    q.fill(tmp, FpExt(Fp(1)), work);
    const size_t wg = 256;
    size_t gs = (work + wg - 1) / wg * wg;
    q.parallel_for(sycl::nd_range<1>(gs, wg), [=](sycl::nd_item<1> it) {
      size_t cycle = it.get_global_id(0);
      if (cycle >= work)
        return;
      Fp* args[5] = {ctrl, global, data, mix, accum};
      step_compute_accum(dctx, total, cycle, args);
    });
    // Inclusive prefix product: per-block scan, exclusive scan of the block
    // totals, multiply-in.
    q.parallel_for(sycl::range<1>(nblk), [=](sycl::id<1> b) {
      uint32_t lo = b[0] * kBlk, hi = sycl::min(lo + kBlk, work);
      FpExt acc = tmp[lo];
      for (uint32_t i = lo + 1; i < hi; i++) {
        acc = acc * tmp[i];
        tmp[i] = acc;
      }
      blk[b[0]] = acc;
    });
    q.single_task([=]() {
      FpExt acc(Fp(1));
      for (uint32_t b = 0; b < nblk; b++) {
        FpExt t = blk[b];
        blk[b] = acc;
        acc = acc * t;
      }
    });
    q.parallel_for(sycl::range<1>(work), [=](sycl::id<1> i) {
      uint32_t b = i[0] / kBlk;
      if (b)
        tmp[i[0]] = blk[b] * tmp[i[0]];
    });
    q.parallel_for(sycl::nd_range<1>(gs, wg), [=](sycl::nd_item<1> it) {
      size_t cycle = it.get_global_id(0);
      if (cycle >= work)
        return;
      Fp* args[5] = {ctrl, global, data, mix, accum};
      step_verify_accum(dctx, total, cycle, args);
    });
    // No wait: the in-order queue orders the following zeroize/commit work.
  } catch (const std::exception& e) {
    return strdup(e.what());
  }
  return nullptr;
}
