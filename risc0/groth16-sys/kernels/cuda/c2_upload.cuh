#pragma once
// Host -> device upload of mmap'd proving-key sections through pinned staging (CUDA).
// A pageable cudaMemcpyAsync holds a driver-wide lock for its whole duration (measured: other
// threads' cuMemAlloc / cuMemcpyHtoD / cudaStreamCreate block on pthread_rwlock_wrlock for
// 90-490 ms each), so a background load stalls any concurrent CUDA work (the STARK prover).
// Here worker threads copy chunks into pinned buffers and each chunk is uploaded with a short
// async copy from pinned memory on the stager's own non-blocking streams; chunk buffers are
// recycled through events.
//
// One process-wide stager (pinned buffers allocated once, 128 MB), used by the background key
// load (`risc0_groth16_cuda_prestart`) and, after it, by later loads. Callers serialize on
// cached_mutex.
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// The Groth16 cold-path optimizations (background key load, compact coefficients, bit-split
// MSMs, tuned MSM batching): on unless RISC0_CUDA_G16_COLD=0 (also read by the Rust side).
static bool g16_cold_enabled() {
  static const bool on = [] {
    const char* e = getenv("RISC0_CUDA_G16_COLD");
    return !(e && e[0] == '0');
  }();
  return on;
}

// Set once a background key load has run: from then on key uploads use the stager.
static std::atomic<bool> g16_staged_upload{false};

static bool g16_staged() { return g16_staged_upload.load(); }

class c2_stager {
  static constexpr size_t CH = 16 << 20;
  static constexpr int NTHR = 4, NB = 2;
  char* bufs[NTHR * NB] = {};
  cudaEvent_t ev[NTHR * NB] = {};
  cudaStream_t st[NTHR] = {};

public:
  c2_stager() {
    for (int i = 0; i < NTHR * NB; i++) {
      CUDA_OK(cudaMallocHost((void**)&bufs[i], CH));
      CUDA_OK(cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming));
    }
    for (int t = 0; t < NTHR; t++)
      CUDA_OK(cudaStreamCreateWithFlags(&st[t], cudaStreamNonBlocking));
  }
  // Never destroyed (process lifetime, like the cached SRS): avoids teardown ordering issues.
  static c2_stager& get() {
    static c2_stager* s = new c2_stager();
    return *s;
  }
  // Copies `bytes` from host `src` to device `dst`; returns when the data is on the device
  // (every chunk's copy has completed, so `src` may be unmapped afterwards).
  void upload(void* dst, const void* src, size_t bytes) {
    size_t nch = (bytes + CH - 1) / CH;
    std::atomic<size_t> next{0};
    std::atomic<int> err{0};
    std::vector<std::thread> th;
    for (int t = 0; t < NTHR; t++)
      th.emplace_back([&, t] {
        int k = 0;
        for (size_t c; (c = next.fetch_add(1)) < nch; k++) {
          int b = t * NB + (k % NB);
          // The buffer's previous copy (if any) must be done before it is overwritten.
          if (cudaEventSynchronize(ev[b]) != cudaSuccess) err = 1;
          size_t o = c * CH, n = std::min(CH, bytes - o);
          memcpy(bufs[b], (const char*)src + o, n);
          if (cudaMemcpyAsync((char*)dst + o, bufs[b], n, cudaMemcpyHostToDevice, st[t]) != cudaSuccess)
            err = 1;
          if (cudaEventRecord(ev[b], st[t]) != cudaSuccess) err = 1;
        }
        if (cudaStreamSynchronize(st[t]) != cudaSuccess) err = 1;
      });
    for (auto& x : th)
      x.join();
    if (err)
      throw sppark_error{EIO, "staged upload failed: "};
  }
};
