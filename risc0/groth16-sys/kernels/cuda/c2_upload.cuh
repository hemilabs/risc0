#pragma once
// C2 prototype: host -> device upload of mmap'd proving-key sections through pinned staging.
// A pageable cudaMemcpyAsync holds a driver-wide lock for its whole duration (measured: other
// threads' cuMemAlloc / cuMemcpyHtoD / cudaStreamCreate block on pthread_rwlock_wrlock for
// 90-490 ms each), so a background load stalls any concurrent CUDA work (identity_p254).
// Here worker threads copy chunks into pinned buffers and each chunk is uploaded with a short
// async copy from pinned memory on the stager's own non-blocking streams; chunk buffers are
// recycled through events.
//
// C2-R2: one process-wide stager (pinned buffers allocated once, 128 MB), used by the
// background key load (`risc0_groth16_cuda_prestart`) and, with RISC0_G16_STAGED, by preload.
// Callers serialize on cached_mutex.
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// C2-R2 switches that default on; "0" disables.
static bool g16_env_on(const char* name) { const char* e = getenv(name); return !(e && e[0] == '0'); }

static std::atomic<bool> g16_force_staged{false};

static bool g16_staged() {
  static bool v = getenv("RISC0_G16_STAGED") != nullptr;
  return v || g16_force_staged.load();
}

class c2_stager {
  static constexpr size_t CH = 16 << 20, ALN = 4096;
  static constexpr int NTHR = 4, NB = 2;
  int nthr_used = NTHR;
  char* bufs[NTHR * NB] = {};
  cudaEvent_t ev[NTHR * NB] = {};
  cudaStream_t st[NTHR] = {};

public:
  c2_stager() {
    for (int i = 0; i < NTHR * NB; i++) {
      CUDA_OK(cudaMallocHost((void**)&bufs[i], CH + 2 * ALN));
      CUDA_OK(cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming));
    }
    for (int t = 0; t < NTHR; t++)
      CUDA_OK(cudaStreamCreateWithFlags(&st[t], cudaStreamNonBlocking));
    if (const char* e = getenv("RISC0_G16_STAGE_THREADS")) {
      int v = atoi(e);
      if (v >= 1 && v <= NTHR) nthr_used = v;
    }
  }
  // Never destroyed (process lifetime, like the cached SRS): avoids teardown ordering issues.
  static c2_stager& get() {
    static c2_stager* s = new c2_stager();
    return *s;
  }
  // Copies `bytes` from host `src` to device `dst`; returns when the data is on the device
  // (every chunk's copy has completed, so `src` may be unmapped afterwards). With `path` and
  // RISC0_G16_STAGE_ODIRECT=1, chunks are read from the file at `file_off` with O_DIRECT
  // straight into the pinned buffers (no CPU copy out of the page cache).
  void upload(void* dst, const void* src, size_t bytes, const char* path = nullptr, size_t file_off = 0) {
    static bool odirect = getenv("RISC0_G16_STAGE_ODIRECT") != nullptr;
    int fd = -1;
    if (odirect && path) fd = open(path, O_RDONLY | O_DIRECT);
    static double rate = getenv("RISC0_G16_STAGE_RATE_MBPS") ? atof(getenv("RISC0_G16_STAGE_RATE_MBPS")) * 1e6 : 0;
    static std::atomic<size_t> sent{0};
    static auto t_start = std::chrono::steady_clock::now();
    size_t nch = (bytes + CH - 1) / CH;
    std::atomic<size_t> next{0};
    std::atomic<int> err{0};
    std::vector<std::thread> th;
    for (int t = 0; t < nthr_used; t++)
      th.emplace_back([&, t] {
        int k = 0;
        for (size_t c; (c = next.fetch_add(1)) < nch; k++) {
          int b = t * NB + (k % NB);
          // The buffer's previous copy (if any) must be done before it is overwritten.
          if (cudaEventSynchronize(ev[b]) != cudaSuccess) err = 1;
          size_t o = c * CH, n = std::min(CH, bytes - o);
          const char* from = bufs[b];
          if (fd >= 0) {
            size_t fo = file_off + o, a0 = fo & ~(ALN - 1), a1 = (fo + n + ALN - 1) & ~(ALN - 1);
            size_t got = 0;
            while (got < a1 - a0) {
              ssize_t r = pread(fd, bufs[b] + got, a1 - a0 - got, a0 + got);
              if (r <= 0) break;
              got += r;
            }
            if (got < fo + n - a0) { err = 2; continue; }
            from = bufs[b] + (fo - a0);
          } else {
            memcpy(bufs[b], (const char*)src + o, n);
          }
          if (cudaMemcpyAsync((char*)dst + o, from, n, cudaMemcpyHostToDevice, st[t]) != cudaSuccess)
            err = 1;
          if (cudaEventRecord(ev[b], st[t]) != cudaSuccess) err = 1;
          if (rate > 0) {  // throttle: keep the process-wide upload rate under `rate` bytes/s
            double due = (double)(sent.fetch_add(n) + n) / rate;
            auto t = t_start + std::chrono::duration<double>(due);
            std::this_thread::sleep_until(t);
          }
        }
        if (cudaStreamSynchronize(st[t]) != cudaSuccess) err = 1;
      });
    for (auto& x : th)
      x.join();
    if (fd >= 0) close(fd);
    if (err)
      throw sppark_error{EIO, "staged upload failed: "};
  }
};
