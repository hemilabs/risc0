#pragma once

// A pinned host copy of a file, read once per process (ROCm and CUDA on Windows).
//
// The Groth16 prover's proving key (the zkey's points and the preprocessed coefficients, ~4.7
// GB) is released after every proof on a card that can't keep it beside the next STARK, so
// every wrap reloads it. From the file (read or mapped, so pageable) that reload ran ~1.5 s
// on an RX 9070 XT; from this copy the uploads are DMA. Returns nullptr when the copy can't be
// made (or RISC0_GROTH16_PINNED_KEY=0), and the callers fall back to the file.
//
// On CUDA it is opt-in: RISC0_GROTH16_PINNED_KEY=1 makes the copy when a file is loaded a
// second time, i.e. when the key was released between proofs (the zk-miner worker does on cards
// below 20 GiB). The ~5 GB of page-locked host memory stays held for the life of the process,
// per worker, which a host with less RAM (or other GPU processes that need pinned memory)
// cannot always spare; =first pins on the first load too.

#if defined(_WIN32) && !defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__HIPCC__) || !defined(__CUDA_ARCH__))
#define GROTH16_PINNED_FILES 1

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <sys/stat.h>

static const unsigned char* groth16_pinned_file(const char* path, size_t* size) {
  static const char* const env = std::getenv("RISC0_GROTH16_PINNED_KEY");
  static const bool enabled = !(env && std::strcmp(env, "0") == 0);
  if (!enabled)
    return nullptr;
  static std::mutex m;
  static std::map<std::string, std::pair<unsigned char*, size_t>> cache;
  std::lock_guard<std::mutex> lock(m);
  auto it = cache.find(path);
  if (it != cache.end()) {
    *size = it->second.second;
    return it->second.first;
  }
#ifndef __HIPCC__
  // CUDA: off unless asked for (see above).
  static const bool cuda_opt_in = env && (std::strcmp(env, "1") == 0 || std::strcmp(env, "first") == 0);
  static const bool cuda_first = env && std::strcmp(env, "first") == 0;
  if (!cuda_opt_in)
    return nullptr;
  static std::map<std::string, int> loads;
  if (++loads[path] < 2 && !cuda_first)
    return nullptr;
#endif
  std::FILE* file = std::fopen(path, "rb");
  if (!file)
    return nullptr;
  struct _stat64 st;
  if (_fstat64(_fileno(file), &st) != 0 || st.st_size <= 0) {
    std::fclose(file);
    return nullptr;
  }
  size_t len = (size_t)st.st_size;
  void* buf = nullptr;
  if (cudaMallocHost(&buf, len) != cudaSuccess || !buf) {
    (void)cudaGetLastError();
    std::fclose(file);
    std::fprintf(stderr, "groth16: could not pin %zu bytes for %s, reading it pageable\n", len, path);
    return nullptr;
  }
  size_t got = std::fread(buf, 1, len, file);
  std::fclose(file);
  if (got != len) {
    cudaFreeHost(buf);
    return nullptr;
  }
  if (std::getenv("RISC0_VERBOSE"))
    std::fprintf(stderr, "groth16: pinned %zu bytes for %s\n", len, path);
  cache[path] = {(unsigned char*)buf, len};
  *size = len;
  return (const unsigned char*)buf;
}
#endif
