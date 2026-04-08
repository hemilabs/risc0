// Intel SYCL version of preflight.h — identical struct layout to CUDA/CXX.
#pragma once
#include <cstdint>

namespace risc0::circuit::rv32im_v2::intel {

struct MemoryTransaction {
  uint32_t addr;
  uint32_t cycle;
  uint32_t word;
  uint32_t prevCycle;
  uint32_t prevWord;
};

struct PreflightCycle {
  uint32_t state;
  uint32_t pc;
  uint8_t major;
  uint8_t minor;
  uint8_t machineMode;
  uint8_t padding;
  uint32_t userCycle;
  uint32_t txnIdx;
  uint32_t pagingIdx;
  uint32_t bigintIdx;
  uint32_t diffCount[2];
};

struct PreflightTrace {
  PreflightCycle* cycles;
  MemoryTransaction* txns;
  uint8_t* bigintBytes;
  uint32_t txnsLen;
  uint32_t bigintBytesLen;
  uint32_t tableSplitCycle;
};

} // namespace risc0::circuit::rv32im_v2::intel
