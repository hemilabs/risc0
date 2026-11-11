// Copyright 2026 RISC Zero, Inc.
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

#pragma once

/// \file
/// Preflight trace types (port of kernels/cuda/preflight.h; Intel precedent intel/preflight.h).
/// `PreflightCycle` and `MemoryTransaction` are the device copies of Rust's `RawPreflightCycle`
/// (36 bytes) and `RawMemoryTransaction` (20 bytes) in rv32im-sys/src/lib.rs, memcpy'd into
/// shared buffers by the HAL: the sizes and every field offset are asserted here and, on the Rust
/// side, by rv32im-sys/tests/r1_layout.rs (which also parses the RV32IM_ASSERT_* lines below).
/// `PreflightTrace` is not shared with the host: CUDA's pointer struct becomes a thread-local
/// struct that a kernel builds from its buffer arguments (plan 2.0, explicit bindings).

#include "cuda_compat.h"

namespace rv32im_metal {

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

// Checked against rv32im-sys/src/lib.rs by tests/r1_layout.rs (keep one assert per line).
RV32IM_ASSERT_SIZE(MemoryTransaction, 20);
RV32IM_ASSERT_OFFSET(MemoryTransaction, addr, 0);
RV32IM_ASSERT_OFFSET(MemoryTransaction, cycle, 4);
RV32IM_ASSERT_OFFSET(MemoryTransaction, word, 8);
RV32IM_ASSERT_OFFSET(MemoryTransaction, prevCycle, 12);
RV32IM_ASSERT_OFFSET(MemoryTransaction, prevWord, 16);
RV32IM_ASSERT_SIZE(PreflightCycle, 36);
RV32IM_ASSERT_OFFSET(PreflightCycle, state, 0);
RV32IM_ASSERT_OFFSET(PreflightCycle, pc, 4);
RV32IM_ASSERT_OFFSET(PreflightCycle, major, 8);
RV32IM_ASSERT_OFFSET(PreflightCycle, minor, 9);
RV32IM_ASSERT_OFFSET(PreflightCycle, machineMode, 10);
RV32IM_ASSERT_OFFSET(PreflightCycle, padding, 11);
RV32IM_ASSERT_OFFSET(PreflightCycle, userCycle, 12);
RV32IM_ASSERT_OFFSET(PreflightCycle, txnIdx, 16);
RV32IM_ASSERT_OFFSET(PreflightCycle, pagingIdx, 20);
RV32IM_ASSERT_OFFSET(PreflightCycle, bigintIdx, 24);
RV32IM_ASSERT_OFFSET(PreflightCycle, diffCount, 28);

/// Thread-local view of the uploaded preflight trace. `cycles` is writable: witgen advances
/// `cycles[c].txnIdx` (extern_getMemoryTxn), which is why the HAL copies the trace into its own
/// buffer (plan 2.6). The lengths let the extern implementations clamp indices taken from the
/// trace (plan 2.9).
struct PreflightTrace {
  device PreflightCycle* cycles;
  const device MemoryTransaction* txns;
  const device uint8_t* bigintBytes;
  uint32_t txnsLen;
  uint32_t bigintBytesLen;
  uint32_t tableSplitCycle;
  uint32_t cyclesLen; // entries in cycles (getDiffCount's clamp; Metal only, not in CUDA)
};

} // namespace rv32im_metal
