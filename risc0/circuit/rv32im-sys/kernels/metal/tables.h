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
/// u8/u16 lookup-table counters (port of kernels/cuda/tables.h; Intel precedent intel/tables.h,
/// which uses explicit relaxed device-scope atomic refs). Counts are relaxed 32-bit atomic adds,
/// as on CUDA (integer adds commute, so the totals do not depend on the order).
///
/// Differences from CUDA, all on invalid input only: CUDA `assert`s (a no-op in release device
/// code) and then indexes anyway; here a bad table id or an out-of-range index sets a bit in
/// `bad` (RV32IM_FAULT_LOOKUP_*) and touches no memory (lookupCurrent returns 0 for it). The kernel
/// ORs `bad` into its fault buffer once (plan 2.9). As on CUDA and the CPU, `count` is ignored:
/// every lookupDelta adds exactly 1.

#include "cuda_compat.h"
#include "rv32im_fp.h"

namespace rv32im_metal {

struct LookupTables {
  device metal::atomic_uint* tableU8;   // 256 counters
  device metal::atomic_uint* tableU16;  // 65536 counters
  uint32_t bad;                         // RV32IM_FAULT_LOOKUP_* bits seen by this thread

  void lookupDelta(Fp table, Fp index, Fp count) thread {
    uint32_t tableU32 = table.asUInt32();
    uint32_t indexU32 = index.asUInt32();
    if (tableU32 == 0) {
      // tableCycle[index] += count;
      return;
    }
    if (tableU32 != 8 && tableU32 != 16) {
      bad |= RV32IM_FAULT_LOOKUP_TABLE;
      return;
    }
    if (indexU32 >= (1u << tableU32)) {
      bad |= RV32IM_FAULT_LOOKUP_INDEX;
      return;
    }
    device metal::atomic_uint* t = tableU32 == 8 ? tableU8 : tableU16;
    metal::atomic_fetch_add_explicit(t + indexU32, 1u, metal::memory_order_relaxed);
  }

  /// Read after the counting dispatch finished (CUDA bypasses L1 with __ldcg; on Metal the
  /// dispatch boundary makes the counts visible, the atomic load just keeps the type honest).
  Fp lookupCurrent(Fp table, Fp index) thread {
    uint32_t tableU32 = table.asUInt32();
    if (tableU32 != 8 && tableU32 != 16) {
      bad |= RV32IM_FAULT_LOOKUP_TABLE;
      return Fp(0);
    }
    uint32_t indexU32 = index.asUInt32();
    if (indexU32 >= (1u << tableU32)) {
      bad |= RV32IM_FAULT_LOOKUP_INDEX;
      return Fp(0);
    }
    device metal::atomic_uint* t = tableU32 == 8 ? tableU8 : tableU16;
    return Fp(metal::atomic_load_explicit(t + indexU32, metal::memory_order_relaxed));
  }
};

inline LookupTables makeLookupTables(device metal::atomic_uint* tableU8,
                                     device metal::atomic_uint* tableU16) {
  LookupTables t;
  t.tableU8 = tableU8;
  t.tableU16 = tableU16;
  t.bad = 0;
  return t;
}

} // namespace rv32im_metal
