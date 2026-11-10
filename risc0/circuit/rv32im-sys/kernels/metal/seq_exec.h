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
/// Sequential (single-thread) witgen drivers, chunked (port of fwd_stepExec / rev_stepExec in
/// kernels/cuda/ffi.cu:710-724, plan 4.2 "port tiny 1x1 dispatch, chunked per 4096 cycles").
///
/// CUDA runs all cycles of a segment in one 1x1 launch. A single GPU thread stepping millions of
/// cycles would run for seconds, so on Metal the host splits the sequence into chunks of
/// RV32IM_SEQ_CHUNK positions and encodes one 1x1 dispatch per chunk, in order, into the same
/// compute encoder (dispatches on hazard-tracked resources in a serial encoder run in encoding
/// order). Each dispatch walks positions [begin, end) of the sequence:
///   forward: position p is cycle p                                  (fwd_stepExec)
///   reverse: positions [0, split) are cycles split-1 .. 0, then positions [split, lastCycle) are
///            cycles lastCycle-1 .. split                            (rev_stepExec)
/// so the chunks together visit exactly CUDA's order.
///
/// The kernels themselves live in the TU that defines the step function (R5's witgen_tu.metal),
/// as `fwd_stepExec` / `rev_stepExec`, written as:
///   kernel void fwd_stepExec(..., constant SeqParams& p [[buffer(k)]], uint gid [[...]]) {
///     if (gid != 0) return;
///     <Step> step{...};               // functor: void operator()(uint32_t cycle) thread
///     rv32im_seq_fwd(step, p);
///   }
/// misc.metal instantiates both drivers with a probe step (seq_fwd_probe / seq_rev_probe) so the
/// chunking is testable before the real step function exists.

#include "cuda_compat.h"

/// Sequence positions per dispatch (mirror on the host).
#define RV32IM_SEQ_CHUNK 4096u

/// Params of a chunked sequential dispatch: positions [begin, end) of the sequence; `split` and
/// `lastCycle` define the reverse order (forward ignores them). 16 bytes.
struct SeqParams {
  uint32_t begin;
  uint32_t end;
  uint32_t split;
  uint32_t lastCycle;
};
RV32IM_ASSERT_SIZE(SeqParams, 16);

/// Cycle at position `pos` of the reverse order (CUDA rev_stepExec).
inline uint32_t rv32im_rev_cycle(uint32_t pos, uint32_t split, uint32_t lastCycle) {
  return pos < split ? split - 1 - pos : lastCycle - 1 - (pos - split);
}

template <typename Step> inline void rv32im_seq_fwd(thread Step& step, constant SeqParams& p) {
  for (uint32_t pos = p.begin; pos < p.end; pos++) {
    step(pos);
  }
}

template <typename Step> inline void rv32im_seq_rev(thread Step& step, constant SeqParams& p) {
  for (uint32_t pos = p.begin; pos < p.end; pos++) {
    step(rv32im_rev_cycle(pos, p.split, p.lastCycle));
  }
}
