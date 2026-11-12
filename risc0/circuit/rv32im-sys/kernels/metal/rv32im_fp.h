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
/// Fp for the rv32im Metal headers: the shared fp.h, or the TU's own copy when it defines
/// RV32IM_FP_HEADER (R17b: witgen_tu.metal uses witgen_fp.h, the shared header with every
/// function always_inline, written by kernels/cuda/witgen_tu_gen.py gen_fp). The rv32im headers
/// include this instead of fp.h, so both copies are never in one TU.
#ifdef RV32IM_FP_HEADER
#include RV32IM_FP_HEADER
#else
#include "fp.h"
#endif
