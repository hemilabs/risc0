// Amalgamation of all eval_check files + ffi_supra into a single compilation unit.
// This enables the compiler to inline the deep poly_fp call chain (keccak_0 through keccak_47),
// dramatically reducing per-thread stack usage from ~82KB to ~8KB.
// Without this, separate compilation (--device-c) prevents cross-TU inlining,
// causing CUDA OOM at po2=18 (1M threads * 82KB stack = 82GB > VRAM).

#include "eval_check_0.cu"
#include "eval_check_1.cu"
#include "eval_check_2.cu"
#include "eval_check_3.cu"
#include "eval_check_4.cu"
#include "ffi_supra.cu"
