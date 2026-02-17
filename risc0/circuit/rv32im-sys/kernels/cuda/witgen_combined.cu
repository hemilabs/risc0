// Combined compilation of all witgen device functions + kernels into a single
// translation unit, compiled WITHOUT -dc (separate compilation). This allows
// NVCC to inline the 210 device functions in the step_Top / step_TopAccum call
// chains, eliminating cross-TU function calls with full ABI overhead (register
// save/restore, parameter passing through stack memory).
//
// Same technique as eval_check_combined.cu.

// Include all device function definitions first
#include "steps.cu"

// Then include the kernels and host wrappers
#include "ffi.cu"
