# Groth16 Metal kernels

Hand-written MSL for the Metal Groth16 prover (docs/metal-port/PLAN.md section 4.5). The Intel
SYCL prover in `../intel` is the template.

- Every `*.metal` file directly in this directory is compiled by `build.rs`
  (`build_metal_kernels`) into `metal_kernels_groth16.metallib`, embedded as
  `risc0_groth16_sys::METAL_LIB`. Adding a file needs no build-script change.
- `risc0_groth16_sys::METAL_KERNEL_NAMES` is generated from the sources: each
  `kernel void NAME(...)` outside a template, and each explicit instantiation's
  `[[host_name("NAME")]]`. Names must be unique across the directory (the build fails otherwise).
- Headers (`bn254.h`, `ntt_params.h`, ...) live here too; every file in this directory is a
  dependency of every source, so editing a header recompiles the library. Shared field headers
  (`fp.h`, `cuda2metal.h`, ...) come from risc0-build-kernel's `METAL_INCS`; never add
  `kernels/cuda` to the include path.
- Conventions (plan 2.1-2.4): global-scope kernels (no namespace), explicit `[[buffer(n)]]`
  indices documented at the top of each file, `[[max_total_threads_per_threadgroup(N)]]` before
  `kernel void`, `Fp(0)` rather than `= 0`, threadgroup memory at most 32768 bytes (asserted on
  the host), BN254 arithmetic in 32-bit limbs.
- With no `.metal` file present the metallib is an empty file and the name list is empty.
