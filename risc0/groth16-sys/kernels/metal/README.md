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
- `bn254_fast.h` (item G9) adds faster formulations next to `bn254.h` (whose functions and
  semantics stay as they are for Z10): Montgomery multiplication with 32x32->64-bit
  multiply-accumulates (`mul_wide`, gnark's no-carry `mul_nc`), the field types `FpF`/`FrF`/`Fp2L`
  (same layout and values as `Fp`/`Fr`/`Fp2`) that the `_g1w`/`_g2w` MSM and `_w` NTT kernels
  instantiate. `g9_bench.metal` holds the micro-benchmarks of the variants (tests.rs
  `g9_fast_field_ops`), including the rejected ones (see docs/metal-port/notes/G9.md).
- `witness_poly.metal`'s `witness_into_poly_compact` (item G10) reads the compact coefficients
  (`(s, value index)` pairs into a table of distinct values, per-row output indices; built and
  disk-cached by `src/metal/compact.rs`); same summation order as `witness_into_poly`, so
  bit-identical.
- With no `.metal` file present the metallib is an empty file and the name list is empty.
