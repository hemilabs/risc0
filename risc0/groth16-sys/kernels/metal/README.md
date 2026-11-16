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
- `msm_window.metal` (item G12) holds `msm_window_reduce_*`, compiled with every function
  always_inline: with out-of-line point additions in its loops (the R17 shape) the G2 kernel
  returned wrong window sums under concurrent GPU load. `msm_reduce.metal` keeps the out-of-line
  additions of the bucket and segment kernels (inlining them was 8 % slower and 2.7x the cold
  pipeline compile; no wrong result was found in them).
- Item G12 (docs/metal-port/notes/G12.md): `msm_sort.metal`'s `msm_len_*` kernels list each
  window's pieces longest first after the sort, and `msm_reduce.metal`'s `msm_buckets_perm_*`
  sum them in that order (the SIMD lanes of a group then add about as many points each).
  `bn254_g12.h` (lazy-reduction Fp2 product, mulhi REDC step) and `g12_bench.metal` (those, and
  a batch-affine accumulation probe) are measured and rejected variants, used only by the tests
  `g12_fast_field_ops` and `g12_batch_affine_probe`.
- With no `.metal` file present the metallib is an empty file and the name list is empty.
