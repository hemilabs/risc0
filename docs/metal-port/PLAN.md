# Metal port of the risc0 fork's GPU kernels — master plan

Repo: `/Users/user/zkminer/deps/risc0` (branch `zkminer/metal`, HEAD `09c6f360a`, based on pinned rev `68b3eacb`).
Crate root for every relative path below: `/Users/user/zkminer/deps/risc0/risc0/`.
Consumer: `/Users/user/zkminer/zk-miner` (`[patch.crates-io]` → `../deps/risc0`).
Target machine: Apple M5 Max, 36 GiB unified memory, macOS 27.2, Apple `metal` 32023.922 (toolchain mounted by `source /Users/user/zkminer/env.sh`).

This document merges five per-unit plans (zkp-hal, rv32im, recursion, keccak, groth16) and two cross-cutting plans (metal-infra, build-plumbing). Every statement marked **verified** was read in code or measured on this machine by a planning agent; everything in section 7 is **unverified** and must be treated as a hypothesis by implementation agents.

---

## 1. Goals, scope, non-goals, status today

### Goals
1. Run the full risc0 proving pipeline on the Apple GPU: rv32im segment proofs, recursion (lift/join/resolve/identity), keccak lifts, and the Groth16 wrap, producing seals that the unchanged CPU verifier accepts and that are **bit-identical to the CPU prover** wherever the CUDA prover is bit-identical (witness, accum, eval_check, NTT, hashes) and group-element identical where CUDA is (Groth16 MSM).
2. Ship a `zkminer-prove-risc0-metal` worker that zk-miner discovers, schedules and benchmarks on macOS.
3. Do it with comprehensive agent fan-out: one agent per work item, strict file ownership, review and optimization rounds per kernel family, CPU-vs-Metal conformance at every step.

### Scope
- `risc0-zkp` generic HAL (`zkp/src/hal/metal.rs`, `sys/kernels/zkp/metal`, `sys/build.rs`), including ports of the supra/sppark NTT, Poseidon2, mdiv, eval_u, mix kernels.
- `risc0-build-kernel` Metal path (`compile_metal`, `cached_compile`, shared headers in `build_kernel/kernels/metal`).
- Circuit HALs + kernels: rv32im (`circuit/rv32im{,-sys}`), recursion (`circuit/recursion{,-sys}`), keccak (`circuit/keccak{,-sys}`), groth16 (`groth16{,-sys}`).
- Cargo feature/build wiring (`risc0-zkvm/metal` → circuits → `-sys` → `risc0-sys`), zkvm orchestration arms, zk-miner worker/discovery/dispatcher.
- Conformance tests, end-to-end tests, benchmarks.

### Non-goals (round 1)
- sppark cooperative-launch kernels (`div_by_x_minus_z`, `prefix_op`, Groth16 pippenger grid syncs): not portable; replaced by multi-dispatch designs or CPU.
- `nttq` (needs > 32 KB threadgroup memory), ROCm-only variants (`mix_poly_coeffs_v6`, `stepAccum` HIP form, `wmsSort/wmsCompare`, `ef_prep_q`, `eval_check_ef_hip`, `recWom*` GPU sorts, `pv::rows0_k`/`fold_q4_k`), Intel/SYCL anything.
- Metal 4 APIs (metal-rs 0.29 has no bindings — **verified**), objc2-metal migration, iOS.
- A C++/Objective-C++ host layer: **all host orchestration is Rust over metal-rs 0.29** (decided by every unit).
- Multi-GPU, CI runners (no macOS job exists in `.github/workflows` — verified).

### Status today (verified)
| Area | State |
|---|---|
| `zkp/src/hal/metal.rs` (1032 lines) | 19 kernels in `sys/kernels/zkp/metal`, 18 DualHal unit tests pass. Creates a **new `ComputePipelineState` and a new command buffer per dispatch** and `wait_until_completed`s each; stores a process-wide `ReentrantMutexGuard` for the HAL's lifetime; `to_vec()` ignores slice offset/size; NTT grid capped at 32 TG × 128 threads; `prefix_products`, `scatter`, `eltwise_zeroize_elem`, `eltwise_copy_elem_slice`, `combos_*`, `gather_samples`, `hash_fold_tree` are CPU/default-trait paths; `metal_prefix_products` feature references non-existent kernels/types. |
| `zkp/src/hal/mod.rs:25` | `pub mod metal` gated on **target** (`macos+aarch64 \| ios`), not on a feature; `sys/build.rs` compiles the zkp metallib on every macOS build, so every macOS `prove` build needs the Metal toolchain. |
| `build_kernel` | `compile_metal` shells `xcrun --sdk macosx metal -c` per file (rayon-parallel) then `metallib`; **ignores `.flag()`** (closure arg `_flags` unused, cache hashed with `&[]`); no `-O`/`-std`; cache keyed by SHA-256 of sources + fp.h/fpext.h + explicit file deps (include dirs not hashed, directory deps panic). `METAL_INCS` = `kernels/metal/{fp.h,fpext.h}`. |
| `build_kernel/kernels/metal/fp.h, fpext.h` | No namespace; `threadgroup Fp x[N]` **fails to compile** (needs address-space constructors); `Fp x; x = 0;` is **ambiguous** (`operator=(uint32_t)` in two address spaces); `constant FpExt*` operands fail (no `constant` operators). |
| recursion | Stale `circuit/recursion-sys/kernels/metal/{eval_check,step_compute_accum,step_verify_accum}.metal` (empty `poly_fp`, old assert form, no entry points, missing `extern.h`); stale `circuit/recursion/src/prove/hal/metal.rs` implements only the pre-#3114 `CircuitHal`, reads `RECURSION_METAL_PATH` that nothing sets, is commented out of `hal/mod.rs`; `prove/mod.rs:162` arm commented out. |
| rv32im, keccak, groth16 | No Metal code. `circuit/rv32im` has no build.rs and no `metal` feature; `circuit/keccak{,-sys}` have no `metal` feature; `groth16-sys` has `metal = []`; groth16 on macOS = Docker prover. |
| Features | `metal = ["prove"]` placeholders in `risc0-zkvm`, `risc0-zkp`, `risc0-circuit-recursion`; `metal = []` in `risc0-sys`, `rv32im-sys`, `recursion-sys`, `groth16-sys`; nothing reads them. |
| zk-miner | Bins/features `cuda|rocm|intel|cpu` only; discovery is Linux sysfs (`GPU_TAGS = cuda/rocm/intel/generic`); `release_groth16_cache_if_tight` releases the SRS unless `ZKMINER_CUDA_VRAM_BYTES ≥ 20 GiB`; `mem_total_bytes`/`get_cpu_info` are Linux-only. |
| Device (measured) | `maxThreadgroupMemoryLength` 32768, `maxThreadsPerThreadgroup` 1024, SIMD width 32, `recommendedMaxWorkingSetSize` 30.15 GB, `maxBufferLength` 22.6 GB, argument buffers Tier2, unified memory. |
| MSL compiler (probed) | Accepts: templates + `[[host_name]]`-style instantiation (probed by zkp unit with `-std=metal3.2`), struct-of-`device`-pointers in `constant`/`device`, dynamic + static threadgroup memory (32 KB), `simd_shuffle_*`/`simd_ballot`/`simd_prefix_exclusive_sum`, 32-bit atomics incl. CAS, `mulhi` 32/64, `reverse_bits`, `__attribute__((noinline))`, `-O0..-O3`, `-std=metal3.2`. **Rejects**: 64-bit atomics, pointer-to-pointer kernel args, lambdas, derived classes/virtual, CTAD, structured bindings over `metal::array`, `array::value_type`, pointer members without address space, namespace-scope `constexpr` non-`constant` data. |
| Compile/PSO times (measured) | Front-end ≈ 10k lines/s (step_exec 53k lines: 8.5 s; keccak eval_check 91k lines: 24 s, 1.8 GB RSS). Pipeline-state (back-end) creation is the real cost: recursion eval_check monolith 20.9 s cold / 0.00 s warm (OS shader cache); keccak whole-chain PSO **did not finish in 20–55 min**; keccak 8-stage cut: stage 0 = 1050 s, others 20–65 s; keccak single witgen arm_50 12 s. |

---

## 2. Shared conventions (binding for every implementation agent)

### 2.0 Cross-unit decisions (contradictions resolved)
| Topic | Units disagreed | Decision |
|---|---|---|
| Backend selection | zkp/recursion/keccak: target cfg; rv32im/build-plumbing/metal-infra: cargo feature | **Cargo feature `metal` everywhere**, fanned out exactly like `cuda`: `risc0-zkvm/metal` → `risc0-zkp/metal`, `risc0-circuit-{rv32im,recursion,keccak}/metal`, `risc0-groth16/metal` → each `-sys/metal` → `risc0-sys/metal`. `zkp/src/hal/mod.rs`: `#[cfg(feature = "metal")] pub mod metal;` plus `compile_error!` if `metal` is enabled on a non-macOS/iOS target, and `compile_error!` for `metal` together with `cuda|rocm|intel` (in `risc0-sys`, every `-sys`, `risc0-zkvm`). Consequence: `sys/build.rs` builds the zkp metallib only when `CARGO_FEATURE_METAL` is set, so plain macOS `prove` builds (zk-miner's `cpu` bin) no longer need the toolchain. cfg_if arms are `else if #[cfg(feature = "metal")]` placed after `rocm`/`intel` and before the CPU fallback. |
| Runtime escape hatch | keccak wanted `RISC0_KECCAK_METAL=0` | One helper `risc0_zkp::hal::metal::enabled(unit: &str) -> bool`: false if `RISC0_METAL=0`, or if `RISC0_METAL_SKIP` (comma list of `zkp,rv32im,recursion,keccak,groth16`) names the unit. Every circuit's cfg_if metal arm calls it and falls back to the CPU prover (groth16: Docker) when false. |
| Where the metallib lives | build-plumbing: `-sys` lib.rs; rv32im/recursion/keccak: circuit build.rs re-exporting `DEP_*` | **`-sys` crate embeds it.** Each `-sys/build.rs` runs `KernelBuild::new(KernelType::Metal)...compile("metal_kernels_<unit>")` and prints `cargo:rustc-env=<UNIT>_METAL_PATH=<OUT_DIR>/metal_kernels_<unit>.metallib`; `-sys/src/lib.rs` exposes `#[cfg(feature = "metal")] pub const METAL_LIB: &[u8] = include_bytes!(env!("<UNIT>_METAL_PATH"));` and `pub const METAL_KERNEL_NAMES: &[&str]`. No new circuit build.rs (rv32im has none). `risc0-zkp` keeps its existing `DEP_RISC0_SYS_METAL_KERNELS_ZKP` → `ZKP_METAL_PATH` chain, gated on `CARGO_FEATURE_METAL`. |
| Shared MSL headers | build-plumbing: `DEP_RISC0_SYS_METAL_ROOT`; zkp-hal: `METAL_INCS` | **`build_kernel/kernels/metal/` via `METAL_INCS`** (auto-written to `<OUT_DIR>/_sys_` and on every `-I` path): `fp.h`, `fpext.h`, `cuda2metal.h`, `sppark_compat.h`. `DEP_RISC0_SYS_METAL_ROOT` stays exported but unused; circuit-specific headers live in `<unit>-sys/kernels/metal/`. Never add `kernels/cuda` to a Metal include path (recursion's `extern.h` shadows). |
| Translator language | metal-infra: Rust in build-kernel; rv32im/recursion: Python; keccak: Rust in build.rs | **Per unit, following that unit's existing generator precedent**: rv32im and recursion use Python (`cuda2metal.py` siblings of their existing generators, invoked at **build time** via `risc0_build_kernel::python_command()` into `OUT_DIR` — the precedent is both crates' Intel paths, verified at `rv32im-sys/build.rs:1610..2066`, `recursion-sys/build.rs:769,842`); keccak uses Rust in `keccak-sys/build.rs` (precedent `write_intel_translation_unit`); zkp and groth16 are hand-written. Generated `.metal` is **never checked in** (delete the stale recursion files). Every translator has pinned rewrite counts and a `--check`/golden test. The shared `cuda2metal.h` conventions are what every translator targets. |
| Generated output location | rv32im: committed; recursion: OUT_DIR | **OUT_DIR for all** (see above). |
| Kernel context passing | zkp mdiv / metal-infra: struct of `gpu_address()` in `constant`/argument buffer + `use_resource`; rv32im B / recursion B: explicit bindings | **Explicit `[[buffer(n)]]` bindings + `set_bytes` POD scalar structs.** Kernels build any thread-local context struct of `device` pointers from their arguments. Pointer-bearing structs passed via `set_bytes` are allowed only as the zkp unit's K5 experiment (`MdJobs`), with an argument-buffer-encoder + `use_resource` fallback. |
| Sync model | zkp A: explicit sync + debug flag; zkp B: lazy via shared event | **Lazy batch model** (2.6) with the debug assertion. |
| Smart pointer to the HAL | rv32im/recursion: `Arc<MetalHal>`; keccak: `Rc` | Mirror what each circuit's `hal/cuda.rs` uses today (`Arc` in rv32im/recursion, `Rc` in keccak — verified) so prover plumbing stays source-compatible. `MetalHal` is constructed per prover; no guard is stored in it. |
| Singleton lock | stored guard vs per-call | `pub fn singleton() -> &'static ReentrantMutex<()>` and `pub fn exclusive() -> ReentrantMutexGuard<'static, ()>`, taken **per call** inside Hal methods that touch the GPU (CUDA precedent `cuda.rs`), and by groth16 around the whole `prove`. `MetalHal::new` no longer holds a guard. |
| Poseidon254 | zkp defers to recursion; recursion defers to groth16; build-plumbing requires it | Phase 1: `recursion_prover("poseidon_254")` on Metal returns the **CPU** recursion prover (identity_p254 only). Item **Z10** (zkp unit) adds `MetalHashPoseidon254` using the BN254 Montgomery header from groth16 item **G2** (`bn254.h`); after Z10 recursion switches p254 to Metal. |
| eval_check split vs monolith (recursion) | A: split; B: monolith | Monolith default, split selectable by `RISC0_METAL_REC_KERNELS`, decided by measurement in C11. |
| keccak eval_check stage count | A: ~12; B: 58 | Start one function per stage (58), sweep 58/29/16/8 in K7. |
| Groth16 host language / MSM | A: ObjC++ + sppark Pippenger; B: Rust + Intel scheme | Rust + Intel scheme first; Booth digits/privatized histogram as measured optimizations. |
| `assert` translation | per-assert atomic vs accumulate-once | Accumulate into a thread-local `bad` flag, OR once per thread into a `device atomic_uint* fault` at kernel end; clamp untrusted indices. |
| `fp.h` ambiguity (`x = 0`) | fix header vs rewrite | **Rewrite**: all MSL (hand-written and translated) writes `Fp(0)`; fp.h gains address-space constructors and by-value free operators (F1) but keeps its operator set otherwise unchanged. |
| Second command queue / eval_check overlap | now vs later | Round 1: one queue, `eval_check_dep()` no-op. Second queue is an optimization-round item (R14, Z9). |

### 2.1 Address spaces (MSL)
- `device` — every trace/witness/scratch/output buffer, including `poly_mix` and other FpExt tables (`fpext.h` has no `constant` operators; `const device FpExt*` works). Read-only data may use `const device`.
- `constant` — only POD scalar/param structs bound with `set_bytes` (≤ 4 KB) and program-scope `constant constexpr` tables (layouts, round constants, twiddle generator constants, Montgomery constants). A `constant Fp`/`FpExt` operand must be copied to a `thread` local before arithmetic.
- `thread` — locals, per-thread context structs (`thread ExecCtx*`), small arrays (`Fp extern_args[96]`). Arrays larger than ~1 KB per thread (keccak's `Fp x13[8237]`, rv32im's `Fp x33[1007]`+`FpExt x34[144]`) become `device` scratch slices with stride = chunk size (`base[i*stride + tid]`, keccak `ScratchRef` precedent) unless the unit measures the thread array as acceptable.
- `threadgroup` — only via `threadgroup T* buf [[threadgroup(i)]]` kernel parameters sized from the host (`set_threadgroup_memory_length`), raw `uint32_t` storage reinterpreted to Fp/FpExt when the header constructors are insufficient; total ≤ 32768 bytes per threadgroup, asserted on the host.
- Every pointer and reference in a signature or struct carries an explicit qualifier; `T&` becomes `thread T&` / `device T&`; `auto&` over device data becomes `device auto&`. No pointer-to-pointer kernel args (`Fp**` → explicit pointers).

### 2.2 Fp/FpExt types
- Only `build_kernel/kernels/metal/fp.h` and `fpext.h` define field types; no unit ships its own Fp. F1 adds (additions only): `constexpr` constructors for `threadgroup`/`device`/`constant` objects, by-value free operators so `threadgroup`/`constant` operands work, nothing else. Owner of these two files during the whole port: the F1 agent, then the zkp-hal unit.
- `sppark_compat.h` (F1) aliases `bb31_t`/`fr_t` → `Fp`, `bb31_4_t`/`fr4_t`/`md4_t` → `FpExt`, `index_t` → `uint`, plus `zero()`, `csel`, `operator^(unsigned)`, `reciprocal()`.
- Write `Fp(0)`, never `= 0`. Access FpExt limbs as `.elems[i]`. BN254 types (groth16/Poseidon254) live in `groth16-sys/kernels/metal/bn254.h` (32-bit limbs only).

### 2.3 `cuda2metal.h` (F1; shipped in `METAL_INCS`)
Aliases, all verified to compile unless noted: `__device__/__host__/__forceinline__/__noinline__` → `inline`/`__attribute__((noinline))`; `__syncthreads()` → `threadgroup_barrier(mem_flags::mem_threadgroup)` (+ `__syncthreads_device()` with `mem_device`); `__syncwarp()` → `simdgroup_barrier`; `__shfl_{down,up,xor}(_sync)` → `simd_shuffle_*`; `__ballot_sync(m, p)` → `(uint)(simd_vote::vote_t)simd_ballot(p)`; `__popc/__clz/__brev/__ffs` → `popcount/clz/reverse_bits/ctz+1`; `__ldg/__ldcg(p)` → `*(p)`; `atomicAdd/Or/CAS` on 32-bit → `atomic_*_explicit(..., memory_order_relaxed)`; `__umulhi/__umul64hi` → `mulhi`; `WARP_SZ`/`warpSize` = 32; `__restrict__`, `asm("")`, `opq()`, `__builtin_assume` → empty/identity; `__builtin_elementwise_min` → `min`; `#undef assert` then `assert(x)` → `((void)0)` (opt-in `R0_FAULT(ctx, cond)` writes the fault flag); `printf` → no-op; `cuda::std::array` → `metal::array` (translators) or a 20-line `r0::array`.
Not aliasable (translators/hand ports must emit): kernel signatures, index attributes, `extern __shared__`, `__constant__` globals, `__launch_bounds__(N)` → `[[max_total_threads_per_threadgroup(N)]]` emitted **before** the `kernel` keyword, template kernel instantiation → `[[host_name("...")]]` (one probe in D0 before any unit relies on it), lambdas, structured bindings, inheritance.

### 2.4 Kernel signature conventions
```metal
// kernels/metal/<file>.metal — global scope, no namespace (namespaced kernels get mangled metallib names)
struct FooParams { uint rows; uint cols; uint count; uint _pad; };   // POD, mirrored #[repr(C)] in Rust, sizes static_assert'ed on both sides
[[max_total_threads_per_threadgroup(256)]]             // replaces __launch_bounds__; MUST precede `kernel void` (post-signature position is rejected, verified)
kernel void foo(device Fp* out            [[buffer(0)]],
                const device Fp* in       [[buffer(1)]],
                device atomic_uint* fault [[buffer(2)]],   // present on every kernel that can detect a bad input
                constant FooParams& p     [[buffer(3)]],
                uint gid [[thread_position_in_grid]])
{
  if (gid >= p.count) return;                           // always bounds-check (dispatch_threads is non-uniform, dispatch_thread_groups is not)
  ...
}
```
- Buffer indices are explicit, contiguous, and documented in a comment block at the top of the kernel's `.metal` file **and** in a `const` table on the Rust side.
- Entry names: a 1:1 port keeps the CUDA `__global__` name (so inventories map); new kernels are `snake_case`; explicit template instantiations are `name_<params>` (`ct_x_8_coalesced`). Names must be unique within a metallib and listed in `METAL_KERNEL_NAMES`; `MetalKernels::new` fails fast if a listed name is missing.
- Index attributes: use `thread_position_in_grid` (1-D/2-D) by default; CUDA-shaped kernels (NTT, mdiv, scans) take the `uint3 threadIdx/blockIdx/blockDim/gridDim` quartet exactly as `sys/kernels/zkp/metal/ntt.metal` does today.
- Threadgroup memory: `threadgroup T* buf [[threadgroup(i)]]`; the host passes `tg_bytes[i]`.
- 64-bit integer math is allowed but flagged in code review (emulated on Apple GPUs); prefer 32-bit formulations; keep 64-bit variants selectable where CUDA has both (`EF_WIDE`, `RecLz`).
- No recursion, no function pointers, no `new`, no virtual, no exceptions, no `printf`.

### 2.5 Launch mapping (CUDA → Metal)
| CUDA | Metal |
|---|---|
| `kernel<<<grid, block, smem>>>(args)` | `hal.dispatch(pso, &args, Grid::Groups { groups: MTLSize(grid), threads: MTLSize(block) }, &[smem])` |
| 1-D elementwise `(count + block-1)/block` | `hal.launch(name, count, &args)` → `dispatch_threads(count, min(256, pso.max_total_threads_per_threadgroup()))` |
| `__launch_bounds__(N[, M])` | `[[max_total_threads_per_threadgroup(N)]]`; M dropped |
| `cudaMemsetAsync` / `fill_u32` | `hal.blit_fill(buf, range, byte)` for byte-repeating patterns (`Fp::INVALID = 0xffffffff`, zero) else a `fill_u32` kernel |
| `cudaMemcpyAsync` H2D/D2H | memcpy into/out of `StorageModeShared` contents (unified memory); `new_buffer_with_bytes_no_copy` only for 16 KB page-aligned, page-multiple host allocations |
| streams / events / `cudaDeviceSynchronize` | one command queue, batch model (2.6); `sync()` |
| `cudaMemcpyToSymbol` (`__constant__`) | a bound buffer (`const device`) or `set_bytes` if ≤ 4 KB |
| thrust/cub sorts & scans | CPU (counting sort / sequential scan on shared buffers) in round 1; GPU scan/sort kernels are optimization items |
| cooperative `grid.sync()` | split into separate dispatches |

### 2.6 Buffer ownership and sync model
- All prover-visible buffers are `risc0_zkp::hal::metal::BufferImpl<T>` (`StorageModeShared`, hazard-tracked, size-keyed pool). `view()/view_mut()/get_at()/to_vec()/copy_from*` **flush and wait** when a batch is open; raw `contents()` pointers obtained before a dispatch are not protected — re-fetch after `sync()`.
- Circuit HALs keep scratch (preflight copies, perm, tables, poly_mix, carry/scratch) in a `RefCell<ScratchBuffers>` grown with 25 % headroom; GPU-only scratch may be `StorageModePrivate` via `hal.alloc_private(bytes)`.
- Preflight traces are **memcpy'd** into HAL-owned shared buffers (kernels mutate `cycles[].txnIdx`/`iopIdx`; the Rust side keeps the trace pristine).
- Batch: `MetalHal` owns `Batch { cmd_buffer, encoder, retained scratch, seq }`. `dispatch`/`launch` only encode (serial dispatch type → dependent dispatches on tracked resources are ordered; use fresh encoders or `memory_barrier_with_resources` only when switching encoder kinds). `flush()` ends encoding and commits; `sync()` flushes and waits. Batches close at: any host access, every 2048 dispatches (until the encoding limit is measured), the end of every Hal method that CUDA ends with `gpu.sync()` (hash_rows/fold/fold_tree, NTT entry points, combos_divide), and the end of every circuit phase (witgen, accum, eval_check). Pool returns are deferred until the batch that used the buffer completes.
- Debug: with `RISC0_METAL_DEBUG=1` the HAL asserts no host read happens through an unsynced batch and runs with the Metal validation layer (`MTL_DEBUG_LAYER=1`). All conformance tests run once under it.
- Untracked resources are forbidden in round 1.
- `gpu_total_memory()` = `recommended_max_working_set_size`, `gpu_free_memory()` = that − `current_allocated_size`; buffers > `max_buffer_length` (22.6 GB) error early.
- Memory budget: po2 22 rv32im is near the 30 GB recommended working set; `release_evaluated/restore_coeffs` host spills become no-ops for unified-memory HALs (prover change, item Z1).

### 2.7 MetalHal API (F0 deliverable; every unit codes against exactly this)
```rust
// risc0/zkp/src/hal/metal.rs (+ metal/{ntt,poseidon2,mdiv,eval}.rs)
pub fn singleton() -> &'static ReentrantMutex<()>;
pub fn exclusive() -> ReentrantMutexGuard<'static, ()>;
pub fn enabled(unit: &str) -> bool;                 // RISC0_METAL / RISC0_METAL_SKIP
pub fn device_name() -> Option<String>;
pub enum KernelArg<'a> { Null, Buffer { buffer: &'a BufferRef, offset: u64 }, Integer(u32), Bytes(&'a [u8]) }
pub enum Grid { Threads(u64), Threads2D(u64, u64), Groups { groups: MTLSize, threads: MTLSize } }
pub struct MetalKernels { pub library: Library, pipelines: HashMap<String, ComputePipelineState> }
impl MetalKernels { pub fn pipeline(&self, name: &str) -> &ComputePipelineState; }
pub struct MetalHal<MH: MetalHash> { pub device: Device, pub cmd_queue: CommandQueue, /* private: zkp MetalKernels, Batch, pool, hash */ }
impl<MH: MetalHash> MetalHal<MH> {
    pub fn new() -> Self;                                       // builds all zkp PSOs once; no lock stored
    pub fn load_library(&self, bytes: &[u8], names: &[&str]) -> MetalKernels;   // for circuit metallibs; fails if a name is missing
    pub fn pipeline(&self, name: &str) -> &ComputePipelineState;                 // zkp kernels
    pub fn dispatch(&self, pso: &ComputePipelineState, args: &[KernelArg], grid: Grid, tg_bytes: &[u64]);  // encodes only
    pub fn launch(&self, name: &str, count: u64, args: &[KernelArg]);             // 1-D over the zkp library
    pub fn blit_fill(&self, buf: &BufferRef, offset: u64, len: u64, byte: u8);
    pub fn alloc_private(&self, bytes: u64) -> Buffer;
    pub fn flush(&self); pub fn sync(&self);
    pub fn with_encoder(&self, f: impl FnOnce(&ComputeCommandEncoderRef));       // escape hatch, documented per use
}
impl<T> BufferImpl<T> { pub fn as_buf(&self)->&BufferRef; pub fn as_arg(&self)->KernelArg; pub fn as_arg_with_offset(&self, off: usize)->KernelArg; pub fn as_ptr(&self)->*const T; pub fn as_device_ptr(&self)->u64; pub fn gpu_address_at(&self, off: usize)->u64; }
pub trait MetalHash { ... fn hash_fold_tree(...) }      // one batch per tree
pub type MetalHalSha256 = MetalHal<MetalHashSha256>; pub type MetalHalPoseidon2 = ...; // MetalHalPoseidon254 after Z10
```
Kept source-compatible for existing callers: `dispatch_by_name`, `dispatch_with_resources` (now encode-only + sync on return, deprecated).

### 2.8 Build, naming, cargo variables
| Thing | Name |
|---|---|
| Metal sources | `<unit>-sys/kernels/metal/*.metal` (hand-written) + `$OUT_DIR/metal/*.metal` (generated) |
| KernelBuild output | `metal_kernels_zkp`, `metal_kernels_rv32im`, `metal_kernels_recursion`, `metal_kernels_keccak`, `metal_kernels_groth16` |
| cargo link-metadata (`links` keys exist in every `-sys`) | `cargo:metal_kernels_<unit>=<path>` → `DEP_RISC0_SYS_METAL_KERNELS_ZKP` (consumed by zkp/build.rs), `DEP_RISC0_CIRCUIT_<X>_SYS_METAL_KERNELS_<X>` (exported, unused) |
| rustc-env | `ZKP_METAL_PATH` (zkp/build.rs), `RV32IM_METAL_PATH`, `RECURSION_METAL_PATH`, `KECCAK_METAL_PATH`, `GROTH16_METAL_PATH` (each `-sys/build.rs`) |
| Embedded lib | `risc0_circuit_<x>_sys::METAL_LIB`, `::METAL_KERNEL_NAMES`; `risc0_groth16_sys::METAL_LIB` |
| Compiler flags | `KernelBuild::flag("-std=metal3.2").flag("-O2")` (after F1 makes flags effective and part of the cache key); `-frecord-sources -gline-tables-only` under `RISC0_METAL_DEBUG_SOURCES=1` |
| Build gating | `CARGO_FEATURE_METAL` && `CARGO_CFG_TARGET_OS ∈ {macos, ios}`; `RISC0_SKIP_BUILD_KERNELS` keeps writing an empty metallib |
| Toolchain | `source /Users/user/zkminer/env.sh` before any cargo command; `build_kernel::find_metal_toolchain()` honours `METAL_TC`/PATH and panics with the env.sh hint |
| Env knobs | `RISC0_METAL`, `RISC0_METAL_SKIP`, `RISC0_METAL_DEBUG`, `RISC0_METAL_REC_KERNELS`, `RISC0_KECCAK_METAL_EVAL_STAGES`, `RISC0_GROTH16_METAL_PROVER=docker`, `RISC0_METAL_G16_COLD`, `RISC0_VERBOSE` timings |

### 2.9 Error handling
- GPU-detectable faults (bad trace, inconsistent set, out-of-range index) → `device atomic_uint* fault` ORed once per thread; the host reads it after `sync()` and `anyhow::bail!`s with the circuit and phase. Indices derived from untrusted data are clamped before use so a corrupt trace cannot read out of bounds.
- `CommandBuffer::status() == Error` after a wait → `bail!` (metal-rs 0.29 has no `error()` accessor; message retrieval via raw `msg_send!` is optional).
- `new_buffer` failure → drain the pool and retry once, then `bail!`.
- Missing kernel name, threadgroup memory > 32768, buffer > `max_buffer_length`, Params size mismatch → panic at HAL construction / dispatch (programming errors).

### 2.10 Running conformance tests
```sh
cd /Users/user/zkminer/deps/risc0/risc0 && source /Users/user/zkminer/env.sh
cargo test -p risc0-zkp --features metal hal::metal                       # DualHal CPU-vs-Metal (18 today, more after Z1..Z6)
cargo test -p risc0-circuit-rv32im --features metal prove::               # basic, multi_read, 46 riscv cases, fwd_rev_ab, hal::metal::tests
cargo test -p risc0-circuit-recursion --features metal                    # eval_check PO2 4, witgen/accum equivalence, lift/join
cargo test -p risc0-circuit-keccak --release --features metal prove::     # basic (po2 8), host_witness, fwd_rev_ab, eval_check
cargo test -p risc0-groth16 --features metal                              # bn254/NTT/MSM vs ark; metal_shrink_wrap (needs RISC0_GROTH16_DIR)
cargo test -p risc0-zkvm --features metal host::recursion::tests          # e2e
RISC0_METAL_DEBUG=1 MTL_DEBUG_LAYER=1 cargo test ...                      # validation-layer pass, once per item
```
Rules: no two agents run GPU perf measurements concurrently; correctness tests may overlap. Use `--target-dir` per agent only if build-lock contention becomes a problem (metallib cache is content-keyed and shared). Every work item ends with its tests green **and** the pre-existing suites of its unit green.

---

## 3. Shared infrastructure work items

| ID | Item | Owner files | Acceptance criteria |
|---|---|---|---|
| **D0** | Day-0 MSL/runtime probes, results recorded in `build_kernel/kernels/metal/probes/*.metal` + `README.md`: `[[host_name]]` on explicit template instantiations; cross-`.air` extern function resolution at `metallib` link; C++ designated initializers in `constant` tables; `gpu_address()` values inside a `set_bytes` struct executed at runtime; counter sampling at dispatch boundary; GPU family value; command-buffer watchdog on a 60 s single-thread dispatch; `assert()` release semantics; `[[max_total_threads_per_threadgroup(768)]]` (non-power-of-two) | `build_kernel/kernels/metal/probes/` | Each probe compiled with `xcrun metal`, run from a tiny Rust bin in the scratchpad, result (ACCEPT/REJECT/value) written in the README; probes compile in CI (`cargo test -p risc0-build-kernel metal_probes`). |
| **F0** | MetalHal runtime rewrite per 2.6/2.7 (PSO cache, Batch + lazy sync, pool with deferred returns, `KernelArg::Bytes`, `Grid`, `load_library`, `exclusive()/singleton()/enabled()/device_name()`, `blit_fill`, `alloc_private`, `hash_fold_tree` on `MetalHash`, `to_vec` fix, memory queries, remove `metal_prefix_products`, `report_nonzero_remainders` cfg widened) | `zkp/src/hal/metal.rs`, new `zkp/src/hal/metal/*.rs`, `zkp/src/hal/mod.rs` (metal-related lines only) | 18 existing tests green via `--features metal`; new tests: 50 dispatches then read without explicit flush; pool reuse across batches; two threads constructing HALs and running ops (no deadlock); `MTL_DEBUG_LAYER=1` clean; `to_vec` on a slice returns the slice. |
| **F1** | Compat headers + build_kernel: `cuda2metal.h`, `sppark_compat.h`, fp.h/fpext.h address-space constructors and by-value operators; `compile_metal` honours `.flag()` and hashes flags, `-std`/`-O` passthrough, recursive include-dir hashing, directory deps, per-file `rerun-if-changed`, `find_metal_toolchain()`; `METAL_INCS` extended | `build_kernel/src/lib.rs`, `build_kernel/kernels/metal/{fp.h,fpext.h,cuda2metal.h,sppark_compat.h}` | Every existing `.metal` in `sys/kernels/zkp/metal` compiles with the new headers; `threadgroup Fp x[8]`, `threadgroup FpExt y[4]`, `constant Fp` copy-then-multiply compile; a cache-invalidation test (touch a header under `-I`, assert rebuild); flags change the cache key. |
| **F2** | Feature/build wiring per 2.0/2.8: `metal` features in zkvm, zkp, every circuit, groth16, every `-sys`, `risc0-sys`; `compile_error!` guards; `sys/build.rs` gated on `CARGO_FEATURE_METAL`; `zkp/build.rs` gated; `zkp/src/hal/mod.rs` feature cfg; skeleton `build_metal_kernels()` in rv32im-sys/recursion-sys/keccak-sys/groth16-sys compiling a placeholder `kernels/metal/noop.metal`; `METAL_LIB`/`METAL_KERNEL_NAMES` consts; `hal/mod.rs` `pub(crate) mod metal` stubs + `prove/mod.rs` cfg_if arms calling `enabled()` and falling back to CPU; delete stale recursion `.metal` + rewrite stale `hal/metal.rs` to a compiling stub; zkvm `metal_implies_prove` kept | all `Cargo.toml`, all `build.rs`, `-sys/src/lib.rs`, `hal/mod.rs`, `prove/mod.rs` of each circuit, `zkvm/Cargo.toml`, `zkvm/src/lib.rs` | `cargo check -p risc0-zkvm --features metal` passes on macOS with stub HALs selecting CPU; `cargo check -p risc0-zkp --features prove` on macOS **does not invoke xcrun**; `cargo check -p risc0-zkvm --features metal,cuda` fails at the guard; `cargo check --target x86_64-unknown-linux-gnu -p risc0-zkp --features metal` fails at the target guard (check only). |
| **M2** | zk-miner host side (independent of risc0 code): `GPU_TAGS += "metal"`, `infer_gpu_tag("-metal")`, `detect_apple_gpus()` on macOS, dispatcher `gpu_env/gpu_vram_bytes/physical_gpu_id` metal arms (VRAM = `hw.memsize` − host reserve), benchmark/memory macOS fallbacks (`sysctl hw.memsize`, `machdep.cpu.brand_string`), ledger behaviour with known mem_total | `zk-miner/crates/zkminer-prover/src/{discovery,dispatcher,benchmark,memory}.rs` | `cargo test -p zkminer-prover discovery:: dispatcher::` extended for `metal`, `gpu-metal0` device id, env has no `*_VISIBLE_DEVICES`, `ZKMINER_CUDA_VRAM_BYTES` set. |

Foundation items D0, F0, F1, F2, M2 touch disjoint files and run in parallel (5 agents). Nothing in section 4 merges before F0, F1 and F2 have landed (unit agents may start translators/headers against scratchpad hand-compiles in the meantime).

---

## 4. Per-unit plans

### 4.1 zkp-hal (generic HAL + supra/sppark kernels)
Owner files: `zkp/src/hal/metal.rs`, `zkp/src/hal/metal/*.rs`, `sys/kernels/zkp/metal/*`, `sys/build.rs` (METAL_KERNELS list), `zkp/src/hal/mod.rs` testutil additions, `zkp/src/prove/{prover,merkle}.rs` (unified-memory no-op spills only).

**Kernels** (decision per CUDA `__global__`):
| CUDA kernel | Decision | Metal shape |
|---|---|---|
| eltwise_add/mul_factor/copy/sum/zeroize_fp(ext) | keep (exist); wire zeroize kernel into `eltwise_zeroize_elem`; `factor` via `Bytes` | 1-D |
| eltwise_copy_fp_region | port 1:1 | 1-D over rows |
| batch_bit_reverse | keep naive for n < 4096; port tiled `bit_rev_z_batched<64>` (16 KB TG) | grid (n/4096, cols) × 64 |
| batch_evaluate_any + eu2_pows/main/reduce | transliterate eu2 trio + Horner fallback; Rust dispatcher reproduces ffi.cu gate (`deg % 4096 == 0`, 16-B aligned) and S selection | eu2_main (S, nEvals) × 256; reduce nEvals × 32 |
| fri_fold, gather_sample, sha_rows/fold | keep | 1-D |
| gather_samples/gather_digests/gather_multi | skip round 1 (unified-memory path uses `view()` — verified in merkle.rs) | — |
| scatter, scatter_bits | port 1:1 (today CPU / 32× expansion) | 1-D |
| mix_poly_coeffs(_v2) | port v2 with TG table, gate `inputSize ≤ 1638`; v1-with-table for 1638..2048; table-less beyond; v6 skipped | 1-D, 256/TG |
| combos_sub | port 1:1 (keeps combos_prepare in-batch); combos_prepare device loop skipped | 1-D |
| batch_lde_expand | keep as fallback; expand-on-load in CT pass 1 | — |
| _CT_X/_GS_X<1,8,8+coalesced>, nttx_zk_shift_batched | transliterate `ntt_x.cuh` via sppark_compat; `[[host_name]]` instantiations; CPU-built twiddle/zk tables per lg; `ct_steps/gs_steps/X_launcher/nttx_commit` 1:1 in Rust; n⁻¹ and zk-shift fused in last GS pass | block 2^(radix−1) ≤ 512, grid (nblocks/8, cols), TG ≤ 16 KB |
| nttx_fill_one/gather_pow | not needed (tables on CPU) | — |
| nttq | **not ported** (> 32 KB TG) | — |
| mdiv_agg0/scan<32..1024>/apply/final | transliterate; `MdJobs` via `Bytes` (D0 probe) or argument buffer; single-TG fallback for po2 < 16; new two-level scan or MD_E=16 for po2 22 | agg0/apply (ntiles, njobs) × 256 |
| d_div_by_x_minus_z, d_prefix_op, prefix_product | not portable (grid sync); single-block form only; `prefix_products` stays CPU as on CUDA | — |
| pv::fold_k/rows_k/fold_tail_k, pvs w/q | transliterate HIP C++ branch; `constant` tables; `hash_fold_tree` in one batch; pvs after pv with measured crossovers | ceil(n/128) × 128; fold_tail 1 × n0 |
| poseidon254 fold/rows | **Z10** after G2 (`bn254.h`) | — |
| current Metal-only kernels (multi_ntt_*, multi_poly_eval, batch_expand, zk_shift, multi_bit_reverse, classic poseidon2) | legacy oracle behind `RISC0_METAL_LEGACY=1`, deleted in Z7 | — |

**Host glue**: all in Rust — `zkp/src/hal/metal/{ntt,poseidon2,mdiv,eval}.rs` port `ffi.cu`/`api.cu`/`ntt.cu` dispatch logic function for function; no `sys/src/metal.rs` externs; `hal/mod.rs` `report_nonzero_remainders` cfg widened.

**Strategy**: faithful-first transliteration of the bit-exactness-sensitive kernels behind the compat shims, preserving CUDA launch shapes; native replacements only where Metal forces it (no grid sync, 32 KB TG, unified memory). Legacy kernels kept as an A/B oracle until DualHal equality holds.

**Work items** (after F0, F1, F2; Z1–Z5 in parallel, one agent each, disjoint files):
| ID | Item | Files | Depends |
|---|---|---|---|
| Z1 | Small kernels + wiring: zeroize, copy_fp_region, scatter, scatter_bits, combos_sub, fill/blit for `alloc_elem_init`/`alloc_extelem_zeroed`, `Bytes` for mix/norm/factor, eltwise_sum arg order; prover no-op spills on unified memory; DualHal tests | `sys/kernels/zkp/metal/{eltwise,misc}.metal`, `zkp/src/hal/metal/small.rs`, `zkp/src/prove/prover.rs` (spill no-ops) | F0 F1 |
| Z2 | NTT: `ntt_x.metal` (CT/GS/bitrev/zk-shift), `NttTables`, launchers, expand-on-load, `commit_coeffs_lde` + `has_commit_coeffs_lde = 10..=23`, `RISC0_METAL_LEGACY` switch, `_large` + `commit_coeffs_lde` tests | `sys/kernels/zkp/metal/ntt_x.metal`, `zkp/src/hal/metal/ntt.rs` | F0 F1 |
| Z3 | Poseidon2 pv/pvs: `pv.metal`, `constant` tables, `hash_fold_tree` one batch, crossovers | `sys/kernels/zkp/metal/pv.metal`, `zkp/src/hal/metal/poseidon2.rs` | F0 F1 |
| Z4 | eval_u (eu2 + Horner + dispatcher) and mix v2/v1-table | `sys/kernels/zkp/metal/{eval,mix}.metal`, `zkp/src/hal/metal/eval.rs` | F0 F1 D0 |
| Z5 | combos_divide: mdiv transliteration, `mdiv_run`, po2 < 16 and po2 22 fallbacks, remainder reporting | `sys/kernels/zkp/metal/mdiv.metal`, `zkp/src/hal/metal/mdiv.rs` | F0 F1 D0 |
| Z6 | Tests + bench: new DualHal testutil functions (scatter, scatter_bits, zeroize, copy_elem_slice, combos_prepare, combos_divide po2 13..22 incl. nonzero remainders, zk_shift variants, eu2/Horner shapes, mix > 1638 and count % 256 ≠ 0), enable CUDA-only tests on Metal, sync/pool stress, benchmark bin (GB/s, perms/s, per-method timings) | `zkp/src/hal/mod.rs` testutil, `zkp/src/hal/metal/tests.rs`, `zkp/benches/metal.rs` | Z1–Z5 |
| Z7 | Retire legacy kernels, regenerate `KERNEL_NAMES`, `sys/build.rs` lists | `sys/kernels/zkp/metal/*`, `sys/build.rs` | Z6 |
| Z8 | Review round (split by kernel family): simd_shuffle out-of-range audit, bounds under grid rounding, every host access through sync, diff review vs CUDA/HIP source | read + fixes in Z1–Z5 files | Z7 |
| Z9 | Optimization round (measure first): empty-dispatch cost, copy GB/s, NTT GB/s by radix/chunk, Poseidon2 perms/s, eu2 S sweep, mdiv vs CPU, mulhi REDC, max_total_threads tuning, second queue, heap placement, nttq-under-32KB go/no-go | per family | Z6 |
| Z10 | `MetalHashPoseidon254` (fold/rows) from `groth16-sys/kernels/metal/bn254.h`; recursion p254 switch | `sys/kernels/zkp/metal/poseidon254.metal`, `zkp/src/hal/metal/poseidon254.rs` | G2, Z3 |

**Acceptance**: 2.10 zkp suite green after every swap; CUDA-only testutil tests (`batch_expand_into_evaluate_ntt_large`, `hash_fold_tree` layers 1..14, `commit_coeffs_lde` 13 shapes incl. po2 22) enabled on Metal; legacy-vs-ported exact equality until Z7; plumbing tests (F0); per-method timings at po2 20/21/22 × 211 columns recorded (not asserted).

**Risks**: no grid barrier (mdiv coverage 2^16..2^21 only); 32 KB TG (nttq, mix gate); runtime validity of `gpu_address` in `Bytes` structs; lazy sync misses → silently wrong proofs (debug assertion + stress tests); bit-exactness of CPU-built twiddle tables vs sppark order (cross-check at init); 64-bit multiply emulation; 36 GiB shared with the CPU prover.

### 4.2 rv32im
Owner files: `circuit/rv32im-sys/kernels/metal/*`, `circuit/rv32im-sys/kernels/cuda/cuda2metal.py` (+ `--metal` flags in `witgen_tu_gen.py`, `accum_tu_gen.py`, new `evalfast/to_metal.py`), `circuit/rv32im-sys/build.rs` (`build_metal_kernels` body), `circuit/rv32im/src/prove/hal/metal/{mod,witgen,accum,eval_check}.rs`, `circuit/rv32im/src/prove/witgen/{mod,tests}.rs` metal arms.

**Kernels**:
| CUDA kernel | Decision | Notes |
|---|---|---|
| par_stepExecW / WITGEN_TU_KERNEL (witgen_tu.cu, 19k lines, 388 `__noinline__`) | port faithful via `witgen_tu_gen.py --metal` | graph stays `__attribute__((noinline))` (CUDA comment: inlined build is not bit-exact); lambdas → loops; 7 structured bindings → temps; kind-tagged BufferObj (intel/witgen.h pattern); real `pre_data` buffer by default until single-buffer bit-exactness is proven |
| fwd_stepExec / rev_stepExec | port tiny 1×1 dispatch, chunked per 4096 cycles | tests / `RISC0_WITGEN_SEQ` only |
| injectCycleCols | port 1:1 (15 lines) | data buffer does not exist during preflight → kernel stays |
| stepAccumA (accum_tu.cu, 48.7k lines) | port via `accum_tu_gen.py --metal`; `[[clang::always_inline]]` first, noinline fallback (go/no-go on compile time) | `AccumParams` via `Bytes` |
| stepAccum (HIP), finalizeAccum (never launched — verified), eval_check_diag, wmsCompare, ef_prep_q, ef_eval_split HIP | **drop** | |
| majorKeys + cub radix sort | **replace**: CPU counting sort → perm buffer; window sweep 11..14 | perm = null is a valid fallback |
| wmsSort | defer (optional optimization) | |
| cub InclusiveScan ×4 | **replace**: CPU scans first (R8), 3-pass `fp_scan_block/sums/apply` kernels in R12 | |
| finalizeZeroizeAccum | port 1:1; `step_accum_zeroizes = true` | |
| eval_check (baseline, eval_check_0..3.cu + .cuh, 20 noinline chain fns) | port as **phase-1 oracle** via `cuda2metal.py` | `poly_mix` threaded as `const device FpExt*` parameter |
| ef_prep / ef_prep_n | **replace**: 627-entry table + 4 quotients computed on CPU into one shared buffer | |
| ef_eval_split, ef_combine (evalfast) | port via `evalfast/to_metal.py` branched after `remat_lu.py` (HIP_LAST): C++ `#else` paths, no `mksm2`/`mksub`, `EfCtx` struct of `device` pointers, `EF_WIDE` A/B | phase 2 |

**Host glue**: Rust-only `MetalRv32imCtx { library, pipelines, table_u8/u16, pf (cycles|txns|bigint, 36/20-byte structs static_assert'ed both sides), perm, scratch, poly_mix }`; explicit buffer bindings + `Bytes` POD params; witgen = one command buffer [fill tables → injectCycleCols → par_stepExec × 2]; accum = CPU perm → [fill, step_accum] → wait → CPU scans → [finalizeZeroizeAccum]; eval_check = CPU tables → one command buffer → wait; `eval_check_dep` no-op; `writes_cycle_cols = true`, `takes_zeroed_data = true`, `DEFER_CYCLE_COLS = true`; `drop_trace` background thread includes metal; `metal_warmup()` builds PSOs. `RawPreflightCycle` = **36 bytes**, `RawMemoryTransaction` = 20 bytes (verified; both plans had it wrong).

**Work items**:
| ID | Item | Files | Depends |
|---|---|---|---|
| R1 | Hand-written MSL support headers: `kernels/metal/{cuda_compat.h, witgen.h, buffers.h, tables.h, preflight.h (static_asserts 36/20), ef_fp.h, ef_fpext.h}` + `misc.metal` (injectCycleCols, finalizeZeroizeAccum, fwd/rev wrappers); xcrun compile | `rv32im-sys/kernels/metal/*` | F1 |
| R2 | `cuda2metal.py` core passes (array/value_type, lambda expansion of the 43 map/reduce sites, structured bindings, address-space policy table, qualifier rewrite, kernel signature synthesis, `Fp**` → pointers, strip launch_bounds/asm/nvtx; `Fp(0)` rule) + `--selftest` compiling extracted snippets | `rv32im-sys/kernels/cuda/cuda2metal.py` | F1 |
| R3 | `types/layout/defs .metal.inc` translation; compile `witgen.h` + the three `.inc` alone; `build_metal_kernels()` body invoking the generators into `OUT_DIR` with `--check` | `rv32im-sys/build.rs` (metal fn), `rv32im-sys/kernels/metal/*.metal.inc` | R1 R2 F2 |
| R4 | `accum_tu_gen.py --metal` → accum_tu.metal; compile + **time** (always_inline go/no-go) | `accum_tu_gen.py` | R3 |
| R5 | `witgen_tu_gen.py --metal` → witgen_tu.metal (+fwd/rev); compile + time | `witgen_tu_gen.py` | R3 |
| R6 | Baseline `eval_check.metal` from eval_check_0..3.cu + .cuh via cuda2metal.py; compile + time | `cuda2metal.py` eval mode | R2 |
| R7 | `hal/metal/mod.rs` skeleton: `MetalRv32imCtx`, library/PSO load from `METAL_LIB`, buffer layout, `segment_prover()`, `metal_warmup`, cfg gates (replaces F2 stub) | `rv32im/src/prove/hal/metal/mod.rs`, `hal/mod.rs`, `prove/mod.rs` arm | F0 F2 |
| R8 | Witgen path (`hal/metal/witgen.rs`): preflight memcpy, one command buffer, seq modes chunked, `witgen_matches_cpu` test | `hal/metal/witgen.rs`, `witgen/tests.rs` metal arm | R5 R7 |
| R9 | Accum path (`hal/metal/accum.rs`): CPU perm, step_accum, CPU scans, finalize; `accum_matches_cpu`, `perm_equivalence` tests | `hal/metal/accum.rs` | R4 R7 |
| R10 | eval_check phase 1 (`hal/metal/eval_check.rs`): CPU poly_mix powers (458 asserted), baseline dispatch, test po2 4/10/16 | `hal/metal/eval_check.rs` | R6 R7 |
| R11 | Integration gates: `prove::tests` basic/multi_read + 46 riscv cases, fwd_rev_ab metal arm, seal verification, seal equality vs CPU under `witgen_debug`; fix diffs | tests + fixes | R8 R9 R10 |
| R12 | evalfast `to_metal.py` (EfCtx, no asm, thread slab, EF_WIDE switch), regen.py EXPECT pins, CPU table builder, ef_eval_split/ef_combine sequencing; bit-exact vs baseline and CPU at po2 16/20 | `evalfast/to_metal.py`, `regen.py`, `hal/metal/eval_check.rs` (evalfast branch) | R10 R11 |
| R13 | O1: GPU Fp scan kernels (3 dispatches, 4 columns) → single accum command buffer | `misc.metal`, `hal/metal/accum.rs` | R9 |
| R14 | O2/O3: blit fills, drop_trace thread, perm window sweep; eval_check on a second queue with `commit_async` + `eval_check_dep` wait | `hal/metal/{mod,eval_check}.rs` | R11 R12 F0 |
| R15 | O4 tuning (independent sub-tracks): TG-size sweeps, always_inline vs noinline accum, EF_THREADS 64/128/256, EF_WIDE on/off, hot-word TG staging, single-buffer witgen once bit-exact over many segments, wmsSort port, tableU16 atomic cost | per sub-track | R12 R11 |
| R16 | Hygiene: `--check` covers .metal outputs, `cuda2metal --selftest` in CI, compile-time logging, README | generators, README | R12 |

Critical path: R2 → R3 → R5 → R8 → R11 → R12 → R14/R15. R1/R2/R7 start together; R4/R5/R6 on three agents; R8/R9/R10 on three agents (disjoint submodules).

**Acceptance**: section 2.10 rv32im line; `hal/metal/tests::eval_check` (po2 4/10/16, baseline and evalfast, canonical inputs); `metal_matches_cpu` (data+global, then all 103 accum cols, repeated N times); `fp_scan` unit test (sizes 1, 255, 256, 2^20+7); `perm_equivalence`; seal equality vs CPU for fixed rand_z; multi-segment composite verifies; generator `--check` round trips; compile-only CI with per-TU times. Perf recorded at po2 20/21 per phase vs CPU HAL on the same machine.

**Risks**: compile/PSO time of the 19k/49k-line TUs (gating experiments R4/R5; fallback noinline accum, split by mux arm); register pressure (324-word slab; `Fp[1007]+FpExt[144]`); back-row load ordering without a compiler barrier (pre_data default); 64-bit `EF_WIDE` cost; macOS watchdog on single-thread kernels (chunked); tableU16 atomics; generator drift (`--check`); po2 22 memory.

### 4.3 recursion
Owner files: `circuit/recursion-sys/kernels/metal/{context.h, extern.h, rec_kernels.metal}`, `circuit/recursion-sys/kernels/cuda/cuda2metal.py`, `circuit/recursion-sys/build.rs` (`build_metal_kernels` body), `circuit/recursion/src/prove/hal/metal/{mod,witgen,accum,eval_check}.rs`, `circuit/recursion/src/prove/{mod,witgen}.rs` host-chain cfg alias.

**Kernels**:
| CUDA kernel | Decision |
|---|---|
| step_exec + 19 noinline arms (53k lines, 4373 asserts, 51 `xN = 0;`) | translate (steps mode): `thread ExecCtx* ctx`, `device Fp* argN`, `device auto& reg`, `xN = Fp(0);` (**mandatory**, verified ambiguous otherwise), `assert(E)` → `ctx->bad |= !(E)` |
| parStepExec / parStepVerifyWom / parStepComputeAccum / parStepVerifyAccum | kernel wrappers emitted by the translator; thread-local ctx of `device` pointers from ordinary buffer args; fault flag ORed at end; TG 256 then sweep |
| *Perm variants, recExecKeys/recAccumKeys (HIP) | CPU-computed, per-program-cached permutations (C11) |
| fwdStepExec / revStepExec / fwdStepVerifyWom | **not ported**: Seq modes run the CPU witgen on the shared buffers |
| markHostChains / scatterHostChains | CPU scatter into shared buffers (kernels kept optional in rec_kernels.metal) |
| thrust sort(womRows) + exclusive_scan(womIndex) | CPU `sortWomFast` port (exact `std::sort` result) + scan between the two witgen command buffers; GPU sort only if measured bottleneck (C12) |
| injectWomBacks, fillOneExt | 1:1 (`WomArgumentRow` = 20 B static_assert) |
| step_compute_accum / step_verify_accum (`c1_bad` form) | translate; `if (c1_bad) __trap()` → `ctx->bad |= c1_bad`; Inl variants unneeded |
| sppark calcPrefixProducts | replaced by the ffi.cu SPPARK_NO_COOPERATIVE_LAUNCH trio (Chunks/Totals/Fixup) |
| poly_fp + eval_check (12k lines) | translate (eval mode); `poly_mix` as `const device FpExt*` first param; PSO once per process (20.9 s cold measured) |
| ec_part<0..5> (eval_check_split) | optional `--split` (CUDA `#else` bodies, text-expanded instantiations); adopted only if measured faster |
| recInitRows / recFillTail / recZeroizeRows | opt round, behind `has_row_passes()` with ZCHECK validation |
| recWomKeys/Gather/Fix | C12 only if CPU sort dominates |
| step_verify_bytes | empty, skipped |

**Host glue**: `MetalCircuitHal<MH> { hal: Arc<MetalHal<MH>>, kernels: MetalKernels, scratch: RefCell<ScratchBuffers> }` rewritten from `hal/hip.rs`; witgen = [CPU copy/scatter/mark] → parStepExec → wait → fault → CPU sort/scan → [injectWomBacks → parStepVerifyWom] → wait → fault; accum = one command buffer [fillOneExt → compute accum → prefix trio → verify accum] → wait → fault; eval_check = poly_mix buffer + one dispatch; `recursion_prover`: poseidon2/sha-256 Metal, poseidon_254 CPU until Z10; host-chain cfg gates become a `host_chains_backend()` alias including metal; `metal_warmup()` on a background thread.

**Work items**:
| ID | Item | Files | Depends |
|---|---|---|---|
| C1 | `context.h`, `extern.h` (thread ctx, `bad` flag, clamped `addr`/`iopIdx`/`idx`), `rec_kernels.metal` (injectWomBacks, fillOneExt, prefix trio, row passes, optional host-chain kernels); hand-compile | `recursion-sys/kernels/metal/*` | F1 |
| C2 | `cuda2metal.py` steps mode + `--check` (pinned: 51 Fp(0), 4373+960 asserts, 1768 `device auto&`, 38 noinline, 486+329 c1_bad, 2 trap) + kernel wrappers; compile the four files; cold PSO times | `recursion-sys/kernels/cuda/cuda2metal.py` | C1 |
| C3 | eval mode (monolith) + `--split`; compile; cold PSO times | same script (eval section) | C1 |
| C4 | `build_metal_kernels()` body (python_command, OUT_DIR, `.include(kernels/metal)` only), stale files deleted (if F2 did not), metallib smoke test | `recursion-sys/build.rs`, `recursion-sys/src/lib.rs` names | C2 C3 F2 |
| C5 | `hal/metal/mod.rs`: struct, library/PSO load, scratch growth, fault check helper, `recursion_prover` factory with p254 CPU fallback, cfg gates, `metal_warmup` | `recursion/src/prove/hal/metal/mod.rs`, `hal/mod.rs`, `prove/mod.rs`, `witgen.rs` cfg alias | F0 C4 |
| C6 | Witgen (`hal/metal/witgen.rs`): Parallel path, CPU sort/scan/scatter, Seq → CPU | `hal/metal/witgen.rs` | C5 |
| C7 | Accum (`hal/metal/accum.rs`): one command buffer, prefix trio, equivalence at n = 1, 2047, 2048, 2049, 2^18 | `hal/metal/accum.rs` | C5 |
| C8 | eval_check (`hal/metal/eval_check.rs`): monolith, optional split selection via `RISC0_METAL_REC_KERNELS`, `testutil::eval_check` PO2 4 | `hal/metal/eval_check.rs` | C5 |
| C9 | Tests: witgen/accum equivalence vs CPU (`first_mismatch`), fault path (corrupted element → `Err`), translator `--check` in CI, metallib smoke, existing prove tests (lift/join/identity) on Metal, identity_p254 via CPU fallback | `recursion/src/prove/hal/metal/tests.rs`, `recursion/src/testutil.rs` (new) | C6 C7 C8 |
| C10 | Review: fault/clamp coverage, buffer growth, cfg gates, p254 path, lock interaction with zk-miner threads, memory vs 36 GiB | read + fixes | C9 |
| C11 | Optimization round 1 (parallel sub-tracks): batching A/B/accum/eval into 4 command buffers; row passes + ZCHECK; CPU-cached permutations + Perm kernels; split vs monolith + MTLBinaryArchive; TG sweep; measure CPU sort/scan/scatter; simd_prefix scan | per track | C10 |
| C12 | Optimization round 2: Metal WOM sort if CPU sort dominates; private storage for GPU-only buffers; lazy-REDC vs plain; `HOST_CHAIN_MIN_LEN` retune | per track | C11 |

**Acceptance**: section 2.10 recursion line; `RISC0_RECURSION_PREWITGEN_CHECK`/`SHADOW_CHECK`/`set_host_chains_check` hooks pass; zkvm `host::recursion::tests` e2e; no PSO compile cost on the second proof in a process; witgen+accum per lift/join below the CPU prover (recorded).

**Risks**: translator keyed on exact generated patterns (pinned counts); cold PSO (20.9 s eval_check; step/accum unverified; keccak-unit probe showed pathological stage PSOs); fault semantics (runs to completion before surfacing); occupancy of the 19-arm kernel; no Poseidon254 until Z10; CPU sort of 2.4M rows unmeasured; python3 + xcrun required for every macOS `--features metal` build.

### 4.4 keccak
Owner files: `circuit/keccak-sys/kernels/metal/{witgen.h, eval_check_prelude.h, scatter.metal}`, `circuit/keccak-sys/build.rs` (`build_metal_kernels`, `write_metal_eval_check`, `write_metal_witgen`), `circuit/keccak/src/prove/hal/metal/{mod,witgen,eval_check}.rs`, `keccak/src/prove/tests.rs` metal arm.

**Kernels**:
| CUDA kernel | Decision |
|---|---|
| eval_check (chain of 58 functions, 91k lines, `Fp x13[8237]`) | **must be staged**: `eval_stage_0..N-1` kernels emitted by `write_metal_eval_check` (Intel rewrites + address spaces + explicit carry emission; `cut_into_stages` run **before** stripping `__device__`; `const Fp*` rewrite before `Fp* argN`; `constexpr` → `constant`; kernels at global scope); chunked launcher (C cycles per chunk, start 32768); last stage multiplies by host `quot[cycle & 3]`; `poly_mix` (34.8 KB) as a bound buffer; scratch/carry private, cached |
| par_stepExec (52k lines) | Phase 1: **host witness** into the shared `data` buffer (`HOST_WITNESS = true`); Phase 2: 12 per-arm kernels `step_arm_<K>` = prologue (cols 932–934) + `step_Top_0`/`step_Top_24` + one-hot cols 0..11 + `step_Top_K` (K ∈ 25..33, 50, 51, 52), dispatched per `(cycle_type, sub_type)` bucket; cheap probe of A's noinline whole-tree kernel first |
| fwd/rev_stepExec | not ported; Seq modes = host path |
| scatter_preflight | 1:1 (`ScatterInfo` 16 B repr(C)) |

**Host glue**: `MetalCircuitHal<MH> { hal: Rc<MetalHal<MH>>, kernels, pipelines (via BinaryArchive when cached), scratch/carry private, error flag }`; `keccak_prover()` → `MetalHalPoseidon2`; `prove/mod.rs` arm uses `enabled("keccak")`; `fwd_rev_ab` metal branch; `.flag()` fix (F1) needed for extra MSL flags.

**Work items**:
| ID | Item | Files | Depends |
|---|---|---|---|
| K1 | Build plumbing + skeleton: `build_metal_kernels()` body, `scatter.metal`, `hal/metal/mod.rs` skeleton (library, PSO map, prover factory), prove/mod.rs arm | `keccak-sys/build.rs`, `keccak-sys/kernels/metal/scatter.metal`, `keccak/src/prove/hal/metal/mod.rs`, `prove/mod.rs` | F2 |
| K2 | Milestone 1 prover: host witness into shared buffers + Metal scatter; eval_check bridged to CPU `poly_fp` over `view()`; `basic`/`host_witness`/`fwd_rev_ab` pass on the Metal prover | `hal/metal/{mod,witgen}.rs` | K1 F0 |
| K3 | eval_check translator: `write_metal_eval_check` + `eval_check_prelude.h` (`ScratchRef`, `EvalCtx`), shape asserts, `RISC0_KECCAK_METAL_EVAL_STAGES` | `keccak-sys/build.rs` (fn), `kernels/metal/eval_check_prelude.h` | K1 F1 |
| K4 | eval_check runtime: chunked stage launcher, private scratch/carry, quot[4], poly_mix buffer; `metal::tests::eval_check` PO2 4 → 8/14/17; PSO + GPU time per stage recorded | `hal/metal/eval_check.rs` | K3 F0 |
| K5 | GPU-binary caching: MTLBinaryArchive under the risc0 cache dir keyed by metallib hash + device + OS build; cold vs warm; evaluate `metal-tt` offline build | `hal/metal/mod.rs` (archive), `build_kernel` (if metal-tt) | K4 |
| K6 | GPU witgen phase 2: MSL `witgen.h`, `write_metal_witgen` (12 per-arm kernels), error flag, per-bucket dispatch; noinline whole-tree probe first; switch `HOST_WITNESS` only when equivalent and faster than CPU | `kernels/metal/witgen.h`, `keccak-sys/build.rs` (fn), `hal/metal/witgen.rs` | K1 F0 |
| K7 | Perf round: stage-count sweep 58/29/16/8, chunk 16384..65536, TG 128/256, bucket batching, GPU fill/zeroize, po2 17/18 breakdown vs CPU, Metal System Trace | per track | K4 K5 K6 |
| K8 | Tests + review: full acceptance list, translator asserts, rerun-if-changed/cache coverage, rationale comments | tests | K4 (+K6) |

**Acceptance**: section 2.10 keccak line (M1, then with GPU witgen); `eval_check` at PO2 4/8/14/17 under stage counts {4, 8, 16, 29, 58} and a multi-chunk shape — bit-exact `check`; `metal_scatter_matches_host`; `metal_witgen_matches_host` (po2 8/12/14) + negative test (inconsistent scatter → error flag → `Err`); metallib name assertion; e2e keccak proofs po2 14..18 verify against `KECCAK_CONTROL_IDS`; zk-miner KeccakAudit path.

**Risks**: PSO compile time dominant (whole chain never finished; per-stage 12–65 s but one fat stage 1050 s) — must be cached and never per-dispatch; ~930 dispatches per eval_check at 58 stages (sweep); per-arm witgen must reproduce the two-arm structure exactly (host path is the oracle); lost printf diagnostics; nothing staged has run on the GPU yet; memory at po2 18 (~7–8 GB + 1 GB scratch).

### 4.5 groth16
Owner files: `groth16-sys/kernels/metal/{bn254.h, ntt_params.h, *.metal}`, `groth16-sys/src/metal/{mod,srs,coeffs,prover,msm,ntt,tests}.rs`, `groth16-sys/build.rs` (metal fn), `groth16-sys/Cargo.toml` optional ark deps, `groth16/src/prove/metal.rs`, `groth16/src/prove/mod.rs` cfg widening.

**Kernels** (hand-written MSL, Intel SYCL prover `kernels/intel/{bn254.hpp,groth16_intel.hpp,ntt_params.hpp}` as template):
| Metal kernel | Replaces | Phase |
|---|---|---|
| `bn254.h` (8×32-limb CIOS Montgomery with `mulhi`, Fp2, Affine/Jacobian/Xyzz, add_affine 8M+2S) + `bn254_test.metal` | sppark `mont_t.cuh` (PTX) | 1 |
| `fr_mul`, `fr_mul_sub_from_mont`, `fr_add` | coeff_wise_mul/_and_sub/_add | 1 |
| `witness_into_poly` | witness_into_poly (sequential sum, bit-identical) | 1 (compact: 3) |
| `chacha_fuzz` | chacha_generate_random_scalars<8> (bit-exact layout: `idx = lane + wid*64`, `l[7] &= 0x1FFFFFFF`) | 1 |
| `fr_powers`, `fr_scale` | twiddle generators | 1 |
| `bit_reverse`, `ntt_stage` (radix-2), `ntt_fused_stages` (radix-2^9, 512 Fr = 16 KB TG) | sppark GS/CT NTT + LDE_distribute_powers | 1 / 2 |
| `msm_hist_count`, `msm_window_scan`, `msm_scatter` | breakdown + sort/sort_y (grid syncs, PTX) | 1; Booth digits / privatized histogram in G9 |
| `msm_buckets_g1/g2`, `msm_segment_reduce`, `msm_window_reduce` | accumulate + integrate* + host collect | 1; G2 lane pairs in G9 |
| split_* / witness_into_poly_compact | cold paths | 3, behind `RISC0_METAL_G16_COLD=1` |
| not ported | bitmap batch_addition, bit_rev_permutation_z, diag_g2_double, cuda_setup | — |

**Host glue**: `groth16-sys/src/metal/` Rust: `metal_preload/prove/release` with the `intel_*` signatures; zkey/coeffs/fuzzed loaders into shared buffers; `MetalProver` with cached PSOs, 3–4 command buffers per proof, host Horner + blinding in ark-ec, proof/public JSON in the CUDA format; `groth16/src/prove/metal.rs` = copy of `intel.rs` with `RISC0_GROTH16_METAL_PROVER=docker` fallback; `component`/`fast`/`docker::fast_enabled` cfgs widened to `metal`; whole prove under `risc0_zkp::hal::metal::exclusive()`.

**Work items**:
| ID | Item | Files | Depends |
|---|---|---|---|
| G1 | groth16-specific plumbing beyond F2: `build_metal_kernels()` body, ark optional deps, `pub mod metal` with stubs, `prove/metal.rs` (intel.rs copy), cfg widening, zkvm feature line; `cargo check` with and without `metal` | `groth16-sys/{build.rs,Cargo.toml,src/lib.rs,src/metal/mod.rs}`, `groth16/{Cargo.toml,src/prove/{mod,metal}.rs}` | F2 |
| G2 | `bn254.h` + `bn254_test.metal` + Rust harness + ark equivalence tests (0/1/p−1/R−1, infinity, P+P, P+(−P)) | `kernels/metal/bn254.h`, `bn254_test.metal`, `src/metal/tests.rs` | F1 |
| G3 | `srs.rs`/`coeffs.rs`/fuzzed loader; VK == `risc0_groth16::verifying_key()` test (settles ark Montgomery compatibility) | `src/metal/{srs,coeffs}.rs` | G1 |
| G4 | eltwise/witness_poly/fuzz/powers kernels + tests (fuzz vs Rust ChaCha8 reference) | `kernels/metal/{eltwise,witness_poly,fuzz,powers}.metal`, `src/metal/small.rs` | G2 |
| G5 | NTT v1 + `NttPlan`; roundtrip + ark-poly coset tests (lg 10/16/23) | `kernels/metal/{bit_rev,ntt}.metal`, `src/metal/ntt.rs` | G2 |
| G6 | MSM v1 (digits/sort/buckets G1+G2/reduce) + `msm.rs`; tests vs ark `VariableBaseMSM` (n ∈ {1,31,32,33,1000,2^16,2^20}, scalars {0,1,p−1}, 7 % infinities, C offset) | `kernels/metal/msm_*.metal`, `src/metal/msm.rs` | G2 |
| G7 | `prover.rs` orchestration, `metal/mod.rs` API, install the rzup component, `MSM(P,f)+fuzzed == O` for A/B1/B2/C, e2e `shrink_wrap` on the identity_p254 fixture verified ×3, release + reprove | `src/metal/{prover,mod}.rs` | G3 G4 G5 G6 |
| G8 | Review: cfg hygiene, error/mutex paths, proof-format parity, pre-blinding dump under `RISC0_VERBOSE` | read + fixes | G7 |
| G9 | Optimization round 1 (measure first): G2 lane-pair accumulate, Booth digits + ones row + wbits sweep, privatized-histogram sort, NTT v2 fused, `witness_par` into the shared buffer, Shared vs Private scratch | per track | G7 |
| G10 | Optimization round 2: deferred-collect overlap via completion handlers, zero-copy mmap of the zkey (probe), compact coefficients + Rust disk cache (backend-tagged key), sppark 3-pass NTT only if fused loses | per track | G9 |
| G11 | Optional: bit-split MSMs behind `RISC0_METAL_G16_COLD=1` | split kernels | G10 |

**Acceptance**: `cargo test -p risc0-groth16 --test verify` over every Metal proof; new gating unit tests (bn254, ChaCha, NTT, witness_into_poly, MSM, VK parse); `MSM(P,f)+fuzzed == O`; `RISC0_GROTH16_DIR=<component> cargo test -p risc0-groth16 --features metal metal_shrink_wrap` ×3; `public.json == seal_to_json`; zkvm `shrink_wrap`/`compress` cfgs widened to `metal`; `cargo check` with/without `metal`; perf: per-stage ms, target wrap < 2 s on M5 Max vs Docker baseline (recorded).

**Risks**: MSM throughput on emulated 254-bit arithmetic; 32 KB TG ceiling (designs stay ≤ 16 KB + host assert); device-atomic contention; bit-exactness dependencies (fuzz layout, root tables, R = 2^256) — gating tests, not optional; ark vs sppark Montgomery form (G3 test); front-end time of inlined 256-bit kernels; unified-memory cache policy (4.5–5 GB resident); feature plumbing must land together or Docker silently stays; macOS hang detection on multi-second dispatches (keep dispatches sub-second).

### 4.6 Integration / zk-miner (M)
| ID | Item | Files | Depends |
|---|---|---|---|
| M1 | zk-miner worker: `metal` feature + `zkminer-prove-risc0-metal` bin, `detect_gpu()` metal branch (`risc0_zkp::hal::metal::device_name()`), unified-memory VRAM semantics in `release_groth16_cache_if_tight`, optional macOS RSS self-guard, `warm_up()` building PSOs | `zk-miner/crates/zkminer-prove-risc0/{Cargo.toml,src/main.rs}` | F2, R11 (first real Metal prover) |
| M2 | zk-miner host side (section 3) | `zkminer-prover/src/*` | — |
| M3 | zkvm orchestration metal arms: `exclusive_gpu()` → `metal::exclusive()`, `release_hal_buffers`/`release_cached_provers`, keccak host-witness threads, `_graph_preload`, `metal_warmup` thread, succinct po2 bump decision | `zkvm/src/host/server/prove/{prover_impl,mod}.rs` | F2 |
| M4 | Docs + targets: README macOS section, `Makefile.mac` targets (`mac-check`, `mac-test-*`, `mac-build-worker`, `mac-install-worker`, `mac-test-miner`, `mac-bench`), env.sh note, `NVCC=off HIPCC=off`, `-j ≥ 4` | `zk-miner/README.md`, `zk-miner/Makefile.mac`, `docs/metal-port/` | M1 M2 |
| M5 | End-to-end + baseline: build/install the worker, `comprehensive_zkvm_test`, a Groth16 wrap (native if G7 landed, else Docker), the worker benchmark run, timings recorded in `docs/metal-port/BASELINES.md` | docs | M1 M2 M3 (+G7) |

---

## 5. Execution schedule

Granularity: **one agent per item ID**, owning exactly the files listed for that item; two concurrently running agents never edit the same file. Shared files (`Cargo.toml`s, `build.rs`, `hal/mod.rs`, `prove/mod.rs`, `fp.h/fpext.h`, `zkp/src/hal/metal.rs`) are edited only by their foundation owner (F1/F2/F0) in wave 0 and afterwards only by the single unit item that lists them. Each item ends with: its tests green, its unit's pre-existing tests green, a validation-layer run, and a short review note (what was transliterated from which CUDA lines, what was verified vs assumed).

| Wave | Items (parallel within a wave) | Gate to next wave |
|---|---|---|
| **0 — foundation** | D0, F0, F1, F2, M2 (5 agents; disjoint files) | F0+F1+F2 merged; `cargo test -p risc0-zkp --features metal hal::metal` 18/18; `cargo check -p risc0-zkvm --features metal` green with CPU-fallback stubs |
| **1 — kernel fan-out** | Z1 Z2 Z3 Z4 Z5 · R1 R2 R7 · C1 · K1 K3 · G1 G2 (13 agents). R2/C1/K3/G2 only need F1 and may start during wave 0 against scratchpad hand-compiles | each item's compile + unit tests |
| **2 — translators & phase-1 host paths** | R3→R4/R5/R6 · C2 C3 → C4 · K2 K6 · G3 G4 G5 G6 · Z6 · M3 | TUs compile, PSO times recorded, phase-1 submodules exist |
| **3 — integration per unit** | R8 R9 R10 → R11 · C5 → C6 C7 C8 → C9 · K4 → K5 · G7 · Z7 · M1 | section-6 conformance green per unit; seals verify |
| **4 — review rounds** | Z8 · C10 · K8 · G8 · R11 fixes · R12 (evalfast) · R16 · M4 | review notes filed; no open P0 |
| **5 — optimization rounds** (measure first, per kernel family, independent sub-tracks) | Z9 · R13 R14 R15 · C11 C12 · K7 · G9 G10 (G11) · Z10 (after G2) → recursion p254 switch | baselines in `BASELINES.md` improved and re-verified bit-exact |
| **6 — end-to-end** | M5 benchmark + Groth16 wrap on the miner | worker scheduled by zk-miner on the Mac |

Critical path: F0/F1/F2 → R2 → R3 → R5 → R8 → R11 → M1 → M5 (rv32im is the first circuit the worker needs; recursion C-chain runs beside it; keccak and groth16 can trail without blocking the worker because `enabled()`/Docker fallbacks keep them on CPU).

Agent-hour totals from the unit plans: zkp ≈ 96, rv32im ≈ 130, recursion ≈ 68, keccak ≈ 68, groth16 ≈ 78 (+18 optional), infra/plumbing ≈ 70. Treat as relative weights only.

---

## 6. Verification plan

### 6.1 CPU-vs-Metal conformance per kernel
- **zkp**: `DualHal::new(CpuHal, MetalHal)` testutil for every Hal method (existing 18 + Z6 additions), on `COUNTS [1, 9, 12, 1001, 1024, 1025, 1M]` and the po2-specific shapes listed in Z6; legacy-vs-ported exact equality until Z7; a full `prove`/`verify` round trip on DualHal.
- **rv32im**: `hal/metal/tests::eval_check` (random canonical inputs, po2 4/10/16, baseline and evalfast); `metal_matches_cpu` (same `PreflightResults`: CPU vs Metal `data`+`global` after witgen+zeroize, then all 103 accum columns incl. the 4 scanned and the finalized machine columns, `first_diff (row, col)`), repeated N times; `fp_scan`, `perm_equivalence`.
- **recursion**: `testutil::eval_check` PO2 4; witgen equivalence (Parallel with/without host chains, Seq modes); accum equivalence incl. the prefix trio at n = 1, 2047, 2048, 2049, 2^18; monolith vs split bit-identity; fault-path test.
- **keccak**: `testutil::eval_check_impl` PO2 4/8/14/17 across stage counts and a multi-chunk shape; `metal_scatter_matches_host`; `metal_witgen_matches_host` + negative test.
- **groth16**: field/Fp2/curve vs ark; ChaCha bit-exact vs a Rust ChaCha8 reference (n = 1..5000, tail); NTT roundtrip + coset evaluation vs ark-poly; `witness_into_poly` vs CPU on the real coefficient file; MSM G1/G2 vs ark; VK parse == `verifying_key()`; `MSM(P,f)+fuzzed == O`.
- Every conformance module runs once under `RISC0_METAL_DEBUG=1 MTL_DEBUG_LAYER=1`.

### 6.2 End-to-end per circuit
- rv32im: `prove::tests` (basic, multi_read, 46 riscv cases), `witgen::tests::fwd_rev_ab_*`, seal equality vs CPU for fixed rand_z, multi-segment composite verifies.
- recursion: lift/join/resolve/identity tests on Metal; `identity_p254` (CPU fallback until Z10); zkvm `host::recursion::tests` e2e.
- keccak: `prove::tests` (basic po2 8, host_witness, fwd_rev_ab); example proofs po2 14..18 verified against `KECCAK_CONTROL_IDS`; zkvm keccak tests.
- groth16: `metal_shrink_wrap` ×3 on the identity_p254 seal; `public.json == seal_to_json`; zkvm `compress`/`shrink_wrap`/`verify_in_guest` with cfg widened to `metal`; Docker vs Metal both verify for one seal.
- zkvm: `metal_implies_prove`; the fibonacci/sha256-chain guests to composite and succinct, verified on CPU.

### 6.3 zk-miner worker
- `cargo test -p zkminer-prover discovery:: dispatcher::` (metal tag, `gpu-metal0`, env), `cargo test -p zkminer-prove-risc0 --features metal`, `--features testing --test worker_lifecycle --test measure_only`, and with the worker installed at `~/.zkminer/provers/zkminer-prove-risc0-metal`: `cargo test -p zkminer-prover --features risc0 --test comprehensive_zkvm_test`.
- The worker benchmark run: `WorkerCommand::Warmup` → `warm_up()` → `run_po2_calibration` through `zkminer simulate` (the exact `--benchmark` CLI flag of the worker is **unverified**; use whatever the worker's existing calibration entry point is and record the command in `BASELINES.md`).

### 6.4 Perf baselines to record (`docs/metal-port/BASELINES.md`, measured with no other GPU work running)
- CPU prover on the same M5 Max: per-segment wall time at po2 18/20/21 (fibonacci guest), recursion lift/join, keccak po2 17 proof, Docker Groth16 wrap.
- Metal: per-phase wall time around waits (witgen ×2, step_accum, scans, finalize, eval_check main/LogUp) at po2 20/21; zkp per-method timings at po2 20/21/22 × 211 columns; NTT GB/s by radix/chunk; Poseidon2 perms/s; mdiv vs CPU by po2; eu2 S sweep; keccak PSO build and GPU time per stage, stage-count sweep; groth16 per-stage ms and total wrap; command-buffer and `sync()` count per proof; PSO creation time cold/warm (and with BinaryArchive); `xcrun metal` compile time per TU; peak `current_allocated_size` vs 30.15 GB working set; pool hit rate.
- Regression gate: per-kernel GPU time vs the stored baseline (reported, not asserted, until the optimization rounds settle).

---

## 7. Open questions and unverified claims (consolidated)

### Open questions (decision owner in parentheses)
1. `[[host_name]]` on explicit template instantiations, cross-`.air` extern resolution, designated initializers in `constant` tables, `assert()` release semantics, `max_total_threads_per_threadgroup(768)` (D0).
2. Runtime validity of `gpu_address()` values inside a `set_bytes` struct — compiles, never executed (D0, Z5).
3. mdiv at po2 22 (two-level scan vs MD_E=16 vs CPU) and po2 < 16 (single-TG vs CPU) (Z5).
4. mix v2 gate: 1638 vs CUDA's 2048 by reading cids from device (Z4).
5. One command queue or two for eval_check overlap; whether a second queue gives overlap at all (Z9, R14).
6. Encoder/command-buffer encoding limits per batch (F0, measured).
7. nttq retune under 32 KB go/no-go (Z9).
8. Compile and PSO creation time for the 19k-line witgen and 49k-line accum TUs; inline strategy; TU layout (R4, R5).
9. Does macOS kill long single-thread command buffers (seq modes, parStepExec with long chains)? (D0 watchdog probe.)
10. Single-buffer witgen bit-exact on Apple without a compiler barrier (R15).
11. evalfast branch point (HIP_LAST vs before close/linkext) given unmeasured 64-bit MAC cost (R12/R15).
12. eval_check default monolith vs split; host chains default and `HOST_CHAIN_MIN_LEN` on Metal (C11).
13. MTLBinaryArchive need/behaviour for cold starts; `metal-tt` offline builds from build.rs (K5).
14. keccak: optimal stage size; whether a noinline whole-chain/whole-tree kernel builds a PSO in minutes (K3/K6 probes).
15. groth16: actual witness length / domain size of the rzup component (install first); where host group ops live (ark in `-sys` vs `risc0-groth16` vs Rust `bn254.hpp` port); Booth vs unsigned windows; keep-vs-release SRS policy on unified memory (G3, G6, G9).
16. zk-miner: what VRAM figure to report for unified memory; host-memory ledger behaviour on macOS; process-per-job vs threads and the singleton lock; should the Metal worker advertise itself while groth16/keccak are still CPU/Docker (M1, M2).
17. Column chunking for groups above `max_buffer_length` (22.6 GB) — needed for any production shape? (Z1 errors early for now.)
18. Apple GPU core count and last-level cache size for sm_count/L2-based chunk sizing (no metal-rs query) (Z9 sweep).
19. metal-rs 0.29 vs objc2-metal before three more HALs are written against the API (decided: stay on 0.29 for round 1).
20. Self-hosted Mac CI in scope? (out of scope for round 1; the `Makefile.mac` targets are the manual CI.)

### Unverified claims carried from the unit plans (must be measured or probed before being relied on)
- Hardware: 32 GPU cores; 500–600 GB/s bandwidth; 64-bit integer MAC cost; M5 `MTLGPUFamily` value vs metal-rs enum; counter sampling at dispatch boundary; command-buffer watchdog threshold.
- All per-stage time estimates in the plans (NTT ~40 ms, commit 150–200 ms, hash_rows 15–30 ms, eval_check 300–600 ms, witgen+accum 20–60 ms, wrap 1–3 s, 50–200 µs command-buffer overhead).
- Cold PSO times for translated recursion step/accum kernels (1.74/1.28/0.06 s reported by one planner, no log) and for `ec_part` splits; keccak noinline-chain "stack space" failure at 101 s; keccak whole-chain/whole-tree "did not finish" wall times (consistent with artifacts, no elapsed record).
- CPU-generated twiddle tables matching sppark `NTTParameters` layout/order bit for bit (values exact; order unverified).
- `simd_shuffle` out-of-range lanes discarded in every reduction (eu2/mdiv per comments; pvs not audited).
- Metal `shaHash` (3-arg) vs CUDA `shaHash(..., false)` padding equivalence beyond the passing tests.
- rv32im `combos_divide` slice count (5) and keccak mix table size (~1100 inputs ≈ 22 KB).
- `Heap::new_buffer_with_offset` in metal-rs 0.29; `new_buffer_with_bytes_no_copy` on a page-aligned read-only mmap on macOS 27; page-aligned preflight Vecs.
- `__attribute__((noinline))` honoured by the Apple back-end for the 19 recursion arms / 388 rv32im functions; register/occupancy profile of 15–60k-line functions.
- `assert()` in `<metal_stdlib>` being a release no-op (if not, `cuda2metal.h` redefines it).
- Shared-buffer CPU/GPU coherence at command-buffer boundaries without `didModifyRange` (Apple-documented; not probed here).
- `witgen::host_ctrl_for` existence for recursion permutation keys (keys derivable from Program rows regardless).
- CPU cost on M5 of: recursion `sortWomFast`/scan/scatter/128 MB zeroize; keccak witgen (1.3 s figure is from Intel hardware); rv32im counting sort.
- ark-bn254 Montgomery form (R = 2^256) matching sppark/zkey limbs (gated by the VK test); Intel `generate_fuzz` bit-exact with the CUDA kernel (gated by `MSM(P,f)+fuzzed == O`).
- Device-atomic contention on Apple for the MSM histogram/scatter; G2 register pressure.
- The AIR compiler not rejecting oversized static threadgroup arrays (host assert covers it).
- metal-rs 0.29 lacking `gpu_start_time/gpu_end_time` and `CommandBuffer::error()` (grep-level; use `msg_send!` if needed).
- po2 22 feasibility in 36 GiB and the zkp `low_mem` path on shared buffers; wired-memory behaviour of multi-GB shared buffers on macOS 27.
- Whether the `-std=metal4.0` flag is accepted (only `metal3.2` was used).
- Existence of an arm64 Docker image for the Groth16 prover (fallback until G7).
- The exact benchmark CLI entry point of the zk-miner worker (`--benchmark` as named by the task harness).

---

## 8. Errata from the completeness review (binding; supersedes conflicting text above)

Source: completeness critic run on 2026-11-09, verified against code and with `xcrun metal` probes (full text: `docs/metal-port/CRITIQUE.md`).

### 8.1 Feature wiring (replaces the F2 feature bullet; F2 now owns **every** `metal` feature line, including groth16)
- `zkvm/Cargo.toml`: `metal = ["prove", "risc0-circuit-keccak/metal", "risc0-circuit-recursion/metal", "risc0-circuit-rv32im/metal", "risc0-groth16/metal", "risc0-zkp/metal"]`.
- `zkp`: `metal = ["prove", "dep:metal", "risc0-sys/metal"]`; the macOS/iOS `metal` dependency becomes `optional = true`.
- `circuit/rv32im` and `circuit/keccak`: new `[target.'cfg(any(target_os = "macos", target_os = "ios"))'.dependencies] metal = { workspace = true, optional = true }`; `metal = ["prove", "dep:metal", "risc0-circuit-<x>-sys/metal", "risc0-zkp/metal"]`.
- `circuit/recursion`: `metal = ["prove", "dep:metal", "risc0-circuit-recursion-sys/metal", "risc0-zkp/metal", "std"]` (its metal dep also becomes optional).
- `rv32im-sys`, `recursion-sys`, `keccak-sys`: `metal = ["risc0-sys/metal"]` (keccak-sys gains the feature).
- `groth16`: `metal` mirrors the `intel` list (`dep:risc0-groth16-sys`, `dep:rayon`, `dep:circom-witnesscalc`, `dep:wtns-file`, `prove`, `risc0-groth16-sys/metal`, `risc0-zkp/metal`). `groth16-sys`: `metal = ["dep:risc0-sys", "risc0-sys/metal", "dep:metal"]` plus an optional target-gated `metal` dependency.
- F2 adds the feature lines and compiling stubs for groth16 too (`pub mod metal` stub, every cfg site below gets a `metal` arm that falls back to Docker until G7). G1 then owns the groth16 metal module contents and no longer edits Cargo.toml.
- groth16 cfg sites that must gain a `metal` arm (otherwise Docker silently stays): `prove/mod.rs` `mod component`, `mod docker`, `mod fast`/`mod intel`, `preload_graph`, `preload_graph_background`, `prestart_prover`, `prestart_key`/`prestart_key_go`, `preload_srs`, `release_srs`, `invalidate_caches`, `shrink_wrap_with`, `shrink_wrap`, and `docker.rs` `fast_enabled` (×2).
- zk-miner (M1): `metal = ["risc0-zkvm/metal", "risc0-groth16/metal", "risc0-circuit-keccak/metal"]` and `[[bin]] name = "zkminer-prove-risc0-metal", required-features = ["metal"]`.
- zkvm GPU-gated tests that M3/C9 widen with `feature = "metal"`: `zkvm/src/host/server/prove/tests.rs` (`compress`, `fake_compress`, `shrink_wrap`, `verify_in_guest`) and `zkvm/src/host/recursion/tests.rs` (`test_recursion_host_chains_check`).

### 8.2 File ownership fixes (no two concurrently running items share a file)
- keccak: K3's translator lives in a new `keccak-sys/build_metal_eval.rs` (`#[path]` module of build.rs); K1 adds only the `mod` line. K6 moves to wave 3 and starts after K2 merges; K6's witgen translator lives in `keccak-sys/build_metal_witgen.rs`.
- recursion: C2 owns `cuda2metal_common.py` + `cuda2metal_steps.py`; C3 owns `cuda2metal_eval.py` and depends on C2.
- rv32im: R15 sub-tracks start after R13 and R14 merge.
- Wave 0 sequencing: F2 must merge before F0 starts (F0 tests need `--features metal`, and both touch `zkp/src/hal/mod.rs`: F2 owns the `pub mod metal` cfg line, F0 owns everything else in that file's metal-related code).
- Placeholder `kernels/metal/noop.metal` created by F2 is deleted by the unit's first kernel item (R3, C4, K1, G1), which also removes it from `METAL_KERNEL_NAMES`.

### 8.3 Inventory additions
- recursion: translate `step_verify_mem.cu` (a `__device__` fn called by the VerifyWom kernels) in steps mode alongside `step_exec.cu`, `step_compute_accum.cu`, `step_verify_accum.cu`; `step_verify_bytes.cu` and `sppark.cu` have no kernels. Host externs: eval_check/warmup → C8/C5 `metal_warmup`; pinned `_host_alloc/_host_free` → none (non-pinned `pinned_ptr` stub path).
- zkp: `k_coset_tabs/k_line_tabs/k_pow_table/pass_h/pass_k` (nttq only), `bit_rev_t2_batched/bit_rev_t2_oop` (HIP/nttq), `fold_tail_q4_k` → not ported; `bit_rev_permutation` → existing naive Metal bit-reverse for n < 4096; `combos_prepare` → Z1; `sppark_init`/`sppark_poseidon2_init` → CPU-built tables (Z2/Z3); `risc0_zkp_cuda_warmup` → `MetalHal::new`; `set_oom_hook` → pool drain-and-retry; `sync_stream` → `sync()`.
- groth16: `batch_diff` not ported; `split_bit_sum` with phase-3 split_*; `integrate_combine/integrate_split` folded into `msm_window_reduce`; host entries `prove/preload/release` → `metal_*`; `prestart`/`stager_init`/`pinned_release`/`host_buffer`/`set_bit_signals` → no-ops on unified memory.

### 8.4 Trait methods
- Every circuit Metal HAL implements `CircuitHal::accumulate` exactly as its `cuda.rs` does (R9, C7, K2).
- recursion `upload_host_witness` = memcpy into shared `data`/`global` (C6); `eval_check_dep` no-op for recursion and keccak.
- keccak phase 1: `type PreferredPreflightOrder = ForwardPreflightOrder`; phase 2 (K6): `MetalPreflightOrder` keyed by (cycle_type, sub_type). State whether the Metal HAL needs a `Drop` (CUDA has one).
- zkp: `trim_device_memory` drains the pool (F0); `prefill_elem` default no-op; `batch_get_digest_at` default; `batch_interpolate_ntt_zk_shift_from` is part of Z2.

### 8.5 Probe facts (verified with xcrun metal 32023.922)
- `mulhi` on 32 and 64-bit: accepted. `atomic_ulong` fetch_add: rejected. `-std=metal4.0`: accepted.
- Static `threadgroup` arrays above 32 KB compile and link without a diagnostic: the host-side 32 KB assert is mandatory; D0 adds a runtime PSO/dispatch probe.
- Explicit template instantiation with `[[host_name("...")]]` compiles and links across `.air` files; D0 verifies runtime lookup.
- `assert()` from `<metal_assert>` is `((void)0)` in all modes.
- `threadgroup Fp x[8]` with today's fp.h: rejected (F1 fixes).

### 8.6 Command and tooling corrections
- `cargo test -p zkminer-prover -- discovery:: dispatcher::` (one filter before `--`).
- Worker lifecycle tests: `cargo test -p zkminer-prover --features testing --test worker_lifecycle --test measure_only` (they live in zkminer-prover).
- D0 adds `build_kernel/tests/metal_probes.rs` (new file; not F1's `src/lib.rs`).
- The non-macOS target guard check needs `rustup target add x86_64-unknown-linux-gnu` first.
- The worker's `--benchmark` flag is verified (`zkminer-prove-risc0/src/main.rs:68`).
- F1: `compile_metal` must strip the source prefix before joining into `OUT_DIR` so `.air` files for absolute generated sources land under `<OUT_DIR>/<output>/`; `cuda2metal.h`/`sppark_compat.h` are appended to the existing `METAL_INCS` const.

---

## 9. Target hardware and wave-1 findings (binding from wave 2 on)

Source: `/Users/user/zkminer/reports/Mac hardware prosumer distribution.md` (research, 2026-11-10) and the wave-1 item notes.

### 9.1 Target hardware matrix
The development machine (M5 Max, 40 GPU cores, 36 GB) is a **high-end** machine, not the tuning target. Steam (Oct 2026) puts Apple-GPU Macs at ~67% base chip, 21% Pro, 7% Max, Ultra unlisted; 16 GB is the most common memory size (43%), 8 GB still 29%.

| Tier | Hardware | GPU budget, owner active / idle | Expectation |
|---|---|---|---|
| Not supported | Intel Macs, any 8 GB Mac, MacBook Neo (A18 Pro) | — | worker refuses with a clear message |
| Minimum | M1–M6 base chip, 16 GB, 7–10 GPU cores, 68–170 GB/s | ~4 GB / ~7–8 GB | must prove correctly; small segments (po2 ≤ 18 active, ≤ 19 idle); MacBook Air derated (fanless) |
| Primary tuning target | M4/M5 base 10-core at 24 GB; M4/M5 Pro 16/20-core at 24–48 GB | 6–12 GB / 12–28 GB | default segment size; tune here |
| High end | Max 36–128 GB, Ultra 96–512 GB | 16–64 GB / 40–180+ GB | concurrency, large segments; first tier for SP1 |

Binding consequences:
- **Never assume a fraction of RAM.** Read `recommendedMaxWorkingSetSize` at runtime (≈75% of RAM by default, 84% measured on one 128 GB machine) and size segments/pools from it minus a host reserve.
- **Peak GPU memory per phase is a first-class metric.** Every circuit item reports peak `current_allocated_size` per phase at the po2 values it tests, and the conformance suites run once under an artificial budget: `RISC0_METAL_MAX_WORKING_SET_BYTES=<n>` (F0 follow-up in Z6: the HAL treats it as the working-set limit and errors early when exceeded). The budgets to test are 4 GB and 8 GB.
- **Tune for 8–10 and 16–20 GPU cores**, not 40: grid sizes, chunking and crossovers must not hard-code core counts; prefer occupancy-driven sizing. Optimization rounds (wave 5) record results at simulated small budgets and note anything that depends on the 40-core GPU.
- **Cold-start cost matters on consumer machines.** Keccak's 58-stage eval_check takes ~94 s of cold PSO creation (K3), instant once the OS shader cache is warm. The worker must warm PSOs once at install/first start (M1, K5), never inside a job deadline.

### 9.2 Wave-1 findings that change later items
- **64-bit vs 32-bit arithmetic is kernel-specific.** D0 measured 64-bit mulhi ~5x slower than 32-bit mul in a microbenchmark, but Z3's Poseidon2 int64 HIP form beat its 32-bit mulhi form at every size. Optimization rounds measure both forms per kernel; neither is a blanket rule.
- **GPU loses below po2 16 for combos_divide** (Z5: CPU 0.38 ms vs GPU 1.1 ms at po2 13) and similar small-size overheads exist elsewhere: Z9 sets per-method CPU/GPU crossovers.
- **Poseidon2 crossovers** chosen on a shared GPU (pvs_w ≤ 256, pvs_q ≤ 2048, pv above) are provisional (Z9).
- **groth16-sys now has optional arkworks dependencies** (ark-bn254/ec/ff/poly/serialize/std behind `metal`) and `rand`/`rand_chacha` dev-dependencies; G3–G6 use them and do not edit Cargo.toml.
- **Open cross-item issues carried into wave 2:** the `poly_group.rs::new_impl` coeffs host spill still runs on unified memory (owner: Z6); `fpext.h` lacks mixed Fp±FpExt operators, so R2's translator ships a shim (owner: R3, coordinate with zkp-hal); F0's runtime re-sets threadgroup memory length redundantly (owner: Z6); fri.metal/mix.metal scalars should be `constant` (owner: Z6).
