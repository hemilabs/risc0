# Day-0 Metal probes (plan item D0)

Facts about the Metal shader compiler and runtime that the Metal port relies on. Every row was
measured on this machine, not taken from documentation:

- Apple M5 Max (`applegpu_g17s`), macOS 27.2, Metal toolchain 32023.922 (`metalfe-32023.922.4`), metal-rs 0.29.0
- measured 2026-11-09, with other agents' builds running (timings are indicative only)

Tooling:

- **Compile facts** (`EXPECT: ACCEPT|REJECT` line at the top of every `*.metal`) are checked in CI by
  `cargo test -p risc0-build-kernel metal_probes` (`risc0/build_kernel/tests/metal_probes.rs`).
  It compiles every probe with `xcrun metal -std=metal3.2 -O2`, checks the two reject
  diagnostics, links the extern pair, and preprocesses `assert`.
- **Runtime facts** come from the standalone runner in `runner/`. It is not part of the risc0
  workspace: its `Cargo.toml` has an empty `[workspace]` table. The runner compiles the probes with
  the same offline toolchain, loads them with `new_library_with_data` and runs them on the GPU.
  `all` runs each case in a child process, so a crash shows up as `CRASHED` and does not stop
  the other cases.

```sh
source /Users/user/zkminer/env.sh
# from the risc0 repo root; R abbreviates the runner command used in the table:
R="cargo run --release --manifest-path risc0/build_kernel/kernels/metal/probes/runner/Cargo.toml --"
$R all              # every case below except the watchdog (~5 s)
$R watchdog 60      # 60 s single-thread dispatch
```

## Results

| Probe (source) | Result | Value / observation | Command |
|---|---|---|---|
| `[[host_name]]` on explicit template instantiations (`host_name.metal`) | **ACCEPT** | `function_names()` = `["ct_x_1_plain","ct_x_8_coalesced"]`. Both are found by name and produce correct output. The bare template name `ct_x` is not in the library. | `$R host-name` |
| Cross-`.air` extern function (`extern_use.metal` + `extern_def.metal`) | **ACCEPT** | `xcrun metallib use.air def.air` links and runs correctly (`out = 5*gid+1`). Linking `extern_use.air` alone fails at **metallib link time** with `LLVM ERROR: Undefined symbol: _Z6ext_fnj`, so a missing definition is never deferred to runtime. Extern functions use C++ mangling. | `$R extern` |
| Designated initializers in `constant constexpr` tables (`designated_init.metal`) | **ACCEPT** | Array-of-struct and scalar-struct tables. An omitted field (`.b`) is zero-initialised. Values verified at runtime. | `$R designated` |
| `gpu_address()` values in a `set_bytes` struct, dereferenced in the kernel (`gpu_address.metal`) | **ACCEPT only with `use_resource`** | Passing `struct { device uint* ptrs[4]; uint n; uint _pad; }` (40 B, host `[u64;4]+u32+u32`) via `set_bytes` and calling `enc.use_resource(buf, Write)` for every pointed-to buffer: 0/16384 mismatches. **Without `use_resource`, all 16384 writes are silently lost.** Status stays `Completed` and the validation layer (`MTL_DEBUG_LAYER=1`) reports nothing. Any K5 `MdJobs`-style struct must call `use_resource` on every referenced buffer. | `$R gpu-address use-resource` / `$R gpu-address no-use-resource` |
| Static `threadgroup` array, 32 KB (`tg_static_32k.metal`) | **ACCEPT** | PSO `staticThreadgroupMemoryLength=32768`, maxTotalThreads 1024, dispatch correct. | `$R tg-static 32k` |
| Static `threadgroup` array, 36 KB (`tg_static_36k.metal`) | Compile **ACCEPT**, PSO **REJECT** | The compiler and `metallib` accept it with no diagnostic. `newComputePipelineState` fails with `Threadgroup memory size (36864) exceeds the maximum threadgroup memory allowed (32768)`. The failure is clean and happens at PSO creation, so it shows up when `MetalKernels::new` builds the PSO. | `$R tg-static 36k` |
| Dynamic threadgroup memory via `set_threadgroup_memory_length` (`tg_dynamic.metal`) | ≤ 32768 **ACCEPT**; above it **undefined** | `device.maxThreadgroupMemoryLength = 32768`. 16384 and 32768 are correct. 32784, 36864 and 65536 run without error and return correct data, but this is outside the API contract: with `MTL_DEBUG_LAYER=1` the process aborts with ``-[MTLDebugComputeCommandEncoder setThreadgroupMemoryLength:atIndex:] failed assertion `length(32784) must be <= 32768.'`` (checked for 32784 and 36864). 131072 gives **silent corruption** (16384 bad words) while status still reads `Completed`. The plan's host-side assert `tg_bytes ≤ 32768` (2.1/2.9) is mandatory. | `$R tg-dynamic <bytes>`; `MTL_DEBUG_LAYER=1 $R tg-dynamic 32784` |
| Counter sampling (no shader; uses `dispatch.metal`) | Dispatch boundary **REJECT**; stage boundary **ACCEPT** | Counter sets: only `timestamp` [`GPUTimestamp`]. `supportsCounterSampling`: stage=true, draw/dispatch/tile/blit=false. So per-dispatch timing must split work into separate encoders, or use command-buffer `GPUStartTime/GPUEndTime` (works through `msg_send!`). Stage-boundary samples through `MTLComputePassDescriptor.sampleBufferAttachments` work when resolved on the CPU with `-[MTLCounterSampleBuffer resolveCounterRange:]`. Example: delta 122458 ticks against a command-buffer GPU time of 0.127 ms; the CPU/GPU tick ratio from `sampleTimestamps` is 1.0, so ticks are ns. A blit `resolveCounters` into a shared buffer returned zeros (Shared-storage sample buffer). Unsampled slots resolve to 0. | `$R counters` |
| GPU family (no shader) | **VALUE** | `supportsFamily`: Apple7/8/9 = true, Mac2 = true, Common3 = true, Metal3 = true, **Apple10 (raw 1010) = true**, Apple11 (raw 1011) = false, **Metal4 (raw 5002) = true**. The metal-rs 0.29 enum ends at Apple9/Metal3, so Apple10/Metal4 need a raw `msg_send![dev, supportsFamily: 1010i64]`. Device: unified memory; recommendedMaxWorkingSetSize 28.08 GiB; maxBufferLength 21.06 GiB; maxThreadsPerThreadgroup 1024. | `$R device` |
| Command-buffer watchdog, 1 thread spinning ~60 s (`watchdog.metal`) | **NOT KILLED** | Calibrated at 13.08 ns/iteration, then ran 4.59e9 iterations: status `Completed`, no error, GPU time 59.45 s. The next small dispatch worked normally. A headless compute dispatch of 60 s is not killed on macOS 27.2. Longer runs were not tried. | `$R watchdog 60` |
| `assert()` release semantics (`assert_release.metal`) | **ACCEPT (no-op)** | With only `<metal_stdlib>`, `assert(cond && "msg")` preprocesses to `((void) 0)` (checked by the CI test with `-E`). At runtime, the store after an always-false assert executes. | `$R assert` |
| `[[max_total_threads_per_threadgroup(768)]]` before `kernel void` (`maxthreads_768.metal`) | **ACCEPT** | PSO `maxTotalThreadsPerThreadgroup` = **768** (the kernel without the attribute gets 1024); `threadExecutionWidth` = 32; 768-thread groups give correct output. | `$R maxthreads` |
| Same attribute after the parameter list (`maxthreads_768_post.metal`) | **REJECT** | `'max_total_threads_per_threadgroup' attribute cannot be applied to types`. Translators must emit the attribute before `kernel`. | CI test only |
| 32-bit vs 64-bit integer multiply throughput (`mul_bench.metal`) | **VALUE** | 1M threads × 2048 iterations × 4 dependent chains, one dispatch, best of 3, GPU time: `mul32_lo` **3029 Gop/s**, `mul32_hi` (mulhi) **1652**, `mul32_wide` (32×32→64, take hi) **1439**, `mul64_lo` **787**, `mul64_hi` (mulhi ulong) **353**. 64-bit lo is about 3.8× slower than 32-bit lo; 64-bit mulhi is about 4.7× slower than 32-bit mulhi. | `$R mul-bench` |
| `simd_shuffle` out-of-range lanes (`simd_shuffle.metal`) | **VALUE** | Own value = 1000+lane, delta 4. Full 32-lane group: `simd_shuffle_down` lanes 28..31 return their **own value** (1028..1031); `simd_shuffle_up` lanes 0..3 return their own value. `simd_shuffle(v, lane+40)` returned 1008 on lanes 0..3, i.e. the index is not taken mod 32 per lane, so treat it as undefined. Partial group (24 threads): `shuffle_down` from lanes 20..23 into the inactive lanes 24..27 returns **0**. Reductions are safe on full SIMD groups only when they ignore out-of-range lanes (own value comes back); with partial groups the inactive source reads 0. | `$R simd-shuffle` |
| Empty-dispatch cost (`dispatch.metal` `empty_kernel`) | **VALUE** | N=1000. One command buffer and one encoder: **1.2 µs/dispatch** wall (0.46 µs/dispatch GPU). One command buffer per dispatch, commit+wait each: **144 µs/dispatch**. One command buffer per dispatch, all committed, wait on the last: **10.9 µs/dispatch**. An earlier run on the shared machine gave 2.6 / 273 / 14.7 µs, so these numbers are noisy. | `$R empty-dispatch` |
| Maximum dispatches per command buffer (`dispatch.metal` `count_kernel`) | **ACCEPT up to 262144** | One encoder; each dispatch does one atomic increment, and the counter is checked. 2048: ok (2.0 µs/dispatch wall); 8192: ok (2.1); 65536: ok (1.05); 262144: ok (0.91; 239 ms total). No failure found. The 2048-per-batch cap in plan 2.6 is a latency choice, not a limit. | `$R max-dispatches <n>` |
| `new_buffer_with_bytes_no_copy` on 16 KB-aligned page-multiple memory (no shader) | **ACCEPT** | 64 KiB allocated with `Layout(align=16384)`: buffer created, `contents()` == host pointer, GPU writes visible through the host pointer. A plain `vec![0u8; 1 MiB]` happened to be 16 KB-aligned (macOS malloc, large size) and also works. Also observed outside the documented contract: 4 KB-aligned, **64 B-aligned**, and non-page-multiple length (65540) were all accepted and worked, even under `MTL_DEBUG_LAYER=1` (4 KB case). The port should still keep to the documented 16 KB page / page-multiple rule (plan 2.5). | `$R no-copy aligned` (`unaligned`, `unaligned64`, `oddlen`, `vec`) |
| `atomic_ulong` `atomic_fetch_add_explicit` (`atomic_ulong.metal`) | **REJECT** | `no matching function for call to 'atomic_fetch_add_explicit'` (re-checks errata 8.5). | CI test only |
| `mulhi(ulong, ulong)` (`mulhi64.metal`) | **ACCEPT** | Re-checks errata 8.5; cost is in the `mul-bench` row. | CI test only |

## Consequences for other items

- F0/2.9: the host-side assert `tg_bytes ≤ device.max_threadgroup_memory_length()` is mandatory. Oversized dynamic threadgroup memory is not reported as an error and can corrupt data silently. Static oversize fails at PSO creation.
- Z5 (`MdJobs`) and any pointer-in-`set_bytes` struct: call `use_resource(buf, usage)` for every referenced buffer, or writes are dropped silently.
- Per-dispatch GPU timing: dispatch-boundary sampling is unsupported. Use encoder-boundary (stage) samples or command-buffer `GPUStartTime/GPUEndTime`.
- 2.4 "64-bit integer math flagged": measured about 4–5× slower than 32-bit for mul/mulhi.
- Batches: one command buffer with many dispatches costs about 1 µs/dispatch. Waiting per command buffer costs about 150 µs. 262144 dispatches in one encoder worked.
- `[[host_name]]`, cross-`.air` externs, designated initializers, and `assert` → `((void)0)` all behave as plan 2.3 assumes.
