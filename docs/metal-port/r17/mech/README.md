# Metal compute kernel reads a stale value from a thread-private array right after a loop that contains a function call

**Area:** Metal / GPU shader compiler (AGX back end) or GPU hardware
**Type:** Incorrect or unexpected behaviour (silent wrong results)
**Machine:** Apple M5 Max (Mac17,14), GPU architecture `applegpu_g17s`, macOS 27.2.1 (26B131)

## Summary
A compute kernel keeps a 4-element `uint` array in thread (private) memory. A non-unrolled loop calls a
non-inlined function and then stores `x[i]`. Straight after the loop, the kernel reads the array back. The
element stored in the **last** iteration often reads back with its **previous** value. On a dispatch of
1,048,576 threads, about 40% of threads get the wrong value, in 10 of 10 dispatches.

The kernel has no undefined behaviour. It needs no device-memory access inside the loop: the callee only
does arithmetic. The result is nondeterministic. A dispatch of 32 to 256 threads never fails, and whole
SIMD groups fail together.

The memory does hold the right value. A later read of the same element through a data-dependent index
returns the stored value in every thread whose first read was wrong. So the first load runs before the
last store in the loop reaches it: a read-after-write hazard on thread-private (stack) memory. The store is
not lost, and a late initialisation store does not overwrite it.

## Bug case (`repro.metal`, `VARIANT 0`, shown without the variant switches)
```metal
__attribute__((noinline)) static uint mix(uint v) { return (v * 2246822519u) ^ (v >> 13); }

kernel void test_kernel(device uint* out [[buffer(0)]], constant uint2& prm [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= prm.x) return;
  uint x[4];
  for (thread uint* p = x; p != x + 4; ++p) *p = 100u + (uint)(p - x);  // x = {100, 101, 102, 103}
  uint acc = 0;
#pragma nounroll
  for (uint i = 0; i < 4; i++) {
    uint v = (gid + i + 1u) * 2654435761u;
    acc ^= mix(v);
    x[i] = v;
  }
  uint bad = 0;
  for (uint i = 0; i < 4; i++) if (x[i] != (gid + i + 1u) * 2654435761u) bad |= 1u << i;
  out[gid * 32 + 0] = bad;          // expected 0 for every thread
  out[gid * 32 + 1] = x[3];
  out[gid * 32 + 3] = acc;
}
```

## Steps to reproduce
1. `./build.sh`. This needs only the Command Line Tools (`clang`). Metal compiles the shader at run time
   (`newLibraryWithSource`), so neither Xcode nor the offline Metal compiler is needed. A prebuilt arm64
   `repro` (macOS 12+) is included.
2. `./repro` runs the bug case, 6 controls and 5 probes. Each one is 10 dispatches of 1,048,576 threads
   with threadgroup size 256.
3. `./repro --sweep` sweeps dispatch size against threadgroup size. `./repro --stress` sweeps the output
   layout, 1M/4M threads and threadgroup sizes 256 and 1024.
4. `./check_this_mac.sh` runs all three and saves the output to `result-<model>-<date>.txt`.

## Expected results
`bad == 0` for every thread in every variant. `./repro` prints `0/10` in the "wrong" column for all rows.

## Actual results (`sample_output.txt`)
```
v  variant                                            wrong     thr/disp     first wrong    2nd read wrong
0  BUG: store after call in non-unrolled loop         10/10     438296       mask 0x8 =103  n/a
1  control: store before the call                     0/10      0            -              n/a
2  control: loop unrolled (no #pragma nounroll)       0/10      0            -              n/a
3  control: callee inlined                            0/10      0            -              n/a
4  control: index-loop init instead of pointer loop   0/10      0            -              n/a
5  control: read back through a dynamic index         0/10      0            -              n/a
6  control: threadgroup_barrier(mem_none) after loop  0/10      0            -              n/a
7  probe: later data-dependent second read            10/10     533889       mask 0x8 =103  0 threads
8  probe: two inits (100+i then 200+i)                10/10     452091       mask 0x8 =203  0 threads
9  no help: atomic_thread_fence(thread scope)         10/10     603960       mask 0x8 =103  n/a
10 no help: simdgroup_barrier(mem_none)               10/10     571260       mask 0x8 =103  n/a
11 no help: another call after the loop               10/10     239147       mask 0x8 =103  n/a
```
- Only `x[3]` is wrong (mask 0x8), the element stored in the last iteration. It reads 103, its previous
  value.
- **Probe 7 (read after write, not a lost store).** A second read of the array uses an index that depends
  on the first read's result (it is always 0 at run time). It returns the correct value in all of the
  ~5.3M threads whose first read was wrong.
- **Probe 8 (not a late init store).** The array is initialised twice, first to 100+i and then to 200+i.
  The stale value is 203, the most recent earlier value, not the first initialiser 103. So no
  initialisation store reaches memory after the loop's store. The first load after the loop sees memory
  as it was before the last store.

## Frequency
Always on Apple M5 Max: 10/10 dispatches for every configuration from 65,536 threads up
(`sample_sweep.txt`, `sample_stress.txt`).
| threads | tg 32 | tg 256 | tg 1024 |
|---|---|---|---|
| 32 / 256 | 0/10 | 0/10 | 0/10 |
| 1,024 | 0/10 | 0/10 | 10/10 (~121 threads) |
| 4,096 | 0/10 | 0/10 | 10/10 (~409) |
| 65,536 | 10/10 (~2.6k) | 10/10 (~1.4k) | 10/10 (~2.9k) |
| 1,048,576 | 10/10 (~391k) | 10/10 (~421k) | 10/10 (~431k) |

The rate depends on the number of SIMD groups resident on a GPU core, and on memory traffic from
*other* threads.
- One threadgroup of 1, 32, 64 or 128 threads: 0/50 each. 256 threads: 7/50. 512 or 1024 threads: 50/50
  (`envelope/k.metal`, same pattern).
- The output layout matters. With `REPRO_STRIDE=4` (16 contiguous bytes per thread) the same kernel is
  0/10. With 8, 32 or 64 words per thread it is 10/10. The machine code for STRIDE=4 and STRIDE=8 differs
  in one 16-bit word, which we read as the address shift. The failure is a timing or memory-system
  effect, not a fixed miscompiled value.

## Controls: what makes it correct and what does not (1,048,576 threads, 5-10 dispatches each)
| change | result |
|---|---|
| store `x[i]` **before** the call | correct |
| loop unrolled, or callee `always_inline`, or callee body empty (call removed) | correct |
| read the array through a dynamic index `x[(i + zero) & 3]`, or through a `volatile thread uint*` | correct |
| `threadgroup_barrier(mem_flags::mem_none)` after the loop | correct |
| `atomic_thread_fence(mem_device, seq_cst, thread_scope_device)` after the loop | correct |
| array in `threadgroup` memory instead of thread memory | correct |
| aggregate init `{100,101,...}`, value-init `{}`, index-loop init, no init | correct (in the G13 code these keep the array in registers, see below) |
| callee does device stores, device loads, device atomic stores, threadgroup stores, a store through a thread pointer, or pure arithmetic | **wrong** in every case |
| init through a C++ default constructor (`struct Elem { uint v; Elem() : v(77) {} }`), with or without a struct wrapper | **wrong** (`supplementary/ctor_shape`) |
| `atomic_thread_fence(..., thread_scope_thread)` (emits no instruction: the G17 binary is byte-identical), `simdgroup_barrier(mem_none)`, a second call between the loop and the read | **wrong** |
| runtime compile with language version 2.4, 3.0, 3.1 or 3.2, `optimizationLevel = Size`, `mathMode = Safe` | **wrong** |
| offline `metal -O2` / `-O3` with `-std=metal3.0`, 3.2 or 4.0 | **wrong** |
| offline `metal -O0`, `-O1` or `-Os` (this small kernel) | correct. The larger original program still fails at -O0 (lower rate) and -Os. |

Envelope of the bug shape (`envelope/k.metal`, the same pattern with macros, 1M threads x 5):
- **Array size.** 3, 4, 5, 6, 7 or 8 elements: wrong. 2, 12 or 16 elements: correct.
- **Loop trip count.** 3 or more: wrong, and the element lost is always the one stored in the last
  iteration (`TRIP-1`), even when the array is longer. A trip count of 1 or 2 is correct.

## Analysis (what we could and could not see)
- **The input to the GPU back end is correct.** The LLVM IR given to the AGX back end
  (`applegpu-nt -mllvm -print-after-all`, last dump; see `evidence/ir_handed_to_agx_backend_k_NE4.ll`) has
  the alloca and its init loop, then a loop with `call fastcc @callee` followed by `store i32 %v, gep(alloca, i)`,
  then loads of every element after the loop.
- **The runtime binary matches the offline one.** We serialized the runtime-compiled pipeline with
  `MTLBinaryArchive` and thinned it to `applegpu_g17s`. Its `__text` is byte-identical to the offline
  `applegpu-nt -arch applegpu_g17s` output, so offline binaries are a valid stand-in.
- **No public G17 decoder.** No public disassembler decodes G17 (M5) code, and `metal-objdump` has no
  instruction printer for agx3. The raw code is in `evidence/g17s_machine_code_variant0.hex`.
- **G13 (M1) translation, as a proxy.** We translated the same kernel for `applegpu_g13g` and decoded it
  with the open-source G13 decoder (dougallj/applegpu; third-party, best effort).
  - Bug case (`evidence/g13_disasm_variant0.txt`): `stack_store xyzw` (init), then a loop of `call` and a
    scalar `stack_store x` to a register address, then **directly** `stack_load xyzw` at immediate offset
    0 and `wait 0`.
  - Store-before-call control: the same instructions with the store before the call.
  - Dynamic-index control: four scalar register-addressed `stack_load x` in place of one wide load.
  - Barrier control: adds `threadgroup_barrier` before the load.
  - The correct init shapes (aggregate, value-init, index loop) keep the array in registers, with no stack
    access.
  - We did not run G13 code here. These listings show the instruction sequence the back end produces, not
    G17 behaviour.
- **Inference (not verified at ISA level on G17).** A wide, immediate-offset load from private (stack)
  memory that follows a scalar, register-addressed stack store issued after a `call`/`ret` is not ordered
  after that store on G17. Either the compiler omits a dependency or wait, or the hardware's hazard
  tracking misses it. It shows up when many SIMD groups share the core and memory traffic is high.

## Workarounds we found (any one is enough here)
- store before the call;
- keep the array out of stack memory (initialise with an index loop or an aggregate);
- unroll the loop;
- add a `threadgroup_barrier(mem_flags::mem_none)` or a device-scope fence after the loop;
- read through a dynamic index.

In the original program (a Metal port of a CUDA ZK witness generator, where `ToBits<N>` decompositions use
this shape), making the field-element default constructor trivial removed the failure in its default
build. It did not remove it when the compiler was left to decide inlining, so we believe other code shapes
reach the same defect.

## Configuration
- Hardware: Mac17,14, Apple M5 Max, 32-core GPU, `MTLDevice.architecture.name = applegpu_g17s`, supports
  Apple1-Apple10 / Metal 4 families, not Apple11.
- OS: macOS 27.2.1 (26B131).
- Runtime compiler: the system Metal framework (`newLibraryWithSource`).
- Offline toolchain used for the analysis: Metal Toolchain 27B47 (`metal` 32023.922, metalfe-32023.922.4),
  macOS SDK 27.2, Apple clang 21.0.0 (clang-2100.4.1.1) / Command Line Tools 27.2.
- We have no older Apple Silicon machine to compare. `check_this_mac.sh` is provided for that.

## Files
| file | content |
|---|---|
| `repro.metal` | bug case, controls and probes (`VARIANT` 0-11) |
| `main.m`, `build.sh`, `repro` | host (Objective-C, Command Line Tools only); prebuilt arm64 binary |
| `check_this_mac.sh` | runs everything and saves `result-<model>-<date>.txt` |
| `sample_output.txt`, `sample_sweep.txt`, `sample_stress.txt` | output on the M5 Max |
| `supplementary/ctor_shape/` | the first reduction: default-constructor init, a 3- and an 8-element version, and the original program's shape (`repro_original_shape.metal`), with its own host `main.m` (`./repro file.metal [threads] [reps] [tg]`) |
| `envelope/k.metal`, `envelope/host.m` | the parametric kernel and host used to map the envelope (macros listed at the top of `k.metal`) |
| `evidence/` | device info, IR handed to the back end, G13 disassembly of the variants, G17 machine code (hex) |
