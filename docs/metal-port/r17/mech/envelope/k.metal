// mech parametric kernel. Macros (defaults in brackets):
// NE [3] elements; TRIP [NE] loop trip count (elements >= TRIP keep their init value)
// INIT [0]: 0 Elem ctor v(100) (same for all); 1 container ctor pointer loop, v=100+idx; 2 kernel-body index loop 100+i;
//           3 aggregate {100,101,...}; 4 Elem ctor v(100) THEN body index loop v=200+i (WAW discriminator);
//           5 none (all elements written by the loop); 6 container ctor index loop v=100+i; 7 value-init A x{} (zero);
//           8 Elem ctor v(100) + container ctor that does nothing extra (non-trivial A)
// WRAP [1]: array inside struct A; 0 = bare Elem x[NE]
// CALLEE [0]: 0 device store; 1 device load (result summed to a sink); 2 threadgroup store; 3 empty; 4 pure ALU returning value;
//             5 device atomic store; 6 store through thread pointer to a different local; 7 device store, callee has 2 stores
// ASPACE [0]: 0 thread; 1 threadgroup slice per thread (INIT forced to pointer loop)
// NOUNROLL [1]; READ2 [0]: 1 delayed data-dependent second read; DELAY [64] dependent ALU steps before READ2
// STORE_BEFORE [0]: 1 store x[i] before the call; CALL_ARG_X [0]: 1 pass x[i]'s address-free value? (unused)
#include <metal_stdlib>
using namespace metal;
#ifndef NE
#define NE 3
#endif
#ifndef TRIP
#define TRIP NE
#endif
#ifndef INIT
#define INIT 0
#endif
#ifndef WRAP
#define WRAP 1
#endif
#ifndef CALLEE
#define CALLEE 0
#endif
#ifndef ASPACE
#define ASPACE 0
#endif
#ifndef NOUNROLL
#define NOUNROLL 1
#endif
#ifndef READ2
#define READ2 0
#endif
#ifndef DELAY
#define DELAY 64
#endif
#ifndef STORE_BEFORE
#define STORE_BEFORE 0
#endif
#ifndef NOINL
#define NOINL 1
#endif
#if NOINL
#define CALLATTR __attribute__((noinline))
#else
#define CALLATTR inline
#endif

#if CALLEE == 0
CALLATTR static uint callee(device uint* row, threadgroup uint* tg, thread uint* oth, uint col, uint v) { row[col] = v; return 0; }
#elif CALLEE == 1
CALLATTR static uint callee(device uint* row, threadgroup uint* tg, thread uint* oth, uint col, uint v) { return row[col + 8] + v; }
#elif CALLEE == 2
CALLATTR static uint callee(device uint* row, threadgroup uint* tg, thread uint* oth, uint col, uint v) { tg[col] = v; return 0; }
#elif CALLEE == 3
CALLATTR static uint callee(device uint* row, threadgroup uint* tg, thread uint* oth, uint col, uint v) { return 0; }
#elif CALLEE == 4
CALLATTR static uint callee(device uint* row, threadgroup uint* tg, thread uint* oth, uint col, uint v) { return (v * 2246822519u) ^ (v >> 13) ^ col; }
#elif CALLEE == 5
CALLATTR static uint callee(device uint* row, threadgroup uint* tg, thread uint* oth, uint col, uint v) { atomic_store_explicit((device atomic_uint*)(row + col), v, memory_order_relaxed); return 0; }
#elif CALLEE == 6
CALLATTR static uint callee(device uint* row, threadgroup uint* tg, thread uint* oth, uint col, uint v) { oth[col & 3] = v; return 0; }
#elif CALLEE == 7
CALLATTR static uint callee(device uint* row, threadgroup uint* tg, thread uint* oth, uint col, uint v) { row[col] = v; row[col + 16] = v ^ 1u; return 0; }
#endif

#if INIT == 0 || INIT == 4 || INIT == 8
struct Elem { uint v; Elem() : v(100u) {} };
#else
struct Elem { uint v; };
#endif
#if WRAP
struct A {
  Elem e[NE];
#if INIT == 1
  A() { for (auto p = e; p != e + NE; ++p) p->v = 100u + (uint)(p - e); }
#elif INIT == 6
  A() { for (uint i = 0; i < NE; i++) e[i].v = 100u + i; }
#elif INIT == 8
  A() {}
#endif
};
#define EL(i) x.e[i].v
#define EL0 x.e[0]
#else
#define EL(i) x[i].v
#define EL0 x[0]
#endif

static inline uint expv(uint gid, uint i) { return (gid + i + 1) * 2654435761u; }
static inline uint initv(uint i) {
#if INIT == 4
  return 200u + i;
#elif INIT == 0 || INIT == 8
  return 100u;
#elif INIT == 7
  return 0u;
#elif INIT == 10
  return 100u;
#else
  return 100u + i;
#endif
}

kernel void test_kernel(device uint* scratch [[buffer(0)]], device uint* out [[buffer(1)]],
                        constant uint4& prm [[buffer(3)]], threadgroup uint* tgm [[threadgroup(0)]],
                        uint gid [[thread_position_in_grid]], uint lid [[thread_position_in_threadgroup]]) {
  if (gid >= prm.x) return;
  device uint* row = scratch + gid * prm.z;
  threadgroup uint* tgrow = tgm + (lid & 255) * 4;   // only used by CALLEE 2 (racy across >256 lanes is fine: value unused)
  uint other[4] = {0, 0, 0, 0};
  uint sink = 0;
#if ASPACE == 1
  threadgroup Elem* xs = (threadgroup Elem*)(tgm + 1024) + (lid & 255) * NE;  // needs tg <= 256 and NE*256+1024 words <= 4096
  for (threadgroup Elem* p = xs; p != xs + NE; ++p) p->v = 100u + (uint)(p - xs);
#define EL2(i) xs[i].v
#undef EL
#define EL(i) xs[i].v
#else
#if WRAP
#if INIT == 3
  A x = {{
#else
#if INIT == 7
  A x{};
#else
  A x;
#endif
#endif
#else
#if INIT == 3
  Elem x[NE] = {
#else
  Elem x[NE];
#endif
#endif
#if INIT == 3
#define AG(k) {100u + k},
    AG(0) AG(1)
#if NE > 2
    AG(2)
#endif
#if NE > 3
    AG(3)
#endif
#if NE > 4
    AG(4)
#endif
#if NE > 5
    AG(5)
#endif
#if NE > 6
    AG(6)
#endif
#if NE > 7
    AG(7)
#endif
#if WRAP
  }};
#else
  };
#endif
#endif
#if INIT == 2 || INIT == 4
  for (uint i = 0; i < NE; i++) EL(i) = initv(i);
#elif INIT == 9
  for (thread Elem* p = &EL0; p != &EL0 + NE; ++p) p->v = 100u + (uint)(p - &EL0);
#elif INIT == 10
  for (thread Elem* p = &EL0; p != &EL0 + NE; ++p) p->v = 100u;
#endif
#endif
#if NOUNROLL
#pragma nounroll
#endif
  for (uint i = 0; i < TRIP; i++) {
    uint v = expv(gid, i);
#if STORE_BEFORE
    EL(i) = v;
#endif
    sink += callee(row, tgrow, other, i, v);
#if !STORE_BEFORE
    EL(i) = v;
#endif
  }
#ifndef FENCE
#define FENCE 0
#endif
#if FENCE == 1
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_thread);
#elif FENCE == 2
  threadgroup_barrier(mem_flags::mem_threadgroup);
#elif FENCE == 3
  simdgroup_barrier(mem_flags::mem_none);
#elif FENCE == 4
  sink += callee(row, tgrow, other, 9, sink);
#elif FENCE == 5
  threadgroup_barrier(mem_flags::mem_none);
#elif FENCE == 6
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
#endif
  device uint* o = out + gid * 32;
  uint m1 = 0;
#ifndef FIRSTDYN
#define FIRSTDYN 0
#endif
#ifndef REVREAD
#define REVREAD 0
#endif
#if REVREAD
  for (int ii = NE - 1; ii >= 0; ii--) { uint i = (uint)ii;
#else
  for (uint i = 0; i < NE; i++) {
#endif
    uint want = i < TRIP ? expv(gid, i) : initv(i);
#if FIRSTDYN == 2
    uint got = ((volatile thread uint*)&EL(0))[i];
#elif FIRSTDYN
    uint got = EL((i + prm.w) % NE);
#else
    uint got = EL(i);
#endif
    if (got != want) m1 |= 1u << i;
    o[2 + i] = got;
  }
  uint m2 = 0;
#if READ2
  // delay: DELAY dependent ALU steps on m1, then read via an index that depends on the result (prm.w == 0 at run time)
  uint h = m1 ^ gid;
  for (uint k = 0; k < DELAY; k++) h = h * 1664525u + 1013904223u;
  uint z = h & prm.w;   // always 0
  for (uint i = 0; i < NE && i < 14; i++) {
    uint want = i < TRIP ? expv(gid, i) : initv(i);
    uint got = EL((i + z) % NE);
    if (got != want) m2 |= 1u << i;
    o[18 + i] = got;
  }
#endif
  o[0] = m1; o[1] = m2;
  if (sink == 0xdeadbeefu || other[gid & 3] == 0xdeadbeefu) o[31] = sink;
}
