// harness SRC.metal REPS N_COLS BZ T [VARIANT]: runs z13_shape REPS times over (n = N_COLS, BZ blocks of
// 16 rows), threadgroups of T threads; compares each rep to a CPU reference that has the kernel's own
// semantics (the twist is applied only where the guard holds; CONTROL applies it everywhere, as an
// identity for bz == 0) and to rep 0; prints wrong reps and outputs.
// Env: DISP=k encodes k identical dispatches per command buffer (default 1).
// W7-T1 fix round 1: the first version's reference applied t = ONE, st = prm.z to the bz == 0 blocks
// (a twist by st^m), which the repro kernel never computes: its "20/20 wrong" was a reference bug.
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const uint32_t P = 0x78000001u, MI = 0x77ffffffu, ONE = 0x0ffffffeu;
static uint32_t mul(uint32_t a, uint32_t b) { uint64_t o = (uint64_t)a * b; uint32_t red = (uint32_t)o * MI; o += (uint64_t)red * P; uint32_t r = o >> 32; return r >= P ? r - P : r; }
static uint32_t add(uint32_t a, uint32_t b) { uint32_t r = a + b; return r >= P ? r - P : r; }
static uint32_t sub(uint32_t a, uint32_t b) { return a >= b ? a - b : a + P - b; }
static void bfly(uint32_t *v, const uint32_t *w) { for (int s = 0; s < 4; s++) { uint32_t h = 8u >> s; for (int i = 0; i < 16; i++) if (!(i & h)) { uint32_t a = v[i], b = mul(v[i + h], w[s * 16 + i]); v[i] = add(a, b); v[i + h] = sub(a, b); } } }
static int is(int argc, char **argv, const char *v) { return argc > 6 && !strcmp(argv[6], v); }
int main(int argc, char **argv) {
  @autoreleasepool {
    NSString *src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:nil];
    int reps = atoi(argv[2]); uint32_t n = atoi(argv[3]), BZ = atoi(argv[4]), T = atoi(argv[5]);
    int disp = getenv("DISP") ? atoi(getenv("DISP")) : 1;
    if (argc > 6) src = [[NSString stringWithFormat:@"#define %s 1\n", argv[6]] stringByAppendingString:src];
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
    if (!lib) { printf("compile: %s\n", err.description.UTF8String); return 2; }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"z13_shape"] error:&err];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    size_t N = (size_t)BZ * 16 * n;
    id<MTLBuffer> in = [dev newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [dev newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> twb = [dev newBufferWithLength:(size_t)BZ * n * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> stwb = [dev newBufferWithLength:BZ * 4 options:MTLResourceStorageModeShared];
    uint32_t *pi = in.contents, *tw = twb.contents, *stw = stwb.contents, w[64], prm[4] = {n, 0x1234567u % P, 0x2468ace % P, 0};
    uint64_t s = 0x9e3779b97f4a7c15ull;
#define RND (s ^= s << 13, s ^= s >> 7, s ^= s << 17, (uint32_t)(s % P))
    for (size_t i = 0; i < N; i++) pi[i] = RND;
    for (int i = 0; i < 64; i++) w[i] = RND;
    for (uint32_t bz = 0; bz < BZ; bz++) { stw[bz] = bz ? RND : ONE; for (uint32_t c = 0; c < n; c++) tw[bz * n + c] = bz ? RND : ONE; }
    int nodyn = is(argc, argv, "NODYN"), gdiv = is(argc, argv, "GUARD_DIV"), single = is(argc, argv, "SINGLE"),
        control = is(argc, argv, "CONTROL"), spill = is(argc, argv, "SPILL"), table = is(argc, argv, "TABLE");
    uint32_t *ref = malloc(N * 4);
    for (uint32_t bz = 0; bz < BZ; bz++) for (uint32_t g = 0; g < n / T; g++) {
      uint32_t (*V)[16] = malloc(T * 64);
      for (uint32_t l = 0; l < T; l++) { uint32_t col = g * T + l, *v = V[l];
        for (int m = 0; m < 16; m++) v[m] = pi[(bz * 16 + m) * n + col];
        uint32_t t = mul(prm[1], bz + col + 1), st = prm[2];
        if (table) { t = tw[bz * n + col]; st = stw[bz]; }
        int guard = gdiv ? (col & 1) != 0 : bz != 0;
        if (control) {
          if (bz == 0) { t = ONE; st = ONE; }
          for (int m = 0; m < 16; m++) { v[m] = mul(v[m], t); t = mul(t, st); }
        } else if (single) {
          if (guard) v[0] = mul(v[0], t);
        } else if (spill) {
          uint32_t u[16];
          for (int m = 0; m < 16; m++) u[m] = v[(m * 7 + l) & 15];
          if (guard) for (int m = 0; m < 16; m++) { v[m] = mul(v[m], t); u[m] = mul(u[m], st); t = mul(t, st); }
          for (int m = 0; m < 16; m++) v[m] = add(v[m], u[(m + l) & 15]);
        } else if (guard) {
          for (int m = 0; m < 16; m++) { v[m] = mul(v[m], t); t = mul(t, st); }
        }
        bfly(v, w); }
      uint32_t *sh = malloc(T * 64);
      for (uint32_t l = 0; l < T; l++) for (int m = 0; m < 16; m++) sh[m * T + l] = V[l][nodyn ? m : ((m + l) & 15)];
      for (uint32_t l = 0; l < T; l++) { uint32_t v[16], col = g * T + l;
        for (int m = 0; m < 16; m++) v[m] = sh[m * T + ((l + m) % T)];
        bfly(v, w); for (int m = 0; m < 16; m++) ref[(bz * 16 + m) * n + col] = v[m]; }
      free(V); free(sh);
    }
    int bad = 0, vs0 = 0; size_t badel = 0, bad_bz0 = 0; uint32_t *first = malloc(N * 4);
    for (int r = 0; r < reps; r++) {
      memset(out.contents, 0, N * 4);
      id<MTLCommandBuffer> cb = [q commandBuffer];
      for (int d = 0; d < disp; d++) {
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso]; [e setBuffer:in offset:0 atIndex:0]; [e setBuffer:out offset:0 atIndex:1];
        [e setBytes:prm length:16 atIndex:2]; [e setBytes:w length:256 atIndex:3];
        [e setBuffer:twb offset:0 atIndex:4]; [e setBuffer:stwb offset:0 atIndex:5]; [e setThreadgroupMemoryLength:T * 64 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(n / T, BZ, 1) threadsPerThreadgroup:MTLSizeMake(T, 1, 1)];
        [e endEncoding];
      }
      [cb commit]; [cb waitUntilCompleted];
      uint32_t *po = out.contents; size_t k = 0, k0 = 0;
      for (size_t i = 0; i < N; i++) if (po[i] != ref[i]) { k++; if (i < 16 * (size_t)n) k0++; }
      if (r == 0) memcpy(first, po, N * 4); else if (memcmp(first, po, N * 4)) vs0++;
      if (k) { bad++; badel += k; bad_bz0 += k0; }
    }
    printf("z13shape %-9s: %d/%d reps wrong (%zu wrong outputs, %zu in bz==0 blocks), %d/%d differ from rep 0; n=%u BZ=%u T=%u disp=%d\n",
           argc > 6 ? argv[6] : "repro", bad, reps, badel, bad_bz0, vs0, reps, n, BZ, T, disp);
  }
  return 0;
}
