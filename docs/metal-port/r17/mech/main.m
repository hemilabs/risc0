// Host for repro.metal. Builds with the Command Line Tools only (no Xcode, no offline Metal compiler needed):
//   clang -fobjc-arc -O2 -mmacosx-version-min=12.0 -framework Metal -framework Foundation main.m -o repro
// Usage:
//   ./repro                      run the bug case, the controls and the probes (default 1048576 threads x 10)
//   ./repro <variant> [threads] [reps] [threadgroup]
//   ./repro --sweep              bug case over dispatch sizes and threadgroup sizes
//   ./repro --stress             bug case over output layouts, 1M/4M threads, threadgroup 256/1024 (older Macs)
// env REPRO_STRIDE=<words>     output words per thread (default 32; 4 hides the bug on M5 Max, see README)
// env REPRO_METAL=<path> selects the shader source (default: repro.metal next to the executable, then ./repro.metal)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/sysctl.h>

static const char *kName[] = {
  "BUG: store after call in non-unrolled loop",
  "control: store before the call",
  "control: loop unrolled (no #pragma nounroll)",
  "control: callee inlined",
  "control: index-loop init instead of pointer loop",
  "control: read back through a dynamic index",
  "control: threadgroup_barrier(mem_none) after loop",
  "probe: later data-dependent second read",
  "probe: two inits (100+i then 200+i)",
  "no help: atomic_thread_fence(thread scope)",
  "no help: simdgroup_barrier(mem_none)",
  "no help: another call after the loop",
};
static const int kExpectBug[] = {1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
enum { kNumVariants = 12 };

static id<MTLDevice> gDev;
static NSString *gSrc;
static uint32_t gStride = 32;   // words per thread in the output buffer (see README: 4 hides the bug)

typedef struct { int badReps, reps; uint64_t badThreads, bad2Threads; uint32_t mask, stale; } Result;

static id<MTLComputePipelineState> pipeline(int variant) {
  MTLCompileOptions *o = [MTLCompileOptions new];
  o.preprocessorMacros = @{ @"VARIANT" : @(variant), @"STRIDE" : @(gStride) };
  NSError *err = nil;
  id<MTLLibrary> lib = [gDev newLibraryWithSource:gSrc options:o error:&err];
  if (!lib) { fprintf(stderr, "compile variant %d: %s\n", variant, err.description.UTF8String); exit(3); }
  id<MTLComputePipelineState> p = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"test_kernel"] error:&err];
  if (!p) { fprintf(stderr, "pipeline variant %d: %s\n", variant, err.description.UTF8String); exit(3); }
  return p;
}

static Result run(int variant, uint32_t n, int reps, NSUInteger tg) {
  Result r = {0, reps, 0, 0, 0, 0};
  id<MTLComputePipelineState> pso = pipeline(variant);
  if (tg > pso.maxTotalThreadsPerThreadgroup) tg = pso.maxTotalThreadsPerThreadgroup;
  id<MTLBuffer> out = [gDev newBufferWithLength:(NSUInteger)n * gStride * 4 options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> q = [gDev newCommandQueue];
  for (int rep = 0; rep < reps; rep++) {
    memset(out.contents, 0xff, out.length);
    uint32_t prm[2] = {n, 0};
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pso];
    [e setBuffer:out offset:0 atIndex:0];
    [e setBytes:prm length:sizeof(prm) atIndex:1];
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.error) { fprintf(stderr, "command buffer: %s\n", cb.error.description.UTF8String); exit(3); }
    const uint32_t *o = out.contents;
    uint64_t bad = 0;
    for (uint32_t g = 0; g < n; g++) {
      if (o[gStride * g]) { if (!bad && !r.badThreads) { r.mask = o[gStride * g]; r.stale = o[gStride * g + 1]; } bad++; }
      if (o[gStride * g + 2]) r.bad2Threads++;
    }
    if (bad) r.badReps++;
    r.badThreads += bad;
  }
  return r;
}

static void sysinfo(void) {
  char model[256] = "?"; size_t len = sizeof(model); sysctlbyname("hw.model", model, &len, NULL, 0);
  char cpu[256] = "?"; len = sizeof(cpu); sysctlbyname("machdep.cpu.brand_string", cpu, &len, NULL, 0);
  printf("Mac model     : %s (%s)\n", model, cpu);
  printf("macOS         : %s\n", NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
  printf("GPU           : %s", gDev.name.UTF8String);
  if ([gDev respondsToSelector:@selector(architecture)]) {
    id arch = [gDev performSelector:@selector(architecture)];
    printf(", architecture %s", [[arch valueForKey:@"name"] UTF8String]);
  }
  printf("\nGPU families  :");
  for (int f = 1001; f <= 1012; f++) if ([gDev supportsFamily:(MTLGPUFamily)f]) printf(" Apple%d", f - 1000);
  printf("\n");
}

int main(int argc, const char **argv) {
  @autoreleasepool {
    gDev = MTLCreateSystemDefaultDevice();
    if (getenv("REPRO_STRIDE")) gStride = (uint32_t)atoi(getenv("REPRO_STRIDE"));
    if (!gDev) { fprintf(stderr, "no Metal device\n"); return 3; }
    NSString *path = getenv("REPRO_METAL") ? @(getenv("REPRO_METAL")) : nil;
    if (!path) {
      NSString *exeDir = [@(argv[0]) stringByDeletingLastPathComponent];
      path = [exeDir stringByAppendingPathComponent:@"repro.metal"];
      if (![NSFileManager.defaultManager fileExistsAtPath:path]) path = @"repro.metal";
    }
    NSError *err = nil;
    gSrc = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
    if (!gSrc) { fprintf(stderr, "cannot read %s\n", path.UTF8String); return 3; }
    sysinfo();

    if (argc > 1 && !strcmp(argv[1], "--sweep")) {
      uint32_t ns[] = {32, 256, 1024, 4096, 65536, 1048576};
      NSUInteger tgs[] = {32, 256, 1024};
      printf("\nBUG case (variant 0), 10 dispatches per cell: wrong dispatches, wrong threads per dispatch\n");
      printf("%10s", "threads");
      for (int j = 0; j < 3; j++) printf("   tg=%-16lu", (unsigned long)tgs[j]);
      printf("\n");
      for (int i = 0; i < 6; i++) {
        printf("%10u", ns[i]);
        for (int j = 0; j < 3; j++) {
          NSUInteger tg = tgs[j] > ns[i] ? ns[i] : tgs[j];
          Result r = run(0, ns[i], 10, tg);
          char cell[64]; snprintf(cell, sizeof cell, "%d/10, %llu", r.badReps, (unsigned long long)(r.badThreads / 10));
          printf("   %-19s", cell);
        }
        printf("\n");
      }
      return 0;
    }

    if (argc > 1 && !strcmp(argv[1], "--stress")) {
      // Bug case under several output layouts, dispatch sizes and threadgroup sizes (for older or smaller GPUs).
      uint32_t strides[] = {8, 32, 64}; uint32_t ns[] = {1u << 20, 1u << 22}; NSUInteger tgs[] = {256, 1024};
      printf("\nBUG case (variant 0), 20 dispatches per row\n%-8s %-9s %-6s %-8s %s\n", "stride", "threads", "tg", "wrong", "threads/dispatch");
      int any = 0;
      for (int a = 0; a < 3; a++) for (int b = 0; b < 2; b++) for (int c = 0; c < 2; c++) {
        gStride = strides[a];
        Result r = run(0, ns[b], 20, tgs[c]);
        printf("%-8u %-9u %-6lu %2d/20    %llu\n", strides[a], ns[b], (unsigned long)tgs[c], r.badReps, (unsigned long long)(r.badThreads / 20));
        if (r.badReps) any = 1;
      }
      printf("\nRESULT: %s\n", any ? "BUG REPRODUCED" : "no wrong result in any configuration");
      return any ? 1 : 0;
    }

    int first = 0, last = kNumVariants - 1;
    uint32_t n = 1u << 20; int reps = 10; NSUInteger tg = 256;
    if (argc > 1) { first = last = atoi(argv[1]); }
    if (argc > 2) n = (uint32_t)atoi(argv[2]);
    if (argc > 3) reps = atoi(argv[3]);
    if (argc > 4) tg = (NSUInteger)atoi(argv[4]);
    printf("Dispatch      : %u threads, threadgroup %lu, %d dispatches per variant\n\n", n, (unsigned long)tg, reps);
    printf("%-2s %-50s %-9s %-12s %-14s %s\n", "v", "variant", "wrong", "thr/disp", "first wrong", "2nd read wrong");
    int bugSeen = 0;
    for (int v = first; v <= last; v++) {
      if (v < 0 || v >= kNumVariants) { fprintf(stderr, "variant 0..%d\n", kNumVariants - 1); return 2; }
      Result r = run(v, n, reps, tg);
      char w[32]; snprintf(w, sizeof w, "%d/%d", r.badReps, r.reps);
      char f[48] = "-";
      if (r.badThreads) snprintf(f, sizeof f, "mask 0x%x =%u", r.mask, r.stale);
      char s2[32] = "n/a";
      if (v == 7 || v == 8) snprintf(s2, sizeof s2, "%llu threads", (unsigned long long)r.bad2Threads);
      printf("%-2d %-50s %-9s %-12llu %-14s %s\n", v, kName[v], w, (unsigned long long)(r.badThreads / (uint64_t)reps), f, s2);
      if (v == 0 && r.badReps) bugSeen = 1;
      (void)kExpectBug;
    }
    if (first == 0) printf("\nRESULT: %s\n", bugSeen ? "BUG REPRODUCED (variant 0 read stale values)" : "variant 0 correct on this machine");
    return bugSeen ? 1 : 0;
  }
}
