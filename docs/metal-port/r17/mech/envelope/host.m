// Generic host for mech variants. Usage: host <file.metal|.metallib> [threads] [reps] [tg]
// env: DEFS="N=4,T=4,..." preprocessor macros; STD=3.0|3.1|3.2|... language version; OPTSIZE=1 -> optimizationLevel Size;
//      FASTMATH=0 ; NOINLINE_FN ; QUIET=1 (only summary)
// Kernel writes out[gid*32 + 0] = mask(first read), [1] = mask(second read), [2..17] = first-read values,
// [18..31] = second read values (up to 14). INIT(i) = INITV(i) defined below; EXP(gid,i)=(gid+i+1)*2654435761u.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
static uint32_t EXP(uint32_t g, uint32_t i) { return (g + i + 1) * 2654435761u; }
int main(int argc, const char **argv) {
  @autoreleasepool {
    if (argc < 2) return 2;
    NSString *path = [NSString stringWithUTF8String:argv[1]];
    uint32_t n = argc > 2 ? (uint32_t)atoi(argv[2]) : (1u << 20);
    int reps = argc > 3 ? atoi(argv[3]) : 10;
    NSUInteger tg = argc > 4 ? (NSUInteger)atoi(argv[4]) : 256;
    const uint32_t stride = 256;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil; id<MTLLibrary> lib;
    const char *quiet = getenv("QUIET");
    int initBase = 100; if (getenv("INITBASE")) initBase = atoi(getenv("INITBASE"));
    if ([path hasSuffix:@".metallib"]) lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    else {
      NSString *src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
      MTLCompileOptions *o = [MTLCompileOptions new];
      NSMutableDictionary *m = [NSMutableDictionary new];
      const char *d = getenv("DEFS");
      if (d && *d) for (NSString *kv in [[NSString stringWithUTF8String:d] componentsSeparatedByString:@","]) {
        NSArray *p = [kv componentsSeparatedByString:@"="];
        m[p[0]] = p.count > 1 ? @([p[1] intValue]) : @1;
      }
      o.preprocessorMacros = m;
      const char *s = getenv("STD");
      if (s) {
        if (!strcmp(s,"2.4")) o.languageVersion = MTLLanguageVersion2_4;
        else if (!strcmp(s,"3.0")) o.languageVersion = MTLLanguageVersion3_0;
        else if (!strcmp(s,"3.1")) o.languageVersion = MTLLanguageVersion3_1;
        else if (!strcmp(s,"3.2")) o.languageVersion = MTLLanguageVersion3_2;
        else if (!strcmp(s,"2.0")) o.languageVersion = MTLLanguageVersion2_0;
      }
      if (getenv("OPTSIZE")) o.optimizationLevel = MTLLibraryOptimizationLevelSize;
      if (getenv("FASTMATH0")) o.mathMode = MTLMathModeSafe;
      lib = [dev newLibraryWithSource:src options:o error:&err];
    }
    if (!lib) { fprintf(stderr, "library: %s\n", err.description.UTF8String); return 3; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"test_kernel"];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { fprintf(stderr, "pipeline: %s\n", err.description.UTF8String); return 3; }
    if (tg > pso.maxTotalThreadsPerThreadgroup) tg = pso.maxTotalThreadsPerThreadgroup;
    id<MTLBuffer> scratch = [dev newBufferWithLength:(NSUInteger)n * stride * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [dev newBufferWithLength:(NSUInteger)n * 32 * 4 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    int badReps = 0; uint64_t totBad = 0;
    NSMutableDictionary<NSString*, NSNumber*> *sig = [NSMutableDictionary new];
    for (int rep = 0; rep < reps; rep++) {
      memset(scratch.contents, 0, scratch.length);
      memset(out.contents, 0, out.length);
      uint32_t prm[4] = {n, (uint32_t)rep, stride, 0};
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:scratch offset:0 atIndex:0];
      [enc setBuffer:out offset:0 atIndex:1];
      [enc setBytes:prm length:sizeof(prm) atIndex:3];
      [enc setThreadgroupMemoryLength:16384 atIndex:0];
      [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
      if (cb.error) { fprintf(stderr, "cb: %s\n", cb.error.description.UTF8String); return 3; }
      const uint32_t *o = (const uint32_t *)out.contents;
      uint64_t bad = 0;
      for (uint32_t g = 0; g < n; g++) {
        const uint32_t *r = o + (size_t)g * 32;
        if (!r[0] && !r[1]) continue;
        bad++;
        if (sig.count > 64) continue;
        NSMutableString *k = [NSMutableString stringWithFormat:@"m1=0x%x m2=0x%x", r[0], r[1]];
        for (int pass = 0; pass < 2; pass++) {
          uint32_t mask = r[pass]; const uint32_t *v = r + (pass ? 18 : 2);
          for (int i = 0; i < (pass ? 14 : 16); i++) if (mask & (1u << i)) {
            uint32_t x = v[i]; NSString *c = nil;
            for (int j = 0; j < 16 && !c; j++) if (x == (uint32_t)(initBase + j)) c = [NSString stringWithFormat:@"init[%d]", j];
            for (int j = 0; j < 16 && !c; j++) if (x == EXP(g, j)) c = [NSString stringWithFormat:@"exp[%d]", j];
            if (!c) c = [NSString stringWithFormat:@"%u", x];
            [k appendFormat:@" r%d[%d]=%@", pass + 1, i, c];
          }
        }
        sig[k] = @(sig[k].intValue + 1);
      }
      if (bad) badReps++;
      totBad += bad;
      if (!quiet) printf("rep %d: %llu/%u wrong\n", rep, (unsigned long long)bad, n);
    }
    NSArray *keys = [sig keysSortedByValueUsingComparator:^NSComparisonResult(id a, id b) { return [b compare:a]; }];
    int shown = 0;
    for (NSString *k in keys) { if (shown++ >= 4) break; printf("  sig %s : %d\n", k.UTF8String, sig[k].intValue); }
    printf("SUMMARY %s DEFS=%s STD=%s n=%u tg=%lu: %d/%d dispatches wrong, %llu threads wrong total\n", argv[1],
           getenv("DEFS") ?: "", getenv("STD") ?: "def", n, (unsigned long)tg, badReps, reps, (unsigned long long)totBad);
    return 0;
  }
}
