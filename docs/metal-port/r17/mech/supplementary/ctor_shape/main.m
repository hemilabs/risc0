// Host for repro.metal. Build: clang -fobjc-arc -O2 -framework Metal -framework Foundation main.m -o repro
// Usage: ./repro <repro.metal | repro.metallib> [threads=1048576] [reps=10] [threadgroup=256]
// A .metal path is compiled at run time (newLibraryWithSource); a .metallib is loaded as is.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(int argc, const char **argv) {
  @autoreleasepool {
    if (argc < 2) { fprintf(stderr, "usage: %s <repro.metal|repro.metallib> [threads] [reps] [tg]\n", argv[0]); return 2; }
    NSString *path = [NSString stringWithUTF8String:argv[1]];
    uint32_t n = argc > 2 ? (uint32_t)atoi(argv[2]) : (1u << 20);
    int reps = argc > 3 ? atoi(argv[3]) : 10;
    NSUInteger tg = argc > 4 ? (NSUInteger)atoi(argv[4]) : 256;
    const uint32_t stride = 256;  // words of scratch per thread (each thread owns its own row)
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("device: %s\n", dev.name.UTF8String);
    NSError *err = nil;
    id<MTLLibrary> lib;
    if ([path hasSuffix:@".metallib"]) {
      lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    } else {
      NSString *src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
      if (!src) { fprintf(stderr, "read: %s\n", err.description.UTF8String); return 1; }
      lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
    }
    if (!lib) { fprintf(stderr, "library: %s\n", err.description.UTF8String); return 1; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"test_kernel"];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { fprintf(stderr, "pipeline: %s\n", err.description.UTF8String); return 1; }
    if (tg > pso.maxTotalThreadsPerThreadgroup) tg = pso.maxTotalThreadsPerThreadgroup;
    id<MTLBuffer> scratch = [dev newBufferWithLength:(NSUInteger)n * stride * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [dev newBufferWithLength:(NSUInteger)n * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> aux = [dev newBufferWithLength:(65536 + 256) * 4 options:MTLResourceStorageModeShared];  // only bound for repro_original_shape.metal
    memset(aux.contents, 0, aux.length);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    int badReps = 0;
    for (int rep = 0; rep < reps; rep++) {
      memset(scratch.contents, 0, scratch.length);
      memset(out.contents, 0, out.length);
      uint32_t prm[4] = {n, (uint32_t)rep, stride, 0};
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:scratch offset:0 atIndex:0];
      [enc setBuffer:out offset:0 atIndex:1];
      [enc setBuffer:aux offset:0 atIndex:2];
      [enc setBytes:prm length:sizeof(prm) atIndex:3];
      [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) { fprintf(stderr, "command buffer: %s\n", cb.error.description.UTF8String); return 1; }
      const uint32_t *o = (const uint32_t *)out.contents;
      uint64_t bad = 0; int64_t first = -1; uint32_t firstCode = 0;
      for (uint32_t i = 0; i < n; i++) if (o[i]) { if (first < 0) { first = i; firstCode = o[i]; } bad++; }
      if (bad) badReps++;
      printf("rep %d: %llu of %u threads wrong", rep, (unsigned long long)bad, n);
      if (bad) printf(" (first thread %lld: element mask 0x%x, last element read back as %u)", first, firstCode & 0xff, firstCode >> 8);
      printf("\n");
    }
    printf("SUMMARY: %d of %d dispatches produced wrong results (expected 0)\n", badReps, reps);
    return badReps ? 1 : 0;
  }
}
