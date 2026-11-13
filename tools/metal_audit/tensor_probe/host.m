// N1 tensor probe host: runs Metal kernels from one metallib interleaved in rounds and reports
// GPU time (command buffer GPUStartTime..GPUEndTime) per kernel. Swift-free: clang + Metal.
//
//   host <metallib> <rounds> <spec>...      spec = name:tg:ntg:iters:outBytes:inBytes[:chk]
//
// Buffers: 0 = out (zeroed), 1 = in A, 2 = in B (inBytes each, deterministic pseudo-random
// bytes, seed fixed), 3 = uint iters, 4 = uint aux (= ntg*tg, the thread count).
// chk = an output index group: specs with the same chk value must produce identical out buffers
// (bit-exact comparison across kernels after the last round); prints CHK lines.
// Env N1_DUMP=<dir> writes each kernel's out buffer to <dir>/<name>.bin.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>

typedef struct { NSString* name; NSUInteger tg, ntg, iters, outBytes, inBytes; NSString* chk;
                 id<MTLComputePipelineState> pso; id<MTLBuffer> out; NSMutableArray* t; } Spec;

static int cmpd(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return x < y ? -1 : x > y; }

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 4) { fprintf(stderr, "usage: host lib rounds spec...\n"); return 2; }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("DEVICE %s apple10=%d apple9=%d metal4=%d maxTGmem=%lu\n", dev.name.UTF8String,
           [dev supportsFamily:MTLGPUFamilyApple10], [dev supportsFamily:MTLGPUFamilyApple9],
           [dev supportsFamily:MTLGPUFamilyMetal4], (unsigned long)dev.maxThreadgroupMemoryLength);
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
    if (!lib) { fprintf(stderr, "lib: %s\n", err.description.UTF8String); return 1; }
    int rounds = atoi(argv[2]);
    NSMutableArray* specs = [NSMutableArray array];
    NSUInteger maxIn = 0;
    for (int i = 3; i < argc; i++) {
      NSArray* f = [@(argv[i]) componentsSeparatedByString:@":"];
      Spec* s = calloc(1, sizeof(Spec));
      s->name = f[0]; s->tg = [f[1] integerValue]; s->ntg = [f[2] integerValue];
      s->iters = [f[3] integerValue]; s->outBytes = [f[4] integerValue]; s->inBytes = [f[5] integerValue];
      s->chk = f.count > 6 ? f[6] : nil;
      id<MTLFunction> fn = [lib newFunctionWithName:s->name];
      if (!fn) { fprintf(stderr, "no function %s\n", s->name.UTF8String); return 1; }
      // N1_ARCHIVE=<dir>: check the R14 binary-archive path for this function first: compile into a
      // fresh MTLBinaryArchive, serialize it, reload it and build the PSO with
      // FailOnBinaryArchiveMiss (a hit = the archive holds the function's binary).
      const char* adir = getenv("N1_ARCHIVE");
      if (adir) {
        MTLComputePipelineDescriptor* cd = [MTLComputePipelineDescriptor new];
        cd.computeFunction = fn;
        MTLBinaryArchiveDescriptor* ad = [MTLBinaryArchiveDescriptor new];
        NSError* e1 = nil;
        id<MTLBinaryArchive> ar = [dev newBinaryArchiveWithDescriptor:ad error:&e1];
        BOOL added = ar && [ar addComputePipelineFunctionsWithDescriptor:cd error:&e1];
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithFormat:@"%s/%@.mtlarchive", adir, s->name]];
        BOOL saved = added && [ar serializeToURL:url error:&e1];
        BOOL hit = NO;
        if (saved) {
          MTLBinaryArchiveDescriptor* ad2 = [MTLBinaryArchiveDescriptor new];
          ad2.url = url;
          id<MTLBinaryArchive> ar2 = [dev newBinaryArchiveWithDescriptor:ad2 error:&e1];
          cd.binaryArchives = ar2 ? @[ ar2 ] : @[];
          id<MTLComputePipelineState> p2 = [dev newComputePipelineStateWithDescriptor:cd
              options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&e1];
          hit = p2 != nil;
        }
        NSNumber* sz = saved ? [[NSFileManager defaultManager] attributesOfItemAtPath:url.path error:nil][NSFileSize] : @0;
        printf("ARCHIVE %s add=%d serialize=%d reload_hit=%d bytes=%lld%s%s\n", s->name.UTF8String, added, saved, hit,
               sz.longLongValue, e1 ? " err=" : "", e1 ? e1.localizedDescription.UTF8String : "");
      }
      s->pso = [dev newComputePipelineStateWithFunction:fn error:&err];
      if (!s->pso) { fprintf(stderr, "pso %s: %s\n", s->name.UTF8String, err.description.UTF8String); return 1; }
      if (s->tg > s->pso.maxTotalThreadsPerThreadgroup) {
        fprintf(stderr, "%s: tg %lu > max %lu\n", s->name.UTF8String, s->tg, s->pso.maxTotalThreadsPerThreadgroup); return 1; }
      s->out = [dev newBufferWithLength:s->outBytes options:MTLResourceStorageModeShared];
      s->t = [NSMutableArray array];
      if (s->inBytes > maxIn) maxIn = s->inBytes;
      [specs addObject:[NSValue valueWithPointer:s]];
      printf("PSO %s maxTG=%lu exec=%lu staticTGmem=%lu\n", s->name.UTF8String, s->pso.maxTotalThreadsPerThreadgroup,
             s->pso.threadExecutionWidth, s->pso.staticThreadgroupMemoryLength);
    }
    // N1_INA / N1_INB: optional files that replace the start of buffer 1 / 2 (tables, Q tiles)
    const char* fa = getenv("N1_INA"); const char* fb = getenv("N1_INB");
    NSData* da = fa ? [NSData dataWithContentsOfFile:@(fa)] : nil;
    NSData* db = fb ? [NSData dataWithContentsOfFile:@(fb)] : nil;
    if (da.length > maxIn) maxIn = da.length;
    if (db.length > maxIn) maxIn = db.length;
    id<MTLBuffer> inA = [dev newBufferWithLength:maxIn options:MTLResourceStorageModeShared];
    id<MTLBuffer> inB = [dev newBufferWithLength:maxIn options:MTLResourceStorageModeShared];
    uint64_t x = 0x9e3779b97f4a7c15ull;
    for (NSUInteger i = 0; i < maxIn / 4; i++) {
      x ^= x << 13; x ^= x >> 7; x ^= x << 17; ((uint32_t*)inA.contents)[i] = (uint32_t)x;
      x ^= x << 13; x ^= x >> 7; x ^= x << 17; ((uint32_t*)inB.contents)[i] = (uint32_t)(x >> 32);
    }
    if (da) memcpy(inA.contents, da.bytes, da.length);
    if (db) memcpy(inB.contents, db.bytes, db.length);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    for (int r = -1; r < rounds; r++) {  // round -1 = warm-up, not recorded
      NSUInteger n = specs.count;
      for (NSUInteger k = 0; k < n; k++) {
        NSUInteger idx = (r & 1) ? n - 1 - k : k;  // odd rounds reverse the order
        Spec* s = [specs[idx] pointerValue];
        memset(s->out.contents, 0, s->outBytes);
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:s->pso];
        [e setBuffer:s->out offset:0 atIndex:0];
        [e setBuffer:inA offset:0 atIndex:1];
        [e setBuffer:inB offset:0 atIndex:2];
        uint32_t it = (uint32_t)s->iters, aux = (uint32_t)(s->ntg * s->tg);
        [e setBytes:&it length:4 atIndex:3];
        [e setBytes:&aux length:4 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(s->ntg, 1, 1) threadsPerThreadgroup:MTLSizeMake(s->tg, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "%s: %s\n", s->name.UTF8String, cb.error.description.UTF8String); return 1; }
        if (r >= 0) [s->t addObject:@((cb.GPUEndTime - cb.GPUStartTime) * 1e3)];
      }
    }
    const char* dump = getenv("N1_DUMP");
    for (NSValue* v in specs) {
      Spec* s = [v pointerValue];
      NSUInteger n = s->t.count; double* a = malloc(n * sizeof(double));
      for (NSUInteger i = 0; i < n; i++) a[i] = [s->t[i] doubleValue];
      qsort(a, n, sizeof(double), cmpd);
      printf("TIME %s tg=%lu ntg=%lu iters=%lu med=%.4f min=%.4f max=%.4f n=%lu\n", s->name.UTF8String, s->tg,
             s->ntg, s->iters, a[n / 2], a[0], a[n - 1], n);
      free(a);
      if (dump) {
        NSString* p = [NSString stringWithFormat:@"%s/%@.bin", dump, s->name];
        [[NSData dataWithBytes:s->out.contents length:s->outBytes] writeToFile:p atomically:NO];
      }
    }
    for (NSUInteger i = 0; i < specs.count; i++) {
      Spec* s = [specs[i] pointerValue];
      if (!s->chk) continue;
      for (NSUInteger j = 0; j < i; j++) {
        Spec* o = [specs[j] pointerValue];
        if (!o->chk || ![o->chk isEqualToString:s->chk]) continue;
        NSUInteger nb = MIN(s->outBytes, o->outBytes), bad = 0, first = NSNotFound;
        const uint32_t *pa = s->out.contents, *pb = o->out.contents;
        for (NSUInteger w = 0; w < nb / 4; w++) if (pa[w] != pb[w]) { bad++; if (first == NSNotFound) first = w; }
        printf("CHK %s vs %s: %lu/%lu words differ%s\n", s->name.UTF8String, o->name.UTF8String, bad, nb / 4,
               bad ? [NSString stringWithFormat:@" (first %lu: %08x vs %08x)", first, pa[first], pb[first]].UTF8String : "");
        break;
      }
    }
  }
  return 0;
}
