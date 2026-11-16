// PROBE extern (part 2 of 2): definition of ext_fn used by extern_use.metal.
// EXPECT: ACCEPT
#include <metal_stdlib>
using namespace metal;

uint ext_fn(uint x) { return x * 5u + 1u; }
