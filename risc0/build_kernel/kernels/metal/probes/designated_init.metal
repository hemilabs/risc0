// PROBE designated_init: C++20 designated initializers in program-scope `constant` tables.
// EXPECT: ACCEPT
// Runtime (runner `designated`): out[i] = TAB[i].a * 1000 + TAB[i].b * 10 + TAB[i].c, i < 4.
#include <metal_stdlib>
using namespace metal;

struct Entry {
  uint a;
  uint b;
  uint c;
};

constant constexpr Entry TAB[4] = {
    {.a = 1, .b = 2, .c = 3},
    {.a = 4, .b = 5, .c = 6},
    {.a = 7, .c = 9},  // .b omitted -> zero-initialized
    {.a = 10, .b = 11, .c = 12},
};

struct Layout {
  uint offset;
  uint size;
};
constant constexpr Layout LAYOUT = {.offset = 100, .size = 7};

kernel void designated(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  if (gid < 4) {
    out[gid] = TAB[gid].a * 1000u + TAB[gid].b * 10u + TAB[gid].c;
  } else if (gid == 4) {
    out[gid] = LAYOUT.offset + LAYOUT.size;
  }
}
