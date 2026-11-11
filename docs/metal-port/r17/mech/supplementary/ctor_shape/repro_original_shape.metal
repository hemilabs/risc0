// R17 minrepro m1: hand-built copy of the witgen ToBits<5> chain with the real constructs.
#include <metal_stdlib>
using namespace metal;

namespace r0 {
template <typename T, size_t N> struct array {
  T elems[N];
  thread T& operator[](size_t i) thread { return elems[i]; }
  const thread T& operator[](size_t i) const thread { return elems[i]; }
  const constant T& operator[](size_t i) const constant { return elems[i]; }
  static constexpr size_t size() { return N; }
};
}

class Fp {
public:
  static constant uint32_t P = 15 * (uint32_t(1) << 27) + 1;
  static constant uint32_t M = 0x88000001;
  static constant uint32_t R2 = 1172168163;
private:
  uint32_t val;
  static constexpr uint32_t add(uint32_t a, uint32_t b) { uint32_t r = a + b; return (r >= P ? r - P : r); }
  static constexpr uint32_t sub(uint32_t a, uint32_t b) { uint32_t r = a - b; return (r > P ? r + P : r); }
  static constexpr uint32_t mul(uint32_t a, uint32_t b) {
    uint64_t o64 = uint64_t(a) * uint64_t(b);
    uint32_t low = -uint32_t(o64);
    uint32_t red = M * low;
    o64 += uint64_t(red) * uint64_t(P);
    uint32_t ret = o64 >> 32;
    return (ret >= P ? ret - P : ret);
  }
  static constexpr uint32_t encode(uint32_t a) { return mul(R2, a); }
  static constexpr uint32_t decode(uint32_t a) { return mul(1, a); }
  constexpr Fp(uint32_t val, bool) : val(val) {}
  constexpr Fp(uint32_t val, bool) device : val(val) {}
public:
  constexpr Fp() : val(0) {}
  constexpr Fp() device : val(0) {}
  constexpr Fp(uint32_t val) : val(encode(val)) {}
  constexpr Fp(uint32_t val) constant : val(encode(val)) {}
  constexpr uint32_t asUInt32() const { return decode(val); }
  friend constexpr Fp operator+(Fp a, Fp b) { return Fp(add(a.val, b.val), true); }
  friend constexpr Fp operator-(Fp a, Fp b) { return Fp(sub(a.val, b.val), true); }
  friend constexpr Fp operator*(Fp a, Fp b) { return Fp(mul(a.val, b.val), true); }
  friend constexpr bool operator==(Fp a, Fp b) { return a.val == b.val; }
};

inline Fp inv_chain(Fp x) {
  Fp x2 = x * x;
  Fp x3 = x2 * x;
  Fp x6 = x3 * x3;
  Fp y3 = x6 * x;
  Fp t = y3;
#pragma unroll
  for (int i = 0; i < 3; i++) t = t * t;
  Fp y6 = t * y3;
  Fp z = y3;
#pragma unroll
  for (int i = 0; i < 4; i++) z = z * z;
  z = z * y3;
#pragma unroll
  for (int k = 0; k < 4; k++) {
#pragma unroll
    for (int i = 0; i < 6; i++) z = z * z;
    z = z * y6;
  }
  return z;
}

using Val = Fp;
struct Buffer {
  device Fp* buf;
  uint32_t rows;
  uint32_t cols;
  void set(uint32_t row, uint32_t col, Fp v) const { buf[col * rows + row] = v; }
  Fp get(uint32_t row, uint32_t col) const { return buf[col * rows + row]; }
};
struct LookupTables { device atomic_uint* t8; device atomic_uint* t16; uint32_t bad; };
struct ExecContext {
  ExecContext(thread LookupTables& tables, uint32_t cycle) : tables(tables), cycle(cycle) {}
  thread LookupTables& tables;
  uint32_t cycle;
};
struct BufferObj {
  uint32_t kind;
  Buffer buf;
  Buffer preDataBuf;
  uint32_t zeroBack;
  Val loadMutable(thread const ExecContext& ctx, uint32_t col, uint32_t back) const {
    if (zeroBack && col > zeroBack && back > 0) return Val(0);
    uint32_t backRow = (ctx.cycle - back) & (buf.rows - 1);
    if (back > 0 && preDataBuf.buf) return preDataBuf.get(backRow, col);
    return buf.get(backRow, col);
  }
  void storeMutable(thread const ExecContext& ctx, uint32_t col, Val v) const { buf.set(ctx.cycle, col, v); }
  Val load(thread const ExecContext& ctx, uint32_t col, uint32_t back) const {
    if (kind == 1) return buf.get(0, col);
    return loadMutable(ctx, col, back);
  }
  void store(thread const ExecContext& ctx, uint32_t col, Val v) const {
    if (kind == 1) buf.set(0, col, v); else storeMutable(ctx, col, v);
  }
};
template <typename T> struct BoundLayout {
  BoundLayout(const constant T& layout, thread BufferObj* buf) : layout(layout), buf(buf) {}
  const constant T& layout;
  thread BufferObj* buf;
};
template <typename T> inline BoundLayout<T> bindLayout(const constant T& layout, thread BufferObj* buf) {
  return BoundLayout<T>(layout, buf);
}
inline size_t to_size_t(Val v) { return v.asUInt32(); }
inline Val inv_0(Val x) { return inv_chain(x); }
inline Val bitAnd(Val a, Val b) { return Val(a.asUInt32() & b.asUInt32()); }
struct Reg {
  constexpr Reg(size_t col) : col(uint32_t(col)) {}
  constexpr Reg(size_t col) constant : col(uint32_t(col)) {}
  uint32_t col;
};
inline void store(thread ExecContext& ctx, BoundLayout<Reg> reg, Val val) { reg.buf->store(ctx, reg.layout.col, val); }
inline Val load(thread ExecContext& ctx, BoundLayout<Reg> reg, size_t back) { return reg.buf->load(ctx, reg.layout.col, back); }
#define LAYOUT_LOOKUP(orig, elem) bindLayout(orig.layout.elem, orig.buf)
#define LAYOUT_SUBSCRIPT(orig, index) bindLayout(orig.layout[index], orig.buf)
#define LOAD(reg, back) load(ctx, reg, back)
#define STORE(reg, val) store(ctx, reg, val)

struct NondetRegLayout { Reg _super; };
using NondetRegLayout5LayoutArray = r0::array<NondetRegLayout, 5>;
struct ToBits_5_Layout { NondetRegLayout5LayoutArray _super; };
struct NondetRegStruct { Val _super; };
using Val5Array = r0::array<Val, 5>;
using Val30Array = r0::array<Val, 30>;
using NondetRegStruct5Array = r0::array<NondetRegStruct, 5>;
struct ToBits_5_Struct { NondetRegStruct5Array _super; };

constant ToBits_5_Layout kLayout_low5 = ToBits_5_Layout{._super = NondetRegLayout5LayoutArray{
    NondetRegLayout{._super = 134}, NondetRegLayout{._super = 135}, NondetRegLayout{._super = 136},
    NondetRegLayout{._super = 137}, NondetRegLayout{._super = 138}}};

#define FNATTR __attribute__((noinline))

static FNATTR NondetRegStruct exec_NondetReg_W(thread ExecContext& ctx, Val arg0, BoundLayout<NondetRegLayout> layout1) {
  STORE(LAYOUT_LOOKUP(layout1, _super), arg0);
  NondetRegStruct x2 = NondetRegStruct{._super = LOAD(LAYOUT_LOOKUP(layout1, _super), 0)};
  return x2;
}
static FNATTR NondetRegStruct exec_NondetBitReg_W(thread ExecContext& ctx, Val arg0, BoundLayout<NondetRegLayout> layout1) {
  NondetRegStruct x2 = exec_NondetReg_W(ctx, arg0, layout1);
  Val x3 = (x2._super * (Val(1) - x2._super));
  (void)x3;
  return x2;
}
static FNATTR ToBits_5_Struct exec_ToBits_5__W(thread ExecContext& ctx, Val arg0, BoundLayout<ToBits_5_Layout> layout1) {
  auto c2m_a7 = Val5Array{Val(0), Val(1), Val(2), Val(3), Val(4)};
  auto c2m_b7 = LAYOUT_LOOKUP(layout1, _super);
  NondetRegStruct5Array x2;
  for (uint c2m_i7 = 0; c2m_i7 < c2m_a7.size(); c2m_i7++) {
    Val x3 = c2m_a7[c2m_i7];
    BoundLayout<NondetRegLayout> x4 = LAYOUT_SUBSCRIPT(c2m_b7, c2m_i7);
    Val x5 = inv_0(Val30Array{Val(1), Val(2), Val(4), Val(8), Val(16), Val(32), Val(64), Val(128),
                              Val(256), Val(512), Val(1024), Val(2048), Val(4096), Val(8192), Val(16384), Val(32768),
                              Val(65536), Val(131072), Val(262144), Val(524288), Val(1048576), Val(2097152),
                              Val(4194304), Val(8388608), Val(16777216), Val(33554432), Val(67108864),
                              Val(134217728), Val(268435456), Val(536870912)}[to_size_t(x3)]);
    NondetRegStruct x6 = exec_NondetBitReg_W(
        ctx,
        (bitAnd(arg0, Val30Array{Val(1), Val(2), Val(4), Val(8), Val(16), Val(32), Val(64), Val(128),
                                 Val(256), Val(512), Val(1024), Val(2048), Val(4096), Val(8192), Val(16384), Val(32768),
                                 Val(65536), Val(131072), Val(262144), Val(524288), Val(1048576), Val(2097152),
                                 Val(4194304), Val(8388608), Val(16777216), Val(33554432), Val(67108864),
                                 Val(134217728), Val(268435456), Val(536870912)}[to_size_t(x3)]) * x5),
        x4);
    x2[c2m_i7] = x6;
  }
  return ToBits_5_Struct{._super = x2};
}

kernel void test_kernel(device Fp* scratch [[buffer(0)]], device uint* out [[buffer(1)]],
                        device atomic_uint* aux [[buffer(2)]], constant uint4& prm [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= prm.x) return;
  LookupTables tables{aux + 65536, aux, 0};
  ExecContext ctx(tables, 0);
  BufferObj obj;
  obj.kind = 0;
  obj.buf = Buffer{scratch + gid * prm.z, 1, 211};
  obj.preDataBuf = Buffer{nullptr, 0, 0};
  obj.zeroBack = 0;
  uint shift = (gid * 7u + prm.y) & 31u;
  ToBits_5_Struct r5 = exec_ToBits_5__W(ctx, Val(shift), bindLayout(kLayout_low5, &obj));
  uint code = 0;
  for (uint i = 0; i < 5; i++)
    if (r5._super[i]._super.asUInt32() != ((shift >> i) & 1u)) code |= (1u << i);
  if (tables.bad) code |= 0x80;
  out[gid] = code ? (code | (shift << 8)) : 0;
}
