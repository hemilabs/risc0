// Land-ready GPU kernels for the finalize "combos" phase (Intel HAL).
//
//  combos_prepare_v2 : device-side U-coefficient subtraction (replaces the
//                      80 MB D2H + host loop + 80 MB H2D in esimd_combos_prepare).
//  combos_divide_v2  : blocked synthetic division of every combo column by
//                      (x - z_k), batched per "round" over columns, with a
//                      W-lane SLM suffix scan for the inter-block carries.
//                      Remainders are read back and checked == 0 (hard error).
//  mix_poly_coeffs_v2: register-accumulating variant of esimd_mix_poly_coeffs
//                      (one RMW of the combos rows at the end instead of one
//                      gather/scatter RMW per input polynomial).
//
// All arithmetic is exact BabyBear / BabyBear^4 (Montgomery, canonical < P),
// so results are bit-identical to the CPU path regardless of evaluation order.
//
// Self-contained (only <sycl/sycl.hpp>) so it can be pasted into intel_ffi.cpp
// / fri_poly_ops.cpp of risc0/sys/kernels/zkp/intel.
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

#ifndef COMBOS_BOUNDS_CHECK
#define COMBOS_BOUNDS_CHECK 0
#endif

namespace combos_gpu {

static constexpr uint32_t P = 0x78000001;
static constexpr uint32_t M = 0x88000001;
static constexpr uint32_t NBETA = 0x40000018;  // Montgomery(-11)
static constexpr uint32_t ONE = 0x0ffffffe;    // Montgomery(1) = 2^32 mod P

static inline uint32_t fp_add(uint32_t a, uint32_t b) { uint32_t r = a + b; return r >= P ? r - P : r; }
static inline uint32_t fp_sub(uint32_t a, uint32_t b) { uint32_t r = a - b; return r > P ? r + P : r; }
static inline uint32_t fp_mul(uint32_t a, uint32_t b) {
  uint64_t o64 = uint64_t(a) * uint64_t(b);
  uint32_t low = -uint32_t(o64);
  uint32_t red = M * low;
  o64 += uint64_t(red) * uint64_t(P);
  uint32_t ret = uint32_t(o64 >> 32);
  return ret >= P ? ret - P : ret;
}
struct E4 { uint32_t e[4]; };
static inline E4 e_zero() { return {{0, 0, 0, 0}}; }
static inline E4 e_one() { return {{ONE, 0, 0, 0}}; }
static inline E4 e_add(E4 a, E4 b) { return {{fp_add(a.e[0], b.e[0]), fp_add(a.e[1], b.e[1]), fp_add(a.e[2], b.e[2]), fp_add(a.e[3], b.e[3])}}; }
static inline E4 e_sub(E4 a, E4 b) { return {{fp_sub(a.e[0], b.e[0]), fp_sub(a.e[1], b.e[1]), fp_sub(a.e[2], b.e[2]), fp_sub(a.e[3], b.e[3])}}; }
static inline E4 e_mul(E4 a, E4 b) {
  uint32_t nb = NBETA;
  return {{fp_add(fp_mul(a.e[0], b.e[0]), fp_mul(nb, fp_add(fp_add(fp_mul(a.e[1], b.e[3]), fp_mul(a.e[2], b.e[2])), fp_mul(a.e[3], b.e[1])))),
           fp_add(fp_add(fp_mul(a.e[0], b.e[1]), fp_mul(a.e[1], b.e[0])), fp_mul(nb, fp_add(fp_mul(a.e[2], b.e[3]), fp_mul(a.e[3], b.e[2])))),
           fp_add(fp_add(fp_add(fp_mul(a.e[0], b.e[2]), fp_mul(a.e[1], b.e[1])), fp_mul(a.e[2], b.e[0])), fp_mul(nb, fp_mul(a.e[3], b.e[3]))),
           fp_add(fp_add(fp_add(fp_mul(a.e[0], b.e[3]), fp_mul(a.e[1], b.e[2])), fp_mul(a.e[2], b.e[1])), fp_mul(a.e[3], b.e[0]))}};
}
static inline E4 e_pow(E4 x, uint64_t n) { E4 r = e_one(); while (n) { if (n & 1) r = e_mul(r, x); x = e_mul(x, x); n >>= 1; } return r; }
static inline bool e_is_zero(E4 a) { return (a.e[0] | a.e[1] | a.e[2] | a.e[3]) == 0; }
static inline E4 e_ld(const uint32_t* p) { return {{p[0], p[1], p[2], p[3]}}; }
static inline void e_st(uint32_t* p, E4 v) { p[0] = v.e[0]; p[1] = v.e[1]; p[2] = v.e[2]; p[3] = v.e[3]; }

// Bounds-check helper (debug builds / CPU validation runs only).
#if COMBOS_BOUNDS_CHECK
#define CB_CHECK(flag, cond) do { if (!(cond)) { sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> a(*(flag)); a.fetch_add(1u); } } while (0)
#else
#define CB_CHECK(flag, cond) do { (void)(flag); } while (0)
#endif

// ---------------------------------------------------------------------------
// combos_prepare_v2
//   combos[(cycles*combo_id(rg) + j)*4 ..] -= mix^rg * coeffU[pos(rg)+j]    j < size(rg)
//   combos[(cycles*combo_count)*4 ..]      -= mix^(regs+ci) * coeffU[posEnd+ci]  ci < check
// One work-group of PW lanes. Phase 1: mix powers + reg prefix offsets in SLM.
// Phase 2: one lane per (combo, j) target, summing its contributions.
// regs+check must be <= MAXR (rv32im 331, recursion < 1024); else error.
// ---------------------------------------------------------------------------
static constexpr uint32_t PW = 1024;
static constexpr uint32_t MAXR = 2048;
static constexpr uint32_t MAXJ = 64;   // max reg size (rv32im 6); targets per combo handled with stride
inline sycl::event combos_prepare_v2(sycl::queue& q, uint32_t* combos, const uint32_t* coeffU,
                                     uint32_t comboCount, uint32_t cycles, uint32_t regsCount,
                                     const uint32_t* regSizes, const uint32_t* regComboIds,
                                     uint32_t checkSize, const uint32_t* mixp, uint32_t* errflag) {
  if (regsCount + checkSize > MAXR) throw std::runtime_error("combos_prepare_v2: too many regs");
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> powS(MAXR * 4, h);
    sycl::local_accessor<uint32_t, 1> posS(MAXR + 1, h);
    sycl::local_accessor<uint32_t, 1> maxS(1, h);
    h.parallel_for(sycl::nd_range<1>(PW, PW), [=](sycl::nd_item<1> it) {
      const uint32_t l = it.get_local_id(0);
      const uint32_t nr = regsCount + checkSize;
      const E4 mix = e_ld(mixp);
      for (uint32_t k = l; k < nr; k += PW) e_st(&powS[k * 4], e_pow(mix, k));
      if (l == 0) {
        uint32_t pos = 0, mx = 0;
        for (uint32_t rg = 0; rg < regsCount; rg++) {
          posS[rg] = pos; uint32_t s = regSizes[rg]; pos += s; mx = s > mx ? s : mx;
          CB_CHECK(errflag, regComboIds[rg] < comboCount);
          CB_CHECK(errflag, s <= cycles);
        }
        posS[regsCount] = pos; maxS[0] = mx;
      }
      sycl::group_barrier(it.get_group());
      const uint32_t mx = maxS[0];
      // targets: combo c in [0, comboCount), row j in [0, mx); plus (comboCount, 0) for check
      const uint32_t ntar = comboCount * mx + 1;
      for (uint32_t t = l; t < ntar; t += PW) {
        E4 acc = e_zero(); bool any = false; size_t dst;
        if (t < comboCount * mx) {
          const uint32_t c = t / mx, j = t % mx;
          for (uint32_t rg = 0; rg < regsCount; rg++) {
            if (regComboIds[rg] == c && j < regSizes[rg]) {
              const E4 pw = e_ld(&powS[rg * 4]);
              acc = e_add(acc, e_mul(pw, e_ld(coeffU + size_t(posS[rg] + j) * 4)));
              any = true;
            }
          }
          dst = (size_t(cycles) * c + j) * 4;
        } else {
          const uint32_t pe = posS[regsCount];
          for (uint32_t ci = 0; ci < checkSize; ci++) {
            acc = e_add(acc, e_mul(e_ld(&powS[(regsCount + ci) * 4]), e_ld(coeffU + size_t(pe + ci) * 4)));
            any = true;
          }
          dst = size_t(cycles) * comboCount * 4;
        }
        if (any) {
          CB_CHECK(errflag, dst + 4 <= size_t(comboCount + 1) * cycles * 4);
          e_st(combos + dst, e_sub(e_ld(combos + dst), acc));
        }
      }
    });
  });
}

// ---------------------------------------------------------------------------
// combos_divide_v2
// ---------------------------------------------------------------------------
struct DivCol {           // one division in one round (device-side parameter block)
  uint64_t off;           // element (u32) offset of the column = col * rows * 4
  E4 z, zB;               // z and z^B
  E4 pw[16];              // (zB^R)^(2^i)
  E4 zn;                  // z of this column's next division (fused C->A)
  uint32_t rem_idx;       // index into remainder array
  int32_t next;           // slot of this column in the next round, or -1
  uint32_t pad[2];
};

struct DivPlan { uint32_t B, nb, W, R; };
inline DivPlan div_plan(uint32_t rows) {
  // rows must be a power of two
  DivPlan p;
  p.B = rows < 64 ? rows : 64;
  p.nb = rows / p.B;
  p.W = p.nb < 1024 ? p.nb : 1024;
  p.R = p.nb / p.W;
  return p;
}

// info: host array of (col, z[4]) entries in caller order (per column, the
// division order must be preserved). Returns via h_rem the remainders in the
// same order. Synchronous: waits for completion (remainder check needs it).
// Scratch L/C are allocated per call (rows/B * maxcols E4 each, ~1.25 MB at po2=20).
struct DivEntry { uint32_t col; E4 z; };
inline std::string combos_divide_v2(sycl::queue& q, uint32_t* combos, uint32_t rows, uint32_t ncols,
                                    const std::vector<DivEntry>& info, std::vector<E4>& h_rem,
                                    uint32_t* errflag, std::vector<sycl::event>* evs = nullptr,
                                    bool fuse = true) {
  if (rows == 0 || (rows & (rows - 1))) return "combos_divide_v2: rows not a power of two";
  for (auto& e : info) if (e.col >= ncols) return "combos_divide_v2: column out of range";
  const DivPlan pl = div_plan(rows);
  const uint32_t B = pl.B, nb = pl.nb, W = pl.W, R = pl.R;
  uint32_t lgW = 0; while ((1u << lgW) < W) lgW++;
  // Group into rounds: round r holds the r-th division of each column.
  std::vector<std::vector<DivCol>> rounds;
  std::vector<uint32_t> seen(ncols, 0);
  for (uint32_t k = 0; k < info.size(); k++) {
    const auto& e = info[k];
    uint32_t r = seen[e.col]++;
    if (rounds.size() <= r) rounds.resize(r + 1);
    DivCol d{};
    d.off = uint64_t(e.col) * rows * 4;
    d.z = e.z; d.zB = e_pow(e.z, B);
    E4 t = e_pow(d.zB, R);
    for (int i = 0; i < 16; i++) { d.pw[i] = t; t = e_mul(t, t); }
    d.rem_idx = k; d.next = -1;
    rounds[r].push_back(d);
  }
  // link each slot to the same column's slot in the following round
  for (size_t r = 0; r + 1 < rounds.size(); r++)
    for (auto& d : rounds[r])
      for (size_t t = 0; t < rounds[r + 1].size(); t++)
        if (rounds[r + 1][t].off == d.off) { d.next = int32_t(t); d.zn = rounds[r + 1][t].z; }
  size_t maxnc = 0, tot = 0;
  for (auto& r : rounds) { maxnc = std::max(maxnc, r.size()); tot += r.size(); }
  if (tot == 0) { h_rem.clear(); return ""; }
  E4* L = sycl::malloc_device<E4>(size_t(nb) * maxnc, q);
  E4* C = sycl::malloc_device<E4>(size_t(nb) * maxnc, q);
  DivCol* dP = sycl::malloc_device<DivCol>(tot, q);
  E4* dRem = sycl::malloc_device<E4>(info.size(), q);
  std::vector<DivCol> flat; flat.reserve(tot);
  for (auto& r : rounds) flat.insert(flat.end(), r.begin(), r.end());
  q.memcpy(dP, flat.data(), tot * sizeof(DivCol));
  const size_t colElems = size_t(rows) * 4, total = colElems * ncols, Ltot = size_t(nb) * maxnc;
  size_t base = 0;
  for (size_t ri = 0; ri < rounds.size(); ri++) {
    auto& rr = rounds[ri];
    const uint32_t nc = uint32_t(rr.size());
    const DivCol* cp = dP + base;
    base += nc;
    // (A) per-block Horner with zero carry-in (only round 0 when fused)
    sycl::event ea;
    if (!fuse || ri == 0) ea = q.parallel_for(sycl::range<1>(size_t(nb) * nc), [=](sycl::id<1> id) {
      const size_t g = id[0], c = g / nb, b = g % nb;
      const DivCol& d = cp[c];
      const uint32_t* s = combos + d.off + b * B * 4;
      CB_CHECK(errflag, d.off + (b + 1) * B * 4 <= total && g < Ltot);
      E4 acc = e_zero();
      for (uint32_t k = B; k-- > 0;) acc = e_add(e_mul(acc, d.z), e_ld(s + k * 4));
      L[g] = acc;
    });
    // (B) per column: lane aggregates over R blocks, suffix scan, carries
    auto eb = q.submit([&](sycl::handler& h) {
      sycl::local_accessor<E4, 1> S(W, h), D(W, h);
      h.parallel_for(sycl::nd_range<1>(size_t(W) * nc, W), [=](sycl::nd_item<1> it) {
        const uint32_t l = it.get_local_id(0); const size_t c = it.get_group(0);
        const DivCol& d = cp[c];
        const E4* Lc = L + c * nb; E4* Cc = C + c * nb; const E4 zB = d.zB;
        E4 s = e_zero();
        for (uint32_t j = R; j-- > 0;) s = e_add(e_mul(s, zB), Lc[size_t(l) * R + j]);
        S[l] = s;
        sycl::group_barrier(it.get_group());
        E4 v = (l + 1 < W) ? S[l + 1] : e_zero();
        D[l] = v;
        sycl::group_barrier(it.get_group());
        for (uint32_t st = 1, i = 0; i < lgW; st <<= 1, i++) {
          E4 o = (l + st < W) ? D[l + st] : e_zero();
          sycl::group_barrier(it.get_group());
          v = e_add(v, e_mul(d.pw[i], o));
          D[l] = v;
          sycl::group_barrier(it.get_group());
        }
        E4 cc = D[l];
        for (uint32_t j = R; j-- > 0;) { const size_t b = size_t(l) * R + j; Cc[b] = cc; cc = e_add(Lc[b], e_mul(zB, cc)); }
        if (l == 0) dRem[d.rem_idx] = cc;
      });
    });
    // (C) rewrite each block with its carry-in
    auto ec = q.parallel_for(sycl::range<1>(size_t(nb) * nc), [=](sycl::id<1> id) {
      const size_t g = id[0], c = g / nb, b = g % nb;
      const DivCol& d = cp[c];
      uint32_t* s = combos + d.off + b * B * 4;
      E4 cur = C[g];
      if (fuse && d.next >= 0) {
        // also produce next round's block Horner value over the new quotient block
        const E4 zn = d.zn; E4 acc = e_zero();
        for (uint32_t k = B; k-- > 0;) { E4 nx = e_add(e_mul(d.z, cur), e_ld(s + k * 4)); e_st(s + k * 4, cur); acc = e_add(e_mul(acc, zn), cur); cur = nx; }
        CB_CHECK(errflag, size_t(d.next) * nb + b < Ltot);
        L[size_t(d.next) * nb + b] = acc;
      } else {
        for (uint32_t k = B; k-- > 0;) { E4 nx = e_add(e_mul(d.z, cur), e_ld(s + k * 4)); e_st(s + k * 4, cur); cur = nx; }
      }
    });
    if (evs) { if (!fuse || ri == 0) evs->push_back(ea); evs->push_back(eb); evs->push_back(ec); }
  }
  h_rem.resize(info.size());
  q.memcpy(h_rem.data(), dRem, info.size() * sizeof(E4));
  q.wait();
  sycl::free(L, q); sycl::free(C, q); sycl::free(dP, q); sycl::free(dRem, q);
  for (size_t k = 0; k < h_rem.size(); k++)
    if (!e_is_zero(h_rem[k])) return "combos_divide_v2: nonzero remainder at entry " + std::to_string(k) + " (col " + std::to_string(info[k].col) + ")";
  return "";
}

// ---------------------------------------------------------------------------
// mix_poly_coeffs_v2: one work-item per row i.
//   out[(count*combo(p) + i)] += mix_start*mix^p * in[count*p + i]
// Step 1 (tiny kernel): pw[p] = mix_start * mix^p into a device scratch table.
// Step 2: per row, accumulate per combo id in registers (ids < MAXC); one RMW
// per (combo,row) at the end. Ids >= MAXC fall back to an immediate RMW (each
// row is owned by exactly one work-item, so no atomics are needed either way).
// ---------------------------------------------------------------------------
static constexpr uint32_t MAXC = 8;
inline sycl::event mix_pw_table(sycl::queue& q, uint32_t* pw, const uint32_t* mixStart, const uint32_t* mixp, uint32_t inputSize) {
  return q.parallel_for(sycl::range<1>(inputSize), [=](sycl::id<1> id) {
    const uint32_t p = id[0];
    e_st(pw + size_t(p) * 4, e_mul(e_ld(mixStart), e_pow(e_ld(mixp), p)));
  });
}
inline sycl::event mix_poly_coeffs_v2(sycl::queue& q, uint32_t* out, const uint32_t* in, const uint32_t* combos,
                                      const uint32_t* pw, uint32_t inputSize, uint32_t count, uint32_t* errflag,
                                      size_t outElems) {
  return q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
    const size_t i = id[0];
    E4 acc[MAXC];
#pragma unroll
    for (uint32_t c = 0; c < MAXC; c++) acc[c] = e_zero();
    uint32_t used = 0;
    for (uint32_t p = 0; p < inputSize; p++) {
      const uint32_t cid = combos[p];
      const uint32_t x = in[size_t(count) * p + i];
      const E4 cur = e_ld(pw + size_t(p) * 4);
      const E4 t = {{fp_mul(cur.e[0], x), fp_mul(cur.e[1], x), fp_mul(cur.e[2], x), fp_mul(cur.e[3], x)}};
      if (cid < MAXC) {
#pragma unroll
        for (uint32_t c = 0; c < MAXC; c++) if (c == cid) acc[c] = e_add(acc[c], t);
        used |= 1u << cid;
      } else {
        uint32_t* o = out + (size_t(count) * cid + i) * 4;
        CB_CHECK(errflag, (size_t(count) * cid + i) * 4 + 4 <= outElems);
        e_st(o, e_add(e_ld(o), t));
      }
    }
#pragma unroll
    for (uint32_t c = 0; c < MAXC; c++) if (used & (1u << c)) {
      CB_CHECK(errflag, (size_t(count) * c + i) * 4 + 4 <= outElems);
      uint32_t* o = out + (size_t(count) * c + i) * 4;
      e_st(o, e_add(e_ld(o), acc[c]));
    }
  });
}

// mix_poly_coeffs_v3: 4 consecutive rows per work-item (16-byte input loads), one
// running accumulator per row, flushed (RMW of 64 contiguous bytes) whenever the
// combo id changes between consecutive inputs (rv32im groups have 3/1/9/1 runs).
// Requires count % 4 == 0 (caller falls back to v2 otherwise).
inline sycl::event mix_poly_coeffs_v3(sycl::queue& q, uint32_t* out, const uint32_t* in, const uint32_t* combos,
                                      const uint32_t* pw, uint32_t inputSize, uint32_t count, uint32_t* errflag,
                                      size_t outElems) {
  return q.parallel_for(sycl::range<1>(count / 4), [=](sycl::id<1> id) {
    const size_t i0 = id[0] * 4;
    E4 acc[4];
    uint32_t cur = 0xffffffffu;
    auto flush = [&]() {
      if (cur == 0xffffffffu) return;
      uint32_t* o = out + (size_t(count) * cur + i0) * 4;
      CB_CHECK(errflag, (size_t(count) * cur + i0) * 4 + 16 <= outElems);
      sycl::vec<uint32_t, 16> v = *reinterpret_cast<sycl::vec<uint32_t, 16>*>(o);
#pragma unroll
      for (int r = 0; r < 4; r++) {
        v[4 * r + 0] = fp_add(v[4 * r + 0], acc[r].e[0]); v[4 * r + 1] = fp_add(v[4 * r + 1], acc[r].e[1]);
        v[4 * r + 2] = fp_add(v[4 * r + 2], acc[r].e[2]); v[4 * r + 3] = fp_add(v[4 * r + 3], acc[r].e[3]);
      }
      *reinterpret_cast<sycl::vec<uint32_t, 16>*>(o) = v;
    };
    for (uint32_t p = 0; p < inputSize; p++) {
      const uint32_t cid = combos[p];
      if (cid != cur) {
        flush();
        cur = cid;
#pragma unroll
        for (int r = 0; r < 4; r++) acc[r] = e_zero();
      }
      const sycl::vec<uint32_t, 4> x = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(in + size_t(count) * p + i0);
      const E4 w = e_ld(pw + size_t(p) * 4);
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const uint32_t xr = x[r];
        acc[r] = e_add(acc[r], E4{{fp_mul(w.e[0], xr), fp_mul(w.e[1], xr), fp_mul(w.e[2], xr), fp_mul(w.e[3], xr)}});
      }
    }
    flush();
  });
}

}  // namespace combos_gpu
