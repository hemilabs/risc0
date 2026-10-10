extern "C++" {
// v3: SLM pass with T threads per 2^14 block (T=32 -> 2 WGs/Xe-core can co-reside).
// Thread lid holds dv[c] = block[c*CHUNK + lid*16 + lane], CHUNK = 16*T, c < LOADS = 1024/T.
//   lane bits (0..3)        -> stages 1..4 in register (iselect)
//   lid  bits (4..LGC-1)    -> stages 5..LGC through SLM
//   c    bits (LGC..13)     -> stages LGC+1..14 in register across dv[]
// Per-butterfly arithmetic identical to production => bit-identical output.

template <uint32_t T> struct SlmGeo {
  static constexpr uint32_t LOADS = SLM_BLOCK / (T * 16);
  static constexpr uint32_t CHUNK = T * 16;
  static constexpr uint32_t LGC = (T == 16) ? 8 : (T == 32) ? 9 : (T == 64) ? 10 : (T == 128) ? 11 : 0;
  static constexpr uint32_t PAIRS = SLM_BLOCK / 32 / T;
};

template <uint32_t T, uint32_t S>
ESIMD_INLINE void reg_ct(bb31::Vec16* dv, const uint32_t* ptw, uint32_t off, uint32_t lid) {
  using G = SlmGeo<T>;
  constexpr uint32_t b = S - (G::LGC + 1);
#pragma unroll
  for (uint32_t ct = 0; ct < G::LOADS; ct++) {
    if (ct & (1u << b)) continue;
    uint32_t cb = ct | (1u << b);
    uint32_t j = (ct & ((1u << b) - 1)) * G::CHUNK + lid * 16u;
    auto twv = esimd::block_load<uint32_t, 16>(ptw + off + j);
    auto tv = bb31::mont_mul(twv, dv[cb]);
    auto u = dv[ct];
    dv[ct] = bb31::field_add(u, tv);
    dv[cb] = bb31::field_sub(u, tv);
  }
}
template <uint32_t T, uint32_t S>
ESIMD_INLINE void reg_gs(bb31::Vec16* dv, const uint32_t* ptw, uint32_t off, uint32_t lid) {
  using G = SlmGeo<T>;
  constexpr uint32_t b = S - (G::LGC + 1);
#pragma unroll
  for (uint32_t ct = 0; ct < G::LOADS; ct++) {
    if (ct & (1u << b)) continue;
    uint32_t cb = ct | (1u << b);
    uint32_t j = (ct & ((1u << b) - 1)) * G::CHUNK + lid * 16u;
    auto u = dv[ct];
    auto v = dv[cb];
    auto sv = bb31::field_add(u, v);
    auto d2 = bb31::field_sub(u, v);
    auto twv = esimd::block_load<uint32_t, 16>(ptw + off + j);
    dv[ct] = sv;
    dv[cb] = bb31::mont_mul(twv, d2);
  }
}
template <uint32_t T>
ESIMD_INLINE void slm_ct(uint32_t STAGE, const uint32_t* ptw, uint32_t off, uint32_t lid) {
  using G = SlmGeo<T>;
  uint32_t half = 1u << (STAGE - 1); uint32_t m = 1u << STAGE;
#pragma unroll
  for (uint32_t p = 0; p < G::PAIRS; p++) {
    uint32_t kb = (lid + p * T) * 16; uint32_t grp = kb / half; uint32_t jb = kb % half;
    uint32_t top_off = (grp * m + jb) * 4; uint32_t bot_off = top_off + half * 4;
    auto u = esimd::slm_block_load<uint32_t, 16>(top_off);
    auto v = esimd::slm_block_load<uint32_t, 16>(bot_off);
    auto twv = esimd::block_load<uint32_t, 16>(ptw + off + jb);
    auto tv = bb31::mont_mul(twv, v);
    esimd::slm_block_store<uint32_t, 16>(top_off, bb31::field_add(u, tv));
    esimd::slm_block_store<uint32_t, 16>(bot_off, bb31::field_sub(u, tv));
  }
  esimd::barrier();
}
template <uint32_t T>
ESIMD_INLINE void slm_gs(uint32_t STAGE, const uint32_t* ptw, uint32_t off, uint32_t lid) {
  using G = SlmGeo<T>;
  uint32_t half = 1u << (STAGE - 1); uint32_t m = 1u << STAGE;
#pragma unroll
  for (uint32_t p = 0; p < G::PAIRS; p++) {
    uint32_t kb = (lid + p * T) * 16; uint32_t grp = kb / half; uint32_t jb = kb % half;
    uint32_t top_off = (grp * m + jb) * 4; uint32_t bot_off = top_off + half * 4;
    auto u = esimd::slm_block_load<uint32_t, 16>(top_off);
    auto v = esimd::slm_block_load<uint32_t, 16>(bot_off);
    auto sv = bb31::field_add(u, v);
    auto d2 = bb31::field_sub(u, v);
    auto twv = esimd::block_load<uint32_t, 16>(ptw + off + jb);
    auto dtw = bb31::mont_mul(twv, d2);
    esimd::slm_block_store<uint32_t, 16>(top_off, sv);
    esimd::slm_block_store<uint32_t, 16>(bot_off, dtw);
  }
  esimd::barrier();
}

template <uint32_t T, bool EXPAND>
static void ntt_ct_slm_v3(sycl::queue& q, uint32_t* d_out, const uint32_t* d_in, uint32_t lg_n,
                          const uint32_t* d_twiddles, const TwiddleTables& tw) {
  using G = SlmGeo<T>;
  uint32_t n = 1u << lg_n;
  uint32_t num_groups = n / SLM_BLOCK;
  auto* pd = d_out;
  auto* pin = d_in;
  auto* ptw = d_twiddles;
  uint32_t offs[16];
  for (int s = 1; s <= 14; s++) offs[s] = tw.offsets[s];
  uint32_t o2 = offs[2], o3 = offs[3], o4 = offs[4], o5 = offs[5], o6 = offs[6], o7 = offs[7], o8 = offs[8];
  uint32_t o9 = offs[9], o10 = offs[10], o11 = offs[11], o12 = offs[12], o13 = offs[13], o14 = offs[14];
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(num_groups * T, T),
      [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
        esimd::slm_init<SLM_BYTES>();
        uint32_t lid = item.get_local_id(0);
        uint32_t gid = item.get_group(0);
        uint32_t global_base = gid * SLM_BLOCK;
        esimd::simd<uint32_t, 16> lane32(0u, 1u);
        bb31::Vec16 dv[G::LOADS];
        if constexpr (EXPAND) {
#pragma unroll
          for (uint32_t c = 0; c < G::LOADS; c++) {
            uint32_t e0 = global_base + c * G::CHUNK + lid * 16;
            esimd::simd<uint32_t, 16> off = ((e0 >> 2) + (lane32 >> 2)) * 4u;
            dv[c] = esimd::gather<uint32_t, 16>(pin, off);
          }
          esimd::simd<uint16_t, 16> xi4(lane32 ^ esimd::simd<uint32_t, 16>(4u));
          esimd::simd<uint16_t, 16> xi8(lane32 ^ esimd::simd<uint32_t, 16>(8u));
          auto t4 = (lane32 & 4u) == esimd::simd<uint32_t, 16>(0u);
          auto t8 = (lane32 & 8u) == esimd::simd<uint32_t, 16>(0u);
          auto tw3 = esimd::block_load<uint32_t, 16>(ptw + o3);
          auto tw4 = esimd::block_load<uint32_t, 16>(ptw + o4);
#pragma unroll
          for (uint32_t c = 0; c < G::LOADS; c++) {
            auto& data = dv[c];
            { auto p = data.iselect(xi4);
              bb31::Vec16 bv = data; bv.merge(p, t4);
              bb31::Vec16 tv = p; tv.merge(data, t4);
              auto r = bb31::mont_mul(tw3, bv);
              auto s = bb31::field_add(tv, r); auto d = bb31::field_sub(tv, r);
              data = d; data.merge(s, t4); }
            { auto p = data.iselect(xi8);
              bb31::Vec16 bv = data; bv.merge(p, t8);
              bb31::Vec16 tv = p; tv.merge(data, t8);
              auto r = bb31::mont_mul(tw4, bv);
              auto s = bb31::field_add(tv, r); auto d = bb31::field_sub(tv, r);
              data = d; data.merge(s, t8); }
          }
        } else {
#pragma unroll
          for (uint32_t c = 0; c < G::LOADS; c++)
            dv[c] = esimd::block_load<uint32_t, 16>(pin + global_base + c * G::CHUNK + lid * 16);
#pragma unroll
          for (uint32_t c = 0; c < G::LOADS; c++)
            FUSED_CT_STAGES_1_4(dv[c], ptw, o2, o3, o4)
        }
#pragma unroll
        for (uint32_t c = 0; c < G::LOADS; c++)
          esimd::slm_block_store<uint32_t, 16>((c * G::CHUNK + lid * 16) * 4u, dv[c]);
        esimd::barrier();
        slm_ct<T>(5, ptw, o5, lid);
        slm_ct<T>(6, ptw, o6, lid);
        slm_ct<T>(7, ptw, o7, lid);
        slm_ct<T>(8, ptw, o8, lid);
        if constexpr (G::LGC >= 9) slm_ct<T>(9, ptw, o9, lid);
        if constexpr (G::LGC >= 10) slm_ct<T>(10, ptw, o10, lid);
        if constexpr (G::LGC >= 11) slm_ct<T>(11, ptw, o11, lid);
#pragma unroll
        for (uint32_t c = 0; c < G::LOADS; c++)
          dv[c] = esimd::slm_block_load<uint32_t, 16>((c * G::CHUNK + lid * 16) * 4u);
        if constexpr (G::LGC < 9) reg_ct<T, 9>(dv, ptw, o9, lid);
        if constexpr (G::LGC < 10) reg_ct<T, 10>(dv, ptw, o10, lid);
        if constexpr (G::LGC < 11) reg_ct<T, 11>(dv, ptw, o11, lid);
        reg_ct<T, 12>(dv, ptw, o12, lid);
        reg_ct<T, 13>(dv, ptw, o13, lid);
        reg_ct<T, 14>(dv, ptw, o14, lid);
#pragma unroll
        for (uint32_t c = 0; c < G::LOADS; c++)
          esimd::block_store(pd + global_base + c * G::CHUNK + lid * 16, dv[c]);
      });
  });
}

template <uint32_t T>
static void ntt_gs_slm_v3(sycl::queue& q, uint32_t* d_data, uint32_t lg_n,
                          const uint32_t* d_twiddles, const TwiddleTables& tw,
                          uint32_t scale_factor, const uint32_t* d_zk_powers) {
  using G = SlmGeo<T>;
  uint32_t n = 1u << lg_n;
  uint32_t num_groups = n / SLM_BLOCK;
  auto* pd = d_data;
  auto* ptw = d_twiddles;
  auto* pzk = d_zk_powers;
  uint32_t o2 = tw.offsets[2], o3 = tw.offsets[3], o4 = tw.offsets[4], o5 = tw.offsets[5], o6 = tw.offsets[6];
  uint32_t o7 = tw.offsets[7], o8 = tw.offsets[8], o9 = tw.offsets[9], o10 = tw.offsets[10], o11 = tw.offsets[11];
  uint32_t o12 = tw.offsets[12], o13 = tw.offsets[13], o14 = tw.offsets[14];
  uint32_t sf = scale_factor;
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(num_groups * T, T),
      [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
        esimd::slm_init<SLM_BYTES>();
        uint32_t lid = item.get_local_id(0);
        uint32_t gid = item.get_group(0);
        uint32_t global_base = gid * SLM_BLOCK;
        bb31::Vec16 dv[G::LOADS];
#pragma unroll
        for (uint32_t c = 0; c < G::LOADS; c++)
          dv[c] = esimd::block_load<uint32_t, 16>(pd + global_base + c * G::CHUNK + lid * 16);
        reg_gs<T, 14>(dv, ptw, o14, lid);
        reg_gs<T, 13>(dv, ptw, o13, lid);
        reg_gs<T, 12>(dv, ptw, o12, lid);
        if constexpr (G::LGC < 11) reg_gs<T, 11>(dv, ptw, o11, lid);
        if constexpr (G::LGC < 10) reg_gs<T, 10>(dv, ptw, o10, lid);
        if constexpr (G::LGC < 9) reg_gs<T, 9>(dv, ptw, o9, lid);
#pragma unroll
        for (uint32_t c = 0; c < G::LOADS; c++)
          esimd::slm_block_store<uint32_t, 16>((c * G::CHUNK + lid * 16) * 4u, dv[c]);
        esimd::barrier();
        if constexpr (G::LGC >= 11) slm_gs<T>(11, ptw, o11, lid);
        if constexpr (G::LGC >= 10) slm_gs<T>(10, ptw, o10, lid);
        if constexpr (G::LGC >= 9) slm_gs<T>(9, ptw, o9, lid);
        slm_gs<T>(8, ptw, o8, lid);
        slm_gs<T>(7, ptw, o7, lid);
        slm_gs<T>(6, ptw, o6, lid);
        slm_gs<T>(5, ptw, o5, lid);
#pragma unroll
        for (uint32_t c = 0; c < G::LOADS; c++)
          dv[c] = esimd::slm_block_load<uint32_t, 16>((c * G::CHUNK + lid * 16) * 4u);
#pragma unroll
        for (uint32_t c = 0; c < G::LOADS; c++)
          FUSED_GS_STAGES_4_1(dv[c], ptw, o2, o3, o4)
        if (sf != bb31::ONE) {
#pragma unroll
          for (uint32_t c = 0; c < G::LOADS; c++) dv[c] = bb31::mont_mul(dv[c], bb31::Vec16(sf));
        }
        if (pzk) {
#pragma unroll
          for (uint32_t c = 0; c < G::LOADS; c++) {
            auto pows = esimd::block_load<uint32_t, 16>(pzk + global_base + c * G::CHUNK + lid * 16);
            dv[c] = bb31::mont_mul(dv[c], pows);
          }
        }
#pragma unroll
        for (uint32_t c = 0; c < G::LOADS; c++)
          esimd::block_store(pd + global_base + c * G::CHUNK + lid * 16, dv[c]);
      });
  });
}

template <bool DERIVE>
static void ntt_ct_4stage_v(sycl::queue& q, uint32_t* d_data, uint32_t lg_n, uint32_t s,
                            const uint32_t* tw0, const uint32_t* tw1, const uint32_t* tw2, const uint32_t* tw3,
                            uint32_t npolys, uint32_t stride) {
  uint32_t n = 1u << lg_n;
  uint32_t hs = 1u << (s - 1), ms = 1u << s, ms1 = ms << 1, ms2 = ms << 2, ms3 = ms << 3;
  uint32_t hs1 = ms, hs2 = ms1, hs3 = ms2;
  uint32_t hexadecs_per_group = hs;
  uint32_t tpp = (n / ms3) * hexadecs_per_group / 16;  // threads per poly
  // uniform twiddle multipliers: stage s+1 root^(hs), stage s+2 root^(k*hs), stage s+3 root^(k*hs)
  uint32_t c1 = host_mont_pow(ntt::forward_roots[s + 1], hs);
  uint32_t c2[4], c3[8];
  for (int k = 0; k < 4; k++) c2[k] = host_mont_pow(ntt::forward_roots[s + 2], k * hs);
  for (int k = 0; k < 8; k++) c3[k] = host_mont_pow(ntt::forward_roots[s + 3], k * hs);
  uint32_t c2_1 = c2[1], c2_2 = c2[2], c2_3 = c2[3];
  uint32_t c3_1 = c3[1], c3_2 = c3[2], c3_3 = c3[3], c3_4 = c3[4], c3_5 = c3[5], c3_6 = c3[6], c3_7 = c3[7];
  q.parallel_for(sycl::range<1>(tpp * npolys), [=](sycl::id<1> idx) [[intel::sycl_explicit_simd]] {
    uint32_t t = idx[0];
    uint32_t poly = t / tpp;
    uint32_t tid = t % tpp;
    uint32_t* pd = d_data + (size_t)poly * stride;
    uint32_t k_base = tid * 16;
    uint32_t grp = k_base / hexadecs_per_group;
    uint32_t j = k_base % hexadecs_per_group;
    uint32_t base = grp * ms3 + j;
    auto v0000 = esimd::block_load<uint32_t, 16>(pd + base, LOAD_STREAMING);
    auto v0001 = esimd::block_load<uint32_t, 16>(pd + base + hs, LOAD_STREAMING);
    auto v0010 = esimd::block_load<uint32_t, 16>(pd + base + hs1, LOAD_STREAMING);
    auto v0011 = esimd::block_load<uint32_t, 16>(pd + base + hs1 + hs, LOAD_STREAMING);
    auto v0100 = esimd::block_load<uint32_t, 16>(pd + base + hs2, LOAD_STREAMING);
    auto v0101 = esimd::block_load<uint32_t, 16>(pd + base + hs2 + hs, LOAD_STREAMING);
    auto v0110 = esimd::block_load<uint32_t, 16>(pd + base + hs2 + hs1, LOAD_STREAMING);
    auto v0111 = esimd::block_load<uint32_t, 16>(pd + base + hs2 + hs1 + hs, LOAD_STREAMING);
    auto v1000 = esimd::block_load<uint32_t, 16>(pd + base + hs3, LOAD_STREAMING);
    auto v1001 = esimd::block_load<uint32_t, 16>(pd + base + hs3 + hs, LOAD_STREAMING);
    auto v1010 = esimd::block_load<uint32_t, 16>(pd + base + hs3 + hs1, LOAD_STREAMING);
    auto v1011 = esimd::block_load<uint32_t, 16>(pd + base + hs3 + hs1 + hs, LOAD_STREAMING);
    auto v1100 = esimd::block_load<uint32_t, 16>(pd + base + hs3 + hs2, LOAD_STREAMING);
    auto v1101 = esimd::block_load<uint32_t, 16>(pd + base + hs3 + hs2 + hs, LOAD_STREAMING);
    auto v1110 = esimd::block_load<uint32_t, 16>(pd + base + hs3 + hs2 + hs1, LOAD_STREAMING);
    auto v1111 = esimd::block_load<uint32_t, 16>(pd + base + hs3 + hs2 + hs1 + hs, LOAD_STREAMING);
#define BFY(TOP, BOT, TW) { auto tt = bb31::mont_mul(TW, BOT); auto tmp = TOP; TOP = bb31::field_add(tmp, tt); BOT = bb31::field_sub(tmp, tt); }
    auto t0 = esimd::block_load<uint32_t, 16>(tw0 + j);
    BFY(v0000, v0001, t0) BFY(v0010, v0011, t0) BFY(v0100, v0101, t0) BFY(v0110, v0111, t0)
    BFY(v1000, v1001, t0) BFY(v1010, v1011, t0) BFY(v1100, v1101, t0) BFY(v1110, v1111, t0)
    bb31::Vec16 t1_lo = esimd::block_load<uint32_t, 16>(tw1 + j), t1_hi;
    if constexpr (DERIVE) t1_hi = bb31::mont_mul(t1_lo, bb31::Vec16(c1));
    else t1_hi = esimd::block_load<uint32_t, 16>(tw1 + j + hs);
    BFY(v0000, v0010, t1_lo) BFY(v0001, v0011, t1_hi) BFY(v0100, v0110, t1_lo) BFY(v0101, v0111, t1_hi)
    BFY(v1000, v1010, t1_lo) BFY(v1001, v1011, t1_hi) BFY(v1100, v1110, t1_lo) BFY(v1101, v1111, t1_hi)
    bb31::Vec16 t2_00 = esimd::block_load<uint32_t, 16>(tw2 + j), t2_01, t2_10, t2_11;
    if constexpr (DERIVE) {
      t2_01 = bb31::mont_mul(t2_00, bb31::Vec16(c2_1)); t2_10 = bb31::mont_mul(t2_00, bb31::Vec16(c2_2)); t2_11 = bb31::mont_mul(t2_00, bb31::Vec16(c2_3));
    } else {
      t2_01 = esimd::block_load<uint32_t, 16>(tw2 + j + hs); t2_10 = esimd::block_load<uint32_t, 16>(tw2 + j + hs1);
      t2_11 = esimd::block_load<uint32_t, 16>(tw2 + j + hs1 + hs);
    }
    BFY(v0000, v0100, t2_00) BFY(v0001, v0101, t2_01) BFY(v0010, v0110, t2_10) BFY(v0011, v0111, t2_11)
    BFY(v1000, v1100, t2_00) BFY(v1001, v1101, t2_01) BFY(v1010, v1110, t2_10) BFY(v1011, v1111, t2_11)
    bb31::Vec16 t3_0 = esimd::block_load<uint32_t, 16>(tw3 + j), t3_1, t3_2, t3_3, t3_4, t3_5, t3_6, t3_7;
    if constexpr (DERIVE) {
      t3_1 = bb31::mont_mul(t3_0, bb31::Vec16(c3_1)); t3_2 = bb31::mont_mul(t3_0, bb31::Vec16(c3_2));
      t3_3 = bb31::mont_mul(t3_0, bb31::Vec16(c3_3)); t3_4 = bb31::mont_mul(t3_0, bb31::Vec16(c3_4));
      t3_5 = bb31::mont_mul(t3_0, bb31::Vec16(c3_5)); t3_6 = bb31::mont_mul(t3_0, bb31::Vec16(c3_6));
      t3_7 = bb31::mont_mul(t3_0, bb31::Vec16(c3_7));
    } else {
      t3_1 = esimd::block_load<uint32_t, 16>(tw3 + j + hs); t3_2 = esimd::block_load<uint32_t, 16>(tw3 + j + hs1);
      t3_3 = esimd::block_load<uint32_t, 16>(tw3 + j + hs1 + hs); t3_4 = esimd::block_load<uint32_t, 16>(tw3 + j + hs2);
      t3_5 = esimd::block_load<uint32_t, 16>(tw3 + j + hs2 + hs); t3_6 = esimd::block_load<uint32_t, 16>(tw3 + j + hs2 + hs1);
      t3_7 = esimd::block_load<uint32_t, 16>(tw3 + j + hs2 + hs1 + hs);
    }
    BFY(v0000, v1000, t3_0) BFY(v0001, v1001, t3_1) BFY(v0010, v1010, t3_2) BFY(v0011, v1011, t3_3)
    BFY(v0100, v1100, t3_4) BFY(v0101, v1101, t3_5) BFY(v0110, v1110, t3_6) BFY(v0111, v1111, t3_7)
#undef BFY
    esimd::block_store(pd + base, v0000, STORE_STREAMING);
    esimd::block_store(pd + base + hs, v0001, STORE_STREAMING);
    esimd::block_store(pd + base + hs1, v0010, STORE_STREAMING);
    esimd::block_store(pd + base + hs1 + hs, v0011, STORE_STREAMING);
    esimd::block_store(pd + base + hs2, v0100, STORE_STREAMING);
    esimd::block_store(pd + base + hs2 + hs, v0101, STORE_STREAMING);
    esimd::block_store(pd + base + hs2 + hs1, v0110, STORE_STREAMING);
    esimd::block_store(pd + base + hs2 + hs1 + hs, v0111, STORE_STREAMING);
    esimd::block_store(pd + base + hs3, v1000, STORE_STREAMING);
    esimd::block_store(pd + base + hs3 + hs, v1001, STORE_STREAMING);
    esimd::block_store(pd + base + hs3 + hs1, v1010, STORE_STREAMING);
    esimd::block_store(pd + base + hs3 + hs1 + hs, v1011, STORE_STREAMING);
    esimd::block_store(pd + base + hs3 + hs2, v1100, STORE_STREAMING);
    esimd::block_store(pd + base + hs3 + hs2 + hs, v1101, STORE_STREAMING);
    esimd::block_store(pd + base + hs3 + hs2 + hs1, v1110, STORE_STREAMING);
    esimd::block_store(pd + base + hs3 + hs2 + hs1 + hs, v1111, STORE_STREAMING);
  });
}



// Single-pass "tail" kernels for the stages above the 2^14 SLM block.
// Positions p = j + 2^14*k, j in [0,2^14) (16-lane column blocks), k in [0, 2^(2H)).
// Stage s = 15+b acts on bit b of k. One WG (2^H threads, SIMD16) owns one 16-wide
// column block and all 2^(2H) k-values (2^(2H)*16 elements, 2^(2H)*64 B of SLM):
//   phase 1: thread t holds R = 2^H vectors over one half of the k bits (in registers),
//   SLM transpose, phase 2: the other half.
// Twiddle tw_s[j + 2^14*m] == tw_s[j] * w_s^(2^14*m) exactly (canonical field values),
// so each stage loads one twiddle vector and multiplies by uniform constants:
//   w_s^(2^14*m) for m < 2^H                  -> table T1 (index 2^i-1+m)
//   m = c'*2^H + t:  A[c'] * B[t], A = w_s^(2^14*2^H*c'), B = w_s^(2^14*t)
// Bit-identical to the per-stage kernels.

struct TailTables { uint32_t* d; uint32_t lg_n; bool fwd; uint32_t H; };
// layout: [0,16) T1 ; [16,32) A ; [32, 32+16*H) B rows (row i' = stage 15+H+i')
static TailTables make_tail_tables(sycl::queue& q, uint32_t H, bool fwd) {
  std::vector<uint32_t> h(32 + 16 * H, bb31::ONE);
  const uint32_t* roots = fwd ? ntt::forward_roots : ntt::inverse_roots;
  for (uint32_t i = 0; i < H; i++)
    for (uint32_t m = 0; m < (1u << i); m++)
      h[(1u << i) - 1 + m] = host_mont_pow(roots[15 + i], (1u << 14) * m);
  for (uint32_t i = 0; i < H; i++) {
    uint32_t s = 15 + H + i;
    for (uint32_t c = 0; c < (1u << i); c++)
      h[16 + (1u << i) - 1 + c] = host_mont_pow(roots[s], (1u << 14) * (1u << H) * c);
    for (uint32_t t = 0; t < (1u << H); t++)
      h[32 + 16 * i + t] = host_mont_pow(roots[s], (1u << 14) * t);
  }
  TailTables tt{sycl::malloc_device<uint32_t>(h.size(), q), 0, fwd, H};
  q.memcpy(tt.d, h.data(), h.size() * 4).wait();
  return tt;
}

template <uint32_t H>
static void ntt_ct_tail(sycl::queue& q, uint32_t* d_data, uint32_t lg_n, const TwiddleTables& tw, const uint32_t* d_tab) {
  static_assert(H >= 1 && H <= 4, "H");
  constexpr uint32_t R = 1u << H, TH = R, K = 2 * H;
  uint32_t n = 1u << lg_n;  // must equal 2^(14+K)
  uint32_t num_wg = n >> (K + 4);  // one WG per 16-wide column block = 2^14/16
  uint32_t offs[8] = {};  // only offs[0..K) are used
  for (uint32_t b = 0; b < K; b++) offs[b] = tw.offsets[15 + b];
  uint32_t o0 = offs[0], o1 = offs[1], o2 = offs[2], o3 = offs[3], o4 = offs[4], o5 = offs[5], o6 = offs[6], o7 = offs[7];
  const uint32_t* ptw = tw.d_buffer;
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(num_wg * TH, TH), [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
      esimd::slm_init<R * R * 64>();
      const uint32_t of[8] = {o0, o1, o2, o3, o4, o5, o6, o7};
      uint32_t t = item.get_local_id(0);
      uint32_t jbase = item.get_group(0) * 16;
      bb31::Vec16 T1 = esimd::block_load<uint32_t, 16>(d_tab);
      bb31::Vec16 TA = esimd::block_load<uint32_t, 16>(d_tab + 16);
      bb31::Vec16 dv[R];
      // phase 1: k = t*R + c
#pragma unroll
      for (uint32_t c = 0; c < R; c++)
        dv[c] = esimd::block_load<uint32_t, 16>(d_data + jbase + ((size_t)(t * R + c) << 14));
#pragma unroll
      for (uint32_t i = 0; i < H; i++) {
        bb31::Vec16 base = esimd::block_load<uint32_t, 16>(ptw + of[i] + jbase);
#pragma unroll
        for (uint32_t ct = 0; ct < R; ct++) {
          if (ct & (1u << i)) continue;
          uint32_t cb = ct | (1u << i);
          uint32_t m = ct & ((1u << i) - 1);
          bb31::Vec16 twv = (m == 0) ? base : bb31::mont_mul(base, bb31::Vec16(T1[(1u << i) - 1 + m]));
          auto tv = bb31::mont_mul(twv, dv[cb]);
          auto u = dv[ct];
          dv[ct] = bb31::field_add(u, tv);
          dv[cb] = bb31::field_sub(u, tv);
        }
      }
#pragma unroll
      for (uint32_t c = 0; c < R; c++) esimd::slm_block_store<uint32_t, 16>((t * R + c) * 64u, dv[c]);
      esimd::barrier();
      // phase 2: k = c*R + t
#pragma unroll
      for (uint32_t c = 0; c < R; c++) dv[c] = esimd::slm_block_load<uint32_t, 16>((c * R + t) * 64u);
#pragma unroll
      for (uint32_t i = 0; i < H; i++) {
        bb31::Vec16 Brow = esimd::block_load<uint32_t, 16>(d_tab + 32 + 16 * i);
        uint32_t Bt = Brow[t];
        bb31::Vec16 base = esimd::block_load<uint32_t, 16>(ptw + of[H + i] + jbase);
        base = bb31::mont_mul(base, bb31::Vec16(Bt));
#pragma unroll
        for (uint32_t ct = 0; ct < R; ct++) {
          if (ct & (1u << i)) continue;
          uint32_t cb = ct | (1u << i);
          uint32_t m = ct & ((1u << i) - 1);
          bb31::Vec16 twv = (m == 0) ? base : bb31::mont_mul(base, bb31::Vec16(TA[(1u << i) - 1 + m]));
          auto tv = bb31::mont_mul(twv, dv[cb]);
          auto u = dv[ct];
          dv[ct] = bb31::field_add(u, tv);
          dv[cb] = bb31::field_sub(u, tv);
        }
      }
#pragma unroll
      for (uint32_t c = 0; c < R; c++)
        esimd::block_store(d_data + jbase + ((size_t)(c * R + t) << 14), dv[c]);
    });
  });
}

template <uint32_t H>
static void ntt_gs_tail(sycl::queue& q, uint32_t* d_data, uint32_t lg_n, const TwiddleTables& tw, const uint32_t* d_tab) {
  constexpr uint32_t R = 1u << H, TH = R, K = 2 * H;
  uint32_t n = 1u << lg_n;
  uint32_t num_wg = n >> (K + 4);
  uint32_t offs[8] = {0};
  for (uint32_t b = 0; b < K; b++) offs[b] = tw.offsets[15 + b];
  uint32_t o0 = offs[0], o1 = offs[1], o2 = offs[2], o3 = offs[3], o4 = offs[4], o5 = offs[5], o6 = offs[6], o7 = offs[7];
  const uint32_t* ptw = tw.d_buffer;
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(num_wg * TH, TH), [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
      esimd::slm_init<R * R * 64>();
      const uint32_t of[8] = {o0, o1, o2, o3, o4, o5, o6, o7};
      uint32_t t = item.get_local_id(0);
      uint32_t jbase = item.get_group(0) * 16;
      bb31::Vec16 T1 = esimd::block_load<uint32_t, 16>(d_tab);
      bb31::Vec16 TA = esimd::block_load<uint32_t, 16>(d_tab + 16);
      bb31::Vec16 dv[R];
      // phase 1 (high k bits, descending stages): k = c*R + t
#pragma unroll
      for (uint32_t c = 0; c < R; c++)
        dv[c] = esimd::block_load<uint32_t, 16>(d_data + jbase + ((size_t)(c * R + t) << 14));
#pragma unroll
      for (int ii = H - 1; ii >= 0; ii--) {
        uint32_t i = ii;
        bb31::Vec16 Brow = esimd::block_load<uint32_t, 16>(d_tab + 32 + 16 * i);
        uint32_t Bt = Brow[t];
        bb31::Vec16 base = esimd::block_load<uint32_t, 16>(ptw + of[H + i] + jbase);
        base = bb31::mont_mul(base, bb31::Vec16(Bt));
#pragma unroll
        for (uint32_t ct = 0; ct < R; ct++) {
          if (ct & (1u << i)) continue;
          uint32_t cb = ct | (1u << i);
          uint32_t m = ct & ((1u << i) - 1);
          bb31::Vec16 twv = (m == 0) ? base : bb31::mont_mul(base, bb31::Vec16(TA[(1u << i) - 1 + m]));
          auto u = dv[ct], v = dv[cb];
          dv[ct] = bb31::field_add(u, v);
          dv[cb] = bb31::mont_mul(twv, bb31::field_sub(u, v));
        }
      }
#pragma unroll
      for (uint32_t c = 0; c < R; c++) esimd::slm_block_store<uint32_t, 16>((c * R + t) * 64u, dv[c]);
      esimd::barrier();
      // phase 2 (low k bits): k = t*R + c
#pragma unroll
      for (uint32_t c = 0; c < R; c++) dv[c] = esimd::slm_block_load<uint32_t, 16>((t * R + c) * 64u);
#pragma unroll
      for (int ii = H - 1; ii >= 0; ii--) {
        uint32_t i = ii;
        bb31::Vec16 base = esimd::block_load<uint32_t, 16>(ptw + of[i] + jbase);
#pragma unroll
        for (uint32_t ct = 0; ct < R; ct++) {
          if (ct & (1u << i)) continue;
          uint32_t cb = ct | (1u << i);
          uint32_t m = ct & ((1u << i) - 1);
          bb31::Vec16 twv = (m == 0) ? base : bb31::mont_mul(base, bb31::Vec16(T1[(1u << i) - 1 + m]));
          auto u = dv[ct], v = dv[cb];
          dv[ct] = bb31::field_add(u, v);
          dv[cb] = bb31::mont_mul(twv, bb31::field_sub(u, v));
        }
      }
#pragma unroll
      for (uint32_t c = 0; c < R; c++)
        esimd::block_store(d_data + jbase + ((size_t)(t * R + c) << 14), dv[c]);
    });
  });
}

// ---- commit-path NTT entry points ----
static bool g_ntt_v3_on = !std::getenv("RISC0_NTT_V3_OFF");
static bool g_ntt_tail_on = g_ntt_v3_on && !std::getenv("RISC0_NTT_TAIL_OFF");

static const uint32_t* get_tail_table(sycl::queue& q, uint32_t H, bool fwd) {
    static uint32_t* tabs[2][5] = {};
    uint32_t*& t = tabs[fwd ? 1 : 0][H];
    if (!t) t = make_tail_tables(q, H, fwd).d;
    return t;
}
// Run all stages above the 2^14 SLM block in ONE pass when (lg_n-14) is even, <= 8.
static bool ct_tail_single(sycl::queue& q, uint32_t* d, uint32_t lg_n, const TwiddleTables& tw) {
    if (!g_ntt_tail_on) return false;
    switch (lg_n) {
        case 20: ntt_ct_tail<3>(q, d, lg_n, tw, get_tail_table(q, 3, true)); return true;
        case 22: ntt_ct_tail<4>(q, d, lg_n, tw, get_tail_table(q, 4, true)); return true;
        default: return false;
    }
}
static bool gs_tail_single(sycl::queue& q, uint32_t* d, uint32_t lg_n, const TwiddleTables& tw) {
    if (!g_ntt_tail_on) return false;
    switch (lg_n) {
        case 20: ntt_gs_tail<3>(q, d, lg_n, tw, get_tail_table(q, 3, false)); return true;
        case 22: ntt_gs_tail<4>(q, d, lg_n, tw, get_tail_table(q, 4, false)); return true;
        default: return false;
    }
}

static void ct_tail_passes(sycl::queue& q, uint32_t* d_data, uint32_t lg_n, const TwiddleTables& tw) {
    if (ct_tail_single(q, d_data, lg_n, tw)) return;
    uint32_t s = SLM_LG_BLOCK + 1;
    uint32_t remaining = lg_n - s + 1;
    for (; remaining >= 4 && (1u << (s - 1)) >= 16; s += 4, remaining = lg_n - s + 1) {
        const uint32_t *t0 = tw.d_buffer+tw.offsets[s], *t1 = tw.d_buffer+tw.offsets[s+1], *t2 = tw.d_buffer+tw.offsets[s+2], *t3 = tw.d_buffer+tw.offsets[s+3];
        // Derive 11 of 15 twiddle vectors from uniform constants when the stage
        // tables are large (s+3 >= 21: >= 4 MB, poor cache reuse); bit-identical.
        if (g_ntt_v3_on && s + 3 >= 21) ntt_ct_4stage_v<true>(q, d_data, lg_n, s, t0, t1, t2, t3, 1, 0);
        else ntt_ct_fused_4stage(q, d_data, lg_n, s, t0, t1, t2, t3);
    }
    for (; remaining >= 3 && (1u << (s - 1)) >= 64; s += 3, remaining = lg_n - s + 1)
        ntt_ct_fused_3stage(q, d_data, lg_n, s, tw.d_buffer+tw.offsets[s], tw.d_buffer+tw.offsets[s+1], tw.d_buffer+tw.offsets[s+2]);
    for (; s + 1 <= lg_n; s += 2)
        ntt_ct_fused_2stage(q, d_data, lg_n, s, tw.d_buffer+tw.offsets[s], tw.d_buffer+tw.offsets[s+1]);
    if (s <= lg_n) ntt_ct_stage_fast(q, d_data, lg_n, s, ntt::forward_roots, tw.d_buffer+tw.offsets[s]);
}

} // extern "C++"
