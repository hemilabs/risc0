// Four-step fused commit NTT ("nttq") for lg_domain_size 16..21, lg_blowup 2 (CUDA only).
//   per column: W (evaluations) --colpass--> T1 --rowpass (* n^-1 * g^k)--> C (natural coefficients)
//               C (rows < Lc/4 of the [Lc][Lr] LDE matrix) --coset colpass--> T2 --rowpass--> E
// Each pass computes batched line DFTs of length L = 2^LOGL (256..2048) as 16 x 16 x P3 stages in
// registers with two shared-memory exchanges, TB lines per block (512-thread blocks where possible).
// The forward column pass splits the zero-padded column DFT of length Lc into its 4 cosets
// k = 4a + b (blockIdx.z = b): DFT_{Lc/4}(x_m * w_Lc^(m b)), so it reads only the nonzero rows.
// Four-step twiddles and zk-shift factors are chained per thread (one table gather per thread).
// Scratch T1/T2 lines are discarded from L2 after their last read (no write-back); the final LDE
// is written with streaming stores. All arithmetic is sppark's canonical Montgomery bb31, so the
// output equals the radix-2 path bit for bit.
// The forward passes run CHF columns at a time, alternating between NS side streams with their own
// T2 scratch (L2-resident); the side streams are joined back into the gpu stream before returning.
#pragma once
#include <mutex>
#include "../device_alloc.h"

namespace nttq {

__host__ __device__ constexpr int brev_c(int i, int bits) {
    int r = 0;
    for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    return r;
}
template<int LOGP, int LG>
__device__ __forceinline__ void dit_level(fr_t* w, const fr_t* rootsP)
{
    if constexpr (LG <= LOGP) {
        constexpr int P = 1 << LOGP, len = 1 << LG, half = len >> 1;
        #pragma unroll
        for (int s = 0; s < P; s += len) {
            #pragma unroll
            for (int k = 0; k < half; k++) {
                fr_t u = w[s + k], t = w[s + k + half];
                if (k != 0) t = t * rootsP[k << (LOGP - LG)];
                w[s + k] = u + t;
                w[s + k + half] = u - t;
            }
        }
        dit_level<LOGP, LG + 1>(w, rootsP);
    }
}
// in-register DFT_P, natural in / natural out (radix-2 DIT); rootsP[e] = w_P^e, e < P/2
template<int LOGP>
__device__ __forceinline__ void reg_dft(fr_t* v, const fr_t* rootsP)
{
    constexpr int P = 1 << LOGP;
    fr_t w[P];
    #pragma unroll
    for (int i = 0; i < P; i++) w[i] = v[brev_c(i, LOGP)];
    dit_level<LOGP, 1>(w, rootsP);
    #pragma unroll
    for (int i = 0; i < P; i++) v[i] = w[i];
}

struct Tw {             // w_N^e = A[e & 4095] * B[e >> 12]
    const fr_t* A;
    const fr_t* B;
    unsigned lgN;
};

// Per-(line size, direction) tables:
//   r16[e] = w_16^e (e < 8), r16[16 + e] = w_16^e (e < 10); rP3[e] = w_P3^e (e < P3/2)
//   t2[m*16 + k] = w_256^(k*m) (m, k < 16); t3[m*256 + k] = w_L^(k*m) (m < P3, k < 256)
struct LineTabs { const fr_t* r16; const fr_t* rP3; const fr_t* t2; const fr_t* t3; };

template<int LOGL, int TB> struct Smem {
    static constexpr int L = 1 << LOGL;
    static constexpr int LS = L + L / 32 + (32 / TB > 0 ? 32 / TB : 1);  // padded line stride
    static constexpr int TW = 256 + 256 * (1 << (LOGL - 8));            // t2 + t3 staged after the lines
    __device__ static __forceinline__ int a(int i) { return i + (i >> 5); }
};

__device__ __forceinline__ fr_t twv(const Tw& tw, unsigned e) {
    e &= (1u << tw.lgN) - 1;
    return tw.A[e & 4095] * tw.B[e >> 12];
}

enum : unsigned { F_DISCARD_IN = 1, F_CS_OUT = 2 };
#ifndef __HIPCC__
__device__ __forceinline__ void st_cs(fr_t* p, fr_t x) { asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(p), "r"(*(uint32_t*)&x) : "memory"); }
// the 128-byte line's contents become undefined; it is dropped from L2 without a write-back.
// `discard` needs sm_80: on Turing the line is left for L2 to evict, as any other.
__device__ __forceinline__ void discard_l2(const void* p) {
#if __CUDA_ARCH__ >= 800
    asm volatile("discard.global.L2 [%0], 128;" :: "l"(p) : "memory");
#else
    (void)p;
#endif
}
#else
__device__ __forceinline__ void st_cs(fr_t* p, fr_t x) { *p = x; }
__device__ __forceinline__ void discard_l2(const void*) {}
#endif

template<int LOGL, int TB, int MODE, int MINB>
__global__ void __launch_bounds__(TB * (1 << LOGL) / 16, MINB)
pass_k(const fr_t* __restrict__ in, fr_t* __restrict__ out, size_t ics, size_t ocs,
       unsigned lgX, unsigned rows_in, LineTabs T, Tw tw,
       const fr_t* __restrict__ lo, const fr_t* __restrict__ hi, fr_t scale, unsigned flags)
{
    constexpr int L = 1 << LOGL, LOGP3 = LOGL - 8, P3 = 1 << LOGP3;
    static_assert(LOGP3 >= 0 && LOGP3 <= 4, "LOGL 8..12");
    using SM = Smem<LOGL, TB>;
    constexpr int NT = TB * L / 16;
    constexpr int LP1 = L / 16;
    extern __shared__ fr_t smem[];
    in += blockIdx.y * ics;
    out += blockIdx.y * ocs;
    const size_t base0 = (size_t)blockIdx.x * TB;
    const int tid = threadIdx.x;
    const unsigned bz = MODE == 2 ? blockIdx.z : 0;

    fr_t r16[8];
    #pragma unroll
    for (int e = 0; e < 8; e++) r16[e] = T.r16[e];
    fr_t* st2 = smem + TB * SM::LS;
    fr_t* st3 = st2 + 256;
    for (int i = tid; i < 256; i += NT) st2[i] = T.t2[i];
    for (int i = tid; i < 256 * P3; i += NT) st3[i] = T.t3[i];
    const fr_t* T2p = st2; const fr_t* T3p = st3;

    {
        int c, j;
        if (MODE != 1) { c = tid % TB; j = tid / TB; }
        else           { c = tid / LP1; j = tid % LP1; }
        fr_t v[16];
        if (MODE == 0) {
            const size_t S = (size_t)1 << lgX;
            const fr_t* src = in + base0 + c + (size_t)j * S;
            if (rows_in == L / 4) {
                #pragma unroll
                for (int m = 0; m < 4; m++) v[m] = src[(size_t)m * LP1 * S];
                fr_t x1 = v[1], x2 = v[2], x3 = v[3], x0 = v[0];
                fr_t r4[2] = {r16[0], r16[4]};
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    fr_t y[4];
                    y[0] = x0;
                    y[1] = b == 0 ? x1 : x1 * T.r16[16 + 1 * b];
                    y[2] = b == 0 ? x2 : x2 * T.r16[16 + 2 * b];
                    y[3] = b == 0 ? x3 : x3 * T.r16[16 + 3 * b];
                    reg_dft<2>(y, r4);
                    #pragma unroll
                    for (int a = 0; a < 4; a++) v[4 * a + b] = y[a];
                }
            } else {
                #pragma unroll
                #pragma unroll
                for (int m = 0; m < 16; m++) v[m] = src[(size_t)m * LP1 * S];
                reg_dft<4>(v, r16);
            }
        } else if (MODE == 2) {
            const size_t S = (size_t)1 << lgX;
            const fr_t* src = in + base0 + c + (size_t)j * S;
            #pragma unroll
            for (int m = 0; m < 16; m++) v[m] = src[(size_t)m * LP1 * S];
            if (bz != 0) {
                // twist x_row *= w_Lc^(row*b) = w_N^(row*b*S), row = j + m*LP1
                fr_t t = twv(tw, (unsigned)(j * bz) << lgX);
                fr_t st = twv(tw, (unsigned)(LP1 * bz) << lgX);
                #pragma unroll
                for (int m = 0; m < 16; m++) {
                    v[m] = v[m] * t;
                    if (m + 1 < 16) t = t * st;
                }
            }
            reg_dft<4>(v, r16);
        } else {
            const fr_t* src = in + (base0 + c) * L;
            #pragma unroll
            for (int m = 0; m < 16; m++) v[m] = src[j + m * LP1];
            reg_dft<4>(v, r16);
        }
        fr_t* sl = smem + c * SM::LS;
        #pragma unroll
        for (int m = 0; m < 16; m++) sl[SM::a(j * 16 + m)] = v[m];
    }
    __syncthreads();
    if (MODE == 1 && (flags & F_DISCARD_IN)) {
        // the block's input rows [base0, base0 + TB) * L are contiguous and dead now
        constexpr int NL = TB * L / 32;               // 128-byte lines
        const char* b = (const char*)(in + base0 * L);
        for (int i = tid; i < NL; i += NT) discard_l2(b + (size_t)i * 128);
    }
    {
        int c = tid / LP1, j = tid % LP1, k = j & 15;
        fr_t* sl = smem + c * SM::LS;
        fr_t v[16];
        #pragma unroll
        for (int m = 0; m < 16; m++) v[m] = sl[SM::a(j + m * LP1)];
        #pragma unroll
        for (int m = 1; m < 16; m++) v[m] = v[m] * T2p[m * 16 + k];
        reg_dft<4>(v, r16);
        __syncthreads();
        int b = (j >> 4) * 256 + k;
        #pragma unroll
        for (int m = 0; m < 16; m++) sl[SM::a(b + m * 16)] = v[m];
    }
    __syncthreads();
    {
        fr_t rP[P3 / 2 > 0 ? P3 / 2 : 1];
        #pragma unroll
        for (int e = 0; e < P3 / 2; e++) rP[e] = T.rP3[e];
        constexpr int ITEMS = TB * 256, IPT = ITEMS / NT;
        constexpr unsigned KM = MODE == 2 ? 4 : 1;
        fr_t step, tI, sI, gI, gsI, gs;
        {
            const int c0 = tid % TB, j0 = tid / TB;
            constexpr unsigned JS = NT / TB;                 // j advance per item
            if (MODE != 1) {
                unsigned col = (unsigned)(base0 + c0);
                step = twv(tw, col * (256u * KM));
                tI = twv(tw, col * ((unsigned)j0 * KM + bz));
                sI = twv(tw, col * (JS * KM));
            } else if (lo) {
                size_t k0 = base0 + c0 + ((size_t)j0 << lgX);
                gI = scale * (lo[k0 & 1023] * hi[k0 >> 10]);
                size_t ki = (size_t)JS << lgX, ks = (size_t)256 << lgX;
                gsI = lo[ki & 1023] * hi[ki >> 10];
                gs = lo[ks & 1023] * hi[ks >> 10];
            }
        }
        #pragma unroll
        for (int it = 0; it < IPT; it++) {
            int item = tid + it * NT;
            int c = item % TB, j = item / TB;
            const fr_t* sl = smem + c * SM::LS;
            fr_t v[P3];
            #pragma unroll
            for (int m = 0; m < P3; m++) v[m] = sl[SM::a(j + m * 256)];
            #pragma unroll
            for (int m = 1; m < P3; m++) v[m] = v[m] * T3p[m * 256 + j];
            reg_dft<LOGP3>(v, rP);
            fr_t t = tI, g = gI;
            #pragma unroll
            for (int m = 0; m < P3; m++) {
                unsigned k = (j + m * 256) * KM + bz;
                fr_t x = v[m];
                if (MODE != 1) {
                    x = x * t;
                    if (m + 1 < P3) t = t * step;
                    out[((size_t)k << lgX) + base0 + c] = x;
                } else {
                    size_t kk = base0 + c + ((size_t)k << lgX);
                    if (lo) { x = x * g; if (m + 1 < P3) g = g * gs; }
                    if (flags & F_CS_OUT) st_cs(out + kk, x); else out[kk] = x;
                }
            }
            if (it + 1 < IPT) {
                if (MODE != 1) tI = tI * sI;
                else if (lo) gI = gI * gsI;
            }
        }
    }
}

__global__ void k_pow_table(fr_t* out, fr_t base, unsigned count, unsigned stride)
{
    unsigned e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < count) out[e] = base ^ (e * stride);
}
__global__ void k_line_tabs(fr_t* r16, fr_t* rP3, fr_t* t2, fr_t* t3, fr_t wN, int lgN, int LOGL)
{
    fr_t wL = wN ^ (1u << (lgN - LOGL));
    int L = 1 << LOGL, P3 = 1 << (LOGL - 8);
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < 8) r16[t] = wL ^ (unsigned)(t * (L / 16));
    if (t >= 16 && t < 26) r16[t] = wL ^ (unsigned)((t - 16) * (L / 16));
    if (t < P3 / 2) rP3[t] = wL ^ (unsigned)(t * (L / P3));
    if (t < 256) { int m = t / 16, k = t % 16; t2[t] = wL ^ (unsigned)((k * m) * (L / 256)); }
    if (t < 256 * P3) { int m = t / 256, k = t % 256; t3[t] = wL ^ (unsigned)(k * m); }
}


static LineTabs make_line(const gpu_t& gpu, fr_t w, unsigned lgN, unsigned LOGL)
{
    fr_t *r16, *rP3, *t2, *t3;
    int P3 = 1 << (LOGL - 8);
    CUDA_OK(risc0_cuda_malloc(&r16, 32 * sizeof(fr_t)));
    CUDA_OK(risc0_cuda_malloc(&rP3, 8 * sizeof(fr_t)));
    CUDA_OK(risc0_cuda_malloc(&t2, 256 * sizeof(fr_t)));
    CUDA_OK(risc0_cuda_malloc(&t3, 256 * P3 * sizeof(fr_t)));
    k_line_tabs<<<(256 * P3 + 255) / 256, 256, 0, gpu>>>(r16, rP3, t2, t3, w, (int)lgN, (int)LOGL);
    CUDA_OK(cudaGetLastError());
    return LineTabs{r16, rP3, t2, t3};
}

template<int LOGL, int TB, int MODE, int MINB>
static void launch(cudaStream_t st, unsigned cols, const fr_t* in, fr_t* out, size_t ics, size_t ocs,
                   unsigned lgX, unsigned rows_in, const LineTabs& LT, Tw tw,
                   const fr_t* lo, const fr_t* hi, fr_t scale, unsigned flags)
{
    constexpr int L = 1 << LOGL;
    const size_t sh = ((size_t)TB * Smem<LOGL, TB>::LS + Smem<LOGL, TB>::TW) * sizeof(fr_t);
    // per device: the attribute is device state
    static std::once_flag once[16];
    int dev;
    CUDA_OK(cudaGetDevice(&dev));
    std::call_once(once[dev], [&] {
        CUDA_OK(cudaFuncSetAttribute(pass_k<LOGL, TB, MODE, MINB>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    });
    dim3 grid((unsigned)(((size_t)1 << lgX) / TB), cols, MODE == 2 ? 4 : 1);
    pass_k<LOGL, TB, MODE, MINB><<<grid, TB * L / 16, sh, st>>>(in, out, ics, ocs, lgX, rows_in, LT, tw,
                                                                lo, hi, scale, flags);
    CUDA_OK(cudaGetLastError());
}

template<int MODE>
static void pass(cudaStream_t st, unsigned LOGL, unsigned TB, unsigned cols, const fr_t* in, fr_t* out,
                 size_t ics, size_t ocs, unsigned lgX, unsigned rows_in, const LineTabs& LT, Tw tw,
                 const fr_t* lo = nullptr, const fr_t* hi = nullptr, fr_t scale = fr_t(1), unsigned flags = 0)
{
#define NTTQ_INST(LL, TT, MB) if (LOGL == LL && TB == TT) { launch<LL, TT, MODE, MB>(st, cols, in, out, ics, ocs, lgX, rows_in, LT, tw, lo, hi, scale, flags); return; }
    NTTQ_INST(8, 16, 4) NTTQ_INST(8, 32, 2) NTTQ_INST(9, 16, 2) NTTQ_INST(9, 32, 1)
    NTTQ_INST(9, 8, 3) NTTQ_INST(10, 8, 2) NTTQ_INST(11, 8, 1)
#undef NTTQ_INST
    throw cuda_error(cudaErrorInvalidValue, "nttq: no instance");
}

struct Tables {
    LineTabs col, row;
    fr_t* A = nullptr; fr_t* B = nullptr;
    unsigned lgN, lgLc, lgLr;
};
// lgN = lgLc + lgLr; the column line has length 2^colL (colL = lgLc, or lgLc - 2 for the coset pass)
static const Tables& tables(const gpu_t& gpu, unsigned lgN, unsigned lgLr, unsigned colL, bool inverse)
{
    static Tables tabs[16][2][24][13];
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    Tables& T = tabs[gpu.id()][inverse][lgN][lgLr];
    if (T.A == nullptr) {
        const fr_t* roots = inverse ? inverse_roots_of_unity : forward_roots_of_unity;
        T.lgN = lgN; T.lgLr = lgLr; T.lgLc = lgN - lgLr;
        T.col = make_line(gpu, roots[lgN], lgN, colL);
        T.row = make_line(gpu, roots[lgN], lgN, lgLr);
        size_t nB = lgN > 12 ? (size_t)1 << (lgN - 12) : 1;
        fr_t *A, *B;
        CUDA_OK(risc0_cuda_malloc(&A, 4096 * sizeof(fr_t)));
        CUDA_OK(risc0_cuda_malloc(&B, nB * sizeof(fr_t)));
        k_pow_table<<<16, 256, 0, gpu>>>(A, roots[lgN], 4096, 1);
        k_pow_table<<<(unsigned)((nB + 255) / 256), 256, 0, gpu>>>(B, roots[lgN], (unsigned)nB, 4096);
        CUDA_OK(cudaGetLastError());
        gpu.sync();
        T.B = B;
        T.A = A;
    }
    return T;
}

// Per-size schedule (measured on RTX 5090): inverse row line 2^IR (column 2^(lg-IR)),
// forward row line 2^FR (coset column 2^(lg-FR)), TB per pass, side streams, columns per forward
// launch, columns per inverse chunk.
struct Cfg { unsigned IR, FR, ITC, ITR, FTC, FTR, NS, CHF, CHI; };
static Cfg cfg(unsigned lg) {
    switch (lg) {
    case 16: return Cfg{8, 8, 16, 16, 16, 16, 0, 32, 32};
    case 17: return Cfg{8, 9, 32, 32, 16, 16, 0, 32, 32};
    case 18: return Cfg{9, 10, 16, 16, 16, 8, 2, 2, 8};
    case 19: return Cfg{10, 10, 8, 8, 16, 8, 2, 1, 2};
    case 20: return Cfg{10, 10, 8, 8, 8, 8, 2, 1, 2};
    default: return Cfg{10, 10, 8, 8, 8, 8, 2, 1, 1};   // 21
    }
}

static bool supported(unsigned lg, unsigned lgB) { return lgB == 2 && lg >= 16 && lg <= 21; }

// The four-step commit NTT and LDE (this file): on unless RISC0_CUDA_NTTQ=0.
static bool enabled() {
    static const bool on = [] { const char* e = getenv("RISC0_CUDA_NTTQ"); return !(e && e[0] == '0'); }();
    return on;
}

// Scratch T2 of the forward passes: one allocation sized for the largest schedule below, made on
// first use and never regrown. A regrow at a po2 increase came after the HAL's buffer pool and the
// new working set had filled the device, and failed (composite 20,21 in one process).
// NS == 0 uses T2[0] only (N * CHF elements); NS > 0 alternates T2[0] and T2[1].
static size_t t2_elems(unsigned lg)
{
    const Cfg G = cfg(lg);
    return ((size_t)4 << lg) * G.CHF * (G.NS ? 2 : 1);
}
static size_t t2_capacity()
{
    size_t m = 0;
    for (unsigned lg = 16; lg <= 21; lg++) m = std::max(m, t2_elems(lg));
    return m;
}

struct Side {
    cudaStream_t s[2] = {};
    cudaEvent_t ev[2] = {};
    cudaEvent_t inv = {};
    fr_t* T2 = nullptr;
    fr_t* t2(unsigned lg, unsigned x) const { return T2 + x * (t2_elems(lg) / 2); }
};
// one Side per device shared by commit and forward (both hold the mutex and join before returning)
static std::mutex side_mtx;
static Side& side(const gpu_t& gpu)
{
    static Side sides[16];
    Side& S = sides[gpu.id()];
    if (S.s[0] == nullptr) {
        CUDA_OK(risc0_cuda_malloc(&S.T2, t2_capacity() * sizeof(fr_t)));
        for (int i = 0; i < 2; i++) {
            CUDA_OK(cudaStreamCreateWithFlags(&S.s[i], cudaStreamNonBlocking));
            CUDA_OK(cudaEventCreateWithFlags(&S.ev[i], cudaEventDisableTiming));
        }
        CUDA_OK(cudaEventCreateWithFlags(&S.inv, cudaEventDisableTiming));
    }
    return S;
}

// forward LDE of `cols` natural-order coefficient columns Cn into E (issued on stream s)
static void fwd_cols(cudaStream_t s, const Tables& F, const Cfg& G, fr_t* T2, unsigned lg, unsigned kf,
                     const fr_t* Cn, fr_t* E)
{
    const unsigned lgN = lg + 2;
    const size_t n = (size_t)1 << lg, N = n << 2;
    pass<2>(s, lg - F.lgLr, G.FTC, kf, Cn, T2, n, N, F.lgLr, 0, F.col, Tw{F.A, F.B, lgN});
    pass<1>(s, F.lgLr, G.FTR, kf, T2, E, N, N, F.lgLc, 0, F.row, Tw{nullptr, nullptr, 0},
            nullptr, nullptr, fr_t(1), F_DISCARD_IN | F_CS_OUT);
}

// C (natural shifted coefficients) and E (LDE) of `cols` columns of W; W unchanged.
// lo/hi: zk-shift factor tables (nttx_tables), so lo[k & 1023] * hi[k >> 10] = g^k.
static void commit(const gpu_t& gpu, fr_t* C, fr_t* E, const fr_t* W, unsigned lg, unsigned lgB,
                   unsigned cols, const fr_t* lo, const fr_t* hi)
{
    const Cfg G = cfg(lg);
    const unsigned lgN = lg + lgB;
    const size_t n = (size_t)1 << lg, N = n << lgB;
    const Tables& I = tables(gpu, lg, G.IR, lg - G.IR, true);
    const Tables& F = tables(gpu, lgN, G.FR, lg - G.FR, false);
    std::lock_guard<std::mutex> lock(side_mtx);
    Side& S = side(gpu);
    cudaStream_t st = gpu;
    const fr_t ninv = domain_size_inverse[lg];
    unsigned rr = 0;
    for (unsigned c0 = 0; c0 < cols; c0 += G.CHI) {
        const unsigned k = std::min(G.CHI, cols - c0);
        // T1 lives in E of these columns: their forward passes are only issued below.
        fr_t* T1 = E + c0 * N;
        pass<0>(st, I.lgLc, G.ITC, k, W + c0 * n, T1, n, n, I.lgLr, 1u << I.lgLc, I.col, Tw{I.A, I.B, lg});
        pass<1>(st, I.lgLr, G.ITR, k, T1, C + c0 * n, n, n, I.lgLc, 0, I.row, Tw{nullptr, nullptr, 0},
                lo, hi, ninv, F_DISCARD_IN);
        if (G.NS == 0) {
            for (unsigned c = c0; c < c0 + k; c += G.CHF)
                fwd_cols(st, F, G, S.t2(lg, 0), lg, std::min(G.CHF, c0 + k - c), C + c * n, E + c * N);
            continue;
        }
        CUDA_OK(cudaEventRecord(S.inv, st));
        for (int i = 0; i < 2; i++) CUDA_OK(cudaStreamWaitEvent(S.s[i], S.inv, 0));
        for (unsigned c = c0; c < c0 + k; c += G.CHF) {
            const unsigned x = rr++ & 1;
            fwd_cols(S.s[x], F, G, S.t2(lg, x), lg, std::min(G.CHF, c0 + k - c), C + c * n, E + c * N);
        }
    }
    if (G.NS) for (int i = 0; i < 2; i++) {
        CUDA_OK(cudaEventRecord(S.ev[i], S.s[i]));
        CUDA_OK(cudaStreamWaitEvent(st, S.ev[i], 0));
    }
}

// E = LDE of natural-order coefficients Cn (cols columns), same scheme as commit's forward half.
static void forward(const gpu_t& gpu, fr_t* E, const fr_t* Cn, unsigned lg, unsigned lgB, unsigned cols)
{
    const Cfg G = cfg(lg);
    const unsigned lgN = lg + lgB;
    const size_t n = (size_t)1 << lg, N = n << lgB;
    const Tables& F = tables(gpu, lgN, G.FR, lg - G.FR, false);
    std::lock_guard<std::mutex> lock(side_mtx);
    Side& S = side(gpu);
    cudaStream_t st = gpu;
    if (G.NS == 0) {
        for (unsigned c = 0; c < cols; c += G.CHF)
            fwd_cols(st, F, G, S.t2(lg, 0), lg, std::min(G.CHF, cols - c), Cn + c * n, E + c * N);
        return;
    }
    CUDA_OK(cudaEventRecord(S.inv, st));
    for (int i = 0; i < 2; i++) CUDA_OK(cudaStreamWaitEvent(S.s[i], S.inv, 0));
    unsigned rr = 0;
    for (unsigned c = 0; c < cols; c += G.CHF) {
        const unsigned x = rr++ & 1;
        fwd_cols(S.s[x], F, G, S.t2(lg, x), lg, std::min(G.CHF, cols - c), Cn + c * n, E + c * N);
    }
    for (int i = 0; i < 2; i++) {
        CUDA_OK(cudaEventRecord(S.ev[i], S.s[i]));
        CUDA_OK(cudaStreamWaitEvent(st, S.ev[i], 0));
    }
}

} // namespace nttq
