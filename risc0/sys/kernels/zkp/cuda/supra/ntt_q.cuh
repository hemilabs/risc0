// Two-pass (four-step) fused commit NTT for lg_domain_size 20..22, lg_blowup 2 (CUDA only).
//   per column: W (evaluations) --colpass--> T1 --rowpass (* n^-1 * g^k)--> C (natural coefficients)
//               C zero-padded to N --colpass--> T2 --rowpass--> E (natural-order LDE)
// Each pass computes batched line DFTs of length L = 2^LOGL (1024..4096) as 16 x 16 x P3 stages in
// registers with two shared-memory exchanges (Stockham), TB lines per block; the column pass
// multiplies the four-step twiddle w_N^(col*k). All arithmetic is sppark's canonical Montgomery
// bb31, so the output equals the radix-2 path bit for bit.
// The forward passes run one column at a time, alternating between two side streams with their own
// T2 scratch, so T2 stays L2-resident and the kernels' tails overlap; the side streams are joined
// back into the gpu stream before returning.
#pragma once

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
    static constexpr int LS = L + L / 32 + 32 / TB;  // padded line stride
    __device__ static __forceinline__ int a(int i) { return i + (i >> 5); }
};

// MODE 0: column pass. Matrix [L rows][S = 2^lgX cols], lines = columns; only rows < rows_in are
//         read (rows_in == L/4: zero-padded expand). out[k*S + col] = DFT_L(col)[k] * w_N^(col*k).
// MODE 1: row pass. Matrix [R = 2^lgX rows][L cols], lines = rows.
//         out[row + R*k] = DFT_L(row)[k] (* scale * lo[i & 1023] * hi[i >> 10], i = row + R*k).
template<int LOGL, int TB, int MODE>
__global__ void __launch_bounds__(TB * (1 << LOGL) / 16)
pass_k(const fr_t* __restrict__ in, fr_t* __restrict__ out, size_t ics, size_t ocs,
       unsigned lgX, unsigned rows_in, LineTabs T, Tw tw,
       const fr_t* __restrict__ lo, const fr_t* __restrict__ hi, fr_t scale)
{
    constexpr int L = 1 << LOGL, LOGP3 = LOGL - 8, P3 = 1 << LOGP3;
    static_assert(LOGP3 >= 1 && LOGP3 <= 4, "LOGL 9..12");
    using SM = Smem<LOGL, TB>;
    constexpr int NT = TB * L / 16;
    constexpr int LP1 = L / 16;
    extern __shared__ fr_t smem[];
    in += blockIdx.y * ics;
    out += blockIdx.y * ocs;
    const size_t base0 = (size_t)blockIdx.x * TB;
    const int tid = threadIdx.x;

    fr_t r16[8];
    #pragma unroll
    for (int e = 0; e < 8; e++) r16[e] = T.r16[e];

    // stage 1 (Ns = 1, P = 16): global -> registers -> smem
    {
        int c, j;
        if (MODE == 0) { c = tid % TB; j = tid / TB; }
        else           { c = tid / LP1; j = tid % LP1; }
        fr_t v[16];
        if (MODE == 0) {
            const size_t S = (size_t)1 << lgX;
            const fr_t* src = in + base0 + c + (size_t)j * S;
            if (rows_in == L / 4) {
                #pragma unroll
                for (int m = 0; m < 4; m++) v[m] = src[(size_t)m * LP1 * S];
                // only v[0..3] nonzero: X[4a+b] = DFT4_a(x_m * w16^(m*b))
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
                for (int m = 0; m < 16; m++) v[m] = src[(size_t)m * LP1 * S];
                reg_dft<4>(v, r16);
            }
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
    // stage 2 (Ns = 16, P = 16): smem -> smem
    {
        int c = tid / LP1, j = tid % LP1, k = j & 15;
        fr_t* sl = smem + c * SM::LS;
        fr_t v[16];
        #pragma unroll
        for (int m = 0; m < 16; m++) v[m] = sl[SM::a(j + m * LP1)];
        #pragma unroll
        for (int m = 1; m < 16; m++) v[m] = v[m] * T.t2[m * 16 + k];
        reg_dft<4>(v, r16);
        __syncthreads();
        int b = (j >> 4) * 256 + k;
        #pragma unroll
        for (int m = 0; m < 16; m++) sl[SM::a(b + m * 16)] = v[m];
    }
    __syncthreads();
    // stage 3 (Ns = 256, P = P3): smem -> registers -> global
    {
        fr_t rP[P3 / 2];
        #pragma unroll
        for (int e = 0; e < P3 / 2; e++) rP[e] = T.rP3[e];
        constexpr int ITEMS = TB * 256, IPT = ITEMS / NT;
        const unsigned Nmask = (1u << tw.lgN) - 1;
        fr_t step;                                       // w_N^(256*col); col is fixed per thread
        if (MODE == 0) {
            unsigned e = ((unsigned)(base0 + tid % TB) * 256u) & Nmask;
            step = tw.A[e & 4095] * tw.B[e >> 12];
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
            for (int m = 1; m < P3; m++) v[m] = v[m] * T.t3[m * 256 + j];
            reg_dft<LOGP3>(v, rP);
            fr_t t;
            if (MODE == 0) {
                unsigned e = ((unsigned)(base0 + c) * (unsigned)j) & Nmask;
                t = tw.A[e & 4095] * tw.B[e >> 12];
            }
            #pragma unroll
            for (int m = 0; m < P3; m++) {
                unsigned k = j + m * 256;
                fr_t x = v[m];
                if (MODE == 0) {
                    x = x * t;
                    if (m + 1 < P3) t = t * step;
                    out[((size_t)k << lgX) + base0 + c] = x;
                } else {
                    size_t kk = base0 + c + ((size_t)k << lgX);
                    if (lo) x = x * scale * (lo[kk & 1023] * hi[kk >> 10]);
                    out[kk] = x;
                }
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

struct Tables {
    LineTabs col, row;
    fr_t* A = nullptr; fr_t* B = nullptr;
    unsigned lgN, lgLc, lgLr;
};

static LineTabs make_line(const gpu_t& gpu, fr_t w, unsigned lgN, unsigned LOGL)
{
    fr_t *r16, *rP3, *t2, *t3;
    int P3 = 1 << (LOGL - 8);
    CUDA_OK(cudaMalloc(&r16, 32 * sizeof(fr_t)));
    CUDA_OK(cudaMalloc(&rP3, 8 * sizeof(fr_t)));
    CUDA_OK(cudaMalloc(&t2, 256 * sizeof(fr_t)));
    CUDA_OK(cudaMalloc(&t3, 256 * P3 * sizeof(fr_t)));
    k_line_tabs<<<(256 * P3 + 255) / 256, 256, 0, gpu>>>(r16, rP3, t2, t3, w, (int)lgN, (int)LOGL);
    CUDA_OK(cudaGetLastError());
    return LineTabs{r16, rP3, t2, t3};
}

// lgN in 20..24: line sizes lgN/2 and lgN - lgN/2 are in 10..12
static const Tables& tables(const gpu_t& gpu, unsigned lgN, bool inverse)
{
    static Tables tabs[16][2][32];
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    Tables& T = tabs[gpu.id()][inverse][lgN];
    if (T.A == nullptr) {
        const fr_t* roots = inverse ? inverse_roots_of_unity : forward_roots_of_unity;
        T.lgN = lgN; T.lgLr = lgN / 2; T.lgLc = lgN - T.lgLr;
        T.col = make_line(gpu, roots[lgN], lgN, T.lgLc);
        T.row = make_line(gpu, roots[lgN], lgN, T.lgLr);
        size_t nB = (size_t)1 << (lgN - 12);
        fr_t *A, *B;
        CUDA_OK(cudaMalloc(&A, 4096 * sizeof(fr_t)));
        CUDA_OK(cudaMalloc(&B, nB * sizeof(fr_t)));
        k_pow_table<<<16, 256, 0, gpu>>>(A, roots[lgN], 4096, 1);
        k_pow_table<<<(unsigned)((nB + 255) / 256), 256, 0, gpu>>>(B, roots[lgN], (unsigned)nB, 4096);
        CUDA_OK(cudaGetLastError());
        gpu.sync();
        T.B = B;
        T.A = A;
    }
    return T;
}

template<int LOGL, int TB, int MODE>
static void launch(cudaStream_t st, unsigned cols, const fr_t* in, fr_t* out, size_t ics, size_t ocs,
                   unsigned lgX, unsigned rows_in, const LineTabs& LT, Tw tw,
                   const fr_t* lo, const fr_t* hi, fr_t scale)
{
    constexpr int L = 1 << LOGL;
    const size_t sh = (size_t)TB * Smem<LOGL, TB>::LS * sizeof(fr_t);
    static std::once_flag once;
    std::call_once(once, [&] {
        CUDA_OK(cudaFuncSetAttribute(pass_k<LOGL, TB, MODE>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    });
    dim3 grid((unsigned)(((size_t)1 << lgX) / TB), cols);
    pass_k<LOGL, TB, MODE><<<grid, TB * L / 16, sh, st>>>(in, out, ics, ocs, lgX, rows_in, LT, tw,
                                                          lo, hi, scale);
    CUDA_OK(cudaGetLastError());
}

// TB per line size: 16 lines of 1024, 8 of 2048, 4 of 4096 (1024 threads, <= 68 KB smem).
template<int MODE>
static void pass(cudaStream_t st, unsigned LOGL, unsigned cols, const fr_t* in, fr_t* out,
                 size_t ics, size_t ocs, unsigned lgX, unsigned rows_in, const LineTabs& LT, Tw tw,
                 const fr_t* lo = nullptr, const fr_t* hi = nullptr, fr_t scale = fr_t(1))
{
    switch (LOGL) {
    case 10: launch<10, 16, MODE>(st, cols, in, out, ics, ocs, lgX, rows_in, LT, tw, lo, hi, scale); break;
    case 11: launch<11, 8, MODE>(st, cols, in, out, ics, ocs, lgX, rows_in, LT, tw, lo, hi, scale); break;
    case 12: launch<12, 4, MODE>(st, cols, in, out, ics, ocs, lgX, rows_in, LT, tw, lo, hi, scale); break;
    default: throw cuda_error(cudaErrorInvalidValue, "nttq: line size");
    }
}

// lg22: 1.01x (no gain) for 256 MB of scratch, so lg 20-21 only.
static bool supported(unsigned lg, unsigned lgB) { return lgB == 2 && lg >= 20 && lg <= 21; }

static bool enabled() {
    static int on = -1;
    if (on < 0) { const char* e = getenv("RISC0_CUDA_NTTQ"); on = (e && e[0] == '0') ? 0 : 1; }
    return on == 1;
}

struct Side {
    cudaStream_t s[2] = {};
    cudaEvent_t ev[2] = {};
    cudaEvent_t inv = {};
    fr_t* T2[2] = {};
    size_t t2_elems = 0;
};

// C (natural shifted coefficients) and E (LDE) of `cols` columns of W; W unchanged.
// lo/hi: zk-shift factor tables (nttx_tables), so lo[k & 1023] * hi[k >> 10] = g^k.
static void commit(const gpu_t& gpu, fr_t* C, fr_t* E, const fr_t* W, unsigned lg, unsigned lgB,
                   unsigned cols, const fr_t* lo, const fr_t* hi)
{
    static Side sides[16];
    static std::mutex mtx;
    const unsigned lgN = lg + lgB;
    const size_t n = (size_t)1 << lg, N = n << lgB;
    const Tables& I = tables(gpu, lg, true);
    const Tables& F = tables(gpu, lgN, false);
    std::lock_guard<std::mutex> lock(mtx);
    Side& S = sides[gpu.id()];
    if (S.s[0] == nullptr) {
        for (int i = 0; i < 2; i++) {
            CUDA_OK(cudaStreamCreateWithFlags(&S.s[i], cudaStreamNonBlocking));
            CUDA_OK(cudaEventCreateWithFlags(&S.ev[i], cudaEventDisableTiming));
        }
        CUDA_OK(cudaEventCreateWithFlags(&S.inv, cudaEventDisableTiming));
    }
    if (S.t2_elems < N) {
        // previous users of T2 finished: every commit joins the side streams before returning
        for (int i = 0; i < 2; i++) {
            if (S.T2[i]) CUDA_OK(cudaFree(S.T2[i]));
            CUDA_OK(cudaMalloc(&S.T2[i], N * sizeof(fr_t)));
        }
        S.t2_elems = N;
    }
    cudaStream_t st = gpu;
    const fr_t ninv = domain_size_inverse[lg];
    constexpr unsigned CHI = 8;
    unsigned rr = 0;
    for (unsigned c0 = 0; c0 < cols; c0 += CHI) {
        const unsigned k = std::min(CHI, cols - c0);
        // T1 lives in E of these columns: their forward passes are only issued below.
        fr_t* T1 = E + c0 * N;
        pass<0>(st, I.lgLc, k, W + c0 * n, T1, n, n, I.lgLr, 1u << I.lgLc, I.col, Tw{I.A, I.B, lg});
        pass<1>(st, I.lgLr, k, T1, C + c0 * n, n, n, I.lgLc, 0, I.row, Tw{nullptr, nullptr, 0},
                lo, hi, ninv);
        CUDA_OK(cudaEventRecord(S.inv, st));
        for (int i = 0; i < 2; i++) CUDA_OK(cudaStreamWaitEvent(S.s[i], S.inv, 0));
        for (unsigned c = c0; c < c0 + k; c++) {
            const unsigned x = rr++ & 1;
            pass<0>(S.s[x], F.lgLc, 1, C + c * n, S.T2[x], n, N, F.lgLr, (unsigned)(n >> F.lgLr),
                    F.col, Tw{F.A, F.B, lgN});
            pass<1>(S.s[x], F.lgLr, 1, S.T2[x], E + c * N, N, N, F.lgLc, 0, F.row,
                    Tw{nullptr, nullptr, 0});
        }
    }
    for (int i = 0; i < 2; i++) {
        CUDA_OK(cudaEventRecord(S.ev[i], S.s[i]));
        CUDA_OK(cudaStreamWaitEvent(st, S.ev[i], 0));
    }
}

// E = LDE of natural-order coefficients Cn (cols columns), same scheme as commit's forward half.
static void forward(const gpu_t& gpu, fr_t* E, const fr_t* Cn, unsigned lg, unsigned lgB, unsigned cols)
{
    static Side sides[16];
    static std::mutex mtx;
    const unsigned lgN = lg + lgB;
    const size_t n = (size_t)1 << lg, N = n << lgB;
    const Tables& F = tables(gpu, lgN, false);
    std::lock_guard<std::mutex> lock(mtx);
    Side& S = sides[gpu.id()];
    if (S.s[0] == nullptr) {
        for (int i = 0; i < 2; i++) {
            CUDA_OK(cudaStreamCreateWithFlags(&S.s[i], cudaStreamNonBlocking));
            CUDA_OK(cudaEventCreateWithFlags(&S.ev[i], cudaEventDisableTiming));
        }
        CUDA_OK(cudaEventCreateWithFlags(&S.inv, cudaEventDisableTiming));
    }
    if (S.t2_elems < N) {
        for (int i = 0; i < 2; i++) {
            if (S.T2[i]) CUDA_OK(cudaFree(S.T2[i]));
            CUDA_OK(cudaMalloc(&S.T2[i], N * sizeof(fr_t)));
        }
        S.t2_elems = N;
    }
    cudaStream_t st = gpu;
    CUDA_OK(cudaEventRecord(S.inv, st));
    for (int i = 0; i < 2; i++) CUDA_OK(cudaStreamWaitEvent(S.s[i], S.inv, 0));
    for (unsigned c = 0; c < cols; c++) {
        const unsigned x = c & 1;
        pass<0>(S.s[x], F.lgLc, 1, Cn + c * n, S.T2[x], n, N, F.lgLr, (unsigned)(n >> F.lgLr),
                F.col, Tw{F.A, F.B, lgN});
        pass<1>(S.s[x], F.lgLr, 1, S.T2[x], E + c * N, N, N, F.lgLc, 0, F.row, Tw{nullptr, nullptr, 0});
    }
    for (int i = 0; i < 2; i++) {
        CUDA_OK(cudaEventRecord(S.ev[i], S.s[i]));
        CUDA_OK(cudaStreamWaitEvent(st, S.ev[i], 0));
    }
}

static bool enabled_lde() {
    static int on = -1;
    if (on < 0) { const char* e = getenv("RISC0_CUDA_NTTQ_LDE"); on = (e && e[0] == '0') ? 0 : 1; }
    return on == 1 && enabled();
}

} // namespace nttq
