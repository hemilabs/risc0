// Intel SYCL eval_check kernel for rv32im circuit (monolithic).
// This file is appended to the monolithic amalgamation.
// Multi-pass kernels are in separate .so files (pass1, pass2).

#include <sycl/sycl.hpp>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

using namespace risc0;

static const char* make_error(const char* msg) { return strdup(msg); }

#ifdef RK_NWORDS
// eval_slab.py moved poly_fp's per-row arrays into a slot-major global slab:
// RK_NWORDS u32 words per row, in blocks of RK_STRIDE rows. The launch is tiled
// so the slab covers one tile; tiles run in order on the single in-order queue
// and each row only reads words it wrote, so reuse across tiles is safe.
// g_slab_mutex guards the slab globals: eval_check holds it while it picks
// the slab and enqueues the tiles, and slab_free (also on release) waits for
// the queue before freeing, so kernels already enqueued by any thread finish
// first.
static std::mutex g_slab_mutex;
static uint32_t* g_slab = nullptr;
static size_t g_slab_words = 0;
static sycl::queue* g_slab_queue = nullptr;
// Largest tile that could be allocated; after a failure, later calls start
// from it instead of draining the queue to retry the full size every time.
static uint32_t g_tile_cap = 1u << 26;

static void slab_free() {
    if (g_slab) {
        g_slab_queue->wait();  // sycl::free does not wait for queued kernels
        sycl::free(g_slab, *g_slab_queue);
    }
    g_slab = nullptr;
    g_slab_words = 0;
    g_slab_queue = nullptr;
}

static size_t slab_words_for(uint32_t rows) {
    return (size_t(rows) + RK_STRIDE - 1) / RK_STRIDE * RK_NWORDS * RK_STRIDE;
}

// Rows per launch: min(tile, domain). The slab is always sized for a full
// tile (2^20 rows = 1.4 GB by default, RISC0_EVAL_SLAB_TILE_LOG2 overrides),
// independent of the domain, so segments of different po2 never regrow it
// (a regrow drains the queue). If that allocation fails the tile halves, down
// to 2^14 rows, and g_tile_cap keeps the smaller size until release.
static uint32_t slab_tile(sycl::queue* q, uint32_t domain) {
    uint32_t tile = 1u << 20;
    if (const char* s = std::getenv("RISC0_EVAL_SLAB_TILE_LOG2")) {
        int v = std::atoi(s);
        if (v >= 9 && v <= 26) tile = 1u << v;
    }
    if (tile > g_tile_cap) tile = g_tile_cap;
    for (;;) {
        const uint32_t launch = tile < domain ? tile : domain;
        if (g_slab && g_slab_queue == q && g_slab_words >= slab_words_for(launch)) return launch;
        slab_free();
        const size_t need = slab_words_for(tile);
        g_slab = sycl::malloc_device<uint32_t>(need, *q);
        if (g_slab) {
            g_slab_words = need;
            g_slab_queue = q;
            return launch;
        }
        if (tile <= (1u << 14)) throw std::runtime_error("eval_check: slab allocation failed");
        tile >>= 1;
        g_tile_cap = tile;
    }
}
#endif

extern "C" const char* risc0_circuit_rv32im_intel_eval_check(
    void* queue_ptr, void* d_check,
    const void* d_data, const void* d_accum, const void* d_out, const void* d_mix,
    const void* d_poly_mix, uint32_t rou_raw, uint32_t po2, uint32_t domain)
{
    auto* q = static_cast<sycl::queue*>(queue_ptr);
    try {
        auto* check = static_cast<Fp*>(d_check);
        auto* data = static_cast<Fp*>(const_cast<void*>(d_data));
        auto* accum = static_cast<Fp*>(const_cast<void*>(d_accum));
        auto* out = static_cast<Fp*>(const_cast<void*>(d_out));
        auto* mix = static_cast<Fp*>(const_cast<void*>(d_mix));
        auto* poly_mix = static_cast<FpExt*>(const_cast<void*>(d_poly_mix));
#ifdef RISC0_INTEL_EVAL_FAST
        // The vanishing quotient 1/((3*rou^cycle)^(2^po2) - 1) takes only 4
        // values over the coset (rou^(2^po2) is a primitive 4th root), so the
        // host appends them after the 458 poly_mix powers (see intel.rs).
        const Fp* qtab = reinterpret_cast<const Fp*>(poly_mix + 458);
#endif
        Fp rou_val;
        std::memcpy(&rou_val, &rou_raw, sizeof(uint32_t));
        // WG_SIZE=512 measured 11% E2E faster than WG=256 on B70 (BMG-G31)
        // at PO2=20 segments; +44% on tiny workloads. eval_check kernel
        // resource budget caps WG at 512 (compiled with `simd_size=16`,
        // `eu_thread_count=4`, 256-GRF; 1024 exceeds resource limit).
        // Runtime-tunable via RISC0_EVAL_CHECK_WG for further sweeps.
        uint32_t WG_SIZE = 512;
        if (const char* s = std::getenv("RISC0_EVAL_CHECK_WG")) {
            int v = std::atoi(s);
            if (v == 16 || v == 32 || v == 64 || v == 128 || v == 256 || v == 512) {
                WG_SIZE = (uint32_t)v;
            }
        }
        const uint32_t wgsz = WG_SIZE;
        constexpr uint32_t POLY_MIX_COUNT = 458;
#ifdef RK_NWORDS
        std::lock_guard<std::mutex> slab_lock(g_slab_mutex);
        const uint32_t tile = slab_tile(q, domain);
        uint32_t* slab = g_slab;
        for (uint32_t tile_base = 0; tile_base < domain; tile_base += tile) {
        const uint32_t rows = (domain - tile_base) < tile ? (domain - tile_base) : tile;
        uint32_t global_size = ((rows + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
#else
        uint32_t global_size = ((domain + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
#endif
        q->submit([&](sycl::handler& h) {
            sycl::local_accessor<FpExt, 1> slm(sycl::range<1>(POLY_MIX_COUNT), h);
            h.parallel_for(sycl::nd_range<1>(global_size, WG_SIZE),
                [=](sycl::nd_item<1> item) {
#ifdef RK_NWORDS
                    uint32_t lrow = item.get_global_id(0);
                    uint32_t cycle = tile_base + lrow;
#else
                    uint32_t cycle = item.get_global_id(0);
#endif
                    uint32_t lid = item.get_local_id(0);
                    for (uint32_t i = lid; i < POLY_MIX_COUNT; i += wgsz) slm[i] = poly_mix[i];
                    sycl::group_barrier(item.get_group());
#ifdef RK_NWORDS
                    if (lrow >= rows) return;
                    Fp* args[5] = {accum, data, out, mix,
                                   reinterpret_cast<Fp*>(slab + (size_t)(lrow / RK_STRIDE) * RK_NWORDS * RK_STRIDE +
                                                         (lrow % RK_STRIDE))};
#else
                    if (cycle >= domain) return;
                    Fp* args[4] = {accum, data, out, mix};
#endif
                    FpExt* pm = slm.get_multi_ptr<sycl::access::decorated::no>().get();
                    FpExt tot = circuit::rv32im_v2::poly_fp(
                        (size_t)cycle, (size_t)domain, pm, args);
#ifdef RISC0_INTEL_EVAL_FAST
                    Fp quot = qtab[cycle & 3];
#else
                    Fp x = Fp(3) * pow(rou_val, cycle);
                    Fp y = pow(x, uint32_t(1) << po2);
                    Fp quot = inv(y - Fp(1));
#endif
                    for (uint32_t i = 0; i < 4; i++)
                        check[i * domain + cycle] = tot.elems[i] * quot;
                });
        });
#ifdef RK_NWORDS
        }
#endif
        return nullptr;
    } catch (const sycl::exception& e) { return make_error(e.what()); }
    catch (const std::exception& e) { return make_error(e.what()); }
    catch (...) { return make_error("Unknown error in intel eval_check"); }
}

extern "C" const char* risc0_circuit_rv32im_intel_eval_check_sync(void* queue_ptr) {
    auto* q = static_cast<sycl::queue*>(queue_ptr);
    try { q->wait(); return nullptr; }
    catch (const sycl::exception& e) { return make_error(e.what()); }
    catch (const std::exception& e) { return make_error(e.what()); }
    catch (...) { return make_error("Unknown error in eval_check_sync"); }
}

// Frees eval_check's scratch (the RK_NWORDS slab) between proving sessions so
// it does not stay resident; it is reallocated on the next eval_check.
extern "C" const char* risc0_circuit_rv32im_intel_eval_check_release() {
    try {
#ifdef RK_NWORDS
        std::lock_guard<std::mutex> slab_lock(g_slab_mutex);
        slab_free();
        g_tile_cap = 1u << 26;  // next session may try the full tile again
#endif
        return nullptr;
    } catch (const sycl::exception& e) { return make_error(e.what()); }
    catch (const std::exception& e) { return make_error(e.what()); }
    catch (...) { return make_error("Unknown error in eval_check_release"); }
}
