// Intel SYCL eval_check kernel for rv32im circuit (monolithic).
// This file is appended to the monolithic amalgamation.
// Multi-pass kernels are in separate .so files (pass1, pass2).

#include <sycl/sycl.hpp>
#include <cstring>

using namespace risc0;

static const char* make_error(const char* msg) { return strdup(msg); }

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
        Fp rou_val;
        std::memcpy(&rou_val, &rou_raw, sizeof(uint32_t));
        constexpr uint32_t WG_SIZE = 256, POLY_MIX_COUNT = 458;
        uint32_t global_size = ((domain + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
        q->submit([&](sycl::handler& h) {
            sycl::local_accessor<FpExt, 1> slm(sycl::range<1>(POLY_MIX_COUNT), h);
            h.parallel_for(sycl::nd_range<1>(global_size, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    uint32_t cycle = item.get_global_id(0);
                    uint32_t lid = item.get_local_id(0);
                    for (uint32_t i = lid; i < POLY_MIX_COUNT; i += WG_SIZE) slm[i] = poly_mix[i];
                    sycl::group_barrier(item.get_group());
                    if (cycle >= domain) return;
                    Fp* args[4] = {accum, data, out, mix};
                    FpExt* pm = slm.get_multi_ptr<sycl::access::decorated::no>().get();
                    FpExt tot = circuit::rv32im_v2::poly_fp(
                        (size_t)cycle, (size_t)domain, pm, args);
                    Fp x = Fp(3) * pow(rou_val, cycle);
                    Fp y = pow(x, uint32_t(1) << po2);
                    Fp quot = inv(y - Fp(1));
                    for (uint32_t i = 0; i < 4; i++)
                        check[i * domain + cycle] = tot.elems[i] * quot;
                });
        });
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
