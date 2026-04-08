// Intel SYCL eval_check kernel for recursion circuit.
// This file is appended to an amalgamation that already contains:
//   - fp.h, fpext.h includes
//   - The poly_fp function definition (from poly_fp.cpp)
//   - namespace risc0::circuit::recursion with kInvRate defined
// So we only need the SYCL kernel wrapper here.

#include <sycl/sycl.hpp>
#include <cstring>

using namespace risc0;

static const char* make_error(const char* msg) { return strdup(msg); }

extern "C" const char* risc0_circuit_recursion_intel_eval_check(
    void* queue_ptr,
    void* d_check,
    const void* d_ctrl,
    const void* d_data,
    const void* d_accum,
    const void* d_mix,
    const void* d_out,
    const void* d_poly_mix,
    uint32_t rou_raw,
    uint32_t po2,
    uint32_t domain)
{
    auto* q = static_cast<sycl::queue*>(queue_ptr);

    try {
        auto* check = static_cast<Fp*>(d_check);
        auto* ctrl = static_cast<Fp*>(const_cast<void*>(d_ctrl));
        auto* data = static_cast<Fp*>(const_cast<void*>(d_data));
        auto* accum = static_cast<Fp*>(const_cast<void*>(d_accum));
        auto* mix = static_cast<Fp*>(const_cast<void*>(d_mix));
        auto* out = static_cast<Fp*>(const_cast<void*>(d_out));
        auto* poly_mix = static_cast<FpExt*>(const_cast<void*>(d_poly_mix));

        // Construct rou from raw Montgomery bits
        Fp rou_val;
        std::memcpy(&rou_val, &rou_raw, sizeof(uint32_t));

        constexpr uint32_t WG_SIZE = 1024;
        uint32_t global_size = ((domain + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

        q->parallel_for(
            sycl::nd_range<1>(global_size, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                uint32_t cycle = item.get_global_id(0);
                if (cycle >= domain) return;

                // args order: ctrl(0), out/global(1), data(2), mix(3), accum(4)
                Fp* args[5] = {ctrl, out, data, mix, accum};

                FpExt tot = circuit::recursion::poly_fp(
                    (size_t)cycle, (size_t)domain, poly_mix, args);

                Fp x = Fp(3) * pow(rou_val, cycle);
                Fp y = pow(x, uint32_t(1) << po2);
                Fp quot = inv(y - Fp(1));

                for (uint32_t i = 0; i < 4; i++) {
                    check[i * domain + cycle] = tot.elems[i] * quot;
                }
            });
        q->wait();

        return nullptr;
    } catch (const sycl::exception& e) {
        return make_error(e.what());
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("Unknown error in intel eval_check");
    }
}
