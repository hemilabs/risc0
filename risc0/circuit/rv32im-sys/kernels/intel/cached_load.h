// LSC cache-hint helper for non-ESIMD SYCL device code on Intel BMG.
// Wraps a scalar load through annotated_ptr with cache_control hints
// (L1+L3 cached). Tier B4 experiment: gated by RISC0_LSC_HINTS env var
// at build time. CSE script wraps `argK[expr]` -> `cached_load(&argK[expr])`
// only when --lsc-hints is passed.
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>

namespace risc0 {
namespace lsc {

template <typename T>
static inline T cached_load(const T* ptr) {
    namespace exp = sycl::ext::oneapi::experimental;
    namespace iex = sycl::ext::intel::experimental;
    auto props = exp::properties{
        iex::read_hint<iex::cache_control<iex::cache_mode::cached,
                                          exp::cache_level::L1,
                                          exp::cache_level::L3>>};
    exp::annotated_ptr<const T, decltype(props)> ap(ptr, props);
    return *ap;
}

} // namespace lsc
} // namespace risc0
