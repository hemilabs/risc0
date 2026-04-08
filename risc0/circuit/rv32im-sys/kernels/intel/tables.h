// Intel SYCL version of tables.h — uses sycl::atomic_ref instead of cuda::atomic.
#pragma once
#include "fp.h"
#include <sycl/sycl.hpp>
#include <cstdint>

namespace risc0::circuit::rv32im_v2::intel {

struct LookupTables {
  uint32_t* tableU8;   // 256 entries, device-allocated
  uint32_t* tableU16;  // 65536 entries, device-allocated

  inline void lookupDelta(Fp table, Fp index, Fp count) {
    uint32_t tableU32 = table.asUInt32();
    uint32_t indexU32 = index.asUInt32();
    if (tableU32 == 0) {
      return;  // cycle table — no-op
    }
    if (tableU32 == 8) {
      sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          ref(tableU8[indexU32]);
      ref.fetch_add(1);
    } else {
      sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          ref(tableU16[indexU32]);
      ref.fetch_add(1);
    }
  }

  inline Fp lookupCurrent(Fp table, Fp index) {
    uint32_t tableU32 = table.asUInt32();
    uint32_t indexU32 = index.asUInt32();
    if (tableU32 == 8) {
      sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          ref(tableU8[indexU32]);
      return Fp(ref.load());
    } else {
      sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          ref(tableU16[indexU32]);
      return Fp(ref.load());
    }
  }
};

} // namespace risc0::circuit::rv32im_v2::intel
