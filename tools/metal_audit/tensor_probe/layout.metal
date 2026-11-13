#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp::tensor_ops;
// For each lane: capacity and the (dim0, dim1) index of every element of the right-input and
// destination cooperative tensors of the chain-shape matmul (16 x 32 x 128, right transposed).
kernel void lay(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr auto desc = matmul2d_descriptor(16, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto cR = op.get_right_input_cooperative_tensor<uint8_t, uint8_t, int32_t>();
  using EQ = extents<int32_t, 128, 16>; using EB = extents<int32_t, 128, 32>;
  using TQ = tensor<device uint8_t, EQ, tensor_inline>; using TB = tensor<device uint8_t, EB, tensor_inline>;
  auto cD = op.get_destination_cooperative_tensor<TQ, TB, int32_t>();
  device int* o = out + lane * 1024;
  o[0] = cR.get_capacity(); o[1] = cD.get_capacity();
  for (uint16_t i = 0; i < cR.get_capacity() && i < 200; i++) {
    auto ix = cR.get_multidimensional_index(i); o[2 + 2*i] = cR.is_valid_element(i) ? ix[0] : -1; o[3 + 2*i] = ix[1]; }
  for (uint16_t i = 0; i < cD.get_capacity() && i < 100; i++) {
    auto ix = cD.get_multidimensional_index(i); o[500 + 2*i] = cD.is_valid_element(i) ? ix[0] : -1; o[501 + 2*i] = ix[1]; }
}
