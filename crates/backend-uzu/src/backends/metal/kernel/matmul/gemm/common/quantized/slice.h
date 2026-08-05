#pragma once

#include <metal_stdlib>

#include "../../../common/quant_pack.h"

using namespace metal;

namespace uzu {
namespace gemm {

struct QuantizedSlice {
  int row_stride_bytes;
  int groups_per_row;
  int first_group;
};

template <typename RightOperand>
static METAL_FUNC QuantizedSlice make_quantized_slice(const uint k_offset, const int k_elements) {
  constexpr int pack_factor = get_pack_factor<RightOperand::BITS, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<RightOperand::BITS>();
  return QuantizedSlice{
      k_elements * bytes_per_pack / pack_factor,
      (k_elements + RightOperand::GROUP_SIZE - 1) / RightOperand::GROUP_SIZE,
      int(k_offset) / RightOperand::GROUP_SIZE,
  };
}

} // namespace gemm
} // namespace uzu
