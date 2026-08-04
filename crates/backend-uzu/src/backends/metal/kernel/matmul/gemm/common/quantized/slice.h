#pragma once

#include <metal_stdlib>

#include "../../../common/mxu_fragment/integer_formats.h"
#include "../../../common/quant_pack.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <typename Format>
struct QuantizedSlice {
  int row_stride_bytes;
  int groups_per_row;
  int first_group;
  const device uint8_t* storage;
};

template <typename Format, int GROUP_SIZE, typename ElementType>
static METAL_FUNC QuantizedSlice<Format> make_quantized_slice(
    const device ElementType* storage,
    const size_t block_index,
    const uint k_offset,
    const int k_elements
) {
  constexpr int pack_factor = get_pack_factor<Format::BITS, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<Format::BITS>();
  static_assert(GROUP_SIZE > 0, "quantized weight groups must be non-empty");
  const int row_stride_bytes = k_elements * bytes_per_pack / pack_factor;
  return QuantizedSlice<Format>{
      row_stride_bytes,
      (k_elements + GROUP_SIZE - 1) / GROUP_SIZE,
      int(k_offset) / GROUP_SIZE,
      reinterpret_cast<const device uint8_t*>(storage) + block_index * row_stride_bytes +
          int(k_offset) * bytes_per_pack / pack_factor,
  };
}

} // namespace gemm
} // namespace uzu
