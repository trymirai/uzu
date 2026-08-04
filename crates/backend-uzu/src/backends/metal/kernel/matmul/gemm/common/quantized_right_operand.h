#pragma once

#include <metal_stdlib>

#include "../../common/fragment.h"
#include "../../common/mxu_fragment/ops.h"
#include "quant_unpack.h"

using namespace metal;

namespace uzu {
namespace gemm {

using uzu::matmul::DeviceTensorOperand;
using uzu::matmul::IntegerFormat;
using uzu::matmul::Signedness;

using SignedInt4Format = IntegerFormat<4, Signedness::Signed>;
using SignedInt8Format = IntegerFormat<8, Signedness::Signed>;

template <typename Format, bool ALIGNED_N, ushort TILES_N, ushort TILES_K, ushort SIMDGROUP_BLOCK_K>
struct QuantizedRightOperand;

template <ushort TILES_N, ushort TILES_K, ushort SIMDGROUP_BLOCK_K>
struct QuantizedRightOperand<SignedInt4Format, true, TILES_N, TILES_K, SIMDGROUP_BLOCK_K> {
  static METAL_FUNC DeviceTensorOperand<SignedInt4Format> make(
      const device uint8_t* b_packed_simdgroup,
      const int k_element_offset,
      const int b_row_stride_bytes,
      const short,
      const short2,
      const ushort
  ) {
    return DeviceTensorOperand<SignedInt4Format>{b_packed_simdgroup + (k_element_offset >> 1), b_row_stride_bytes};
  }
};

template <ushort TILES_N, ushort TILES_K, ushort SIMDGROUP_BLOCK_K>
struct QuantizedRightOperand<SignedInt4Format, false, TILES_N, TILES_K, SIMDGROUP_BLOCK_K> {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using RegisterTile = uzu::matmul::Fragment<int8_t, TILES_N, TILES_K, Ops>;

  static METAL_FUNC RegisterTile make(
      const device uint8_t* b_packed_simdgroup,
      const int k_element_offset,
      const int b_row_stride_bytes,
      const short simdgroup_limit_n,
      const short2 position,
      const ushort
  ) {
    RegisterTile right_tile;
    METAL_PRAGMA_UNROLL
    for (ushort tile_n = 0; tile_n < TILES_N; ++tile_n) {
      METAL_PRAGMA_UNROLL
      for (ushort tile_k = 0; tile_k < TILES_K; ++tile_k) {
        thread auto& weight_vector = right_tile.fragment_at(tile_n, tile_k);
        METAL_PRAGMA_UNROLL
        for (ushort thread_row = 0; thread_row < Ops::THREAD_ELEMENT_ROWS; ++thread_row) {
          const short row =
              short(tile_n * Ops::FRAGMENT_ROWS) + position.y + short(thread_row * Ops::THREAD_ELEMENT_ROW_STRIDE);
          const ushort element_base = thread_row * Ops::THREAD_ELEMENT_COLS;
          char4 codes = char4(0);
          if (row < simdgroup_limit_n) {
            const int k_base = k_element_offset + int(tile_k * Ops::FRAGMENT_COLS) + int(position.x);
            const ushort packed = *reinterpret_cast<const device ushort*>(
                b_packed_simdgroup + int(row) * b_row_stride_bytes + (k_base >> 1)
            );
            codes = unpack_signed_nibbles_to_int8(uint(packed));
          }
          weight_vector[element_base + 0] = codes.x;
          weight_vector[element_base + 1] = codes.y;
          weight_vector[element_base + 2] = codes.z;
          weight_vector[element_base + 3] = codes.w;
        }
      }
    }
    return right_tile;
  }
};

template <bool ALIGNED_N, ushort TILES_N, ushort TILES_K, ushort SIMDGROUP_BLOCK_K>
struct QuantizedRightOperand<SignedInt8Format, ALIGNED_N, TILES_N, TILES_K, SIMDGROUP_BLOCK_K> {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using RegisterTile = uzu::matmul::Fragment<int8_t, TILES_N, TILES_K, Ops>;

  static METAL_FUNC RegisterTile make(
      const device uint8_t* b_packed_simdgroup,
      const int k_element_offset,
      const int b_row_stride_bytes,
      const short simdgroup_limit_n,
      const short2,
      const ushort simd_lane_id
  ) {
    RegisterTile right_tile;
    auto right_src = uzu::matmul::fragment_source(
        reinterpret_cast<const device int8_t*>(b_packed_simdgroup) + k_element_offset,
        b_row_stride_bytes
    );
    if constexpr (!ALIGNED_N) {
      right_src = right_src.bounded(simdgroup_limit_n, SIMDGROUP_BLOCK_K);
    }
    right_tile.load_from(simd_lane_id, right_src);
    return right_tile;
  }
};

} // namespace gemm
} // namespace uzu
