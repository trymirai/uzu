#pragma once

#include <metal_stdlib>

#include "../../common/defines.h"

namespace uzu {
namespace gemm {

template <typename AT_, typename BT_, typename DT_>
struct GemvOperands {
  using AT = AT_;
  using BT = BT_;
  using DT = DT_;
  const device uint32_t* b;
  const device BT* scales;
  const device uint8_t* zero_points;
  const device BT* biases;
  const device AT* a;
  device DT* d;
  const device BT* output_bias;
  const device int32_t* hadamard_factors;
  const device uint* gather_indices;
};

struct GemvParams {
  uint in_vec_size;
  uint out_vec_size;
  uint output_row_stride;
  uint scale_output_stride;
  uint scale_group_stride;
  uint zero_point_output_stride;
  uint zero_point_group_stride;
  uint batch_size;
  float ab_scale;
  float soft_cap;
  GemmDTransform output_transform;
  bool gathered;
  bool signed_codes;
  GemmBLayout gemm_b_layout;
  uint gather_first_row;
  uint gather_row_count;

  METAL_FUNC bool gathers(uint gather_index) const thread { return gather_index - gather_first_row < gather_row_count; }

  METAL_FUNC uint gathered_row(uint gather_index) const thread {
    return gathers(gather_index) ? gather_index - gather_first_row : 0;
  }

  METAL_FUNC uint output_row(uint row) const thread {
    if (gemm_b_layout != GemmBLayout::MxuTiled || gathered) {
      return row;
    }
    const uint block_row = row % 32;
    return row - block_row + block_row / 16 * 4 + block_row / 4 % 4 + block_row % 4 * 8;
  }
};

} // namespace gemm
} // namespace uzu
