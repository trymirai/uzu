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
  const device BT* outer_scales;
  const device AT* a;
  device DT* d;
  const device BT* output_bias;
  const device int32_t* hadamard_factors;
  const device uint* gather_indices;
};

struct GemvParams {
  uint in_vec_size;
  uint out_vec_size;
  uint batch_size;
  float ab_scale;
  float soft_cap;
  GemmDTransform output_transform;
  bool gathered;
  bool signed_codes;
};

} // namespace gemm
} // namespace uzu
