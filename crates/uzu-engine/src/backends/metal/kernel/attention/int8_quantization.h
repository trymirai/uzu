#pragma once

#include <metal_stdlib>
#include "../common/defines.h"

UZU_CONST float INT8_MAX = 127.0f;

template <ushort VALUES_PER_LANE>
METAL_FUNC bfloat kv_quantize_row(const thread float* values, thread int* codes) {
  float abs_max = 0.0f;
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    abs_max = metal::max(abs_max, metal::abs(values[value_index]));
  }
  abs_max = metal::simd_max(abs_max);
  const bfloat scale = bfloat(abs_max > 0.0f ? metal::precise::divide(abs_max, INT8_MAX) : 1.0f);
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    codes[value_index] =
        int(metal::clamp(metal::rint(metal::precise::divide(values[value_index], float(scale))), -INT8_MAX, INT8_MAX));
  }
  return scale;
}
