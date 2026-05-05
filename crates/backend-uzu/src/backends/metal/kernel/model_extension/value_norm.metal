#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;

#define GRAIN_SIZE 8

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(ValueNorm)(
    device T* qkv,
    constant uint& batch_size,
    constant uint& num_heads,
    constant uint& num_groups,
    constant uint& head_dim,
    constant float& epsilon,
    const uint row GROUPS(batch_size * num_groups),
    const uint lane_id THREADS(METAL_SIMD_SIZE)
) {
  const uint batch = row / num_groups;
  const uint group = row % num_groups;
  const uint row_stride = (num_heads + 2 * num_groups) * head_dim;
  const uint values_offset = (num_heads + num_groups) * head_dim;
  const uint offset = batch * row_stride + values_offset + group * head_dim;

  float partial_sum = 0.0f;
  for (uint base_dim = lane_id * GRAIN_SIZE; base_dim < head_dim;
       base_dim += METAL_SIMD_SIZE * GRAIN_SIZE) {
    for (uint grain_index = 0; grain_index < GRAIN_SIZE; ++grain_index) {
      const uint dim = base_dim + grain_index;
      if (dim >= head_dim) {
        continue;
      }
      float value = float(qkv[offset + dim]);
      partial_sum += value * value;
    }
  }

  float sum_sq = simd_sum(partial_sum);
  float rms_inv = rsqrt(sum_sq / float(head_dim) + epsilon);

  for (uint base_dim = lane_id * GRAIN_SIZE; base_dim < head_dim;
       base_dim += METAL_SIMD_SIZE * GRAIN_SIZE) {
    for (uint grain_index = 0; grain_index < GRAIN_SIZE; ++grain_index) {
      const uint dim = base_dim + grain_index;
      if (dim >= head_dim) {
        continue;
      }
      qkv[offset + dim] = T(float(qkv[offset + dim]) * rms_inv);
    }
  }
}
