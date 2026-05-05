#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;

#define GRAIN_SIZE 8

template <typename T, typename ScaleT>
VARIANTS(T, float, half, bfloat)
VARIANTS(ScaleT, float, half, bfloat)
PUBLIC KERNEL(PerLayerEmbeddingCombine)(
    const device T* token_ple,
    const device T* model_ple,
    const device ScaleT* scales,
    device T* combined,
    constant uint& suffix_length,
    constant uint& num_layers,
    constant uint& ple_dim,
    constant float& model_projection_scale,
    constant float& input_scale,
    constant float& epsilon,
    constant float& scale_offset,
    const uint row GROUPS(suffix_length * num_layers),
    const uint lane_id THREADS(METAL_SIMD_SIZE)
) {
  const uint token = row / num_layers;
  const uint layer = row % num_layers;
  const uint total_ple_dim = num_layers * ple_dim;
  const uint offset = token * total_ple_dim + layer * ple_dim;

  float partial_sum = 0.0f;
  for (uint base_dim = lane_id * GRAIN_SIZE; base_dim < ple_dim;
       base_dim += METAL_SIMD_SIZE * GRAIN_SIZE) {
    for (uint grain_index = 0; grain_index < GRAIN_SIZE; ++grain_index) {
      const uint dim = base_dim + grain_index;
      if (dim >= ple_dim) {
        continue;
      }
      float value = float(model_ple[offset + dim]) * model_projection_scale;
      partial_sum += value * value;
    }
  }

  float sum_sq = simd_sum(partial_sum);
  float rms_inv = rsqrt(sum_sq / float(ple_dim) + epsilon);

  for (uint base_dim = lane_id * GRAIN_SIZE; base_dim < ple_dim;
       base_dim += METAL_SIMD_SIZE * GRAIN_SIZE) {
    for (uint grain_index = 0; grain_index < GRAIN_SIZE; ++grain_index) {
      const uint dim = base_dim + grain_index;
      if (dim >= ple_dim) {
        continue;
      }
      float model_value =
          float(model_ple[offset + dim]) * model_projection_scale;
      float token_value = float(token_ple[offset + dim]);
      float scale = float(scales[dim]) + scale_offset;
      combined[offset + dim] =
          T((token_value + model_value * rms_inv * scale) * input_scale);
    }
  }
}
