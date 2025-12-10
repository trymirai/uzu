#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

#define BLOCK_SIZE 1024
#define SIMD_SIZE 32
#define GRAIN_SIZE 4

// LayerNorm: subtract mean, then normalize (for BERT-style models)
template <typename InputT, typename ScaleT, typename OutputT, typename AccumT>
void layer_norm_core(
    const device InputT* input_data,
    const device ScaleT* scales_data,
    device OutputT* output_data,
    uint element_count,
    constant float& epsilon,
    constant float& scale_offset,
    threadgroup AccumT* shared_mean,
    threadgroup AccumT* shared_variance,
    uint thread_in_row,
    bool full_layer
) {
  AccumT partial_sum = static_cast<AccumT>(0.0f);

  // Compute mean: sum all elements
  for (uint base_i = thread_in_row * GRAIN_SIZE; base_i < element_count;
       base_i += BLOCK_SIZE * GRAIN_SIZE) {
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i < element_count) {
        partial_sum += static_cast<AccumT>(input_data[i]);
      }
    }
  }

  // Reduce to get total mean
  AccumT total_sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(
      partial_sum,
      shared_mean,
      thread_in_row
  );
  AccumT mean = total_sum / static_cast<AccumT>(element_count);

  // Compute variance: sum of squared deviations
  AccumT partial_var_sum = static_cast<AccumT>(0.0f);
  for (uint base_i = thread_in_row * GRAIN_SIZE; base_i < element_count;
       base_i += BLOCK_SIZE * GRAIN_SIZE) {
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i < element_count) {
        AccumT centered = static_cast<AccumT>(input_data[i]) - mean;
        partial_var_sum += centered * centered;
      }
    }
  }

  // Reduce to get total variance
  AccumT total_var = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(
      partial_var_sum,
      shared_variance,
      thread_in_row
  );
  AccumT variance = total_var / static_cast<AccumT>(element_count);
  AccumT inv_std = rsqrt(variance + static_cast<AccumT>(epsilon));

  // Apply normalization and scaling
  for (uint base_i = thread_in_row * GRAIN_SIZE; base_i < element_count;
       base_i += BLOCK_SIZE * GRAIN_SIZE) {
    AccumT vals[GRAIN_SIZE];
    AccumT scaled_vals[GRAIN_SIZE];

    // Load and center
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      vals[j] = (i < element_count)
                    ? (static_cast<AccumT>(input_data[i]) - mean)
                    : 0.0f;
    }

    // Normalize and scale
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i >= element_count) {
        continue;
      }

      AccumT normalized_high = vals[j] * inv_std;

      if (full_layer) {
        AccumT scale_value_high = static_cast<AccumT>(scales_data[i]) +
                                  static_cast<AccumT>(scale_offset);
        scaled_vals[j] = normalized_high * scale_value_high;
      } else {
        OutputT normalized_low = static_cast<OutputT>(normalized_high);
        OutputT scale_value_low = static_cast<OutputT>(
            static_cast<AccumT>(scales_data[i]) +
            static_cast<AccumT>(scale_offset)
        );
        OutputT product_low = normalized_low * scale_value_low;
        scaled_vals[j] = static_cast<AccumT>(product_low);
      }
    }

    // Store
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i < element_count) {
        output_data[i] = static_cast<OutputT>(scaled_vals[j]);
      }
    }
  }
}

// X-macro table defining all supported type combinations (same as RMSNorm)
#define FOREACH_LAYER_NORM_COMBO(_)                                            \
  /* bfloat combinations most commonly used */                                 \
  _(bfloat, bfloat, bfloat, float, false, bf16_bf16_bf16_f32_norm)             \
  _(bfloat, bfloat, bfloat, float, true, bf16_bf16_bf16_f32_full)              \
  _(bfloat, float, bfloat, float, false, bf16_f32_bf16_f32_norm)               \
  _(bfloat, float, bfloat, float, true, bf16_f32_bf16_f32_full)                \
  /* float combinations */                                                     \
  _(float, float, float, float, false, f32_f32_f32_f32_norm)                   \
  _(float, float, float, float, true, f32_f32_f32_f32_full)                    \
  /* half combinations */                                                      \
  _(half, half, half, float, false, f16_f16_f16_f32_norm)                      \
  _(half, half, half, float, true, f16_f16_f16_f32_full)

// Generate LayerNorm kernels
#define DEFINE_LAYER_NORM_KERNEL(IN, SC, OUT, ACC, FULL_LAYER, SUF)            \
  kernel void layer_norm_##SUF(                                                \
      const device IN* input [[buffer(0)]],                                    \
      const device SC* scales [[buffer(1)]],                                   \
      device OUT* output [[buffer(2)]],                                        \
      constant uint& batch_size [[buffer(3)]],                                 \
      constant uint& model_dim [[buffer(4)]],                                  \
      constant float& epsilon [[buffer(5)]],                                   \
      constant float& scale_offset [[buffer(6)]],                              \
      uint batch_idx [[threadgroup_position_in_grid]],                         \
      uint thread_in_row [[thread_position_in_threadgroup]]                    \
  ) {                                                                          \
    if (batch_idx >= batch_size)                                               \
      return;                                                                  \
                                                                               \
    threadgroup ACC shared_mean[SIMD_SIZE];                                    \
    threadgroup ACC shared_variance[SIMD_SIZE];                                \
    const uint input_offset = batch_idx * model_dim;                           \
                                                                               \
    layer_norm_core<IN, SC, OUT, ACC>(                                         \
        input + input_offset,                                                  \
        scales,                                                                \
        output + input_offset,                                                 \
        model_dim,                                                             \
        epsilon,                                                               \
        scale_offset,                                                          \
        shared_mean,                                                           \
        shared_variance,                                                       \
        thread_in_row,                                                         \
        FULL_LAYER                                                             \
    );                                                                         \
  }

FOREACH_LAYER_NORM_COMBO(DEFINE_LAYER_NORM_KERNEL)
#undef DEFINE_LAYER_NORM_KERNEL
