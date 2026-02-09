#include <metal_stdlib>
#include "../definitions.metal"

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

template <typename IN, typename SC, typename OUT, typename ACC>
VARIANTS(IN, float, half, bfloat)
VARIANTS(SC, float, half, bfloat)
VARIANTS(OUT, float, half, bfloat)
VARIANTS(ACC, float)
KERNEL(LayerNorm) (
    const device IN* input,
    const device SC* scales,
    device OUT* output,
    constant uint& batch_size,
    constant uint& model_dim,
    constant float& epsilon,
    constant float& scale_offset,
    constant uint& full_layer,
    threadgroup ACC shared_mean[SIMD_SIZE],
    threadgroup ACC shared_variance[SIMD_SIZE],
    uint batch_idx GROUPS(batch_size),
    uint thread_in_row THREADS(BLOCK_SIZE)
) {
  const uint input_offset = batch_idx * model_dim;
  layer_norm_core<IN, SC, OUT, ACC>(
    input + input_offset,
    scales,
    output + input_offset,
    model_dim,
    epsilon,
    scale_offset,
    shared_mean,
    shared_variance,
    thread_in_row,
    full_layer
  );
}
