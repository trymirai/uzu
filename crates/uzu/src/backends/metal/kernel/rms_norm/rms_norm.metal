#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

#define BLOCK_SIZE 1024
#define SIMD_SIZE 32
#define GRAIN_SIZE 4

template <typename InputT, typename ScaleT, typename OutputT, typename AccumT>
VARIANTS(InputT, float, half, bfloat)
VARIANTS(ScaleT, float, half, bfloat)
VARIANTS(OutputT, float, half, bfloat)
VARIANTS(AccumT, float, half)
KERNEL(RMSNorm)(
    const device InputT* input,
    const device ScaleT* scales,
    device OutputT* output,
    constant uint& batch_size,
    constant uint& element_count,
    constant float& epsilon,
    constant float& scale_offset,
    constant bool& full_layer,
    threadgroup AccumT shared_sum[SIMD_SIZE],
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(1024)
) {
  const uint input_offset = batch_idx * element_count;
  const device InputT* input_data = input + input_offset;
  const device ScaleT* scales_data = scales;
  device OutputT* output_data = output + input_offset;

  AccumT partial_sum = static_cast<AccumT>(0.0f);

  // Compute thread local partial sum
  for (uint base_i = thread_in_row * GRAIN_SIZE; base_i < element_count;
       base_i += BLOCK_SIZE * GRAIN_SIZE) {
    AccumT vals[GRAIN_SIZE];

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      vals[j] = (i < element_count) ? static_cast<AccumT>(input_data[i]) : 0.0f;
    }

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      partial_sum += vals[j] * vals[j];
    }
  }

  // Compute total sum across threadgroup
  AccumT total_sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(
      partial_sum,
      shared_sum,
      thread_in_row
  );

  // Compute RMS norm factor
  AccumT mean_square =
      static_cast<AccumT>(total_sum) / static_cast<AccumT>(element_count);
  AccumT rms_norm = rsqrt(mean_square + static_cast<AccumT>(epsilon));

  // Apply normalization and scaling using the same vectorized pattern
  for (uint base_i = thread_in_row * GRAIN_SIZE; base_i < element_count;
       base_i += BLOCK_SIZE * GRAIN_SIZE) {
    AccumT vals[GRAIN_SIZE];
    AccumT scaled_vals[GRAIN_SIZE];

    // Load GRAIN_SIZE input elements
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      vals[j] = (i < element_count) ? static_cast<AccumT>(input_data[i]) : 0.0f;
    }

    // Process GRAIN_SIZE elements: normalize and scale
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i >= element_count) {
        continue;
      }

      AccumT normalized_high = vals[j] * rms_norm;

      if (full_layer) {
        // Full-layer: keep everything in accumulation precision
        AccumT scale_value_high = static_cast<AccumT>(scales_data[i]) +
                                  static_cast<AccumT>(scale_offset);
        scaled_vals[j] = normalized_high * scale_value_high;
      } else {
        // Only-normalization: cast down for the scale multiply
        OutputT normalized_low = static_cast<OutputT>(normalized_high);
        OutputT scale_value_low = static_cast<OutputT>(
            static_cast<AccumT>(scales_data[i]) +
            static_cast<AccumT>(scale_offset)
        );
        OutputT product_low = normalized_low * scale_value_low;
        scaled_vals[j] = static_cast<AccumT>(product_low);
      }
    }

    // Store GRAIN_SIZE output elements
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i < element_count) {
        output_data[i] = static_cast<OutputT>(scaled_vals[j]);
      }
    }
  }
}
