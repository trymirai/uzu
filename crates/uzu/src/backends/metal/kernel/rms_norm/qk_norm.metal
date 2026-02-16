#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

#define SIMD_SIZE 32
#define GRAIN_SIZE 4

// QK norm: normalize per-head vectors (small head_dim) efficiently.
//
// Strategy:
// - One SIMD-group (32 threads) processes one head.
// - One threadgroup (one SIMD-group) is dispatched per head.
template <typename InputT, typename ScaleT, typename OutputT, typename AccumT>
VARIANTS(InputT, float, half, bfloat)
VARIANTS(ScaleT, float, half, bfloat)
VARIANTS(OutputT, float, half, bfloat)
VARIANTS(AccumT, float, half)
KERNEL(QKNorm)(
    const device InputT* qkv_input,
    const device ScaleT* scales,
    device OutputT* qkv_output,
    constant uint& batch_size,
    constant uint& num_q_heads,
    constant uint& num_kv_heads,
    constant uint& head_dim,
    constant float& epsilon,
    constant float& scale_offset,
    constant uint& head_offset,
    constant uint& head_count,
    constant bool& full_layer,
    const uint batch_idx GROUPS(batch_size),
    const uint head_idx GROUPS(head_count),
    const uint lane_id THREADS(SIMD_SIZE)
) {
  if (head_count == 0u || head_dim == 0u)
    return;

  const uint total_heads_in_buffer = num_q_heads + 2u * num_kv_heads;
  const uint logical_head_idx = head_offset + head_idx;
  if (logical_head_idx >= total_heads_in_buffer)
    return;

  const ulong slice_offset =
      (ulong)batch_idx * (ulong)total_heads_in_buffer * (ulong)head_dim +
      (ulong)logical_head_idx * (ulong)head_dim;

  const device InputT* input_data = qkv_input + slice_offset;
  const device ScaleT* scales_data = scales;
  device OutputT* output_data = qkv_output + slice_offset;
  const uint element_count = head_dim;

  AccumT partial_sum = static_cast<AccumT>(0.0f);

  // Sum of squares: each lane processes GRAIN_SIZE elements per iteration.
  for (uint base_i = lane_id * GRAIN_SIZE; base_i < element_count;
       base_i += SIMD_SIZE * GRAIN_SIZE) {
    AccumT vals[GRAIN_SIZE];
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      vals[j] = (i < element_count) ? static_cast<AccumT>(input_data[i]) : 0.0f;
    }
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      partial_sum += vals[j] * vals[j];
    }
  }

  // SIMD-group reduction.
  AccumT total_sum = simd_sum(partial_sum);

  // RMS factor.
  AccumT mean_square =
      static_cast<AccumT>(total_sum) / static_cast<AccumT>(element_count);
  AccumT rms_norm = rsqrt(mean_square + static_cast<AccumT>(epsilon));

  // Normalize + scale.
  for (uint base_i = lane_id * GRAIN_SIZE; base_i < element_count;
       base_i += SIMD_SIZE * GRAIN_SIZE) {
    AccumT vals[GRAIN_SIZE];
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      vals[j] = (i < element_count) ? static_cast<AccumT>(input_data[i]) : 0.0f;
    }

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i >= element_count)
        continue;

      AccumT normalized_high = vals[j] * rms_norm;

      if (full_layer) {
        AccumT scale_value_high = static_cast<AccumT>(scales_data[i]) +
                                  static_cast<AccumT>(scale_offset);
        output_data[i] =
            static_cast<OutputT>(normalized_high * scale_value_high);
      } else {
        OutputT normalized_low = static_cast<OutputT>(normalized_high);
        OutputT scale_value_low = static_cast<OutputT>(
            static_cast<AccumT>(scales_data[i]) +
            static_cast<AccumT>(scale_offset)
        );
        OutputT product_low = normalized_low * scale_value_low;
        output_data[i] = static_cast<OutputT>(product_low);
      }
    }
  }
}
