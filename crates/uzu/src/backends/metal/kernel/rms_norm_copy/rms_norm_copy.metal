#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"

using namespace metal;

#define BLOCK_SIZE 1024
#define GRAIN_SIZE 4

// Fused TensorCopy + RMSNorm: reads Main once, writes unchanged copy to
// Shortcut during the reduction pass, then writes normalized result to Main.
// Eliminates one separate TensorCopy dispatch per layer.
template <typename ScaleT, typename DataT, typename AccumT>
VARIANTS(ScaleT, float, half, bfloat)
VARIANTS(DataT, float, half, bfloat)
VARIANTS(AccumT, float, half)
PUBLIC KERNEL(RMSNormCopy)(
    device DataT* main_buffer,
    device DataT* shortcut_buffer,
    const device ScaleT* scales,
    constant uint& batch_size,
    constant uint& element_count,
    constant float& epsilon,
    constant float& scale_offset,
    constant bool& full_layer,
    threadgroup AccumT shared_sum[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(1024)
) {
  const uint offset = batch_idx * element_count;
  device DataT* main_data = main_buffer + offset;
  device DataT* shortcut_data = shortcut_buffer + offset;
  const device ScaleT* scales_data = scales;

  AccumT partial_sum = static_cast<AccumT>(0.0f);

  // Pass 1: sum-of-squares + copy to shortcut
  for (uint base_i = thread_in_row * GRAIN_SIZE; base_i < element_count;
       base_i += BLOCK_SIZE * GRAIN_SIZE) {
    AccumT vals[GRAIN_SIZE];

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i < element_count) {
        DataT raw = main_data[i];
        vals[j] = static_cast<AccumT>(raw);
        shortcut_data[i] = raw;
      } else {
        vals[j] = 0.0f;
      }
    }

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      partial_sum += vals[j] * vals[j];
    }
  }

  AccumT total_sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(
      partial_sum,
      shared_sum,
      thread_in_row,
      thread_context
  );

  AccumT mean_square =
      static_cast<AccumT>(total_sum) / static_cast<AccumT>(element_count);
  AccumT rms_norm = rsqrt(mean_square + static_cast<AccumT>(epsilon));

  // Pass 2: normalize + scale, write to main
  for (uint base_i = thread_in_row * GRAIN_SIZE; base_i < element_count;
       base_i += BLOCK_SIZE * GRAIN_SIZE) {
    AccumT vals[GRAIN_SIZE];
    AccumT scaled_vals[GRAIN_SIZE];

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      vals[j] = (i < element_count) ? static_cast<AccumT>(main_data[i]) : 0.0f;
    }

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i >= element_count)
        continue;

      AccumT normalized_high = vals[j] * rms_norm;

      if (full_layer) {
        AccumT scale_value_high = static_cast<AccumT>(scales_data[i]) +
                                  static_cast<AccumT>(scale_offset);
        scaled_vals[j] = normalized_high * scale_value_high;
      } else {
        DataT normalized_low = static_cast<DataT>(normalized_high);
        DataT scale_value_low = static_cast<DataT>(
            static_cast<AccumT>(scales_data[i]) +
            static_cast<AccumT>(scale_offset)
        );
        DataT product_low = normalized_low * scale_value_low;
        scaled_vals[j] = static_cast<AccumT>(product_low);
      }
    }

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i < element_count) {
        main_data[i] = static_cast<DataT>(scaled_vals[j]);
      }
    }
  }
}
