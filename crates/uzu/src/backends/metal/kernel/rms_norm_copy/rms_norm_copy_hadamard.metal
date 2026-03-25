#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"

using namespace metal;

#define BLOCK_SIZE 1024
#define GRAIN_SIZE 4
#define STAGING_SIZE (BLOCK_SIZE * GRAIN_SIZE * 2)

template <typename ScaleT, typename DataT, typename AccumT>
VARIANTS(ScaleT, float, half, bfloat)
VARIANTS(DataT, float, half, bfloat)
VARIANTS(AccumT, float, half)
PUBLIC KERNEL(RMSNormCopyHadamardMul)(
    device DataT* main_buffer,
    device DataT* shortcut_buffer,
    const device ScaleT* scales,
    const device DataT* hadamard_factors,
    constant uint& batch_size,
    constant uint& element_count,
    constant float& epsilon,
    constant float& scale_offset,
    constant bool& full_layer,
    threadgroup float staging[STAGING_SIZE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(1024)
) {
  threadgroup AccumT* shared_sum =
      reinterpret_cast<threadgroup AccumT*>(&staging[STAGING_SIZE - METAL_SIMD_SIZE]);

  const uint offset = batch_idx * element_count;
  device DataT* main_data = main_buffer + offset;
  device DataT* shortcut_data = shortcut_buffer + offset;
  const device ScaleT* scales_data = scales;

  AccumT partial_sum = static_cast<AccumT>(0.0f);

  // Pass 1: sum-of-squares + copy to shortcut + cache to staging
  for (uint base_i = thread_in_row * GRAIN_SIZE; base_i < element_count;
       base_i += BLOCK_SIZE * GRAIN_SIZE) {
    AccumT vals[GRAIN_SIZE];

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i < element_count) {
        DataT raw = main_data[i];
        vals[j] = static_cast<AccumT>(raw);
        shortcut_data[i] = raw;
        staging[i] = float(vals[j]);
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
  AccumT rms_norm_val = rsqrt(mean_square + static_cast<AccumT>(epsilon));

  // Pass 2: normalize + scale, write to staging
  for (uint base_i = thread_in_row * GRAIN_SIZE; base_i < element_count;
       base_i += BLOCK_SIZE * GRAIN_SIZE) {
    AccumT vals[GRAIN_SIZE];

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      vals[j] = (i < element_count) ? static_cast<AccumT>(staging[i]) : 0.0f;
    }

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i >= element_count)
        continue;

      AccumT normalized_high = vals[j] * rms_norm_val;
      float result;

      if (full_layer) {
        AccumT scale_value_high = static_cast<AccumT>(scales_data[i]) +
                                  static_cast<AccumT>(scale_offset);
        result = float(normalized_high * scale_value_high);
      } else {
        DataT normalized_low = static_cast<DataT>(normalized_high);
        DataT scale_value_low = static_cast<DataT>(
            static_cast<AccumT>(scales_data[i]) +
            static_cast<AccumT>(scale_offset)
        );
        result = float(normalized_low * scale_value_low);
      }

      staging[i] = result;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Pass 3: Hadamard transform from staging to main
  const uint lane = thread_in_row % METAL_SIMD_SIZE;
  const uint simd_group_id = thread_in_row / METAL_SIMD_SIZE;
  const uint total_simd_groups = BLOCK_SIZE / METAL_SIMD_SIZE;
  const uint total_blocks = element_count / METAL_SIMD_SIZE;

  for (uint block = simd_group_id; block < total_blocks;
       block += total_simd_groups) {
    uint elem_idx = block * METAL_SIMD_SIZE + lane;

    float value = staging[elem_idx] * float(hadamard_factors[elem_idx]);

    for (uint stride = 1; stride < METAL_SIMD_SIZE; stride <<= 1) {
      float partner = simd_shuffle_xor(value, static_cast<ushort>(stride));
      value = (lane & stride) ? (partner - value) : (partner + value);
    }

    constexpr float normalization_factor = 1.0f / 5.656854249f;
    main_data[elem_idx] = DataT(value * normalization_factor);
  }
}
