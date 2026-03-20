#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"

using namespace metal;

#define BLOCK_SIZE 1024
#define GRAIN_SIZE 4

// Fused RMSNorm + Hadamard transform using threadgroup memory staging.
// Phase 1: Standard RMSNorm reduction + normalization, writing to threadgroup memory.
// Phase 2: SIMD groups read from threadgroup memory, apply Hadamard, write to device.
// No device memory barrier needed -- data passes through threadgroup memory only.
template <typename InputT, typename ScaleT, typename OutputT, typename AccumT>
VARIANTS(InputT, float, half, bfloat)
VARIANTS(ScaleT, float, half, bfloat)
VARIANTS(OutputT, float, half, bfloat)
VARIANTS(AccumT, float, half)
PUBLIC KERNEL(RMSNormHadamardMul)(
    const device InputT* input OPTIONAL(!in_place),
    const device ScaleT* scales,
    device OutputT* output,
    const device OutputT* hadamard_factors,
    constant uint& batch_size,
    constant uint& element_count,
    constant float& epsilon,
    constant float& scale_offset,
    constant bool& full_layer,
    const bool in_place SPECIALIZE,
    threadgroup AccumT shared_sum[METAL_SIMD_SIZE],
    threadgroup float staging[4096],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(1024)
) {
  if (in_place) {
    input = reinterpret_cast<const device InputT*>(output);
  }

  const uint input_offset = batch_idx * element_count;
  const device InputT* input_data = input + input_offset;
  const device ScaleT* scales_data = scales;
  device OutputT* output_data = output + input_offset;

  // ── Phase 1: RMSNorm reduction ────────────────────────────────────

  AccumT partial_sum = static_cast<AccumT>(0.0f);

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

  AccumT total_sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(
      partial_sum, shared_sum, thread_in_row, thread_context);

  AccumT mean_square =
      static_cast<AccumT>(total_sum) / static_cast<AccumT>(element_count);
  AccumT rms_norm = rsqrt(mean_square + static_cast<AccumT>(epsilon));

  // ── Phase 1b: Normalize + scale, write to threadgroup staging ─────

  for (uint base_i = thread_in_row * GRAIN_SIZE; base_i < element_count;
       base_i += BLOCK_SIZE * GRAIN_SIZE) {
    AccumT vals[GRAIN_SIZE];

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      vals[j] = (i < element_count) ? static_cast<AccumT>(input_data[i]) : 0.0f;
    }

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i >= element_count) continue;

      AccumT normalized_high = vals[j] * rms_norm;
      float result;

      if (full_layer) {
        AccumT scale_value_high = static_cast<AccumT>(scales_data[i]) +
                                  static_cast<AccumT>(scale_offset);
        result = float(normalized_high * scale_value_high);
      } else {
        OutputT normalized_low = static_cast<OutputT>(normalized_high);
        OutputT scale_value_low = static_cast<OutputT>(
            static_cast<AccumT>(scales_data[i]) +
            static_cast<AccumT>(scale_offset));
        result = float(normalized_low * scale_value_low);
      }

      staging[i] = result;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ── Phase 2: Hadamard transform from threadgroup memory ───────────

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
    output_data[elem_idx] = OutputT(value * normalization_factor);
  }
}
