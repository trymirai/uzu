#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/activation_transform.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;
using namespace uzu::activation_transform;

UZU_CONST uint ACTIVATION_TILE_SIZE = 128;
UZU_CONST uint SIMDGROUPS_PER_THREADGROUP = ACTIVATION_TILE_SIZE / METAL_SIMD_SIZE;
UZU_CONST float INT8_QMAX = 127.0f;

#define QUANTIZED (ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums)
#define EMITS_GROUP_SUMS (ops == ActivationTransformOp::QuantizeWithGroupSums)

template <typename T, typename SimdReduce, typename Combine>
METAL_FUNC T reduce_quantization_group(
    const T value,
    const uint group_size,
    threadgroup T* partials,
    const thread ThreadContext& thread_context,
    SimdReduce simd_reduce,
    Combine combine
) {
  const ushort lane_index = thread_context.simd_lane_id;
  const ushort simdgroup_index = thread_context.simdgroup_index;
  const T simdgroup_value = simd_reduce(value);
  if (group_size == METAL_SIMD_SIZE) {
    return simdgroup_value;
  }
  if (lane_index == 0) {
    partials[simdgroup_index] = simdgroup_value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint simdgroups_per_group = group_size / METAL_SIMD_SIZE;
  const uint simdgroup_base = simdgroup_index / simdgroups_per_group * simdgroups_per_group;
  T result = partials[simdgroup_base];
  METAL_PRAGMA_UNROLL
  for (uint index = 1; index < simdgroups_per_group; ++index) {
    result = combine(result, partials[simdgroup_base + index]);
  }
  return result;
}

template <typename T>
METAL_FUNC void write_quantization_group(
    device T* output,
    const T value,
    const uint group_size,
    const uint element_count,
    const uint tile_index,
    const uint batch_index,
    const thread ThreadContext& thread_context
) {
  const ushort simdgroup_index = thread_context.simdgroup_index;
  const uint simdgroups_per_group = group_size / METAL_SIMD_SIZE;
  const uint simdgroup_offset = tile_index * ACTIVATION_TILE_SIZE + simdgroup_index * METAL_SIMD_SIZE;
  if (thread_context.simd_lane_id == 0 && simdgroup_offset < element_count &&
      (simdgroup_index + 1) % simdgroups_per_group == 0) {
    const uint groups_per_row = element_count / group_size;
    const uint groups_per_tile = ACTIVATION_TILE_SIZE / group_size;
    const uint group_index = tile_index * groups_per_tile + simdgroup_index / simdgroups_per_group;
    output[batch_index * groups_per_row + group_index] = value;
  }
}

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(ActivationTransform)(
    const device T* input OPTIONAL(!in_place),
    device T* fp_out OPTIONAL(!QUANTIZED),
    device int8_t* q_out OPTIONAL(QUANTIZED),
    device float* scales_out OPTIONAL(QUANTIZED),
    device int32_t* group_sums_out OPTIONAL(EMITS_GROUP_SUMS),
    const device int32_t* rht_factors,
    constant uint& batch_size,
    constant uint& element_count,
    const ActivationTransformOp ops SPECIALIZE,
    const bool in_place SPECIALIZE,
    const uint activation_scale_group_size SPECIALIZE,
    const uint correction_group_size SPECIALIZE,
    threadgroup float partial_max OPTIONAL(QUANTIZED && activation_scale_group_size > METAL_SIMD_SIZE)[4],
    threadgroup int partial_sums OPTIONAL(EMITS_GROUP_SUMS && correction_group_size > METAL_SIMD_SIZE)[4],
    uint activation_tile_index GROUPS(element_count.div_ceil(128)),
    uint batch_index GROUPS(batch_size),
    uint thread_index THREADS(128),
    const ThreadContext thread_context
) {
  (void)thread_index;
  if (in_place) {
    input = reinterpret_cast<const device T*>(fp_out);
  }

  const bool input_rht = ops != ActivationTransformOp::OutputRht;
  const ushort lane_index = thread_context.simd_lane_id;
  const ushort simdgroup_index = thread_context.simdgroup_index;
  const uint tile_offset = activation_tile_index * ACTIVATION_TILE_SIZE;
  const uint simdgroup_offset = tile_offset + simdgroup_index * METAL_SIMD_SIZE;
  const bool in_bounds = simdgroup_offset < element_count;
  const uint factor_index = simdgroup_offset + lane_index;
  const uint element_index = batch_index * element_count + factor_index;

  float value = 0.0f;
  if (in_bounds) {
    value = static_cast<float>(input[element_index]);
    value = input_rht ? simdgroup_input_random_hadamard_transform(lane_index, value, rht_factors[factor_index])
                      : simdgroup_output_random_hadamard_transform(lane_index, value, rht_factors[factor_index]);
  }

  if (!QUANTIZED) {
    if (in_bounds) {
      fp_out[element_index] = static_cast<T>(value);
    }
    return;
  }

  const float maximum = reduce_quantization_group(
      fabs(value),
      activation_scale_group_size,
      partial_max,
      thread_context,
      [](float x) { return simd_max(x); },
      [](float x, float y) { return max(x, y); }
  );
  const float scale = isfinite(maximum) && maximum > 0.0f ? maximum / INT8_QMAX : 1.0f;

  const int8_t code = static_cast<int8_t>(clamp(round(value / scale), -INT8_QMAX, INT8_QMAX));
  if (in_bounds) {
    q_out[element_index] = code;
  }
  write_quantization_group(
      scales_out,
      scale,
      activation_scale_group_size,
      element_count,
      activation_tile_index,
      batch_index,
      thread_context
  );

  if (EMITS_GROUP_SUMS) {
    const int sum = reduce_quantization_group(
        int(code),
        correction_group_size,
        partial_sums,
        thread_context,
        [](int x) { return simd_sum(x); },
        [](int x, int y) { return x + y; }
    );
    write_quantization_group(
        group_sums_out,
        sum,
        correction_group_size,
        element_count,
        activation_tile_index,
        batch_index,
        thread_context
    );
  }
}
