#include <metal_stdlib>
#include "../common/dsl.h"
#include "../generated/attention.h"
#include "rotated_output.h"

using namespace metal;

template <typename T, uint BD>
VARIANTS(T, bfloat, float)
VARIANTS(BD, 64, 128, 256, 512)
KERNEL(AttentionMatrixCombine)(
    const device bfloat* partials,
    const device float* partial_maxs,
    const device float* partial_sums,
    device T* output,
    const device T* gate OPTIONAL(has_gate),
    const device T* sinks OPTIONAL(has_sinks),
    const constant uzu::attention::AttnParams& params,
    const constant uint& grouped_rows,
    const constant uint& padded_rows,
    const constant uint& kv_heads,
    const constant uint& num_splits,
    const constant uint& gate_row_stride OPTIONAL(has_gate),
    const bool has_gate SPECIALIZE,
    const bool has_sinks SPECIALIZE,
    threadgroup float rotation[2 * BD],
    const uint grouped_row_idx GROUPS(grouped_rows),
    const uint kv_head_idx GROUPS(kv_heads),
    const uint column THREADS(BD)
) {
  const size_t row = size_t(kv_head_idx) * padded_rows + grouped_row_idx;
  const uint split_rows = kv_heads * padded_rows;
  const uint q_head_idx = kv_head_idx * params.gqa_factor + grouped_row_idx / params.q_len;
  const device float* row_maxs = partial_maxs + row;
  const device float* row_sums = partial_sums + row;
  const device bfloat* row_values = partials + row * BD + column;

  float numerator = 0.0f;
  float denominator = 0.0f;
  constexpr uint SIMDGROUPS = BD / METAL_SIMD_SIZE;
  const float sink = has_sinks ? float(sinks[q_head_idx]) * M_LOG2E_F : -INFINITY;
  float global_max = sink;
  for (uint s = column; s < num_splits; s += BD) {
    if (row_sums[s * split_rows] > 0.0f) {
      global_max = metal::max(global_max, row_maxs[s * split_rows]);
    }
  }
  global_max = simd_max(global_max);
  if (column % METAL_SIMD_SIZE == 0) {
    rotation[column / METAL_SIMD_SIZE] = global_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  global_max = rotation[0];
  for (uint simdgroup = 1; simdgroup < SIMDGROUPS; ++simdgroup) {
    global_max = metal::max(global_max, rotation[simdgroup]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float* split_weights = rotation;
  threadgroup float* split_denominators = rotation + BD;
  for (uint chunk_start = 0; chunk_start < num_splits; chunk_start += BD) {
    const uint split = chunk_start + column;
    float weight = 0.0f;
    float scaled_denominator = 0.0f;
    if (split < num_splits && row_sums[split * split_rows] > 0.0f) {
      weight = fast::exp2(row_maxs[split * split_rows] - global_max);
      scaled_denominator = weight * row_sums[split * split_rows];
    }
    split_weights[column] = weight;
    split_denominators[column] = scaled_denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint chunk_size = metal::min(num_splits - chunk_start, uint(BD));
    _Pragma("clang loop unroll_count(8)") for (uint i = 0; i < chunk_size; ++i) {
      const float split_weight = split_weights[i];
      denominator += split_denominators[i];
      numerator += split_weight * float(row_values[size_t(chunk_start + i) * split_rows * BD]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  rotation[column] = numerator;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (column >= METAL_SIMD_SIZE) {
    return;
  }
  constexpr ushort VALUES_PER_LANE = BD / METAL_SIMD_SIZE;
  float rotated[VALUES_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    rotated[value_index] = rotation[value_index * METAL_SIMD_SIZE + column];
  }
  store_output_row<T, BD>(
      ushort(column),
      rotated,
      global_max,
      denominator,
      sinks,
      has_sinks,
      output,
      gate,
      params,
      kv_head_idx,
      grouped_row_idx,
      gate_row_stride,
      has_gate
  );
}
