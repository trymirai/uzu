#include <metal_stdlib>
#include "../../common/dsl.h"
#include "../../generated/attention.h"
#include "../../hadamard_transform/hadamard_transform.h"
#include "../attention_gate.h"
#include "../kv_layout.h"

using namespace metal;

template <typename T, uint BD>
VARIANTS(T, bfloat)
VARIANTS(BD, 256)
KERNEL(AttentionGemmGroupedInt8Combine)(
    const device bfloat* partials,
    const device float* partial_maxs,
    const device float* partial_sums,
    device T* o,
    const device T* gate OPTIONAL(has_gate),
    const constant uzu::attention::AttnParams& params,
    const constant uint& grouped_rows,
    const constant uint& padded_rows,
    const constant uint& kv_heads,
    const constant uint& num_splits,
    const constant uint& gate_row_stride OPTIONAL(has_gate),
    const bool has_gate SPECIALIZE,
    threadgroup float rotation[BD],
    const uint grouped_row_idx GROUPS(grouped_rows),
    const uint kv_head_idx GROUPS(kv_heads),
    const uint column THREADS(BD)
) {
  const size_t row = size_t(kv_head_idx) * padded_rows + grouped_row_idx;
  const uint split_rows = kv_heads * padded_rows;
  const uint head_in_group = grouped_row_idx / params.q_len;
  const uint query_position = grouped_row_idx - head_in_group * params.q_len;
  const uint q_head_idx = kv_head_idx * params.gqa_factor + head_in_group;
  const device float* row_maxs = partial_maxs + row;
  const device float* row_sums = partial_sums + row;
  const device bfloat* row_means = partials + row * BD + column;

  float numerator = 0.0f;
  float denominator = 0.0f;
  constexpr uint SIMDGROUPS = BD / METAL_SIMD_SIZE;
  float global_max = -metal::numeric_limits<float>::infinity();
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
  for (uint chunk_start = 0; chunk_start < num_splits; chunk_start += BD) {
    const uint split = chunk_start + column;
    float weight = 0.0f;
    if (split < num_splits && row_sums[split * split_rows] > 0.0f) {
      weight = fast::exp2(row_maxs[split * split_rows] - global_max) * row_sums[split * split_rows];
    }
    split_weights[column] = weight;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint chunk_size = metal::min(num_splits - chunk_start, uint(BD));
    _Pragma("clang loop unroll_count(8)") for (uint i = 0; i < chunk_size; ++i) {
      const float split_weight = split_weights[i];
      denominator += split_weight;
      numerator += split_weight * float(row_means[size_t(chunk_start + i) * split_rows * BD]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float mean = denominator > 0.0f ? numerator / denominator : 0.0f;

  rotation[column] = mean;
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
  simdgroup_hadamard_transform<VALUES_PER_LANE, METAL_SIMD_SIZE>(ushort(column), rotated);
  device T* output_row = o + size_t(q_head_idx) * params.o_strides[1] + size_t(query_position) * params.o_strides[2];
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    const uint channel = value_index * METAL_SIMD_SIZE + column;
    T result = static_cast<T>(rotated[value_index] * uzu::kv_layout::rotation_sign(channel));
    if (has_gate) {
      result = attention_apply_gate(result, gate[query_position * gate_row_stride + q_head_idx * BD + channel]);
    }
    output_row[channel] = result;
  }
}
