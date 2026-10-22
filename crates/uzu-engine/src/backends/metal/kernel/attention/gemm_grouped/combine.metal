#include <metal_stdlib>
#include "../../common/dsl.h"
#include "../../generated/attention.h"
#include "../attention_gate.h"

using namespace metal;
using namespace uzu::attention;

template <typename T, uint BD>
VARIANTS(T, bfloat)
VARIANTS(BD, 128, 256)
KERNEL(AttentionGemmGroupedCombine)(
    const device T* partials,
    const device float* partial_maxs,
    const device float* partial_sums,
    device T* o,
    const device T* gate OPTIONAL(has_gate),
    const constant AttnParams& params,
    const constant uint& grouped_rows,
    const constant uint& padded_rows,
    const constant uint& kv_heads,
    const constant uint& num_splits,
    const constant uint& gate_row_stride OPTIONAL(has_gate),
    const bool has_gate SPECIALIZE,
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
  const device T* row_partials = partials + row * BD + column;

  float numerator = 0.0f;
  float denominator = 0.0f;
  float global_max = -INFINITY;
  for (uint s = 0; s < num_splits; ++s) {
    if (row_sums[s * split_rows] > 0.0f) {
      global_max = metal::max(global_max, row_maxs[s * split_rows]);
    }
  }
  if (global_max > -INFINITY) {
    for (uint s = 0; s < num_splits; ++s) {
      const float split_sum = row_sums[s * split_rows];
      if (split_sum <= 0.0f) {
        continue;
      }
      const float rescale = fast::exp2(row_maxs[s * split_rows] - global_max);
      const float weight = rescale * split_sum;
      denominator += weight;
      numerator += rescale * float(row_partials[size_t(s) * split_rows * BD]);
    }
  }
  const float mean = denominator > 0.0f ? numerator / denominator : 0.0f;
  T result = static_cast<T>(mean);
  if (has_gate) {
    result = attention_apply_gate(result, gate[query_position * gate_row_stride + q_head_idx * BD + column]);
  }
  o += size_t(q_head_idx) * params.o_strides[1] + size_t(query_position) * params.o_strides[2] + column;
  o[0] = result;
}
