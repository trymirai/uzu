#pragma once

#include "../common/sigmoid_gate.h"
#include "../generated/attention.h"
#include "../hadamard_transform/hadamard_transform.h"
#include "kv_layout.h"

template <typename T, ushort HEAD_DIM>
METAL_FUNC void store_output_row(
    const ushort lane,
    thread float* rotated,
    device T* output,
    const device T* gate,
    const constant uzu::attention::AttnParams& params,
    const uint kv_head_index,
    const uint grouped_row,
    const uint gate_row_stride,
    const bool has_gate
) {
  constexpr ushort VALUES_PER_LANE = HEAD_DIM / METAL_SIMD_SIZE;
  const uint head_in_group = grouped_row / params.q_len;
  const uint query_position = grouped_row - head_in_group * params.q_len;
  const uint q_head_index = kv_head_index * params.gqa_factor + head_in_group;
  device T* output_row =
      output + size_t(q_head_index) * params.o_strides[1] + size_t(query_position) * params.o_strides[2];
  const device T* gate_row = gate + query_position * gate_row_stride + q_head_index * HEAD_DIM;
  hadamard::simdgroup_transform<VALUES_PER_LANE, METAL_SIMD_SIZE>(lane, rotated);
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    const uint channel = value_index * METAL_SIMD_SIZE + lane;
    T result = static_cast<T>(rotated[value_index] * uzu::kv_layout::rotation_sign(channel));
    if (has_gate) {
      result = uzu::apply_sigmoid_gate(result, gate_row[channel]);
    }
    output_row[channel] = result;
  }
}
