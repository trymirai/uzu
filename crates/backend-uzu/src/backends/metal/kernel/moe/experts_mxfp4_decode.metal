#include <metal_stdlib>
#include <metal_simdgroup>
#include "../activation/activations.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
using namespace metal;

constant float MXFP4_VALUES[16] = {
    0.0f,
    0.5f,
    1.0f,
    1.5f,
    2.0f,
    3.0f,
    4.0f,
    6.0f,
    -0.0f,
    -0.5f,
    -1.0f,
    -1.5f,
    -2.0f,
    -3.0f,
    -4.0f,
    -6.0f,
};

/// E8M0 is an IEEE-754 exponent byte with an implicit unit mantissa.
inline float e8m0_scale(const uchar exponent) {
  const uint bits = exponent == 0 ? 0x00400000u : uint(exponent) << 23;
  return as_type<float>(bits);
}

/// Tail-only scalar access for fixture dimensions smaller than one MXFP4 group.
inline float mxfp4_value(
    device const uchar* blocks,
    device const uchar* scales,
    const ulong row,
    const uint groups_per_row,
    const uint group,
    const uint element_in_group
) {
  const ulong group_offset = row * (ulong)groups_per_row + group;
  const uchar packed = blocks[group_offset * 16 + element_in_group / 2];
  const uchar code = element_in_group % 2 == 0 ? packed & 0x0f : packed >> 4;
  return e8m0_scale(scales[group_offset]) * MXFP4_VALUES[code];
}

/// Two lanes consume one 32-value group without redundantly loading its scale 16 times.
inline float dot_mxfp4_16(
    device const uchar* blocks,
    device const uchar* scales,
    threadgroup const float* values,
    const ulong row,
    const uint groups_per_row,
    const uint group,
    const uint half_group,
    const float4 x0,
    const float4 x1,
    const float4 x2,
    const float4 x3
) {
  const ulong block_offset = (row * (ulong)groups_per_row + group) * 16;
  device const uchar* packed = blocks + block_offset + half_group * 8;
  const float scale = e8m0_scale(scales[row * (ulong)groups_per_row + group]);

  const float4 even0 = float4(x0.x, x0.z, x1.x, x1.z);
  const float4 odd0 = float4(x0.y, x0.w, x1.y, x1.w);
  const float4 even1 = float4(x2.x, x2.z, x3.x, x3.z);
  const float4 odd1 = float4(x2.y, x2.w, x3.y, x3.w);
  const float4 weights_even0 = float4(
      values[packed[0] & 0x0f],
      values[packed[1] & 0x0f],
      values[packed[2] & 0x0f],
      values[packed[3] & 0x0f]
  );
  const float4 weights_odd0 = float4(
      values[packed[0] >> 4],
      values[packed[1] >> 4],
      values[packed[2] >> 4],
      values[packed[3] >> 4]
  );
  const float4 weights_even1 = float4(
      values[packed[4] & 0x0f],
      values[packed[5] & 0x0f],
      values[packed[6] & 0x0f],
      values[packed[7] & 0x0f]
  );
  const float4 weights_odd1 = float4(
      values[packed[4] >> 4],
      values[packed[5] >> 4],
      values[packed[6] >> 4],
      values[packed[7] >> 4]
  );
  return scale *
         (dot(even0, weights_even0) + dot(odd0, weights_odd0) + dot(even1, weights_even1) + dot(odd1, weights_odd1));
}

/// One lane consumes one complete 16-value group.
inline float dot_mxfp4_group16(
    device const uchar* blocks,
    device const uchar* scales,
    threadgroup const float* values,
    const ulong row,
    const uint groups_per_row,
    const uint group,
    const float4 x0,
    const float4 x1,
    const float4 x2,
    const float4 x3
) {
  const ulong group_offset = row * (ulong)groups_per_row + group;
  device const uchar* packed = blocks + group_offset * 8;
  const float scale = e8m0_scale(scales[group_offset]);

  const float4 even0 = float4(x0.x, x0.z, x1.x, x1.z);
  const float4 odd0 = float4(x0.y, x0.w, x1.y, x1.w);
  const float4 even1 = float4(x2.x, x2.z, x3.x, x3.z);
  const float4 odd1 = float4(x2.y, x2.w, x3.y, x3.w);
  const float4 weights_even0 = float4(
      values[packed[0] & 0x0f],
      values[packed[1] & 0x0f],
      values[packed[2] & 0x0f],
      values[packed[3] & 0x0f]
  );
  const float4 weights_odd0 = float4(
      values[packed[0] >> 4],
      values[packed[1] >> 4],
      values[packed[2] >> 4],
      values[packed[3] >> 4]
  );
  const float4 weights_even1 = float4(
      values[packed[4] & 0x0f],
      values[packed[5] & 0x0f],
      values[packed[6] & 0x0f],
      values[packed[7] & 0x0f]
  );
  const float4 weights_odd1 = float4(
      values[packed[4] >> 4],
      values[packed[5] >> 4],
      values[packed[6] >> 4],
      values[packed[7] >> 4]
  );
  return scale *
         (dot(even0, weights_even0) + dot(odd0, weights_odd0) + dot(even1, weights_even1) + dot(odd1, weights_odd1));
}

/// GPT-OSS clips each projection before applying its selected gated activation.
inline float activate_expert_value(
    const float acc_up,
    const float acc_gate,
    const float up_bias,
    const float gate_bias,
    const float gate_clip_min,
    const float gate_clip_max,
    const float up_clip_min,
    const float up_clip_max,
    const float silu_alpha,
    const uint gating_sel
) {
  const float up = clamp(acc_up + up_bias, up_clip_min, up_clip_max);
  if (gating_sel <= 1)
    return gating_sel == 0 ? activate_gelu(up) : activate_silu_alpha(up, silu_alpha);

  const float gate = clamp(acc_gate + gate_bias, gate_clip_min, gate_clip_max);
  const float gate_activated = gating_sel == 2 ? activate_silu_alpha(gate, silu_alpha) : activate_gelu(gate);
  return gate_activated * up;
}

// Adjacent output rows share each activation load while retaining independent reductions.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MoeExpertsMxfp4DecodePassA)(
    device const T* x_perm,
    device const uint* expert_offsets,
    device const uchar* w13_blocks,
    device const uchar* w13_scales,
    device const T* w13_global_scale,
    device float* hidden_out,
    device const T* up_biases,
    constant uint& d_model,
    constant uint& d_ff,
    constant uint& e,
    constant float& gate_clip_min,
    constant float& gate_clip_max,
    constant float& up_clip_min,
    constant float& up_clip_max,
    constant float& silu_alpha,
    device const uint* tile_map,
    threadgroup float mxfp4_values[16],
    const uint gating_sel SPECIALIZE,
    const ThreadContext thread_context,
    const uint tile_idx GROUPS(INDIRECT),
    const uint tid THREADS(128)
) {
  if (tid < 16)
    mxfp4_values[tid] = MXFP4_VALUES[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint h_block_idx = tile_map[tile_idx * 3 + 0];
  const uint expert_idx = tile_map[tile_idx * 3 + 1];
  const uint row_in_expert = tile_map[tile_idx * 3 + 2];

  const uint segment_start = expert_offsets[expert_idx];
  const uint segment_end = expert_offsets[expert_idx + 1];
  const uint global_row = segment_start + row_in_expert;
  if (global_row >= segment_end)
    return;

  const uint h_idx0 = h_block_idx * 8 + thread_context.simdgroup_index * 2;
  if (h_idx0 >= d_ff)
    return;
  const uint h_idx1 = h_idx0 + 1;
  const bool has_second_output = h_idx1 < d_ff;

  const uint groups_per_row = (d_model + 15) / 16;
  const ulong rows_per_expert = (ulong)(2 * d_ff);
  const ulong expert_row_base = (ulong)expert_idx * rows_per_expert;
  const ulong up_row0 = expert_row_base + h_idx0;
  const ulong gate_row0 = expert_row_base + d_ff + h_idx0;
  const ulong up_row1 = expert_row_base + h_idx1;
  const ulong gate_row1 = expert_row_base + d_ff + h_idx1;
  const ulong x_row_base = (ulong)global_row * d_model;

  float acc_up0 = 0.0f;
  float acc_gate0 = 0.0f;
  float acc_up1 = 0.0f;
  float acc_gate1 = 0.0f;
  const uint group_lane = thread_context.simd_lane_id;
  using Vec4 = typename metal::vec<T, 4>;
  for (uint group_base = 0; group_base < groups_per_row; group_base += 32) {
    const uint group = group_base + group_lane;
    if (group >= groups_per_row)
      continue;

    const uint element = group * 16;
    if (element + 15 >= d_model)
      continue;

    device const Vec4* x = reinterpret_cast<device const Vec4*>(x_perm + x_row_base + element);
    const float4 x0 = float4(x[0]);
    const float4 x1 = float4(x[1]);
    const float4 x2 = float4(x[2]);
    const float4 x3 = float4(x[3]);
    acc_up0 +=
        dot_mxfp4_group16(w13_blocks, w13_scales, mxfp4_values, up_row0, groups_per_row, group, x0, x1, x2, x3);
    if (gating_sel > 1) {
      acc_gate0 +=
          dot_mxfp4_group16(w13_blocks, w13_scales, mxfp4_values, gate_row0, groups_per_row, group, x0, x1, x2, x3);
    }
    if (has_second_output) {
      acc_up1 +=
          dot_mxfp4_group16(w13_blocks, w13_scales, mxfp4_values, up_row1, groups_per_row, group, x0, x1, x2, x3);
      if (gating_sel > 1) {
        acc_gate1 +=
            dot_mxfp4_group16(w13_blocks, w13_scales, mxfp4_values, gate_row1, groups_per_row, group, x0, x1, x2, x3);
      }
    }
  }

  acc_up0 = simd_sum(acc_up0);
  acc_up1 = simd_sum(acc_up1);
  if (gating_sel > 1) {
    acc_gate0 = simd_sum(acc_gate0);
    acc_gate1 = simd_sum(acc_gate1);
  }
  const float global_scale = float(w13_global_scale[expert_idx]);
  acc_up0 *= global_scale;
  acc_gate0 *= global_scale;
  acc_up1 *= global_scale;
  acc_gate1 *= global_scale;

  if (thread_context.simd_lane_id != 0)
    return;

  const float activated0 = activate_expert_value(
      acc_up0,
      acc_gate0,
      float(up_biases[up_row0]),
      float(up_biases[gate_row0]),
      gate_clip_min,
      gate_clip_max,
      up_clip_min,
      up_clip_max,
      silu_alpha,
      gating_sel
  );
  hidden_out[(ulong)global_row * d_ff + h_idx0] = activated0;
  if (!has_second_output)
    return;

  const float activated1 = activate_expert_value(
      acc_up1,
      acc_gate1,
      float(up_biases[up_row1]),
      float(up_biases[gate_row1]),
      gate_clip_min,
      gate_clip_max,
      up_clip_min,
      up_clip_max,
      silu_alpha,
      gating_sel
  );
  hidden_out[(ulong)global_row * d_ff + h_idx1] = activated1;
}

#define THREADS_PER_SIMD 32
#define SIMDGROUPS_PER_TG 4
#define OUTPUTS_PER_SIMDGROUP 2
#define OUTPUTS_PER_TG 8

// The down projection decodes only the values consumed by the dot product;
// no dense expert weight buffer is allocated on either side of this kernel.
template <typename T, typename AccumT>
VARIANTS(T, float, half, bfloat)
VARIANTS(AccumT, float)
PUBLIC KERNEL(MoeExpertsMxfp4DecodeDownFused2D)(
    device const float* hidden,
    device const uint* row_expert_map,
    device const uchar* w2_blocks,
    device const uchar* w2_scales,
    device const T* w2_global_scale,
    device const T* down_biases,
    device T* y_out,
    constant uint& total_rows,
    constant uint& d_model,
    constant uint& d_ff,
    constant uint& e,
    threadgroup float mxfp4_values[16],
    const ThreadContext thread_context,
    const uint tgpig_x GROUPS(d_model.div_ceil(OUTPUTS_PER_TG)),
    const uint tgpig_y GROUPS(total_rows),
    const uint tid THREADS(128)
) {
  if (tid < 16)
    mxfp4_values[tid] = MXFP4_VALUES[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint row_idx = tgpig_y;
  const uint output_idx0 = tgpig_x * OUTPUTS_PER_TG + thread_context.simdgroup_index * OUTPUTS_PER_SIMDGROUP;
  if (output_idx0 >= d_model)
    return;
  const uint output_idx1 = output_idx0 + 1;
  const bool has_second_output = output_idx1 < d_model;

  const uint expert_idx = row_expert_map[row_idx];
  const uint groups_per_row = (d_ff + 31) / 32;
  const ulong weight_row0 = (ulong)expert_idx * d_model + output_idx0;
  const ulong weight_row1 = (ulong)expert_idx * d_model + output_idx1;
  const ulong hidden_base = (ulong)row_idx * d_ff;

  AccumT acc0 = AccumT(0.0);
  AccumT acc1 = AccumT(0.0);
  const uint group_lane = thread_context.simd_lane_id / 2;
  const uint half_group = thread_context.simd_lane_id % 2;
  for (uint group_base = 0; group_base < groups_per_row; group_base += 16) {
    const uint group = group_base + group_lane;
    if (group >= groups_per_row)
      continue;

    const uint element = group * 32 + half_group * 16;
    if (element >= d_ff)
      continue;

    if (element + 15 >= d_ff) {
      for (uint offset = 0; offset < 16 && element + offset < d_ff; ++offset) {
        const float value = hidden[hidden_base + element + offset];
        acc0 += AccumT(value * mxfp4_value(
                                  w2_blocks, w2_scales, weight_row0, groups_per_row, group, half_group * 16 + offset
                              ));
        if (has_second_output) {
          acc1 += AccumT(value * mxfp4_value(
                                    w2_blocks, w2_scales, weight_row1, groups_per_row, group, half_group * 16 + offset
                                ));
        }
      }
      continue;
    }

    device const float4* values = reinterpret_cast<device const float4*>(hidden + hidden_base + element);
    const float4 x0 = values[0];
    const float4 x1 = values[1];
    const float4 x2 = values[2];
    const float4 x3 = values[3];
    acc0 += AccumT(
        dot_mxfp4_16(
            w2_blocks, w2_scales, mxfp4_values, weight_row0, groups_per_row, group, half_group, x0, x1, x2, x3
        )
    );
    if (has_second_output) {
      acc1 += AccumT(
          dot_mxfp4_16(
              w2_blocks, w2_scales, mxfp4_values, weight_row1, groups_per_row, group, half_group, x0, x1, x2, x3
          )
      );
    }
  }

  const AccumT global_scale = AccumT(w2_global_scale[expert_idx]);
  AccumT result0 = simd_sum(acc0) * global_scale;
  AccumT result1 = simd_sum(acc1) * global_scale;
  if (thread_context.simd_lane_id != 0)
    return;

  result0 += AccumT(down_biases[(ulong)expert_idx * d_model + output_idx0]);
  y_out[(ulong)row_idx * d_model + output_idx0] = T(result0);
  if (!has_second_output)
    return;

  result1 += AccumT(down_biases[(ulong)expert_idx * d_model + output_idx1]);
  y_out[(ulong)row_idx * d_model + output_idx1] = T(result1);
}
