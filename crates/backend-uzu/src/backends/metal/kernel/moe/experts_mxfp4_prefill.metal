#include <metal_stdlib>
#include <metal_simdgroup>
#include "../activation/activations.h"
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
using namespace metal;

constant float MXFP4_PREFILL_VALUES[16] = {
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
inline float mxfp4_prefill_e8m0_scale(const uchar exponent) {
  const uint bits = exponent == 0 ? 0x00400000u : uint(exponent) << 23;
  return as_type<float>(bits);
}

/// One cooperative load expands eight packed values into an MMA-ready tile.
template <typename T>
inline void stage_mxfp4_eight(
    device const uchar* blocks,
    device const uchar* scales,
    threadgroup const float* values,
    threadgroup T* destination,
    const ulong weight_row,
    const uint groups_per_row,
    const uint group_size,
    const uint element,
    const uint dimension,
    const float global_scale
) {
  if (element >= dimension) {
    METAL_PRAGMA_UNROLL
    for (uint index = 0; index < 8; ++index)
      destination[index] = T(0.0f);
    return;
  }

  const uint group = element / group_size;
  const uint pair = (element % group_size) / 2;
  const ulong group_offset = weight_row * (ulong)groups_per_row + group;
  const ulong block_offset = group_offset * (group_size / 2) + pair;
  const float scale = global_scale * mxfp4_prefill_e8m0_scale(scales[group_offset]);
  const uchar4 packed_values = *reinterpret_cast<device const uchar4*>(blocks + block_offset);
  METAL_PRAGMA_UNROLL
  for (uint pair_index = 0; pair_index < 4; ++pair_index) {
    const uchar packed = packed_values[pair_index];
    const uint output = pair_index * 2;
    destination[output] = T(scale * values[packed & 0x0f]);
    destination[output + 1] = element + output + 1 < dimension ? T(scale * values[packed >> 4]) : T(0.0f);
  }
}

/// GPT-OSS clips each projection before applying its selected gated activation.
inline float activate_mxfp4_prefill_value(
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

constant uint MXFP4_PASSA_BM = 16;
constant uint MXFP4_PASSA_BN = 32;
constant uint MXFP4_PASSA_BK = 32;
constant uint MXFP4_PASSA_LD = 36;
constant uint MXFP4_PASSA_ROW_FRAGMENTS = 2;

// Each packed weight tile is expanded once and reused by up to 16 routed rows.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MoeExpertsMxfp4PrefillPassA)(
    device const T* x_perm,
    device const uint* expert_offsets,
    device const uchar* w13_blocks,
    device const uchar* w13_scales,
    device const T* w13_global_scale,
    device const T* up_biases,
    device float* hidden_out,
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
    threadgroup T Xs[MXFP4_PASSA_BM * MXFP4_PASSA_LD],
    threadgroup T Wk_up[MXFP4_PASSA_BN * MXFP4_PASSA_LD],
    threadgroup T Wk_gate[MXFP4_PASSA_BN * MXFP4_PASSA_LD],
    const uint gating_sel SPECIALIZE,
    const uint n_tile_idx GROUPS(INDIRECT),
    const uint row_tile_idx GROUPS(INDIRECT),
    const uint lin THREADS(128),
    const ThreadContext thread_context
) {
  if (lin < 16)
    mxfp4_values[lin] = MXFP4_PREFILL_VALUES[lin];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint map_offset = row_tile_idx * 3;
  const uint expert_idx = tile_map[map_offset];
  if (expert_idx >= e)
    return;

  const uint segment_start = expert_offsets[expert_idx];
  const uint segment_end = expert_offsets[expert_idx + 1];
  const uint segment_length = segment_end - segment_start;
  if (segment_length == 0)
    return;

  const uint row_tile_offset = tile_map[map_offset + 2];
  const uint column_tile_offset = n_tile_idx * MXFP4_PASSA_BN;
  if (row_tile_offset >= segment_length || column_tile_offset >= d_ff)
    return;

  const uint row_count = min(MXFP4_PASSA_BM, segment_length - row_tile_offset);
  const uint column_count = min(MXFP4_PASSA_BN, d_ff - column_tile_offset);
  const uint groups_per_row = d_model / 16;
  const ulong expert_row_base = (ulong)expert_idx * (2 * d_ff);
  const float global_scale = float(w13_global_scale[expert_idx]);

  const uint column_simdgroup = thread_context.simdgroup_index;
  metal::simdgroup_float8x8 output_up[MXFP4_PASSA_ROW_FRAGMENTS];
  metal::simdgroup_float8x8 output_gate[MXFP4_PASSA_ROW_FRAGMENTS];
  METAL_PRAGMA_UNROLL
  for (uint row_fragment = 0; row_fragment < MXFP4_PASSA_ROW_FRAGMENTS; ++row_fragment) {
    output_up[row_fragment] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    output_gate[row_fragment] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }

  for (uint k_offset = 0; k_offset < d_model; k_offset += MXFP4_PASSA_BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint input_index = lin; input_index < MXFP4_PASSA_BM * 4; input_index += 128) {
      const uint row = input_index / 4;
      const uint element_offset = (input_index % 4) * 8;
      threadgroup T* x_destination = Xs + row * MXFP4_PASSA_LD + element_offset;
      if (row < row_count) {
        const ulong input_base = (ulong)(segment_start + row_tile_offset + row) * d_model + k_offset + element_offset;
        METAL_PRAGMA_UNROLL
        for (uint index = 0; index < 8; ++index)
          x_destination[index] = T(float(x_perm[input_base + index]));
      } else {
        METAL_PRAGMA_UNROLL
        for (uint index = 0; index < 8; ++index)
          x_destination[index] = T(0.0f);
      }
    }

    if (lin < MXFP4_PASSA_BN * 4) {
      const uint column = lin / 4;
      const uint element_offset = (lin % 4) * 8;
      const uint global_column = column_tile_offset + column;
      threadgroup T* up_destination = Wk_up + column * MXFP4_PASSA_LD + element_offset;
      threadgroup T* gate_destination = Wk_gate + column * MXFP4_PASSA_LD + element_offset;
      if (column < column_count) {
        const ulong up_row = expert_row_base + global_column;
        const ulong gate_row = expert_row_base + d_ff + global_column;
        stage_mxfp4_eight(
            w13_blocks,
            w13_scales,
            mxfp4_values,
            up_destination,
            up_row,
            groups_per_row,
            16,
            k_offset + element_offset,
            d_model,
            global_scale
        );
        if (gating_sel > 1) {
          stage_mxfp4_eight(
              w13_blocks,
              w13_scales,
              mxfp4_values,
              gate_destination,
              gate_row,
              groups_per_row,
              16,
              k_offset + element_offset,
              d_model,
              global_scale
          );
        }
      } else {
        METAL_PRAGMA_UNROLL
        for (uint index = 0; index < 8; ++index) {
          up_destination[index] = T(0.0f);
          gate_destination[index] = T(0.0f);
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    METAL_PRAGMA_UNROLL
    for (uint k = 0; k < MXFP4_PASSA_BK; k += 8) {
      metal::simdgroup_matrix<T, 8, 8> rhs_up;
      simdgroup_load(rhs_up, Wk_up, MXFP4_PASSA_LD, ulong2(k, column_simdgroup * 8), true);
      metal::simdgroup_matrix<T, 8, 8> rhs_gate;
      if (gating_sel > 1) {
        simdgroup_load(rhs_gate, Wk_gate, MXFP4_PASSA_LD, ulong2(k, column_simdgroup * 8), true);
      }
      METAL_PRAGMA_UNROLL
      for (uint row_fragment = 0; row_fragment < MXFP4_PASSA_ROW_FRAGMENTS; ++row_fragment) {
        metal::simdgroup_matrix<T, 8, 8> lhs;
        const uint row = row_fragment * 8;
        simdgroup_load(lhs, Xs, MXFP4_PASSA_LD, ulong2(k, row));
        simdgroup_multiply_accumulate(output_up[row_fragment], lhs, rhs_up, output_up[row_fragment]);
        if (gating_sel > 1)
          simdgroup_multiply_accumulate(output_gate[row_fragment], lhs, rhs_gate, output_gate[row_fragment]);
      }
    }
  }

  const uint lane_quadrant = thread_context.simd_lane_id >> 2;
  const uint lane_row = (lane_quadrant & 4) + ((thread_context.simd_lane_id >> 1) & 3);
  const uint lane_column = ((lane_quadrant & 2) * 2) + ((thread_context.simd_lane_id & 1) * 2);
  const uint local_column = column_simdgroup * 8 + lane_column;
  if (local_column >= column_count)
    return;

  METAL_PRAGMA_UNROLL
  for (uint row_fragment = 0; row_fragment < MXFP4_PASSA_ROW_FRAGMENTS; ++row_fragment) {
    const uint local_row = row_fragment * 8 + lane_row;
    if (local_row >= row_count)
      continue;

    const auto up = output_up[row_fragment].thread_elements();
    const auto gate = output_gate[row_fragment].thread_elements();
    const ulong output_row = segment_start + row_tile_offset + local_row;
    METAL_PRAGMA_UNROLL
    for (uint index = 0; index < 2; ++index) {
      const uint column = local_column + index;
      if (column >= column_count)
        continue;

      const uint global_column = column_tile_offset + column;
      const ulong up_row = expert_row_base + global_column;
      const ulong gate_row = expert_row_base + d_ff + global_column;
      const float activated = activate_mxfp4_prefill_value(
          up[index],
          gate[index],
          float(up_biases[up_row]),
          float(up_biases[gate_row]),
          gate_clip_min,
          gate_clip_max,
          up_clip_min,
          up_clip_max,
          silu_alpha,
          gating_sel
      );
      hidden_out[output_row * (ulong)d_ff + global_column] = activated;
    }
  }
}

constant uint MXFP4_PASSB_BM = 16;
constant uint MXFP4_PASSB_BN = 32;
constant uint MXFP4_PASSB_BK = 32;
constant uint MXFP4_PASSB_LD = 36;
constant uint MXFP4_PASSB_ROW_FRAGMENTS = 2;

// The down projection reuses each expanded tile across the same routed row block.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MoeExpertsMxfp4PrefillPassB)(
    device const float* hidden,
    device const uint* expert_offsets,
    device const uchar* w2_blocks,
    device const uchar* w2_scales,
    device const T* w2_global_scale,
    device const T* down_biases,
    device T* output,
    constant uint& d_model,
    constant uint& d_ff,
    constant uint& e,
    device const uint* tile_map,
    threadgroup float mxfp4_values[16],
    threadgroup float Hs[MXFP4_PASSB_BM * MXFP4_PASSB_LD],
    threadgroup T Wk[MXFP4_PASSB_BN * MXFP4_PASSB_LD],
    const uint n_tile_idx GROUPS(INDIRECT),
    const uint row_tile_idx GROUPS(INDIRECT),
    const uint lin THREADS(128),
    const ThreadContext thread_context
) {
  if (lin < 16)
    mxfp4_values[lin] = MXFP4_PREFILL_VALUES[lin];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint map_offset = row_tile_idx * 3;
  const uint expert_idx = tile_map[map_offset];
  if (expert_idx >= e)
    return;

  const uint segment_start = expert_offsets[expert_idx];
  const uint segment_end = expert_offsets[expert_idx + 1];
  const uint segment_length = segment_end - segment_start;
  if (segment_length == 0)
    return;

  const uint row_tile_offset = tile_map[map_offset + 2];
  const uint column_tile_offset = n_tile_idx * MXFP4_PASSB_BN;
  if (row_tile_offset >= segment_length || column_tile_offset >= d_model)
    return;

  const uint row_count = min(MXFP4_PASSB_BM, segment_length - row_tile_offset);
  const uint column_count = min(MXFP4_PASSB_BN, d_model - column_tile_offset);
  const uint groups_per_row = d_ff / 32;
  const ulong expert_row_base = (ulong)expert_idx * d_model;
  const float global_scale = float(w2_global_scale[expert_idx]);

  const uint column_simdgroup = thread_context.simdgroup_index;
  metal::simdgroup_float8x8 output_tile[MXFP4_PASSB_ROW_FRAGMENTS];
  METAL_PRAGMA_UNROLL
  for (uint row_fragment = 0; row_fragment < MXFP4_PASSB_ROW_FRAGMENTS; ++row_fragment) {
    output_tile[row_fragment] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }

  for (uint k_offset = 0; k_offset < d_ff; k_offset += MXFP4_PASSB_BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint input_index = lin; input_index < MXFP4_PASSB_BM * 4; input_index += 128) {
      const uint row = input_index / 4;
      const uint element_offset = (input_index % 4) * 8;
      threadgroup float* destination = Hs + row * MXFP4_PASSB_LD + element_offset;
      if (row < row_count) {
        const ulong input_base = (ulong)(segment_start + row_tile_offset + row) * d_ff + k_offset + element_offset;
        METAL_PRAGMA_UNROLL
        for (uint index = 0; index < 8; ++index)
          destination[index] = hidden[input_base + index];
      } else {
        METAL_PRAGMA_UNROLL
        for (uint index = 0; index < 8; ++index)
          destination[index] = 0.0f;
      }
    }

    for (uint weight_index = lin; weight_index < MXFP4_PASSB_BN * 4; weight_index += 128) {
      const uint column = weight_index / 4;
      const uint element_offset = (weight_index % 4) * 8;
      const uint global_column = column_tile_offset + column;
      threadgroup T* destination = Wk + column * MXFP4_PASSB_LD + element_offset;
      if (column < column_count) {
        const ulong weight_row = expert_row_base + global_column;
        stage_mxfp4_eight(
            w2_blocks,
            w2_scales,
            mxfp4_values,
            destination,
            weight_row,
            groups_per_row,
            32,
            k_offset + element_offset,
            d_ff,
            global_scale
        );
      } else {
        METAL_PRAGMA_UNROLL
        for (uint index = 0; index < 8; ++index)
          destination[index] = T(0.0f);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    METAL_PRAGMA_UNROLL
    for (uint k = 0; k < MXFP4_PASSB_BK; k += 8) {
      metal::simdgroup_matrix<T, 8, 8> rhs;
      simdgroup_load(rhs, Wk, MXFP4_PASSB_LD, ulong2(k, column_simdgroup * 8), true);
      METAL_PRAGMA_UNROLL
      for (uint row_fragment = 0; row_fragment < MXFP4_PASSB_ROW_FRAGMENTS; ++row_fragment) {
        metal::simdgroup_float8x8 lhs;
        const uint row = row_fragment * 8;
        simdgroup_load(lhs, Hs, MXFP4_PASSB_LD, ulong2(k, row));
        simdgroup_multiply_accumulate(output_tile[row_fragment], lhs, rhs, output_tile[row_fragment]);
      }
    }
  }

  const uint lane_quadrant = thread_context.simd_lane_id >> 2;
  const uint lane_row = (lane_quadrant & 4) + ((thread_context.simd_lane_id >> 1) & 3);
  const uint lane_column = ((lane_quadrant & 2) * 2) + ((thread_context.simd_lane_id & 1) * 2);
  METAL_PRAGMA_UNROLL
  for (uint row_fragment = 0; row_fragment < MXFP4_PASSB_ROW_FRAGMENTS; ++row_fragment) {
    const uint local_row = row_fragment * 8 + lane_row;
    if (local_row >= row_count)
      continue;

    const ulong output_row = segment_start + row_tile_offset + local_row;
    const auto values = output_tile[row_fragment].thread_elements();
    const uint local_column = column_simdgroup * 8 + lane_column;
    METAL_PRAGMA_UNROLL
    for (uint index = 0; index < 2; ++index) {
      const uint column = local_column + index;
      if (column >= column_count)
        continue;

      const uint global_column = column_tile_offset + column;
      const float value = values[index] + float(down_biases[expert_row_base + global_column]);
      output[output_row * (ulong)d_model + global_column] = T(value);
    }
  }
}
