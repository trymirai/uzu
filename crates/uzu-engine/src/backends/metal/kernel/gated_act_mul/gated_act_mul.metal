#include <metal_stdlib>
#include "../common/activation_quantization.h"
#include "../common/dsl.h"
#include "../common/gated_act_mul.h"
#include "../common/thread_context.h"
#include "../generated/gated_act_mul.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;
using namespace uzu::activation_type;
using namespace uzu::gated_act_mul;

#define NUM_SIMDGROUPS 4
#define NUM_THREADS NUM_SIMDGROUPS* METAL_SIMD_SIZE
// Each thread owns VEC consecutive elements: a threadgroup covers TILE_ELEMENTS elements, and a 32-element
// Hadamard block / quantization group is owned by (block size / VEC) adjacent lanes of one simdgroup.
#define VEC 4
#define LANES_PER_BLOCK (HADAMARD_TRANSFORM_BLOCK_SIZE / VEC)
// No outer parentheses (like NUM_THREADS): the DSL already wraps GROUPS expressions.
#define TILE_ELEMENTS NUM_THREADS* VEC

#define QUANTIZED (ops == GatedActMulOp::Quantize || ops == GatedActMulOp::QuantizeWithGroupSums)
#define EMITS_GROUP_SUMS (ops == GatedActMulOp::QuantizeWithGroupSums)

// Bit-identical to simdgroup_hadamard_transform: same stage order (strides 1, 2, 4, 8, 16) and operand order, with
// element p = VEC * lane + k. Strides 1 and 2 stay inside the thread; strides 4, 8, 16 are lane xors 1, 2, 4.
static METAL_FUNC void hadamard_vec4(thread float (&x)[VEC], const ushort lane_index) {
  {
    const float a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3];
    x[0] = a1 + a0;
    x[1] = a0 - a1;
    x[2] = a3 + a2;
    x[3] = a2 - a3;
  }
  {
    const float a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3];
    x[0] = a2 + a0;
    x[2] = a0 - a2;
    x[1] = a3 + a1;
    x[3] = a1 - a3;
  }
  for (ushort lane_stride = 1; lane_stride < LANES_PER_BLOCK; lane_stride <<= 1) {
    METAL_PRAGMA_UNROLL
    for (uint k = 0; k < VEC; ++k) {
      const float partner = simd_shuffle_xor(x[k], lane_stride);
      x[k] = (lane_index & lane_stride) ? (partner - x[k]) : (partner + x[k]);
    }
  }
  METAL_PRAGMA_UNROLL
  for (uint k = 0; k < VEC; ++k) {
    x[k] = x[k] / sqrt(static_cast<float>(HADAMARD_TRANSFORM_BLOCK_SIZE));
  }
}

// Reduces over the group_size / VEC adjacent lanes that own one group (group_size <= METAL_SIMD_SIZE * VEC).
template <typename V, typename Combine>
static METAL_FUNC V reduce_lane_group(V value, const uint group_size, Combine combine) {
  for (ushort lane_stride = 1; lane_stride < group_size / VEC; lane_stride <<= 1) {
    value = combine(value, simd_shuffle_xor(value, lane_stride));
  }
  return value;
}

template <typename T>
static METAL_FUNC void load_vec(const device T* source, thread T (&destination)[VEC]) {
  const vec<T, VEC> loaded = *reinterpret_cast<const device packed_vec<T, VEC>*>(source);
  METAL_PRAGMA_UNROLL
  for (uint k = 0; k < VEC; ++k) {
    destination[k] = loaded[k];
  }
}

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(GatedActMul) (
    const device T* act_operand,
    const device T* value_operand OPTIONAL(!interleaved),
    device T* fp_out OPTIONAL(!QUANTIZED),
    device int8_t* q_out OPTIONAL(QUANTIZED),
    device float* scales_out OPTIONAL(QUANTIZED),
    device int32_t* group_sums_out OPTIONAL(EMITS_GROUP_SUMS),
    const device int32_t* hadamard_factors OPTIONAL(use_hadamard),
    const constant uint& gated_dim,
    const constant uint& batch_dim,
    const constant uint& value_offset,
    const constant uint& value_row_stride,
    const constant ActivationType& act_type,
    const constant float& activation_alpha OPTIONAL(custom_activation_alpha),
    const constant float& gate_clip_min OPTIONAL(clip_gate),
    const constant float& gate_clip_max OPTIONAL(clip_gate),
    const constant float& value_clip_min OPTIONAL(clip_value),
    const constant float& value_clip_max OPTIONAL(clip_value),
    const GatedActMulOp ops SPECIALIZE,
    const bool grouped_by_weight_nibble SPECIALIZE,
    const bool interleaved SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    const uint activation_scale_group_size SPECIALIZE,
    const uint sum_group_size SPECIALIZE,
    const bool custom_activation_alpha SPECIALIZE,
    const bool clip_gate SPECIALIZE,
    const bool clip_value SPECIALIZE,
    uint activation_tile_index GROUPS(gated_dim.div_ceil(TILE_ELEMENTS)),
    uint batch_idx GROUPS(batch_dim),
    uint thread_index THREADS(NUM_THREADS),
    const ThreadContext thread_context
) {
  const uint first_idx = activation_tile_index * TILE_ELEMENTS + thread_index * VEC;
  const ushort lane_index = static_cast<ushort>(thread_context.simd_lane_id);
  const bool first_in_bounds = first_idx < gated_dim;
  const uint element_count = first_in_bounds ? min(static_cast<uint>(VEC), gated_dim - first_idx) : 0u;
  // Whole-vector accesses need every row base to stay VEC-aligned, otherwise fall back to element accesses.
  const bool rows_aligned =
      interleaved ? (gated_dim % VEC == 0) : ((gated_dim | value_offset | value_row_stride) % VEC == 0);
  const bool use_vector_io = element_count == VEC && rows_aligned;

  T value[VEC];
  T gate[VEC];
  METAL_PRAGMA_UNROLL
  for (uint k = 0; k < VEC; ++k) {
    value[k] = static_cast<T>(0);
    gate[k] = static_cast<T>(0);
  }
  {
    const device T* value_row;
    const device T* gate_row;
    if (interleaved) {
      value_row = act_operand + batch_idx * (2 * gated_dim);
      gate_row = value_row + gated_dim;
    } else {
      value_row = value_operand + batch_idx * value_row_stride + value_offset;
      gate_row = act_operand + batch_idx * gated_dim;
    }
    if (use_vector_io) {
      load_vec(value_row + first_idx, value);
      load_vec(gate_row + first_idx, gate);
    } else {
      for (uint k = 0; k < element_count; ++k) {
        value[k] = value_row[first_idx + k];
        gate[k] = gate_row[first_idx + k];
      }
    }
  }
  METAL_PRAGMA_UNROLL
  for (uint k = 0; k < VEC; ++k) {
    if (clip_gate) {
      gate[k] = static_cast<T>(clamp(float(gate[k]), gate_clip_min, gate_clip_max));
    }
    if (clip_value) {
      value[k] = static_cast<T>(clamp(float(value[k]), value_clip_min, value_clip_max));
    }
  }

  // Hadamard blocks are 32-element aligned; a block that is not entirely inside the row is never transformed.
  const bool block_in_bounds =
      (first_idx / HADAMARD_TRANSFORM_BLOCK_SIZE + 1) * HADAMARD_TRANSFORM_BLOCK_SIZE <= gated_dim;
  const bool transform = (QUANTIZED || use_hadamard) && block_in_bounds;
  float result[VEC];
  METAL_PRAGMA_UNROLL
  for (uint k = 0; k < VEC; ++k) {
    result[k] = 0.0f;
    if (k < element_count && (!(QUANTIZED || use_hadamard) || block_in_bounds)) {
      if (custom_activation_alpha && act_type == ActivationType::SILU) {
        const T activated = activate_silu_alpha(gate[k], activation_alpha);
        const T gated = value[k] * activated;
        result[k] = static_cast<float>(gated);
      } else {
        result[k] = gated_act_mul(value[k], gate[k], act_type);
      }
    }
  }
  if (transform) {
    // Elements are only multiplied by their factor here; all lanes of a block reach the shuffles together.
    int32_t factors[VEC];
    load_vec(hadamard_factors + first_idx, factors);
    METAL_PRAGMA_UNROLL
    for (uint k = 0; k < VEC; ++k) {
      result[k] = result[k] * float(factors[k]);
    }
  }
  if (QUANTIZED || use_hadamard) {
    // Lanes of a block that is out of the row only hold zeros; running the butterfly keeps shuffles uniform.
    hadamard_vec4(result, lane_index);
    if (!transform) {
      METAL_PRAGMA_UNROLL
      for (uint k = 0; k < VEC; ++k) {
        result[k] = 0.0f;
      }
    }
  }

  if (!QUANTIZED) {
    if (first_in_bounds) {
      device T* out_row = fp_out + batch_idx * gated_dim;
      if (use_vector_io) {
        vec<T, VEC> packed;
        METAL_PRAGMA_UNROLL
        for (uint k = 0; k < VEC; ++k) {
          packed[k] = static_cast<T>(result[k]);
        }
        *reinterpret_cast<device packed_vec<T, VEC>*>(out_row + first_idx) = packed;
      } else {
        for (uint k = 0; k < element_count; ++k) {
          out_row[first_idx + k] = static_cast<T>(result[k]);
        }
      }
    }
    return;
  }

  const float magnitude = max(max(fabs(result[0]), fabs(result[1])), max(fabs(result[2]), fabs(result[3])));
  const float maximum =
      reduce_lane_group(magnitude, activation_scale_group_size, [](float x, float y) { return max(x, y); });
  const float scale = isfinite(maximum) && maximum > 0.0f ? maximum / ACTIVATION_QUANT_INT8_MAX : 1.0f;
  int8_t code[VEC];
  int code_sum = 0;
  uint packed_codes = 0;
  METAL_PRAGMA_UNROLL
  for (uint k = 0; k < VEC; ++k) {
    code[k] = quantize_activation_int8(result[k], scale);
    code_sum += static_cast<int>(code[k]);
    packed_codes |= static_cast<uint>(static_cast<uchar>(code[k])) << (8 * k);
  }

  // gated_dim is a multiple of every group size and of 32, so a group is either fully inside or fully outside.
  const uint row_base = batch_idx * gated_dim;
  if (grouped_by_weight_nibble) {
    // [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 4, 1, 5, 2, 6, 3, 7]: a lane pair owns one 8-code word pair. The even lane
    // writes codes (0, 2, 4, 6), the odd lane codes (1, 3, 5, 7).
    const uint partner_codes = simd_shuffle_xor(packed_codes, 1);
    const uint odd = lane_index & 1;
    const uint even_codes = odd ? partner_codes : packed_codes;
    const uint odd_codes = odd ? packed_codes : partner_codes;
    if (first_in_bounds) {
      const uint shift = 8 * odd;
      const packed_char4 word = packed_char4(
          static_cast<char>(even_codes >> shift),
          static_cast<char>(even_codes >> (shift + 16)),
          static_cast<char>(odd_codes >> shift),
          static_cast<char>(odd_codes >> (shift + 16))
      );
      *reinterpret_cast<device packed_char4*>(q_out + row_base + (first_idx & ~7u) + 4 * odd) = word;
    }
  } else if (first_in_bounds) {
    *reinterpret_cast<device packed_char4*>(q_out + row_base + first_idx) =
        packed_char4(code[0], code[1], code[2], code[3]);
  }

  if (first_in_bounds && lane_index % (activation_scale_group_size / VEC) == 0) {
    scales_out[batch_idx * (gated_dim / activation_scale_group_size) + first_idx / activation_scale_group_size] = scale;
  }

  if (EMITS_GROUP_SUMS) {
    const int sum = reduce_lane_group(code_sum, sum_group_size, [](int x, int y) { return x + y; });
    if (first_in_bounds && lane_index % (sum_group_size / VEC) == 0) {
      group_sums_out[batch_idx * (gated_dim / sum_group_size) + first_idx / sum_group_size] = sum;
    }
  }
}
