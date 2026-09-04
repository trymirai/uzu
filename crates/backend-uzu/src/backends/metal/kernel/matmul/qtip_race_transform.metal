// Full-dimension incoherence transform + signed-A8 quantisation, race version.
//
// Bit-exact replacement for `QtipFullIncoherenceA8` (same fp32 operation
// order: per-element mixing sum over q ascending, Walsh-Hadamard butterflies at
// strides 1, 2, 4, ..., power/2 in that order, bf16 rounding, per-token max,
// scale = max / 127, round-to-nearest int8 with clamp), restructured so that
//
//   * the `order` input values of each element are loaded once and kept in
//     registers instead of being re-read for every output q;
//   * the WHT runs strides 1..16 as simdgroup shuffles, one threadgroup
//     transpose, then strides 32.. as shuffles / in-register butterflies, i.e.
//     two barriers per WHT instead of log2(power).
//
// One threadgroup (512 threads) per token. power = 1024 (5120 / 17408 columns,
// 2 elements per thread) or 2048 (6144 columns, 4 elements per thread).

#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "common/defines.h"

using namespace metal;

// Butterfly at lane distance `mask` on a value held one-per-lane. Both partners
// compute (a + b) or (a - b) exactly as the reference: the lower index keeps a + b.
static inline float qtip_race_wht_shuffle_stage(float value, ushort lane, ushort mask) {
  const float other = simd_shuffle_xor(value, mask);
  const bool upper = (lane & mask) != 0;
  // lower partner: a = value, b = other -> a + b ; upper partner: a = other, b = value -> a - b
  return upper ? (other - value) : (value + other);
}

template <uint POWER, uint ORDER>
static inline void qtip_race_full_incoherence_a8(
    device const bfloat* input,
    device const float* signs,
    device const float* small_q,
    device int8_t* output,
    device float* scales,
    uint active_batch,
    uint dimension,
    threadgroup float* values,       // POWER floats
    threadgroup float* partial_max,  // 16 floats
    threadgroup float* q_matrix,     // ORDER * ORDER floats
    uint token,
    uint thread_index,
    const thread ThreadContext& thread_context) {
  constexpr uint THREADS = 512u;
  constexpr uint PER_THREAD = POWER / THREADS;  // 2 or 4
  static_assert(POWER == 1024u || POWER == 2048u, "unsupported power");
  static_assert(ORDER <= 17u, "unsupported order");

  if (token >= active_batch) {
    for (uint element = thread_index; element < dimension; element += THREADS) {
      output[token * dimension + element] = int8_t(0);
    }
    if (thread_index == 0u) {
      scales[token] = 1.0f;
    }
    return;
  }

  const ushort lane = ushort(thread_context.simd_lane_id);
  const ushort simdgroup = ushort(thread_context.simdgroup_index);  // 0..15

  for (uint index = thread_index; index < ORDER * ORDER; index += THREADS) {
    q_matrix[index] = small_q[index];
  }

  // Phase-1 element assignment: h = 32 * a + lane with a = simdgroup + 16 * i.
  // Phase-2 (after the transpose): h = 32 * lane + b with b = simdgroup + 16 * i
  // (POWER 1024) or additionally a-bit 5 in-register (POWER 2048: a = lane, lane + 32).
  float mixed_inputs[PER_THREAD][ORDER];
  METAL_PRAGMA_UNROLL
  for (uint i = 0; i < PER_THREAD; ++i) {
    const uint a = uint(simdgroup) + 16u * i;
    const uint h = 32u * a + uint(lane);
    const uint base = token * dimension + h * ORDER;
    METAL_PRAGMA_UNROLL
    for (uint q = 0; q < ORDER; ++q) {
      mixed_inputs[i][q] = float(input[base + q]) * signs[h * ORDER + q];
    }
  }

  const float normalization = rsqrt(float(POWER));
  float local_maximum = 0.0f;
  float transformed[PER_THREAD * ORDER];

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint q_out = 0u; q_out < ORDER; ++q_out) {
    // mixing (same fp32 order as the reference: accumulate q ascending from 0)
    float element[PER_THREAD];
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < PER_THREAD; ++i) {
      float value = 0.0f;
      METAL_PRAGMA_UNROLL
      for (uint q = 0; q < ORDER; ++q) {
        value += mixed_inputs[i][q] * q_matrix[q_out * ORDER + q];
      }
      element[i] = value;
    }

    // strides 1..16: lanes hold consecutive h within a 32-block
    METAL_PRAGMA_UNROLL
    for (ushort mask = 1; mask <= 16; mask <<= 1) {
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < PER_THREAD; ++i) {
        element[i] = qtip_race_wht_shuffle_stage(element[i], lane, mask);
      }
    }

    // transpose through threadgroup memory: values[h]
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < PER_THREAD; ++i) {
      const uint a = uint(simdgroup) + 16u * i;
      values[32u * a + uint(lane)] = element[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // phase 2: thread holds h = 32 * a + b with a = lane (+32 for POWER 2048), b = simdgroup + 16 * i
    if constexpr (POWER == 1024u) {
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < PER_THREAD; ++i) {
        const uint b = uint(simdgroup) + 16u * i;
        element[i] = values[32u * uint(lane) + b];
      }
      // strides 32..512 = a bits 0..4 = lane bits
      METAL_PRAGMA_UNROLL
      for (ushort mask = 1; mask <= 16; mask <<= 1) {
        METAL_PRAGMA_UNROLL
        for (uint i = 0; i < PER_THREAD; ++i) {
          element[i] = qtip_race_wht_shuffle_stage(element[i], lane, mask);
        }
      }
    } else {
      // POWER 2048: a in 0..63; thread holds (a = lane, b), (a = lane + 32, b) for b = simdgroup, simdgroup + 16
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < 2; ++i) {
        const uint b = uint(simdgroup) + 16u * i;
        element[2 * i] = values[32u * uint(lane) + b];
        element[2 * i + 1] = values[32u * (uint(lane) + 32u) + b];
      }
      METAL_PRAGMA_UNROLL
      for (ushort mask = 1; mask <= 16; mask <<= 1) {
        METAL_PRAGMA_UNROLL
        for (uint i = 0; i < 4; ++i) {
          element[i] = qtip_race_wht_shuffle_stage(element[i], lane, mask);
        }
      }
      // stride 1024 = a bit 5: in-register pair (2i, 2i+1)
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < 2; ++i) {
        const float lower = element[2 * i];
        const float upper = element[2 * i + 1];
        element[2 * i] = lower + upper;
        element[2 * i + 1] = lower - upper;
      }
    }

    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < PER_THREAD; ++i) {
      const float value = float(bfloat(element[i] * normalization));
      transformed[q_out * PER_THREAD + i] = value;
      local_maximum = max(local_maximum, abs(value));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float simdgroup_maximum = simd_max(local_maximum);
  if (lane == 0) {
    partial_max[simdgroup] = simdgroup_maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maximum = lane < 16 ? partial_max[lane] : 0.0f;
  maximum = simd_max(maximum);
  const float scale = isfinite(maximum) && maximum > 0.0f ? maximum / 127.0f : 1.0f;

  // phase-2 element h for (i): POWER 1024: h = 32 * lane + simdgroup + 16 * i
  //                            POWER 2048: h = 32 * (lane + 32 * (i & 1)) + simdgroup + 16 * (i >> 1)
  METAL_PRAGMA_UNROLL
  for (uint q_out = 0u; q_out < ORDER; ++q_out) {
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < PER_THREAD; ++i) {
      uint h;
      if constexpr (POWER == 1024u) {
        h = 32u * uint(lane) + uint(simdgroup) + 16u * i;
      } else {
        h = 32u * (uint(lane) + 32u * (i & 1u)) + uint(simdgroup) + 16u * (i >> 1u);
      }
      const float value = transformed[q_out * PER_THREAD + i];
      output[token * dimension + h * ORDER + q_out] =
          int8_t(clamp(round(value / scale), -127.0f, 127.0f));
    }
  }
  if (thread_index == 0u) {
    scales[token] = scale;
  }
}

#define QTIP_RACE_TRANSFORM_KERNEL(NAME, POWER, ORDER) \
KERNEL(NAME)( \
    device const bfloat* input, \
    device const float* signs, \
    device const float* small_q, \
    device int8_t* output, \
    device float* scales, \
    const constant uint& active_batch, \
    const constant uint& padded_batch, \
    const constant uint& dimension, \
    threadgroup float values[2048], \
    threadgroup float partial_max[16], \
    threadgroup float q_matrix[289], \
    const uint token GROUPS(padded_batch), \
    const uint thread_index THREADS(512), \
    const ThreadContext thread_context \
) { \
  qtip_race_full_incoherence_a8<POWER, ORDER>( \
      input, signs, small_q, output, scales, active_batch, dimension, \
      values, partial_max, q_matrix, token, thread_index, thread_context); \
}

QTIP_RACE_TRANSFORM_KERNEL(QtipRaceTransform5120, 1024, 5)
QTIP_RACE_TRANSFORM_KERNEL(QtipRaceTransform6144, 2048, 3)
QTIP_RACE_TRANSFORM_KERNEL(QtipRaceTransform17408, 1024, 17)

#undef QTIP_RACE_TRANSFORM_KERNEL
