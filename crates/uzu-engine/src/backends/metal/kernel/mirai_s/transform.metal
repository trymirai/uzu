#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"

using namespace metal;

#define TRANSFORM_THREADS 512

// Butterfly across lanes at distance `mask`: the lower index keeps a + b, the upper a - b.
static METAL_FUNC float butterfly(float value, ushort lane, ushort mask) {
  const float other = simd_shuffle_xor(value, mask);
  return (lane & mask) != 0 ? other - value : value + other;
}

// Hadamard index h of a thread's element i after the transpose: the lane walks h bits 5..9, so the
// remaining strides are lane shuffles again (POWER 2048 keeps h bit 10 in registers as i % 2).
template <uint POWER>
static METAL_FUNC uint transposed_index(ushort lane, ushort simdgroup, uint i) {
  if (POWER == 1024) {
    return 32 * lane + simdgroup + 16 * i;
  }
  return 32 * (lane + 32 * (i % 2)) + simdgroup + 16 * (i / 2);
}

// Input rotation of a Mirai S linear, one threadgroup per token. Column k = h * ORDER + q:
//   x * signs -> mixing[ORDER x ORDER] over q -> Walsh-Hadamard over h (strides ascending) -> / sqrt(POWER)
//   -> bf16 -> int8 with scale max|x| / 127.
// token_statistics[2 * token] sums the int8 values of the columns k = j (mod 4) for j = 0..3 and
// token_statistics[2 * token + 1] is (scale, 0, 0, 0); the projection folds the codebook offsets in with the sums.
// The WHT runs strides 1..16 as lane shuffles, transposes through threadgroup memory, and runs the remaining
// strides the same way (stride 1024 in registers).
template <uint DIMENSION>
VARIANTS(DIMENSION, 5120, 6144, 17408)
KERNEL(MiraiSTransform)(
    device const bfloat* input,
    device const float* signs,
    device const float* mixing,
    device int8_t* activations,
    device float4* token_statistics,
    constant uint& batch,
    threadgroup float values[2048],
    threadgroup float partial_maximum[16],
    threadgroup float mixing_shared[17 * 17],
    const uint token GROUPS(batch),
    const uint thread_index THREADS(TRANSFORM_THREADS),
    const ThreadContext thread_context
) {
  // ORDER is the odd part of DIMENSION (`mixing_order` in common/kernel/mirai_s.rs), POWER its lowest set bit
  constexpr uint POWER = DIMENSION & (0u - DIMENSION);
  constexpr uint ORDER = DIMENSION / POWER;
  constexpr uint PER_THREAD = POWER / TRANSFORM_THREADS;
  // 1 / sqrt(POWER), exact in f32 up to the rounding of 1 / sqrt(2)
  constexpr float NORMALIZATION = (POWER == 2048 ? M_SQRT1_2_F : 1.0f) / 32.0f;
  static_assert(POWER == 1024 || POWER == 2048, "unsupported power");

  const ushort lane = ushort(thread_context.simd_lane_id);
  const ushort simdgroup = ushort(thread_context.simdgroup_index);

  for (uint index = thread_index; index < ORDER * ORDER; index += TRANSFORM_THREADS) {
    mixing_shared[index] = mixing[index];
  }

  // before the transpose a thread holds h = 32 * (simdgroup + 16 * i) + lane
  float signed_inputs[PER_THREAD][ORDER];
  METAL_PRAGMA_UNROLL
  for (uint i = 0; i < PER_THREAD; ++i) {
    const uint h = 32 * (simdgroup + 16 * i) + lane;
    METAL_PRAGMA_UNROLL
    for (uint q = 0; q < ORDER; ++q) {
      signed_inputs[i][q] = float(input[token * DIMENSION + h * ORDER + q]) * signs[h * ORDER + q];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float transformed[ORDER][PER_THREAD];
  float local_maximum = 0.0f;
  for (uint q_out = 0; q_out < ORDER; ++q_out) {
    float element[PER_THREAD];
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < PER_THREAD; ++i) {
      element[i] = 0.0f;
      METAL_PRAGMA_UNROLL
      for (uint q = 0; q < ORDER; ++q) {
        element[i] = fma(signed_inputs[i][q], mixing_shared[q_out * ORDER + q], element[i]);
      }
    }
    METAL_PRAGMA_UNROLL
    for (ushort mask = 1; mask <= 16; mask <<= 1) {
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < PER_THREAD; ++i) {
        element[i] = butterfly(element[i], lane, mask);
      }
    }
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < PER_THREAD; ++i) {
      values[32 * (simdgroup + 16 * i) + lane] = element[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < PER_THREAD; ++i) {
      element[i] = values[transposed_index<POWER>(lane, simdgroup, i)];
    }
    METAL_PRAGMA_UNROLL
    for (ushort mask = 1; mask <= 16; mask <<= 1) {
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < PER_THREAD; ++i) {
        element[i] = butterfly(element[i], lane, mask);
      }
    }
    if constexpr (POWER == 2048) {
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < PER_THREAD; i += 2) {
        const float lower = element[i];
        const float upper = element[i + 1];
        element[i] = lower + upper;
        element[i + 1] = lower - upper;
      }
    }

    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < PER_THREAD; ++i) {
      transformed[q_out][i] = float(bfloat(element[i] * NORMALIZATION));
      local_maximum = max(local_maximum, abs(transformed[q_out][i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float simdgroup_maximum = simd_max(local_maximum);
  if (lane == 0) {
    partial_maximum[simdgroup] = simdgroup_maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float maximum = simd_max(lane < 16 ? partial_maximum[lane] : 0.0f);
  const float scale = maximum > 0.0f ? precise::divide(maximum, 127.0f) : 1.0f;

  int class_sums[4] = {0, 0, 0, 0};
  METAL_PRAGMA_UNROLL
  for (uint q_out = 0; q_out < ORDER; ++q_out) {
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < PER_THREAD; ++i) {
      const uint column = transposed_index<POWER>(lane, simdgroup, i) * ORDER + q_out;
      const int8_t quantized = int8_t(clamp(round(precise::divide(transformed[q_out][i], scale)), -127.0f, 127.0f));
      activations[token * DIMENSION + column] = quantized;
      METAL_PRAGMA_UNROLL
      for (uint j = 0; j < 4; ++j) {
        class_sums[j] += (column & 3) == j ? int(quantized) : 0;
      }
    }
  }
  // `values` is free after the last barrier of the WHT loop
  METAL_PRAGMA_UNROLL
  for (uint j = 0; j < 4; ++j) {
    const int simdgroup_sum = simd_sum(class_sums[j]);
    if (lane == 0) {
      values[4 * simdgroup + j] = float(simdgroup_sum);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float4 sums = float4(0.0f);
    for (uint group = 0; group < 16; ++group) {
      sums += float4(values[4 * group], values[4 * group + 1], values[4 * group + 2], values[4 * group + 3]);
    }
    token_statistics[2 * token] = sums;
    token_statistics[2 * token + 1] = float4(scale, 0.0f, 0.0f, 0.0f);
  }
}
