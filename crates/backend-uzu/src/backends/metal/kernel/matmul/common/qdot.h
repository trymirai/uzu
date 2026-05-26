#pragma once

#include <metal_stdlib>

#include "../../common/defines.h"
#include "quant_unpack.h"

using namespace metal;

namespace uzu {
namespace gemm {

// Pre-scales each activation lane k by 2^-(BITS*k). qdot then reads the
// matching weight nibble/byte in place (value = q * 2^(BITS*k)) without
// shifting it down to the low bits; the positional 2^(BITS*k) factor cancels
// the 2^-(BITS*k) here, so the dot product is unchanged. All factors are powers
// of two, so this is bit-exact.
template <typename T, typename U, int VALUES_PER_THREAD, int BITS>
METAL_FUNC U load_vector(const device T* x, thread U* x_thread) {
  using U4 = vec<U, 4>;
  const U4 inv =
      U4(U(1),
         U(1) / U(1u << BITS),
         U(1) / U(1u << (2u * BITS)),
         U(1) / U(1u << (3u * BITS)));
  U sum = 0;
  thread U4* x_vec4 = reinterpret_cast<thread U4*>(x_thread);
  METAL_PRAGMA_UNROLL
  for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
    U4 v = U4(x[4 * i], x[4 * i + 1], x[4 * i + 2], x[4 * i + 3]);
    sum += v[0] + v[1] + v[2] + v[3];
    x_vec4[i] = v * inv;
  }
  return sum;
}

template <typename T, typename U, int VALUES_PER_THREAD>
METAL_FUNC void load_vector_unscaled(const device T* x, thread U* x_thread) {
  using U4 = vec<U, 4>;
  thread U4* x_vec4 = reinterpret_cast<thread U4*>(x_thread);
  METAL_PRAGMA_UNROLL
  for (int index = 0; index < VALUES_PER_THREAD / 4; index++) {
    x_vec4[index] = U4(x[4 * index], x[4 * index + 1], x[4 * index + 2], x[4 * index + 3]);
  }
}

template <typename T, typename U, int VALUES_PER_THREAD>
METAL_FUNC U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  U sum = 0;
  METAL_PRAGMA_UNROLL
  for (int i = 0; i < VALUES_PER_THREAD; ++i) {
    x_thread[i] = 0;
  }
  for (int i = 0; i < N; ++i) {
    U v = x[i];
    sum += v;
    x_thread[i] = v;
  }
  return sum;
}

template <typename U, int VALUES_PER_THREAD, int BITS>
METAL_FUNC U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum
) {
  static_assert(BITS == 4 || BITS == 8, "Only int4 and int8 supported");

  U accumulator = 0;
  if constexpr (BITS == 4) {
    using U4 = vec<U, 4>;
    const device ushort* weight_words =
        reinterpret_cast<const device ushort*>(w);
    const thread U4* x_vec4 = reinterpret_cast<const thread U4*>(x_thread);
    METAL_PRAGMA_UNROLL
    for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
      // Mask each nibble in place (no shifts); value of lane k is n_k << (4*k),
      // i.e. n_k * 16^k, which is < 2^23 so the magic-number convert is valid.
      // The matching x lane was pre-divided by 16^k in load_vector.
      const uint4 lanes =
          uint4(weight_words[i]) & uint4(0x000fu, 0x00f0u, 0x0f00u, 0xf000u);
      const U4 weight_vec4 =
          U4(as_type<float4>(lanes | uint4(0x4b000000u)) - float4(8388608.0f));
      accumulator += dot(x_vec4[i], weight_vec4);
    }
  } else if constexpr (BITS == 8) {
    using U4 = vec<U, 4>;
    const device uint* weight_words = reinterpret_cast<const device uint*>(w);
    const thread U4* x_vec4 = reinterpret_cast<const thread U4*>(x_thread);
    METAL_PRAGMA_UNROLL
    for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
      // Mask each byte in place (no shifts); lane k value is b_k * 256^k. This
      // exceeds the magic-number range for k=3, so use the hardware convert
      // (exact for b_k * 256^k). x lane k was pre-divided by 256^k.
      const uint4 lanes =
          uint4(weight_words[i]) &
          uint4(0x000000ffu, 0x0000ff00u, 0x00ff0000u, 0xff000000u);
      const U4 weight_vec4 = U4(float4(lanes));
      accumulator += dot(x_vec4[i], weight_vec4);
    }
  }
  return scale * accumulator + sum * bias;
}

template <typename U, int VALUES_PER_THREAD, int BITS>
METAL_FUNC U qdot_codebook(
    const device uint8_t* w,
    const thread U* x_thread,
    const threadgroup half* codebook,
    U scale
) {
  static_assert(BITS == 4, "Only int4 codebook QMV is supported");

  using U4 = vec<U, 4>;
  U accumulator = 0;
  const device ushort* weight_words = reinterpret_cast<const device ushort*>(w);
  const thread U4* x_vec4 = reinterpret_cast<const thread U4*>(x_thread);

  for (int value_idx = 0; value_idx < (VALUES_PER_THREAD / 4); value_idx++) {
    uint weight_word = weight_words[value_idx];
    U4 weight_vec4 =
        U4(static_cast<U>(codebook[weight_word & 0x0fu]),
           static_cast<U>(codebook[(weight_word >> 4) & 0x0fu]),
           static_cast<U>(codebook[(weight_word >> 8) & 0x0fu]),
           static_cast<U>(codebook[(weight_word >> 12) & 0x0fu]));
    accumulator += dot(x_vec4[value_idx], weight_vec4);
  }

  return scale * accumulator;
}

template <typename U, int VALUES_PER_THREAD, int BITS>
METAL_FUNC U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N
) {
  static_assert(BITS == 4 || BITS == 8, "Only int4 and int8 supported");

  U accumulator = 0;
  if constexpr (BITS == 4) {
    using U4 = vec<U, 4>;
    const device uint16_t* weight_words =
        reinterpret_cast<const device uint16_t*>(w);
    const thread U4* x_vec4 = reinterpret_cast<const thread U4*>(x_thread);

    int full_chunks = N / 4;
    for (int i = 0; i < full_chunks; i++) {
      uint16_t weight_word = weight_words[i];
      U4 weight_vec4 = uint4_to_fp4<U, 4>(uint4(
          weight_word,
          weight_word >> 4,
          weight_word >> 8,
          weight_word >> 12
      ));
      accumulator += dot(x_vec4[i], weight_vec4);
    }

    int remainder = N & 3;
    if (remainder > 0) {
      uint16_t weight_word = weight_words[full_chunks];
      int base_index = 4 * full_chunks;
      accumulator += x_thread[base_index] * uint_to_fp<U>(weight_word & 0xf);
      if (remainder > 1)
        accumulator +=
            x_thread[base_index + 1] * uint_to_fp<U>((weight_word >> 4) & 0xf);
      if (remainder > 2)
        accumulator +=
            x_thread[base_index + 2] * uint_to_fp<U>((weight_word >> 8) & 0xf);
    }
  } else if constexpr (BITS == 8) {
    for (int i = 0; i < N; i++) {
      accumulator += x_thread[i] * w[i];
    }
  }

  return scale * accumulator + sum * bias;
}

} // namespace gemm
} // namespace uzu
