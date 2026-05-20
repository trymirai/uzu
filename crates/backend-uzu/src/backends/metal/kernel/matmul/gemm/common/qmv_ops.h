#pragma once

#include <metal_stdlib>

#include "../../common/defines.h"
#include "quant_unpack.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <typename T, typename U, int VALUES_PER_THREAD, int BITS>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(BITS == 4 || BITS == 8, "Only int4 and int8 supported");

  using U4 = vec<U, 4>;
  U sum = 0;
  thread U4* x_vec4 = (thread U4*)x_thread;
  for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
    U4 v = U4(x[4 * i], x[4 * i + 1], x[4 * i + 2], x[4 * i + 3]);
    sum += v[0] + v[1] + v[2] + v[3];
    x_vec4[i] = v;
  }
  return sum;
}

template <typename T, typename U, int VALUES_PER_THREAD, int BITS>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(BITS == 4 || BITS == 8, "Only int4 and int8 supported");

  U sum = 0;
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
inline void qouter(
    const thread uint8_t* w,
    U x,
    U scale,
    U bias,
    thread U* result
) {
  static_assert(BITS == 4 || BITS == 8, "Only int4 and int8 supported");

  if (BITS == 4) {
    for (int i = 0; i < (VALUES_PER_THREAD / 2); i++) {
      result[2 * i] += x * (scale * uint_to_fp<U>(w[i] & 0x0fu) + bias);
      result[2 * i + 1] +=
          x * (scale * uint_to_fp<U>((w[i] >> 4) & 0x0fu) + bias);
    }
  } else if (BITS == 8) {
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      result[i] += x * (scale * w[i] + bias);
    }
  }
}

template <typename U, int VALUES_PER_THREAD, int BITS>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum
) {
  static_assert(BITS == 4 || BITS == 8, "Only int4 and int8 supported");

  U accumulator = 0;
  if (BITS == 4) {
    using U4 = vec<U, 4>;
    const device ushort* weight_words = (const device ushort*)w;
    const thread U4* x_vec4 = (const thread U4*)x_thread;
    for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
      uint weight_word = weight_words[i];
      U4 weight_vec4 = uint4_to_fp4<U, 4>(uint4(
          weight_word,
          weight_word >> 4,
          weight_word >> 8,
          weight_word >> 12
      ));
      accumulator += dot(x_vec4[i], weight_vec4);
    }
  } else if (BITS == 8) {
    using U4 = vec<U, 4>;
    const device uint* weight_words = (const device uint*)w;
    const thread U4* x_vec4 = (const thread U4*)x_thread;
    for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
      uint weight_word = weight_words[i];
      U4 weight_vec4 = uint4_to_fp4<U, 8>(uint4(
          weight_word,
          weight_word >> 8,
          weight_word >> 16,
          weight_word >> 24
      ));
      accumulator += dot(x_vec4[i], weight_vec4);
    }
  }
  return scale * accumulator + sum * bias;
}

template <typename U, int VALUES_PER_THREAD, int BITS>
inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N
) {
  static_assert(BITS == 4 || BITS == 8, "Only int4 and int8 supported");

  U accumulator = 0;
  if (BITS == 4) {
    using U4 = vec<U, 4>;
    const device uint16_t* weight_words = (const device uint16_t*)w;
    const thread U4* x_vec4 = (const thread U4*)x_thread;

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
      if (remainder > 0)
        accumulator += x_thread[base_index] * uint_to_fp<U>(weight_word & 0xf);
      if (remainder > 1)
        accumulator +=
            x_thread[base_index + 1] * uint_to_fp<U>((weight_word >> 4) & 0xf);
      if (remainder > 2)
        accumulator +=
            x_thread[base_index + 2] * uint_to_fp<U>((weight_word >> 8) & 0xf);
    }
  } else if (BITS == 8) {
    for (int i = 0; i < N; i++) {
      accumulator += x_thread[i] * w[i];
    }
  }

  return scale * accumulator + sum * bias;
}

} // namespace gemm
} // namespace uzu
