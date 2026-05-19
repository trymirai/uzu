#pragma once

#include <metal_stdlib>

#include "../../common/defines.h"
#include "quant_unpack.h"

using namespace metal;

namespace uzu {
namespace gemm {

// Inner-loop helpers for the standalone QMV / QMV-Fast kernels. The unified
// `GemmQuantKernel` does not use these — its dequantize path is the block
// loader in `quant_scale_bias.h` / `quant_scale_zero_point.h`.

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  using U4 = vec<U, 4>;
  U sum = 0;
  thread U4* x4 = (thread U4*)x_thread;
  for (int i = 0; i < values_per_thread / 4; i++) {
    U4 v = U4(x[4 * i], x[4 * i + 1], x[4 * i + 2], x[4 * i + 3]);
    sum += v[0] + v[1] + v[2] + v[3];
    x4[i] = v;
  }
  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  U sum = 0;
  for (int i = 0; i < values_per_thread; ++i) {
    x_thread[i] = 0;
  }
  for (int i = 0; i < N; ++i) {
    U v = x[i];
    sum += v;
    x_thread[i] = v;
  }
  return sum;
}

template <typename U, int values_per_thread, int bits>
inline void qouter(
    const thread uint8_t* w,
    U x,
    U scale,
    U bias,
    thread U* result
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  if (bits == 4) {
    for (int i = 0; i < (values_per_thread / 2); i++) {
      result[2 * i] += x * (scale * uint_to_fp<U>(w[i] & 0x0fu) + bias);
      result[2 * i + 1] +=
          x * (scale * uint_to_fp<U>((w[i] >> 4) & 0x0fu) + bias);
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      result[i] += x * (scale * w[i] + bias);
    }
  }
}

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  U accum = 0;
  if (bits == 4) {
    using U4 = vec<U, 4>;
    const device ushort* ws = (const device ushort*)w;
    const thread U4* x4 = (const thread U4*)x_thread;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint wi = ws[i];
      U4 w_vec = uint4_to_fp4<U, 4>(uint4(wi, wi >> 4, wi >> 8, wi >> 12));
      accum += dot(x4[i], w_vec);
    }
  } else if (bits == 8) {
    using U4 = vec<U, 4>;
    const device uint* ws = (const device uint*)w;
    const thread U4* x4 = (const thread U4*)x_thread;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint wi = ws[i];
      U4 w_vec = uint4_to_fp4<U, 8>(uint4(wi, wi >> 8, wi >> 16, wi >> 24));
      accum += dot(x4[i], w_vec);
    }
  }
  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  U accum = 0;
  if (bits == 4) {
    using U4 = vec<U, 4>;
    const device uint16_t* ws = (const device uint16_t*)w;
    const thread U4* x4 = (const thread U4*)x_thread;

    int full = N / 4;
    for (int i = 0; i < full; i++) {
      uint16_t wi = ws[i];
      U4 w_vec = uint4_to_fp4<U, 4>(uint4(wi, wi >> 4, wi >> 8, wi >> 12));
      accum += dot(x4[i], w_vec);
    }

    int rem = N & 3;
    if (rem > 0) {
      uint16_t wv = ws[full];
      int base = 4 * full;
      if (rem > 0)
        accum += x_thread[base] * uint_to_fp<U>(wv & 0xf);
      if (rem > 1)
        accum += x_thread[base + 1] * uint_to_fp<U>((wv >> 4) & 0xf);
      if (rem > 2)
        accum += x_thread[base + 2] * uint_to_fp<U>((wv >> 8) & 0xf);
    }
  } else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

} // namespace gemm
} // namespace uzu
