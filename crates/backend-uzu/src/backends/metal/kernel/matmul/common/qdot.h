#pragma once

#include <metal_stdlib>

#include "../../common/defines.h"
#include "quant_unpack.h"

using namespace metal;

namespace uzu {
namespace gemm {

// For packed unsigned weights, pre-scales each activation lane k by
// 2^-(BITS*k). qdot then reads the matching nibble/byte in place (value =
// q * 2^(BITS*k)); the factors cancel, so the dot product is unchanged. All
// factors are powers of two, so this is bit-exact.
// Signed int8 weights are already directly loadable and need no pre-scaling.
template <typename T, typename U, int VALUES_PER_THREAD, int BITS>
METAL_FUNC U load_vector(const device T* x, thread U* x_thread, const bool signed_codes) {
  using U4 = vec<U, 4>;
  const U4 inv = (BITS == 8 && signed_codes)
                     ? U4(U(1))
                     : U4(U(1), U(1) / U(1u << BITS), U(1) / U(1u << (2u * BITS)), U(1) / U(1u << (3u * BITS)));
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
METAL_FUNC U qdot(const device uint8_t* w, const thread U* x_thread, U scale, U bias, U sum, const bool signed_codes) {
  static_assert(BITS == 4 || BITS == 8, "Only int4 and int8 supported");

  U accumulator = 0;
  if constexpr (BITS == 4) {
    using U4 = vec<U, 4>;
    const device ushort* weight_words = reinterpret_cast<const device ushort*>(w);
    const thread U4* x_vec4 = reinterpret_cast<const thread U4*>(x_thread);
    const ushort packed_mask = signed_codes ? 0x8888u : 0u;
    METAL_PRAGMA_UNROLL
    for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
      // Mask each nibble in place (no shifts); value of lane k is n_k << (4*k),
      // i.e. n_k * 16^k, which is < 2^23 so the magic-number convert is valid.
      // The matching x lane was pre-divided by 16^k in load_vector.
      const uint4 lanes = uint4(ushort(weight_words[i] ^ packed_mask)) & uint4(0x000fu, 0x00f0u, 0x0f00u, 0xf000u);
      const U4 weight_vec4 = U4(as_type<float4>(lanes | uint4(0x4b000000u)) - float4(8388608.0f));
      accumulator += dot(x_vec4[i], weight_vec4);
    }
  } else if constexpr (BITS == 8) {
    using U4 = vec<U, 4>;
    const thread U4* x_vec4 = reinterpret_cast<const thread U4*>(x_thread);
    if (signed_codes) {
      const device char4* weight_vectors = reinterpret_cast<const device char4*>(w);
      METAL_PRAGMA_UNROLL
      for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
        accumulator += dot(x_vec4[i], U4(weight_vectors[i]));
      }
    } else {
      const device uint* weight_words = reinterpret_cast<const device uint*>(w);
      METAL_PRAGMA_UNROLL
      for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
        // Keep each byte in place. Lane k is b_k * 256^k, while the
        // matching activation lane was pre-divided by 256^k.
        const uint4 lanes = uint4(weight_words[i]) & uint4(0x000000ffu, 0x0000ff00u, 0x00ff0000u, 0xff000000u);
        accumulator += dot(x_vec4[i], U4(float4(lanes)));
      }
    }
  }
  const U adjusted_bias = (BITS == 8 && signed_codes) ? bias + scale * U(128) : bias;
  return scale * accumulator + sum * adjusted_bias;
}

template <typename U, int VALUES_PER_THREAD, int BITS>
METAL_FUNC U
qdot_safe(const device uint8_t* w, const thread U* x_thread, U scale, U bias, U sum, int N, const bool signed_codes) {
  static_assert(BITS == 4 || BITS == 8, "Only int4 and int8 supported");

  U accumulator = 0;
  if constexpr (BITS == 4) {
    using U4 = vec<U, 4>;
    const device uint16_t* weight_words = reinterpret_cast<const device uint16_t*>(w);
    const thread U4* x_vec4 = reinterpret_cast<const thread U4*>(x_thread);
    const uint16_t packed_mask = signed_codes ? 0x8888u : 0u;

    int full_chunks = N / 4;
    for (int i = 0; i < full_chunks; i++) {
      uint16_t weight_word = weight_words[i] ^ packed_mask;
      U4 weight_vec4 = uint4_to_fp4<U, 4>(uint4(weight_word, weight_word >> 4, weight_word >> 8, weight_word >> 12));
      accumulator += dot(x_vec4[i], weight_vec4);
    }

    int remainder = N & 3;
    if (remainder > 0) {
      uint16_t weight_word = weight_words[full_chunks] ^ packed_mask;
      int base_index = 4 * full_chunks;
      accumulator += x_thread[base_index] * uint_to_fp<U>(weight_word & 0xf);
      if (remainder > 1)
        accumulator += x_thread[base_index + 1] * uint_to_fp<U>((weight_word >> 4) & 0xf);
      if (remainder > 2)
        accumulator += x_thread[base_index + 2] * uint_to_fp<U>((weight_word >> 8) & 0xf);
    }
  } else if constexpr (BITS == 8) {
    if (signed_codes) {
      const device int8_t* signed_weights = reinterpret_cast<const device int8_t*>(w);
      for (int i = 0; i < N; i++) {
        accumulator += x_thread[i] * U(signed_weights[i]);
      }
    } else {
      for (int i = 0; i < N; i++) {
        accumulator += x_thread[i] * U(w[i]);
      }
    }
  }

  const U adjusted_bias = (BITS == 8 && signed_codes) ? bias + scale * U(128) : bias;
  return scale * accumulator + sum * adjusted_bias;
}

} // namespace gemm
} // namespace uzu
