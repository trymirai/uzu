#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/dsl.h"
#include "nf4_common.h"
#include "quant_matmul.h"

using namespace metal;

// NF4 qdot: 4-bit nibble → codebook lookup via constant address space.
template <int values_per_thread>
inline float qdot_nf4_constant(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale
) {
  using U4 = vec<float, 4>;
  float accum = 0;
  const thread U4* x4 = (const thread U4*)x_thread;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    uint8_t b0 = w[2 * i];
    uint8_t b1 = w[2 * i + 1];
    half h0 = nf4_codebook[b0 & 0x0f];
    half h1 = nf4_codebook[(b0 >> 4) & 0x0f];
    half h2 = nf4_codebook[b1 & 0x0f];
    half h3 = nf4_codebook[(b1 >> 4) & 0x0f];
    U4 w_vec = U4(float(h0), float(h1), float(h2), float(h3));
    accum += dot(x4[i], w_vec);
  }
  return scale * accum;
}

// NF4 qdot with an E4M3 (1-byte FP8) per-group scale. Identical to
// qdot_nf4_constant except the caller passes the raw FP8 byte and we
// decode it once here. Decode is amortized once per group.
template <int values_per_thread>
inline float qdot_nf4_e4m3(
    const device uint8_t* w,
    const thread float* x_thread,
    uint8_t scale_fp8
) {
  using U4 = vec<float, 4>;
  float accum = 0;
  const thread U4* x4 = (const thread U4*)x_thread;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    uint8_t b0 = w[2 * i];
    uint8_t b1 = w[2 * i + 1];
    half h0 = nf4_codebook[b0 & 0x0f];
    half h1 = nf4_codebook[(b0 >> 4) & 0x0f];
    half h2 = nf4_codebook[b1 & 0x0f];
    half h3 = nf4_codebook[(b1 >> 4) & 0x0f];
    U4 w_vec = U4(float(h0), float(h1), float(h2), float(h3));
    accum += dot(x4[i], w_vec);
  }
  return float(e4m3_to_half(scale_fp8)) * accum;
}

// NF4 qdot: 4-bit nibble → codebook lookup via threadgroup memory.
template <int values_per_thread>
inline float qdot_nf4_tg(
    const device uint8_t* w,
    const thread float* x_thread,
    const threadgroup half* cb,
    float scale
) {
  using U4 = vec<float, 4>;
  float accum = 0;
  const thread U4* x4 = (const thread U4*)x_thread;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    uint8_t b0 = w[2 * i];
    uint8_t b1 = w[2 * i + 1];
    half h0 = cb[b0 & 0x0f];
    half h1 = cb[(b0 >> 4) & 0x0f];
    half h2 = cb[b1 & 0x0f];
    half h3 = cb[(b1 >> 4) & 0x0f];
    U4 w_vec = U4(float(h0), float(h1), float(h2), float(h3));
    accum += dot(x4[i], w_vec);
  }
  return scale * accum;
}
