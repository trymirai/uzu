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

// NF4 qdot: 4-bit nibble → codebook value via a zero-memory in-thread
// switch-of-literals select (`nf4_select_entry`). No constant/threadgroup
// codebook, no cross-lane op, no array indexing. Numerically identical to
// `qdot_nf4_constant` (same 16 half literals; only the value source differs).
template <int values_per_thread>
inline float qdot_nf4_select(
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
    half h0 = nf4_select_entry(b0 & 0x0f);
    half h1 = nf4_select_entry((b0 >> 4) & 0x0f);
    half h2 = nf4_select_entry(b1 & 0x0f);
    half h3 = nf4_select_entry((b1 >> 4) & 0x0f);
    U4 w_vec = U4(float(h0), float(h1), float(h2), float(h3));
    accum += dot(x4[i], w_vec);
  }
  return scale * accum;
}

// NF4-ZP qdot: 4-bit nibble → codebook lookup (constant addr space) plus a
// per-group zero-point offset `zp_off` added to each codebook value before the
// scale multiply: out = scale * Σ (codebook[nibble] + zp_off) · x.
template <int values_per_thread>
inline float qdot_nf4_zp(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float zp_off
) {
  using U4 = vec<float, 4>;
  float accum = 0;
  const thread U4* x4 = (const thread U4*)x_thread;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    uint8_t b0 = w[2 * i];
    uint8_t b1 = w[2 * i + 1];
    float h0 = float(nf4_codebook[b0 & 0x0f]) + zp_off;
    float h1 = float(nf4_codebook[(b0 >> 4) & 0x0f]) + zp_off;
    float h2 = float(nf4_codebook[b1 & 0x0f]) + zp_off;
    float h3 = float(nf4_codebook[(b1 >> 4) & 0x0f]) + zp_off;
    U4 w_vec = U4(h0, h1, h2, h3);
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

// NF4 qdot: byte-batched 256-entry threadgroup half2 LUT. Each packed weight
// byte indexes a `half2` holding {codebook[low nibble], codebook[high nibble]}
// (NF4 codebook values, NOT integer nibbles). One threadgroup load yields two
// dequant values. The per-group scale is NOT in the table (the table is the
// global codebook); it is applied once on accumulate, exactly as
// `qdot_nf4_constant` does.
template <int values_per_thread>
inline float qdot_nf4_byte_lut(
    const device uint8_t* w,
    const thread float* x_thread,
    const threadgroup half2* lut,
    float scale
) {
  using U4 = vec<float, 4>;
  float accum = 0;
  const thread U4* x4 = (const thread U4*)x_thread;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    const half2 w01 = lut[w[2 * i]];
    const half2 w23 = lut[w[2 * i + 1]];
    const U4 w_vec = U4(float2(w01), float2(w23));
    accum += dot(x4[i], w_vec);
  }
  return scale * accum;
}

// NF4 qdot: byte-batched 256-entry threadgroup `half2` LUT, REPLICATED DUP
// times. The LUT base `lut` points at `pair_lut_dup[256 * DUP]`. For each
// weight byte `b`, the lane reads `lut[b * DUP + dup]` where `dup` is this
// lane's per-call replica selector (e.g. `simd_lane & (DUP - 1)`). Every
// replica slot holds the SAME codebook pair, so numerically identical to
// `qdot_nf4_byte_lut`; only the bank distribution of the load differs.
template <int values_per_thread, uint DUP>
inline float qdot_nf4_byte_lut_dup(
    const device uint8_t* w,
    const thread float* x_thread,
    const threadgroup half2* lut,
    uint dup,
    float scale
) {
  using U4 = vec<float, 4>;
  float accum = 0;
  const thread U4* x4 = (const thread U4*)x_thread;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    const half2 w01 = lut[uint(w[2 * i]) * DUP + dup];
    const half2 w23 = lut[uint(w[2 * i + 1]) * DUP + dup];
    const U4 w_vec = U4(float2(w01), float2(w23));
    accum += dot(x4[i], w_vec);
  }
  return scale * accum;
}

// NF4 qdot: zero-memory register-shuffle codebook. `my_entry` is THIS lane's
// register-held codebook value (lane i holds entry i for i<S). Each weight
// nibble n is dequantized via `simd_shuffle(my_entry, n)` — a cross-lane
// register op with NO memory/LSU traffic. The per-group scale is applied once
// on accumulate, exactly as `qdot_nf4_constant`. For S=8 the nibble is masked
// to 3 bits so the shuffle source lane stays in [0, 7]; for S=16/32 the 4-bit
// nibble (0..15) is a valid source lane. The shuffle op cost is independent
// of S.
template <int values_per_thread, uint S>
inline float qdot_nf4_shuffle(
    const device uint8_t* w,
    const thread float* x_thread,
    half my_entry,
    float scale
) {
  using U4 = vec<float, 4>;
  constexpr uint NIBBLE_MASK = (S == 8u) ? 0x07u : 0x0fu;
  float accum = 0;
  const thread U4* x4 = (const thread U4*)x_thread;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    uint8_t b0 = w[2 * i];
    uint8_t b1 = w[2 * i + 1];
    ushort n0 = ushort(b0 & NIBBLE_MASK);
    ushort n1 = ushort((b0 >> 4) & NIBBLE_MASK);
    ushort n2 = ushort(b1 & NIBBLE_MASK);
    ushort n3 = ushort((b1 >> 4) & NIBBLE_MASK);
    half h0 = simd_shuffle(my_entry, n0);
    half h1 = simd_shuffle(my_entry, n1);
    half h2 = simd_shuffle(my_entry, n2);
    half h3 = simd_shuffle(my_entry, n3);
    U4 w_vec = U4(float(h0), float(h1), float(h2), float(h3));
    accum += dot(x4[i], w_vec);
  }
  return scale * accum;
}

// NF4 qdot: vec4-replicated TG codebook (half[16][4]). Lane reads
// `cb[nibble*4 + replica]`. Replica is `simd_lane & 3`.
template <int values_per_thread>
inline float qdot_nf4_vec4(
    const device uint8_t* w,
    const thread float* x_thread,
    const threadgroup half* cb,
    uint replica,
    float scale
) {
  using U4 = vec<float, 4>;
  float accum = 0;
  const thread U4* x4 = (const thread U4*)x_thread;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    uint8_t b0 = w[2 * i];
    uint8_t b1 = w[2 * i + 1];
    half h0 = cb[((b0 & 0x0fu) << 2) | replica];
    half h1 = cb[(((b0 >> 4) & 0x0fu) << 2) | replica];
    half h2 = cb[((b1 & 0x0fu) << 2) | replica];
    half h3 = cb[(((b1 >> 4) & 0x0fu) << 2) | replica];
    U4 w_vec = U4(float(h0), float(h1), float(h2), float(h3));
    accum += dot(x4[i], w_vec);
  }
  return scale * accum;
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

// Multi-accumulator variant of qdot_nf4_tg. Breaks the per-iter
// load->convert->dot->accum dependency chain by maintaining 2 independent
// accumulators. Each outer iter issues 4 byte loads + 8 threadgroup loads
// back-to-back before any dot consumes them, exposing more ILP and letting
// more loads be in flight under the memory-latency-bound regime that
// Nf4QmvTg hits on M4 (Instruction Throughput Limiter ~90%).
//
// Numerically: same 16 NF4 codebook values dequantized in the same order,
// but the accumulation order differs (two partial sums combined at the end
// instead of one). Within bf16 ULP slack of qdot_nf4_tg.
template <int values_per_thread>
inline float qdot_nf4_tg_ilp(
    const device uint8_t* w,
    const thread float* x_thread,
    const threadgroup half* cb,
    float scale
) {
  using U4 = vec<float, 4>;
  float accum0 = 0;
  float accum1 = 0;
  const thread U4* x4 = (const thread U4*)x_thread;
  // values_per_thread=16, so 2 outer iters, each processing 8 values via 2
  // independent accumulators.
  for (int i = 0; i < (values_per_thread / 8); i++) {
    // Issue 4 byte loads.
    uint8_t b0 = w[4 * i];
    uint8_t b1 = w[4 * i + 1];
    uint8_t b2 = w[4 * i + 2];
    uint8_t b3 = w[4 * i + 3];
    // Issue 8 TG loads back-to-back (no immediate consumer).
    half h0 = cb[b0 & 0x0f];
    half h1 = cb[(b0 >> 4) & 0x0f];
    half h2 = cb[b1 & 0x0f];
    half h3 = cb[(b1 >> 4) & 0x0f];
    half h4 = cb[b2 & 0x0f];
    half h5 = cb[(b2 >> 4) & 0x0f];
    half h6 = cb[b3 & 0x0f];
    half h7 = cb[(b3 >> 4) & 0x0f];
    // Two independent dots → write to different accumulators.
    U4 v0 = U4(float(h0), float(h1), float(h2), float(h3));
    U4 v1 = U4(float(h4), float(h5), float(h6), float(h7));
    accum0 += dot(x4[2 * i], v0);
    accum1 += dot(x4[2 * i + 1], v1);
  }
  return scale * (accum0 + accum1);
}
