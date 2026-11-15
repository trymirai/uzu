#pragma once

#include <metal_stdlib>

using namespace metal;

// Shared 1280-point complex FFT for the Liquid-Codec STFT/ISTFT pair.
//
// 1280 = 5 * 256, so the transform is a radix-5/radix-2 Cooley-Tukey:
//   X[k] = sum_{s=0..4} W_1280^(s*k) * E_s[k mod 256],
//   E_s = FFT_256(x[5*j + s]).
// The caller pre-scatters the input into bit-reversed order per subsequence
// (`sbuf[s*256 + j] = x[5*rev8(j) + s]`), and passes a twiddle table
// `tw[k] = exp(-2*pi*i*k/1280)` in f32. Everything stays in threadgroup memory
// (1280 float2 = 10 KB), one threadgroup per frame.

inline uint codec_rev8(uint v) {
  v = ((v & 0xF0u) >> 4) | ((v & 0x0Fu) << 4);
  v = ((v & 0xCCu) >> 2) | ((v & 0x33u) << 2);
  v = ((v & 0xAAu) >> 1) | ((v & 0x55u) << 1);
  return v;
}

// Unscaled forward FFT (the `e^{-i...}` sign). Caller scatters the input.
inline void codec_fft1280(threadgroup float2* sbuf, device const float2* tw, uint tid) {
  // 5 interleaved 256-point transforms, 640 butterflies per stage.
  for (uint stage = 0; stage < 8u; ++stage) {
    const uint len = 2u << stage;
    const uint half_len = len >> 1;
    const uint tw_stride = 1280u / len;
    for (uint b = tid; b < 640u; b += 256u) {
      const uint s = b / 128u;
      const uint butterfly = b % 128u;
      const uint group = butterfly / half_len;
      const uint k = butterfly % half_len;
      const uint base = s * 256u;
      const uint i1 = base + group * len + k;
      const uint i2 = i1 + half_len;
      const float2 w = tw[(k * tw_stride) % 1280u];
      const float2 a = sbuf[i1];
      const float2 c = sbuf[i2];
      const float2 t = float2(c.x * w.x - c.y * w.y, c.x * w.y + c.y * w.x);
      sbuf[i1] = float2(a.x + t.x, a.y + t.y);
      sbuf[i2] = float2(a.x - t.x, a.y - t.y);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Scatter real input `x[5*j + s]` (already windowed) into bit-reversed,
// subsequence-major threadgroup layout.
inline void codec_scatter_real(
    threadgroup float2* sbuf,
    device const float* x,
    device const float* window,
    uint tid
) {
  for (uint idx = tid; idx < 1280u; idx += 256u) {
    // idx = s*256 + j  ->  source sample 5*rev8(j) + s
    const uint s = idx / 256u;
    const uint j = idx % 256u;
    const uint src = 5u * codec_rev8(j) + s;
    sbuf[idx] = float2(x[src] * window[src], 0.0f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Combine the five 256-point transforms into bin `k` of the 1280-point output.
inline float2 codec_combine(
    threadgroup const float2* sbuf,
    device const float2* tw,
    uint k
) {
  const uint j = k % 256u;
  float2 acc = float2(0.0f, 0.0f);
  for (uint s = 0; s < 5u; ++s) {
    const float2 e = sbuf[s * 256u + j];
    const float2 w = (s == 0u) ? float2(1.0f, 0.0f) : tw[(s * k) % 1280u];
    acc += float2(e.x * w.x - e.y * w.y, e.x * w.y + e.y * w.x);
  }
  return acc;
}
