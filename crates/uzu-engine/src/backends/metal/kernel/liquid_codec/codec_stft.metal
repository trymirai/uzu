#include <metal_stdlib>
#include "../common/dsl.h"
#include "codec_fft.h"

using namespace metal;

// ---------------------------------------------------------------------------
// Liquid-Codec analysis / synthesis transforms.
//
// Fusions, by construction:
// - CodecStft: Hann window * input, 1280-point FFT and the
//   [log1p(|X|) | cos | sin] feature extraction happen in one pass; no
//   intermediate spectrum touches global memory.
// - CodecIstft: polar reconstruction (exp / cos / sin), the inverse FFT and the
//   synthesised Hann window are fused per frame.
// - CodecOverlapAdd: overlap-add of the 78 frame buffers against a
//   precomputed reciprocal Hann^2 envelope (computed once at load), so the
//   normalisation costs one multiply, not a division.
// ---------------------------------------------------------------------------

#define CODEC_N_FFT 1280u
#define CODEC_HOP 320u
#define CODEC_BINS 641u

// forward: audio [padded], window [1280], tw [1280] ->
// features [frames, 3 * 641] in [log1p(|X|) | cos | sin] order.
KERNEL(CodecStft)(
    device const float* audio,
    device const float* window,
    device const float2* tw,
    device bfloat* features,
    constant const uint& num_frames,
    const uint frame GROUPS(num_frames),
    threadgroup float2 sbuf[CODEC_N_FFT],
    const uint tid THREADS(256)
) {
  codec_scatter_real(sbuf, audio + frame * CODEC_HOP, window, tid);
  codec_fft1280(sbuf, tw, tid);

  const uint row = frame * (3u * CODEC_BINS);
  for (uint r = 0; r < 5u; ++r) {
    const uint k = tid + 256u * r;
    if (k > 640u) {
      continue;
    }
    const float2 x = codec_combine(sbuf, tw, k);
    const float mag = max(sqrt(x.x * x.x + x.y * x.y), 1e-5f);
    features[row + k] = bfloat(log(mag + 1.0f));
    features[row + CODEC_BINS + k] = bfloat(x.x / mag);
    features[row + 2u * CODEC_BINS + k] = bfloat(x.y / mag);
  }
}

// inverse: dec_lin output [frames, 2 * 641] ([log|X| | angle] interleaved) ->
// windowed synthesised frames [frames, 1280] (float, overlap-added later).
KERNEL(CodecIstft)(
    device const bfloat* dec_out,
    device const float* window,
    device const float2* tw,
    device float* frames_out,
    constant const uint& num_frames,
    constant const uint& row_stride,
    const uint frame GROUPS(num_frames),
    threadgroup float2 sbuf[CODEC_N_FFT],
    const uint tid THREADS(256)
) {
  const uint row = frame * row_stride;

  // X[k] with Hermitian symmetry, conjugated for the forward-FFT identity
  // ifft(X) = conj(fft(conj(X))) / n.
  for (uint idx = tid; idx < CODEC_N_FFT; idx += 256u) {
    const uint s = idx / 256u;
    const uint j = idx % 256u;
    const uint k = 5u * codec_rev8(j) + s;
    const uint bin = k <= 640u ? k : CODEC_N_FFT - k;
    const float mag = exp(float(dec_out[row + bin]));
    const float a = float(dec_out[row + CODEC_BINS + bin]);
    const float re = mag * cos(a);
    const float im = mag * sin(a);
    const bool mirrored = k > 640u;
    sbuf[idx] = float2(re, mirrored ? im : -im); // conj(X) (and conj for mirrored bins)
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  codec_fft1280(sbuf, tw, tid);

  for (uint r = 0; r < 5u; ++r) {
    const uint n = tid + 256u * r;
    const float2 y = codec_combine(sbuf, tw, n);
    frames_out[frame * CODEC_N_FFT + n] = (y.x / float(CODEC_N_FFT)) * window[n];
  }
}

// Plain overlap-add: audio[s] = sum_t frames[t][s - t*hop]. Normalisation
// against the fold(hann^2) envelope happens on the host, because the streaming
// path has to accumulate audio and envelope across chunks separately (the
// reference divides only once, after summing both).
KERNEL(CodecOverlapAdd)(
    device const float* frames,
    device float* audio,
    constant const uint& num_frames,
    const uint sample AXIS((num_frames - 1u) * 320u + 1280u, 256)
) {
  const uint t_last = min(sample / 320u, num_frames - 1u);
  const uint t_first = sample >= 1279u ? (sample - 1279u + 319u) / 320u : 0u;
  float acc = 0.0f;
  for (uint t = t_first; t <= t_last; ++t) {
    acc += frames[t * CODEC_N_FFT + (sample - t * 320u)];
  }
  audio[sample] = acc;
}
