#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

// ---------------------------------------------------------------------------
// LFM3-Audio speech encoder downsampler & projector kernels (BF16).
// ---------------------------------------------------------------------------

KERNEL(AudioDownsamplerConvIn)(
    device const float* input,
    device const bfloat* kernel_w,
    device const bfloat* bias,
    device bfloat* output,
    constant const uint& in_f,
    constant const uint& in_t,
    constant const uint& out_f,
    constant const uint& out_t,
    constant const uint& out_ch,
    constant const uint& out_valid,
    const uint index AXIS(out_f * out_t * out_ch, 256)
) {
  const uint oc = index % out_ch;
  const uint ft = index / out_ch;
  const uint t = ft % out_t;
  const uint f = ft / out_t;

  if (t >= out_valid) {
    output[index] = bfloat(0.0f);
    return;
  }

  float sum = float(bias[oc]);
  const uint w_base = oc * 9u;
  for (uint kh = 0u; kh < 3u; ++kh) {
    const int sf = int(f * 2u + kh) - 1;
    if (sf < 0 || uint(sf) >= in_f) continue;
    for (uint kw = 0u; kw < 3u; ++kw) {
      const int st = int(t * 2u + kw) - 1;
      if (st < 0 || uint(st) >= in_t) continue;
      sum += input[uint(sf) * in_t + uint(st)] * float(kernel_w[w_base + kh * 3u + kw]);
    }
  }
  const float v = float(bfloat(sum));
  output[index] = bfloat(v / (1.0f + exp(-v)));
}

KERNEL(AudioDownsamplerDepthwiseConv)(
    device const bfloat* input,
    device const bfloat* kernel_w,
    device const bfloat* bias,
    device bfloat* output,
    constant const uint& in_f,
    constant const uint& in_t,
    constant const uint& out_f,
    constant const uint& out_t,
    constant const uint& ch,
    const uint index AXIS(out_f * out_t * ch, 256)
) {
  const uint oc = index % ch;
  const uint ft = index / ch;
  const uint t = ft % out_t;
  const uint f = ft / out_t;

  float sum = float(bias[oc]);
  const uint w_base = oc * 9u;
  for (uint kh = 0u; kh < 3u; ++kh) {
    const int sf = int(f * 2u + kh) - 1;
    if (sf < 0 || uint(sf) >= in_f) continue;
    for (uint kw = 0u; kw < 3u; ++kw) {
      const int st = int(t * 2u + kw) - 1;
      if (st < 0 || uint(st) >= in_t) continue;
      sum += float(input[(uint(sf) * in_t + uint(st)) * ch + oc]) * float(kernel_w[w_base + kh * 3u + kw]);
    }
  }
  output[index] = bfloat(sum);
}

KERNEL(AudioSiluMask)(
    device bfloat* data,
    constant const uint& total,
    constant const uint& out_t,
    constant const uint& ch,
    constant const uint& out_valid,
    const uint index AXIS(total, 256)
) {
  const uint t = (index / ch) % out_t;
  if (t < out_valid) {
    const float v = float(data[index]);
    data[index] = bfloat(v / (1.0f + exp(-v)));
  } else {
    data[index] = bfloat(0.0f);
  }
}

KERNEL(AudioDownsamplerFlatten)(
    device const bfloat* input,
    device bfloat* output,
    constant const uint& out_f,
    constant const uint& out_t,
    constant const uint& ch,
    constant const uint& tokens,
    const uint index AXIS(tokens * ch * out_f, 256)
) {
  const uint cf = ch * out_f;
  const uint t = index / cf;
  const uint rem = index % cf;
  const uint c = rem / out_f;
  const uint f = rem % out_f;
  output[index] = input[(f * out_t + t) * ch + c];
}

KERNEL(AudioLayerNorm)(
    device const bfloat* input,
    device const bfloat* gamma,
    device const bfloat* beta,
    device bfloat* output,
    constant const uint& tokens,
    constant const uint& hidden,
    constant const float& eps,
    const uint token GROUPS(tokens),
    const uint tid THREADS(32)
) {
  const uint base = token * hidden;

  float local_sum = 0.0f;
  for (uint d = tid; d < hidden; d += 32u) {
    local_sum += float(input[base + d]);
  }
  const float mean = simd_sum(local_sum) / float(hidden);

  float local_var = 0.0f;
  for (uint d = tid; d < hidden; d += 32u) {
    const float diff = float(input[base + d]) - mean;
    local_var += diff * diff;
  }
  const float inv_std = rsqrt(simd_sum(local_var) / float(hidden) + eps);

  for (uint d = tid; d < hidden; d += 32u) {
    const float v = float(input[base + d]);
    output[base + d] = bfloat((v - mean) * inv_std * float(gamma[d]) + float(beta[d]));
  }
}

KERNEL(AudioGeluErf)(
    device bfloat* data,
    constant const uint& count,
    const uint index AXIS(count, 256)
) {
  const float x = float(data[index]);
  const float ax = abs(x * 0.7071067811865475f);
  const float t = 1.0f / (1.0f + 0.3275911f * ax);
  const float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
  const float erf_abs = 1.0f - poly * exp(-ax * ax);
  const float erf_val = (x >= 0.0f) ? erf_abs : -erf_abs;
  data[index] = bfloat(0.5f * x * (1.0f + erf_val));
}
