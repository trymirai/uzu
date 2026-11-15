#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;

// ---------------------------------------------------------------------------
// Haru-4B three-stream audio/text glue, MeanFlowHead, and VibeVoiceCodec kernels.
// Non-PUBLIC KERNEL definitions generate `*MetalKernel` directly for Metal backend.
// ---------------------------------------------------------------------------

KERNEL(HaruLatentAffine)(
    const device float* z,
    const device float* scale,
    const device float* bias,
    device bfloat* out,
    const constant uint& total,
    const uint gid AXIS(total, 256)
) {
  if (gid >= total) {
    return;
  }
  out[gid] = bfloat(fma(z[gid], scale[0], bias[0]));
}

KERNEL(HaruSampleUserLatent)(
    const device bfloat* mean_bf16,
    const device float* unit_noise,
    device bfloat* out_bf16,
    const constant float& noise_scale,
    const constant float& proj_scale,
    const constant float& proj_bias,
    const constant uint& total,
    const uint gid AXIS(total, 256)
) {
  if (gid >= total) {
    return;
  }
  const float z = fma(unit_noise[gid], noise_scale, float(mean_bf16[gid]));
  out_bf16[gid] = bfloat(fma(z, proj_scale, proj_bias));
}

KERNEL(HaruAddBias)(
    device bfloat* x,
    const device float* bias,
    const constant uint& dim,
    const constant uint& rows,
    const uint c AXIS(dim, 256),
    const uint r AXIS(rows, 1)
) {
  if (c >= dim || r >= rows) {
    return;
  }
  const size_t idx = size_t(r) * dim + c;
  x[idx] = bfloat(float(x[idx]) + bias[c]);
}

KERNEL(HaruStreamEmbedSum)(
    const device bfloat* tok_emb,
    const device bfloat* model_emb,
    const device bfloat* user_emb,
    const device uint* model_mask,
    const device uint* user_mask,
    device bfloat* out,
    const constant uint& dim,
    const constant uint& seq_len,
    const uint c AXIS(dim, 256),
    const uint l AXIS(seq_len, 1)
) {
  if (c >= dim || l >= seq_len) {
    return;
  }
  const size_t idx = size_t(l) * dim + c;
  float v = float(tok_emb[idx]);
  if (model_mask[l] != 0u) {
    v += float(model_emb[idx]);
  }
  if (user_mask[l] != 0u) {
    v += float(user_emb[idx]);
  }
  out[idx] = bfloat(v);
}

template <typename T>
VARIANTS(T, bfloat, float)
KERNEL(HaruPatchLogitsAndDry)(
    const device bfloat* final_hidden,
    const device float* pad_row_fp32,
    const device float* eop_row_fp32,
    device T* logits,
    const device uint* dry_ids,
    const device float* dry_pens,
    const constant uint& dim,
    const constant uint& pad_id,
    const constant uint& eop_id,
    const constant uint& eos_id,
    const constant float& eop_bias,
    const constant uint& dry_count,
    const constant uint& force_text,
    threadgroup float sh_p[32],
    threadgroup float sh_e[32],
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint sg THREADS(8u),
    const uint tg_unused GROUPS(1)
) {
  (void)tg_unused;
  const uint tid = sg * METAL_SIMD_SIZE + lane;
  const uint threads = 8u * METAL_SIMD_SIZE;
  float dp = 0.0f;
  float de = 0.0f;
  for (uint c = tid; c < dim; c += threads) {
    const float h = float(final_hidden[c]);
    dp = fma(h, pad_row_fp32[c], dp);
    de = fma(h, eop_row_fp32[c], de);
  }
  dp = simd_sum(dp);
  de = simd_sum(de);
  if (lane == 0u) {
    sh_p[sg] = dp;
    sh_e[sg] = de;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    const float sp = simd_sum(lane < 8u ? sh_p[lane] : 0.0f);
    const float se = simd_sum(lane < 8u ? sh_e[lane] : 0.0f);
    if (lane == 0u) {
      logits[pad_id] = T(sp);
      logits[eop_id] = T(se + eop_bias);
      if (force_text != 0u) {
        logits[pad_id] = T(-INFINITY);
        logits[eop_id] = T(-INFINITY);
        logits[eos_id] = T(-INFINITY);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint i = tid; i < dry_count; i += threads) {
    const uint tok = dry_ids[i];
    if (force_text == 0u || (tok != pad_id && tok != eop_id && tok != eos_id)) {
      logits[tok] = T(float(logits[tok]) - dry_pens[i]);
    }
  }
}

KERNEL(HaruAddSilu)(
    const device bfloat* cond,
    const device float* static_time_emb,
    device bfloat* out,
    const constant uint& dim,
    const uint gid AXIS(dim, 256)
) {
  if (gid >= dim) {
    return;
  }
  const float c = float(cond[gid]) + static_time_emb[gid];
  out[gid] = bfloat(c / (1.0f + fast::exp(-c)));
}

KERNEL(HaruAdaLnModulate)(
    const device bfloat* x,
    const device float* norm_w,
    const device bfloat* modulation,
    device bfloat* out,
    const constant uint& dim,
    const constant uint& mod_offset,
    const constant uint& has_norm_w,
    const constant float& eps,
    threadgroup float sh[32],
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint sg THREADS(8u),
    const uint tg_unused GROUPS(1)
) {
  (void)tg_unused;
  const uint tid = sg * METAL_SIMD_SIZE + lane;
  const uint threads = 8u * METAL_SIMD_SIZE;
  float ss = 0.0f;
  for (uint c = tid; c < dim; c += threads) {
    const float v = float(x[c]);
    ss = fma(v, v, ss);
  }
  ss = simd_sum(ss);
  if (lane == 0u) {
    sh[sg] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    const float t = simd_sum(lane < 8u ? sh[lane] : 0.0f);
    if (lane == 0u) {
      sh[0] = rsqrt(t / float(dim) + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sh[0];

  const device bfloat* shift = modulation + mod_offset;
  const device bfloat* scale = shift + dim;
  for (uint c = tid; c < dim; c += threads) {
    float n = float(x[c]) * inv;
    if (has_norm_w != 0u) {
      n *= norm_w[c];
    }
    out[c] = bfloat(fma(n, 1.0f + float(scale[c]), float(shift[c])));
  }
}

KERNEL(HaruGatedResidual)(
    device bfloat* x,
    const device bfloat* ffn_out,
    const device bfloat* modulation,
    const constant uint& dim,
    const constant uint& gate_offset,
    const uint gid AXIS(dim, 256)
) {
  if (gid >= dim) {
    return;
  }
  x[gid] = bfloat(fma(float(modulation[gate_offset + gid]), float(ffn_out[gid]), float(x[gid])));
}

KERNEL(HaruVelocityReadout)(
    const device float* z_in,
    const device bfloat* v,
    device float* z_out,
    const constant float& step_h,
    const constant uint& dim,
    const uint gid AXIS(dim, 256)
) {
  if (gid >= dim) {
    return;
  }
  z_out[gid] = fma(step_h, float(v[gid]), z_in[gid]);
}

KERNEL(VibeVoiceRmsNorm)(
    const device bfloat* x,
    const device float* weight,
    device bfloat* out,
    const constant uint& steps,
    const constant uint& c_dim,
    const constant float& eps,
    threadgroup float sh[32],
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint sg THREADS(8u),
    const uint step GROUPS(steps)
) {
  const uint tid = sg * METAL_SIMD_SIZE + lane;
  const uint threads = 8u * METAL_SIMD_SIZE;
  const size_t base = size_t(step) * c_dim;
  float ss = 0.0f;
  for (uint c = tid; c < c_dim; c += threads) {
    const float v = float(x[base + c]);
    ss = fma(v, v, ss);
  }
  ss = simd_sum(ss);
  if (lane == 0u) {
    sh[sg] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    const float t = simd_sum(lane < 8u ? sh[lane] : 0.0f);
    if (lane == 0u) {
      sh[0] = rsqrt(t / float(c_dim) + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = sh[0];
  for (uint c = tid; c < c_dim; c += threads) {
    out[base + c] = bfloat(float(x[base + c]) * inv * weight[c]);
  }
}

KERNEL(VibeVoiceDepthwiseMixer)(
    const device bfloat* x,
    const device bfloat* n_in,
    const device bfloat* state_in,
    const device float* weight,
    const device float* bias,
    const device float* gamma,
    device bfloat* out,
    device bfloat* state_out,
    const constant uint& steps,
    const constant uint& c_dim,
    const constant uint& state_steps_max,
    const uint c AXIS(c_dim, 32),
    const uint t AXIS(state_steps_max, 4)
) {
  if (c >= c_dim || t >= state_steps_max) {
    return;
  }
  if (t < steps) {
    const size_t wbase = size_t(c) * 7u;
    float acc = bias[c];
    for (uint k = 0u; k < 7u; ++k) {
      const uint idx = t + k;
      const float tap = (idx < 6u)
          ? float(state_in[size_t(idx) * c_dim + c])
          : float(n_in[size_t(idx - 6u) * c_dim + c]);
      acc = fma(tap, weight[wbase + k], acc);
    }
    const size_t elem = size_t(t) * c_dim + c;
    out[elem] = bfloat(fma(gamma[c], acc, float(x[elem])));
  }
  if (t < 6u) {
    const uint src_pos = steps + t;
    const bfloat v = (src_pos < 6u)
        ? state_in[size_t(src_pos) * c_dim + c]
        : n_in[size_t(src_pos - 6u) * c_dim + c];
    state_out[size_t(t) * c_dim + c] = v;
  }
}

KERNEL(VibeVoiceCausalConvPack)(
    const device bfloat* x,
    const device bfloat* state_in,
    device bfloat* windows,
    device bfloat* state_out,
    const constant uint& in_steps,
    const constant uint& out_steps,
    const constant uint& c_in,
    const constant uint& k_len,
    const constant uint& stride,
    const constant uint& y_rows,
    const uint c AXIS(c_in, 32),
    const uint out_t AXIS(y_rows, 4)
) {
  if (c >= c_in || out_t >= y_rows) {
    return;
  }
  const uint ctx = k_len - stride;
  if (out_t < out_steps) {
    const uint base_t = out_t * stride;
    const size_t dst_row = size_t(out_t) * (k_len * c_in);
    for (uint k = 0u; k < k_len; ++k) {
      const uint pos = base_t + k;
      const bfloat v = (pos < ctx)
          ? state_in[size_t(pos) * c_in + c]
          : x[size_t(pos - ctx) * c_in + c];
      windows[dst_row + size_t(k) * c_in + c] = v;
    }
  }
  if (out_t < ctx) {
    const uint src_pos = in_steps + out_t;
    const bfloat v = (src_pos < ctx)
        ? state_in[size_t(src_pos) * c_in + c]
        : x[size_t(src_pos - ctx) * c_in + c];
    state_out[size_t(out_t) * c_in + c] = v;
  }
}

KERNEL(VibeVoiceConvTransposeOverlap)(
    const device bfloat* matmul_y,
    const device float* bias,
    const device bfloat* state_in,
    device bfloat* out,
    device bfloat* state_out,
    const constant uint& t_steps,
    const constant uint& c_out,
    const constant uint& stride,
    const uint c AXIS(c_out, 16),
    const uint s AXIS(stride, 4),
    const uint t AXIS(t_steps, 1)
) {
  if (c >= c_out || s >= stride || t >= t_steps) {
    return;
  }
  const size_t y_stride = size_t(c_out) * 2u * stride;
  const float first_v = float(matmul_y[size_t(t) * y_stride + (size_t(c) * 2u) * stride + s]);
  const float ov_v = (t == 0u)
      ? float(state_in[size_t(s) * c_out + c])
      : float(matmul_y[size_t(t - 1u) * y_stride + (size_t(c) * 2u + 1u) * stride + s]);
  out[(size_t(t) * stride + s) * c_out + c] = bfloat(first_v + ov_v + bias[c]);

  if (t == t_steps - 1u) {
    state_out[size_t(s) * c_out + c] = matmul_y[size_t(t) * y_stride + (size_t(c) * 2u + 1u) * stride + s];
  }
}

inline float vibevoice_fast_erf(float x) {
  const float s = (x < 0.0f) ? -1.0f : 1.0f;
  const float ax = fabs(x);
  const float t = 1.0f / fma(0.3275911f, ax, 1.0f);
  const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
  return s * y;
}

KERNEL(VibeVoiceAddBiasGelu)(
    device bfloat* x,
    const device float* bias,
    const constant uint& dim,
    const constant uint& rows,
    const uint c AXIS(dim, 256),
    const uint r AXIS(rows, 1)
) {
  if (c >= dim || r >= rows) {
    return;
  }
  const size_t idx = size_t(r) * dim + c;
  const float v = float(x[idx]) + bias[c];
  x[idx] = bfloat(0.5f * v * (1.0f + vibevoice_fast_erf(v * 0.7071067811865475f)));
}

KERNEL(VibeVoiceFfnResidualBias)(
    device bfloat* x,
    const device bfloat* fc2_out,
    const device float* bias,
    const device float* gamma,
    const constant uint& dim,
    const constant uint& rows,
    const uint c AXIS(dim, 256),
    const uint r AXIS(rows, 1)
) {
  if (c >= dim || r >= rows) {
    return;
  }
  const size_t idx = size_t(r) * dim + c;
  const float h = float(fc2_out[idx]) + bias[c];
  x[idx] = bfloat(fma(gamma[c], h, float(x[idx])));
}
