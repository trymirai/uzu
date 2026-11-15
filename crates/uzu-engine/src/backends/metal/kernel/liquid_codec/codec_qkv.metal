#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

#define CODEC_HEAD_DIM 64u

// Fused QKV + gate tail for the Liquid-Codec LFM3 attention layer.
//
// Input is the single fused projection of q, k, v and the per-head gate:
//   fused[token] = [q (q_heads * 64) | k (kv_heads * 64) | v (kv_heads * 64) | gate (q_heads)]
// One threadgroup handles one (token, q head): it normalises and rotates q, and
// when the head is also a kv head it does the same for k and copies v, writing
// straight into the KV cache slots. This replaces three dispatches (q head norm
// + rope, k head norm + rope, v copy) with one.
KERNEL(CodecQkvNormRope)(
    device const bfloat* fused,
    device const bfloat* q_norm,
    device const bfloat* k_norm,
    device const float* inv_freq,
    device bfloat* q_out,
    device bfloat* k_cache,
    device bfloat* v_cache,
    constant const uint& tokens,
    constant const uint& q_heads,
    constant const uint& kv_heads,
    constant const uint& fused_stride,
    constant const uint& q_offset,
    constant const uint& k_row_offset,
    constant const uint& v_row_offset,
    constant const float& eps,
    constant const uint& pos_base,
    const uint work GROUPS(tokens * 8),
    threadgroup float scratch[2],
    threadgroup float total[1],
    threadgroup float normed[CODEC_HEAD_DIM],
    const uint tid THREADS(64)
) {
  const uint t = work / q_heads;
  const uint h = work % q_heads;
  const uint kv_dim = kv_heads * CODEC_HEAD_DIM;

  // --- q head: RMSNorm over the head dim, then NeoX half-rotation RoPE ---
  const device bfloat* q_src = fused + t * fused_stride + q_offset + h * CODEC_HEAD_DIM;
  float v = float(q_src[tid]);
  float ss = simd_sum(v * v);
  if (tid % 32u == 0u) {
    scratch[tid / 32u] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    total[0] = scratch[0] + scratch[1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float q_inv = rsqrt(total[0] / float(CODEC_HEAD_DIM) + eps);
  normed[tid] = float(bfloat(float(q_norm[tid]) * float(bfloat(v * q_inv))));
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float angle = float(pos_base + t) * inv_freq[tid % 32u];
  const float c = cos(angle);
  const float s = sin(angle);
  float q_out_value;
  if (tid < 32u) {
    q_out_value = normed[tid] * c - normed[tid + 32u] * s;
  } else {
    q_out_value = normed[tid] * c + normed[tid - 32u] * s;
  }
  q_out[t * q_heads * CODEC_HEAD_DIM + h * CODEC_HEAD_DIM + tid] = bfloat(q_out_value);

  if (h >= kv_heads) {
    return; // only kv heads have k/v
  }

  // --- k head ---
  const device bfloat* k_src =
      fused + t * fused_stride + q_offset + q_heads * CODEC_HEAD_DIM + h * CODEC_HEAD_DIM;
  float kv = float(k_src[tid]);
  ss = simd_sum(kv * kv);
  if (tid % 32u == 0u) {
    scratch[tid / 32u] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    total[0] = scratch[0] + scratch[1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float k_inv = rsqrt(total[0] / float(CODEC_HEAD_DIM) + eps);
  normed[tid] = float(bfloat(float(k_norm[tid]) * float(bfloat(kv * k_inv))));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float k_out_value;
  if (tid < 32u) {
    k_out_value = normed[tid] * c - normed[tid + 32u] * s;
  } else {
    k_out_value = normed[tid] * c + normed[tid - 32u] * s;
  }
  k_cache[(k_row_offset + t) * kv_dim + h * CODEC_HEAD_DIM + tid] = bfloat(k_out_value);

  // --- v head: straight copy into the cache slot ---
  const device bfloat* v_src = fused + t * fused_stride + q_offset + (q_heads + kv_heads) * CODEC_HEAD_DIM
      + h * CODEC_HEAD_DIM;
  v_cache[(v_row_offset + t) * kv_dim + h * CODEC_HEAD_DIM + tid] = v_src[tid];
}

// Depthwise causal conv mixer with the state roll folded in.
// One thread owns a channel and walks all tokens, so the state write cannot race
// with another thread's read of the old state:
//   gated[t] = (conv over [state | pre_gate * x])[t] * post_gate
//   state    = last `taps` entries of [state | pre_gate * x]
KERNEL(CodecConvMixerState)(
    device const bfloat* in_proj,
    device const float* weight,
    device bfloat* output,
    device float* state,
    constant const uint& tokens,
    constant const uint& dim,
    constant const uint& taps,
    const uint channel THREADS(256)
) {
  const uint stride = 3u * dim;
  for (uint d = channel; d < dim; d += 256u) {
    for (uint t = 0; t < tokens; ++t) {
      float acc = 0.0f;
      for (uint k = 0; k < taps + 1u; ++k) {
        const int src = int(t) + int(k) - int(taps);
        const float sample = src >= 0
            ? float(in_proj[uint(src) * stride + d]) * float(in_proj[uint(src) * stride + 2u * dim + d])
            : state[d * taps + uint(src + int(taps))];
        acc += weight[d * (taps + 1u) + k] * sample;
      }
      const float post_gate = float(in_proj[t * stride + dim + d]);
      output[t * dim + d] = bfloat(acc * post_gate);
    }

    for (uint j = 0; j < taps; ++j) {
      const uint i = tokens + j;
      if (i < taps) {
        state[d * taps + j] = state[d * taps + i];
      } else {
        const uint row = i - taps;
        state[d * taps + j] = float(in_proj[row * stride + d]) * float(in_proj[row * stride + 2u * dim + d]);
      }
    }
  }
}
