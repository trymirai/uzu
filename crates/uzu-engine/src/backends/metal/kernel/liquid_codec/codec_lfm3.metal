#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

// ---------------------------------------------------------------------------
// Liquid-Codec LFM3 layer kernels.
//
// GEMMs stay on the generic Metal matmul kernel (bf16 weights, f32 accumulate,
// bias epilogue); the elementwise / attention tail is fused here:
// - CodecRmsNorm: pre-norm (variance in f32, weight multiply rounded to bf16).
// - CodecRmsNormResidual: post-norm + residual add in one pass.
// - CodecSiluMul: SwiGLU activation with the reference's rounding points.
// - CodecHeadNormRope: per-head RMSNorm + NeoX half-rotation RoPE, writing
//   straight into the query buffer / KV cache slot.
// - CodecAttention: causal (optionally sliding-window) attention with the
//   sigmoid per-head gate fused; scores/probabilities rounded to bf16 exactly
//   like the reference.
// - CodecConvMixer: B|C|x gating + causal depthwise conv (state carried for
//   streaming) + post gate.
// ---------------------------------------------------------------------------

#define CODEC_HEAD_DIM 64u

inline void codec_row_sum_of_squares(
    device const bfloat* row,
    uint dim,
    uint tid,
    threadgroup float* scratch,
    threadgroup float* total,
    uint threads
) {
  float ss = 0.0f;
  for (uint i = tid; i < dim; i += threads) {
    const float v = float(row[i]);
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (tid % 32u == 0u) {
    scratch[tid / 32u] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float sum = 0.0f;
    const uint groups = threads / 32u;
    for (uint i = 0; i < groups; ++i) {
      sum += scratch[i];
    }
    total[0] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

KERNEL(CodecRmsNorm)(
    device const bfloat* input,
    device const bfloat* weight,
    device bfloat* output,
    constant const uint& rows,
    constant const uint& dim,
    constant const float& eps,
    const uint row GROUPS(rows),
    threadgroup float scratch[4],
    threadgroup float total[1],
    const uint tid THREADS(128)
) {
  const device bfloat* row_in = input + row * dim;
  codec_row_sum_of_squares(row_in, dim, tid, scratch, total, 128u);
  const float inv = rsqrt(total[0] / float(dim) + eps);
  for (uint i = tid; i < dim; i += 128u) {
    const bfloat normed = bfloat(float(row_in[i]) * inv);
    output[row * dim + i] = bfloat(float(weight[i]) * float(normed));
  }
}

KERNEL(CodecRmsNormResidual)(
    device const bfloat* residual,
    device const bfloat* input,
    device const bfloat* weight,
    device bfloat* output,
    constant const uint& rows,
    constant const uint& dim,
    constant const float& eps,
    const uint row GROUPS(rows),
    threadgroup float scratch[4],
    threadgroup float total[1],
    const uint tid THREADS(128)
) {
  const device bfloat* row_in = input + row * dim;
  const device bfloat* row_res = residual + row * dim;
  codec_row_sum_of_squares(row_in, dim, tid, scratch, total, 128u);
  const float inv = rsqrt(total[0] / float(dim) + eps);
  for (uint i = tid; i < dim; i += 128u) {
    const bfloat normed = bfloat(float(row_in[i]) * inv);
    const bfloat scaled = bfloat(float(weight[i]) * float(normed));
    output[row * dim + i] = bfloat(float(row_res[i]) + float(scaled));
  }
}

KERNEL(CodecSilu)(
    device const bfloat* input,
    device bfloat* output,
    constant const uint& count,
    const uint index AXIS(count, 256)
) {
  const float x = float(input[index]);
  output[index] = bfloat(x / (1.0f + exp(-x)));
}

KERNEL(CodecSiluMul)(
    device const bfloat* gate,
    device const bfloat* up,
    device bfloat* output,
    constant const uint& count,
    const uint index AXIS(count, 256)
) {
  const float g = float(gate[index]);
  const bfloat silu = bfloat(g / (1.0f + exp(-g)));
  output[index] = bfloat(float(silu) * float(up[index]));
}


// Causal attention with the per-head sigmoid gate. `window` = 0 means full
// attention; otherwise keys older than `abs_pos + 1 - window` are masked out.
// q: [tokens, heads * 64]; k/v cache: [kv_len, kv_heads * 64]; gate: [tokens, heads].
KERNEL(CodecAttention)(
    device const bfloat* q,
    device const bfloat* k_cache,
    device const bfloat* v_cache,
    device const bfloat* gate,
    device bfloat* output,
    constant const uint& tokens,
    constant const uint& heads,
    constant const uint& kv_heads,
    constant const uint& window,
    constant const uint& pos_base,
    constant const float& scale,
    constant const uint& gate_stride,
    const uint work GROUPS(tokens * 8),
    threadgroup float scores[128],
    threadgroup float gate_cache[8],
    const uint tid THREADS(64)
) {
  const uint t = work / heads;
  const uint h = work % heads;
  const uint kvh = h / (heads / kv_heads);
  const uint abs_pos = pos_base + t;
  const uint first = window == 0u ? 0u : (abs_pos + 1u > window ? abs_pos + 1u - window : 0u);
  const uint n_keys = abs_pos + 1u - first;
  // q rows are [token][heads * head_dim]; gate rows are [token][heads].
  const device bfloat* q_row = q + (t * heads + h) * CODEC_HEAD_DIM;
  const uint kv_stride = kv_heads * CODEC_HEAD_DIM;

  // gate is per (token, head); cache it once per threadgroup so all heads of
  // this token read the same values and only one thread touches global memory
  if (tid < 8u) {
    gate_cache[tid] = float(gate[t * gate_stride + tid]);
  }
  // gate_cache must be visible to every thread of the group.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint j = tid; j < 128u; j += 64u) {
    scores[j] = -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = tid; j < n_keys; j += 64u) {
    const device bfloat* k_row = k_cache + (first + j) * kv_stride + kvh * CODEC_HEAD_DIM;
    float acc = 0.0f;
    for (uint d = 0; d < CODEC_HEAD_DIM; ++d) {
      acc += float(q_row[d]) * float(k_row[d]);
    }
    // reference: bf16 matmul result, times `scale`, rounded to bf16
    scores[j] = float(bfloat(float(bfloat(acc)) * scale));
  }
  // threads with j >= n_keys must not leave stale threadgroup data behind:
  // a query processed by a later threadgroup reuses this buffer only after its
  // own barrier, but tid 0's loops below read up to n_keys entries.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    float max_score = -INFINITY;
    for (uint j = 0; j < n_keys; ++j) {
      max_score = max(max_score, scores[j]);
    }
    float sum = 0.0f;
    for (uint j = 0; j < n_keys; ++j) {
      const float e = exp(scores[j] - max_score);
      scores[j] = e;
      sum += e;
    }
    const float inv_sum = 1.0f / sum;
    for (uint j = 0; j < n_keys; ++j) {
      scores[j] = float(bfloat(scores[j] * inv_sum));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc = 0.0f;
  for (uint j = 0; j < n_keys; ++j) {
    acc += scores[j] * float(v_cache[(first + j) * kv_stride + kvh * CODEC_HEAD_DIM + tid]);
  }
  const bfloat context = bfloat(acc);
  const float g = gate_cache[h];
  const bfloat sigmoid = bfloat(1.0f / (1.0f + exp(-g)));
  output[(t * heads + h) * CODEC_HEAD_DIM + tid] = bfloat(float(context) * float(sigmoid));
}

// in_proj output [tokens, 3 * dim] laid out [B | C | x]; depthwise causal conv
// with kernel taps over `pre_gate * x`; post gate `C`; streaming state keeps
// the last `kernel - 1` gated inputs per channel.
KERNEL(CodecConvMixer)(
    device const bfloat* in_proj,
    device const float* weight,
    device bfloat* output,
    device const float* state,
    constant const uint& tokens,
    constant const uint& dim,
    constant const uint& taps_count,
    const uint index AXIS(tokens * dim, 256) // one thread per (token, channel)
) {
  const uint t = index / dim;
  const uint d = index % dim;
  const uint taps = taps_count;

  auto gated_at = [&](int src) -> float {
    if (src >= 0) {
      const uint row = uint(src) * (3u * dim);
      return float(in_proj[row + d]) * float(in_proj[row + 2u * dim + d]);
    }
    return state[d * taps + uint(src + int(taps))];
  };

  float acc = 0.0f;
  for (uint k = 0; k < taps + 1u; ++k) {
    const int src = int(t) + int(k) - int(taps);
    acc += weight[d * (taps + 1u) + k] * gated_at(src);
  }
  const float post_gate = float(in_proj[t * (3u * dim) + dim + d]);
  output[t * dim + d] = bfloat(acc * post_gate);
}

