#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

// ---------------------------------------------------------------------------
// LFM3-Audio MaskGIT depth decoder kernels (head_dim == 32, slots <= 32).
//
// GEMMs use the generic BF16 MatmulKernel; RMSNorm and SwiGLU reuse
// CodecRmsNorm / CodecSiluMul. These 4 kernels cover the depth-specific ops.
// ---------------------------------------------------------------------------

KERNEL(DepthEmbedSlots)(
    device const bfloat* projected,
    device const bfloat* mask,
    device const bfloat* embeddings,
    device const int* codes,
    device const uint* masked,
    device bfloat* output,
    constant const uint& slots,
    constant const uint& hidden,
    constant const uint& vocab,
    const uint index AXIS(slots * hidden, 256)
) {
  const uint slot = index / hidden;
  const uint d = index % hidden;
  const bfloat base = (masked[slot] != 0u)
      ? mask[d]
      : embeddings[((slot * vocab + uint(codes[slot])) * hidden) + d];
  output[index] = bfloat(float(projected[index]) + float(base));
}

KERNEL(DepthQkvNormRope)(
    device bfloat* qkvg,
    device const bfloat* q_norm,
    device const bfloat* k_norm,
    device const float* rope,
    constant const uint& slots,
    constant const uint& heads,
    constant const uint& kv_heads,
    constant const uint& qkvg_stride,
    constant const float& eps,
    const uint work GROUPS(slots * (heads + kv_heads)),
    const uint tid THREADS(32)
) {
  const uint total_heads = heads + kv_heads;
  const uint slot = work / total_heads;
  const uint h = work % total_heads;
  const bool is_q = h < heads;
  const uint head_idx = is_q ? h : (h - heads);
  const uint col_off = is_q ? (head_idx * 32u) : (heads * 32u + head_idx * 32u);
  device const bfloat* nw = is_q ? q_norm : k_norm;

  const uint base = slot * qkvg_stride + col_off;
  const float v = float(qkvg[base + tid]);
  const float ss = simd_sum(v * v);
  const float inv = rsqrt(ss / 32.0f + eps);
  const float normed = float(bfloat(v * inv * float(nw[tid])));

  const uint half_dim = 16u;
  const float other = simd_shuffle_xor(normed, half_dim);
  const uint rbase = (slot * half_dim + (tid & 15u)) * 2u;
  const float c = rope[rbase];
  const float s = rope[rbase + 1u];
  const float rotated = (tid < half_dim)
      ? (normed * c - other * s)
      : (normed * c + other * s);
  qkvg[base + tid] = bfloat(rotated);
}

KERNEL(DepthAttention)(
    device const bfloat* qkvg,
    device bfloat* output,
    constant const uint& slots,
    constant const uint& heads,
    constant const uint& kv_heads,
    constant const uint& qkvg_stride,
    constant const float& scale,
    const uint work GROUPS(slots * heads),
    const uint tid THREADS(32)
) {
  const uint row = work / heads;
  const uint head = work % heads;
  const uint kv_head = head / (heads / kv_heads);
  const uint k_off = heads * 32u + kv_head * 32u;
  const uint v_off = (heads + kv_heads) * 32u + kv_head * 32u;
  const uint g_off = (heads + 2u * kv_heads) * 32u;

  const float q_val = float(qkvg[row * qkvg_stride + head * 32u + tid]);
  float my_score = -INFINITY;
  for (uint col = 0u; col < slots; ++col) {
    const float k_val = float(qkvg[col * qkvg_stride + k_off + tid]);
    const float dot = simd_sum(q_val * k_val) * scale;
    if (tid == col) {
      my_score = dot;
    }
  }

  const float max_s = simd_max(my_score);
  const float w = (tid < slots) ? exp(my_score - max_s) : 0.0f;
  const float sum_w = simd_sum(w);
  const float my_prob = w / sum_w;

  float acc = 0.0f;
  for (uint col = 0u; col < slots; ++col) {
    const float prob = simd_broadcast(my_prob, col);
    const float v_val = float(qkvg[col * qkvg_stride + v_off + tid]);
    acc += prob * v_val;
  }

  const float g = float(qkvg[row * qkvg_stride + g_off + head]);
  const float gate_val = 1.0f / (1.0f + exp(-g));
  output[(row * heads + head) * 32u + tid] = bfloat(acc * gate_val);
}

KERNEL(DepthSiluMulFused)(
    device const bfloat* w13,
    device bfloat* output,
    constant const uint& slots,
    constant const uint& inter,
    const uint index AXIS(slots * inter, 256)
) {
  const uint slot = index / inter;
  const uint d = index % inter;
  const float g = float(w13[slot * (2u * inter) + d]);
  const float u = float(w13[slot * (2u * inter) + inter + d]);
  const bfloat silu = bfloat(g / (1.0f + exp(-g)));
  output[index] = bfloat(float(silu) * u);
}

KERNEL(DepthResidualAdd)(
    device bfloat* x,
    device const bfloat* delta,
    constant const uint& count,
    const uint index AXIS(count, 256)
) {
  x[index] = bfloat(float(x[index]) + float(delta[index]));
}

KERNEL(DepthHeadNormSlot)(
    device const bfloat* input,
    device const bfloat* weight,
    device bfloat* output,
    constant const uint& slot,
    constant const uint& dim,
    constant const float& eps,
    threadgroup float scratch[4],
    threadgroup float total[1],
    const uint tid THREADS(128)
) {
  const device bfloat* row = input + slot * dim;
  float ss = 0.0f;
  for (uint i = tid; i < dim; i += 128u) {
    const float v = float(row[i]);
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (tid % 32u == 0u) {
    scratch[tid / 32u] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    total[0] = scratch[0] + scratch[1] + scratch[2] + scratch[3];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(total[0] / float(dim) + eps);
  for (uint i = tid; i < dim; i += 128u) {
    output[i] = bfloat(float(row[i]) * inv * float(weight[i]));
  }
}
