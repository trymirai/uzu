#include <metal_stdlib>
#include "../definitions.metal"
#include "../ssm/ssm_common.h"

using namespace metal;

// TODO: support different head_v_dim via VARIANTS template when needed
#define HEAD_V_DIM 128
#define K_SPLIT 4
#define DN_UPDATE_THREADS (HEAD_V_DIM * K_SPLIT)

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(DeltaNetUpdate)(
    device const T* in_proj,
    device const T* a_log,
    device const T* dt_bias,
    device const T* norm_weight,
    device T* state,
    device T* out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_k_dim,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const float& norm_epsilon,
    threadgroup float shared_qk[DN_UPDATE_THREADS * 2],
    threadgroup float shared_scratch[32],
    const Simd simd,
    const uint hv_idx GROUPS(num_v_heads),
    const uint tid THREADS(DN_UPDATE_THREADS)
) {
  const uint vi = tid / K_SPLIT;
  const uint ki_part = tid % K_SPLIT;
  const bool active = (vi < head_v_dim);
  const uint bk = head_k_dim / K_SPLIT;

  threadgroup float* shared_q = shared_qk;
  threadgroup float* shared_k = shared_qk + DN_UPDATE_THREADS;

  const uint conv_dim = 2 * key_dim + value_dim;
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk = hv_idx / groups_per_head;

  if (tid < head_k_dim) {
    shared_q[tid] = float(in_proj[hk * head_k_dim + tid]);
    shared_k[tid] = float(in_proj[key_dim + hk * head_k_dim + tid]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float q_partial = (tid < head_k_dim) ? shared_q[tid] * shared_q[tid] : 0.0f;
  float q_norm_sq = threadgroup_cooperative_reduce_sum<DN_UPDATE_THREADS>(
      q_partial,
      shared_scratch,
      tid,
      simd
  );
  float q_inv_norm = rsqrt(q_norm_sq + 1e-6f);

  float k_partial = (tid < head_k_dim) ? shared_k[tid] * shared_k[tid] : 0.0f;
  float k_norm_sq = threadgroup_cooperative_reduce_sum<DN_UPDATE_THREADS>(
      k_partial,
      shared_scratch,
      tid,
      simd
  );
  float k_inv_norm = rsqrt(k_norm_sq + 1e-6f);

  float q_scale = rsqrt(float(head_k_dim));
  if (tid < head_k_dim) {
    shared_q[tid] *= q_inv_norm * q_scale;
    shared_k[tid] *= k_inv_norm;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float kq_p = (tid < head_k_dim) ? shared_k[tid] * shared_q[tid] : 0.0f;
  float kq_dot = threadgroup_cooperative_reduce_sum<DN_UPDATE_THREADS>(
      kq_p,
      shared_scratch,
      tid,
      simd
  );

  float beta_raw = float(in_proj[conv_dim + value_dim + hv_idx]);
  float beta = 1.0f / (1.0f + fast::exp(-beta_raw));
  float a_raw = float(in_proj[conv_dim + value_dim + num_v_heads + hv_idx]);
  float sp = softplus(a_raw + float(dt_bias[hv_idx]));
  float decay = fast::exp(-fast::exp(float(a_log[hv_idx])) * sp);

  float v_i =
      active ? float(in_proj[2 * key_dim + hv_idx * head_v_dim + vi]) : 0.0f;
  float z_i =
      active ? float(in_proj[conv_dim + hv_idx * head_v_dim + vi]) : 0.0f;

  // State read — each thread handles bk rows
  float sq_partial = 0.0f;
  float sk_partial = 0.0f;
  const uint state_head_offset = hv_idx * head_k_dim * head_v_dim;
  const uint k_base = ki_part * bk;
  if (active) {
    for (uint j = 0; j < bk; ++j) {
      float s =
          float(state[state_head_offset + (k_base + j) * head_v_dim + vi]);
      sq_partial += s * shared_q[k_base + j];
      sk_partial += s * shared_k[k_base + j];
    }
  }

  // Reduce across k-parts (4 consecutive lanes = one quad group)
  float sq_acc = quad_sum(sq_partial);
  float sk_acc = quad_sum(sk_partial);

  float retrieved_i = decay * sk_acc;
  float delta_i = beta * (v_i - retrieved_i);
  float o_i = decay * sq_acc + delta_i * kq_dot;

  // State update — each thread updates its k-part
  if (active) {
    for (uint j = 0; j < bk; ++j) {
      uint idx = state_head_offset + (k_base + j) * head_v_dim + vi;
      float s = float(state[idx]);
      state[idx] = static_cast<T>(decay * s + shared_k[k_base + j] * delta_i);
    }
  }

  // RMSNorm + SiLU gate — only ki_part==0 contributes
  float o_sq = (active && ki_part == 0) ? o_i * o_i : 0.0f;
  float o_sumsq = threadgroup_cooperative_reduce_sum<DN_UPDATE_THREADS>(
      o_sq,
      shared_scratch,
      tid,
      simd
  );
  float inv_rms = rsqrt(o_sumsq / float(head_v_dim) + norm_epsilon);

  if (active && ki_part == 0) {
    float nw = float(norm_weight[vi]);
    float z_silu = apply_silu(z_i);
    out[hv_idx * head_v_dim + vi] = static_cast<T>(o_i * inv_rms * nw * z_silu);
  }
}
