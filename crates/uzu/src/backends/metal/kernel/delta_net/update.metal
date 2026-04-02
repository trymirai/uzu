#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../ssm/ssm_common.h"

using namespace metal;

#define UPDATE_THREADS 512
constexpr constant uint UPDATE_K_SPLIT = 4;

static_assert(
    UPDATE_THREADS % UPDATE_K_SPLIT == 0,
    "UPDATE_THREADS must be a multiple of UPDATE_K_SPLIT"
);

template <typename T, uint HEAD_K_DIM>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_K_DIM, 128)
PUBLIC KERNEL(DeltaNetUpdate)(
    device const T* in_proj,
    device const T* a_log,
    device const T* dt_bias,
    device const T* norm_weight,
    device T* state,
    device T* out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const float& norm_epsilon,
    threadgroup float shared_qk[512 /* UPDATE_THREADS */ * 2],
    threadgroup float shared_scratch[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint hv_idx GROUPS(num_v_heads),
    const uint tid THREADS(UPDATE_THREADS)
) {
  static_assert(
      HEAD_K_DIM % UPDATE_K_SPLIT == 0,
      "HEAD_K_DIM must be a multiple of UPDATE_K_SPLIT"
  );
  constexpr uint bk = HEAD_K_DIM / UPDATE_K_SPLIT;

  const uint vi = tid / UPDATE_K_SPLIT;
  const uint ki_part = tid % UPDATE_K_SPLIT;
  const bool active = (vi < head_v_dim);

  threadgroup float* shared_q = shared_qk;
  threadgroup float* shared_k = shared_qk + UPDATE_THREADS;

  const uint conv_dim = 2 * key_dim + value_dim;
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk = hv_idx / groups_per_head;

  if (tid < HEAD_K_DIM) {
    shared_q[tid] = float(in_proj[hk * HEAD_K_DIM + tid]);
    shared_k[tid] = float(in_proj[key_dim + hk * HEAD_K_DIM + tid]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float q_partial = (tid < HEAD_K_DIM) ? shared_q[tid] * shared_q[tid] : 0.0f;
  float q_norm_sq =
      threadgroup_cooperative_reduce<SimdReduceSum<float>, UPDATE_THREADS>(
          q_partial,
          shared_scratch,
          thread_context
      );
  float q_inv_norm = rsqrt(q_norm_sq + 1e-6f);

  float k_partial = (tid < HEAD_K_DIM) ? shared_k[tid] * shared_k[tid] : 0.0f;
  float k_norm_sq =
      threadgroup_cooperative_reduce<SimdReduceSum<float>, UPDATE_THREADS>(
          k_partial,
          shared_scratch,
          thread_context
      );
  float k_inv_norm = rsqrt(k_norm_sq + 1e-6f);

  float q_scale = rsqrt(float(HEAD_K_DIM));
  if (tid < HEAD_K_DIM) {
    shared_q[tid] *= q_inv_norm * q_scale;
    shared_k[tid] *= k_inv_norm;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float kq_p = (tid < HEAD_K_DIM) ? shared_k[tid] * shared_q[tid] : 0.0f;
  float kq_dot =
      threadgroup_cooperative_reduce<SimdReduceSum<float>, UPDATE_THREADS>(
          kq_p,
          shared_scratch,
          thread_context
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
  // State layout: [Hv, Dv, Dk]
  float sq_partial = 0.0f;
  float sk_partial = 0.0f;
  const uint state_row_offset =
      hv_idx * head_v_dim * HEAD_K_DIM + vi * HEAD_K_DIM;
  const uint k_base = ki_part * bk;
  if (active) {
    for (uint j = 0; j < bk; ++j) {
      float s = float(state[state_row_offset + k_base + j]);
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
      uint idx = state_row_offset + k_base + j;
      float s = float(state[idx]);
      state[idx] = static_cast<T>(decay * s + shared_k[k_base + j] * delta_i);
    }
  }

  // RMSNorm + SiLU gate — only ki_part==0 contributes
  float o_sq = (active && ki_part == 0) ? o_i * o_i : 0.0f;
  float o_sumsq =
      threadgroup_cooperative_reduce<SimdReduceSum<float>, UPDATE_THREADS>(
          o_sq,
          shared_scratch,
          thread_context
      );
  float inv_rms = rsqrt(o_sumsq / float(head_v_dim) + norm_epsilon);

  if (active && ki_part == 0) {
    float nw = float(norm_weight[vi]);
    float z_silu = activate_silu(z_i);
    out[hv_idx * head_v_dim + vi] = static_cast<T>(o_i * inv_rms * nw * z_silu);
  }
}
