#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../ssm/ssm_common.h"

using namespace metal;

// Multi-token delta net prefill: sequential scan, k-tiled state.
// 128 threads per threadgroup, BV=64 v-columns per tile.
// Thread mapping: vi = tid/2, ki_half = tid%2 — paired threads in same SIMD
// group for simd_shuffle reduction.
// Each thread holds head_k_dim/2 state floats. No RMSNorm/SiLU — outputs raw o.
//
// Grid: num_v_heads x num_v_tiles threadgroups

#define DN_TILED_BLOCK_SIZE 128
#define DN_TILED_BV 64
#define DN_TILED_MAX_BK_HALF 64

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(DeltaNetPrefill)(
    device const T* in_proj,
    device const T* a_log,
    device const T* dt_bias,
    device T* state,
    device T* out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_k_dim,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const uint& suffix_len,
    constant const uint& num_v_tiles,
    threadgroup float shared_qk[DN_TILED_BLOCK_SIZE * 2],
    threadgroup float shared_scratch[32],
    const ThreadContext thread_context,
    const uint hv_idx GROUPS(num_v_heads),
    const uint v_tile GROUPS(num_v_tiles),
    const uint tid THREADS(DN_TILED_BLOCK_SIZE)
) {
  const uint vi_local = tid / 2;
  const uint ki_half = tid % 2;
  const uint vi_global = v_tile * DN_TILED_BV + vi_local;
  const bool active = (vi_global < head_v_dim);

  threadgroup float* shared_q = shared_qk;
  threadgroup float* shared_k = shared_qk + DN_TILED_BLOCK_SIZE;

  const uint conv_dim = 2 * key_dim + value_dim;
  const uint total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk = hv_idx / groups_per_head;
  const uint state_head_offset = hv_idx * head_k_dim * head_v_dim;

  const uint bk_half = head_k_dim / 2;

  // Load initial state into thread-local storage
  thread float s_col[DN_TILED_MAX_BK_HALF];
  if (active) {
    const uint k_base = ki_half * bk_half;
    for (uint j = 0; j < bk_half; ++j) {
      s_col[j] = float(
          state[state_head_offset + (k_base + j) * head_v_dim + vi_global]
      );
    }
  }

  const float a_log_val = float(a_log[hv_idx]);
  const float dt_bias_val = float(dt_bias[hv_idx]);

  for (uint t = 0; t < suffix_len; ++t) {
    const uint tok = t * total_proj_dim;

    // Load q, k into shared memory
    if (tid < head_k_dim) {
      shared_q[tid] = float(in_proj[tok + hk * head_k_dim + tid]);
      shared_k[tid] = float(in_proj[tok + key_dim + hk * head_k_dim + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // L2 normalize q
    float q_partial = (tid < head_k_dim) ? shared_q[tid] * shared_q[tid] : 0.0f;
    float q_norm_sq = threadgroup_cooperative_reduce_sum<DN_TILED_BLOCK_SIZE>(
        q_partial,
        shared_scratch,
        tid,
        thread_context
    );
    float q_inv_norm = rsqrt(q_norm_sq + 1e-6f);

    // L2 normalize k
    float k_partial = (tid < head_k_dim) ? shared_k[tid] * shared_k[tid] : 0.0f;
    float k_norm_sq = threadgroup_cooperative_reduce_sum<DN_TILED_BLOCK_SIZE>(
        k_partial,
        shared_scratch,
        tid,
        thread_context
    );
    float k_inv_norm = rsqrt(k_norm_sq + 1e-6f);

    // Apply normalization + scale q
    float q_scale = rsqrt(float(head_k_dim));
    if (tid < head_k_dim) {
      shared_q[tid] *= q_inv_norm * q_scale;
      shared_k[tid] *= k_inv_norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dot(k, q)
    float kq_partial =
        (tid < head_k_dim) ? shared_k[tid] * shared_q[tid] : 0.0f;
    float kq_dot = threadgroup_cooperative_reduce_sum<DN_TILED_BLOCK_SIZE>(
        kq_partial,
        shared_scratch,
        tid,
        thread_context
    );

    // Gating scalars
    float beta_raw = float(in_proj[tok + conv_dim + value_dim + hv_idx]);
    float beta = 1.0f / (1.0f + fast::exp(-beta_raw));

    float a_raw =
        float(in_proj[tok + conv_dim + value_dim + num_v_heads + hv_idx]);
    float sp = softplus(a_raw + dt_bias_val);
    float decay = fast::exp(-fast::exp(a_log_val) * sp);

    // Load v for this v-column
    float v_i =
        active
            ? float(
                  in_proj[tok + 2 * key_dim + hv_idx * head_v_dim + vi_global]
              )
            : 0.0f;

    // State retrieval: partial dot products over ki_half's k-dims
    float sq_partial = 0.0f;
    float sk_partial = 0.0f;
    if (active) {
      const uint k_base = ki_half * bk_half;
      for (uint j = 0; j < bk_half; ++j) {
        sq_partial += s_col[j] * shared_q[k_base + j];
        sk_partial += s_col[j] * shared_k[k_base + j];
      }
    }

    // Reduce across k-halves via simd_shuffle (partner is adjacent lane)
    float sq_acc = sq_partial + simd_shuffle_xor(sq_partial, 1);
    float sk_acc = sk_partial + simd_shuffle_xor(sk_partial, 1);

    // Delta rule + output
    float retrieved_i = decay * sk_acc;
    float delta_i = beta * (v_i - retrieved_i);
    float o_i = decay * sq_acc + delta_i * kq_dot;

    if (active && ki_half == 0) {
      out[t * value_dim + hv_idx * head_v_dim + vi_global] =
          static_cast<T>(o_i);
    }

    // Update state in registers
    if (active) {
      const uint k_base = ki_half * bk_half;
      for (uint j = 0; j < bk_half; ++j) {
        s_col[j] = decay * s_col[j] + shared_k[k_base + j] * delta_i;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write final state back
  if (active) {
    const uint k_base = ki_half * bk_half;
    for (uint j = 0; j < bk_half; ++j) {
      state[state_head_offset + (k_base + j) * head_v_dim + vi_global] =
          static_cast<T>(s_col[j]);
    }
  }
}
