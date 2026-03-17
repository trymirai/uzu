#include <metal_stdlib>
#include "../definitions.metal"
#include "../ssm/ssm_common.h"

using namespace metal;

// Single-token delta net update: decay state, apply delta rule, RMSNorm + SiLU
// gate. One threadgroup per v_head, HEAD_V_DIM threads per threadgroup.
//
// State layout: [num_v_heads, head_k_dim, head_v_dim]

// TODO: support different head_v_dim via VARIANTS template when needed
#define HEAD_V_DIM 128

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
    threadgroup float shared_qk[HEAD_V_DIM * 2],
    threadgroup float shared_scratch[32],
    const Simd simd,
    const uint hv_idx GROUPS(num_v_heads),
    const uint lane_i THREADS(HEAD_V_DIM)
) {
  const bool active = (lane_i < head_v_dim);

  threadgroup float* shared_q = shared_qk;
  threadgroup float* shared_k = shared_qk + HEAD_V_DIM;

  const uint conv_dim = 2 * key_dim + value_dim;
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk = hv_idx / groups_per_head;

  // Phase 0: Load q and k into shared memory (cooperatively)
  if (lane_i < head_k_dim) {
    shared_q[lane_i] = float(in_proj[hk * head_k_dim + lane_i]);
    shared_k[lane_i] = float(in_proj[key_dim + hk * head_k_dim + lane_i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // L2 normalize q
  float q_partial =
      (lane_i < head_k_dim) ? shared_q[lane_i] * shared_q[lane_i] : 0.0f;
  float q_norm_sq = threadgroup_cooperative_reduce_sum<HEAD_V_DIM>(
      q_partial, shared_scratch, lane_i, simd);
  float q_inv_norm = rsqrt(q_norm_sq + 1e-6f);

  // L2 normalize k
  float k_partial =
      (lane_i < head_k_dim) ? shared_k[lane_i] * shared_k[lane_i] : 0.0f;
  float k_norm_sq = threadgroup_cooperative_reduce_sum<HEAD_V_DIM>(
      k_partial, shared_scratch, lane_i, simd);
  float k_inv_norm = rsqrt(k_norm_sq + 1e-6f);

  // Apply normalization + scale q by head_k_dim^-0.5
  float q_scale = rsqrt(float(head_k_dim));
  if (lane_i < head_k_dim) {
    shared_q[lane_i] *= q_inv_norm * q_scale;
    shared_k[lane_i] *= k_inv_norm;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Compute dot(k, q) cooperatively
  float kq_partial =
      (lane_i < head_k_dim) ? shared_k[lane_i] * shared_q[lane_i] : 0.0f;
  float kq_dot = threadgroup_cooperative_reduce_sum<HEAD_V_DIM>(
      kq_partial, shared_scratch, lane_i, simd);

  // Phase 1: Compute gating
  float beta_raw = float(in_proj[conv_dim + value_dim + hv_idx]);
  float beta = 1.0f / (1.0f + fast::exp(-beta_raw));

  float a_raw = float(in_proj[conv_dim + value_dim + num_v_heads + hv_idx]);
  float a_log_val = float(a_log[hv_idx]);
  float dt_bias_val = float(dt_bias[hv_idx]);
  float sp_input = a_raw + dt_bias_val;
  float sp = softplus(sp_input);
  float g = -fast::exp(a_log_val) * float(sp);
  float decay = fast::exp(g);

  // Phase 2: Load v and z for this lane
  float v_i = active
                  ? float(in_proj[2 * key_dim + hv_idx * head_v_dim + lane_i])
                  : 0.0f;
  float z_i =
      active ? float(in_proj[conv_dim + hv_idx * head_v_dim + lane_i]) : 0.0f;

  // Phase 3: Read state column (coalesced: adjacent lanes read adjacent addrs)
  float sq_acc = 0.0f;
  float sk_acc = 0.0f;
  const uint state_head_offset = hv_idx * head_k_dim * head_v_dim;
  if (active) {
    for (uint j = 0; j < head_k_dim; ++j) {
      float s = float(state[state_head_offset + j * head_v_dim + lane_i]);
      sq_acc += s * shared_q[j];
      sk_acc += s * shared_k[j];
    }
  }

  // Phase 4: Compute delta and output
  float retrieved_i = decay * sk_acc;
  float delta_i = beta * (v_i - retrieved_i);
  float o_i = decay * sq_acc + delta_i * kq_dot;

  // Phase 5: Update state (coalesced writes)
  if (active) {
    for (uint j = 0; j < head_k_dim; ++j) {
      uint idx = state_head_offset + j * head_v_dim + lane_i;
      float s = float(state[idx]);
      state[idx] = static_cast<T>(decay * s + shared_k[j] * delta_i);
    }
  }

  // Phase 6: RMSNorm + SiLU gate
  float o_sq = active ? o_i * o_i : 0.0f;
  float o_sumsq = threadgroup_cooperative_reduce_sum<HEAD_V_DIM>(
      o_sq, shared_scratch, lane_i, simd);
  float inv_rms = rsqrt(o_sumsq / float(head_v_dim) + norm_epsilon);

  if (active) {
    float nw = float(norm_weight[lane_i]);
    float z_silu = apply_silu(z_i);
    float final_val = o_i * inv_rms * nw * z_silu;
    out[hv_idx * head_v_dim + lane_i] = static_cast<T>(final_val);
  }
}
