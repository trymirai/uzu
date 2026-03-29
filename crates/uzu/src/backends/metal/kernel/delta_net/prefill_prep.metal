#include <metal_stdlib>
#include "../common/dsl.h"
#include "../ssm/ssm_common.h"

using namespace metal;

// Pre-compute normalized q, k, beta, decay for DeltaNet prefill.
// Removes L2 norm reductions and gating from the inner loop.
//
// Grid: suffix_len × num_k_heads threadgroups, 32 threads each (1 SIMD group).
// Each threadgroup normalizes one (token, k_head) pair of q and k vectors,
// and fills beta/decay for the corresponding v-heads.

#define PREP_SIMD_SIZE 32
#define PREP_MAX_ELEMS_PER_THREAD 8

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(DeltaNetPrefillPrep)(
    device const T* in_proj,
    device const T* a_log,
    device const T* dt_bias,
    device float* q_norm_out,
    device float* k_norm_out,
    device float* beta_out,
    device float* decay_out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_k_dim,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const uint& suffix_len,
    const uint t_idx GROUPS(suffix_len),
    const uint hk_idx GROUPS(num_k_heads),
    const uint lane THREADS(PREP_SIMD_SIZE)
) {
  const uint conv_dim = 2 * key_dim + value_dim;
  const uint total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint elems_per_thread = head_k_dim / PREP_SIMD_SIZE;

  const uint tok_offset = t_idx * total_proj_dim;
  const uint q_base = tok_offset + hk_idx * head_k_dim;
  const uint k_base = tok_offset + key_dim + hk_idx * head_k_dim;

  // Load q and k elements (strided by SIMD width)
  float q_vals[PREP_MAX_ELEMS_PER_THREAD];
  float k_vals[PREP_MAX_ELEMS_PER_THREAD];
  float q_sq = 0.0f;
  float k_sq = 0.0f;

  for (uint i = 0; i < elems_per_thread; ++i) {
    uint idx = lane + i * PREP_SIMD_SIZE;
    q_vals[i] = (idx < head_k_dim) ? float(in_proj[q_base + idx]) : 0.0f;
    k_vals[i] = (idx < head_k_dim) ? float(in_proj[k_base + idx]) : 0.0f;
    q_sq += q_vals[i] * q_vals[i];
    k_sq += k_vals[i] * k_vals[i];
  }

  // L2 normalize via simd_sum (no threadgroup barrier needed)
  float q_norm_sq = simd_sum(q_sq);
  float k_norm_sq = simd_sum(k_sq);
  float q_inv_norm = rsqrt(q_norm_sq + 1e-6f);
  float k_inv_norm = rsqrt(k_norm_sq + 1e-6f);
  float q_scale = rsqrt(float(head_k_dim));

  // Normalize and scale q, write q and k
  const uint out_q_base = t_idx * key_dim + hk_idx * head_k_dim;
  const uint out_k_base = t_idx * key_dim + hk_idx * head_k_dim;

  for (uint i = 0; i < elems_per_thread; ++i) {
    uint idx = lane + i * PREP_SIMD_SIZE;
    float q_n = q_vals[i] * q_inv_norm * q_scale;
    float k_n = k_vals[i] * k_inv_norm;
    if (idx < head_k_dim) {
      q_norm_out[out_q_base + idx] = q_n;
      k_norm_out[out_k_base + idx] = k_n;
    }
  }

  // Compute beta and decay for v-heads belonging to this k-head
  // groups_per_head v-heads share this k-head; distribute across lanes
  for (uint g = lane; g < groups_per_head; g += PREP_SIMD_SIZE) {
    uint hv = hk_idx * groups_per_head + g;

    float beta_raw = float(in_proj[tok_offset + conv_dim + value_dim + hv]);
    float beta = 1.0f / (1.0f + fast::exp(-beta_raw));

    float a_raw =
        float(in_proj[tok_offset + conv_dim + value_dim + num_v_heads + hv]);
    float sp = softplus(a_raw + float(dt_bias[hv]));
    float decay = fast::exp(-fast::exp(float(a_log[hv])) * sp);

    beta_out[t_idx * num_v_heads + hv] = beta;
    decay_out[t_idx * num_v_heads + hv] = decay;
  }
}
