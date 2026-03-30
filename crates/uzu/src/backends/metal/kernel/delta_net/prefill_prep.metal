#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../ssm/ssm_common.h"

using namespace metal;

// Pre-compute normalized q, k, beta, decay for DeltaNet prefill.
// Grid: suffix_len × num_k_heads threadgroups, METAL_SIMD_SIZE threads each (1
// SIMD group).

template <typename T, uint HEAD_K_DIM>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_K_DIM, 128)
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
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const uint& suffix_len,
    const uint t_idx GROUPS(suffix_len),
    const uint hk_idx GROUPS(num_k_heads),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  static_assert(
      HEAD_K_DIM % METAL_SIMD_SIZE == 0,
      "HEAD_K_DIM must be a multiple of METAL_SIMD_SIZE"
  );
  constexpr uint elems_per_thread = HEAD_K_DIM / METAL_SIMD_SIZE;

  const uint conv_dim = 2 * key_dim + value_dim;
  const uint total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
  const uint groups_per_head = num_v_heads / num_k_heads;

  const uint tok_offset = t_idx * total_proj_dim;
  const uint q_base = tok_offset + hk_idx * HEAD_K_DIM;
  const uint k_base = tok_offset + key_dim + hk_idx * HEAD_K_DIM;

  // Load q and k elements (strided by SIMD width)
  float q_vals[elems_per_thread];
  float k_vals[elems_per_thread];
  float q_sq = 0.0f;
  float k_sq = 0.0f;

  for (uint i = 0; i < elems_per_thread; ++i) {
    uint idx = lane + i * METAL_SIMD_SIZE;
    q_vals[i] = float(in_proj[q_base + idx]);
    k_vals[i] = float(in_proj[k_base + idx]);
    q_sq += q_vals[i] * q_vals[i];
    k_sq += k_vals[i] * k_vals[i];
  }

  // L2 normalize via simd_sum (no threadgroup barrier needed)
  float q_norm_sq = simd_sum(q_sq);
  float k_norm_sq = simd_sum(k_sq);
  float q_inv_norm = rsqrt(q_norm_sq + 1e-6f);
  float k_inv_norm = rsqrt(k_norm_sq + 1e-6f);
  float q_scale = rsqrt(float(HEAD_K_DIM));

  // Normalize and scale q, write q and k
  const uint out_q_base = t_idx * key_dim + hk_idx * HEAD_K_DIM;
  const uint out_k_base = t_idx * key_dim + hk_idx * HEAD_K_DIM;

  for (uint i = 0; i < elems_per_thread; ++i) {
    uint idx = lane + i * METAL_SIMD_SIZE;
    float q_n = q_vals[i] * q_inv_norm * q_scale;
    float k_n = k_vals[i] * k_inv_norm;
    if (idx < HEAD_K_DIM) {
      q_norm_out[out_q_base + idx] = q_n;
      k_norm_out[out_k_base + idx] = k_n;
    }
  }

  // Compute beta and decay for v-heads belonging to this k-head
  // groups_per_head v-heads share this k-head; distribute across lanes
  for (uint g = lane; g < groups_per_head; g += METAL_SIMD_SIZE) {
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
