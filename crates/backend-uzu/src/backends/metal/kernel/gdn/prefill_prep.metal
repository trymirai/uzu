#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;

template <typename T, uint HEAD_K_DIM>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_K_DIM, 128)
PUBLIC KERNEL(DeltaNetPrefillPrep)(
    device const T* in_proj,
    device const float* a_log,
    device const float* dt_bias,
    device float* q_norm_out,
    device float* k_norm_out,
    device float* beta_out,
    device float* decay_out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const uint& suffix_len,
    const uint token_idx GROUPS(suffix_len),
    const uint hk_idx GROUPS(num_k_heads),
    const bool write_log_decay SPECIALIZE,
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  static_assert(HEAD_K_DIM % METAL_SIMD_SIZE == 0, "HEAD_K_DIM must be a multiple of METAL_SIMD_SIZE");
  constexpr uint elems_per_thread = HEAD_K_DIM / METAL_SIMD_SIZE;

  const uint conv_dim = 2 * key_dim + value_dim;
  const uint total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
  const uint groups_per_head = num_v_heads / num_k_heads;

  const uint tok_offset = token_idx * total_proj_dim;
  const uint q_base = tok_offset + hk_idx * HEAD_K_DIM;
  const uint k_base = tok_offset + key_dim + hk_idx * HEAD_K_DIM;

  float q_vals[elems_per_thread];
  float k_vals[elems_per_thread];
  float q_sq = 0.0f;
  float k_sq = 0.0f;

  for (uint i = 0; i < elems_per_thread; ++i) {
    const uint idx = lane + i * METAL_SIMD_SIZE;
    q_vals[i] = float(in_proj[q_base + idx]);
    k_vals[i] = float(in_proj[k_base + idx]);
    q_sq += q_vals[i] * q_vals[i];
    k_sq += k_vals[i] * k_vals[i];
  }

  const float q_inv_norm = rsqrt(simd_sum(q_sq) + 1e-6f);
  const float k_inv_norm = rsqrt(simd_sum(k_sq) + 1e-6f);
  const float q_scale = rsqrt(float(HEAD_K_DIM));

  const uint out_base = token_idx * key_dim + hk_idx * HEAD_K_DIM;
  for (uint i = 0; i < elems_per_thread; ++i) {
    const uint idx = lane + i * METAL_SIMD_SIZE;
    q_norm_out[out_base + idx] = q_vals[i] * q_inv_norm * q_scale;
    k_norm_out[out_base + idx] = k_vals[i] * k_inv_norm;
  }

  for (uint group = lane; group < groups_per_head; group += METAL_SIMD_SIZE) {
    const uint hv = hk_idx * groups_per_head + group;
    const float beta_raw = float(in_proj[tok_offset + conv_dim + value_dim + hv]);
    const float a_raw = float(in_proj[tok_offset + conv_dim + value_dim + num_v_heads + hv]);
    const float log_decay = -fast::exp(a_log[hv]) * activate_softplus(a_raw + dt_bias[hv]);

    beta_out[token_idx * num_v_heads + hv] = 1.0f / (1.0f + fast::exp(-beta_raw));
    if (write_log_decay) {
      decay_out[token_idx * num_v_heads + hv] = log_decay;
    } else {
      decay_out[token_idx * num_v_heads + hv] = fast::exp(log_decay);
    }
  }
}
