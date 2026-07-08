#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../common/prep.h"

using namespace metal;

template <typename T, uint HEAD_K_DIM>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_K_DIM, 128)
KERNEL(DeltaNetChunkedPrep)(
    device const T* in_proj,
    device const float* a_log,
    device const float* dt_bias,
    device float* q_norm_out,
    device float* k_norm_out,
    device float* beta_out,
    device float* log_decay_out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const uint& suffix_len,
    const uint token_idx GROUPS(suffix_len),
    const uint hk_idx GROUPS(num_k_heads),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  gdn_prepare_qk_beta_decay<T, HEAD_K_DIM, true>(
      in_proj,
      a_log,
      dt_bias,
      q_norm_out,
      k_norm_out,
      beta_out,
      log_decay_out,
      num_v_heads,
      num_k_heads,
      key_dim,
      value_dim,
      token_idx,
      hk_idx,
      lane
  );
}

KERNEL(DeltaNetChunkedCumsum)(
    device const float* log_decay,
    device float* g_out,
    constant const uint& num_v_heads,
    constant const uint& suffix_len,
    constant const uint& chunk_size,
    const uint chunk_idx GROUPS(suffix_len.div_ceil(chunk_size)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  if (lane != 0) {
    return;
  }

  const uint token_begin = chunk_idx * chunk_size;
  const uint token_end = min(token_begin + chunk_size, suffix_len);
  float acc = 0.0f;
  for (uint token = token_begin; token < token_end; ++token) {
    const uint offset = token * num_v_heads + hv_idx;
    acc += log_decay[offset];
    g_out[offset] = acc;
  }
}
