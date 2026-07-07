#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../common/prep.h"

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
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  gdn_prepare_qk_beta_decay<T, HEAD_K_DIM, false>(
      in_proj,
      a_log,
      dt_bias,
      q_norm_out,
      k_norm_out,
      beta_out,
      decay_out,
      num_v_heads,
      num_k_heads,
      key_dim,
      value_dim,
      token_idx,
      hk_idx,
      lane
  );
}
