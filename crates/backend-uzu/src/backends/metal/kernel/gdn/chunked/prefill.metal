#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../common/heads.h"

using namespace metal;

// DeltaNet prefill: one simdgroup owns 4 value rows while lanes cover Dk via
// float4. Grid is num_v_heads x num_dv_groups.

#define PREFILL_THREADS 128
#define DV_PER_SIMDGROUP 4

static_assert(PREFILL_THREADS % METAL_SIMD_SIZE == 0, "PREFILL_THREADS must be a multiple of METAL_SIMD_SIZE");

template <typename T, uint HEAD_K_DIM>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_K_DIM, 128)
PUBLIC KERNEL(DeltaNetPrefill)(
    device const float* q_norm,
    device const float* k_norm,
    device const float* beta_buf,
    device const float* decay_buf,
    device const T* in_proj,
    device float* state,
    device T* out,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    constant const uint& key_dim,
    constant const uint& value_dim,
    constant const uint& suffix_len,
    constant const uint& num_dv_groups,
    const uint hv_idx GROUPS(num_v_heads),
    const uint dv_group GROUPS(num_dv_groups),
    const uint tid THREADS(PREFILL_THREADS)
) {
  static_assert(HEAD_K_DIM % METAL_SIMD_SIZE == 0, "HEAD_K_DIM must be a multiple of METAL_SIMD_SIZE");
  constexpr uint ELEMS = HEAD_K_DIM / METAL_SIMD_SIZE;
  constexpr uint NUM_SG = PREFILL_THREADS / METAL_SIMD_SIZE;
  static_assert(ELEMS == 4, "float4 prefill requires ELEMS == 4");
  static_assert(DV_PER_SIMDGROUP == 4, "packs dv rows into a float4");

  const uint lane = tid % METAL_SIMD_SIZE;
  const uint dv_local = tid / METAL_SIMD_SIZE;
  const uint dv_idx = (dv_group * NUM_SG + dv_local) * DV_PER_SIMDGROUP;

  const uint hk = gdn_key_head_for_value_head(hv_idx, num_v_heads, num_k_heads);
  const uint conv_dim = 2 * key_dim + value_dim;
  const uint total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
  const uint dk_base = lane * ELEMS;

  device const float* q_ptr = q_norm + hk * HEAD_K_DIM + dk_base;
  device const float* k_ptr = k_norm + hk * HEAD_K_DIM + dk_base;
  device const float* beta_ptr = beta_buf + hv_idx;
  device const float* decay_ptr = decay_buf + hv_idx;
  device const T* v_row = in_proj + 2 * key_dim + hv_idx * head_v_dim + dv_idx;
  device T* out_row = out + hv_idx * head_v_dim + dv_idx;

  // State layout: [Hv, Dv, Dk], contiguous along Dk (HEAD_K_DIM floats/row).
  // Unrolled by hand: METAL_PRAGMA_UNROLL is ignored under metal4.0 -O2, so a
  // loop over s[i] keeps the array in an alloca and reloads it every token.
  device const float* s_in = state + (hv_idx * head_v_dim + dv_idx) * HEAD_K_DIM + dk_base;
  float4 s[DV_PER_SIMDGROUP];
  s[0] = *reinterpret_cast<device const float4*>(s_in + 0 * HEAD_K_DIM);
  s[1] = *reinterpret_cast<device const float4*>(s_in + 1 * HEAD_K_DIM);
  s[2] = *reinterpret_cast<device const float4*>(s_in + 2 * HEAD_K_DIM);
  s[3] = *reinterpret_cast<device const float4*>(s_in + 3 * HEAD_K_DIM);

  for (uint token = 0; token < suffix_len; ++token) {
    float decay = *decay_ptr;
    float beta = *beta_ptr;
    float4 k = *reinterpret_cast<device const float4*>(k_ptr);
    float4 q = *reinterpret_cast<device const float4*>(q_ptr);

    s[0] *= decay;
    s[1] *= decay;
    s[2] *= decay;
    s[3] *= decay;
    float4 kv_mem = float4(dot(s[0], k), dot(s[1], k), dot(s[2], k), dot(s[3], k));
    kv_mem = simd_sum(kv_mem);

    float4 v_val = float4(float(v_row[0]), float(v_row[1]), float(v_row[2]), float(v_row[3]));
    float4 delta = beta * (v_val - kv_mem);
    s[0] += k * delta[0];
    s[1] += k * delta[1];
    s[2] += k * delta[2];
    s[3] += k * delta[3];
    float4 o = float4(dot(s[0], q), dot(s[1], q), dot(s[2], q), dot(s[3], q));
    o = simd_sum(o);

    if (lane == 0) {
      out_row[0] = static_cast<T>(o[0]);
      out_row[1] = static_cast<T>(o[1]);
      out_row[2] = static_cast<T>(o[2]);
      out_row[3] = static_cast<T>(o[3]);
    }

    q_ptr += key_dim;
    k_ptr += key_dim;
    beta_ptr += num_v_heads;
    decay_ptr += num_v_heads;
    v_row += total_proj_dim;
    out_row += value_dim;
  }

  device float* s_out = state + (hv_idx * head_v_dim + dv_idx) * HEAD_K_DIM + dk_base;
  *reinterpret_cast<device float4*>(s_out + 0 * HEAD_K_DIM) = s[0];
  *reinterpret_cast<device float4*>(s_out + 1 * HEAD_K_DIM) = s[1];
  *reinterpret_cast<device float4*>(s_out + 2 * HEAD_K_DIM) = s[2];
  *reinterpret_cast<device float4*>(s_out + 3 * HEAD_K_DIM) = s[3];
}
