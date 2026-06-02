#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;

// DeltaNet prefill: one simdgroup owns 4 value rows while lanes cover Dk via
// float4. Grid is num_v_heads x num_dv_groups.

#define PREFILL_THREADS 128
#define PREFILL_DV_PER_SIMDGROUP 4

static_assert(
    PREFILL_THREADS % METAL_SIMD_SIZE == 0,
    "PREFILL_THREADS must be a multiple of METAL_SIMD_SIZE"
);

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
  static_assert(
      HEAD_K_DIM % METAL_SIMD_SIZE == 0,
      "HEAD_K_DIM must be a multiple of METAL_SIMD_SIZE"
  );
  constexpr uint SIMD = METAL_SIMD_SIZE;
  constexpr uint ELEMS = HEAD_K_DIM / SIMD;
  constexpr uint NUM_SG = PREFILL_THREADS / SIMD;
  static_assert(ELEMS == 4, "float4 prefill requires ELEMS == 4");
  static_assert(PREFILL_DV_PER_SIMDGROUP == 4, "packs dv rows into a float4");

  const uint lane = tid % SIMD;
  const uint dv_local = tid / SIMD;
  const uint dv_idx = (dv_group * NUM_SG + dv_local) * PREFILL_DV_PER_SIMDGROUP;

  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk = hv_idx / groups_per_head;
  const uint conv_dim = 2 * key_dim + value_dim;
  const uint total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
  const uint dk_base = lane * ELEMS;

  device const float* q_ptr = q_norm + hk * HEAD_K_DIM + dk_base;
  device const float* k_ptr = k_norm + hk * HEAD_K_DIM + dk_base;
  device const float* beta_ptr = beta_buf + hv_idx;
  device const float* decay_ptr = decay_buf + hv_idx;
  device const T* v_ptr = in_proj + 2 * key_dim + hv_idx * head_v_dim;
  device T* out_ptr = out + hv_idx * head_v_dim;

  float4 s[PREFILL_DV_PER_SIMDGROUP];
  METAL_PRAGMA_UNROLL
  for (uint i = 0; i < PREFILL_DV_PER_SIMDGROUP; ++i) {
    uint dv = dv_idx + i;
    const bool active = (dv < head_v_dim);
    s[i] = active
               ? *reinterpret_cast<device const float4*>(
                     state + (hv_idx * head_v_dim + dv) * HEAD_K_DIM + dk_base
                 )
               : float4(0.0f);
  }

  for (uint token = 0; token < suffix_len; ++token) {
    float decay = *decay_ptr;
    float beta = *beta_ptr;
    float4 k = *reinterpret_cast<device const float4*>(k_ptr);
    float4 q = *reinterpret_cast<device const float4*>(q_ptr);

    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < PREFILL_DV_PER_SIMDGROUP; ++i)
      s[i] *= decay;
    float4 retrieved =
        float4(dot(s[0], k), dot(s[1], k), dot(s[2], k), dot(s[3], k));
    retrieved = simd_sum(retrieved);

    float4 o;
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < PREFILL_DV_PER_SIMDGROUP; ++i) {
      uint dv = dv_idx + i;
      const bool active = (dv < head_v_dim);
      float v_val = active ? float(v_ptr[token * total_proj_dim + dv]) : 0.0f;
      float delta = beta * (v_val - retrieved[i]);
      s[i] += k * delta;
      o[i] = dot(s[i], q);
    }
    o = simd_sum(o);

    if (lane == 0) {
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < PREFILL_DV_PER_SIMDGROUP; ++i) {
        uint dv = dv_idx + i;
        const bool active = (dv < head_v_dim);
        if (active)
          out_ptr[token * value_dim + dv] = static_cast<T>(o[i]);
      }
    }

    q_ptr += key_dim;
    k_ptr += key_dim;
    beta_ptr += num_v_heads;
    decay_ptr += num_v_heads;
  }

  METAL_PRAGMA_UNROLL
  for (uint i = 0; i < PREFILL_DV_PER_SIMDGROUP; ++i) {
    uint dv = dv_idx + i;
    const bool active = (dv < head_v_dim);
    if (active) {
      *reinterpret_cast<device float4*>(
          state + (hv_idx * head_v_dim + dv) * HEAD_K_DIM + dk_base
      ) = s[i];
    }
  }
}
