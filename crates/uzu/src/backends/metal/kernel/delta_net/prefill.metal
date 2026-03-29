#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../ssm/ssm_common.h"

using namespace metal;

// DeltaNet prefill. One SIMD group per (v-head, dv_idx), Dk split across lanes.
// Grid: num_v_heads × num_dv_groups, PREFILL_THREADS threads each.

#define PREFILL_THREADS 256

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
    device T* state,
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
  constexpr uint n_per_t = HEAD_K_DIM / METAL_SIMD_SIZE;

  const uint dk_lane = tid % METAL_SIMD_SIZE;
  const uint dv_local = tid / METAL_SIMD_SIZE;
  const uint dv_idx = dv_group * (PREFILL_THREADS / METAL_SIMD_SIZE) + dv_local;
  const bool active = (dv_idx < head_v_dim);

  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk = hv_idx / groups_per_head;
  const uint conv_dim = 2 * key_dim + value_dim;
  const uint total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
  const uint dk_base = dk_lane * n_per_t;

  // Pointer bases — increment per token instead of re-computing index
  device const float* q_ptr = q_norm + hk * HEAD_K_DIM + dk_base;
  device const float* k_ptr = k_norm + hk * HEAD_K_DIM + dk_base;
  device const float* beta_ptr = beta_buf + hv_idx;
  device const float* decay_ptr = decay_buf + hv_idx;
  device const T* v_ptr = in_proj + 2 * key_dim + hv_idx * head_v_dim + dv_idx;

  // State layout: [Hv, Dv, Dk] — contiguous along Dk
  device T* state_ptr =
      state + (hv_idx * head_v_dim + dv_idx) * HEAD_K_DIM + dk_base;
  device T* out_ptr = out + hv_idx * head_v_dim + dv_idx;

  // Load state into registers
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  if (active) {
    s0 = float(state_ptr[0]);
    s1 = float(state_ptr[1]);
    s2 = float(state_ptr[2]);
    s3 = float(state_ptr[3]);
  }

  for (uint t = 0; t < suffix_len; ++t) {
    float decay = *decay_ptr;
    float beta = *beta_ptr;

    // Load and cache k_norm (used in both passes)
    float k0 = k_ptr[0], k1 = k_ptr[1], k2 = k_ptr[2], k3 = k_ptr[3];

    // Pass 1: decay state + kv_mem = (decayed S) @ k
    s0 *= decay;
    s1 *= decay;
    s2 *= decay;
    s3 *= decay;
    float kv_partial = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
    float kv_mem = simd_sum(kv_partial);

    // Delta
    float v_val = active ? float(*v_ptr) : 0.0f;
    float delta = beta * (v_val - kv_mem);

    // Pass 2: update state + output = new_S @ q
    s0 += k0 * delta;
    s1 += k1 * delta;
    s2 += k2 * delta;
    s3 += k3 * delta;
    float q0 = q_ptr[0], q1 = q_ptr[1], q2 = q_ptr[2], q3 = q_ptr[3];
    float out_partial = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
    float o_val = simd_sum(out_partial);

    if (active && dk_lane == 0) {
      *out_ptr = static_cast<T>(o_val);
    }

    // Advance pointers to next token
    q_ptr += key_dim;
    k_ptr += key_dim;
    beta_ptr += num_v_heads;
    decay_ptr += num_v_heads;
    v_ptr += total_proj_dim;
    out_ptr += value_dim;
  }

  // Write final state
  if (active) {
    state_ptr[0] = static_cast<T>(s0);
    state_ptr[1] = static_cast<T>(s1);
    state_ptr[2] = static_cast<T>(s2);
    state_ptr[3] = static_cast<T>(s3);
  }
}

#undef PREFILL_THREADS
