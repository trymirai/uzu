#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"

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
  constexpr uint elems_per_thread = HEAD_K_DIM / METAL_SIMD_SIZE;

  const uint dk_lane = tid % METAL_SIMD_SIZE;
  const uint dv_local = tid / METAL_SIMD_SIZE;
  const uint dv_idx = dv_group * (PREFILL_THREADS / METAL_SIMD_SIZE) + dv_local;
  const bool active = (dv_idx < head_v_dim);

  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk = hv_idx / groups_per_head;
  const uint conv_dim = 2 * key_dim + value_dim;
  const uint total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
  const uint dk_base = dk_lane * elems_per_thread;

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
  float s[elems_per_thread];
  METAL_PRAGMA_UNROLL
  for (uint i = 0; i < elems_per_thread; ++i)
    s[i] = active ? float(state_ptr[i]) : 0.0f;

  for (uint token = 0; token < suffix_len; ++token) {
    float decay = *decay_ptr;
    float beta = *beta_ptr;

    // Load and cache k_norm (used in both passes)
    float k[elems_per_thread];
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < elems_per_thread; ++i)
      k[i] = k_ptr[i];

    // Pass 1: decay state + kv_mem = (decayed S) @ k
    float kv_partial = 0.0f;
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < elems_per_thread; ++i) {
      s[i] *= decay;
      kv_partial += s[i] * k[i];
    }
    float kv_mem = simd_sum(kv_partial);

    // Delta
    float v_val = active ? float(*v_ptr) : 0.0f;
    float delta = beta * (v_val - kv_mem);

    // Pass 2: update state + output = new_S @ q
    float out_partial = 0.0f;
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < elems_per_thread; ++i) {
      s[i] += k[i] * delta;
      out_partial += s[i] * q_ptr[i];
    }
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
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < elems_per_thread; ++i)
      state_ptr[i] = static_cast<T>(s[i]);
  }
}
