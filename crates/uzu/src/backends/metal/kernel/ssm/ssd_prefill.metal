#include <metal_stdlib>
#include <metal_simdgroup>
#include "../definitions.metal"
#include "ssm_common.h"

using namespace metal;

constant ushort SSM_PREFILL_MAX_STATE = 256;

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(SSDPrefill64)(
    device const T* x,      // (suffix, h, dh)
    device const T* dt_raw, // (suffix, h) - raw dt values
    device const T* b,      // (suffix, g, n)
    device const T* c,      // (suffix, g, n)
    device const T* d,      // (h)
    device const T* z,      // (suffix, h, dh)
    device T* state,        // (h, dh, n)
    device T* y,            // (suffix, h, dh)
    constant const uint& suffix_len,
    constant const int& group_size,
    constant const int& state_size,
    constant const uint* x_strides,
    constant const uint* dt_strides,
    constant const uint* cb_strides,
    constant const uint* state_strides,
    constant const uint& num_heads,
    constant const uint& head_dim,
    const uint pair_idx GROUPS(num_heads * head_dim),
    const uint lane_idx THREADS(32)
) {
  const uint simd_width = 32;
  const uint h_idx = pair_idx / head_dim;
  const uint dh_idx = pair_idx % head_dim;
  const uint safe_group = uint(max(group_size, 1));
  const uint group_idx = h_idx / safe_group;

  const uint x_token_stride = x_strides[0];
  const uint x_head_stride = x_strides[1];
  const uint x_dim_stride = x_strides[2];
  const uint dt_token_stride = dt_strides[0];
  const uint dt_head_stride = dt_strides[1];
  const uint cb_token_stride = cb_strides[0];
  const uint cb_group_stride = cb_strides[1];
  const uint cb_state_stride = cb_strides[2];
  const uint state_head_stride = state_strides[0];
  const uint state_dim_stride = state_strides[1];
  const uint state_inner_stride = state_strides[2];

  const uint x_base = h_idx * x_head_stride + dh_idx * x_dim_stride;
  const uint dt_base = h_idx * dt_head_stride;
  const uint state_base = h_idx * state_head_stride + dh_idx * state_dim_stride;
  const uint cb_group_base = group_idx * cb_group_stride;
  const float d_scalar = float(d[h_idx]);

  const uint idx0 = lane_idx;
  const uint idx1 = lane_idx + simd_width;

  float state0 = 0.0f;
  float state1 = 0.0f;
  uint state_idx0 = 0;
  uint state_idx1 = 0;
  state_idx0 = state_base + idx0 * state_inner_stride;
  state0 = float(state[state_idx0]);

  state_idx1 = state_base + idx1 * state_inner_stride;
  state1 = float(state[state_idx1]);

  uint cb_idx0 = 0;
  uint cb_idx1 = 0;
  cb_idx0 = cb_group_base + idx0 * cb_state_stride;
  cb_idx1 = cb_group_base + idx1 * cb_state_stride;

  for (uint token = 0; token < suffix_len; ++token) {
    const uint x_idx = token * x_token_stride + x_base;
    const uint dt_idx = token * dt_token_stride + dt_base;

    const float x_val = float(x[x_idx]);
    const float decay_val = fast::exp(-float(softplus(float(dt_raw[dt_idx]))));
    const float gate = float(apply_silu(z[x_idx]));
    const float skip = d_scalar * x_val;
    const float dt_scaled_input = x_val;

    float contrib = 0.0f;
    const float new_state0 =
        decay_val * state0 + dt_scaled_input * float(b[cb_idx0]);
    state0 = new_state0;
    contrib += new_state0 * float(c[cb_idx0]);
    cb_idx0 += cb_token_stride;

    const float new_state1 =
        decay_val * state1 + dt_scaled_input * float(b[cb_idx1]);
    state1 = new_state1;
    contrib += new_state1 * float(c[cb_idx1]);
    cb_idx1 += cb_token_stride;

    float dot = simd_sum(contrib);
    if (lane_idx == 0) {
      y[x_idx] = static_cast<T>((skip + dot) * gate);
    }
  }

  state[state_idx0] = static_cast<T>(state0);
  state[state_idx1] = static_cast<T>(state1);
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(SSDPrefill)(
    device const T* x,      // (suffix, h, dh)
    device const T* dt_raw, // (suffix, h) - raw dt values
    device const T* b,      // (suffix, g, n)
    device const T* c,      // (suffix, g, n)
    device const T* d,      // (h)
    device const T* z,      // (suffix, h, dh)
    device T* state,        // (h, dh, n)
    device T* y,            // (suffix, h, dh)
    constant const uint& suffix_len,
    constant const int& group_size,
    constant const int& state_size,
    constant const uint* x_strides,
    constant const uint* dt_strides,
    constant const uint* cb_strides,
    constant const uint* state_strides,
    constant const uint& num_heads,
    constant const uint& head_dim,
    const uint pair_idx GROUPS(num_heads * head_dim),
    const uint lane_idx THREADS(32)
) {
  const uint simd_width = 32;
  const int state_dim = state_size;
  const uint h_idx = pair_idx / head_dim;
  const uint dh_idx = pair_idx % head_dim;
  const uint safe_group = uint(max(group_size, 1));
  const uint group_idx = h_idx / safe_group;

  const uint x_token_stride = x_strides[0];
  const uint x_head_stride = x_strides[1];
  const uint x_dim_stride = x_strides[2];
  const uint dt_token_stride = dt_strides[0];
  const uint dt_head_stride = dt_strides[1];
  const uint cb_token_stride = cb_strides[0];
  const uint cb_group_stride = cb_strides[1];
  const uint cb_state_stride = cb_strides[2];
  const uint state_head_stride = state_strides[0];
  const uint state_dim_stride = state_strides[1];
  const uint state_inner_stride = state_strides[2];

  const uint x_base = h_idx * x_head_stride + dh_idx * x_dim_stride;
  const uint dt_base = h_idx * dt_head_stride;
  const uint state_base = h_idx * state_head_stride + dh_idx * state_dim_stride;
  const uint cb_group_base = group_idx * cb_group_stride;
  const float d_scalar = float(d[h_idx]);

  const int max_chunks = SSM_PREFILL_MAX_STATE / simd_width;
  const int chunk_count = (state_dim + int(simd_width) - 1) / int(simd_width);
  if (state_dim <= 0 || chunk_count > max_chunks) {
    return;
  }

  thread float lane_states[SSM_PREFILL_MAX_STATE / 32];

#pragma unroll
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    const int idx = chunk * int(simd_width) + int(lane_idx);
    const uint state_idx = state_base + idx * state_inner_stride;
    lane_states[chunk] = float(state[state_idx]);
  }

  for (uint token = 0; token < suffix_len; ++token) {
    const uint x_idx = token * x_token_stride + x_base;
    const uint dt_idx = token * dt_token_stride + dt_base;

    const float x_val = float(x[x_idx]);
    const float decay_val = fast::exp(-float(softplus(float(dt_raw[dt_idx]))));
    const float gate = float(apply_silu(z[x_idx]));
    const float skip = d_scalar * x_val;
    const float dt_scaled_input = x_val;

    float contrib_sum = 0.0f;
#pragma unroll
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
      const int idx = chunk * int(simd_width) + int(lane_idx);
      const uint cb_idx =
          cb_group_base + uint(idx) * cb_state_stride + token * cb_token_stride;
      const float new_state =
          decay_val * lane_states[chunk] + dt_scaled_input * float(b[cb_idx]);
      lane_states[chunk] = new_state;
      contrib_sum += new_state * float(c[cb_idx]);
    }

    float dot = simd_sum(contrib_sum);
    if (lane_idx == 0) {
      y[x_idx] = static_cast<T>((skip + dot) * gate);
    }
  }

#pragma unroll
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    const int idx = chunk * int(simd_width) + int(lane_idx);
    const uint state_idx = state_base + uint(idx) * state_inner_stride;
    state[state_idx] = static_cast<T>(lane_states[chunk]);
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(SSDPrefillSequential)(
    device const T* x,      // (suffix, h, dh)
    device const T* dt_raw, // (suffix, h) - raw dt values
    device const T* b,      // (suffix, g, n)
    device const T* c,      // (suffix, g, n)
    device const T* d,      // (h)
    device const T* z,      // (suffix, h, dh)
    device T* state,        // (h, dh, n)
    device T* y,            // (suffix, h, dh)
    constant const uint& suffix_len,
    constant const int& group_size,
    constant const int& state_size,
    constant const uint* x_strides,
    constant const uint* dt_strides,
    constant const uint* cb_strides,
    constant const uint* state_strides,
    constant const uint& channels,
    constant const uint& head_dim,
    const uint h_idx AXIS(channels, 32),
    const uint dh_idx AXIS(head_dim, 32)
) {
  const uint safe_group = uint(max(group_size, 1));
  const uint group_idx = h_idx / safe_group;
  device T* state_row = state + size_t(h_idx) * state_strides[0] +
                        size_t(dh_idx) * state_strides[1];

  for (uint token = 0; token < suffix_len; ++token) {
    const uint x_idx =
        token * x_strides[0] + h_idx * x_strides[1] + dh_idx * x_strides[2];
    const uint dt_idx = token * dt_strides[0] + h_idx * dt_strides[1];
    const uint cb_base = token * cb_strides[0] + group_idx * cb_strides[1];

    const T this_x = x[x_idx];
    const T dt_raw_val = dt_raw[dt_idx];
    const T this_dt = softplus(dt_raw_val);
    const T this_decay = static_cast<T>(fast::exp(-float(this_dt)));
    const T this_D = d[h_idx];
    const T this_z = apply_silu(z[x_idx]);
    const T dt_scaled_input = this_x;

    T acc = T(0);
    int s = 0;
    const int vec_bound = (state_size / 4) * 4;
    for (; s < vec_bound; s += 4) {
      const uint state_idx = s * state_strides[2];
      const uint cb_idx = cb_base + s * cb_strides[2];
      auto prev_state =
          *reinterpret_cast<device vec<T, 4>*>(state_row + state_idx);
      auto b_vec = *reinterpret_cast<device const vec<T, 4>*>(b + cb_idx);
      auto c_vec = *reinterpret_cast<device const vec<T, 4>*>(c + cb_idx);
      vec<T, 4> new_state = prev_state * this_decay + b_vec * dt_scaled_input;
      *reinterpret_cast<device vec<T, 4>*>(state_row + state_idx) = new_state;
      vec<T, 4> prod = new_state * c_vec;
      acc += prod.x + prod.y + prod.z + prod.w;
    }
    for (; s < state_size; ++s) {
      const uint state_idx = s * state_strides[2];
      const T prev_state = state_row[state_idx];
      const uint cb_idx = cb_base + s * cb_strides[2];
      const T new_state = prev_state * this_decay + b[cb_idx] * dt_scaled_input;
      state_row[state_idx] = new_state;
      acc += new_state * c[cb_idx];
    }

    acc += this_D * this_x;
    acc *= this_z;
    y[x_idx] = acc;
  }
}
