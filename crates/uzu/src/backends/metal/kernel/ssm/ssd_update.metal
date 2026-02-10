#include <metal_stdlib>
#include "../definitions.metal"
#include "ssm_common.h"

using namespace metal;

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(SSDUpdate)(
    // Input
    device const T* x,      // (b, h, dh)
    device const T* dt_raw, // (b, h)
    device const T* b,      // (b, g, n)
    device const T* c,      // (b, g, n)
    device const T* d,      // (h)
    device const T* z,      // (b, d)
    device const T* state,  // (b, h, dh, n)
    device T* y,
    device T* next_state,
    // Parameters
    constant const uint& group_size, // h = group_size * g
    constant const uint& state_size,
    // Strides
    constant const uint x_strides[3],
    constant const uint dt_strides[2],
    constant const uint cb_strides[3],
    constant const uint state_strides[4],
    // Threads count
    constant const uint& b_size,
    constant const uint& h_size,
    constant const uint& dh_size,
    // Grid
    const uint b_idx AXIS(b_size, 32),
    const uint h_idx AXIS(h_size, 32),
    const uint dh_idx AXIS(dh_size, 1)
) {
  const uint cb_start_idx =
      b_idx * cb_strides[0] + (h_idx / group_size) * cb_strides[1];
  const uint x_idx =
      b_idx * x_strides[0] + h_idx * x_strides[1] + dh_idx * x_strides[2];
  const uint dt_idx = b_idx * dt_strides[0] + h_idx * dt_strides[1];
  const uint state_start_idx = b_idx * state_strides[0] +
                               h_idx * state_strides[1] +
                               dh_idx * state_strides[2];

  // load data
  T this_x = x[x_idx];
  T dt_raw_val = dt_raw[dt_idx];
  T this_dt = softplus(dt_raw_val);
  T this_decay = static_cast<T>(fast::exp(-float(this_dt)));
  T this_d = d[h_idx];
  T this_z = z[x_idx];
  this_z = apply_silu(this_z);
  T dt_scaled_input = this_x;

  T temp = T(0);
#pragma unroll
  for (uint i = 0; i < state_size; ++i) {
    uint cb_idx = cb_start_idx + i;
    uint state_idx = state_start_idx + i;
    T this_new_state =
        state[state_idx] * this_decay + b[cb_idx] * dt_scaled_input;
    next_state[state_idx] = this_new_state;
    temp = temp + this_new_state * c[cb_idx];
  }
  temp = temp + this_d * this_x;
  temp = temp * this_z;
  y[x_idx] = temp;
}