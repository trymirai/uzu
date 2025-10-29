#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

struct SILU {
  template <typename T>
  T operator()(T x) {
    float xf = float(x);
    float y = 1.0f / (1.0f + fast::exp(-fabs(xf)));
    float out = (xf < 0.0f) ? (1.0f - y) * xf : y * xf;
    return static_cast<T>(out);
  }
};

struct Softplus {
  template <typename T>
  T operator()(T x) {
    float xf = float(x);
    float y = log(1.0f + fast::exp(xf));
    return static_cast<T>((xf > 20.0f) ? xf : y);
  }
};

struct Exp {
  template <typename T>
  T operator()(T x) {
    float xf = float(x);
    return static_cast<T>(fast::exp(xf));
  };
};

template <typename T>
kernel void ssm_update_kernel(
    device const T* x [[ buffer(0) ]],      // (b, h)
    device const T* dt [[ buffer(1) ]],     // (b, h)
    device const T* A [[ buffer(2) ]],      // (n)
    device const T* B [[ buffer(3) ]],      // (b, n)
    device const T* C [[ buffer(4) ]],      // (b, n)
    device const T* D [[ buffer(5) ]],      // (h)
    device const T* z [[ buffer(6) ]],      // (b, h)
    device const T* state [[ buffer(7) ]],  // (b, h, n)
    device T* y [[ buffer(8) ]],            // (b, h)
    device T* next_state [[ buffer(9) ]],   // (b, h, n)
    uint3 grid_idx [[ thread_position_in_grid ]],
    uint3 grid_size [[ threads_per_grid ]]
) {
    const int state_size = 16; // matches reference
    const int channel_size = grid_size.y;

    const int batch_idx = grid_idx.x;
    const int channel_idx = grid_idx.y;

    const int cb_start_idx = batch_idx * state_size;
    const int x_idx = batch_idx * channel_size + channel_idx;
    const int state_start_idx = x_idx * state_size;

    T this_x = x[x_idx];
    this_x = SILU{}(this_x);

    T this_z = z[x_idx];
    this_z = SILU{}(this_z);

    T delta = Softplus{}(dt[x_idx]);

    T temp = T(0);
    #pragma unroll
    for (int i = 0; i < state_size; ++i) {
        int cb_idx = cb_start_idx + i;
        int state_idx = state_start_idx + i;
        T this_new_state = state[state_idx] * Exp{}(A[i] * delta) + B[cb_idx] * delta * this_x;
        next_state[state_idx] = this_new_state;
        temp = temp + this_new_state * C[cb_idx];
    }
    temp = temp + D[channel_idx] * this_x;
    temp = temp * this_z;
    y[x_idx] = temp;
}

#define instantiate_ssm_update_kernel(type_name, type)      \
  template [[host_name("ssm_update_kernel_" #type_name)]]   \
  kernel void ssm_update_kernel<type>(                      \
    device const type* x [[ buffer(0) ]],                   \
    device const type* dt [[ buffer(1) ]],                  \
    device const type* A [[ buffer(2) ]],                   \
    device const type* B [[ buffer(3) ]],                   \
    device const type* C [[ buffer(4) ]],                   \
    device const type* D [[ buffer(5) ]],                   \
    device const type* z [[ buffer(6) ]],                   \
    device const type* state [[ buffer(7) ]],               \
    device type* y [[ buffer(8) ]],                         \
    device type* next_state [[ buffer(9) ]],                \
    uint3 grid_idx [[ thread_position_in_grid ]],           \
    uint3 grid_size [[ threads_per_grid ]]);

instantiate_ssm_update_kernel(float, float);
instantiate_ssm_update_kernel(bfloat, bfloat);
instantiate_ssm_update_kernel(half, half);


