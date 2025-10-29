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

template <typename T>
kernel void ssd_update_kernel(
    // Input 
    device const T* x [[ buffer(0) ]],      // (b, h, dh)
    device const T* dt [[ buffer(1) ]],     // (b, h)
    device const T* decay [[ buffer(2) ]],  // (b, h)
    device const T* B [[ buffer(3) ]],      // (b, g, n)
    device const T* C [[ buffer(4) ]],      // (b, g, n)
    device const T* D [[ buffer(5) ]],      // (h)
    device const T* z [[ buffer(6) ]],      // (b, d)
    device const T* state [[ buffer(7) ]],  // (b, h, dh, n)
    device T* y [[ buffer(8) ]],
    device T* next_state [[ buffer(9) ]],
    // Parameters
    constant const int& group_size [[ buffer(10) ]],   // h = group_size * g
    constant const int& state_size [[ buffer(11) ]],
    // Strides
    constant const size_t* x_strides [[ buffer(12) ]],
    constant const size_t* dt_strides [[ buffer(13) ]],
    constant const size_t* CB_strides [[ buffer(14) ]],
    constant const size_t* state_strides [[ buffer(15) ]],
    // Grid
    uint3 grid_idx [[ thread_position_in_grid ]],
    uint3 grid_size [[ threads_per_grid ]]
) {
    const int b_idx = grid_idx.x;
    const int h_idx = grid_idx.y;
    const int dh_idx = grid_idx.z;

    const int CB_start_idx = static_cast<int>(b_idx * CB_strides[0] + (h_idx / group_size) * CB_strides[1]);
    const int x_idx = static_cast<int>(b_idx * x_strides[0] + h_idx * x_strides[1] + dh_idx * x_strides[2]);
    const int dt_idx = static_cast<int>(b_idx * dt_strides[0] + h_idx * dt_strides[1]);
    const int state_start_idx = static_cast<int>(b_idx * state_strides[0] + h_idx * state_strides[1] + dh_idx * state_strides[2]);

    // load data
    T this_x = x[x_idx]; // assumed SILU by conv kernel
    T this_dt = dt[dt_idx];
    T this_decay = decay[dt_idx];
    T this_D = D[h_idx];
    T this_z = z[x_idx];
    this_z = SILU{}(this_z);

    T temp = T(0);
    #pragma unroll
    for (int i = 0; i < state_size; ++i) {
        int CB_idx = CB_start_idx + i;
        int state_idx = state_start_idx + i;
        T this_new_state = state[state_idx] * this_decay + B[CB_idx] * this_dt * this_x;
        next_state[state_idx] = this_new_state;
        temp = temp + this_new_state * C[CB_idx];
    }
    temp = temp + this_D * this_x;
    temp = temp * this_z;
    y[x_idx] = temp;
}

#define instantiate_ssd_update_kernel(type_name, type)      \
  template [[host_name("ssd_update_kernel_" #type_name)]]   \
  kernel void ssd_update_kernel<type>(                  \
    device const type* x [[buffer(0)]],                     \
    device const type* dt [[buffer(1)]],                    \
    device const type* decay [[buffer(2)]],                 \
    device const type* B [[buffer(3)]],                     \
    device const type* C [[buffer(4)]],                     \
    device const type* D [[buffer(5)]],                     \
    device const type* z [[buffer(6)]],                     \
    device const type* state [[buffer(7)]],                 \
    device type* y [[buffer(8)]],                           \
    device type* next_state [[buffer(9)]],                  \
    constant const int& group_size [[buffer(10)]],          \
    constant const int& state_size [[buffer(11)]],          \
    constant const size_t* x_strides [[buffer(12)]],        \
    constant const size_t* dt_strides [[buffer(13)]],       \
    constant const size_t* CB_strides [[buffer(14)]],       \
    constant const size_t* state_strides [[buffer(15)]],    \
    uint3 grid_idx [[thread_position_in_grid]],             \
    uint3 grid_size [[threads_per_grid]]);

instantiate_ssd_update_kernel(float, float);
instantiate_ssd_update_kernel(bfloat, bfloat);
instantiate_ssd_update_kernel(half, half);


