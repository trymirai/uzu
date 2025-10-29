#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
inline void ssd_update_no_z_kernel(
    // Input 
    device const T* x,      // (b, h, dh)
    device const T* dt,     // (b, h)
    device const T* decay,  // (b, h)
    device const T* B,      // (b, g, n)
    device const T* C,      // (b, g, n)
    device const T* D,      // (h)
    device const T* state,  // (b, h, dh, n)
    device T* y,
    device T* next_state,
    // Parameters
    constant const int& group_size,  // h = group_size * g
    constant const int& state_size,
    // Strides
    constant const size_t* x_strides,
    constant const size_t* dt_strides,
    constant const size_t* CB_strides,
    constant const size_t* state_strides,
    // Grid
    uint3 grid_idx,
    uint3 /*grid_size*/
) {
    const int b_idx = grid_idx.x;
    const int h_idx = grid_idx.y;
    const int dh_idx = grid_idx.z;

    const int CB_start_idx = static_cast<int>(b_idx * CB_strides[0] + (h_idx / group_size) * CB_strides[1]);
    const int x_idx = static_cast<int>(b_idx * x_strides[0] + h_idx * x_strides[1] + dh_idx * x_strides[2]);
    const int dt_idx = static_cast<int>(b_idx * dt_strides[0] + h_idx * dt_strides[1]);
    const int state_start_idx = static_cast<int>(b_idx * state_strides[0] + h_idx * state_strides[1] + dh_idx * state_strides[2]);

    T this_x = x[x_idx];
    T this_dt = dt[dt_idx];
    T this_decay = decay[dt_idx];
    T this_D = D[h_idx];

    T temp = T(0);
    #pragma unroll
    for (int i = 0; i < state_size; ++i) {
        int CB_idx = CB_start_idx + i;
        int state_idx = state_start_idx + i;
        T this_new_state = state[state_idx] * this_decay + B[CB_idx] * this_dt * this_x; 
        next_state[state_idx] = this_new_state;
        temp = temp + this_new_state * C[CB_idx];
    }
    temp = temp + this_D * this_x;  // Skip connection
    y[x_idx] = temp; 
}

#define outerArguments(T)                                                      \
(device const T* x           [[ buffer(0) ]],                                   \
 device const T* dt          [[ buffer(1) ]],                                   \
 device const T* decay       [[ buffer(2) ]],                                   \
 device const T* B           [[ buffer(3) ]],                                   \
 device const T* C           [[ buffer(4) ]],                                   \
 device const T* D           [[ buffer(5) ]],                                   \
 device const T* state       [[ buffer(6) ]],                                   \
 device       T* y           [[ buffer(7) ]],                                   \
 device       T* next_state  [[ buffer(8) ]],                                   \
 constant const int& group_size [[ buffer(9) ]],                                 \
 constant const int& state_size [[ buffer(10) ]],                                \
 constant const size_t* x_strides [[ buffer(11) ]],                              \
 constant const size_t* dt_strides [[ buffer(12) ]],                             \
 constant const size_t* CB_strides [[ buffer(13) ]],                             \
 constant const size_t* state_strides [[ buffer(14) ]],                          \
 uint3 grid_idx             [[ thread_position_in_grid ]],                      \
 uint3 grid_size            [[ threads_per_grid ]])

#define innerArguments (x, dt, decay, B, C, D, state, y, next_state, group_size, state_size, x_strides, dt_strides, CB_strides, state_strides, grid_idx, grid_size)

generateKernels(ssd_update_no_z_kernel)

#undef outerArguments
#undef innerArguments


