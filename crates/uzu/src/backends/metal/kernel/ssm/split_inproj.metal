#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
kernel void ssm_split_inproj_kernel(
    device const T* input [[ buffer(0) ]],
    device       T* conv_out [[ buffer(1) ]],
    device       T* z_out [[ buffer(2) ]],
    device       T* dt_out [[ buffer(3) ]],
    constant const int& total_dim [[ buffer(4) ]],
    constant const int& conv_dim [[ buffer(5) ]],
    constant const int& inner_dim [[ buffer(6) ]],
    constant const int& num_heads [[ buffer(7) ]],
    uint2 grid_idx [[ thread_position_in_grid ]],
    uint2 grid_size [[ threads_per_grid ]]
) {
    const int row = static_cast<int>(grid_idx.x);
    const int col = static_cast<int>(grid_idx.y);

    if (row >= grid_size.x || col >= grid_size.y || col >= total_dim) {
        return;
    }

    const int input_idx = row * total_dim + col;
    if (col < conv_dim) {
        const int dst = row * conv_dim + col;
        conv_out[dst] = input[input_idx];
    } else if (col < conv_dim + inner_dim) {
        const int dst = row * inner_dim + (col - conv_dim);
        z_out[dst] = input[input_idx];
    } else if (col < conv_dim + inner_dim + num_heads) {
        const int dst = row * num_heads + (col - conv_dim - inner_dim);
        dt_out[dst] = input[input_idx];
    }
}

#define instantiate_ssm_split_inproj_kernel(type_name, type)      \
  template [[host_name("ssm_split_inproj_kernel_" #type_name)]]   \
  kernel void ssm_split_inproj_kernel<type>(                      \
    device const type* input [[ buffer(0) ]],                     \
    device       type* conv_out [[ buffer(1) ]],                  \
    device       type* z_out [[ buffer(2) ]],                     \
    device       type* dt_out [[ buffer(3) ]],                    \
    constant const int& total_dim [[ buffer(4) ]],                \
    constant const int& conv_dim [[ buffer(5) ]],                 \
    constant const int& inner_dim [[ buffer(6) ]],                \
    constant const int& num_heads [[ buffer(7) ]],                \
    uint2 grid_idx [[ thread_position_in_grid ]],                 \
    uint2 grid_size [[ threads_per_grid ]]                        \
  );

instantiate_ssm_split_inproj_kernel(float, float);
instantiate_ssm_split_inproj_kernel(bfloat, bfloat);
instantiate_ssm_split_inproj_kernel(half, half);

#undef instantiate_ssm_split_inproj_kernel
