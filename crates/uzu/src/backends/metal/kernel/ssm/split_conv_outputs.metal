#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
kernel void ssm_split_conv_outputs_kernel(
    device const T* conv_input [[ buffer(0) ]],
    device       T* x_out [[ buffer(1) ]],
    device       T* b_out [[ buffer(2) ]],
    device       T* c_out [[ buffer(3) ]],
    constant const int& conv_dim [[ buffer(4) ]],
    constant const int& inner_dim [[ buffer(5) ]],
    constant const int& proj_dim [[ buffer(6) ]],
    uint2 grid_idx [[ thread_position_in_grid ]],
    uint2 grid_size [[ threads_per_grid ]]
) {
    const int row = static_cast<int>(grid_idx.x);
    const int col = static_cast<int>(grid_idx.y);

    if (row >= grid_size.x || col >= grid_size.y || col >= conv_dim) {
        return;
    }

    const int input_idx = row * conv_dim + col;
    if (col < inner_dim) {
        const int dst = row * inner_dim + col;
        x_out[dst] = conv_input[input_idx];
    } else if (col < inner_dim + proj_dim) {
        const int dst = row * proj_dim + (col - inner_dim);
        b_out[dst] = conv_input[input_idx];
    } else if (col < inner_dim + 2 * proj_dim) {
        const int dst = row * proj_dim + (col - inner_dim - proj_dim);
        c_out[dst] = conv_input[input_idx];
    }
}

#define instantiate_ssm_split_conv_outputs_kernel(type_name, type)      \
  template [[host_name("ssm_split_conv_outputs_kernel_" #type_name)]]   \
  kernel void ssm_split_conv_outputs_kernel<type>(                      \
    device const type* conv_input [[ buffer(0) ]],                      \
    device       type* x_out [[ buffer(1) ]],                           \
    device       type* b_out [[ buffer(2) ]],                           \
    device       type* c_out [[ buffer(3) ]],                           \
    constant const int& conv_dim [[ buffer(4) ]],                       \
    constant const int& inner_dim [[ buffer(5) ]],                      \
    constant const int& proj_dim [[ buffer(6) ]],                       \
    uint2 grid_idx [[ thread_position_in_grid ]],                       \
    uint2 grid_size [[ threads_per_grid ]]                              \
  );

instantiate_ssm_split_conv_outputs_kernel(float, float);
instantiate_ssm_split_conv_outputs_kernel(bfloat, bfloat);
instantiate_ssm_split_conv_outputs_kernel(half, half);

#undef instantiate_ssm_split_conv_outputs_kernel
