#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
inline T softplus(T x) {
    float xf = float(x);
    if (xf > 20.0f) {
        return x;
    }
    return static_cast<T>(log(1.0f + fast::exp(xf)));
}

template <typename T>
kernel void ssm_dt_decay_kernel(
    device       T* dt [[ buffer(0) ]],
    device       T* decay [[ buffer(1) ]],
    constant const int& num_heads [[ buffer(2) ]],
    constant const int& suffix_length [[ buffer(3) ]],
    uint2 grid_idx [[ thread_position_in_grid ]],
    uint2 grid_size [[ threads_per_grid ]]
) {
    const int row = static_cast<int>(grid_idx.x);
    const int head = static_cast<int>(grid_idx.y);

    if (row >= suffix_length || head >= num_heads) {
        return;
    }

    const int idx = row * num_heads + head;
    const T raw = dt[idx];
    const T delta = softplus(raw);
    dt[idx] = delta;
    decay[idx] = static_cast<T>(fast::exp(-float(delta)));
}

#define instantiate_ssm_dt_decay_kernel(type_name, type)      \
  template [[host_name("ssm_dt_decay_kernel_" #type_name)]]   \
  kernel void ssm_dt_decay_kernel<type>(                      \
    device       type* dt [[ buffer(0) ]],                    \
    device       type* decay [[ buffer(1) ]],                 \
    constant const int& num_heads [[ buffer(2) ]],            \
    constant const int& suffix_length [[ buffer(3) ]],        \
    uint2 grid_idx [[ thread_position_in_grid ]],             \
    uint2 grid_size [[ threads_per_grid ]]                    \
  );

instantiate_ssm_dt_decay_kernel(float, float);
instantiate_ssm_dt_decay_kernel(bfloat, bfloat);
instantiate_ssm_dt_decay_kernel(half, half);

#undef instantiate_ssm_dt_decay_kernel
