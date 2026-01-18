#include <metal_stdlib>
using namespace metal;

template <typename T>
[[max_total_threads_per_threadgroup(1)]]
kernel void update_attention_mask(
    device T* mask [[buffer(0)]],
    constant int& unmask_col [[buffer(1)]],
    constant int& mask_col [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
  if (tid == 0) {
    if (unmask_col >= 0) {
      mask[unmask_col] = T(0);
    }
    if (mask_col >= 0) {
      mask[mask_col] = -numeric_limits<T>::infinity();
    }
  }
}

#define instantiate_mask_update(name, type)                                    \
  template [[host_name("update_attention_mask_" #name)]]                       \
  kernel void update_attention_mask<type>(                                     \
      device type * mask [[buffer(0)]],                                        \
      constant int& unmask_col [[buffer(1)]],                                  \
      constant int& mask_col [[buffer(2)]],                                    \
      uint tid [[thread_position_in_grid]]                                     \
  );

instantiate_mask_update(float, float) instantiate_mask_update(
    half,
    half
) instantiate_mask_update(bfloat, bfloat)
