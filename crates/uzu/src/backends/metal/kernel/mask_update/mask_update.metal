#include <metal_stdlib>
using namespace metal;

/// Updates attention mask between async passes.
/// Optionally sets one column to unmask_value (typically 0.0 for the new KV position).
/// Optionally sets another column to mask_value (typically -inf for evicted position in sliding window).
/// Pass -1 for unmask_col or mask_col to skip that update.
template <typename T>
kernel void update_attention_mask(
    device T* mask [[buffer(0)]],
    constant int& unmask_col [[buffer(1)]],    // -1 if no unmask needed
    constant int& mask_col [[buffer(2)]],      // -1 if no eviction
    constant T& unmask_value [[buffer(3)]],    // 0.0
    constant T& mask_value [[buffer(4)]],      // -inf
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        if (unmask_col >= 0) {
            mask[unmask_col] = unmask_value;
        }
        if (mask_col >= 0) {
            mask[mask_col] = mask_value;
        }
    }
}

#define instantiate_mask_update(name, type) \
    template [[host_name("update_attention_mask_" #name)]] \
    kernel void update_attention_mask<type>( \
        device type* mask [[buffer(0)]], \
        constant int& unmask_col [[buffer(1)]], \
        constant int& mask_col [[buffer(2)]], \
        constant type& unmask_value [[buffer(3)]], \
        constant type& mask_value [[buffer(4)]], \
        uint tid [[thread_position_in_grid]] \
    );

instantiate_mask_update(float, float)
instantiate_mask_update(half, half)
instantiate_mask_update(bfloat, bfloat)

