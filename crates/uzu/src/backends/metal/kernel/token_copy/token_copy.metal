#include <metal_stdlib>
using namespace metal;

kernel void copy_sampled_token(
    device const uint32_t* src [[buffer(0)]],
    device uint64_t* dst [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx == 0) {
        dst[0] = static_cast<uint64_t>(src[0]);
    }
}

