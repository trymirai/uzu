#include <metal_stdlib>
using namespace metal;

// Transform fused expert weights from [E, d_model, 2*d_ff] to separate W1[E, d_ff, d_model] and W3[E, d_ff, d_model]
kernel void transpose_split_fused_expert_weights_f16(
    device const half* fused_src [[buffer(0)]],  // [E, d_model, 2*d_ff]
    device half* w1_dst [[buffer(1)]],           // [E, d_ff, d_model]
    device half* w3_dst [[buffer(2)]],           // [E, d_ff, d_model]
    constant uint& E [[buffer(3)]],
    constant uint& d_model [[buffer(4)]],
    constant uint& d_ff [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint expert_idx = gid.z;
    const uint ff_idx = gid.y;
    const uint dm_idx = gid.x;
    
    if (expert_idx >= E || ff_idx >= d_ff || dm_idx >= d_model) return;
    
    // Source: [E, d_model, 2*d_ff] layout
    const ulong src_idx = (ulong)expert_idx * (ulong)d_model * (ulong)(d_ff * 2)
                        + (ulong)dm_idx * (ulong)(d_ff * 2)
                        + (ulong)ff_idx;
    
    // Destination: [E, d_ff, d_model] layout
    const ulong dst_idx = (ulong)expert_idx * (ulong)d_ff * (ulong)d_model
                        + (ulong)ff_idx * (ulong)d_model
                        + (ulong)dm_idx;
    
    // W1 is first d_ff elements, W3 is second d_ff elements
    w1_dst[dst_idx] = fused_src[src_idx];
    w3_dst[dst_idx] = fused_src[src_idx + d_ff];
}

kernel void transpose_split_fused_expert_weights_bf16(
    device const bfloat* fused_src [[buffer(0)]],  // [E, d_model, 2*d_ff]
    device bfloat* w1_dst [[buffer(1)]],           // [E, d_ff, d_model]
    device bfloat* w3_dst [[buffer(2)]],           // [E, d_ff, d_model]
    constant uint& E [[buffer(3)]],
    constant uint& d_model [[buffer(4)]],
    constant uint& d_ff [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint expert_idx = gid.z;
    const uint ff_idx = gid.y;
    const uint dm_idx = gid.x;
    
    if (expert_idx >= E || ff_idx >= d_ff || dm_idx >= d_model) return;
    
    const ulong src_idx = (ulong)expert_idx * (ulong)d_model * (ulong)(d_ff * 2)
                        + (ulong)dm_idx * (ulong)(d_ff * 2)
                        + (ulong)ff_idx;
    
    const ulong dst_idx = (ulong)expert_idx * (ulong)d_ff * (ulong)d_model
                        + (ulong)ff_idx * (ulong)d_model
                        + (ulong)dm_idx;
    
    // W1 is first d_ff elements, W3 is second d_ff elements
    w1_dst[dst_idx] = fused_src[src_idx];
    w3_dst[dst_idx] = fused_src[src_idx + d_ff];
}
