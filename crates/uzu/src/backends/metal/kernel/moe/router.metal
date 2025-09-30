#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Router matmul: X[T, d_model] · W[d_model, E] + b → logits[T, E]
kernel void moe_router_bf16(
    const device bfloat4* input [[buffer(0)]],     // [T, d_model/4]
    const device bfloat4* weight [[buffer(1)]],    // [E, d_model/4]
    const device bfloat* bias [[buffer(2)]],       // [E]
    device bfloat* output [[buffer(3)]],           // [T, E]
    constant uint& T [[buffer(4)]],
    constant uint& d_model [[buffer(5)]],
    constant uint& E [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simdgroup_tid [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]])
{
    const uint simdgroup_size = 32;
    const uint num_column_vecs = d_model / 4;
    const uint row = gid.x * num_simdgroups + simdgroup_idx;
    
    if (gid.y >= T || row >= E) return;
    
    input += gid.y * num_column_vecs + simdgroup_tid;
    weight += num_column_vecs * row + simdgroup_tid;
    bias += row;
    output += gid.y * E + row;
    
    uint num_iter = (num_column_vecs - simdgroup_tid + (simdgroup_size - 1)) / simdgroup_size;
    
    float4 sum4 = 0.0f;
    do {
        const bfloat4 w = *weight;
        const float4 i = float4(*input);
        sum4 = metal::fma(float4(w), i, sum4);
        
        weight += simdgroup_size;
        input += simdgroup_size;
    } while (--num_iter != 0);
    
    const float2 sum2 = sum4.xy + sum4.zw;
    float sum = sum2.x + sum2.y;
    sum = metal::simd_sum(sum);
    
    if (metal::simd_is_first()) {
        sum += float(*bias);
        *output = bfloat(sum);
    }
}

kernel void moe_router_f16(
    const device half4* input [[buffer(0)]],       // [T, d_model/4]
    const device half4* weight [[buffer(1)]],      // [E, d_model/4]
    const device half* bias [[buffer(2)]],         // [E]
    device half* output [[buffer(3)]],             // [T, E]
    constant uint& T [[buffer(4)]],
    constant uint& d_model [[buffer(5)]],
    constant uint& E [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simdgroup_tid [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]])
{
    const uint simdgroup_size = 32;
    const uint num_column_vecs = d_model / 4;
    const uint row = gid.x * num_simdgroups + simdgroup_idx;
    
    if (gid.y >= T || row >= E) return;
    
    input += gid.y * num_column_vecs + simdgroup_tid;
    weight += num_column_vecs * row + simdgroup_tid;
    bias += row;
    output += gid.y * E + row;
    
    uint num_iter = (num_column_vecs - simdgroup_tid + (simdgroup_size - 1)) / simdgroup_size;
    
    float4 sum4 = 0.0f;
    do {
        const half4 w = *weight;
        const half4 i = *input;
        sum4 = metal::fma(float4(w), float4(i), sum4);
        
        weight += simdgroup_size;
        input += simdgroup_size;
    } while (--num_iter != 0);
    
    const float2 sum2 = sum4.xy + sum4.zw;
    float sum = sum2.x + sum2.y;
    sum = metal::simd_sum(sum);
    
    if (metal::simd_is_first()) {
        sum += float(*bias);
        *output = half(sum);
    }
}
