#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Router matmul via simdgroup dot-product reduction with vectorized loads.
// Optimized: Threadgroup caches input vector, 8 simdgroups per TG (256 threads)
// - threadsPerTG = (256, 1, 1)  [8 simdgroups × 32 lanes]
// - threadgroups = (ceil_div(E, 8), T, 1)

kernel void moe_router_bf16(
    const device bfloat4* input [[buffer(0)]],   // [T, d_model/4]
    const device bfloat4* weight [[buffer(1)]],  // [E, d_model/4]
    const device bfloat* bias [[buffer(2)]],     // [E]
    device bfloat* output [[buffer(3)]],         // [T, E]
    constant uint& T [[buffer(4)]],
    constant uint& d_model [[buffer(5)]],
    constant uint& E [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]],
    uint lid [[thread_index_in_threadgroup]])
{
    const uint token_idx = gid.y;
    const uint num_column_vecs = d_model / 4u;
    const uint row = gid.x * num_simdgroups + simdgroup_idx;
    if (token_idx >= T || row >= E) return;

    const device bfloat4* x_vec = input + (ulong)token_idx * (ulong)num_column_vecs;
    
    // Threadgroup cache: load input once, share across simdgroups
    threadgroup float4 x_cache[1024]; // Max 4096/4
    for (uint c = lid; c < num_column_vecs; c += (32u * num_simdgroups)) {
        x_cache[c] = float4(x_vec[c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device bfloat4* w_vec = weight + (ulong)row * (ulong)num_column_vecs;

    float4 sum4 = float4(0.0f);
    for (uint c = simd_lane; c < num_column_vecs; c += 32u) {
        const float4 wv = float4(w_vec[c]);
        const float4 xv = x_cache[c]; // Read from TG cache
        sum4 = fma(wv, xv, sum4);
    }
    float sum = (sum4.x + sum4.y) + (sum4.z + sum4.w);
    sum = simd_sum(sum);
    if (simd_is_first()) {
        sum += float(bias[row]);
        output[(ulong)token_idx * (ulong)E + (ulong)row] = bfloat(sum);
    }
}

kernel void moe_router_f16(
    const device half4* input [[buffer(0)]],     // [T, d_model/4]
    const device half4* weight [[buffer(1)]],    // [E, d_model/4]
    const device half* bias [[buffer(2)]],       // [E]
    device half* output [[buffer(3)]],           // [T, E]
    constant uint& T [[buffer(4)]],
    constant uint& d_model [[buffer(5)]],
    constant uint& E [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]],
    uint lid [[thread_index_in_threadgroup]])
{
    const uint token_idx = gid.y;
    const uint num_column_vecs = d_model / 4u;
    const uint row = gid.x * num_simdgroups + simdgroup_idx;
    if (token_idx >= T || row >= E) return;

    const device half4* x_vec = input + (ulong)token_idx * (ulong)num_column_vecs;
    
    // Threadgroup cache: load input once, share across simdgroups
    threadgroup float4 x_cache[1024];
    for (uint c = lid; c < num_column_vecs; c += (32u * num_simdgroups)) {
        x_cache[c] = float4(x_vec[c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device half4* w_vec = weight + (ulong)row * (ulong)num_column_vecs;

    float4 sum4 = float4(0.0f);
    for (uint c = simd_lane; c < num_column_vecs; c += 32u) {
        const float4 wv = float4(w_vec[c]);
        const float4 xv = x_cache[c];
        sum4 = fma(wv, xv, sum4);
    }
    float sum = (sum4.x + sum4.y) + (sum4.z + sum4.w);
    sum = simd_sum(sum);
    if (simd_is_first()) {
        sum += float(bias[row]);
        output[(ulong)token_idx * (ulong)E + (ulong)row] = half(sum);
    }
}

kernel void moe_router_f32(
    const device float4* input [[buffer(0)]],    // [T, d_model/4]
    const device float4* weight [[buffer(1)]],   // [E, d_model/4]
    const device float* bias [[buffer(2)]],      // [E]
    device float* output [[buffer(3)]],          // [T, E]
    constant uint& T [[buffer(4)]],
    constant uint& d_model [[buffer(5)]],
    constant uint& E [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]],
    uint lid [[thread_index_in_threadgroup]])
{
    const uint token_idx = gid.y;
    const uint num_column_vecs = d_model / 4u;
    const uint row = gid.x * num_simdgroups + simdgroup_idx;
    if (token_idx >= T || row >= E) return;

    const device float4* x_vec = input + (ulong)token_idx * (ulong)num_column_vecs;
    
    // Threadgroup cache: load input once, share across simdgroups
    threadgroup float4 x_cache[1024];
    for (uint c = lid; c < num_column_vecs; c += (32u * num_simdgroups)) {
        x_cache[c] = x_vec[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device float4* w_vec = weight + (ulong)row * (ulong)num_column_vecs;

    float4 sum4 = float4(0.0f);
    for (uint c = simd_lane; c < num_column_vecs; c += 32u) {
        const float4 wv = w_vec[c];
        const float4 xv = x_cache[c];
        sum4 = fma(wv, xv, sum4);
    }
    float sum = (sum4.x + sum4.y) + (sum4.z + sum4.w);
    sum = simd_sum(sum);
    if (simd_is_first()) {
        sum += bias[row];
        output[(ulong)token_idx * (ulong)E + (ulong)row] = sum;
    }
}