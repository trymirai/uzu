#include <metal_stdlib>
using namespace metal;

kernel void moe_gather_x_perm_f16(
    device const half* X [[buffer(0)]],
    device const int* bucketed_ids [[buffer(1)]],
    device half* x_perm [[buffer(2)]],
    device const uint* sumk_buf [[buffer(3)]],
    constant uint& d_model [[buffer(4)]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]])
{
    const uint total_rows = sumk_buf[0];
    if (total_rows == 0u || d_model == 0u) {
        return;
    }
    for (uint row = lid; row < total_rows; row += threads_per_threadgroup.x) {
        int token = bucketed_ids[row];
        if (token < 0) {
            continue;
        }
        const device half* src = X + (ulong)(uint)token * (ulong)d_model;
        device half* dst = x_perm + (ulong)row * (ulong)d_model;
        uint col = 0u;
        for (; col + 8u <= d_model; col += 8u) {
            half4 v0 = *(reinterpret_cast<const device half4*>(src + col + 0u));
            half4 v1 = *(reinterpret_cast<const device half4*>(src + col + 4u));
            *(reinterpret_cast<device half4*>(dst + col + 0u)) = v0;
            *(reinterpret_cast<device half4*>(dst + col + 4u)) = v1;
        }
        for (; col < d_model; ++col) {
            dst[col] = src[col];
        }
    }
}

kernel void moe_gather_x_perm_f32(
    device const float* X [[buffer(0)]],
    device const int* bucketed_ids [[buffer(1)]],
    device float* x_perm [[buffer(2)]],
    device const uint* sumk_buf [[buffer(3)]],
    constant uint& d_model [[buffer(4)]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]])
{
    const uint total_rows = sumk_buf[0];
    if (total_rows == 0u || d_model == 0u) {
        return;
    }
    for (uint row = lid; row < total_rows; row += threads_per_threadgroup.x) {
        int token = bucketed_ids[row];
        if (token < 0) {
            continue;
        }
        const device float* src = X + (ulong)(uint)token * (ulong)d_model;
        device float* dst = x_perm + (ulong)row * (ulong)d_model;
        uint col = 0u;
        for (; col + 8u <= d_model; col += 8u) {
            float4 v0 = *(reinterpret_cast<const device float4*>(src + col + 0u));
            float4 v1 = *(reinterpret_cast<const device float4*>(src + col + 4u));
            *(reinterpret_cast<device float4*>(dst + col + 0u)) = v0;
            *(reinterpret_cast<device float4*>(dst + col + 4u)) = v1;
        }
        for (; col < d_model; ++col) {
            dst[col] = src[col];
        }
    }
}

kernel void moe_gather_x_perm_bf16(
    device const bfloat* X [[buffer(0)]],
    device const int* bucketed_ids [[buffer(1)]],
    device bfloat* x_perm [[buffer(2)]],
    device const uint* sumk_buf [[buffer(3)]],
    constant uint& d_model [[buffer(4)]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]])
{
    const uint total_rows = sumk_buf[0];
    if (total_rows == 0u || d_model == 0u) {
        return;
    }
    for (uint row = lid; row < total_rows; row += threads_per_threadgroup.x) {
        int token = bucketed_ids[row];
        if (token < 0) {
            continue;
        }
        const device bfloat* src = X + (ulong)(uint)token * (ulong)d_model;
        device bfloat* dst = x_perm + (ulong)row * (ulong)d_model;
        for (uint col = 0u; col < d_model; ++col) {
            dst[col] = src[col];
        }
    }
}
