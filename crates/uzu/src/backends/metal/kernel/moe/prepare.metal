#include <metal_stdlib>
using namespace metal;

// Tiling config for 2D gather
constant uint ROWS_PER_TG = 8u;
constant uint COL_VECS = 32u;  // 256 threads = 8×32
constant uint ELEMS_PER_THREAD = 8u;  // each thread copies 8 elements (2× vec4)

// ========== 2D Tiled Gather Template ==========
template<typename T>
void moe_gather_x_perm_2d_impl(
    device const T* X,
    device const int* bucketed_ids,
    device T* x_perm,
    device const uint* sumk_buf,
    uint d_model,
    uint3 tgid,
    uint lid)
{
    const uint total_rows = sumk_buf[0];
    if (total_rows == 0u || d_model == 0u) return;

    // Thread organization: lid = row_in_tg * COL_VECS + col_vec
    const uint row_in_tg = lid / COL_VECS;
    const uint col_vec = lid % COL_VECS;

    // Global row assignment
    const uint base_row = tgid.x * ROWS_PER_TG;
    const uint row = base_row + row_in_tg;

    if (row >= total_rows) return;

    int token = bucketed_ids[row];
    if (token < 0) return;

    const device T* src = X + (ulong)(uint)token * (ulong)d_model;
    device T* dst = x_perm + (ulong)row * (ulong)d_model;

    // Each thread copies ELEMS_PER_THREAD elements
    // Thread i processes: col_vec*8, col_vec*8+256, col_vec*8+512, ...
    uint col = col_vec * ELEMS_PER_THREAD;
    const uint stride = COL_VECS * ELEMS_PER_THREAD;

    // Vectorized copy: 8 elements = 2× vec4
    using Vec4 = typename metal::vec<T, 4>;
    while (col + 7u < d_model) {
        Vec4 v0 = *(reinterpret_cast<const device Vec4*>(src + col));
        Vec4 v1 = *(reinterpret_cast<const device Vec4*>(src + col + 4u));
        *(reinterpret_cast<device Vec4*>(dst + col)) = v0;
        *(reinterpret_cast<device Vec4*>(dst + col + 4u)) = v1;
        col += stride;
    }

    // Scalar remainder
    if (col < d_model) {
        const uint limit = min(d_model, col + ELEMS_PER_THREAD);
        for (uint tail = col; tail < limit; ++tail) {
            dst[tail] = src[tail];
        }
    }
}

// ========== 1D Gather Template ==========
template<typename T>
void moe_gather_x_perm_1d_impl(
    device const T* X,
    device const int* bucketed_ids,
    device T* x_perm,
    device const uint* sumk_buf,
    uint d_model,
    uint lid,
    uint threads_x)
{
    const uint total_rows = sumk_buf[0];
    if (total_rows == 0u || d_model == 0u) return;

    const uint num_vec4 = d_model / 4u;

    // Each thread processes multiple rows sequentially
    for (uint row = lid; row < total_rows; row += threads_x) {
        int token = bucketed_ids[row];
        if (token < 0) continue;

        const device T* src = X + (ulong)(uint)token * (ulong)d_model;
        device T* dst = x_perm + (ulong)row * (ulong)d_model;

        // Vectorized copy: 4 elements at a time
        using Vec4 = typename metal::vec<T, 4>;
        for (uint vec_idx = 0u; vec_idx < num_vec4; vec_idx++) {
            uint col = vec_idx * 4u;
            Vec4 v = *(reinterpret_cast<const device Vec4*>(src + col));
            *(reinterpret_cast<device Vec4*>(dst + col)) = v;
        }

        // Scalar remainder
        for (uint col = num_vec4 * 4u; col < d_model; col++) {
            dst[col] = src[col];
        }
    }
}

// ========== Kernel Entry Points ==========

// 2D Tiled kernels
#define GATHER_2D_KERNEL(DTYPE, SUFFIX) \
kernel void moe_gather_x_perm_##SUFFIX##_2d( \
    device const DTYPE* X [[buffer(0)]], \
    device const int* bucketed_ids [[buffer(1)]], \
    device DTYPE* x_perm [[buffer(2)]], \
    device const uint* sumk_buf [[buffer(3)]], \
    constant uint& d_model [[buffer(4)]], \
    uint3 tgid [[threadgroup_position_in_grid]], \
    uint lid [[thread_index_in_threadgroup]]) \
{ \
    moe_gather_x_perm_2d_impl<DTYPE>(X, bucketed_ids, x_perm, sumk_buf, d_model, tgid, lid); \
}

// 1D kernels
#define GATHER_1D_KERNEL(DTYPE, SUFFIX) \
kernel void moe_gather_x_perm_##SUFFIX( \
    device const DTYPE* X [[buffer(0)]], \
    device const int* bucketed_ids [[buffer(1)]], \
    device DTYPE* x_perm [[buffer(2)]], \
    device const uint* sumk_buf [[buffer(3)]], \
    constant uint& d_model [[buffer(4)]], \
    uint lid [[thread_index_in_threadgroup]], \
    uint3 threads_per_threadgroup [[threads_per_threadgroup]]) \
{ \
    moe_gather_x_perm_1d_impl<DTYPE>(X, bucketed_ids, x_perm, sumk_buf, d_model, lid, threads_per_threadgroup.x); \
}

// Instantiate for all types
GATHER_2D_KERNEL(bfloat, bf16)
GATHER_2D_KERNEL(half, f16)
GATHER_2D_KERNEL(float, f32)

GATHER_1D_KERNEL(bfloat, bf16)
GATHER_1D_KERNEL(half, f16)
GATHER_1D_KERNEL(float, f32)
