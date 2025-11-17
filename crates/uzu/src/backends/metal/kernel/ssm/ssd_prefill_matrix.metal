#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "../definitions.metal"

using namespace metal;

constant ushort SSD_M2_CHUNK = 64;

struct SILU {
  template <typename T>
  T operator()(T x) const {
    float xf = float(x);
    float y = 1.0f / (1.0f + fast::exp(-fabs(xf)));
    float out = (xf < 0.0f) ? (1.0f - y) * xf : y * xf;
    return static_cast<T>(out);
  }
};

//------------------------------------------------------------------------------
// DT preprocess + dtA build (dtA = -dt for current models)
//------------------------------------------------------------------------------

template <typename T>
kernel void ssd_m2_dt_preprocess_kernel(
    device const T* decay_in   [[ buffer(0) ]],
    device       float* dtA    [[ buffer(1) ]],
    constant const uint& T_len [[ buffer(2) ]],
    constant const uint& H     [[ buffer(3) ]],
    uint2 tid [[ thread_position_in_grid ]]
) {
    const uint t = tid.x;
    const uint h = tid.y;
    if (t >= T_len || h >= H) {
        return;
    }

    const size_t idx = size_t(t) * H + h;
    const float decay_val = clamp(float(decay_in[idx]), 1e-6f, 1.0f);
    dtA[idx] = log(decay_val);
}

//------------------------------------------------------------------------------
// Chunk prefix scan over dtA
//------------------------------------------------------------------------------

kernel void ssd_m2_dtA_prefix_chunk_kernel(
    device const float* dtA         [[ buffer(0) ]],
    device       float* prefix      [[ buffer(1) ]],
    device       float* chunk_sums  [[ buffer(2) ]],
    constant const uint& T_len      [[ buffer(3) ]],
    constant const uint& H          [[ buffer(4) ]],
    constant const uint& num_chunks [[ buffer(5) ]],
    uint2 tg_pos  [[ threadgroup_position_in_grid ]],
    ushort lid    [[ thread_index_in_threadgroup ]]
) {
    const uint h = tg_pos.x;
    const uint chunk_idx = tg_pos.y;
    if (h >= H || chunk_idx >= num_chunks) {
        return;
    }

    const size_t chunk_start = size_t(chunk_idx) * size_t(SSD_M2_CHUNK);
    if (chunk_start >= T_len) {
        return;
    }

    const ushort chunk_len = ushort(min(
        size_t(SSD_M2_CHUNK),
        size_t(T_len) - chunk_start));

    threadgroup float shared[SSD_M2_CHUNK];

    float value = 0.0f;
    bool lane_active = lid < chunk_len;
    if (lane_active) {
        const size_t t = chunk_start + size_t(lid);
        const size_t idx = t * H + h;
        value = dtA[idx];
    }

    float prefix_excl = threadgroup_raking_prefix_exclusive_sum<SSD_M2_CHUNK, float>(
        value, shared, lid);

    if (lane_active) {
        const float local_incl = prefix_excl + value;
        const size_t t = chunk_start + size_t(lid);
        prefix[t * H + h] = local_incl;
        if (lid == chunk_len - 1) {
            chunk_sums[size_t(chunk_idx) * H + h] = local_incl;
        }
    }
}

kernel void ssd_m2_dtA_chunk_scan_kernel(
    device const float* chunk_sums    [[ buffer(0) ]],
    device       float* chunk_offsets [[ buffer(1) ]],
    constant const uint& num_chunks   [[ buffer(2) ]],
    constant const uint& H            [[ buffer(3) ]],
    uint h [[ thread_position_in_grid ]]
) {
    if (h >= H) {
        return;
    }
    float running = 0.0f;
    for (uint c = 0; c < num_chunks; ++c) {
        const size_t idx = size_t(c) * H + h;
        chunk_offsets[idx] = running;
        running += chunk_sums[idx];
    }
}

kernel void ssd_m2_dtA_prefix_apply_kernel(
    device       float* prefix        [[ buffer(0) ]],
    device const float* chunk_offsets [[ buffer(1) ]],
    constant const uint& T_len        [[ buffer(2) ]],
    constant const uint& H            [[ buffer(3) ]],
    constant const uint& num_chunks   [[ buffer(4) ]],
    uint2 tg_pos  [[ threadgroup_position_in_grid ]],
    ushort lid    [[ thread_index_in_threadgroup ]]
) {
    const uint h = tg_pos.x;
    const uint chunk_idx = tg_pos.y;
    if (h >= H || chunk_idx >= num_chunks) {
        return;
    }

    const size_t chunk_start = size_t(chunk_idx) * size_t(SSD_M2_CHUNK);
    if (chunk_start >= T_len) {
        return;
    }

    const ushort chunk_len = ushort(min(
        size_t(SSD_M2_CHUNK),
        size_t(T_len) - chunk_start));
    if (lid >= chunk_len) {
        return;
    }

    const float offset = chunk_offsets[size_t(chunk_idx) * H + h];
    const size_t t = chunk_start + size_t(lid);
    prefix[t * H + h] += offset;
}

template <typename T>
kernel void ssd_m2_decay_last_kernel(
    device const float* prefix [[ buffer(0) ]],
    device       T*     decay_last [[ buffer(1) ]],
    constant const uint& T_len [[ buffer(2) ]],
    constant const uint& H     [[ buffer(3) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    const uint t = gid.x;
    const uint h = gid.y;
    if (t >= T_len || h >= H) {
        return;
    }

    const float pref_last = prefix[size_t(T_len - 1) * H + h];
    const float pref_t = prefix[size_t(t) * H + h];
    const float val = fast::exp(pref_last - pref_t);
    decay_last[size_t(h) * T_len + t] = static_cast<T>(val);
}

//------------------------------------------------------------------------------
// Pack B/C into batched GEMM layout
//------------------------------------------------------------------------------

template <typename T>
kernel void ssd_m2_pack_BC_kernel(
    device const T* B_in     [[ buffer(0) ]],
    device const T* C_in     [[ buffer(1) ]],
    device       T* C_packed [[ buffer(2) ]],
    device       T* B_packed [[ buffer(3) ]],
    constant const uint& T_len [[ buffer(4) ]],
    constant const uint& G     [[ buffer(5) ]],
    constant const uint& N     [[ buffer(6) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    const uint n = gid.x;
    const uint t = gid.y;
    const uint g = gid.z;
    if (n >= N || t >= T_len || g >= G) {
        return;
    }

    const size_t base = (size_t(t) * G + g) * N + n;
    C_packed[(size_t(g) * T_len + t) * N + n] = C_in[base];
    B_packed[(size_t(g) * N + n) * T_len + t] = B_in[base];
}

//------------------------------------------------------------------------------
// Build attention from CB groups + decay computed from prefix
//------------------------------------------------------------------------------

template <typename T>
kernel void ssd_m2_attn_elementwise_kernel(
    device const T* CB_groups [[ buffer(0) ]],
    device const float* prefix [[ buffer(1) ]],
    device       T* attn [[ buffer(2) ]],
    constant const uint& T_len [[ buffer(3) ]],
    constant const uint& H     [[ buffer(4) ]],
    constant const uint& G     [[ buffer(5) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    const uint j = gid.x;
    const uint i = gid.y;
    const uint h = gid.z;
    if (i >= T_len || j >= T_len || h >= H) {
        return;
    }

    const uint safe_groups = max(1u, G);
    const uint group_size = max(1u, H / safe_groups);
    const uint g = min(h / group_size, safe_groups - 1u);

    const size_t attn_idx = (size_t(h) * T_len + i) * T_len + j;
    const size_t group_idx = (size_t(g) * T_len + i) * T_len + j;
    if (j > i) {
        attn[attn_idx] = static_cast<T>(0);
        return;
    }

    const float pref_i = prefix[size_t(i) * H + h];
    const float pref_j = prefix[size_t(j) * H + h];
    const float decay = fast::exp(pref_i - pref_j);
    const float cb = float(CB_groups[group_idx]);
    attn[attn_idx] = static_cast<T>(cb * decay);
}

//------------------------------------------------------------------------------
// GEMM helper (8x8 tiles, batched)
//------------------------------------------------------------------------------

template <typename T>
inline float _to_float(T x) {
    return float(x);
}

inline float _to_float(float x) {
    return x;
}

inline float _to_float(bfloat x) {
    return float(x);
}

inline float _to_float(half x) {
    return float(x);
}


template <typename T>
inline T _from_float_generic(float v);

template <>
inline float _from_float_generic<float>(float v) {
    return v;
}

template <>
inline half _from_float_generic<half>(float v) {
    return half(v);
}

template <>
inline bfloat _from_float_generic<bfloat>(float v) {
    return bfloat(v);
}


template <typename T>
kernel void ssd_m2_gemm_batched_kernel(
    device const T* A [[ buffer(0) ]],
    device const T* B [[ buffer(1) ]],
    device       T* C [[ buffer(2) ]],
    constant const uint& M [[ buffer(3) ]],
    constant const uint& N [[ buffer(4) ]],
    constant const uint& K [[ buffer(5) ]],
    constant const uint& lda [[ buffer(6) ]],
    constant const uint& ldb [[ buffer(7) ]],
    constant const uint& ldc [[ buffer(8) ]],
    constant const uint& batch_count [[ buffer(9) ]],
    constant const uint& strideA [[ buffer(10) ]],
    constant const uint& strideB [[ buffer(11) ]],
    constant const uint& strideC [[ buffer(12) ]],
    uint3 tg_pos [[ threadgroup_position_in_grid ]],
    ushort tid [[ thread_index_in_threadgroup ]]
) {
    const uint batch = tg_pos.z;
    if (batch >= batch_count) {
        return;
    }

    const uint tile_n = tg_pos.x;
    const uint tile_m = tg_pos.y;
    const uint m0 = tile_m * 8;
    const uint n0 = tile_n * 8;
    if (m0 >= M || n0 >= N) {
        return;
    }

    device const T* A_base = A + batch * strideA;
    device const T* B_base = B + batch * strideB;
    device       T* C_base = C + batch * strideC;

    threadgroup float As_tile[8][8];
    threadgroup float Bs_tile[8][8];

    simdgroup_matrix<float, 8, 8> acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> a_frag;
    simdgroup_matrix<float, 8, 8> b_frag;

    const uint lane = tid;
    const uint elems = 64;
    const uint per_thread = (elems + 31u) / 32u;

    for (uint k0 = 0; k0 < K; k0 += 8) {
        for (uint e = 0; e < per_thread; ++e) {
            uint idx = lane + e * 32u;
            if (idx < elems) {
                uint r = idx / 8u;
                uint c = idx % 8u;
                uint m = m0 + r;
                uint k = k0 + c;
                As_tile[r][c] = (m < M && k < K)
                    ? _to_float(A_base[m * lda + k])
                    : 0.0f;
                uint kk = k0 + r;
                uint n = n0 + c;
                Bs_tile[r][c] = (kk < K && n < N)
                    ? _to_float(B_base[kk * ldb + n])
                    : 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(a_frag, &As_tile[0][0], 8);
        simdgroup_load(b_frag, &Bs_tile[0][0], 8);
        simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, &As_tile[0][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0; e < per_thread; ++e) {
        uint idx = lane + e * 32u;
        if (idx < elems) {
            uint r = idx / 8u;
            uint c = idx % 8u;
            uint m = m0 + r;
            uint n = n0 + c;
            if (m < M && n < N) {
                float v = As_tile[r][c];
                C_base[m * ldc + n] = _from_float_generic<T>(v);
            }
        }
    }
}

//------------------------------------------------------------------------------
// dtx / dtxdecay / B-head packing
//------------------------------------------------------------------------------

template <typename T>
kernel void ssd_m2_dtx_kernel(
    device const T* x        [[ buffer(0) ]],
    device const T* dt_proc  [[ buffer(1) ]],
    device       T* dtx_out  [[ buffer(2) ]],
    constant const uint& T_len [[ buffer(3) ]],
    constant const uint& H     [[ buffer(4) ]],
    constant const uint& Dh    [[ buffer(5) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    const uint dh = gid.x;
    const uint t = gid.y;
    const uint h = gid.z;
    if (dh >= Dh || t >= T_len || h >= H) {
        return;
    }
    const size_t x_idx = ((size_t(t) * H) + h) * Dh + dh;
    const size_t dt_idx = size_t(t) * H + h;
    const size_t out_idx = ((size_t(h) * T_len) + t) * Dh + dh;
    const float dt_val = float(dt_proc[dt_idx]);
    const float denom = fmax(dt_val, 1e-6f);
    const float scaled = dt_val / denom;
    const float v = float(x[x_idx]) * scaled;
    dtx_out[out_idx] = static_cast<T>(v);
}


template <typename T>
kernel void ssd_m2_dtxdecay_kernel(
    device const T* dtx        [[ buffer(0) ]],
    device const T* decay_last [[ buffer(1) ]],
    device       T* dtxdecay   [[ buffer(2) ]],
    constant const uint& T_len [[ buffer(3) ]],
    constant const uint& H     [[ buffer(4) ]],
    constant const uint& Dh    [[ buffer(5) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    const uint t = gid.x;
    const uint dh = gid.y;
    const uint h = gid.z;
    if (t >= T_len || dh >= Dh || h >= H) {
        return;
    }
    const size_t dtx_idx = ((size_t(h) * T_len) + t) * Dh + dh;
    const size_t decay_idx = size_t(h) * T_len + t;
    const size_t out_idx = ((size_t(h) * Dh) + dh) * T_len + t;
    const float v = float(dtx[dtx_idx]) * float(decay_last[decay_idx]);
    dtxdecay[out_idx] = static_cast<T>(v);
}


template <typename T>
kernel void ssd_m2_pack_b_heads_kernel(
    device const T* B_in   [[ buffer(0) ]],
    device       T* B_head [[ buffer(1) ]],
    constant const uint& T_len [[ buffer(2) ]],
    constant const uint& H     [[ buffer(3) ]],
    constant const uint& G     [[ buffer(4) ]],
    constant const uint& N     [[ buffer(5) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    const uint n = gid.x;
    const uint t = gid.y;
    const uint h = gid.z;
    if (n >= N || t >= T_len || h >= H) {
        return;
    }
    const uint group_size = max(1u, H / max(1u, G));
    const uint g = min(h / group_size, G - 1u);
    const size_t src_idx = (size_t(t) * G + g) * N + n;
    const size_t dst_idx = ((size_t(h) * T_len) + t) * N + n;
    B_head[dst_idx] = B_in[src_idx];
}

template <typename T>
kernel void ssd_m2_pack_c_head_kernel(
    device const T* C_in   [[ buffer(0) ]],
    device       T* C_head [[ buffer(1) ]],
    constant const uint& T_len [[ buffer(2) ]],
    constant const uint& H     [[ buffer(3) ]],
    constant const uint& G     [[ buffer(4) ]],
    constant const uint& N     [[ buffer(5) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    const uint t = gid.x;
    const uint n = gid.y;
    const uint h = gid.z;
    if (t >= T_len || n >= N || h >= H) {
        return;
    }
    const uint group_size = max(1u, H / max(1u, G));
    const uint g = min(h / group_size, G - 1u);
    const size_t src_idx = ((size_t)t * G + g) * N + n;
    const size_t dst_idx = ((size_t)h * N + n) * T_len + t;
    C_head[dst_idx] = C_in[src_idx];
}

//------------------------------------------------------------------------------
// Residual add + gate
//------------------------------------------------------------------------------

template <typename T>
kernel void ssd_m2_residual_y_kernel(
    device const T* x        [[ buffer(0) ]],
    device const T* D        [[ buffer(1) ]],
    device const T* z        [[ buffer(2) ]],
    device const T* y_in     [[ buffer(3) ]],
    device       T* y_out    [[ buffer(4) ]],
    constant const uint& T_len [[ buffer(5) ]],
    constant const uint& H     [[ buffer(6) ]],
    constant const uint& Dh    [[ buffer(7) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    const uint dh = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    if (dh >= Dh || h >= H || t >= T_len) {
        return;
    }

    const size_t vec_idx = ((size_t(h) * T_len) + t) * Dh + dh;
    const size_t x_idx = ((size_t(t) * H) + h) * Dh + dh;
    const size_t out_idx = (size_t(t) * H + h) * Dh + dh;
    float acc = float(y_in[vec_idx]);
    const float skip = float(x[x_idx]) * float(D[h]);
    acc += skip;
    const float gate = float(SILU{}(z[x_idx]));
    y_out[out_idx] = static_cast<T>(acc * gate);
}

kernel void ssd_m2_state_decay_kernel(
    device const float* prefix [[ buffer(0) ]],
    device       float* decay  [[ buffer(1) ]],
    constant const uint& T_len [[ buffer(2) ]],
    constant const uint& H     [[ buffer(3) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    const uint t = gid.x;
    const uint h = gid.y;
    if (t >= T_len || h >= H) {
        return;
    }
    const size_t idx = size_t(t) * H + h;
    decay[idx] = fast::exp(prefix[idx]);
}

template <typename T>
kernel void ssd_m2_scale_c_head_kernel(
    device const T* c_in   [[ buffer(0) ]],
    device const float* decay [[ buffer(1) ]],
    device       T* c_out  [[ buffer(2) ]],
    constant const uint& T_len [[ buffer(3) ]],
    constant const uint& H     [[ buffer(4) ]],
    constant const uint& N     [[ buffer(5) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    const uint n = gid.x;
    const uint t = gid.y;
    const uint h = gid.z;
    if (n >= N || t >= T_len || h >= H) {
        return;
    }
    const size_t idx = ((size_t)h * N + n) * T_len + t;
    const float scale = decay[size_t(t) * H + h];
    c_out[idx] = static_cast<T>(float(c_in[idx]) * scale);
}

template <typename T>
kernel void ssd_m2_accumulate_state_kernel(
    device const T* state_contrib [[ buffer(0) ]],
    device       T* y_tmp         [[ buffer(1) ]],
    constant const uint& T_len [[ buffer(2) ]],
    constant const uint& H     [[ buffer(3) ]],
    constant const uint& Dh    [[ buffer(4) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    const uint dh = gid.x;
    const uint t = gid.y;
    const uint h = gid.z;
    if (dh >= Dh || t >= T_len || h >= H) {
        return;
    }
    const size_t contrib_idx =
        ((size_t)h * Dh + dh) * T_len + t;
    const size_t y_idx = ((size_t)h * T_len + t) * Dh + dh;
    const float acc =
        float(y_tmp[y_idx]) + float(state_contrib[contrib_idx]);
    y_tmp[y_idx] = static_cast<T>(acc);
}

//------------------------------------------------------------------------------
// Instantiations
//------------------------------------------------------------------------------

#define INSTANTIATE_SSD_M2_DT_PREPROCESS(type_name, type) \
  template [[host_name("ssd_m2_dt_preprocess_" #type_name)]] \
  kernel void ssd_m2_dt_preprocess_kernel<type>( \
    device const type*, device float*, \
    constant const uint&, constant const uint&, uint2);

INSTANTIATE_SSD_M2_DT_PREPROCESS(float, float)
INSTANTIATE_SSD_M2_DT_PREPROCESS(bfloat, bfloat)
INSTANTIATE_SSD_M2_DT_PREPROCESS(half, half)
#undef INSTANTIATE_SSD_M2_DT_PREPROCESS

#define INSTANTIATE_SSD_M2_DECAY_LAST(type_name, type) \
  template [[host_name("ssd_m2_decay_last_" #type_name)]] \
  kernel void ssd_m2_decay_last_kernel<type>( \
    device const float*, device type*, \
    constant const uint&, constant const uint&, uint2);

INSTANTIATE_SSD_M2_DECAY_LAST(float, float)
INSTANTIATE_SSD_M2_DECAY_LAST(bfloat, bfloat)
INSTANTIATE_SSD_M2_DECAY_LAST(half, half)
#undef INSTANTIATE_SSD_M2_DECAY_LAST

#define INSTANTIATE_SSD_M2_PACK_BC(type_name, type) \
  template [[host_name("ssd_m2_pack_bc_" #type_name)]] \
  kernel void ssd_m2_pack_BC_kernel<type>( \
    device const type*, device const type*, device type*, device type*, \
    constant const uint&, constant const uint&, constant const uint&, uint3);

INSTANTIATE_SSD_M2_PACK_BC(float, float)
INSTANTIATE_SSD_M2_PACK_BC(bfloat, bfloat)
INSTANTIATE_SSD_M2_PACK_BC(half, half)
#undef INSTANTIATE_SSD_M2_PACK_BC

#define INSTANTIATE_SSD_M2_ATTN(type_name, type) \
  template [[host_name("ssd_m2_attn_" #type_name)]] \
  kernel void ssd_m2_attn_elementwise_kernel<type>( \
    device const type*, device const float*, device type*, \
    constant const uint&, constant const uint&, constant const uint&, uint3);

INSTANTIATE_SSD_M2_ATTN(float, float)
INSTANTIATE_SSD_M2_ATTN(bfloat, bfloat)
INSTANTIATE_SSD_M2_ATTN(half, half)
#undef INSTANTIATE_SSD_M2_ATTN

#define INSTANTIATE_SSD_M2_GEMM(type_name, type) \
  template [[host_name("ssd_m2_gemm_batched_" #type_name)]] \
  kernel void ssd_m2_gemm_batched_kernel<type>( \
    device const type*, device const type*, device type*, \
    constant const uint&, constant const uint&, constant const uint&, \
    constant const uint&, constant const uint&, constant const uint&, \
    constant const uint&, constant const uint&, constant const uint&, \
    constant const uint&, \
    uint3, ushort);

INSTANTIATE_SSD_M2_GEMM(float, float)
INSTANTIATE_SSD_M2_GEMM(bfloat, bfloat)
INSTANTIATE_SSD_M2_GEMM(half, half)
#undef INSTANTIATE_SSD_M2_GEMM

#define INSTANTIATE_SSD_M2_DTX(type_name, type) \
  template [[host_name("ssd_m2_dtx_" #type_name)]] \
  kernel void ssd_m2_dtx_kernel<type>( \
    device const type*, device const type*, device type*, \
    constant const uint&, constant const uint&, constant const uint&, uint3);

INSTANTIATE_SSD_M2_DTX(float, float)
INSTANTIATE_SSD_M2_DTX(bfloat, bfloat)
INSTANTIATE_SSD_M2_DTX(half, half)
#undef INSTANTIATE_SSD_M2_DTX

#define INSTANTIATE_SSD_M2_DTXDECAY(type_name, type) \
  template [[host_name("ssd_m2_dtxdecay_" #type_name)]] \
  kernel void ssd_m2_dtxdecay_kernel<type>( \
    device const type*, device const type*, device type*, \
    constant const uint&, constant const uint&, constant const uint&, uint3);

INSTANTIATE_SSD_M2_DTXDECAY(float, float)
INSTANTIATE_SSD_M2_DTXDECAY(bfloat, bfloat)
INSTANTIATE_SSD_M2_DTXDECAY(half, half)
#undef INSTANTIATE_SSD_M2_DTXDECAY

#define INSTANTIATE_SSD_M2_PACK_B_HEADS(type_name, type) \
  template [[host_name("ssd_m2_pack_b_heads_" #type_name)]] \
  kernel void ssd_m2_pack_b_heads_kernel<type>( \
    device const type*, device type*, \
    constant const uint&, constant const uint&, constant const uint&, constant const uint&, uint3);

INSTANTIATE_SSD_M2_PACK_B_HEADS(float, float)
INSTANTIATE_SSD_M2_PACK_B_HEADS(bfloat, bfloat)
INSTANTIATE_SSD_M2_PACK_B_HEADS(half, half)
#undef INSTANTIATE_SSD_M2_PACK_B_HEADS

#define INSTANTIATE_SSD_M2_PACK_C_HEAD(type_name, type) \
  template [[host_name("ssd_m2_pack_c_head_kernel_" #type_name)]] \
  kernel void ssd_m2_pack_c_head_kernel<type>( \
    device const type*, device type*, \
    constant const uint&, constant const uint&, constant const uint&, constant const uint&, uint3);

INSTANTIATE_SSD_M2_PACK_C_HEAD(float, float)
INSTANTIATE_SSD_M2_PACK_C_HEAD(bfloat, bfloat)
INSTANTIATE_SSD_M2_PACK_C_HEAD(half, half)
#undef INSTANTIATE_SSD_M2_PACK_C_HEAD

#define INSTANTIATE_SSD_M2_SCALE_C(type_name, type) \
  template [[host_name("ssd_m2_scale_c_head_" #type_name)]] \
  kernel void ssd_m2_scale_c_head_kernel<type>( \
    device const type*, device const float*, device type*, \
    constant const uint&, constant const uint&, constant const uint&, uint3);

INSTANTIATE_SSD_M2_SCALE_C(float, float)
INSTANTIATE_SSD_M2_SCALE_C(bfloat, bfloat)
INSTANTIATE_SSD_M2_SCALE_C(half, half)
#undef INSTANTIATE_SSD_M2_SCALE_C

#define INSTANTIATE_SSD_M2_ACCUM_STATE(type_name, type) \
  template [[host_name("ssd_m2_accumulate_state_" #type_name)]] \
  kernel void ssd_m2_accumulate_state_kernel<type>( \
    device const type*, device type*, \
    constant const uint&, constant const uint&, constant const uint&, uint3);

INSTANTIATE_SSD_M2_ACCUM_STATE(float, float)
INSTANTIATE_SSD_M2_ACCUM_STATE(bfloat, bfloat)
INSTANTIATE_SSD_M2_ACCUM_STATE(half, half)
#undef INSTANTIATE_SSD_M2_ACCUM_STATE

#define INSTANTIATE_SSD_M2_RESIDUAL(type_name, type) \
  template [[host_name("ssd_m2_residual_y_" #type_name)]] \
  kernel void ssd_m2_residual_y_kernel<type>( \
    device const type*, device const type*, device const type*, \
    device const type*, device type*, \
    constant const uint&, constant const uint&, constant const uint&, uint3);

INSTANTIATE_SSD_M2_RESIDUAL(float, float)
INSTANTIATE_SSD_M2_RESIDUAL(bfloat, bfloat)
INSTANTIATE_SSD_M2_RESIDUAL(half, half)
#undef INSTANTIATE_SSD_M2_RESIDUAL
