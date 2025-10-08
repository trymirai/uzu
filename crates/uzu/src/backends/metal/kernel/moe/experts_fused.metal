#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "../definitions.metal"
#include "../quant_matmul/mma.h"
using namespace metal;

#define BM 16
#define BN 64
#define BK 32

constant uint GATING_SEL [[function_constant(30)]];
constant uint N_GROUP [[function_constant(31)]];

static inline float gelu_approx(float x) {
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    if (x > 10.0f) return x;
    if (x < -10.0f) return 0.0f;
    float x3 = x * x * x;
    float inner = x + k1 * x3;
    float tanh_arg = clamp(k0 * inner, -10.0f, 10.0f);
    return 0.5f * x * (1.0f + tanh(tanh_arg));
}

static inline float silu(float x, float alpha) {
    return x / (1.0f + exp(-alpha * x));
}

template<typename T>
void moe_fused_expert_mlp_impl(
    device const T* X_perm,
    device const uint* expert_offsets,
    device const T* W13_all,
    device const T* W2_all,
    device T* Y_partial,
    uint T_val,
    uint d_model,
    uint d_ff,
    uint E,
    uint gating_code_val,
    device const T* up_biases,
    device const T* down_biases,
    float gate_clip_min,
    float gate_clip_max,
    float up_clip_min,
    float up_clip_max,
    float silu_alpha,
    device const uint* tile_row_offsets,
    device const uint* tile_map,
    device uint* total_meta_buf,
    uint y_base,
    uint lid,
    uint3 tgpig,
    ushort simd_gid,
    ushort simd_lid,
    threadgroup T* Xs,
    threadgroup T* Wk_up,
    threadgroup T* Wk_gate,
    threadgroup float* Htile,
    threadgroup float* W2sf,
    threadgroup float* Hs)
{
    (void)gating_code_val;
    (void)tile_row_offsets;
    (void)T_val;

    const uint total_tiles = total_meta_buf[0];
    if (total_tiles == 0u) return;

    const uint tile_n_group_start = tgpig.x * N_GROUP;
    const uint tile_y = y_base + tgpig.y;
    const bool tile_y_valid = (tile_y < total_tiles);

    uint expert_idx = 0u;
    uint seg_start = 0u;
    uint tile_m0 = 0u;
    if (tile_y_valid) {
        const uint base = tile_y * 3u;
        expert_idx = tile_map[base + 0u];
        seg_start = tile_map[base + 1u];
        tile_m0 = tile_map[base + 2u];
    }
    bool expert_valid = tile_y_valid && (expert_idx < E);
    uint seg_len = 0u;
    if (expert_valid) {
        const uint seg_end = expert_offsets[expert_idx + 1u];
        seg_len = (seg_end > seg_start) ? (seg_end - seg_start) : 0u;
        }
    const bool rows_ok = expert_valid && (seg_len > tile_m0);
    const uint m_rows = rows_ok ? min((uint)BM, seg_len - tile_m0) : 0u;
    const bool rows_active = rows_ok;

        const ulong W13_base = (ulong)expert_idx * (ulong)d_model * (ulong)(d_ff * 2);
        const ulong W2_base = (ulong)expert_idx * (ulong)d_ff * (ulong)d_model;
        const ulong x_block_base = (ulong)(seg_start + tile_m0) * (ulong)d_model;

    // ===== FC1: Compute once, reuse for N_GROUP tiles =====
    // For simplicity with N_GROUP=8, unroll manually (Metal doesn't support VLA with function constants)
    using mma_fc2_t = matmul_utils::BlockMMA<float, T, BM, BN, BK, 2, 2, false, false, BK, BN, float>;
    mma_fc2_t mma_fc2_0(simd_gid, simd_lid);
    mma_fc2_t mma_fc2_1(simd_gid, simd_lid);
    mma_fc2_t mma_fc2_2(simd_gid, simd_lid);
    mma_fc2_t mma_fc2_3(simd_gid, simd_lid);
    mma_fc2_t mma_fc2_4(simd_gid, simd_lid);
    mma_fc2_t mma_fc2_5(simd_gid, simd_lid);
    mma_fc2_t mma_fc2_6(simd_gid, simd_lid);
    mma_fc2_t mma_fc2_7(simd_gid, simd_lid);
    mma_fc2_0.Ctile.clear();
    if (N_GROUP > 1) mma_fc2_1.Ctile.clear();
    if (N_GROUP > 2) mma_fc2_2.Ctile.clear();
    if (N_GROUP > 3) mma_fc2_3.Ctile.clear();
    if (N_GROUP > 4) mma_fc2_4.Ctile.clear();
    if (N_GROUP > 5) mma_fc2_5.Ctile.clear();
    if (N_GROUP > 6) mma_fc2_6.Ctile.clear();
    if (N_GROUP > 7) mma_fc2_7.Ctile.clear();

        for (uint ff0 = 0; ff0 < d_ff; ff0 += BK) {
            const uint ff_chunk = min((uint)BK, d_ff - ff0);
        matmul_utils::BlockMMA<T, float, BM, BK, BK, 2, 2, false, false, BK, BK, float> mma_fc1_up(simd_gid, simd_lid);
        matmul_utils::BlockMMA<T, float, BM, BK, BK, 2, 2, false, false, BK, BK, float> mma_fc1_gate(simd_gid, simd_lid);
            mma_fc1_up.Ctile.clear();
            if (GATING_SEL > 1u) {
                mma_fc1_gate.Ctile.clear();
            }

            for (uint k0 = 0; k0 < d_model; k0 += BK) {
                const uint k_chunk = min((uint)BK, d_model - k0);
                for (uint idx = lid; idx < (uint)(BM * BK); idx += 128u) {
                    const uint mi = idx / BK;
                    const uint kk = idx % BK;
                T val = T(0.0f);
                if (mi < m_rows && kk < k_chunk && rows_active) {
                        const ulong row_offset = x_block_base
                            + (ulong)mi * (ulong)d_model
                            + (ulong)(k0 + kk);
                    val = X_perm[row_offset];
                    }
                    Xs[mi * BK + kk] = val;
                }
                for (uint idx = lid; idx < (uint)(BK * BK); idx += 128u) {
                    const uint kk = idx / BK;
                    const uint r = idx % BK;
                T up_val = T(0.0f);
                T gate_val = T(0.0f);
                if (kk < k_chunk && r < ff_chunk && rows_active) {
                        // Transposed layout: [E, 2*d_ff, d_model] - d_model is contiguous
                        const ulong fused_idx = W13_base
                            + (ulong)(ff0 + r) * (ulong)d_model
                            + (ulong)(k0 + kk);
                    up_val = W13_all[fused_idx];
                            if (GATING_SEL > 1u) {
                        const ulong gate_idx = W13_base
                            + (ulong)(d_ff + ff0 + r) * (ulong)d_model
                            + (ulong)(k0 + kk);
                        gate_val = W13_all[gate_idx];
                    }
                    }
                    Wk_up[kk * BK + r] = up_val;
                    Wk_gate[kk * BK + r] = gate_val;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                mma_fc1_up.mma(Xs, Wk_up);
                if (GATING_SEL > 1u) {
                    mma_fc1_gate.mma(Xs, Wk_gate);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            mma_fc1_up.store_result_tg(Htile, (int)BK);
            if (GATING_SEL > 1u) {
                mma_fc1_gate.store_result_tg(Hs, (int)BK);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint idx = lid; idx < (uint)(BM * BK); idx += 128u) {
                const uint mi = idx / BK;
                const uint r = idx % BK;
            if (mi < m_rows && r < ff_chunk && rows_active) {
                    const ulong bias_idx = (ulong)expert_idx * (ulong)(d_ff * 2) + (ulong)(ff0 + r);
                float bias_val = float(up_biases[bias_idx]);
                    float val = Htile[idx] + bias_val;
                val = isfinite(val) ? val : 0.0f;
                    Htile[idx] = clamp(val, up_clip_min, up_clip_max);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (GATING_SEL > 1u) {
                for (uint idx = lid; idx < (uint)(BM * BK); idx += 128u) {
                    const uint mi = idx / BK;
                    const uint r = idx % BK;
                if (mi < m_rows && r < ff_chunk && rows_active) {
                        const ulong bias_idx = (ulong)expert_idx * (ulong)(d_ff * 2)
                            + (ulong)d_ff + (ulong)(ff0 + r);
                    float bias_val = float(up_biases[bias_idx]);
                        float val = Hs[idx] + bias_val;
                    val = isfinite(val) ? val : 0.0f;
                        Hs[idx] = clamp(val, gate_clip_min, gate_clip_max);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            for (uint idx = lid; idx < (uint)(BM * BK); idx += 128u) {
                const uint mi = idx / BK;
                const uint r = idx % BK;
                float result = 0.0f;
            if (mi < m_rows && r < ff_chunk && rows_active) {
                    const float up = Htile[idx];
                    if (GATING_SEL <= 1u) {
                        result = (GATING_SEL == 0u) ? gelu_approx(up) : silu(up, silu_alpha);
                    } else {
                        const float gate = Hs[idx];
                        const float swish_y = (GATING_SEL == 2u)
                            ? silu(gate, silu_alpha)
                            : gelu_approx(gate);
                        result = swish_y * up;
                    }
                result = isfinite(result) ? result : 0.0f;
                }
                Hs[idx] = result;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

        // ===== FC2: Loop over N_GROUP tiles, reusing activated Hs =====
        // Manual unroll for N_GROUP (Metal doesn't support indexing with loop var in this context)
        #define DO_FC2_MMA_FOR_G(g_val, mma_obj) \
        if (g_val < N_GROUP) { \
            const uint tile_n_idx = tile_n_group_start + g_val; \
            const uint tile_n0 = tile_n_idx * BN; \
            if (tile_n0 < d_model) { \
                const uint n_cols = min((uint)BN, d_model - tile_n0); \
                const bool is_active = rows_active && (n_cols > 0u); \
                for (uint idx = lid; idx < (uint)(BK * BN); idx += 128u) { \
                    const uint r = idx / BN; \
                    const uint nj = idx % BN; \
                    float v = 0.0f; \
                    if (r < ff_chunk && nj < n_cols && is_active) { \
                        const ulong w2g = W2_base + (ulong)(ff0 + r) * (ulong)d_model + (ulong)(tile_n0 + nj); \
                        v = float(W2_all[w2g]); \
                        v = isfinite(v) ? v : 0.0f; \
                    } \
                    W2sf[r * BN + nj] = v; \
                } \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
                mma_obj.mma(Hs, W2sf); \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
            } \
        }

        DO_FC2_MMA_FOR_G(0, mma_fc2_0)
        DO_FC2_MMA_FOR_G(1, mma_fc2_1)
        DO_FC2_MMA_FOR_G(2, mma_fc2_2)
        DO_FC2_MMA_FOR_G(3, mma_fc2_3)
        DO_FC2_MMA_FOR_G(4, mma_fc2_4)
        DO_FC2_MMA_FOR_G(5, mma_fc2_5)
        DO_FC2_MMA_FOR_G(6, mma_fc2_6)
        DO_FC2_MMA_FOR_G(7, mma_fc2_7)
        #undef DO_FC2_MMA_FOR_G
    }

    // ===== Store FC2 results for each N-tile in the group =====
    #define STORE_FC2_FOR_G(g_val, mma_obj) \
    if (g_val < N_GROUP) { \
        const uint tile_n_idx = tile_n_group_start + g_val; \
        const uint tile_n0 = tile_n_idx * BN; \
        if (tile_n0 < d_model) { \
            const uint n_cols = min((uint)BN, d_model - tile_n0); \
            const bool is_active = rows_active && (n_cols > 0u); \
            const ulong y_offset = (ulong)(seg_start + tile_m0) * (ulong)d_model + (ulong)tile_n0; \
            device T* y_ptr = Y_partial + y_offset; \
            const short num_els = (short)m_rows; \
            const short num_outs = (short)n_cols; \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            if (num_els < (short)BM || num_outs < (short)BN) { \
                mma_obj.store_result_safe(y_ptr, (int)d_model, short2(num_outs, num_els)); \
            } else { \
                mma_obj.store_result(y_ptr, (int)d_model); \
            } \
            if (is_active) { \
                for (uint mi = 0; mi < m_rows; ++mi) { \
                    for (uint nj = lid; nj < n_cols; nj += 128u) { \
                        const ulong bias_idx = (ulong)expert_idx * (ulong)d_model + (ulong)(tile_n0 + nj); \
                        const ulong y_idx = y_offset + (ulong)mi * (ulong)d_model + (ulong)nj; \
                        float prev_val = float(Y_partial[y_idx]); \
                        float bias_val = float(down_biases[bias_idx]); \
                        float out_val = prev_val + bias_val; \
                        out_val = isfinite(out_val) ? out_val : 0.0f; \
                        Y_partial[y_idx] = T(out_val); \
                    } \
                } \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            if (is_active) { \
                for (uint mi = 0; mi < m_rows; ++mi) { \
                    for (uint nj = lid; nj < n_cols; nj += 128u) { \
                        const ulong y_idx = y_offset + (ulong)mi * (ulong)d_model + (ulong)nj; \
                        float val = float(Y_partial[y_idx]); \
                        if (!isfinite(val)) { Y_partial[y_idx] = T(0.0f); } \
                    } \
                } \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
    }

    STORE_FC2_FOR_G(0, mma_fc2_0)
    STORE_FC2_FOR_G(1, mma_fc2_1)
    STORE_FC2_FOR_G(2, mma_fc2_2)
    STORE_FC2_FOR_G(3, mma_fc2_3)
    STORE_FC2_FOR_G(4, mma_fc2_4)
    STORE_FC2_FOR_G(5, mma_fc2_5)
    STORE_FC2_FOR_G(6, mma_fc2_6)
    STORE_FC2_FOR_G(7, mma_fc2_7)
    #undef STORE_FC2_FOR_G
}

#define outerArguments(T) \
(device const T* X_perm [[buffer(0)]], \
 device const uint* expert_offsets [[buffer(1)]], \
 device const T* W13_all [[buffer(2)]], \
 device const T* W2_all [[buffer(3)]], \
 device T* Y_partial [[buffer(4)]], \
 constant uint& T_param [[buffer(5)]], \
 constant uint& d_model [[buffer(6)]], \
 constant uint& d_ff [[buffer(7)]], \
 constant uint& E [[buffer(8)]], \
 constant uint& gating_code [[buffer(9)]], \
 device const T* up_biases [[buffer(10)]], \
 device const T* down_biases [[buffer(11)]], \
 constant float& gate_clip_min [[buffer(12)]], \
 constant float& gate_clip_max [[buffer(13)]], \
 constant float& up_clip_min [[buffer(14)]], \
 constant float& up_clip_max [[buffer(15)]], \
 constant float& silu_alpha [[buffer(16)]], \
 device const uint* tile_row_offsets [[buffer(17)]], \
 device const uint* tile_map [[buffer(18)]], \
 device uint* total_meta_buf [[buffer(19)]], \
 constant uint& y_base [[buffer(20)]], \
 uint lid [[thread_index_in_threadgroup]], \
 uint3 tgpig [[threadgroup_position_in_grid]], \
 ushort simd_gid [[simdgroup_index_in_threadgroup]], \
 ushort simd_lid [[thread_index_in_simdgroup]])

#define innerArguments \
(X_perm, expert_offsets, W13_all, W2_all, Y_partial, T_param, d_model, d_ff, E, gating_code, \
 up_biases, down_biases, gate_clip_min, gate_clip_max, up_clip_min, up_clip_max, silu_alpha, \
 tile_row_offsets, tile_map, total_meta_buf, y_base, lid, tgpig, simd_gid, simd_lid, \
 Xs, Wk_up, Wk_gate, Htile, W2sf, Hs)

kernel void moe_fused_expert_mlp_f16 outerArguments(half) {
    threadgroup half Xs[BM * BK];
    threadgroup half Wk_up[BK * BK];
    threadgroup half Wk_gate[BK * BK];
    threadgroup float Htile[BM * BK];
    threadgroup float W2sf[BK * BN];
    threadgroup float Hs[BM * BK];
    moe_fused_expert_mlp_impl<half> innerArguments;
}

kernel void moe_fused_expert_mlp_bf16 outerArguments(bfloat) {
    threadgroup bfloat Xs[BM * BK];
    threadgroup bfloat Wk_up[BK * BK];
    threadgroup bfloat Wk_gate[BK * BK];
    threadgroup float Htile[BM * BK];
    threadgroup float W2sf[BK * BN];
    threadgroup float Hs[BM * BK];
    moe_fused_expert_mlp_impl<bfloat> innerArguments;
}

kernel void moe_fused_expert_mlp_f32 outerArguments(float) {
    threadgroup float Xs[BM * BK];
    threadgroup float Wk_up[BK * BK];
    threadgroup float Wk_gate[BK * BK];
    threadgroup float Htile[BM * BK];
    threadgroup float W2sf[BK * BN];
    threadgroup float Hs[BM * BK];
    moe_fused_expert_mlp_impl<float> innerArguments;
}
