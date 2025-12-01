// UZU MoE Two-Pass Prefill
// - Pass A: X @ W13 → hidden (f32) with gating
// - Pass B: hidden @ W2 → output (T)
//
//   X_perm          : [total_rows, D]
//   W13_all         : [E, 2*FF, D]   // first FF = up, second FF = gate
//   up_biases       : [E, 2*FF]      // [up | gate]
//   hidden_out (f32): [total_rows, FF]
//   W2_all          : [E, D, FF]     // FF contiguous (col-major-ish use below)
//   down_biases     : [E, D]
//   output (T)      : [total_rows, D]
//
// Notes:
// - 8x8 simdgroup MMA tiles; threadgroup staging is float.
// - Vectorized float4 cooperative loads.
// - Linearized thread id covers both 1-D and 2/3-D TGs safely.
// - We guard tails; K/N can be non-multiples.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "../quant_matmul/mma.h" 

using namespace metal;

// ------------------------ helpers ------------------------
static inline uint ceil_div(uint a, uint b) { return (a + b - 1u) / b; }

static inline uint linear_tid(uint3 tid, uint3 tpg) {
    return tid.z * (tpg.x * tpg.y) + tid.y * tpg.x + tid.x;
}

static inline float gelu_approx(float x) {
    const float k0 = 0.7978845608f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    if (x > 10.0f)  return x;
    if (x < -10.0f) return 0.0f;
    float x3 = x * x * x;
    float t  = clamp(k0 * (x + k1 * x3), -10.0f, 10.0f);
    return 0.5f * x * (1.0f + tanh(t));
}

static inline float silu(float x, float alpha) {
    // alpha=1 gives standard SiLU; keep alpha param for parity with your API
    return x / (1.0f + exp(-alpha * x));
}

template<typename T>
inline void store_vec4(device T* dst, ulong base, float4 vals) {
    dst[base + 0] = T(vals.x);
    dst[base + 1] = T(vals.y);
    dst[base + 2] = T(vals.z);
    dst[base + 3] = T(vals.w);
}

template<>
inline void store_vec4<float>(device float* dst, ulong base, float4 vals) {
    *reinterpret_cast<device float4*>(dst + base) = vals;
}

template<typename T>
struct StageStorage {
    using type = float;
};

template<>
struct StageStorage<half> {
    using type = half;
};

template<>
struct StageStorage<bfloat> {
    using type = bfloat;
};

// 0=GELU(up), 1=SiLU(up), 2=SwiGLU(gate)*up, 3=GEGLU(gate)*up
constant uint GATING_SEL [[function_constant(30)]];

// ------------------------ Pass A (X @ W13 → hidden) ------------------------
// Tile config (tuned for A-series; matches your prior defaults)
constant uint PASSA_BM = 16;
constant uint PASSA_BN = 32;
constant uint PASSA_BK = 64;
constant uint PASSA_SG_BM = 8;
constant uint PASSA_SG_BN = 16;
constant uint PASSA_TG_PAD = 4;  // Padding to avoid bank conflicts

template<typename T>
inline void pass_a_impl(
    device const T*    X_perm,          // [total_rows, D]
    device const uint* expert_offsets,  // [E+1]
    device const T*    W13_all,         // [E, 2*FF, D]
    device const T*    up_biases,       // [E, 2*FF]  (up | gate)
    device float*      hidden_out,      // [total_rows, FF]
    uint d_model, uint d_ff, uint E,
    float gate_clip_min, float gate_clip_max,
    float up_clip_min,   float up_clip_max,
    float silu_alpha,
    // work partition
    uint tile_seg_start, uint expert_idx, uint tile_m, uint tile_n,
    // per-thread info
    uint sg_id, uint3 threads_per_tg, uint3 local_tid,
    // TG storage
    threadgroup typename StageStorage<T>::type* Xs,       // [BM,BK]
    threadgroup typename StageStorage<T>::type* Wk_up,    // [BN,BK]
    threadgroup typename StageStorage<T>::type* Wk_gate,  // [BN,BK]
    threadgroup float* bias_up,       // [BN]
    threadgroup float* bias_gate      // [BN]
) {
    using StageT = typename StageStorage<T>::type;
    using StageVec4 = metal::vec<StageT, 4>;

    constexpr uint Bm = PASSA_BM, Bn = PASSA_BN, Bk = PASSA_BK;
    constexpr uint SgBm = PASSA_SG_BM, SgBn = PASSA_SG_BN;
    constexpr uint TG_PAD = PASSA_TG_PAD;
    constexpr uint X_LD = Bk + TG_PAD;   // Padded stride for X
    constexpr uint W_LD = Bk + TG_PAD;   // Padded stride for weights

    if (expert_idx >= E) return;

    const uint seg_start = expert_offsets[expert_idx];
    const uint seg_end   = expert_offsets[expert_idx + 1];
    const uint seg_len   = (seg_end > seg_start) ? (seg_end - seg_start) : 0u;
    if (seg_len == 0) return;

    const uint row_tg_off = tile_m * Bm;
    const uint col_tg_off = tile_n * Bn;
    if (row_tg_off >= seg_len || col_tg_off >= d_ff) return;

    const uint m_rows = min(Bm, seg_len - row_tg_off);
    const uint n_cols = min(Bn, d_ff   - col_tg_off);

    const ulong x_base          = ((ulong)tile_seg_start + (ulong)row_tg_off) * (ulong)d_model;
    const ulong w13_expert_base = (ulong)expert_idx * (ulong)(2 * d_ff) * (ulong)d_model;
    const ulong bias_base       = (ulong)expert_idx * (ulong)(2 * d_ff);

    // simdgroup tile mapping
    const uint sg_col_count = Bn / SgBn;
    const uint row_sg = sg_id / sg_col_count;
    const uint col_sg = sg_id % sg_col_count;
    const uint row_sg_off = row_sg * SgBm;
    const uint col_sg_off = col_sg * SgBn;

    // Guard against misconfigured TG sizes
    constexpr uint SG_TILE = 8;  // simdgroup fragment dimension (8x8)
    constexpr uint SG_EXPECTED = (Bm / SG_TILE) * (Bn / SG_TILE);
    if (sg_id >= SG_EXPECTED) return;

    // partial accumulators (8x8 tiles per simdgroup)
    constexpr uint TEMP = (SgBm / SG_TILE) * (SgBn / SG_TILE);
    metal::simdgroup_float8x8 OutUp[TEMP];
    metal::simdgroup_float8x8 OutGate[TEMP];
    for (uint i = 0; i < TEMP; ++i) {
        OutUp[i] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        if (GATING_SEL > 1u) {
            OutGate[i] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint threads_total = threads_per_tg.x * threads_per_tg.y * threads_per_tg.z;
    const uint lin = linear_tid(local_tid, threads_per_tg);

    // K-loop (tile along D)
    for (uint k_off = 0; k_off < d_model; k_off += Bk) {
        const uint valid_k = min(Bk, d_model - k_off);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Stage LHS: Xs [Bm,Bk] with vec8 loads and padding ----
        {
            constexpr uint vec_size = 8;
            constexpr uint vec_cols = Bk / vec_size;  // 64/8 = 8
            constexpr uint total_vecs = Bm * vec_cols;  // 16*8 = 128
            const uint iters = ceil_div(total_vecs, threads_total);

UZU_PRAGMA_UNROLL
            for (uint t = 0; t < iters; ++t) {
                const uint i = t * threads_total + lin;
                if (i < total_vecs) {
                    const uint r  = i / vec_cols;
                    const uint c8 = i % vec_cols;
                    const uint col_base = c8 * vec_size;

                    threadgroup StageT* dst = Xs + r * X_LD + col_base;

                    if (r < m_rows && col_base < valid_k) {
                        // Safe to read - at least first element is valid
                        const ulong base = x_base + (ulong)r * (ulong)d_model + (ulong)(k_off + col_base);
UZU_PRAGMA_UNROLL
                        for (uint j = 0; j < vec_size; j++) {
                            dst[j] = (col_base + j < valid_k) ? StageT(float(X_perm[base + j])) : StageT(0.0f);
                        }
                    } else {
                        // Row out of bounds or entire vec8 chunk is out of K bounds
UZU_PRAGMA_UNROLL
                        for (uint j = 0; j < vec_size; j++) {
                            dst[j] = StageT(0.0f);
                        }
                    }
                }
            }
        }

        // ---- Stage RHS: W_up [Bn,Bk], W_gate [Bn,Bk] with vec8 loads and padding ----
        {
            constexpr uint vec_size = 8;
            constexpr uint vec_cols = Bk / vec_size;  // 64/8 = 8
            constexpr uint total_vecs = Bn * vec_cols;  // 32*8 = 256
            const uint iters = ceil_div(total_vecs, threads_total);

UZU_PRAGMA_UNROLL
            for (uint t = 0; t < iters; ++t) {
                const uint i = t * threads_total + lin;
                if (i < total_vecs) {
                    const uint n   = i / vec_cols;  // row within the BN tile
                    const uint c8  = i % vec_cols;
                    const uint col_base = c8 * vec_size;

                    threadgroup StageT* up_dst = Wk_up + n * W_LD + col_base;
                    threadgroup StageT* gt_dst = Wk_gate + n * W_LD + col_base;

                    if (n < n_cols && col_base < valid_k) {
                        // Safe to read - at least first element is valid
                        const uint gcol = col_tg_off + n;
                        const ulong up_base   = w13_expert_base + (ulong)gcol * (ulong)d_model + (ulong)(k_off + col_base);
                        const ulong gate_base = w13_expert_base + (ulong)(d_ff + gcol) * (ulong)d_model + (ulong)(k_off + col_base);
UZU_PRAGMA_UNROLL
                        for (uint j = 0; j < vec_size; j++) {
                            up_dst[j] = (col_base + j < valid_k) ? StageT(float(W13_all[up_base + j])) : StageT(0.0f);
                            if (GATING_SEL > 1u) {
                                gt_dst[j] = (col_base + j < valid_k) ? StageT(float(W13_all[gate_base + j])) : StageT(0.0f);
                            }
                        }
                    } else {
                        // Row out of bounds or entire vec8 chunk is out of K bounds
UZU_PRAGMA_UNROLL
                        for (uint j = 0; j < vec_size; j++) {
                            up_dst[j] = StageT(0.0f);
                            if (GATING_SEL > 1u) gt_dst[j] = StageT(0.0f);
                        }
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- MMA across the BK tile (in simdgroup-sized chunks) ----
UZU_PRAGMA_UNROLL
        for (uint kk = 0; kk < Bk; kk += SG_TILE) {
UZU_PRAGMA_UNROLL
            for (uint m_sub = 0; m_sub < SgBm; m_sub += SG_TILE) {
                const uint r_idx = m_sub / SG_TILE;
                metal::simdgroup_matrix<StageT, 8, 8> lhs_frag;
                simdgroup_load(lhs_frag, Xs, X_LD, ulong2(kk, row_sg_off + m_sub));

UZU_PRAGMA_UNROLL
                for (uint n_sub = 0; n_sub < SgBn; n_sub += SG_TILE) {
                    const uint c_idx = n_sub / SG_TILE;
                    const uint tile  = r_idx * (SgBn / SG_TILE) + c_idx;
                    metal::simdgroup_matrix<StageT, 8, 8> rhs_up;
                    simdgroup_load(rhs_up, Wk_up, W_LD, ulong2(kk, col_sg_off + n_sub), true);
                    simdgroup_multiply_accumulate(OutUp[tile], lhs_frag, rhs_up, OutUp[tile]);

                    if (GATING_SEL > 1u) {
                        metal::simdgroup_matrix<StageT, 8, 8> rhs_gate;
                        simdgroup_load(rhs_gate, Wk_gate, W_LD, ulong2(kk, col_sg_off + n_sub), true);
                        simdgroup_multiply_accumulate(OutGate[tile], lhs_frag, rhs_gate, OutGate[tile]);
                    }
                }
            }
        }
    } // K loop

    // ---- Bias tile [Bn] ----
    // Note: bias layout is [up | gate] in contiguous FF chunks.
    for (uint c_local = lin; c_local < Bn; c_local += threads_total) {
        const uint c_global = col_tg_off + c_local;
        bias_up[c_local] = (c_global < d_ff) ? float(up_biases[bias_base + c_global]) : 0.0f;
        if (GATING_SEL > 1u) {
            bias_gate[c_local] = (c_global < d_ff) ? float(up_biases[bias_base + d_ff + c_global]) : 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Epilogue: activation/gating → hidden_out (f32) ----
    const uint sg_lane_start = sg_id * 32;
    const uint sg_lane_end = sg_lane_start + 32;
    if (lin < sg_lane_start || lin >= sg_lane_end) return;
    const uint sg_lin = lin - sg_lane_start;
    // Lane→(row,col) mapping for 8x8 simdgroup fragments, matches the fragment layout used above.
    const uint lane_qid = sg_lin >> 2;
    const uint lane_row = (lane_qid & 4u) + ((sg_lin >> 1) & 3u);
    const uint lane_col_base = ((lane_qid & 2u) << 1) + ((sg_lin & 1u) << 1);

UZU_PRAGMA_UNROLL
    for (uint n_sub = 0; n_sub < SgBn; n_sub += SG_TILE) {
        const uint c_idx = n_sub / SG_TILE;
UZU_PRAGMA_UNROLL
        for (uint m_sub = 0; m_sub < SgBm; m_sub += SG_TILE) {
            const uint r_idx = m_sub / SG_TILE;
            const uint tile = r_idx * (SgBn / SG_TILE) + c_idx;

            const uint tile_row_base = row_sg_off + m_sub;
            const uint tile_col_base = col_sg_off + n_sub;
            const uint local_row = tile_row_base + lane_row;
            if (local_row >= m_rows || tile_col_base >= n_cols) {
                continue;
            }

            const auto up_frag = OutUp[tile].thread_elements();
            float gate_frag_0 = 0.0f;
            float gate_frag_1 = 0.0f;
            if (GATING_SEL > 1u) {
                const auto gate_frag = OutGate[tile].thread_elements();
                gate_frag_0 = gate_frag[0];
                gate_frag_1 = gate_frag[1];
            }

            const uint col0 = tile_col_base + lane_col_base;
            const uint col1 = col0 + 1;
            const ulong out_row = seg_start + row_tg_off + local_row;

            if (col0 < n_cols) {
                float up_v = clamp(up_frag[0] + bias_up[col0], up_clip_min, up_clip_max);
                float out_val;
                if (GATING_SEL <= 1u) {
                    out_val =
                        (GATING_SEL == 0u) ? gelu_approx(up_v) : silu(up_v, silu_alpha);
                } else {
                    float gate_v =
                        clamp(gate_frag_0 + bias_gate[col0], gate_clip_min, gate_clip_max);
                    const float gate_act =
                        (GATING_SEL == 2u) ? silu(gate_v, silu_alpha) : gelu_approx(gate_v);
                    out_val = gate_act * up_v;
                }
                const ulong out_col = col_tg_off + col0;
                hidden_out[out_row * (ulong)d_ff + out_col] = out_val;
            }

            if (col1 < n_cols) {
                float up_v = clamp(up_frag[1] + bias_up[col1], up_clip_min, up_clip_max);
                float out_val;
                if (GATING_SEL <= 1u) {
                    out_val =
                        (GATING_SEL == 0u) ? gelu_approx(up_v) : silu(up_v, silu_alpha);
                } else {
                    float gate_v =
                        clamp(gate_frag_1 + bias_gate[col1], gate_clip_min, gate_clip_max);
                    const float gate_act =
                        (GATING_SEL == 2u) ? silu(gate_v, silu_alpha) : gelu_approx(gate_v);
                    out_val = gate_act * up_v;
                }
                const ulong out_col = col_tg_off + col1;
                hidden_out[out_row * (ulong)d_ff + out_col] = out_val;
            }
        }
    }
}

// ----- Kernel entry points (Pass A) -----
#define MOE_PASS_A_KERNEL(DTYPE, SUFFIX) \
kernel void moe_two_pass_prefill_pass_a_##SUFFIX( \
    device const DTYPE* X_perm              [[buffer(0)]], \
    device const uint*  expert_offsets      [[buffer(1)]], \
    device const DTYPE* W13_all             [[buffer(2)]], \
    device const DTYPE* up_biases           [[buffer(3)]], \
    device float*       hidden_out          [[buffer(4)]], \
    constant uint&      d_model             [[buffer(5)]], \
    constant uint&      d_ff                [[buffer(6)]], \
    constant uint&      E                   [[buffer(7)]], \
    constant float&     gate_clip_min       [[buffer(8)]], \
    constant float&     gate_clip_max       [[buffer(9)]], \
    constant float&     up_clip_min         [[buffer(10)]], \
    constant float&     up_clip_max         [[buffer(11)]], \
    constant float&     silu_alpha          [[buffer(12)]], \
    uint                sg_id               [[simdgroup_index_in_threadgroup]], \
    uint3               threads_per_tg      [[threads_per_threadgroup]], \
    uint3               tg_pos              [[threadgroup_position_in_grid]], \
    uint3               local_tid           [[thread_position_in_threadgroup]] ) \
{ \
    using StageT = typename StageStorage<DTYPE>::type; \
    threadgroup StageT Xs         [PASSA_BM * (PASSA_BK + PASSA_TG_PAD)]; \
    threadgroup StageT Wk_up      [PASSA_BN * (PASSA_BK + PASSA_TG_PAD)]; \
    threadgroup StageT Wk_gate    [PASSA_BN * (PASSA_BK + PASSA_TG_PAD)]; \
    threadgroup float bias_up    [PASSA_BN]; \
    threadgroup float bias_gate  [PASSA_BN]; \
    pass_a_impl<DTYPE>( \
        X_perm, expert_offsets, W13_all, up_biases, hidden_out, \
        d_model, d_ff, E, \
        gate_clip_min, gate_clip_max, up_clip_min, up_clip_max, silu_alpha, \
        /*tile_seg_start*/ expert_offsets[tg_pos.z], /*expert*/ tg_pos.z, /*tile_m*/ tg_pos.y, /*tile_n*/ tg_pos.x, \
        sg_id, threads_per_tg, local_tid, \
        Xs, Wk_up, Wk_gate, bias_up, bias_gate); \
}

MOE_PASS_A_KERNEL(bfloat, bf16)
MOE_PASS_A_KERNEL(half,   f16)
MOE_PASS_A_KERNEL(float,  f32)

// Indirect variant (consumes [expert, seg_start, tile_row_offset] triples)
template<typename T>
inline void pass_a_indirect_impl(
    device const T*    X_perm,
    device const uint* expert_offsets,
    device const T*    W13_all,
    device const T*    up_biases,
    device float*      hidden_out,
    device const uint* tile_map,           // [tiles * 3]: (expert, seg_start, row_offset_elems)
    uint d_model, uint d_ff, uint E,
    float gate_clip_min, float gate_clip_max,
    float up_clip_min, float up_clip_max,
    float silu_alpha,
    uint row_tile_idx, uint n_tile_idx,
    uint sg_id, uint3 threads_per_tg, uint3 local_tid,
    threadgroup typename StageStorage<T>::type* Xs,
    threadgroup typename StageStorage<T>::type* Wk_up,
    threadgroup typename StageStorage<T>::type* Wk_gate,
    threadgroup float* bias_up,
    threadgroup float* bias_gate)
{
    constexpr uint ROW_TILE = PASSA_BM;
    const uint base = row_tile_idx * 3u;
    const uint expert_idx = tile_map[base + 0u];
    if (expert_idx >= E) return;
    const uint tile_seg_start = tile_map[base + 1u];
    const uint row_off_elems = tile_map[base + 2u];
    const uint tile_m = row_off_elems / ROW_TILE;

    pass_a_impl<T>(
        X_perm, expert_offsets, W13_all, up_biases, hidden_out,
        d_model, d_ff, E,
        gate_clip_min, gate_clip_max, up_clip_min, up_clip_max, silu_alpha,
        tile_seg_start, expert_idx, tile_m, n_tile_idx,
        sg_id, threads_per_tg, local_tid,
        Xs, Wk_up, Wk_gate, bias_up, bias_gate);
}

#define MOE_PASS_A_INDIRECT_KERNEL(DTYPE, SUFFIX) \
kernel void moe_two_pass_prefill_pass_a_indirect_##SUFFIX( \
    device const DTYPE* X_perm              [[buffer(0)]], \
    device const uint*  expert_offsets      [[buffer(1)]], \
    device const DTYPE* W13_all             [[buffer(2)]], \
    device const DTYPE* up_biases           [[buffer(3)]], \
    device float*       hidden_out          [[buffer(4)]], \
    constant uint&      d_model             [[buffer(5)]], \
    constant uint&      d_ff                [[buffer(6)]], \
    constant uint&      E                   [[buffer(7)]], \
    constant float&     gate_clip_min       [[buffer(8)]], \
    constant float&     gate_clip_max       [[buffer(9)]], \
    constant float&     up_clip_min         [[buffer(10)]], \
    constant float&     up_clip_max         [[buffer(11)]], \
    constant float&     silu_alpha          [[buffer(12)]], \
    device const uint*  tile_map            [[buffer(13)]], \
    uint                sg_id               [[simdgroup_index_in_threadgroup]], \
    uint3               threads_per_tg      [[threads_per_threadgroup]], \
    uint3               tg_pos              [[threadgroup_position_in_grid]], \
    uint3               local_tid           [[thread_position_in_threadgroup]]) \
{ \
    using StageT = typename StageStorage<DTYPE>::type; \
    threadgroup StageT Xs         [PASSA_BM * (PASSA_BK + PASSA_TG_PAD)]; \
    threadgroup StageT Wk_up      [PASSA_BN * (PASSA_BK + PASSA_TG_PAD)]; \
    threadgroup StageT Wk_gate    [PASSA_BN * (PASSA_BK + PASSA_TG_PAD)]; \
    threadgroup float bias_up    [PASSA_BN]; \
    threadgroup float bias_gate  [PASSA_BN]; \
    pass_a_indirect_impl<DTYPE>( \
        X_perm, expert_offsets, W13_all, up_biases, hidden_out, tile_map, \
        d_model, d_ff, E, \
        gate_clip_min, gate_clip_max, up_clip_min, up_clip_max, silu_alpha, \
        /*row_tile_idx*/ tg_pos.y, /*n_tile_idx*/ tg_pos.x, \
        sg_id, threads_per_tg, local_tid, \
        Xs, Wk_up, Wk_gate, bias_up, bias_gate); \
}

MOE_PASS_A_INDIRECT_KERNEL(bfloat, bf16)
MOE_PASS_A_INDIRECT_KERNEL(half,   f16)
MOE_PASS_A_INDIRECT_KERNEL(float,  f32)

// ------------------------ Pass B (hidden @ W2 → output) ------------------------
// Optimized config: 16x64 output tile, 4 simdgroups (128 threads)
constant uint PASSB_BM = 16;
constant uint PASSB_BN = 64;
constant uint PASSB_BK = 64;
constant uint PASSB_SG_BM = 8;   // Each simdgroup handles 8x32
constant uint PASSB_SG_BN = 32;
constant uint PASSB_TG_PAD = 4;  // Padding to avoid bank conflicts

template<typename T>
inline void pass_b_impl(
    device const float* hidden,         // [total_rows, FF]
    device const uint*  expert_offsets, // [E+1]
    device const T*     W2_all,         // [E, D, FF]  (FF contiguous)
    device const T*     down_biases,    // [E, D]
    device T*           output,         // [total_rows, D]
    uint d_model, uint d_ff, uint E,
    uint expert_idx, uint tile_m, uint tile_n,
    uint sg_id, uint simd_lid, uint lin,
    threadgroup float* Hs,                                 // [BM, BK+PAD]
    threadgroup typename StageStorage<T>::type* Wk,        // [BN, BK+PAD]
    threadgroup float* bias_tile                           // [BN]
) {
    using StageT = typename StageStorage<T>::type;

    constexpr uint Bm = PASSB_BM, Bn = PASSB_BN, Bk = PASSB_BK;
    constexpr uint SgBm = PASSB_SG_BM, SgBn = PASSB_SG_BN;
    constexpr uint TG_PAD = PASSB_TG_PAD;
    constexpr uint H_LD = Bk + TG_PAD;  // Padded stride for hidden
    constexpr uint W_LD = Bk + TG_PAD;  // Padded stride for weights
    constexpr uint SG_TILE = 8;
    constexpr uint TGP_SIZE = 128;  // 4 simdgroups × 32 threads

    if (expert_idx >= E) return;

    const uint seg_start = expert_offsets[expert_idx];
    const uint seg_end   = expert_offsets[expert_idx + 1];
    const uint seg_len   = (seg_end > seg_start) ? (seg_end - seg_start) : 0u;
    if (seg_len == 0) return;

    const uint row_tg_off = tile_m * Bm;
    const uint col_tg_off = tile_n * Bn;
    if (row_tg_off >= seg_len || col_tg_off >= d_model) return;

    const uint m_rows = min(Bm, seg_len  - row_tg_off);
    const uint n_cols = min(Bn, d_model  - col_tg_off);

    const ulong h_base          = (ulong)(seg_start + row_tg_off) * (ulong)d_ff;
    const ulong w2_expert_base  = (ulong)expert_idx * (ulong)d_model * (ulong)d_ff;
    const ulong bias_base       = (ulong)expert_idx * (ulong)d_model;

    // 2×2 simdgroup layout for 16×64 output (each sg handles 8×32)
    const uint row_sg = sg_id / 2;  // 0-1 → rows
    const uint col_sg = sg_id % 2;  // 0-1 → cols
    const uint row_sg_off = row_sg * SgBm;
    const uint col_sg_off = col_sg * SgBn;

    // Guard against excessive simdgroups
    constexpr uint SG_EXPECTED = (PASSB_BM / SgBm) * (PASSB_BN / SgBn);
    if (sg_id >= SG_EXPECTED) return;

    // 4 accumulators per simdgroup: 1×4 layout of 8×8 tiles
    constexpr uint TEMP = (SgBm / SG_TILE) * (SgBn / SG_TILE);  // 1×4 = 4
    metal::simdgroup_float8x8 Out[TEMP];
UZU_PRAGMA_UNROLL
    for (uint i = 0; i < TEMP; ++i) {
        Out[i] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    // Cooperative load config - vec8 loads for better bandwidth
    constexpr uint H_VEC = 8;
    constexpr uint H_TCOLS = Bk / H_VEC;  // 64/8 = 8
    constexpr uint H_TROWS = TGP_SIZE / H_TCOLS;  // 128/8 = 16
    const uint h_bi = lin / H_TCOLS;
    const uint h_bj = (lin % H_TCOLS) * H_VEC;

    constexpr uint W_VEC = 8;
    constexpr uint W_TCOLS = Bk / W_VEC;  // 64/8 = 8
    constexpr uint W_TROWS = TGP_SIZE / W_TCOLS;  // 128/8 = 16
    const uint w_bi = lin / W_TCOLS;
    const uint w_bj = (lin % W_TCOLS) * W_VEC;

    // K-loop across FF
    for (uint k_off = 0; k_off < d_ff; k_off += Bk) {
        const uint valid_k = min(Bk, d_ff - k_off);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Stage hidden [Bm=16, Bk=64] with vec8 loads ----
        // 128 threads, 8 vec8 loads per row → 16 rows covered in one pass
        {
            threadgroup float* my_dst = Hs + h_bi * H_LD + h_bj;
            if (h_bi < m_rows && h_bj < valid_k) {
                // Safe to read - at least first element is valid
                device const float* my_src = hidden + h_base + (ulong)h_bi * (ulong)d_ff + (ulong)(k_off + h_bj);
UZU_PRAGMA_UNROLL
                for (uint j = 0; j < H_VEC; j++) {
                    my_dst[j] = (h_bj + j < valid_k) ? my_src[j] : 0.0f;
                }
            } else {
UZU_PRAGMA_UNROLL
                for (uint j = 0; j < H_VEC; j++) {
                    my_dst[j] = 0.0f;
                }
            }
        }

        // ---- Stage W2 [Bn=64, Bk=64] with vec8 loads ----
        // Need 4 passes: 128 threads cover 16 rows, Bn=64 needs 4×16
        {
            for (uint pass = 0; pass < 4; pass++) {
                const uint row = w_bi + pass * W_TROWS;
                threadgroup StageT* my_dst = Wk + row * W_LD + w_bj;
                if (row < n_cols && w_bj < valid_k) {
                    // Safe to read - at least first element is valid
                    const uint gcol = col_tg_off + row;
                    device const T* my_src = W2_all + w2_expert_base + (ulong)gcol * (ulong)d_ff + (ulong)(k_off + w_bj);
UZU_PRAGMA_UNROLL
                    for (uint j = 0; j < W_VEC; j++) {
                        my_dst[j] = (w_bj + j < valid_k) ? StageT(float(my_src[j])) : StageT(0.0f);
                    }
                } else {
UZU_PRAGMA_UNROLL
                    for (uint j = 0; j < W_VEC; j++) {
                        my_dst[j] = StageT(0.0f);
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Serpentine MMA: 1×4 tiles per simdgroup ----
UZU_PRAGMA_UNROLL
        for (uint kk = 0; kk < Bk; kk += SG_TILE) {
            // Load 1 LHS fragment
            metal::simdgroup_float8x8 lhs;
            simdgroup_load(lhs, Hs, H_LD, ulong2(kk, row_sg_off));

            // Load 4 RHS fragments (cols)
            metal::simdgroup_matrix<StageT, 8, 8> rhs0, rhs1, rhs2, rhs3;
            simdgroup_load(rhs0, Wk, W_LD, ulong2(kk, col_sg_off), true);
            simdgroup_load(rhs1, Wk, W_LD, ulong2(kk, col_sg_off + 8), true);
            simdgroup_load(rhs2, Wk, W_LD, ulong2(kk, col_sg_off + 16), true);
            simdgroup_load(rhs3, Wk, W_LD, ulong2(kk, col_sg_off + 24), true);

            // Serpentine pattern: 0→1→2→3 (forward for first row)
            simdgroup_multiply_accumulate(Out[0], lhs, rhs0, Out[0]);
            simdgroup_multiply_accumulate(Out[1], lhs, rhs1, Out[1]);
            simdgroup_multiply_accumulate(Out[2], lhs, rhs2, Out[2]);
            simdgroup_multiply_accumulate(Out[3], lhs, rhs3, Out[3]);
        }
    }

    // ---- Bias tile (cooperative load) ----
    for (uint c_local = lin; c_local < Bn; c_local += TGP_SIZE) {
        const uint c_global = col_tg_off + c_local;
        bias_tile[c_local] = (c_global < d_model) ? float(down_biases[bias_base + c_global]) : 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Writeout with fragment extraction ----
    const uint lane_qid = simd_lid >> 2;
    const uint lane_row = (lane_qid & 4u) + ((simd_lid >> 1) & 3u);
    const uint lane_col_base = ((lane_qid & 2u) << 1) + ((simd_lid & 1u) << 1);

    const uint local_row = row_sg_off + lane_row;
    if (local_row >= m_rows) return;
    const ulong out_row = seg_start + row_tg_off + local_row;

UZU_PRAGMA_UNROLL
    for (uint ni = 0; ni < 4; ni++) {  // 4 col tiles per simdgroup
        const auto accum = Out[ni].thread_elements();
        const uint local_col = col_sg_off + ni * 8 + lane_col_base;

        if (local_col < n_cols) {
            const ulong out_col = col_tg_off + local_col;
            const float val = accum[0] + bias_tile[local_col];
            output[out_row * (ulong)d_model + out_col] = T(val);
        }
        if (local_col + 1 < n_cols) {
            const ulong out_col = col_tg_off + local_col + 1;
            const float val = accum[1] + bias_tile[local_col + 1];
            output[out_row * (ulong)d_model + out_col] = T(val);
        }
    }
}

// ----- Kernel entry points (Pass B) -----
#define MOE_PASS_B_KERNEL(DTYPE, SUFFIX) \
kernel void moe_two_pass_prefill_pass_b_##SUFFIX( \
    device const float* hidden           [[buffer(0)]], \
    device const uint*  expert_offsets   [[buffer(1)]], \
    device const DTYPE* W2_all           [[buffer(2)]], \
    device const DTYPE* down_biases      [[buffer(3)]], \
    device DTYPE*       output           [[buffer(4)]], \
    constant uint&      d_model          [[buffer(5)]], \
    constant uint&      d_ff             [[buffer(6)]], \
    constant uint&      E                [[buffer(7)]], \
    uint                sg_id            [[simdgroup_index_in_threadgroup]], \
    uint                simd_lid         [[thread_index_in_simdgroup]], \
    uint3               tg_pos           [[threadgroup_position_in_grid]], \
    uint3               local_tid        [[thread_position_in_threadgroup]]) \
{ \
    using StageT = typename StageStorage<DTYPE>::type; \
    threadgroup float Hs      [PASSB_BM * (PASSB_BK + PASSB_TG_PAD)]; \
    threadgroup StageT Wk     [PASSB_BN * (PASSB_BK + PASSB_TG_PAD)]; \
    threadgroup float bias    [PASSB_BN]; \
    const uint lin = local_tid.x; \
    pass_b_impl<DTYPE>( \
        hidden, expert_offsets, W2_all, down_biases, output, \
        d_model, d_ff, E, \
        /*expert*/ tg_pos.z, /*tile_m*/ tg_pos.y, /*tile_n*/ tg_pos.x, \
        sg_id, simd_lid, lin, \
        Hs, Wk, bias); \
}

MOE_PASS_B_KERNEL(bfloat, bf16)
MOE_PASS_B_KERNEL(half,   f16)
MOE_PASS_B_KERNEL(float,  f32)

// Indirect variant
template<typename T>
inline void pass_b_indirect_impl(
    device const float* hidden,
    device const uint*  expert_offsets,
    device const T*     W2_all,
    device const T*     down_biases,
    device T*           output,
    device const uint*  tile_map,      // [tiles * 3]
    uint d_model, uint d_ff, uint E,
    uint row_tile_idx, uint n_tile_idx,
    uint sg_id, uint simd_lid, uint lin,
    threadgroup float* Hs,
    threadgroup typename StageStorage<T>::type* Wk,
    threadgroup float* bias_tile)
{
    constexpr uint ROW_TILE = PASSB_BM;
    const uint base = row_tile_idx * 3u;
    const uint expert_idx = tile_map[base + 0u];
    if (expert_idx >= E) return;
    const uint row_off_elems = tile_map[base + 2u];
    const uint tile_m = row_off_elems / ROW_TILE;

    pass_b_impl<T>(
        hidden, expert_offsets, W2_all, down_biases, output,
        d_model, d_ff, E,
        expert_idx, tile_m, n_tile_idx,
        sg_id, simd_lid, lin,
        Hs, Wk, bias_tile);
}

#define MOE_PASS_B_INDIRECT_KERNEL(DTYPE, SUFFIX) \
kernel void moe_two_pass_prefill_pass_b_indirect_##SUFFIX( \
    device const float* hidden           [[buffer(0)]], \
    device const uint*  expert_offsets   [[buffer(1)]], \
    device const DTYPE* W2_all           [[buffer(2)]], \
    device const DTYPE* down_biases      [[buffer(3)]], \
    device DTYPE*       output           [[buffer(4)]], \
    constant uint&      d_model          [[buffer(5)]], \
    constant uint&      d_ff             [[buffer(6)]], \
    constant uint&      E                [[buffer(7)]], \
    device const uint*  tile_map         [[buffer(8)]], \
    uint                sg_id            [[simdgroup_index_in_threadgroup]], \
    uint                simd_lid         [[thread_index_in_simdgroup]], \
    uint3               tg_pos           [[threadgroup_position_in_grid]], \
    uint3               local_tid        [[thread_position_in_threadgroup]]) \
{ \
    using StageT = typename StageStorage<DTYPE>::type; \
    threadgroup float Hs      [PASSB_BM * (PASSB_BK + PASSB_TG_PAD)]; \
    threadgroup StageT Wk     [PASSB_BN * (PASSB_BK + PASSB_TG_PAD)]; \
    threadgroup float bias    [PASSB_BN]; \
    const uint lin = local_tid.x; \
    pass_b_indirect_impl<DTYPE>( \
        hidden, expert_offsets, W2_all, down_biases, output, tile_map, \
        d_model, d_ff, E, \
        /*row_tile_idx*/ tg_pos.y, /*n_tile_idx*/ tg_pos.x, \
        sg_id, simd_lid, lin, \
        Hs, Wk, bias); \
}

MOE_PASS_B_INDIRECT_KERNEL(bfloat, bf16)
MOE_PASS_B_INDIRECT_KERNEL(half,   f16)
MOE_PASS_B_INDIRECT_KERNEL(float,  f32)
