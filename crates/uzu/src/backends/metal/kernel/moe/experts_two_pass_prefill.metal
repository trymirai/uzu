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
    // TG scratch
    threadgroup typename StageStorage<T>::type* Xs,       // [BM,BK]
    threadgroup typename StageStorage<T>::type* Wk_up,    // [BN,BK]
    threadgroup typename StageStorage<T>::type* Wk_gate,  // [BN,BK]
    threadgroup float* scratch_up,    // [BM,BN]
    threadgroup float* scratch_gate,  // [BM,BN]
    threadgroup float* bias_up,       // [BN]
    threadgroup float* bias_gate      // [BN]
) {
    using StageT = typename StageStorage<T>::type;
    using StageVec4 = metal::vec<StageT, 4>;

    const uint Bm = PASSA_BM, Bn = PASSA_BN, Bk = PASSA_BK;
    const uint SgBm = PASSA_SG_BM, SgBn = PASSA_SG_BN;

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
    constexpr uint SG_EXPECTED = (PASSA_BM / SG_TILE) * (PASSA_BN / SG_TILE);
    if (sg_id >= SG_EXPECTED) return;

    // partial accumulators (8x8 tiles per simdgroup)
    constexpr uint TEMP = (PASSA_SG_BM / SG_TILE) * (PASSA_SG_BN / SG_TILE);
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

        // ---- Stage LHS: Xs [Bm,Bk] as float, vectorized 4 ----
        {
            constexpr uint row_stride = PASSA_BK;
            constexpr uint vec_cols   = PASSA_BK / 4;
            const uint total_vecs = PASSA_BM * vec_cols;
            const uint iters = ceil_div(total_vecs, threads_total);

UZU_PRAGMA_UNROLL
            for (uint t = 0; t < iters; ++t) {
                const uint i = t * threads_total + lin;
                if (i < total_vecs) {
                    const uint r  = i / vec_cols;
                    const uint c4 = i % vec_cols;

                    threadgroup StageVec4* dst4 =
                        reinterpret_cast<threadgroup StageVec4*>(Xs + r * row_stride + (c4 << 2));

                    StageVec4 packed = StageVec4(StageT(0.0f));
                    if (r < m_rows) {
                        const ulong base = x_base + (ulong)r * (ulong)d_model + (ulong)(k_off + (c4 << 2));
                        const uint base_vec = (c4 << 2);
                        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
                        if (base_vec + 0 < valid_k) v0 = float(X_perm[base + 0]);
                        if (base_vec + 1 < valid_k) v1 = float(X_perm[base + 1]);
                        if (base_vec + 2 < valid_k) v2 = float(X_perm[base + 2]);
                        if (base_vec + 3 < valid_k) v3 = float(X_perm[base + 3]);
                        packed = StageVec4(StageT(v0), StageT(v1), StageT(v2), StageT(v3));
                    }
                    *dst4 = packed;
                }
            }
        }

        // ---- Stage RHS: W_up [Bn,Bk], W_gate [Bn,Bk], vectorized 4 ----
        {
            constexpr uint row_stride = PASSA_BK;
            constexpr uint vec_cols   = PASSA_BK / 4;
            const uint total_vecs = PASSA_BN * vec_cols;
            const uint iters = ceil_div(total_vecs, threads_total);

UZU_PRAGMA_UNROLL
            for (uint t = 0; t < iters; ++t) {
                const uint i = t * threads_total + lin;
                if (i < total_vecs) {
                    const uint n   = i / vec_cols;  // column within the BN tile
                    const uint c4  = i % vec_cols;

                    threadgroup StageVec4* up4 =
                        reinterpret_cast<threadgroup StageVec4*>(Wk_up + n * row_stride + (c4 << 2));
                    StageVec4 upv = StageVec4(StageT(0.0f));

                    threadgroup StageVec4* gt4 =
                        reinterpret_cast<threadgroup StageVec4*>(Wk_gate + n * row_stride + (c4 << 2));
                    StageVec4 gtv = StageVec4(StageT(0.0f));

                    if (n < n_cols) {
                        const uint gcol = col_tg_off + n;
                        const ulong up_base   = w13_expert_base + (ulong)gcol * (ulong)d_model + (ulong)(k_off + (c4 << 2));
                        const ulong gate_base = w13_expert_base + (ulong)(d_ff + gcol) * (ulong)d_model + (ulong)(k_off + (c4 << 2));
                        const uint base_vec = (c4 << 2);
                        float up0 = 0.0f, up1 = 0.0f, up2 = 0.0f, up3 = 0.0f;
                        float gt0 = 0.0f, gt1 = 0.0f, gt2 = 0.0f, gt3 = 0.0f;
                        if (base_vec + 0 < valid_k) { up0 = float(W13_all[up_base + 0]); gt0 = float(W13_all[gate_base + 0]); }
                        if (base_vec + 1 < valid_k) { up1 = float(W13_all[up_base + 1]); gt1 = float(W13_all[gate_base + 1]); }
                        if (base_vec + 2 < valid_k) { up2 = float(W13_all[up_base + 2]); gt2 = float(W13_all[gate_base + 2]); }
                        if (base_vec + 3 < valid_k) { up3 = float(W13_all[up_base + 3]); gt3 = float(W13_all[gate_base + 3]); }
                        upv = StageVec4(StageT(up0), StageT(up1), StageT(up2), StageT(up3));
                        gtv = StageVec4(StageT(gt0), StageT(gt1), StageT(gt2), StageT(gt3));
                    }

                    *up4 = upv;
                    if (GATING_SEL > 1u) *gt4 = gtv;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- MMA across the BK tile (in simdgroup-sized chunks) ----
UZU_PRAGMA_UNROLL
        for (uint kk = 0; kk < PASSA_BK; kk += SG_TILE) {
UZU_PRAGMA_UNROLL
            for (uint m_sub = 0; m_sub < PASSA_SG_BM; m_sub += SG_TILE) {
                const uint r_idx = m_sub / SG_TILE;
                metal::simdgroup_matrix<StageT, 8, 8> lhs_frag;
                simdgroup_load(lhs_frag, Xs, PASSA_BK, ulong2(kk, row_sg_off + m_sub));

UZU_PRAGMA_UNROLL
                for (uint n_sub = 0; n_sub < PASSA_SG_BN; n_sub += SG_TILE) {
                    const uint c_idx = n_sub / SG_TILE;
                    const uint tile  = r_idx * (PASSA_SG_BN / SG_TILE) + c_idx;
                    metal::simdgroup_matrix<StageT, 8, 8> rhs_up;
                    simdgroup_load(rhs_up, Wk_up, PASSA_BK, ulong2(kk, col_sg_off + n_sub), true);
                    simdgroup_multiply_accumulate(OutUp[tile], lhs_frag, rhs_up, OutUp[tile]);

                    if (GATING_SEL > 1u) {
                        metal::simdgroup_matrix<StageT, 8, 8> rhs_gate;
                        simdgroup_load(rhs_gate, Wk_gate, PASSA_BK, ulong2(kk, col_sg_off + n_sub), true);
                        simdgroup_multiply_accumulate(OutGate[tile], lhs_frag, rhs_gate, OutGate[tile]);
                    }
                }
            }
        }
    } // K loop

    // ---- Store accumulators to scratch [Bm,Bn] ----
    UZU_PRAGMA_UNROLL
    for (uint n_sub = 0; n_sub < PASSA_SG_BN; n_sub += SG_TILE) {
        const uint c_idx = n_sub / SG_TILE;
        const uint col_off = col_sg_off + n_sub;
    UZU_PRAGMA_UNROLL
        for (uint m_sub = 0; m_sub < PASSA_SG_BM; m_sub += SG_TILE) {
            const uint r_idx = m_sub / SG_TILE;
            const uint row_off = row_sg_off + m_sub;
            const uint tile = r_idx * (PASSA_SG_BN / SG_TILE) + c_idx;

            simdgroup_store(OutUp[tile],   scratch_up,   PASSA_BN, ulong2(col_off, row_off));
            if (GATING_SEL > 1u) {
                simdgroup_store(OutGate[tile], scratch_gate, PASSA_BN, ulong2(col_off, row_off));
            }
        }
    }

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
    // Compute valid bounds with saturation to avoid unsigned underflow
    const uint row0 = row_sg_off;
    const uint col0 = col_sg_off;
    const uint row_end = min(m_rows, row0 + SgBm);
    const uint col_end = min(n_cols, col0 + SgBn);

    // Early exit if this simdgroup is entirely out of bounds
    if (row0 >= row_end || col0 >= col_end) return;

    const uint valid_rows = row_end - row0;
    const uint valid_cols = col_end - col0;

    // Only let this simdgroup's threads participate
    const uint sg_lane_start = sg_id * 32;
    const uint sg_lane_end = sg_lane_start + 32;
    if (lin < sg_lane_start || lin >= sg_lane_end) return;

    const uint sg_lin = lin - sg_lane_start;  // Thread index within simdgroup (0-31)

    const uint vec_cols = valid_cols / 4u;
    const uint vec_total = valid_rows * vec_cols;
    for (uint vec_idx = sg_lin; vec_idx < vec_total; vec_idx += 32u) {
        const uint lr = vec_idx / vec_cols;
        const uint lc_vec = vec_idx % vec_cols;
        const uint r = row0 + lr;
        const uint c = col0 + (lc_vec << 2);

        const uint scratch_base = r * Bn + c;
        float4 up_vals =
            *reinterpret_cast<threadgroup float4*>(scratch_up + scratch_base);
        const float4 bias_vals = float4(
            bias_up[c + 0],
            bias_up[c + 1],
            bias_up[c + 2],
            bias_up[c + 3]);
        up_vals += bias_vals;
        up_vals = clamp(up_vals, up_clip_min, up_clip_max);

        float4 out_vec;
        if (GATING_SEL <= 1u) {
            out_vec.x = (GATING_SEL == 0u) ? gelu_approx(up_vals.x) : silu(up_vals.x, silu_alpha);
            out_vec.y = (GATING_SEL == 0u) ? gelu_approx(up_vals.y) : silu(up_vals.y, silu_alpha);
            out_vec.z = (GATING_SEL == 0u) ? gelu_approx(up_vals.z) : silu(up_vals.z, silu_alpha);
            out_vec.w = (GATING_SEL == 0u) ? gelu_approx(up_vals.w) : silu(up_vals.w, silu_alpha);
        } else {
            float4 gate_vals =
                *reinterpret_cast<threadgroup float4*>(scratch_gate + scratch_base);
            const float4 gate_bias = float4(
                bias_gate[c + 0],
                bias_gate[c + 1],
                bias_gate[c + 2],
                bias_gate[c + 3]);
            gate_vals += gate_bias;
            gate_vals = clamp(gate_vals, gate_clip_min, gate_clip_max);
            if (GATING_SEL == 2u) {
                out_vec.x = silu(gate_vals.x, silu_alpha) * up_vals.x;
                out_vec.y = silu(gate_vals.y, silu_alpha) * up_vals.y;
                out_vec.z = silu(gate_vals.z, silu_alpha) * up_vals.z;
                out_vec.w = silu(gate_vals.w, silu_alpha) * up_vals.w;
            } else {
                out_vec.x = gelu_approx(gate_vals.x) * up_vals.x;
                out_vec.y = gelu_approx(gate_vals.y) * up_vals.y;
                out_vec.z = gelu_approx(gate_vals.z) * up_vals.z;
                out_vec.w = gelu_approx(gate_vals.w) * up_vals.w;
            }
        }

        const ulong out_base =
            (seg_start + row_tg_off + r) * (ulong)d_ff + (col_tg_off + c);
        *reinterpret_cast<device float4*>(hidden_out + out_base) = out_vec;
    }

    const uint tail_cols = valid_cols - vec_cols * 4u;
    if (tail_cols > 0u) {
        const uint tail_total = valid_rows * tail_cols;
        for (uint idx = sg_lin; idx < tail_total; idx += 32u) {
            const uint lr = idx / tail_cols;
            const uint lc_tail = idx % tail_cols;
            const uint r = row0 + lr;
            const uint c = col0 + vec_cols * 4u + lc_tail;
            const uint s = r * Bn + c;

            float up_v = scratch_up[s] + bias_up[c];
            up_v = clamp(up_v, up_clip_min, up_clip_max);

            float out;
            if (GATING_SEL <= 1u) {
                out = (GATING_SEL == 0u) ? gelu_approx(up_v) : silu(up_v, silu_alpha);
            } else {
                float gate_v = scratch_gate[s] + bias_gate[c];
                gate_v = clamp(gate_v, gate_clip_min, gate_clip_max);
                const float sw = (GATING_SEL == 2u) ? silu(gate_v, silu_alpha)
                                                    : gelu_approx(gate_v);
                out = sw * up_v;
            }

            const ulong out_idx =
                (seg_start + row_tg_off + r) * (ulong)d_ff + (col_tg_off + c);
            hidden_out[out_idx] = out;
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
    threadgroup StageT Xs         [PASSA_BM * PASSA_BK]; \
    threadgroup StageT Wk_up      [PASSA_BN * PASSA_BK]; \
    threadgroup StageT Wk_gate    [PASSA_BN * PASSA_BK]; \
    threadgroup float scratch_up [PASSA_BM * PASSA_BN]; \
    threadgroup float scratch_gt [PASSA_BM * PASSA_BN]; \
    threadgroup float bias_up    [PASSA_BN]; \
    threadgroup float bias_gate  [PASSA_BN]; \
    pass_a_impl<DTYPE>( \
        X_perm, expert_offsets, W13_all, up_biases, hidden_out, \
        d_model, d_ff, E, \
        gate_clip_min, gate_clip_max, up_clip_min, up_clip_max, silu_alpha, \
        /*tile_seg_start*/ expert_offsets[tg_pos.z], /*expert*/ tg_pos.z, /*tile_m*/ tg_pos.y, /*tile_n*/ tg_pos.x, \
        sg_id, threads_per_tg, local_tid, \
        Xs, Wk_up, Wk_gate, scratch_up, scratch_gt, bias_up, bias_gate); \
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
    threadgroup float* scratch_up,
    threadgroup float* scratch_gate,
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
        Xs, Wk_up, Wk_gate, scratch_up, scratch_gate, bias_up, bias_gate);
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
    threadgroup StageT Xs         [PASSA_BM * PASSA_BK]; \
    threadgroup StageT Wk_up      [PASSA_BN * PASSA_BK]; \
    threadgroup StageT Wk_gate    [PASSA_BN * PASSA_BK]; \
    threadgroup float scratch_up [PASSA_BM * PASSA_BN]; \
    threadgroup float scratch_gt [PASSA_BM * PASSA_BN]; \
    threadgroup float bias_up    [PASSA_BN]; \
    threadgroup float bias_gate  [PASSA_BN]; \
    pass_a_indirect_impl<DTYPE>( \
        X_perm, expert_offsets, W13_all, up_biases, hidden_out, tile_map, \
        d_model, d_ff, E, \
        gate_clip_min, gate_clip_max, up_clip_min, up_clip_max, silu_alpha, \
        /*row_tile_idx*/ tg_pos.y, /*n_tile_idx*/ tg_pos.x, \
        sg_id, threads_per_tg, local_tid, \
        Xs, Wk_up, Wk_gate, scratch_up, scratch_gt, bias_up, bias_gate); \
}

MOE_PASS_A_INDIRECT_KERNEL(bfloat, bf16)
MOE_PASS_A_INDIRECT_KERNEL(half,   f16)
MOE_PASS_A_INDIRECT_KERNEL(float,  f32)

// ------------------------ Pass B (hidden @ W2 → output) ------------------------
constant uint PASSB_BM = 16;
constant uint PASSB_BN = 64;
constant uint PASSB_BK = 64;
constant uint PASSB_SG_BM = 8;
constant uint PASSB_SG_BN = 32;

template<typename T>
inline void pass_b_impl(
    device const float* hidden,         // [total_rows, FF]
    device const uint*  expert_offsets, // [E+1]
    device const T*     W2_all,         // [E, D, FF]  (FF contiguous)
    device const T*     down_biases,    // [E, D]
    device T*           output,         // [total_rows, D]
    uint d_model, uint d_ff, uint E,
    uint expert_idx, uint tile_m, uint tile_n,
    uint sg_id, uint3 threads_per_tg, uint3 local_tid,
    threadgroup float* Hs,                                 // [BM,BK]
    threadgroup typename StageStorage<T>::type* Wk,        // [BN,BK]
    threadgroup float* scratch,                           // [BM,BN]
    threadgroup float* bias_tile                          // [BN]
) {
    using StageT = typename StageStorage<T>::type;
    using StageVec4 = metal::vec<StageT, 4>;

    const uint Bm = PASSB_BM, Bn = PASSB_BN, Bk = PASSB_BK;
    const uint SgBm = PASSB_SG_BM, SgBn = PASSB_SG_BN;

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

    const uint sg_col_count = Bn / SgBn;
    const uint row_sg = sg_id / sg_col_count;
    const uint col_sg = sg_id % sg_col_count;
    const uint row_sg_off = row_sg * SgBm;
    const uint col_sg_off = col_sg * SgBn;

    constexpr uint SG_TILE = 8;
    constexpr uint TEMP = (PASSB_SG_BM / SG_TILE) * (PASSB_SG_BN / SG_TILE);
    metal::simdgroup_float8x8 Out[TEMP];
    for (uint i = 0; i < TEMP; ++i) {
        Out[i] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const uint threads_total = threads_per_tg.x * threads_per_tg.y * threads_per_tg.z;
    const uint lin = linear_tid(local_tid, threads_per_tg);

    // K-loop across FF
    for (uint k_off = 0; k_off < d_ff; k_off += Bk) {
        const uint valid_k = min(Bk, d_ff - k_off);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Stage hidden [Bm,Bk] as float4 ----
        {
            constexpr uint row_stride = PASSB_BK;
            constexpr uint vec_cols   = PASSB_BK / 4;
            const uint total_vecs = PASSB_BM * vec_cols;
            const uint iters = ceil_div(total_vecs, threads_total);

UZU_PRAGMA_UNROLL
            for (uint t = 0; t < iters; ++t) {
                const uint i = t * threads_total + lin;
                if (i < total_vecs) {
                    const uint r  = i / vec_cols;
                    const uint c4 = i % vec_cols;

                    threadgroup float4* dst4 =
                        reinterpret_cast<threadgroup float4*>(Hs + r * row_stride + (c4 << 2));

                    float4 v = float4(0.0f);
                    if (r < m_rows) {
                        const uint base_vec = (c4 << 2);
                        const ulong base = h_base + (ulong)r * (ulong)d_ff + (ulong)(k_off + base_vec);
                        if (base_vec + 0u < valid_k) v.x = hidden[base + 0];
                        if (base_vec + 1u < valid_k) v.y = hidden[base + 1];
                        if (base_vec + 2u < valid_k) v.z = hidden[base + 2];
                        if (base_vec + 3u < valid_k) v.w = hidden[base + 3];
                    }
                    *dst4 = v;
                }
            }
        }

        // ---- Stage W2 [Bn,Bk] as StageT (column-major for simd transpose) ----
        {
            constexpr uint row_stride = PASSB_BK;
            constexpr uint vec_cols   = PASSB_BK / 4;
            const uint total_vecs = PASSB_BN * vec_cols;
            const uint iters = ceil_div(total_vecs, threads_total);

UZU_PRAGMA_UNROLL
            for (uint t = 0; t < iters; ++t) {
                const uint i = t * threads_total + lin;
                if (i < total_vecs) {
                    const uint n   = i / vec_cols;
                    const uint c4  = i % vec_cols;

                    threadgroup StageVec4* dst4 =
                        reinterpret_cast<threadgroup StageVec4*>(Wk + n * row_stride + (c4 << 2));

                    StageVec4 packed = StageVec4(StageT(0.0f));
                    if (n < n_cols) {
                        const uint gcol = col_tg_off + n;
                        const ulong base = w2_expert_base + (ulong)gcol * (ulong)d_ff + (ulong)(k_off + (c4 << 2));
                        const uint base_vec = (c4 << 2);
                        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
                        if (base_vec + 0u < valid_k) v0 = float(W2_all[base + 0]);
                        if (base_vec + 1u < valid_k) v1 = float(W2_all[base + 1]);
                        if (base_vec + 2u < valid_k) v2 = float(W2_all[base + 2]);
                        if (base_vec + 3u < valid_k) v3 = float(W2_all[base + 3]);
                        packed = StageVec4(StageT(v0), StageT(v1), StageT(v2), StageT(v3));
                    }
                    *dst4 = packed;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- MMA ----
UZU_PRAGMA_UNROLL
        for (uint kk = 0; kk < PASSB_BK; kk += SG_TILE) {
UZU_PRAGMA_UNROLL
            for (uint m_sub = 0; m_sub < PASSB_SG_BM; m_sub += SG_TILE) {
                const uint r_idx = m_sub / SG_TILE;
                metal::simdgroup_float8x8 lhs;
                simdgroup_load(lhs, Hs, PASSB_BK, ulong2(kk, row_sg_off + m_sub));
UZU_PRAGMA_UNROLL
                for (uint n_sub = 0; n_sub < PASSB_SG_BN; n_sub += SG_TILE) {
                    const uint c_idx = n_sub / SG_TILE;
                    const uint tile  = r_idx * (PASSB_SG_BN / SG_TILE) + c_idx;
                    metal::simdgroup_matrix<StageT, 8, 8> rhs;
                    simdgroup_load(rhs, Wk, PASSB_BK, ulong2(kk, col_sg_off + n_sub), true);
                    simdgroup_multiply_accumulate(Out[tile], lhs, rhs, Out[tile]);
                }
            }
        }
    }

    // ---- Store to scratch ----
UZU_PRAGMA_UNROLL
    for (uint n_sub = 0; n_sub < PASSB_SG_BN; n_sub += SG_TILE) {
        const uint c_idx = n_sub / SG_TILE;
        const uint col_off = col_sg_off + n_sub;
UZU_PRAGMA_UNROLL
        for (uint m_sub = 0; m_sub < PASSB_SG_BM; m_sub += SG_TILE) {
            const uint r_idx = m_sub / SG_TILE;
            const uint row_off = row_sg_off + m_sub;
            const uint tile = r_idx * (PASSB_SG_BN / SG_TILE) + c_idx;
            simdgroup_store(Out[tile], scratch, PASSB_BN, ulong2(col_off, row_off));
        }
    }

    // ---- Bias tile ----
    for (uint c_local = lin; c_local < Bn; c_local += threads_total) {
        const uint c_global = col_tg_off + c_local;
        bias_tile[c_local] = (c_global < d_model) ? float(down_biases[bias_base + c_global]) : 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Writeout ----
    const uint row0 = row_sg_off;
    const uint col0 = col_sg_off;
    const uint row_end = min(m_rows, row0 + SgBm);
    const uint col_end = min(n_cols, col0 + SgBn);

    // Early exit if this simdgroup is entirely out of bounds
    if (row0 >= row_end || col0 >= col_end) return;

    const uint valid_rows = row_end - row0;
    const uint valid_cols = col_end - col0;

    // Only let this simdgroup's threads participate
    const uint sg_lane_start = sg_id * 32;
    const uint sg_lane_end = sg_lane_start + 32;
    if (lin < sg_lane_start || lin >= sg_lane_end) return;

    const uint sg_lin = lin - sg_lane_start;  // Thread index within simdgroup (0-31)

    const uint vec_cols = valid_cols / 4u;
    const uint vec_total = valid_rows * vec_cols;
    for (uint vec_idx = sg_lin; vec_idx < vec_total; vec_idx += 32u) {
        const uint lr = vec_idx / vec_cols;
        const uint lc_vec = vec_idx % vec_cols;
        const uint r = row0 + lr;
        const uint c = col0 + (lc_vec << 2);

        const uint scratch_base = r * Bn + c;
        float4 vals =
            *reinterpret_cast<threadgroup float4*>(scratch + scratch_base);
        vals += float4(
            bias_tile[c + 0],
            bias_tile[c + 1],
            bias_tile[c + 2],
            bias_tile[c + 3]);

        const ulong out_base =
            (seg_start + row_tg_off + r) * (ulong)d_model + (col_tg_off + c);
        store_vec4<T>(output, out_base, vals);
    }

    const uint tail_cols = valid_cols - vec_cols * 4u;
    if (tail_cols > 0u) {
        const uint tail_total = valid_rows * tail_cols;
        for (uint idx = sg_lin; idx < tail_total; idx += 32u) {
            const uint lr = idx / tail_cols;
            const uint lc_tail = idx % tail_cols;
            const uint r = row0 + lr;
            const uint c = col0 + vec_cols * 4u + lc_tail;
            const float val = scratch[r * Bn + c] + bias_tile[c];

            const ulong out_idx =
                (seg_start + row_tg_off + r) * (ulong)d_model + (col_tg_off + c);
            output[out_idx] = T(val);
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
    uint3               threads_per_tg   [[threads_per_threadgroup]], \
    uint3               tg_pos           [[threadgroup_position_in_grid]], \
    uint3               local_tid        [[thread_position_in_threadgroup]]) \
{ \
    using StageT = typename StageStorage<DTYPE>::type; \
    threadgroup float Hs      [PASSB_BM * PASSB_BK]; \
    threadgroup StageT Wk     [PASSB_BN * PASSB_BK]; \
    threadgroup float scratch [PASSB_BM * PASSB_BN]; \
    threadgroup float bias    [PASSB_BN]; \
    pass_b_impl<DTYPE>( \
        hidden, expert_offsets, W2_all, down_biases, output, \
        d_model, d_ff, E, \
        /*expert*/ tg_pos.z, /*tile_m*/ tg_pos.y, /*tile_n*/ tg_pos.x, \
        sg_id, threads_per_tg, local_tid, \
        Hs, Wk, scratch, bias); \
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
    uint sg_id, uint3 threads_per_tg, uint3 local_tid,
    threadgroup float* Hs,
    threadgroup typename StageStorage<T>::type* Wk,
    threadgroup float* scratch,
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
        sg_id, threads_per_tg, local_tid,
        Hs, Wk, scratch, bias_tile);
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
    uint3               threads_per_tg   [[threads_per_threadgroup]], \
    uint3               tg_pos           [[threadgroup_position_in_grid]], \
    uint3               local_tid        [[thread_position_in_threadgroup]]) \
{ \
    using StageT = typename StageStorage<DTYPE>::type; \
    threadgroup float Hs      [PASSB_BM * PASSB_BK]; \
    threadgroup StageT Wk     [PASSB_BN * PASSB_BK]; \
    threadgroup float scratch [PASSB_BM * PASSB_BN]; \
    threadgroup float bias    [PASSB_BN]; \
    pass_b_indirect_impl<DTYPE>( \
        hidden, expert_offsets, W2_all, down_biases, output, tile_map, \
        d_model, d_ff, E, \
        /*row_tile_idx*/ tg_pos.y, /*n_tile_idx*/ tg_pos.x, \
        sg_id, threads_per_tg, local_tid, \
        Hs, Wk, scratch, bias); \
}

MOE_PASS_B_INDIRECT_KERNEL(bfloat, bf16)
MOE_PASS_B_INDIRECT_KERNEL(half,   f16)
MOE_PASS_B_INDIRECT_KERNEL(float,  f32)
