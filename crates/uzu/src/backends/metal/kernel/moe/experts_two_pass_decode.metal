#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant uint GATING_SEL [[function_constant(30)]]; // 0=GELU,1=SiLU,2=SwiGLU,3=GEGLU
constant uint TILE_H [[function_constant(32)]];    // tile size for hidden dimension in Pass A
constant uint K_TILE [[function_constant(33)]];    // split-K tile size for Pass B

static inline float gelu_approx(float x) {
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    if (x > 10.0f) return x;
    if (x < -10.0f) return 0.0f;
    return 0.5f * x * (1.0f + tanh(clamp(k0 * (x + k1 * x * x * x), -10.0f, 10.0f)));
}

static inline float silu(float x, float alpha) {
    return x / (1.0f + exp(-alpha * x));
}

#define MTL_CONST static constant constexpr const
#define MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// Tiling configuration (tunable)
MTL_CONST uint BM = 1;   // Threadgroups in M dimension (usually 1 for GEMV)
MTL_CONST uint BN = 4;   // Threadgroups in N dimension
MTL_CONST uint SM = 1;   // Simdgroup rows
MTL_CONST uint SN = 32;  // Simdgroup cols (must be power of 2, ≤32)
MTL_CONST uint TM = 4;   // Thread work in M (output elements per thread)
MTL_CONST uint TN = 4;   // Thread work in N (input elements per iteration)

MTL_CONST uint THREADS_M = BM * SM;
MTL_CONST uint THREADS_N = BN * SN;
MTL_CONST uint BLOCK_M = THREADS_M * TM;
MTL_CONST uint BLOCK_N = THREADS_N * TN;

static_assert(SM * SN == 32, "simdgroup must have 32 threads");
static_assert(SN == 4 || SN == 8 || SN == 16 || SN == 32, "SN must be 4, 8, 16, or 32");

template<uint SIMD_SIZE>
static inline float simdgroup_sum(float val, uint simd_lid) {
    MTL_PRAGMA_UNROLL
    for (uint offset = SIMD_SIZE / 2; offset >= 1; offset >>= 1) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

// === Pass A: Optimized GEMV with hierarchical tiling ===
// Computes: hidden[row, h] = activation(x[row, d] @ W13[d, 2*h])
template<typename T>
void moe_experts_decode_pass_a_impl(
    device const T* X_perm,              // [total_rows, d_model]
    device const uint* expert_offsets,   // [E + 1]
    device const T* W13_all,             // [E, d_model, 2*d_ff]
    device const T* up_biases,           // [E, 2*d_ff]
    device T* hidden_out,                // [total_rows, d_ff]
    uint d_model,
    uint d_ff,
    uint E,
    float gate_clip_min,
    float gate_clip_max,
    float up_clip_min,
    float up_clip_max,
    float silu_alpha,
    uint expert_idx,
    uint row_in_expert,
    uint h_block_idx,
    uint simd_gid,
    uint simd_lid,
    threadgroup float* tgp_memory
) {
    // Thread position within simdgroup
    const uint thrM = SN != 32 ? simd_lid / SN : 0;
    const uint thrN = SN != 32 ? simd_lid % SN : uint(simd_lid);

    // Simdgroup position within threadgroup
    const uint sgN = BN != 1 ? (simd_gid % BN) : 0;
    const uint simdM = BN != 1 ? SM * (simd_gid / BN) : uint(SM * simd_gid);
    const uint simdN = BN != 1 ? SN * (simd_gid % BN) : 0;

    // Thread's work block
    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;

    // Output row position
    const uint seg_start = expert_offsets[expert_idx];
    const uint seg_end = expert_offsets[expert_idx + 1];
    uint global_row = seg_start + row_in_expert;

    if (global_row >= seg_end) return;

    // Calculate output h index
    int h_row = h_block_idx * BLOCK_M + bm;

    // Adjust tail block to ensure in-bounds reads
    if (h_row + TM > d_ff) {
        h_row = d_ff > TM ? int(d_ff - TM) : 0;
        bm = h_row - int(h_block_idx * BLOCK_M);
    }

    if (h_row >= int(d_ff)) return;

    // Accumulation arrays for up and gate projections
    thread float result_up[TM] = {0};
    thread float result_gate[TM] = {0};
    thread T weights_up[TN];
    thread T weights_gate[TN];
    thread float x_vals[TN];

    // Expert weight base addresses
    const ulong w13_base = (ulong)expert_idx * (ulong)d_model * (ulong)(2 * d_ff);
    const ulong bias_base = (ulong)expert_idx * (ulong)(2 * d_ff);
    const ulong x_row_base = (ulong)global_row * (ulong)d_model;

    // Main accumulation loop over d_model in blocks of BLOCK_N
    const uint n_iter = d_model / BLOCK_N;
    const uint leftover = d_model - (n_iter * BLOCK_N);

    for (uint i = 0; i < n_iter; ++i) {
        // Load x values for this block
        MTL_PRAGMA_UNROLL
        for (uint tn = 0; tn < TN; tn++) {
            uint d_idx = bn + tn;
            x_vals[tn] = float(X_perm[x_row_base + d_idx]);
        }

        // Accumulate for each output row
        MTL_PRAGMA_UNROLL
        for (uint tm = 0; tm < TM; tm++) {
            uint h_idx = h_row + tm;

            // Load weights for up projection
            MTL_PRAGMA_UNROLL
            for (uint tn = 0; tn < TN; tn++) {
                uint d_idx = bn + tn;
                ulong w_idx = w13_base + (ulong)d_idx * (ulong)(2 * d_ff) + (ulong)h_idx;
                weights_up[tn] = W13_all[w_idx];
            }

            // Accumulate up projection
            MTL_PRAGMA_UNROLL
            for (uint tn = 0; tn < TN; tn++) {
                result_up[tm] += x_vals[tn] * float(weights_up[tn]);
            }

            // Load and accumulate gate projection if needed
            if (GATING_SEL > 1) {
                MTL_PRAGMA_UNROLL
                for (uint tn = 0; tn < TN; tn++) {
                    uint d_idx = bn + tn;
                    ulong w_gate_idx = w13_base + (ulong)d_idx * (ulong)(2 * d_ff) + (ulong)(d_ff + h_idx);
                    weights_gate[tn] = W13_all[w_gate_idx];
                }

                MTL_PRAGMA_UNROLL
                for (uint tn = 0; tn < TN; tn++) {
                    result_gate[tm] += x_vals[tn] * float(weights_gate[tn]);
                }
            }
        }

        bn += BLOCK_N;
    }

    // Handle leftover elements
    if (leftover > 0) {
        MTL_PRAGMA_UNROLL
        for (uint tn = 0; tn < TN; tn++) {
            uint d_idx = bn + tn;
            x_vals[tn] = (d_idx < d_model) ? float(X_perm[x_row_base + d_idx]) : 0.0f;
        }

        MTL_PRAGMA_UNROLL
        for (uint tm = 0; tm < TM; tm++) {
            uint h_idx = h_row + tm;

            MTL_PRAGMA_UNROLL
            for (uint tn = 0; tn < TN; tn++) {
                uint d_idx = bn + tn;
                if (d_idx < d_model) {
                    ulong w_idx = w13_base + (ulong)d_idx * (ulong)(2 * d_ff) + (ulong)h_idx;
                    weights_up[tn] = W13_all[w_idx];
                    result_up[tm] += x_vals[tn] * float(weights_up[tn]);

                    if (GATING_SEL > 1) {
                        ulong w_gate_idx = w13_base + (ulong)d_idx * (ulong)(2 * d_ff) + (ulong)(d_ff + h_idx);
                        weights_gate[tn] = W13_all[w_gate_idx];
                        result_gate[tm] += x_vals[tn] * float(weights_gate[tn]);
                    }
                }
            }
        }
    }

    // Simdgroup reduction across thrN dimension
    MTL_PRAGMA_UNROLL
    for (uint tm = 0; tm < TM; tm++) {
        result_up[tm] = simd_sum(result_up[tm]);
        if (GATING_SEL > 1) {
            result_gate[tm] = simd_sum(result_gate[tm]);
        }
    }

    // Threadgroup reduction if BN > 1
    if (BN > 1) {
        threadgroup float* tgp_up = tgp_memory;
        threadgroup float* tgp_gate = tgp_memory + BN * (BLOCK_M + TM);

        if (thrN == 0) {
            MTL_PRAGMA_UNROLL
            for (uint tm = 0; tm < TM; tm++) {
                tgp_up[sgN * (BLOCK_M + TM) + bm + tm] = result_up[tm];
                if (GATING_SEL > 1) {
                    tgp_gate[sgN * (BLOCK_M + TM) + bm + tm] = result_gate[tm];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgN == 0) {
                MTL_PRAGMA_UNROLL
                for (uint sgn = 1; sgn < BN; sgn++) {
                    MTL_PRAGMA_UNROLL
                    for (uint tm = 0; tm < TM; tm++) {
                        result_up[tm] += tgp_up[sgn * (BLOCK_M + TM) + bm + tm];
                        if (GATING_SEL > 1) {
                            result_gate[tm] += tgp_gate[sgn * (BLOCK_M + TM) + bm + tm];
                        }
                    }
                }
            }
        }
    }

    // Apply activation and write output (only first thread in simdgroup width)
    if (simdN == 0 && thrN == 0) {
        MTL_PRAGMA_UNROLL
        for (uint tm = 0; tm < TM; tm++) {
            uint h_idx = h_row + tm;
            if (h_idx < d_ff) {
                // Add bias and clip
                float up_val = result_up[tm] + float(up_biases[bias_base + h_idx]);
                up_val = clamp(up_val, up_clip_min, up_clip_max);

                float activated_val;
                if (GATING_SEL <= 1) {
                    activated_val = (GATING_SEL == 0) ? gelu_approx(up_val) : silu(up_val, silu_alpha);
                } else {
                    float gate_val = result_gate[tm] + float(up_biases[bias_base + d_ff + h_idx]);
                    gate_val = clamp(gate_val, gate_clip_min, gate_clip_max);
                    float gate_activated = (GATING_SEL == 2) ? silu(gate_val, silu_alpha) : gelu_approx(gate_val);
                    activated_val = gate_activated * up_val;
                }

                ulong hidden_idx = (ulong)global_row * (ulong)d_ff + (ulong)h_idx;
                hidden_out[hidden_idx] = T(activated_val);
            }
        }
    }
}

#define MOE_PASS_A_KERNEL(DTYPE, SUFFIX) \
kernel void moe_experts_decode_pass_a_##SUFFIX( \
    device const DTYPE* X_perm [[buffer(0)]], \
    device const uint* expert_offsets [[buffer(1)]], \
    device const DTYPE* W13_all [[buffer(2)]], \
    device DTYPE* hidden_out [[buffer(3)]], \
    device const DTYPE* up_biases [[buffer(4)]], \
    constant uint& d_model [[buffer(5)]], \
    constant uint& d_ff [[buffer(6)]], \
    constant uint& E [[buffer(7)]], \
    constant float& gate_clip_min [[buffer(8)]], \
    constant float& gate_clip_max [[buffer(9)]], \
    constant float& up_clip_min [[buffer(10)]], \
    constant float& up_clip_max [[buffer(11)]], \
    constant float& silu_alpha [[buffer(12)]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]) \
{ \
    constexpr uint tgp_mem_size = (BN > 1) ? 2 * BN * (BLOCK_M + TM) : 1; \
    threadgroup float tgp_memory[tgp_mem_size]; \
    \
    moe_experts_decode_pass_a_impl<DTYPE>( \
        X_perm, expert_offsets, W13_all, up_biases, hidden_out, \
        d_model, d_ff, E, gate_clip_min, gate_clip_max, \
        up_clip_min, up_clip_max, silu_alpha, \
        tgpig.y, tgpig.z, tgpig.x, \
        simd_gid, simd_lid, \
        (BN > 1) ? tgp_memory : nullptr); \
}

MOE_PASS_A_KERNEL(bfloat, bf16)
MOE_PASS_A_KERNEL(half, f16)
MOE_PASS_A_KERNEL(float, f32)


// === Pass B: compute partial down projections (split-K) ===
template<typename T>
void moe_experts_decode_down_partial_impl(
    device const T* hidden,               // [total_rows, d_ff]
    device const uint* expert_offsets,    // [E + 1]
    device const T* w2_all,               // [E, d_ff, d_model]
    device float* partial_out,            // [num_tiles_k, total_rows, d_model]
    uint total_rows,
    uint d_model,
    uint d_ff,
    uint num_tiles_k,
    uint E,
    uint3 tgpig,
    uint lid
) {
    constexpr uint THREADS_PER_TG = 256;

    const uint col_group = tgpig.x;
    const uint row_idx = tgpig.y;
    const uint tile_idx = tgpig.z;

    if (row_idx >= total_rows) return;
    if (tile_idx >= num_tiles_k) return;

    const uint col0 = col_group * THREADS_PER_TG;
    const uint my_col = col0 + lid;
    if (my_col >= d_model) return;

    uint expert_idx = 0u;
    for (uint expert = 0u; expert < E; ++expert) {
        const uint start = expert_offsets[expert];
        const uint end = expert_offsets[expert + 1u];
        if (row_idx >= start && row_idx < end) {
            expert_idx = expert;
            break;
        }
    }

    const uint k_start = tile_idx * K_TILE;
    if (k_start >= d_ff) return;
    const uint k_chunk = min(K_TILE, d_ff - k_start);

    const ulong hidden_row_base = (ulong)row_idx * (ulong)d_ff + (ulong)k_start;

    float acc = 0.0f;
    for (uint k = 0; k < k_chunk; ++k) {
        const float h_val = float(hidden[hidden_row_base + k]);
        const uint ff_idx = k_start + k;
        const ulong base =
            (ulong)expert_idx * (ulong)d_ff * (ulong)d_model +
            (ulong)ff_idx * (ulong)d_model +
            (ulong)my_col;
        const float w_val = float(w2_all[base]);
        acc = fma(h_val, w_val, acc);
    }

    const ulong partial_idx = ((ulong)tile_idx * (ulong)total_rows + (ulong)row_idx) * (ulong)d_model + (ulong)my_col;
    partial_out[partial_idx] = acc;
}

#define MOE_PASS_B_PARTIAL_KERNEL(DTYPE, SUFFIX) \
kernel void moe_experts_decode_down_partial_##SUFFIX( \
    device const DTYPE* hidden [[buffer(0)]], \
    device const uint* expert_offsets [[buffer(1)]], \
    device const DTYPE* w2_all [[buffer(2)]], \
    device float* partial_out [[buffer(3)]], \
    constant uint& total_rows [[buffer(4)]], \
    constant uint& d_model [[buffer(5)]], \
    constant uint& d_ff [[buffer(6)]], \
    constant uint& num_tiles_k [[buffer(7)]], \
    constant uint& E [[buffer(8)]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    uint lid [[thread_index_in_threadgroup]]) \
{ \
    moe_experts_decode_down_partial_impl<DTYPE>( \
        hidden, expert_offsets, w2_all, partial_out, \
        total_rows, d_model, d_ff, num_tiles_k, \
        E, \
        tgpig, lid); \
}

MOE_PASS_B_PARTIAL_KERNEL(bfloat, bf16)
MOE_PASS_B_PARTIAL_KERNEL(half, f16)
MOE_PASS_B_PARTIAL_KERNEL(float, f32)

// === Pass C: reduce split-K partials, add bias, and cast to output type ===
template<typename T>
void moe_experts_decode_down_reduce_impl(
    device const float* partial_in,       // [num_tiles_k, total_rows, d_model]
    device const uint* expert_offsets,    // [E + 1]
    device const T* down_biases,          // [E, d_model]
    device T* y_out,                      // [total_rows, d_model]
    uint total_rows,
    uint d_model,
    uint num_tiles_k,
    uint d_model_stride,
    uint E,
    uint2 tgpig,
    uint lid
) {
    constexpr uint THREADS_PER_TG = 256;

    const uint col_group = tgpig.x;
    const uint row_idx = tgpig.y;
    if (row_idx >= total_rows) return;

    const uint col0 = col_group * THREADS_PER_TG;
    const uint my_col = col0 + lid;
    if (my_col >= d_model) return;

    float sum = 0.0f;
    for (uint tile = 0; tile < num_tiles_k; ++tile) {
        const ulong partial_idx = ((ulong)tile * (ulong)total_rows + (ulong)row_idx) * (ulong)d_model + (ulong)my_col;
        sum += partial_in[partial_idx];
    }

    uint expert_idx = 0u;
    for (uint expert = 0u; expert < E; ++expert) {
        const uint start = expert_offsets[expert];
        const uint end = expert_offsets[expert + 1u];
        if (row_idx >= start && row_idx < end) {
            expert_idx = expert;
            break;
        }
    }
    const ulong bias_idx = (ulong)expert_idx * (ulong)d_model_stride + (ulong)my_col;
    sum += float(down_biases[bias_idx]);

    const ulong out_idx = (ulong)row_idx * (ulong)d_model_stride + (ulong)my_col;
    y_out[out_idx] = T(sum);
}

#define MOE_PASS_B_REDUCE_KERNEL(DTYPE, SUFFIX) \
kernel void moe_experts_decode_down_reduce_##SUFFIX( \
    device const float* partial_in [[buffer(0)]], \
    device const uint* expert_offsets [[buffer(1)]], \
    device const DTYPE* down_biases [[buffer(2)]], \
    device DTYPE* y_out [[buffer(3)]], \
    constant uint& total_rows [[buffer(4)]], \
    constant uint& d_model [[buffer(5)]], \
    constant uint& num_tiles_k [[buffer(6)]], \
    constant uint& E [[buffer(7)]], \
    uint2 tgpig [[threadgroup_position_in_grid]], \
    uint lid [[thread_index_in_threadgroup]]) \
{ \
    moe_experts_decode_down_reduce_impl<DTYPE>( \
        partial_in, expert_offsets, down_biases, y_out, \
        total_rows, d_model, num_tiles_k, d_model, E, \
        tgpig, lid); \
}

MOE_PASS_B_REDUCE_KERNEL(bfloat, bf16)
MOE_PASS_B_REDUCE_KERNEL(half, f16)
MOE_PASS_B_REDUCE_KERNEL(float, f32)
