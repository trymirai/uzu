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
    device const T* W13_all,             // weights in transposed layout [E, 2*d_ff, d_model]
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
            // Transposed layout: [E, 2*d_ff, d_model] for contiguous d_model access
            MTL_PRAGMA_UNROLL
            for (uint tn = 0; tn < TN; tn++) {
                uint d_idx = bn + tn;
                ulong w_idx = w13_base + (ulong)h_idx * (ulong)d_model + (ulong)d_idx;
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
                    ulong w_gate_idx = w13_base + (ulong)(d_ff + h_idx) * (ulong)d_model + (ulong)d_idx;
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
                    // Transposed layout: [E, 2*d_ff, d_model]
                    ulong w_idx = w13_base + (ulong)h_idx * (ulong)d_model + (ulong)d_idx;
                    weights_up[tn] = W13_all[w_idx];
                    result_up[tm] += x_vals[tn] * float(weights_up[tn]);

                    if (GATING_SEL > 1) {
                        ulong w_gate_idx = w13_base + (ulong)(d_ff + h_idx) * (ulong)d_model + (ulong)d_idx;
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

// === Helper kernels for indirect dispatch of Pass A ===

// Count tiles per expert: tiles = (num_rows > 0) ? num_rows * h_blocks : 0
kernel void moe_pass_a_tile_counts(
    device const uint* expert_offsets [[buffer(0)]],  // [E+1]
    device uint* tile_counts [[buffer(1)]],           // [E]
    constant uint& E [[buffer(2)]],
    constant uint& h_blocks [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= E) return;
    const uint start = expert_offsets[tid];
    const uint end = expert_offsets[tid + 1];
    const uint num_rows = end - start;
    tile_counts[tid] = (num_rows > 0) ? (num_rows * h_blocks) : 0;
}

// Exclusive scan of tile_counts to get tile_offsets and total_tiles
kernel void moe_pass_a_tile_scan(
    device const uint* tile_counts [[buffer(0)]],   // [E]
    device uint* tile_offsets [[buffer(1)]],        // [E+1]
    device uint* total_tiles [[buffer(2)]],         // [1]
    constant uint& E [[buffer(3)]],
    uint lid [[thread_index_in_threadgroup]],
    threadgroup uint* scratch [[threadgroup(0)]]
) {
    // Simple single-threadgroup scan (works for E <= 1024)
    const uint idx = lid;

    // Load into threadgroup memory
    if (idx < E) {
        scratch[idx] = tile_counts[idx];
    } else {
        scratch[idx] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Kogge-Stone scan
    uint val = scratch[idx];
    MTL_PRAGMA_UNROLL
    for (uint offset = 1; offset < 1024; offset *= 2) {
        uint temp = 0;
        if (idx >= offset && idx < E) {
            temp = scratch[idx - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (idx >= offset && idx < E) {
            val += temp;
            scratch[idx] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write exclusive scan (shift right by 1)
    if (idx == 0) {
        tile_offsets[0] = 0;
    }
    if (idx < E) {
        tile_offsets[idx + 1] = scratch[idx];
        if (idx == E - 1) {
            total_tiles[0] = scratch[idx];
        }
    }
}

// Build row→expert map: one thread per routed row
kernel void moe_pass_a_build_row_map(
    device const uint* expert_offsets [[buffer(0)]] , // [E+1]
    device uint* row_expert_map [[buffer(1)]] ,       // [total_rows]
    constant uint& total_rows [[buffer(2)]],
    constant uint& E [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_rows) return;

    uint left = 0u;
    uint right = E;
    const uint row = tid;

    while (left + 1u < right) {
        const uint mid = (left + right) >> 1;
        if (row < expert_offsets[mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }

    row_expert_map[row] = left;
}

// Build tile map entries from row→expert map
kernel void moe_pass_a_build_tile_map(
    device const uint* expert_offsets [[buffer(0)]],  // [E+1]
    device const uint* tile_offsets [[buffer(1)]],    // [E+1]
    device const uint* row_expert_map [[buffer(2)]],  // [total_rows]
    device uint* tile_map [[buffer(3)]],              // [total_tiles * 3]
    constant uint& total_rows [[buffer(4)]],
    constant uint& h_blocks [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint total_tiles = total_rows * h_blocks;
    if (tid >= total_tiles) return;

    const uint row_idx = tid / h_blocks;
    const uint h_block = tid % h_blocks;

    if (row_idx >= total_rows) return;

    const uint expert_idx = row_expert_map[row_idx];
    const uint row_start = expert_offsets[expert_idx];
    const uint row_in_expert = row_idx - row_start;
    const uint tile_base = tile_offsets[expert_idx] + row_in_expert * h_blocks + h_block;

    tile_map[tile_base * 3 + 0] = h_block;
    tile_map[tile_base * 3 + 1] = expert_idx;
    tile_map[tile_base * 3 + 2] = row_in_expert;
}

// Write dispatch args for indirect dispatch (reusable from tiled version)
kernel void moe_pass_a_write_dispatch_args(
    device const uint* total_tiles [[buffer(0)]],  // [1]
    device uint* dispatch_args [[buffer(1)]],      // [3] - MTLDispatchThreadgroupsIndirectArguments
    constant uint& num_tiles_y [[buffer(2)]],      // usually 1 for Pass A
    uint tid [[thread_position_in_grid]]
) {
    if (tid > 0) return;
    dispatch_args[0] = total_tiles[0];  // x dimension = total tiles
    dispatch_args[1] = num_tiles_y;     // y dimension
    dispatch_args[2] = 1;               // z dimension
}

// Modified Pass A that reads from tile map for indirect dispatch
template<typename T>
void moe_experts_decode_pass_a_indirect_impl(
    device const T* X_perm,
    device const uint* expert_offsets,
    device const T* W13_all,
    device const T* up_biases,
    device T* hidden_out,
    device const uint* tile_map,         // [total_tiles * 3]
    uint d_model,
    uint d_ff,
    uint E,
    float gate_clip_min,
    float gate_clip_max,
    float up_clip_min,
    float up_clip_max,
    float silu_alpha,
    uint tile_idx,                       // flat threadgroup index
    uint simd_gid,
    uint simd_lid,
    threadgroup float* tgp_memory
) {
    // Read tile descriptor
    const uint h_block_idx = tile_map[tile_idx * 3 + 0];
    const uint expert_idx = tile_map[tile_idx * 3 + 1];
    const uint row_in_expert = tile_map[tile_idx * 3 + 2];

    // Call original implementation
    moe_experts_decode_pass_a_impl<T>(
        X_perm, expert_offsets, W13_all, up_biases, hidden_out,
        d_model, d_ff, E,
        gate_clip_min, gate_clip_max,
        up_clip_min, up_clip_max, silu_alpha,
        expert_idx, row_in_expert, h_block_idx,
        simd_gid, simd_lid, tgp_memory
    );
}

#define MOE_PASS_A_INDIRECT_KERNEL(DTYPE, SUFFIX) \
kernel void moe_experts_decode_pass_a_indirect_##SUFFIX( \
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
    device const uint* tile_map [[buffer(13)]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]) \
{ \
    constexpr uint tgp_mem_size = (BN > 1) ? 2 * BN * (BLOCK_M + TM) : 1; \
    threadgroup float tgp_memory[tgp_mem_size]; \
    \
    moe_experts_decode_pass_a_indirect_impl<DTYPE>( \
        X_perm, expert_offsets, W13_all, up_biases, hidden_out, tile_map, \
        d_model, d_ff, E, gate_clip_min, gate_clip_max, \
        up_clip_min, up_clip_max, silu_alpha, \
        tgpig.x, simd_gid, simd_lid, \
        (BN > 1) ? tgp_memory : nullptr); \
}

MOE_PASS_A_INDIRECT_KERNEL(bfloat, bf16)
MOE_PASS_A_INDIRECT_KERNEL(half, f16)
MOE_PASS_A_INDIRECT_KERNEL(float, f32)


// === Pass B: compute partial down projections (split-K) ===
template<typename T>
void moe_experts_decode_down_partial_impl(
    device const T* hidden,               // [total_rows, d_ff]
    device const uint* row_expert_map,    // [total_rows] - direct row->expert lookup
    device const T* w2_all,               // [E, d_ff, d_model] - original layout
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

    // Direct lookup - no binary search!
    const uint expert_idx = row_expert_map[row_idx];

    const uint k_start = tile_idx * K_TILE;
    if (k_start >= d_ff) return;
    const uint k_chunk = min(K_TILE, d_ff - k_start);

    // Hoist address calculation outside loop for hidden (contiguous access)
    const ulong hidden_base = (ulong)row_idx * (ulong)d_ff + (ulong)k_start;
    device const T* h_ptr = hidden + hidden_base;

    // Compute W2 base: [E, d_ff, d_model] layout
    // For (expert_idx, ff_idx, my_col): index = expert_idx * d_ff * d_model + ff_idx * d_model + my_col
    const ulong w2_expert_base = (ulong)expert_idx * (ulong)d_ff * (ulong)d_model;
    const ulong w2_col_offset = (ulong)my_col;

    float acc = 0.0f;
    for (uint k = 0; k < k_chunk; ++k) {
        const float h_val = float(h_ptr[k]);
        const uint ff_idx = k_start + k;
        const ulong w_idx = w2_expert_base + (ulong)ff_idx * (ulong)d_model + w2_col_offset;
        const float w_val = float(w2_all[w_idx]);
        acc = fma(h_val, w_val, acc);
    }

    const ulong partial_idx = ((ulong)tile_idx * (ulong)total_rows + (ulong)row_idx) * (ulong)d_model + (ulong)my_col;
    partial_out[partial_idx] = acc;
}

#define MOE_PASS_B_PARTIAL_KERNEL(DTYPE, SUFFIX) \
kernel void moe_experts_decode_down_partial_##SUFFIX( \
    device const DTYPE* hidden [[buffer(0)]], \
    device const uint* row_expert_map [[buffer(1)]], \
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
        hidden, row_expert_map, w2_all, partial_out, \
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
    device const uint* row_expert_map,    // [total_rows] - direct row->expert lookup
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

    // Partial layout: [num_tiles_k, total_rows, d_model]
    // Stride between tiles for same (row, col) is (total_rows * d_model)
    const ulong stride = (ulong)total_rows * (ulong)d_model;
    const ulong row_col_offset = (ulong)row_idx * (ulong)d_model + (ulong)my_col;

    float sum = 0.0f;
    for (uint tile = 0; tile < num_tiles_k; ++tile) {
        const ulong partial_idx = (ulong)tile * stride + row_col_offset;
        sum += partial_in[partial_idx];
    }

    // Direct lookup - no binary search!
    const uint expert_idx = row_expert_map[row_idx];
    const ulong bias_idx = (ulong)expert_idx * (ulong)d_model_stride + (ulong)my_col;
    sum += float(down_biases[bias_idx]);

    const ulong out_idx = (ulong)row_idx * (ulong)d_model_stride + (ulong)my_col;
    y_out[out_idx] = T(sum);
}

#define MOE_PASS_B_REDUCE_KERNEL(DTYPE, SUFFIX) \
kernel void moe_experts_decode_down_reduce_##SUFFIX( \
    device const float* partial_in [[buffer(0)]], \
    device const uint* row_expert_map [[buffer(1)]], \
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
        partial_in, row_expert_map, down_biases, y_out, \
        total_rows, d_model, num_tiles_k, d_model, E, \
        tgpig, lid); \
}

MOE_PASS_B_REDUCE_KERNEL(bfloat, bf16)
MOE_PASS_B_REDUCE_KERNEL(half, f16)
MOE_PASS_B_REDUCE_KERNEL(float, f32)

// === Decode-specific Pass B: Simdgroup cooperation along K for coalescing ===
// W2 layout [E, d_model, d_ff] - 32 threads cooperate on one output, reading consecutive K elements
// Exploits transposed layout for stride-1 memory access across threads

template<typename T>
void moe_experts_decode_down_fused_2d_impl(
    device const T* hidden,               // [total_rows, d_ff]
    device const uint* row_expert_map,    // [total_rows] - direct row->expert lookup
    device const T* w2_all,               // [E, d_model, d_ff] - TRANSPOSED layout
    device const T* down_biases,          // [E, d_model]
    device T* y_out,                      // [total_rows, d_model]
    uint total_rows,
    uint d_model,
    uint d_ff,
    uint E,
    uint2 tgpig,
    uint simd_gid,
    uint simd_lid
) {
    // Thin kernel: 32 threads per output (1 simdgroup), 8 simdgroups per TG = 8 outputs/TG
    constexpr uint THREADS_PER_SIMD = 32;
    constexpr uint SIMDGROUPS_PER_TG = 8;

    const uint row_idx = tgpig.y;
    if (row_idx >= total_rows) return;

    // Each simdgroup computes one output column
    const uint my_col = tgpig.x * SIMDGROUPS_PER_TG + simd_gid;
    if (my_col >= d_model) return;

    const uint expert_idx = row_expert_map[row_idx];

    // Base addresses for this output column
    const ulong hidden_base = (ulong)row_idx * (ulong)d_ff;
    const ulong w2_col_base = (ulong)expert_idx * (ulong)d_model * (ulong)d_ff + (ulong)my_col * (ulong)d_ff;

    // Each thread in simdgroup handles every 32nd element along K
    // With transposed W2, simd_lid gives consecutive memory locations → perfect coalescing!
    float acc = 0.0f;

    // Main loop: stride-32 with vec8 loads for ILP
    const uint k_iters = d_ff / THREADS_PER_SIMD;
    const uint k_vec_iters = k_iters / 8;

    #pragma clang loop unroll_count(4)
    for (uint iter = 0; iter < k_vec_iters; ++iter) {
        const uint k_base = iter * (8 * THREADS_PER_SIMD) + simd_lid;

        // Load 8 values with stride-32
        float h0 = float(hidden[hidden_base + k_base + 0 * THREADS_PER_SIMD]);
        float h1 = float(hidden[hidden_base + k_base + 1 * THREADS_PER_SIMD]);
        float h2 = float(hidden[hidden_base + k_base + 2 * THREADS_PER_SIMD]);
        float h3 = float(hidden[hidden_base + k_base + 3 * THREADS_PER_SIMD]);
        float h4 = float(hidden[hidden_base + k_base + 4 * THREADS_PER_SIMD]);
        float h5 = float(hidden[hidden_base + k_base + 5 * THREADS_PER_SIMD]);
        float h6 = float(hidden[hidden_base + k_base + 6 * THREADS_PER_SIMD]);
        float h7 = float(hidden[hidden_base + k_base + 7 * THREADS_PER_SIMD]);

        // W2 loads: COALESCED - consecutive threads read consecutive addresses!
        float w0 = float(w2_all[w2_col_base + k_base + 0 * THREADS_PER_SIMD]);
        float w1 = float(w2_all[w2_col_base + k_base + 1 * THREADS_PER_SIMD]);
        float w2 = float(w2_all[w2_col_base + k_base + 2 * THREADS_PER_SIMD]);
        float w3 = float(w2_all[w2_col_base + k_base + 3 * THREADS_PER_SIMD]);
        float w4 = float(w2_all[w2_col_base + k_base + 4 * THREADS_PER_SIMD]);
        float w5 = float(w2_all[w2_col_base + k_base + 5 * THREADS_PER_SIMD]);
        float w6 = float(w2_all[w2_col_base + k_base + 6 * THREADS_PER_SIMD]);
        float w7 = float(w2_all[w2_col_base + k_base + 7 * THREADS_PER_SIMD]);

        acc = fma(h0, w0, acc);
        acc = fma(h1, w1, acc);
        acc = fma(h2, w2, acc);
        acc = fma(h3, w3, acc);
        acc = fma(h4, w4, acc);
        acc = fma(h5, w5, acc);
        acc = fma(h6, w6, acc);
        acc = fma(h7, w7, acc);
    }

    // Handle remaining full iterations
    for (uint iter = k_vec_iters * 8; iter < k_iters; ++iter) {
        const uint k = iter * THREADS_PER_SIMD + simd_lid;
        acc = fma(float(hidden[hidden_base + k]), float(w2_all[w2_col_base + k]), acc);
    }

    // Handle leftover elements (d_ff % 32)
    const uint leftover_start = k_iters * THREADS_PER_SIMD;
    if (leftover_start + simd_lid < d_ff) {
        const uint k = leftover_start + simd_lid;
        acc = fma(float(hidden[hidden_base + k]), float(w2_all[w2_col_base + k]), acc);
    }

    // Simdgroup reduction
    float result = simd_sum(acc);

    // Lane 0 writes result
    if (simd_lid == 0) {
        const ulong bias_idx = (ulong)expert_idx * (ulong)d_model + (ulong)my_col;
        result += float(down_biases[bias_idx]);

        const ulong out_idx = (ulong)row_idx * (ulong)d_model + (ulong)my_col;
        y_out[out_idx] = T(result);
    }
}

#define MOE_PASS_B_FUSED_2D_KERNEL(DTYPE, SUFFIX) \
kernel void moe_experts_decode_down_fused_2d_##SUFFIX( \
    device const DTYPE* hidden [[buffer(0)]], \
    device const uint* row_expert_map [[buffer(1)]], \
    device const DTYPE* w2_all [[buffer(2)]], \
    device const DTYPE* down_biases [[buffer(3)]], \
    device DTYPE* y_out [[buffer(4)]], \
    constant uint& total_rows [[buffer(5)]], \
    constant uint& d_model [[buffer(6)]], \
    constant uint& d_ff [[buffer(7)]], \
    constant uint& E [[buffer(8)]], \
    uint2 tgpig [[threadgroup_position_in_grid]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]) \
{ \
    moe_experts_decode_down_fused_2d_impl<DTYPE>( \
        hidden, row_expert_map, w2_all, down_biases, y_out, \
        total_rows, d_model, d_ff, E, \
        tgpig, simd_gid, simd_lid); \
}

MOE_PASS_B_FUSED_2D_KERNEL(bfloat, bf16)
MOE_PASS_B_FUSED_2D_KERNEL(half, f16)
MOE_PASS_B_FUSED_2D_KERNEL(float, f32)
