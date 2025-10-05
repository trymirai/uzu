#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Tiled two-phase GEMV for T=1 decode
// Tiles H dimension to fit in TG memory (~17KB total)

constant bool WEIGHTS_ARE_TRANSPOSED [[function_constant(28)]];
constant uint GATING_SEL [[function_constant(30)]]; // 0=GELU, 1=SiLU, 2=SwiGLU, 3=GEGLU
constant uint TILE_H [[function_constant(32)]]; // H tile size (e.g., 256)

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

template<typename T>
void moe_experts_decode_fused_gemv_v2_impl(
    device const T* X_perm,
    device const uint* expert_offsets,
    device const T* W13_all,
    device const T* W2_all,
    device T* Y_partial,
    uint d_model,
    uint d_ff,
    uint E,
    device const T* up_biases,
    device const T* down_biases,
    float gate_clip_min,
    float gate_clip_max,
    float up_clip_min,
    float up_clip_max,
    float silu_alpha,
    uint2 tgpig,
    uint lid,
    threadgroup float4* x_cache,
    threadgroup float* h_tile
) {
    constexpr uint THREADS_PER_TG = 256;

    const uint col_group = tgpig.x;
    const uint expert_idx = tgpig.y;

    if (expert_idx >= E) return;

    const uint seg_start = expert_offsets[expert_idx];
    const uint seg_end = expert_offsets[expert_idx + 1];
    const uint seg_len = (seg_end > seg_start) ? (seg_end - seg_start) : 0u;
    if (seg_len == 0) return;
    
    const uint col0 = col_group * THREADS_PER_TG;
    if (col0 >= d_model) return;
    
    const ulong w13_expert_base = (ulong)expert_idx * (ulong)d_model * (ulong)(2 * d_ff);
    const ulong w2_expert_base = (ulong)expert_idx * (ulong)d_ff * (ulong)d_model;
    const ulong bias_up_base = (ulong)expert_idx * (ulong)(2 * d_ff);
    const ulong bias_down_base = (ulong)expert_idx * (ulong)d_model;
    
    const uint num_vec4 = d_model / 4;

    // Process all rows in this expert's segment
    for (uint row_idx = 0; row_idx < seg_len; ++row_idx) {
        // Load x for this row (all threads collaborate)
        const ulong x_row_offset = (ulong)(seg_start + row_idx) * (ulong)d_model;
        for (uint i = lid; i < num_vec4; i += THREADS_PER_TG) {
            x_cache[i] = float4(reinterpret_cast<const device typename metal::vec<T,4>*>(X_perm + x_row_offset)[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulator for FC2
        float y_acc = 0.0f;

    // Tile over H dimension
    for (uint h_base = 0; h_base < d_ff; h_base += TILE_H) {
        const uint h_chunk = min(TILE_H, d_ff - h_base);
        
        // === Phase 1: FC1 for this h_tile ===
        for (uint h_local = lid; h_local < h_chunk; h_local += THREADS_PER_TG) {
            const uint ff_idx = h_base + h_local;
            
            float up_acc = 0.0f;
            float gate_acc = 0.0f;
            
            for (uint d4 = 0; d4 < num_vec4; ++d4) {
                const float4 xv = x_cache[d4];
                
                if (WEIGHTS_ARE_TRANSPOSED) {
                    const ulong w_up_base = w13_expert_base + (ulong)ff_idx * (ulong)d_model + (ulong)(d4 * 4);
                    const float4 w_up = float4(reinterpret_cast<const device typename metal::vec<T,4>*>(W13_all + w_up_base)[0]);
                    up_acc += dot(xv, w_up);

                    if (GATING_SEL > 1u) {
                        const ulong w_gate_base = w_up_base + (ulong)d_ff * (ulong)d_model;
                        const float4 w_gate = float4(reinterpret_cast<const device typename metal::vec<T,4>*>(W13_all + w_gate_base)[0]);
                        gate_acc += dot(xv, w_gate);
                    }
                } else {
                    const ulong w_base = w13_expert_base + (ulong)(d4 * 4) * (ulong)(2 * d_ff) + (ulong)ff_idx;
                    float4 w_up;
                    w_up.x = float(W13_all[w_base + 0u * (ulong)(2 * d_ff)]);
                    w_up.y = float(W13_all[w_base + 1u * (ulong)(2 * d_ff)]);
                    w_up.z = float(W13_all[w_base + 2u * (ulong)(2 * d_ff)]);
                    w_up.w = float(W13_all[w_base + 3u * (ulong)(2 * d_ff)]);
                    up_acc += dot(xv, w_up);

                    if (GATING_SEL > 1u) {
                        const ulong gate_base = w_base + (ulong)d_ff;
                        float4 w_gate;
                        w_gate.x = float(W13_all[gate_base + 0u * (ulong)(2 * d_ff)]);
                        w_gate.y = float(W13_all[gate_base + 1u * (ulong)(2 * d_ff)]);
                        w_gate.z = float(W13_all[gate_base + 2u * (ulong)(2 * d_ff)]);
                        w_gate.w = float(W13_all[gate_base + 3u * (ulong)(2 * d_ff)]);
                        gate_acc += dot(xv, w_gate);
                    }
                }
            }
            
            up_acc += float(up_biases[bias_up_base + ff_idx]);
            up_acc = clamp(up_acc, up_clip_min, up_clip_max);

            float activated_val;
            if (GATING_SEL <= 1u) {
                activated_val = (GATING_SEL == 0u) ? gelu_approx(up_acc) : silu(up_acc, silu_alpha);
            } else {
                gate_acc += float(up_biases[bias_up_base + d_ff + ff_idx]);
                gate_acc = clamp(gate_acc, gate_clip_min, gate_clip_max);
                const float swish_y = (GATING_SEL == 2u) ? silu(gate_acc, silu_alpha) : gelu_approx(gate_acc);
                activated_val = swish_y * up_acc;
            }
            
            h_tile[h_local] = activated_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Phase 2: FC2 for this h_tile ===
        const uint my_col = col0 + lid;
        if (my_col < d_model) {
            for (uint h_local = 0; h_local < h_chunk; ++h_local) {
                const uint ff_idx = h_base + h_local;
                const float h_val = h_tile[h_local];
                float w2_val;
                
                if (WEIGHTS_ARE_TRANSPOSED) {
                    const ulong w2_idx = w2_expert_base + (ulong)my_col * (ulong)d_ff + (ulong)ff_idx;
                    w2_val = float(W2_all[w2_idx]);
                } else {
                    const ulong w2_idx = w2_expert_base + (ulong)ff_idx * (ulong)d_model + (ulong)my_col;
                    w2_val = float(W2_all[w2_idx]);
                }
                y_acc = fma(h_val, w2_val, y_acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

        // Write FC2 result first (match tiled MMA's two-stage quantization)
        const uint my_col = col0 + lid;
        if (my_col < d_model) {
            const ulong y_idx = (ulong)(seg_start + row_idx) * (ulong)d_model + (ulong)my_col;
            Y_partial[y_idx] = T(y_acc);  // First BF16 conversion
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Read back, add bias, write (second BF16 conversion - matches tiled MMA)
        if (my_col < d_model) {
            const ulong y_idx = (ulong)(seg_start + row_idx) * (ulong)d_model + (ulong)my_col;
            float prev_val = float(Y_partial[y_idx]);  // BF16 → F32
            float bias_val = float(down_biases[bias_down_base + my_col]);
            float final_val = prev_val + bias_val;
            Y_partial[y_idx] = T(final_val);  // F32 → BF16
        }
    } // End row loop
}

// Kernel entry points
#define MOE_V2_KERNEL(DTYPE, SUFFIX) \
kernel void moe_experts_decode_fused_gemv_v2_##SUFFIX( \
    device const DTYPE* X_perm [[buffer(0)]], \
    device const uint* expert_offsets [[buffer(1)]], \
    device const DTYPE* W13_all [[buffer(2)]], \
    device const DTYPE* W2_all [[buffer(3)]], \
    device DTYPE* Y_partial [[buffer(4)]], \
    constant uint& T [[buffer(5)]], \
    constant uint& d_model [[buffer(6)]], \
    constant uint& d_ff [[buffer(7)]], \
    constant uint& E [[buffer(8)]], \
    device const DTYPE* up_biases [[buffer(9)]], \
    device const DTYPE* down_biases [[buffer(10)]], \
    constant float& gate_clip_min [[buffer(11)]], \
    constant float& gate_clip_max [[buffer(12)]], \
    constant float& up_clip_min [[buffer(13)]], \
    constant float& up_clip_max [[buffer(14)]], \
    constant float& silu_alpha [[buffer(15)]], \
    uint2 tgpig [[threadgroup_position_in_grid]], \
    uint lid [[thread_index_in_threadgroup]]) \
{ \
    threadgroup float4 x_cache_local[1024]; \
    threadgroup float h_tile_local[512]; \
    moe_experts_decode_fused_gemv_v2_impl<DTYPE>( \
        X_perm, expert_offsets, W13_all, W2_all, Y_partial, \
        d_model, d_ff, E, up_biases, down_biases, \
        gate_clip_min, gate_clip_max, up_clip_min, up_clip_max, silu_alpha, \
        tgpig, lid, x_cache_local, h_tile_local); \
}

MOE_V2_KERNEL(bfloat, bf16)
MOE_V2_KERNEL(half, f16)
MOE_V2_KERNEL(float, f32)