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

// === Pass A: compute activated hidden states and store to hidden buffer ===
template<typename T>
void moe_experts_decode_pass_a_impl(
    device const T* X_perm,
    device const uint* expert_offsets,
    device const T* W13_all,
    device const T* up_biases,
    device T* hidden_out,
    uint d_model,
    uint d_ff,
    uint E,
    float gate_clip_min,
    float gate_clip_max,
    float up_clip_min,
    float up_clip_max,
    float silu_alpha,
    uint2 tgpig,
    uint lid,
    threadgroup float4* x_cache
) {
    constexpr uint THREADS_PER_TG = 256;

    const uint expert_idx = tgpig.y;
    if (expert_idx >= E) return;

    const uint seg_start = expert_offsets[expert_idx];
    const uint seg_end = expert_offsets[expert_idx + 1];
    const uint seg_len = (seg_end > seg_start) ? (seg_end - seg_start) : 0u;
    if (seg_len == 0u) return;

    const ulong w13_expert_base = (ulong)expert_idx * (ulong)d_model * (ulong)(2u * d_ff);
    const ulong bias_base = (ulong)expert_idx * (ulong)(2u * d_ff);

    const uint num_vec4 = d_model / 4u;

    for (uint row_idx = 0; row_idx < seg_len; ++row_idx) {
        const ulong x_row_offset = (ulong)(seg_start + row_idx) * (ulong)d_model;
        for (uint i = lid; i < num_vec4; i += THREADS_PER_TG) {
            x_cache[i] = float4(reinterpret_cast<const device typename metal::vec<T,4>*>(X_perm + x_row_offset)[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint h_base = 0; h_base < d_ff; h_base += TILE_H) {
            const uint h_chunk = min(TILE_H, d_ff - h_base);
            for (uint h_local = lid; h_local < h_chunk; h_local += THREADS_PER_TG) {
                const uint ff_idx = h_base + h_local;

                float up_acc = 0.0f;
                float gate_acc = 0.0f;

                for (uint d4 = 0; d4 < num_vec4; ++d4) {
                    const float4 xv = x_cache[d4];
                    const ulong w_base = w13_expert_base + (ulong)(d4 * 4u) * (ulong)(2u * d_ff) + (ulong)ff_idx;
                    float4 w_up;
                    w_up.x = float(W13_all[w_base + 0u * (ulong)(2u * d_ff)]);
                    w_up.y = float(W13_all[w_base + 1u * (ulong)(2u * d_ff)]);
                    w_up.z = float(W13_all[w_base + 2u * (ulong)(2u * d_ff)]);
                    w_up.w = float(W13_all[w_base + 3u * (ulong)(2u * d_ff)]);
                    up_acc += dot(xv, w_up);
                    if (GATING_SEL > 1u) {
                        const ulong gate_base = w_base + (ulong)d_ff;
                        float4 w_gate;
                        w_gate.x = float(W13_all[gate_base + 0u * (ulong)(2u * d_ff)]);
                        w_gate.y = float(W13_all[gate_base + 1u * (ulong)(2u * d_ff)]);
                        w_gate.z = float(W13_all[gate_base + 2u * (ulong)(2u * d_ff)]);
                        w_gate.w = float(W13_all[gate_base + 3u * (ulong)(2u * d_ff)]);
                        gate_acc += dot(xv, w_gate);
                    }
                }

                up_acc += float(up_biases[bias_base + ff_idx]);
                up_acc = clamp(up_acc, up_clip_min, up_clip_max);

                float activated_val;
                if (GATING_SEL <= 1u) {
                    activated_val = (GATING_SEL == 0u) ? gelu_approx(up_acc) : silu(up_acc, silu_alpha);
                } else {
                    gate_acc += float(up_biases[bias_base + d_ff + ff_idx]);
                    gate_acc = clamp(gate_acc, gate_clip_min, gate_clip_max);
                    const float gate_val = (GATING_SEL == 2u) ? silu(gate_acc, silu_alpha) : gelu_approx(gate_acc);
                    activated_val = gate_val * up_acc;
                }

                const ulong hidden_idx = (ulong)(seg_start + row_idx) * (ulong)d_ff + (ulong)ff_idx;
                hidden_out[hidden_idx] = T(activated_val);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
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
    uint2 tgpig [[threadgroup_position_in_grid]], \
    uint lid [[thread_index_in_threadgroup]]) \
{ \
    threadgroup float4 x_cache_local[1024]; \
    moe_experts_decode_pass_a_impl<DTYPE>( \
        X_perm, expert_offsets, W13_all, up_biases, hidden_out, \
        d_model, d_ff, E, gate_clip_min, gate_clip_max, \
        up_clip_min, up_clip_max, silu_alpha, \
        tgpig, lid, x_cache_local); \
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
