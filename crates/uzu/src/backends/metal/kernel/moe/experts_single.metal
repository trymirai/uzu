// UZU MoE Experts Decode Single-Token
// Optimized path for T=1: skips scatter/gather, fuses finalize into Pass B
// Naming convention: moe_experts_decode_single_* (matches moe_experts_decode_*)

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant uint GATING_SEL [[function_constant(30)]]; // 0=GELU, 1=SiLU, 2=SwiGLU, 3=GEGLU

static inline float gelu_approx(float x) {
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    if (x > 10.0f) return x;
    if (x < -10.0f) return 0.0f;
    return 0.5f * x * (1.0f + tanh(clamp(k0 * (x + k1 * x * x * x), -10.0f, 10.0f)));
}

static inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

// ============================================================================
// Pass A: x @ W13[expert] → hidden[k]
// Each threadgroup: 4 simdgroups, outputs 4 elements (1 per simdgroup)
// Each simdgroup: 32 threads reduce d_model with float4 vectorized loads
// Grid: (ceil(d_ff/4), K)
// ============================================================================

template<typename T, typename T4>
inline void moe_experts_decode_single_pass_a_impl(
    device const T* x,
    device const int* topk_ids,
    device const T* W13_all,
    device const T* biases,
    device float* hidden_out,
    uint d_model,
    uint d_ff,
    uint K,
    uint k_slot,
    uint h_block_idx,
    uint simd_gid,
    uint simd_lid
) {
    const int expert_id = topk_ids[k_slot];
    if (expert_id < 0) return;
    const uint expert_u = uint(expert_id);

    const uint h_idx = h_block_idx * 4 + simd_gid;
    if (h_idx >= d_ff) return;

    const ulong w13_stride = (ulong)d_model * (ulong)(2 * d_ff);
    const ulong w13_base = (ulong)expert_u * w13_stride;
    const ulong bias_base = (ulong)expert_u * (ulong)(2 * d_ff);

    device const T* w_up_row = W13_all + w13_base + (ulong)h_idx * (ulong)d_model;
    device const T* w_gate_row = W13_all + w13_base + (ulong)(d_ff + h_idx) * (ulong)d_model;

    float acc_up = 0.0f;
    float acc_gate = 0.0f;

    const uint vec_iters = d_model / 128;

    for (uint i = 0; i < vec_iters; ++i) {
        uint base_idx = i * 128 + simd_lid * 4;

        T4 x_vec = *reinterpret_cast<device const T4*>(x + base_idx);
        T4 w_up_vec = *reinterpret_cast<device const T4*>(w_up_row + base_idx);

        acc_up += float(x_vec.x) * float(w_up_vec.x);
        acc_up += float(x_vec.y) * float(w_up_vec.y);
        acc_up += float(x_vec.z) * float(w_up_vec.z);
        acc_up += float(x_vec.w) * float(w_up_vec.w);

        if (GATING_SEL > 1) {
            T4 w_gate_vec = *reinterpret_cast<device const T4*>(w_gate_row + base_idx);
            acc_gate += float(x_vec.x) * float(w_gate_vec.x);
            acc_gate += float(x_vec.y) * float(w_gate_vec.y);
            acc_gate += float(x_vec.z) * float(w_gate_vec.z);
            acc_gate += float(x_vec.w) * float(w_gate_vec.w);
        }
    }

    uint leftover_start = vec_iters * 128 + simd_lid;
    for (uint idx = leftover_start; idx < d_model; idx += 32) {
        float xv = float(x[idx]);
        acc_up += xv * float(w_up_row[idx]);
        if (GATING_SEL > 1) {
            acc_gate += xv * float(w_gate_row[idx]);
        }
    }

    acc_up = simd_sum(acc_up);
    if (GATING_SEL > 1) {
        acc_gate = simd_sum(acc_gate);
    }

    if (simd_lid == 0) {
        float up_val = acc_up + float(biases[bias_base + h_idx]);

        float activated;
        if (GATING_SEL <= 1) {
            activated = (GATING_SEL == 0) ? gelu_approx(up_val) : silu(up_val);
        } else {
            float gate_val = acc_gate + float(biases[bias_base + d_ff + h_idx]);
            float gate_act = (GATING_SEL == 2) ? silu(gate_val) : gelu_approx(gate_val);
            activated = gate_act * up_val;
        }

        hidden_out[(ulong)k_slot * (ulong)d_ff + (ulong)h_idx] = activated;
    }
}

#define MOE_DECODE_SINGLE_PASS_A_KERNEL(DTYPE, DTYPE4, SUFFIX) \
kernel void moe_experts_decode_single_pass_a_##SUFFIX( \
    device const DTYPE* x [[buffer(0)]], \
    device const int* topk_ids [[buffer(1)]], \
    device const DTYPE* W13_all [[buffer(2)]], \
    device const DTYPE* biases [[buffer(3)]], \
    device float* hidden_out [[buffer(4)]], \
    constant uint& d_model [[buffer(5)]], \
    constant uint& d_ff [[buffer(6)]], \
    constant uint& K [[buffer(7)]], \
    uint2 tgpig [[threadgroup_position_in_grid]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]) \
{ \
    moe_experts_decode_single_pass_a_impl<DTYPE, DTYPE4>( \
        x, topk_ids, W13_all, biases, hidden_out, \
        d_model, d_ff, K, tgpig.y, tgpig.x, simd_gid, simd_lid); \
}

MOE_DECODE_SINGLE_PASS_A_KERNEL(half, half4, f16)
MOE_DECODE_SINGLE_PASS_A_KERNEL(bfloat, bfloat4, bf16)
MOE_DECODE_SINGLE_PASS_A_KERNEL(float, float4, f32)


// ============================================================================
// Pass B (fused with finalize): hidden[k] @ W2[expert] → y (directly)
// Computes: y[d] = Σ_k prob[k] * (hidden[k] @ W2[expert_k, d] + bias)
// Each simdgroup computes one final output element
// Grid: (ceil(d_model/8), 1)  - NOT per K!
// ============================================================================

template<typename T, typename T4>
inline void moe_experts_decode_single_pass_b_impl(
    device const float* hidden,     // [K, d_ff]
    device const int* topk_ids,     // [K]
    device const T* topk_probs,     // [K]
    device const T* W2_all,         // [E, d_model, d_ff]
    device const T* biases,         // [E, d_model]
    device T* y,                    // [d_model]
    uint d_model,
    uint d_ff,
    uint K,
    uint d_block,
    uint simd_gid,
    uint simd_lid
) {
    const uint my_col = d_block * 8 + simd_gid;
    if (my_col >= d_model) return;

    const uint vec_iters = d_ff / 128;
    const ulong w2_expert_stride = (ulong)d_model * (ulong)d_ff;

    float final_acc = 0.0f;

    // Loop over K experts
    for (uint k = 0; k < K; ++k) {
        const uint expert_u = uint(topk_ids[k]);
        const float prob = float(topk_probs[k]);

        device const float* hidden_ptr = hidden + (ulong)k * (ulong)d_ff;
        device const T* w2_ptr = W2_all + (ulong)expert_u * w2_expert_stride + (ulong)my_col * (ulong)d_ff;

        float acc = 0.0f;

        // Vectorized reduction
        for (uint i = 0; i < vec_iters; ++i) {
            uint base_idx = i * 128 + simd_lid * 4;

            float4 h_vec = *reinterpret_cast<device const float4*>(hidden_ptr + base_idx);
            T4 w_vec = *reinterpret_cast<device const T4*>(w2_ptr + base_idx);

            acc += h_vec.x * float(w_vec.x);
            acc += h_vec.y * float(w_vec.y);
            acc += h_vec.z * float(w_vec.z);
            acc += h_vec.w * float(w_vec.w);
        }

        // Remainder
        for (uint idx = vec_iters * 128 + simd_lid; idx < d_ff; idx += 32) {
            acc += hidden_ptr[idx] * float(w2_ptr[idx]);
        }

        // Simdgroup reduction and accumulate
        float result = simd_sum(acc);
        if (simd_lid == 0) {
            result += float(biases[(ulong)expert_u * (ulong)d_model + (ulong)my_col]);
            final_acc += prob * result;
        }
    }

    if (simd_lid == 0) {
        y[my_col] = T(final_acc);
    }
}

#define MOE_DECODE_SINGLE_PASS_B_KERNEL(DTYPE, DTYPE4, SUFFIX) \
kernel void moe_experts_decode_single_pass_b_##SUFFIX( \
    device const float* hidden [[buffer(0)]], \
    device const int* topk_ids [[buffer(1)]], \
    device const DTYPE* topk_probs [[buffer(2)]], \
    device const DTYPE* W2_all [[buffer(3)]], \
    device const DTYPE* biases [[buffer(4)]], \
    device DTYPE* y [[buffer(5)]], \
    constant uint& d_model [[buffer(6)]], \
    constant uint& d_ff [[buffer(7)]], \
    constant uint& K [[buffer(8)]], \
    uint tgpig [[threadgroup_position_in_grid]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]) \
{ \
    moe_experts_decode_single_pass_b_impl<DTYPE, DTYPE4>( \
        hidden, topk_ids, topk_probs, W2_all, biases, y, \
        d_model, d_ff, K, tgpig, simd_gid, simd_lid); \
}

MOE_DECODE_SINGLE_PASS_B_KERNEL(half, half4, f16)
MOE_DECODE_SINGLE_PASS_B_KERNEL(bfloat, bfloat4, bf16)
MOE_DECODE_SINGLE_PASS_B_KERNEL(float, float4, f32)
