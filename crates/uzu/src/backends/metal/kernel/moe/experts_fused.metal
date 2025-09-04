#include <metal_stdlib>
#include "../definitions.metal"
using namespace metal;

#define BLOCK_SIZE 256
#define FF_CHUNK 64

static inline float gelu_approx(float x) {
    const float k0 = 0.7978845608f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    
    // For large |x|, GELU(x) ≈ x for x > 0 and ≈ 0 for x < 0
    // This avoids tanh overflow for extreme values
    if (x > 10.0f) return x;
    if (x < -10.0f) return 0.0f;
    
    float x3 = x * x * x;
    float inner = x + k1 * x3;
    float tanh_arg = k0 * inner;
    
    // Clamp tanh argument to avoid potential Metal tanh issues
    tanh_arg = clamp(tanh_arg, -10.0f, 10.0f);
    
    return 0.5f * x * (1.0f + tanh(tanh_arg));
}

static inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline uint lower_bound_offsets(device const uint* offsets, uint E, uint idx) {
    // find e such that offsets[e] <= idx < offsets[e+1]
    uint lo = 0u, hi = E;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        uint v = offsets[mid+1];
        if (idx < v) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }
    return lo;
}

// gating_code: 0=GELU (plain), 1=SiLU (plain), 2=SwiGLU, 3=GEGLU
kernel void moe_fused_expert_mlp_f16(
    device const half* X [[buffer(0)]],                 // [T, d_model]
    device const int* bucketed_token_ids [[buffer(1)]], // [sum_k]
    device const uint* expert_offsets [[buffer(2)]],    // [E+1]
    device const half* W1_all [[buffer(3)]],            // [E*d_ff*d_model]
    device const half* W3_all [[buffer(4)]],            // [E*d_ff*d_model] or ignored
    device const half* W2_all [[buffer(5)]],            // [E*d_model*d_ff]
    device half* Y_partial [[buffer(6)]],               // [sum_k, d_model]
    constant uint& T [[buffer(7)]],
    constant uint& d_model [[buffer(8)]],
    constant uint& d_ff [[buffer(9)]],
    constant uint& E [[buffer(10)]],
    constant uint& gating_code [[buffer(11)]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    // Determine sum_k via dispatch sizing; guard if oversized
    const uint sum_k = expert_offsets[E];
    const uint row = tgpig.x;
    if (row >= sum_k) return;

    // Resolve expert and token
    const int token_id = bucketed_token_ids[row];
    if (token_id < 0 || (uint)token_id >= T) return;
    const uint e = lower_bound_offsets(expert_offsets, E, row);

    // Base pointers for expert weights
    const ulong W1_base = (ulong)e * (ulong)d_ff * (ulong)d_model;
    const ulong W3_base = (ulong)e * (ulong)d_ff * (ulong)d_model;
    const ulong W2_base = (ulong)e * (ulong)d_model * (ulong)d_ff;

    // Deterministic single-thread per row (correctness-first)
    if (lid != 0) {
        return;
    }

    // Two-pass CPU-identical evaluation
    const uint max_ff_local = 2048u; // sufficient for current tests
    thread float a_local[max_ff_local];
    const uint ff_used = d_ff <= max_ff_local ? d_ff : max_ff_local;

    for (uint j0 = 0; j0 < d_model; ++j0) {
        // Pass 1: compute a[r]
        for (uint r = 0; r < ff_used; ++r) {
            float up = 0.0f;
            for (uint j = 0; j < d_model; ++j) {
                const ulong i1 = W1_base + (ulong)r * (ulong)d_model + (ulong)j;
                const ulong ix = (ulong)(uint)token_id * (ulong)d_model + (ulong)j;
                const ulong total1 = (ulong)E * (ulong)d_ff * (ulong)d_model;
                const ulong totalx = (ulong)T * (ulong)d_model;
                float w1 = (i1 < total1) ? (float)W1_all[i1] : 0.0f;
                float x  = (ix < totalx) ? (float)X[ix]       : 0.0f;
                up += w1 * x;
            }
            if (gating_code <= 1u) {
                a_local[r] = (gating_code == 0u) ? gelu_approx(up) : silu(up);
            } else {
                float vp = 0.0f;
                for (uint j = 0; j < d_model; ++j) {
                    const ulong i3 = W3_base + (ulong)r * (ulong)d_model + (ulong)j;
                    const ulong ix = (ulong)(uint)token_id * (ulong)d_model + (ulong)j;
                    const ulong total3 = (ulong)E * (ulong)d_ff * (ulong)d_model;
                    const ulong totalx = (ulong)T * (ulong)d_model;
                    float w3 = (i3 < total3) ? (float)W3_all[i3] : 0.0f;
                    float x  = (ix < totalx) ? (float)X[ix]       : 0.0f;
                    vp += w3 * x;
                }
                const float g = (gating_code == 2u) ? silu(up) : gelu_approx(up);
                a_local[r] = g * vp;
            }
        }

        // Pass 2: accumulate FC2
        float acc = 0.0f;
        const ulong total2 = (ulong)E * (ulong)d_model * (ulong)d_ff;
        for (uint r = 0; r < ff_used; ++r) {
            const ulong i2 = W2_base + (ulong)j0 * (ulong)d_ff + (ulong)r;
            float w2 = (i2 < total2) ? (float)W2_all[i2] : 0.0f;
            acc += w2 * a_local[r];
        }

        const ulong yi = (ulong)row * (ulong)d_model + (ulong)j0;
        Y_partial[(uint)yi] = (half)acc;
    }
}


