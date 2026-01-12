#pragma once

#include <metal_stdlib>
using namespace metal;

// MLP Fusion Function Constants
// These control whether the kernel computes paired up/gate projections
// and applies activation fusion (SwiGLU/GeGLU style).
//
// Index allocation:
//   50 - MLP_FUSED (bool): enable paired computation + fused activation
//   51 - MLP_HIDDEN_DIM (uint): offset to gate weights (= hidden_dim)
//   52 - MLP_ACTIVATION (uint): 0=SiLU, 1=GELU

constant bool MLP_FUSED [[function_constant(50)]];
constant uint MLP_HIDDEN_DIM [[function_constant(51)]];
constant uint MLP_ACTIVATION [[function_constant(52)]];

// Activation type enum for readability
enum MlpActivationType : uint {
    MLP_ACT_SILU = 0,
    MLP_ACT_GELU = 1,
};

// SiLU activation: x * sigmoid(x)
template <typename T>
inline T mlp_silu(T x) {
    float xf = float(x);
    float result = xf / (1.0f + exp(-xf));
    return T(result);
}

// GELU approximation (tanh-based)
template <typename T>
inline T mlp_gelu(T x) {
    constexpr float k0 = 0.044715f;
    constexpr float k1 = 0.7978845608f; // sqrt(2/pi)
    float xf = float(x);
    float t = k1 * (xf + k0 * xf * xf * xf);
    float result = 0.5f * xf * (1.0f + metal::precise::tanh(t));
    return T(result);
}

// Apply activation based on MLP_ACTIVATION constant
template <typename T>
inline T mlp_activate(T x) {
    if (MLP_ACTIVATION == MLP_ACT_SILU) {
        return mlp_silu(x);
    }
    return mlp_gelu(x);
}

// Fused MLP epilogue: up * activation(gate)
// Used in SwiGLU (SiLU) and GeGLU (GELU) style MLPs
template <typename T>
inline T mlp_fused_epilogue(T up, T gate) {
    T activated_gate = mlp_activate(gate);
    return T(float(up) * float(activated_gate));
}

// Fused MLP epilogue with float accumulators
inline float mlp_fused_epilogue_f32(float up, float gate) {
    float activated_gate;
    if (MLP_ACTIVATION == MLP_ACT_SILU) {
        activated_gate = up / (1.0f + exp(-gate));
        // Note: SwiGLU is gate_act * up, not up_act * gate
        // Correcting: activated_gate = gate / (1.0f + exp(-gate))
        activated_gate = gate / (1.0f + exp(-gate));
    } else {
        constexpr float k0 = 0.044715f;
        constexpr float k1 = 0.7978845608f;
        float t = k1 * (gate + k0 * gate * gate * gate);
        activated_gate = 0.5f * gate * (1.0f + metal::precise::tanh(t));
    }
    return up * activated_gate;
}
