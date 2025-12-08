#include <metal_stdlib>
#include "../definitions.metal"
using namespace metal;

enum ActivationType : ushort {
  ACT_SILU = 0,
  ACT_GELU = 1,
};

template <typename T>
inline T silu(T x) {
  float xf = float(x);
  float yf = xf / (1.0f + exp(-xf));
  return T(yf);
}

template <typename T>
inline T gelu_approx(T x) {
  constexpr float k0 = 0.044715f;
  constexpr float k1 = 0.7978845608f; // sqrt(2/pi)
  float xf = float(x);
  float yf = 0.5f * xf * (1.0f + tanh(k1 * (xf + k0 * xf * xf * xf)));
  return T(yf);
}

template <typename T>
inline T activate(T x, ushort act) {
  if (act == ACT_SILU)
    return silu(x);
  return gelu_approx(x);
}

template <typename T>
[[kernel]] void mlp_activation_mul(
    const device T* fused_up [[buffer(0)]],
    device T* hidden [[buffer(1)]],
    const constant int& H [[buffer(2)]],
    const constant int& M [[buffer(3)]],
    const constant ushort& act_type [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
  int row = tid.y;
  int j = tid.x;
  if (row >= M || j >= H)
    return;
  int base = row * (2 * H);
  T up = fused_up[base + j];
  T gate = fused_up[base + H + j];
  T g = activate(gate, act_type);
  hidden[row * H + j] = up * g;
}

// Explicit instantiations with stable host names
template [[host_name("mlp_activation_mul_f16")]] [[kernel]] void
mlp_activation_mul<half>(
    const device half* fused_up [[buffer(0)]],
    device half* hidden [[buffer(1)]],
    const constant int& H [[buffer(2)]],
    const constant int& M [[buffer(3)]],
    const constant ushort& act_type [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
);

template [[host_name("mlp_activation_mul_f32")]] [[kernel]] void
mlp_activation_mul<float>(
    const device float* fused_up [[buffer(0)]],
    device float* hidden [[buffer(1)]],
    const constant int& H [[buffer(2)]],
    const constant int& M [[buffer(3)]],
    const constant ushort& act_type [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
);

template [[host_name("mlp_activation_mul_bf16")]] [[kernel]] void
mlp_activation_mul<bfloat>(
    const device bfloat* fused_up [[buffer(0)]],
    device bfloat* hidden [[buffer(1)]],
    const constant int& H [[buffer(2)]],
    const constant int& M [[buffer(3)]],
    const constant ushort& act_type [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
);
