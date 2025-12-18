#include <metal_stdlib>
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
[[kernel]] void activation(
    const device T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    const constant int& N [[buffer(2)]],
    const constant ushort& act_type [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
  if (int(tid) >= N)
    return;
  output[tid] = activate(input[tid], act_type);
}

// Explicit instantiations
template [[host_name("activation_f16")]] [[kernel]] void
activation<half>(
    const device half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    const constant int& N [[buffer(2)]],
    const constant ushort& act_type [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
);

template [[host_name("activation_f32")]] [[kernel]] void
activation<float>(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const constant int& N [[buffer(2)]],
    const constant ushort& act_type [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
);

template [[host_name("activation_bf16")]] [[kernel]] void
activation<bfloat>(
    const device bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    const constant int& N [[buffer(2)]],
    const constant ushort& act_type [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
);
