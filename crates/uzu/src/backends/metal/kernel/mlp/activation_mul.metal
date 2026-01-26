#include <metal_stdlib>
#include "../definitions.metal"

// TODO: here is the same code as in "activation.metal", maybe move to separate file?

enum ActivationType : uint {
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
  float t = k1 * (xf + k0 * xf * xf * xf);
  float yf = 0.5f * xf * (1.0f + metal::precise::tanh(t));
  return T(yf);
}

template <typename T>
inline T activate(T x, uint act) {
  if (act == ACT_SILU)
    return silu(x);
  return gelu_approx(x);
}

SPECIALIZE(T, float, half, bfloat) KERNEL(MlpGateActMul) (
    const device T* fused_up,
    device T* hidden,
    const constant int& h,
    const constant int& m,
    const constant uint& act_type,
    uint j AXIS(h, 64),
    uint row AXIS(m, 1)
) {
  if (row >= m || j >= h) {
    return;
  }

  int base = row * (2 * h);
  T up = fused_up[base + j];
  T gate = fused_up[base + h + j];
  T g = activate(gate, act_type);
  float out_f = float(up) * float(g);
  hidden[row * h + j] = T(out_f);
}