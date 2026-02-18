#include <metal_stdlib>

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
  float yf =
      0.5f * xf * (1.0f + metal::precise::tanh(k1 * (xf + k0 * xf * xf * xf)));
  return T(yf);
}

template <typename T>
inline T activate(T x, uint act) {
  if (act == ACT_SILU)
    return silu(x);
  return gelu_approx(x);
}
