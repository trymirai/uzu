#include <metal_stdlib>

enum ActivationType : uint {
  ACT_SILU = 0,
  ACT_GELU = 1,
  ACT_TANH = 2,
  ACT_LEAKY_RELU = 3,
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
inline T tanh_activation(T x) {
  float xf = float(x);
  return T(tanh(xf));
}

template <typename T>
inline T leaky_relu_activation(
    T x,
    float alpha
) {
  float xf = float(x);
  float yf = (xf >= 0.0f) ? xf : (alpha * xf);
  return T(yf);
}

template <typename T>
inline T activate(
    T x,
    uint act,
    float alpha
) {
  if (act == ACT_SILU)
    return silu(x);
  if (act == ACT_TANH)
    return tanh_activation(x);
  if (act == ACT_LEAKY_RELU)
    return leaky_relu_activation(x, alpha);
  // Default remains GELU for compatibility.
  return gelu_approx(x);
}

template <typename T>
inline T activate(T x, uint act) {
  return activate(x, act, 0.0f);
}
