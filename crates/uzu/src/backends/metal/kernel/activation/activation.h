#include <metal_stdlib>
#include "activation_type.h"

using namespace uzu::activation_type;

template <typename T>
inline T silu(T x) {
  float xf = float(x);
  float yf = xf / (1.0f + metal::exp(-xf));
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
inline T activate(T x, ActivationType type) {
  switch (type) {
    case SILU:
      return silu(x);
    case GELU:
      return gelu_approx(x);
  }
}
