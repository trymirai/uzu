#include <metal_stdlib>
#include "activation.h"

using namespace uzu::activation;

template <typename T>
inline T activate_silu(T x) {
  float xf = float(x);
  float y = 1.0f / (1.0f + fast::exp(-fabs(xf)));
  float out = (xf < 0.0f) ? (1.0f - y) * xf : y * xf;
  return static_cast<T>(out);
}

template <typename T>
inline T activate_gelu(T x) {
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
      return activate_silu(x);
    case GELU:
      return activate_gelu(x);
    case IDENTITY:
      return x;
  }
}
