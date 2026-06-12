#pragma once

#include <metal_stdlib>
#include "activation_type.h"

using namespace uzu::activation_type;

template <typename T>
inline T activate_silu_alpha(T x, float alpha) {
  float xf = float(x);
  float y = 1.0f / (1.0f + fast::exp(-fabs(xf) * alpha));
  float out = (xf < 0.0f) ? (1.0f - y) * xf : y * xf;
  return static_cast<T>(out);
}

template <typename T>
inline T activate_silu(T x) {
  return activate_silu_alpha(x, 1.0);
}

template <typename T>
inline T activate_gelu(T x) {
  constexpr float k0 = 0.044715f;
  constexpr float k1 = 0.7978845608f; // sqrt(2/pi)
  float xf = float(x);
  float yf = 0.5f * xf * (1.0f + metal::precise::tanh(k1 * (xf + k0 * xf * xf * xf)));
  return T(yf);
}

inline float erf_precise(float a) {
  // Faithfully rounded erff polynomial from Norbert Juffa, also used by MLX.
  float t = metal::abs(a);
  float s = a * a;

  if (t > 0.927734375f) {
    float r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);
    float u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);
    r = metal::fma(r, s, u);
    r = metal::fma(r, t, -1.06777877e-1f);
    r = metal::fma(r, t, -6.34846687e-1f);
    r = metal::fma(r, t, -1.28717512e-1f);
    r = metal::fma(r, t, -t);
    r = 1.0f - metal::precise::exp(r);
    return metal::copysign(r, a);
  }

  float r = -5.96761703e-4f;
  r = metal::fma(r, s, 4.99119423e-3f);
  r = metal::fma(r, s, -2.67681349e-2f);
  r = metal::fma(r, s, 1.12819925e-1f);
  r = metal::fma(r, s, -3.76125336e-1f);
  r = metal::fma(r, s, 1.28379166e-1f);
  return metal::fma(r, a, a);
}

template <typename T>
inline T activate_gelu_exact(T x) {
  constexpr float inv_sqrt_2 = 0.7071067811865475f;
  float xf = float(x);
  float yf = 0.5f * xf * (1.0f + erf_precise(inv_sqrt_2 * xf));
  return T(yf);
}

template <typename T>
inline T activate_leaky_relu(T x, float alpha) {
  const float xf = float(x);
  const float yf = (xf >= 0.0f) ? xf : (alpha * xf);
  return static_cast<T>(yf);
}

template <typename T>
inline T activate_softplus(T x) {
  float xf = static_cast<float>(x);
  if (xf > 20.0f) {
    return x;
  }

  float result = fast::log(1.0f + fast::exp(xf));
  return static_cast<T>(result);
}

template <typename T>
inline T activate_tanh(T x) {
  return static_cast<T>(metal::tanh(float(x)));
}

template <typename T>
inline T activate(T x, ActivationType type) {
  switch (type) {
  case ActivationType::SILU:
    return activate_silu(x);
  case ActivationType::GELUApprox:
    return activate_gelu(x);
  case ActivationType::GELUExact:
    return activate_gelu_exact(x);
  case ActivationType::TANH:
    return activate_tanh(x);
  case ActivationType::IDENTITY:
    return x;
  case ActivationType::SOFTPLUS:
    return activate_softplus(x);
  }
}
