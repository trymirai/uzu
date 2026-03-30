#ifndef __MOE_COMMONS_H__
#define __MOE_COMMONS_H__

#include <metal_stdlib>

static inline float gelu_approx(float x) {
  if (x > 10.0f) {
    return x;
  }
  if (x < -10.0f) {
    return 0.0f;
  }

  const float GELU_APPROX_K0 = 0.7978845608f; // sqrt(2/pi)
  const float GELU_APPROX_K1 = 0.044715f;
  float t = metal::clamp(
      GELU_APPROX_K0 * (x + GELU_APPROX_K1 * x * x * x),
      -10.0f,
      10.0f
  );
  return 0.5f * x * (1.0f + metal::tanh(t));
}

#endif // __MOE_COMMONS_H__