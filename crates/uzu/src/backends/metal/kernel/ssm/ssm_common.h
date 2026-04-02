#pragma once

#include <metal_stdlib>

template <typename T>
inline T softplus(T x) {
  float xf = float(x);
  if (xf > 20.0f) {
    return x;
  }
  return static_cast<T>(log(1.0f + fast::exp(xf)));
}
