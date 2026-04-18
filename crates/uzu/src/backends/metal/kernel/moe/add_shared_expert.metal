#include <metal_stdlib>
#include "../common/dsl.h"

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MoeAddSharedExpert)(
    device const T* shared_out,    // [T, d_model]
    device const T* gate_logits,   // [T, num_shared]
    device T* y,                   // [T, d_model]
    constant uint& t_count,
    constant uint& d_model,
    constant uint& num_shared,
    const uint f AXIS(d_model, 64),
    const uint t AXIS(t_count, 1)
) {
  float gw = 0.0f;
  for (uint i = 0; i < num_shared; ++i) {
    float lg = (float)gate_logits[t * num_shared + i];
    gw += 1.0f / (1.0f + exp(-lg));
  }
  const uint idx = t * d_model + f;
  float acc = (float)y[idx] + gw * (float)shared_out[idx];
  y[idx] = T(acc);
}
