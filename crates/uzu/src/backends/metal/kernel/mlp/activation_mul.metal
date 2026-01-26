#include <metal_stdlib>
#include "../definitions.metal"

#include "../activation/activation.h"

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
