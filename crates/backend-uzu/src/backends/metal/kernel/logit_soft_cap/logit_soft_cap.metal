#include <metal_stdlib>
#include "../common/dsl.h"

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(LogitSoftCap)(
    device T* logits,
    constant uint& length,
    constant float& soft_cap,
    const uint position AXIS(length, 256)
) {
  const float value = float(logits[position]);
  logits[position] = T(fast::tanh(value / soft_cap) * soft_cap);
}
