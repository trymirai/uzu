#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/soft_cap.h"

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(LogitTransform)(
    device T* logits,
    constant uint& length,
    constant float& scale,
    constant float& soft_cap,
    const uint position AXIS(length, 256),
    const bool has_soft_cap SPECIALIZE
) {
  T value = static_cast<T>(static_cast<float>(logits[position]) * scale);
  if (has_soft_cap) {
    value = uzu::apply_soft_cap<T>(value, soft_cap);
  }
  logits[position] = value;
}
