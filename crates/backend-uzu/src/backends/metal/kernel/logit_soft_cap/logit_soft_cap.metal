#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/soft_cap.h"

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(LogitSoftCap)(
    device T* logits,
    constant uint& length,
    constant float& soft_cap,
    const uint position AXIS(length, 256)
) {
  logits[position] = uzu::apply_soft_cap<T>(logits[position], soft_cap);
}
