#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(TensorFinalize)(
    device T* shortcut,
    device T* main,
    const device T* scalar OPTIONAL(has_scalar),
    constant uint& length,
    const bool has_scalar SPECIALIZE,
    const uint position AXIS(length, 256)
) {
  float scale = has_scalar ? float(scalar[0]) : 1.0f;
  shortcut[position] =
      T((float(shortcut[position]) + float(main[position])) * scale);
  main[position] = T(0.0f);
}
