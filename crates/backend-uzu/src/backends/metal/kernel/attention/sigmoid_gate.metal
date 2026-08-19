#include <metal_stdlib>
#include "../common/dsl.h"

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(SigmoidGate)(
    const device T* gate,
    device T* output,
    const constant uint& gate_dim,
    const constant uint& batch_dim,
    const constant uint& gate_row_stride,
    const uint gate_idx AXIS(gate_dim, 256),
    const uint batch_idx AXIS(batch_dim, 1)
) {
  if (gate_idx >= gate_dim)
    return;
  const uint output_idx = batch_idx * gate_dim + gate_idx;
  float sigmoid = 1.0f / (1.0f + exp(-float(gate[batch_idx * gate_row_stride + gate_idx])));
  output[output_idx] = T(float(output[output_idx]) * sigmoid);
}
