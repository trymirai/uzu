#include <metal_stdlib>
#include "../common/dsl.h"
#include "attention_gate.h"

template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(SigmoidGate)(
    const device T* gate,
    device T* output,
    const constant uint& gate_dim,
    const constant uint& batch_dim,
    const constant uint& gate_row_stride,
    const uint gate_idx AXIS(gate_dim, 256),
    const uint batch_idx AXIS(batch_dim, 1)
) {
  const uint output_idx = batch_idx * gate_dim + gate_idx;
  output[output_idx] = attention_apply_gate(output[output_idx], gate[batch_idx * gate_row_stride + gate_idx]);
}
