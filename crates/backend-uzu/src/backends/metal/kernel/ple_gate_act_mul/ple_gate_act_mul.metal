#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/dsl.h"

using namespace uzu::activation_type;

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(PleGateActMul)(
    const device T* gate_out,
    const device T* per_layer_input,
    device T* output,
    const constant int& ple_dim,
    const constant int& batch_dim,
    const constant int& num_layers,
    const constant int& layer_offset,
    const constant ActivationType& act_type,
    uint col AXIS(ple_dim, 64),
    uint row AXIS(batch_dim, 1)
) {
  const uint layer_stride = static_cast<uint>(num_layers * ple_dim);
  const uint gate_index = row * static_cast<uint>(ple_dim) + col;
  const uint input_index =
      row * layer_stride + static_cast<uint>(layer_offset) + col;
  T gate_value = gate_out[gate_index];
  T activated = activate(gate_value, act_type);
  T input_value = per_layer_input[input_index];
  output[gate_index] = static_cast<T>(
      static_cast<float>(activated) * static_cast<float>(input_value)
  );
}
