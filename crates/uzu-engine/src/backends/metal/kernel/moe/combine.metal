#include <metal_stdlib>

#include "../common/dsl.h"
#include "../common/sigmoid_gate.h"

template <typename T, uint BD>
VARIANTS(T, bfloat)
VARIANTS(BD, 256)
PUBLIC KERNEL(MoeCombine)(
    const device T* expert_outputs,
    const device float* expert_weights,
    const device T* shared_output,
    const device T* shared_gate,
    device T* output,
    const constant uint& model_dim,
    const constant uint& top_k,
    const constant uint& batch_size,
    const uint dim_idx AXIS(model_dim, BD),
    const uint token_idx AXIS(batch_size, 1)
) {
  float acc = 0.0f;
  const device float* weights_row = expert_weights + token_idx * top_k;
  const uint base_route = token_idx * top_k;
  for (uint k = 0; k < top_k; ++k) {
    const float expert_weight = weights_row[k];
    const float expert_value = float(expert_outputs[(base_route + k) * model_dim + dim_idx]);
    acc = fma(expert_weight, expert_value, acc);
  }
  acc += uzu::apply_sigmoid_gate(float(shared_output[token_idx * model_dim + dim_idx]), float(shared_gate[token_idx]));
  output[token_idx * model_dim + dim_idx] = T(acc);
}
