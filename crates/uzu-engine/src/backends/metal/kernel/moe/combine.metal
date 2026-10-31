#include <metal_stdlib>

#include "../attention/attention_gate.h"
#include "../common/dsl.h"

template <typename T, uint BD>
VARIANTS(T, bfloat)
VARIANTS(BD, 256)
PUBLIC KERNEL(MoeCombine)(
    const device T* expert_outputs,
    const device float* expert_weights,
    const device T* shared_output OPTIONAL(has_shared),
    const device T* shared_gate OPTIONAL(has_gate),
    device T* output,
    const constant uint& model_dim,
    const constant uint& top_k,
    const constant uint& batch_size,
    const bool has_shared SPECIALIZE,
    const bool has_gate SPECIALIZE,
    const uint col AXIS(model_dim, BD),
    const uint token_idx AXIS(batch_size, 1)
) {
  float acc = 0.0f;
  const device float* weights_row = expert_weights + token_idx * top_k;
  const uint base_route = token_idx * top_k;
  for (uint k = 0; k < top_k; ++k) {
    const float w = weights_row[k];
    const float val = float(expert_outputs[(base_route + k) * model_dim + col]);
    acc = fma(w, val, acc);
  }
  if (has_shared) {
    float s_val = float(shared_output[token_idx * model_dim + col]);
    if (has_gate) {
      s_val = attention_apply_gate(s_val, float(shared_gate[token_idx]));
    }
    acc += s_val;
  }
  output[token_idx * model_dim + col] = T(acc);
}
