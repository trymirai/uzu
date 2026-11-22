#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "expert_common.h"

using namespace metal;
using namespace uzu::moe;

#define COMBINE_THREADS 256

// output[token] = sum over the token's top_k routes of expert_outputs[token * top_k + pick], in float32.
// Requires model_dim % 4 == 0.
template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(MoeCombineRoutes)(
    const device T* expert_outputs,
    device T* output,
    const constant uint& tokens,
    const constant uint& model_dim,
    const constant uint& top_k,
    const uint vector AXIS(model_dim / 4, COMBINE_THREADS),
    const uint token AXIS(tokens, 1)
) {
  if (4 * vector >= model_dim) {
    return;
  }
  const device T* source = expert_outputs + size_t(token) * top_k * model_dim + 4 * vector;
  float4 sum = float4(0.0f);
  for (uint pick = 0; pick < top_k; ++pick) {
    sum += load_float4(source + size_t(pick) * model_dim);
  }
  *reinterpret_cast<device vec<T, 4>*>(output + size_t(token) * model_dim + 4 * vector) = vec<T, 4>(sum);
}
