#include <metal_stdlib>
#include "../common/dsl.h"

// Scatters the key/value halves of the current step's packed QKV rows into their
// node-cache slots. Kept out of AttentionLastQuery so that the node arena stays
// read-only for the whole attention dispatch.
template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(WeaverNodeCacheWrite)(
    const device T* current_qkv,
    device T* node_qkv,
    const device uint* node_indices,
    constant uint& model_dim,
    constant uint& total,
    const uint position AXIS(total, 256)
) {
  const uint row = position / (2 * model_dim);
  const uint offset = position - row * 2 * model_dim;
  const uint qkv_width = 3 * model_dim;
  node_qkv[node_indices[row] * qkv_width + model_dim + offset] = current_qkv[row * qkv_width + model_dim + offset];
}
