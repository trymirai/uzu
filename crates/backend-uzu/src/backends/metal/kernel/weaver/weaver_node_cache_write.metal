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
    constant uint& node_capacity,
    constant uint& total,
    const uint position AXIS(total, 256)
) {
  if (node_capacity == 0) {
    return;
  }
  const uint row = position / (2 * model_dim);
  const uint offset = position - row * 2 * model_dim;
  const uint qkv_width = 3 * model_dim;
  const uint node = min(node_indices[row], node_capacity - 1u);
  node_qkv[node * qkv_width + model_dim + offset] = current_qkv[row * qkv_width + model_dim + offset];
}
