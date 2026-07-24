#include <metal_stdlib>
#include "../common/dsl.h"

// Scatters the key/value halves of the current step's packed QKV rows into their
// node-cache slots. Kept out of AncestorAttention so that the node arena stays
// read-only for the whole attention dispatch.
template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(WeaverNodeCacheWrite)(
    const device T* current_qkv,
    device T* node_kv,
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
  const uint component = offset / model_dim;
  const uint component_offset = offset % model_dim;
  const uint qkv_width = 3 * model_dim;
  const uint node = min(node_indices[row], node_capacity - 1u);
  node_kv[component * node_capacity * model_dim + node * model_dim + component_offset] =
      current_qkv[row * qkv_width + (component + 1) * model_dim + component_offset];
}
