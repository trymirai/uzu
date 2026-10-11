#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;

#define VALUE_TILE 16u

// One batch, T <= 16, initial state slot 0 and output scale 1. Keep U in
// registers between the FP32 solve and output products.
template <typename T>
VARIANTS(T, bfloat)
KERNEL(TreeUpdateOut)(
    const device T* q,
    const device T* v,
    const device float* prefix,
    const device float* beta,
    const device float* kh0,
    const device float* a_inv,
    const device float* qkd,
    const device float* h0,
    device T* o,
    constant const uint& tree_size,
    constant const uint& k_heads,
    constant const uint& value_heads,
    constant const uint& head_k_dim,
    constant const uint& head_v_dim,
    const uint value_head_idx GROUPS(value_heads),
    const uint value_tile_idx GROUPS(head_v_dim.div_ceil(VALUE_TILE)),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  using Ops = SimdgroupFragmentOps;
  using Tile = Fragment<float, 2, VALUE_TILE / 8, Ops>;
  using Matrix = OperandFragment<float, 2, 2, Ops>;
  using Query = OperandFragment<float, 2, 1, Ops>;
  using State = OperandFragment<float, 1, VALUE_TILE / 8, Ops>;
  const uint value_base = value_tile_idx * VALUE_TILE;
  const uint value_stride = value_heads * head_v_dim;
  const uint value_offset = value_head_idx * head_v_dim + value_base;
  const uint valid_cols = min(VALUE_TILE, head_v_dim - value_base);

  Tile rhs;
  rhs.load_from(lane, fragment_source(kh0 + value_offset, int(value_stride)).bounded(tree_size, valid_cols));
  Tile values;
  values.load_from(lane, fragment_source(v + value_offset, int(value_stride)).bounded(tree_size, valid_cols));
  const float lane_beta = lane < tree_size ? beta[lane * value_heads + value_head_idx] : 0.0f;
  const float lane_decay = lane < tree_size ? exp(prefix[lane * value_heads + value_head_idx]) : 0.0f;
  Tile::zip_for_each_coord(
      lane,
      [&](ushort row, ushort, thread float& value, thread float& input) {
        value = simd_shuffle(lane_beta, row) * (input - simd_shuffle(lane_decay, row) * value);
      },
      rhs,
      values
  );

  Matrix inverse;
  inverse.load_from(lane, fragment_source(a_inv + value_head_idx * 256, 16).bounded(tree_size, tree_size));
  Tile solved;
  solved.clear();
  fragment_mma(solved, inverse, rhs);

  const uint key_head_idx = value_head_idx / (value_heads / k_heads);
  const device T* q_head = q + key_head_idx * head_k_dim;
  const device float* h0_tile = h0 + value_offset * head_k_dim;
  Tile output;
  output.clear();
  for (uint kb = 0; kb < head_k_dim; kb += 8) {
    const uint valid_k = min(8u, head_k_dim - kb);
    Query query;
    State state;
    query.load_from(lane, fragment_source(q_head + kb, int(k_heads * head_k_dim)).bounded(tree_size, valid_k));
    state.load_from(lane, fragment_source(h0_tile + kb, 1, int(head_k_dim)).bounded(valid_k, valid_cols));
    fragment_mma(output, query, state);
  }
  output.map_coords(lane, [&](short row, short, float value) { return value * simd_shuffle(lane_decay, ushort(row)); });
  Matrix weights;
  weights.load_from(
      lane,
      fragment_source(qkd + value_head_idx * tree_size * tree_size, int(tree_size)).bounded(tree_size, tree_size)
  );
  fragment_mma(output, weights, solved);
  output.store_safe(lane, o + value_offset, int(value_stride), short2(valid_cols, tree_size));
}
