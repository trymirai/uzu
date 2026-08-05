#include <metal_stdlib>
#include "../common/dsl.h"
#include "weaver_frontier.h"

using namespace metal;

template <typename ElementT, typename RopeT>
VARIANTS(ElementT, bfloat)
VARIANTS(RopeT, float)
PUBLIC KERNEL(WeaverRopeQkv)(
    device ElementT* qkv,
    const device RopeT* cosines,
    const device RopeT* sines,
    const device uint* node_metadata,
    constant uint& num_heads,
    constant uint& head_dim,
    constant uint& max_depth,
    constant uint& rows,
    const uint pair AXIS(head_dim / 2, 64),
    const uint head AXIS(num_heads * 2, 1),
    const uint row AXIS(rows, 1)
) {
  const uint half_dim = head_dim / 2;
  const uint model_dim = num_heads * head_dim;
  const uint qkv_width = 3 * model_dim;

  const uint depth = node_metadata[uint(MetadataIdx::Depth) * rows + row];
  const uint position = min(depth, max_depth - 1u) + 1u;

  const uint head_base = row * qkv_width + head * head_dim;
  device ElementT* low = qkv + head_base + pair;
  device ElementT* high = qkv + head_base + half_dim + pair;

  const float low_value = float(*low);
  const float high_value = float(*high);

  const uint rope_index = position * head_dim + pair;
  const float low_cosine = float(cosines[rope_index]);
  const float low_sine = float(sines[rope_index]);
  const float high_cosine = float(cosines[rope_index + half_dim]);
  const float high_sine = float(sines[rope_index + half_dim]);

  *low = static_cast<ElementT>(low_value * low_cosine - high_value * low_sine);
  *high = static_cast<ElementT>(high_value * high_cosine + low_value * high_sine);
}
