#include <metal_stdlib>
#include "../common/dsl.h"

#include "kv_layout.h"

namespace kv_layout = uzu::kv_layout;

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(KVCacheCompact)(
    device T* in_place_keys,
    device T* in_place_values,
    // [len][indices...], deepest accepted row at word 1
    const device uint* accepted_indices,
    constant const uint& suffix_base,
    constant const uint& ring_write_start,
    constant const uint& ring_capacity,
    constant const uint& element_dim,
    const uint element_idx AXIS(element_dim, 256)
) {
  const uint accepted_len = accepted_indices[0];
  for (uint rank = 0; rank < accepted_len; ++rank) {
    const uint source = suffix_base + accepted_indices[accepted_len - rank];
    uint destination;
    if (ring_capacity == 0) {
      destination = suffix_base + rank;
    } else {
      if (rank + ring_capacity < accepted_len) continue;
      destination = (ring_write_start + rank) % ring_capacity;
    }
    if (source == destination) continue;

    const uint sourceIdx = source * element_dim + element_idx;
    const uint destIdx = destination * element_dim + element_idx;
    in_place_keys[destIdx] = in_place_keys[sourceIdx];
    in_place_values[destIdx] = in_place_values[sourceIdx];
  }
}

PUBLIC KERNEL(KVCacheCompactInt8)(
    device int8_t* in_place_k_codes,
    device uchar* in_place_v_codes,
    device bfloat* in_place_k_scales,
    device bfloat* in_place_v_scales,
    // [len][indices...], deepest accepted row at word 1
    const device uint* accepted_indices,
    constant const uint& suffix_base,
    constant const uint& ring_write_start,
    constant const uint& ring_capacity,
    const uint head_dim SPECIALIZE,
    const uint num_kv_heads SPECIALIZE,
    const uint element AXIS(num_kv_heads * head_dim, 256)
) {
  const kv_layout::Int8KVShape shape{head_dim, num_kv_heads};
  const uint kv_head = element / head_dim;
  const uint channel = element % head_dim;
  const uint accepted_len = accepted_indices[0];
  for (uint rank = 0; rank < accepted_len; ++rank) {
    const uint source = suffix_base + accepted_indices[accepted_len - rank];
    uint destination;
    if (ring_capacity == 0) {
      destination = suffix_base + rank;
    } else {
      if (rank + ring_capacity < accepted_len) continue;
      destination = (ring_write_start + rank) % ring_capacity;
    }
    if (source == destination) continue;

    in_place_k_codes[shape.k_code_index(destination, kv_head, channel)] =
        in_place_k_codes[shape.k_code_index(source, kv_head, channel)];
    in_place_v_codes[shape.v_code_index(destination, kv_head, channel)] =
        in_place_v_codes[shape.v_code_index(source, kv_head, channel)];
    if (channel == 0) {
      const size_t source_scale = shape.scale_index(source, kv_head);
      const size_t destination_scale = shape.scale_index(destination, kv_head);
      in_place_k_scales[destination_scale] = in_place_k_scales[source_scale];
      in_place_v_scales[destination_scale] = in_place_v_scales[source_scale];
    }
  }
}
