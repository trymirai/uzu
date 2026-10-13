#include <metal_stdlib>
#include "../common/dsl.h"

#include "kv_cache_update.h"
#include "attention_kv_layout.h"

using namespace uzu::kv_layout;

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(KVCacheUpdate) (
    device T* in_place_keys,
    device T* in_place_values,
    const constant uzu::kv_cache_update::Copy* copies,
    const constant uint& copy_count,
    const constant uint& element_dim,
    const uint element_idx AXIS(element_dim, 256)
) {
  for (uint i = 0; i < copy_count; ++i) {
    const uint sourceIdx = copies[i].source * element_dim + element_idx;
    const uint destIdx = copies[i].destination * element_dim + element_idx;

    in_place_keys[destIdx] = in_place_keys[sourceIdx];
    in_place_values[destIdx] = in_place_values[sourceIdx];
  }
}

PUBLIC KERNEL(KVCacheUpdateInt8)(
    device int8_t* in_place_k_codes,
    device uchar* in_place_v_codes,
    device bfloat* in_place_k_scales,
    device bfloat* in_place_v_scales,
    const constant uzu::kv_cache_update::Copy* copies,
    const constant uint& copy_count,
    const constant uint& padded_capacity,
    const uint head_dim SPECIALIZE,
    const uint num_kv_heads SPECIALIZE,
    const uint tile_keys SPECIALIZE,
    const uint tile_dims SPECIALIZE,
    const uint element AXIS(num_kv_heads* head_dim, 256)
) {
  const Int8KvShape shape{head_dim, num_kv_heads, tile_keys, tile_dims};
  const uint kv_head = element / head_dim;
  const uint channel = element % head_dim;
  for (uint i = 0; i < copy_count; ++i) {
    const uint source = copies[i].source;
    const uint destination = copies[i].destination;
    in_place_k_codes[shape.k_code_index(destination, kv_head, channel)] =
        in_place_k_codes[shape.k_code_index(source, kv_head, channel)];
    in_place_v_codes[shape.v_code_index(destination, kv_head, channel)] =
        in_place_v_codes[shape.v_code_index(source, kv_head, channel)];
    if (channel == 0) {
      const size_t source_token = per_token_index(source, kv_head, padded_capacity);
      const size_t destination_token = per_token_index(destination, kv_head, padded_capacity);
      in_place_k_scales[destination_token] = in_place_k_scales[source_token];
      in_place_v_scales[destination_token] = in_place_v_scales[source_token];
    }
  }
}
