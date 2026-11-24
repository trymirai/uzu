#include <metal_stdlib>
#include "../common/dsl.h"

#include "kv_cache_update.h"
#include "cache_copy.h"

namespace kv_layout = uzu::kv_layout;
namespace kv_cache = uzu::kv_cache;

PUBLIC KERNEL(KVCacheUpdate)(
    device int8_t* in_place_k_codes,
    device uchar* in_place_v_codes OPTIONAL(!keys_only),
    device bfloat* in_place_k_scales,
    device bfloat* in_place_v_scales OPTIONAL(!keys_only),
    const constant uzu::kv_cache_update::Copy* copies,
    const constant uint& copy_count,
    const uint head_dim SPECIALIZE,
    const uint num_kv_heads SPECIALIZE,
    const uint32_t backend_opaque_layout SPECIALIZE,
    const bool keys_only SPECIALIZE,
    const uint element AXIS(num_kv_heads* head_dim, 256)
) {
  const kv_layout::KVShape shape{head_dim, num_kv_heads, static_cast<kv_layout::KVLayout>(backend_opaque_layout)};
  const uint kv_head = element / head_dim;
  const uint channel = element % head_dim;
  // Tiles load before storing; encode_accept asserts this is safe.
  constexpr uint TILE = 2;
  for (uint first = 0; first < copy_count; first += TILE) {
    kv_cache::KeyElement keys[TILE];
    uchar v_codes[TILE];
    bfloat v_scales[TILE];
    const auto load = [&](uint i) {
      const uint source = copies[first + i].source;
      keys[i] = kv_cache::load_key(in_place_k_codes, in_place_k_scales, shape, source, kv_head, channel);
      if (!keys_only) {
        v_codes[i] = in_place_v_codes[shape.v_code_index(source, kv_head, channel)];
      }
      if (channel == 0 && !keys_only) {
        v_scales[i] = in_place_v_scales[shape.scale_index(source, kv_head)];
      }
    };
    const auto store = [&](uint i) {
      const uint destination = copies[first + i].destination;
      kv_cache::store_key(in_place_k_codes, in_place_k_scales, shape, destination, kv_head, channel, keys[i]);
      if (!keys_only) {
        in_place_v_codes[shape.v_code_index(destination, kv_head, channel)] = v_codes[i];
      }
      if (channel == 0 && !keys_only) {
        in_place_v_scales[shape.scale_index(destination, kv_head)] = v_scales[i];
      }
    };
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < TILE && first + i < copy_count; ++i) {
      load(i);
    }
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < TILE && first + i < copy_count; ++i) {
      store(i);
    }
  }
}
