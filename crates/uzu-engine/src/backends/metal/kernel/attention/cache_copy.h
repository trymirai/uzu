#pragma once

#include "kv_layout.h"

namespace uzu::kv_cache {

struct KeyElement {
  int8_t code;
  bfloat scale;
};

METAL_FUNC KeyElement load_key(
    const device int8_t* codes,
    const device bfloat* scales,
    const thread kv_layout::KVShape& shape,
    uint row,
    uint head,
    uint channel
) {
  return {
      codes[shape.k_code_index(row, head, channel)],
      channel == 0 ? scales[shape.scale_index(row, head)] : bfloat(0)
  };
}

METAL_FUNC void store_key(
    device int8_t* codes,
    device bfloat* scales,
    const thread kv_layout::KVShape& shape,
    uint row,
    uint head,
    uint channel,
    KeyElement value
) {
  codes[shape.k_code_index(row, head, channel)] = value.code;
  if (channel == 0)
    scales[shape.scale_index(row, head)] = value.scale;
}

} // namespace uzu::kv_cache
