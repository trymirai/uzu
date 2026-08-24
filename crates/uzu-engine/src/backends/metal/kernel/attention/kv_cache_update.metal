#include <metal_stdlib>
#include "../common/dsl.h"

#include "kv_cache_update.h"

template <typename T>
VARIANTS(T, float, bfloat, half)
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
