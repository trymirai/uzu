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

KERNEL(KVCacheUpdateInt8)(
    device int8_t* in_place_k_codes,
    device uchar* in_place_v_codes,
    device bfloat* in_place_k_scales,
    device bfloat* in_place_v_scales,
    const constant uzu::kv_cache_update::Copy* copies,
    const constant uint& copy_count,
    const constant uint& num_kv_heads,
    const constant uint& padded_capacity,
    const uint kv_head GROUPS(num_kv_heads),
    const uint channel THREADS(KV_HEAD_DIM)
) {
  for (uint i = 0; i < copy_count; ++i) {
    const uint source = copies[i].source;
    const uint destination = copies[i].destination;
    in_place_k_codes[k_code_index(destination, kv_head, num_kv_heads, channel)] =
        in_place_k_codes[k_code_index(source, kv_head, num_kv_heads, channel)];
    in_place_v_codes[v_code_index(destination, kv_head, num_kv_heads, channel)] =
        in_place_v_codes[v_code_index(source, kv_head, num_kv_heads, channel)];
    if (channel == 0) {
      const size_t sourceIdx = per_token_index(kv_head, source, padded_capacity);
      const size_t destIdx = per_token_index(kv_head, destination, padded_capacity);
      in_place_k_scales[destIdx] = in_place_k_scales[sourceIdx];
      in_place_v_scales[destIdx] = in_place_v_scales[sourceIdx];
    }
  }
}
