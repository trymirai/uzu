// Auto-generated from gpu_types/kv_cache_update.rs - do not edit manually
#pragma once

#ifndef UZU_KV_CACHE_UPDATE_H
#define UZU_KV_CACHE_UPDATE_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

namespace uzu {
namespace kv_cache_update {
#else
#include <stdint.h>
#endif

typedef struct {
  uint32_t source;
  uint32_t destination;
} Swap;

#ifdef __METAL_VERSION__
} // namespace kv_cache_update
} // namespace uzu
#endif

#endif // UZU_KV_CACHE_UPDATE_H
