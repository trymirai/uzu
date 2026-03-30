// Auto-generated from gpu_types/ring.rs - do not edit manually
#pragma once

#ifndef UZU_RING_H
#define UZU_RING_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

namespace uzu {
namespace ring {
#else
#include <stdint.h>
#endif

typedef struct {
  uint32_t ring_offset;
  uint32_t ring_length;
} RingParams;

#ifdef __METAL_VERSION__
} // namespace ring
} // namespace uzu
#endif

#endif // UZU_RING_H
