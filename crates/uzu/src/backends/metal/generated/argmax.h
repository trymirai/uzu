// Auto-generated from gpu_types/argmax.rs - do not edit manually
#pragma once

#ifndef UZU_ARGMAX_H
#define UZU_ARGMAX_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

namespace uzu {
namespace argmax {
#else
#include <stdint.h>
#endif

typedef struct {
  float value;
  uint32_t index;
} ArgmaxPair;

#ifdef __METAL_VERSION__
} // namespace argmax
} // namespace uzu
#endif

#endif // UZU_ARGMAX_H
