// Auto-generated from gpu_types/activation.rs - do not edit manually
#pragma once

#ifndef UZU_ACTIVATION_H
#define UZU_ACTIVATION_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

namespace uzu {
namespace activation {
#else
#include <stdint.h>
#endif

enum ActivationType {
  SILU,
  GELU,
  IDENTITY,
};

#ifdef __METAL_VERSION__
} // namespace activation
} // namespace uzu
#endif

#endif // UZU_ACTIVATION_H
