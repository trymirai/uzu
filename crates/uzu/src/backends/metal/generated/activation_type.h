// Auto-generated from gpu_types/activation_type.rs - do not edit manually
#pragma once

#ifndef UZU_ACTIVATION_TYPE_H
#define UZU_ACTIVATION_TYPE_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

namespace uzu {
namespace activation_type {
#else
#include <stdint.h>
#endif

enum ActivationType : uint32_t {
  SILU,
  GELU,
  TANH,
  IDENTITY,
};

#ifdef __METAL_VERSION__
} // namespace activation_type
} // namespace uzu
#endif

#endif // UZU_ACTIVATION_TYPE_H
