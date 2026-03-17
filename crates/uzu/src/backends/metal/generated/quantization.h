// Auto-generated from gpu_types/quantization.rs - do not edit manually
#pragma once

#ifndef UZU_QUANTIZATION_H
#define UZU_QUANTIZATION_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

namespace uzu {
namespace quantization {
#else
#include <stdint.h>
#endif

enum QuantizationMode {
  UINT4,
  INT8,
  UINT8,
};

#ifdef __METAL_VERSION__
} // namespace quantization
} // namespace uzu
#endif

#endif // UZU_QUANTIZATION_H
