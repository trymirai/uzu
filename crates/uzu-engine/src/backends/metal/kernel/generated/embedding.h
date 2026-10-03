// Auto-generated from gpu_types/embedding - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::embedding {
enum class EmbeddingTableKind : uint32_t {
  Dense = 0,
  Quantized = 1,
  D4S4 = 2,
};
} // namespace uzu::embedding
