#pragma once

#include "../../common/dsl.h"
#include "bits_per_weight.h"

namespace uzu {
namespace unified_gemm {

template <typename T>
struct GemmPipeline {
  static METAL_FUNC void run() {}
};

} // namespace unified_gemm
} // namespace uzu

