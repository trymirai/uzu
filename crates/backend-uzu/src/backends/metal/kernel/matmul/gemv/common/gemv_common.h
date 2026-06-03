#pragma once

#include <metal_stdlib>

#include "../../common/defines.h"
#include "../../../generated/gemm.h"

using namespace metal;

namespace uzu {
namespace gemm {

METAL_CONST uint RESULTS_PER_SIMDGROUP = 4;

} // namespace gemm
} // namespace uzu
