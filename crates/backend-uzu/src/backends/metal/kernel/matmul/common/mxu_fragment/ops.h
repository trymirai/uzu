#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#include "../../../common/integral_constant.h"
#include "../../../common/thread_context.h"
using namespace uzu;

#include "../defines.h"
#include "../loader.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

namespace uzu {
namespace matmul {

UZU_CONST ushort MXU_MMA_ROWS = 16;
UZU_CONST ushort MXU_MMA_COLS = 16;

// RELAXED=false uses the strict MPP layout; it currently performs about the same as simdgroup.
template <bool RELAXED = true>
struct MxuFragmentOps {
  using MatmulMode = mpp::tensor_ops::matmul2d_descriptor::mode;

#include "layout.h"
#include "cooperative_vectors.h"
#include "tile_matmul.h"
#include "fragment_matmul.h"
#include "device_weight_matmul.h"
};

using MxuStrictFragmentOps = MxuFragmentOps<false>;

} // namespace matmul
} // namespace uzu
