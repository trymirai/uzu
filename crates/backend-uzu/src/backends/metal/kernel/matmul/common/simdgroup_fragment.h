#pragma once

#include "fragment.h"
#include "simdgroup_multiply_accumulate.h"

using namespace metal;

namespace uzu {
namespace matmul {

template <
    typename T,
    ushort GRID_ROWS,
    ushort GRID_COLS,
    class SimdgroupOps = SimdgroupFragmentOps<T>>
using SimdgroupFragment = Fragment<T, GRID_ROWS, GRID_COLS, SimdgroupOps>;

} // namespace matmul
} // namespace uzu
