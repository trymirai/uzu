#pragma once

#include "fragment.h"
#include "simdgroup_multiply_accumulate.h"

using namespace metal;

namespace uzu {
namespace matmul {

template <
    typename T,
    int GRID_ROWS_,
    int GRID_COLS_,
    class SimdgroupOps = SimdgroupFragmentOps<T>>
using SimdgroupFragment = Fragment<T, GRID_ROWS_, GRID_COLS_, SimdgroupOps>;

} // namespace matmul
} // namespace uzu
