#pragma once

#include "fragment.h"
#include "mxu_fragment.h"

namespace uzu {
namespace matmul {

template <
    typename T,
    ushort TILE_ROWS_,
    ushort TILE_COLS_,
    class FragmentOps = MxuFragmentOps>
using MxuTile = Fragment<T, TILE_ROWS_, TILE_COLS_, FragmentOps>;

} // namespace matmul
} // namespace uzu
