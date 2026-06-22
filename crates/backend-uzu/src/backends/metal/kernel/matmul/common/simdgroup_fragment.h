#pragma once

#include "fragment.h"
#include "simdgroup_fragment_ops.h"
#include "simdgroup_multiply_accumulate.h"

using namespace metal;

namespace uzu {
namespace matmul {

///////////////////////////////////////////////////////////////////////////////
// Simdgroup tile I/O — both backends use one tile type, Fragment<…, Ops>.
// These free functions are the simdgroup-backend threadgroup load / device store
// (caller pre-offsets the source/dest per lane, with per-simdgroup sub-tile
// strides). They stay free functions because Metal forbids the inheritance that
// would let a tile carry backend-specific methods.
///////////////////////////////////////////////////////////////////////////////

template <typename U, int SIMDGROUP_STRIDE_X, int SIMDGROUP_STRIDE_Y, int STRIDE_X, int STRIDE_Y, typename FragT>
METAL_FUNC void tile_load(thread FragT& frag, const threadgroup U* source) {
  using Sg = SimdgroupMultiplyAccumulate<typename FragT::ElementType, 8, 8>;
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < FragT::TILE_ROWS; ++i) {
    METAL_PRAGMA_UNROLL
    for (ushort j = 0; j < FragT::TILE_COLS; ++j) {
      Sg::load(
          frag.fragment_at(i, j),
          &(source[(i * 8) * SIMDGROUP_STRIDE_X * STRIDE_X + (j * 8) * SIMDGROUP_STRIDE_Y * STRIDE_Y]),
          STRIDE_X,
          STRIDE_Y
      );
    }
  }
}

template <typename U, int SIMDGROUP_STRIDE_X, int SIMDGROUP_STRIDE_Y, typename FragT>
METAL_FUNC void tile_store(thread FragT& frag, device U* destination, const int leading_dimension) {
  using Sg = SimdgroupMultiplyAccumulate<typename FragT::ElementType, 8, 8>;
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < FragT::TILE_ROWS; ++i) {
    METAL_PRAGMA_UNROLL
    for (ushort j = 0; j < FragT::TILE_COLS; ++j) {
      Sg::store(
          frag.fragment_at(i, j),
          &(destination[(i * 8) * SIMDGROUP_STRIDE_X * leading_dimension + (j * 8) * SIMDGROUP_STRIDE_Y]),
          leading_dimension,
          1
      );
    }
  }
}

template <typename U, int SIMDGROUP_STRIDE_X, int SIMDGROUP_STRIDE_Y, typename FragT>
METAL_FUNC void tile_store_safe(
    thread FragT& frag,
    device U* destination,
    const int leading_dimension,
    const short2 destination_tile_dimensions
) {
  using Sg = SimdgroupMultiplyAccumulate<typename FragT::ElementType, 8, 8>;
  METAL_PRAGMA_UNROLL
  for (int i = 0; i < FragT::TILE_ROWS; ++i) {
    METAL_PRAGMA_UNROLL
    for (int j = 0; j < FragT::TILE_COLS; ++j) {
      Sg::store_safe(
          frag.fragment_at(i, j),
          destination,
          leading_dimension,
          1,
          destination_tile_dimensions.y,
          destination_tile_dimensions.x,
          (i * 8) * SIMDGROUP_STRIDE_X,
          (j * 8) * SIMDGROUP_STRIDE_Y
      );
    }
  }
}

} // namespace matmul
} // namespace uzu
