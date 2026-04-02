#pragma once

#include "../../common/defines.h"
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

namespace uzu {
namespace matmul {

template <typename T>
struct PointerElementTypeImpl {
  using type = T;
};

template <typename T>
struct PointerElementTypeImpl<device T*> {
  using type = T;
};

template <typename T>
struct PointerElementTypeImpl<const device T*> {
  using type = T;
};

template <typename T>
struct PointerElementTypeImpl<threadgroup T*> {
  using type = T;
};

template <typename T>
using PointerElementType = typename PointerElementTypeImpl<T>::type;

template <typename OutT, typename InT>
struct TransformNone {
  TransformNone(float = 1.0f, float = 0.0f) {}

  static METAL_FUNC OutT apply(InT x) { return static_cast<OutT>(x); }
  METAL_FUNC OutT apply(InT x, OutT) const { return static_cast<OutT>(x); }
};

template <typename OutT, typename InT>
struct TransformScaleAccumulate {
  const float alpha;
  const float beta;

  TransformScaleAccumulate(float alpha, float beta)
      : alpha(alpha), beta(beta) {}

  static METAL_FUNC OutT apply(InT x) { return static_cast<OutT>(x); }

  METAL_FUNC OutT apply(InT x, OutT c) const {
    return static_cast<OutT>(
        x * static_cast<InT>(alpha) +
        (beta != 0.0f ? (static_cast<OutT>(beta) * c) : static_cast<OutT>(0.0))
    );
  }
};

} // namespace matmul
} // namespace uzu
