#pragma once

#include "../../common/defines.h"
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
using PointerElementType = typename PointerElementTypeImpl<T>::type;

template <typename OutputType, typename InputType>
struct TransformNone {
  TransformNone(float = 1.0f, float = 0.0f) {}

  static METAL_FUNC OutputType apply(InputType x) {
    return static_cast<OutputType>(x);
  }
  METAL_FUNC OutputType apply(InputType x, OutputType) const {
    return static_cast<OutputType>(x);
  }
};

template <typename OutputType, typename InputType>
struct TransformScaleAccumulate {
  const float alpha;
  const float beta;

  TransformScaleAccumulate(float alpha, float beta)
      : alpha(alpha), beta(beta) {}

  static METAL_FUNC OutputType apply(InputType x) {
    return static_cast<OutputType>(x);
  }

  METAL_FUNC OutputType apply(InputType x, OutputType c) const {
    return static_cast<OutputType>(
        x * static_cast<InputType>(alpha) +
        (beta != 0.0f ? (static_cast<OutputType>(beta) * c)
                      : static_cast<OutputType>(0.0))
    );
  }
};

} // namespace matmul
} // namespace uzu
