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
struct PointerElementTypeImpl<threadgroup T*> {
  using type = T;
};

template <typename T>
using PointerElementType = typename PointerElementTypeImpl<T>::type;

template <typename OutputType, typename InputType>
struct TransformNone {
  static METAL_FUNC OutputType apply(InputType x) { return static_cast<OutputType>(x); }
};

template <typename OutputType, typename InputType>
struct TransformScale {
  const float alpha;

  TransformScale(float alpha) : alpha(alpha) {}

  METAL_FUNC OutputType apply(InputType x) const { return static_cast<OutputType>(x * static_cast<InputType>(alpha)); }
};

} // namespace matmul
} // namespace uzu
