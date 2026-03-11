#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#define MTL_CONST static constant constexpr
#define PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define SIMD_SIZE 32

using namespace metal;

namespace uzu {
namespace matmul {

// Pointer element type extraction
template <typename T>
struct PointerElementImpl {
  using type = T;
};
template <typename T>
struct PointerElementImpl<device T*> {
  using type = T;
};
template <typename T>
struct PointerElementImpl<const device T*> {
  using type = T;
};
template <typename T>
struct PointerElementImpl<threadgroup T*> {
  using type = T;
};
template <typename T>
using pointer_element_t = typename PointerElementImpl<T>::type;

// Accumulator type helper
template <typename T>
struct AccumHelper {
  typedef float AccumType;
};

// Transform operations for epilogue
// Note: apply(x) is static for default store_result usage
// apply(x, c) is non-static for epilogue operations with C matrix
template <typename OutputType, typename InputType>
struct TransformNone {
  TransformNone(float = 1.0f, float = 0.0f) {}

  static METAL_FUNC OutputType apply(InputType x) { return static_cast<OutputType>(x); }
  METAL_FUNC OutputType apply(InputType x, OutputType) const { return static_cast<OutputType>(x); }
};

template <typename OutputType, typename InputType>
struct TransformAdd {
  TransformAdd(float = 1.0f, float = 1.0f) {}

  static METAL_FUNC OutputType apply(InputType x) { return static_cast<OutputType>(x); }
  METAL_FUNC OutputType apply(InputType x, OutputType c) const {
    return static_cast<OutputType>(x) + c;
  }
};

template <typename OutputType, typename InputType>
struct TransformAxpby {
  const float alpha;
  const float beta;

  TransformAxpby(float alpha_, float beta_) : alpha(alpha_), beta(beta_) {}

  static METAL_FUNC OutputType apply(InputType x) { return static_cast<OutputType>(x); }

  METAL_FUNC OutputType apply(InputType x, OutputType c) const {
    return static_cast<OutputType>(
        x * static_cast<InputType>(alpha) + (static_cast<OutputType>(beta) * c)
    );
  }
};

} // namespace matmul
} // namespace uzu
