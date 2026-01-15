#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#define UZU_MTL_CONST static constant constexpr
#define UZU_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define SIMD_SIZE 32

using namespace metal;

namespace uzu {
namespace matmul {

// Pointer element type extraction
template <typename T>
struct pointer_element_t_impl {
  using type = T;
};
template <typename T>
struct pointer_element_t_impl<device T*> {
  using type = T;
};
template <typename T>
struct pointer_element_t_impl<const device T*> {
  using type = T;
};
template <typename T>
struct pointer_element_t_impl<threadgroup T*> {
  using type = T;
};
template <typename T>
using pointer_element_t = typename pointer_element_t_impl<T>::type;

// Accumulator type helper
template <typename T>
struct AccumHelper {
  typedef float accum_type;
};

// Transform operations for epilogue
// Note: apply(x) is static for default store_result usage
// apply(x, c) is non-static for epilogue operations with C matrix
template <typename OutT, typename InT>
struct TransformNone {
  TransformNone(float = 1.0f, float = 0.0f) {}

  static METAL_FUNC OutT apply(InT x) { return static_cast<OutT>(x); }
  METAL_FUNC OutT apply(InT x, OutT) const { return static_cast<OutT>(x); }
};

template <typename OutT, typename InT>
struct TransformAdd {
  TransformAdd(float = 1.0f, float = 1.0f) {}

  static METAL_FUNC OutT apply(InT x) { return static_cast<OutT>(x); }
  METAL_FUNC OutT apply(InT x, OutT c) const {
    return static_cast<OutT>(x) + c;
  }
};

template <typename OutT, typename InT>
struct TransformAxpby {
  const float alpha;
  const float beta;

  TransformAxpby(float alpha_, float beta_) : alpha(alpha_), beta(beta_) {}

  static METAL_FUNC OutT apply(InT x) { return static_cast<OutT>(x); }

  METAL_FUNC OutT apply(InT x, OutT c) const {
    return static_cast<OutT>(
        x * static_cast<InT>(alpha) + (static_cast<OutT>(beta) * c)
    );
  }
};

} // namespace matmul
} // namespace uzu
