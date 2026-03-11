

#pragma once

///////////////////////////////////////////////////////////////////////////////
// Transforms and Epilogues
///////////////////////////////////////////////////////////////////////////////

namespace steel {

template <typename OutputType, typename InputType>
struct TransformNone {
  static METAL_FUNC OutputType apply(InputType x) { return static_cast<OutputType>(x); }

  static METAL_FUNC OutputType apply(InputType x, OutputType) { return static_cast<OutputType>(x); }
};

template <typename OutputType, typename InputType>
struct TransformAdd {
  TransformAdd(const float, const float) {}

  static METAL_FUNC OutputType apply(InputType x) { return static_cast<OutputType>(x); }

  static METAL_FUNC OutputType apply(InputType x, OutputType c) {
    return static_cast<OutputType>(x) + c;
  }
};

template <typename OutputType, typename InputType>
struct TransformAxpby {
  const float alpha;
  const float beta;

  TransformAxpby(const float alpha_, const float beta_)
      : alpha(alpha_), beta(beta_) {}

  static METAL_FUNC OutputType apply(InputType x) { return static_cast<OutputType>(x); }

  METAL_FUNC OutputType apply(InputType x, OutputType c) const {
    return static_cast<OutputType>(
        x * static_cast<InputType>(alpha) + (static_cast<OutputType>(beta) * c)
    );
  }
};

template <typename T>
struct AccumHelper {
  typedef float AccumType;
};

struct BlockSwizzle {
  static METAL_FUNC int2
  swizzle(uint3 threadgroup_position [[threadgroup_position_in_grid]], const int swizzle_log) {
    const int swizzle_size = 1 << swizzle_log;
    const int swizzled_x = threadgroup_position.x / swizzle_size;
    const int swizzled_y =
        threadgroup_position.y * swizzle_size + (threadgroup_position.x % swizzle_size);
    return int2(swizzled_x, swizzled_y);
  }
};

} // namespace steel
