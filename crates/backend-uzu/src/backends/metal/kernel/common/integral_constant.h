#pragma once

#include <metal_stdlib>

#pragma METAL internals : enable

namespace uzu {

template <typename T, T v>
struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;

  METAL_FUNC constexpr operator value_type() const noexcept { return value; }
};

template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template <int val>
using Int = integral_constant<int, val>;

#define METAL_INTEGRAL_CONST_BINOP(op, fn)                                       \
  template <typename T, T tv, typename U, U uv>                                \
  METAL_FUNC constexpr auto fn(                                                \
      integral_constant<T, tv>,                                                \
      integral_constant<U, uv>                                                 \
  ) {                                                                          \
    constexpr auto res = tv op uv;                                             \
    return integral_constant<decltype(res), res>{};                            \
  }

METAL_INTEGRAL_CONST_BINOP(+, operator+)
METAL_INTEGRAL_CONST_BINOP(-, operator-)
METAL_INTEGRAL_CONST_BINOP(*, operator*)
METAL_INTEGRAL_CONST_BINOP(/, operator/)

#undef METAL_INTEGRAL_CONST_BINOP

template <int start, int step, typename F, int... Is>
METAL_FUNC constexpr void
const_for_loop_impl(F f, metal::integer_sequence<int, Is...>) {
  (f(Int<start + Is * step>{}), ...);
}

template <int start, int stop, int step, typename F>
METAL_FUNC constexpr void const_for_loop(F f) {
  static_assert(step > 0 && start <= stop);
  constexpr int count = (stop - start + step - 1) / step;
  const_for_loop_impl<start, step>(
      f, metal::make_integer_sequence<int, count>{}
  );
}

template <typename F>
METAL_FUNC void dispatch_bool(bool v, F f) {
  if (v) {
    f(true_type{});
  } else {
    f(false_type{});
  }
}

} // namespace uzu

#pragma METAL internals : disable
