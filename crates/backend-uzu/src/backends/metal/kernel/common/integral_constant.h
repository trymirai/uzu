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

#define UZU_INTEGRAL_CONST_BINOP(op, fn)                                       \
  template <typename T, T tv, typename U, U uv>                                \
  METAL_FUNC constexpr auto fn(                                                \
      integral_constant<T, tv>,                                                \
      integral_constant<U, uv>                                                 \
  ) {                                                                          \
    constexpr auto res = tv op uv;                                             \
    return integral_constant<decltype(res), res>{};                            \
  }

UZU_INTEGRAL_CONST_BINOP(+, operator+)
UZU_INTEGRAL_CONST_BINOP(-, operator-)
UZU_INTEGRAL_CONST_BINOP(*, operator*)
UZU_INTEGRAL_CONST_BINOP(/, operator/)

#undef UZU_INTEGRAL_CONST_BINOP

template <int start, int stop, int step, typename F>
constexpr void const_for_loop(F f) {
  if constexpr (start < stop) {
    f(Int<start>{});
    const_for_loop<start + step, stop, step, F>(f);
  }
}

template <typename F>
void dispatch_bool(bool v, F f) {
  if (v) {
    f(true_type{});
  } else {
    f(false_type{});
  }
}

} // namespace uzu

#pragma METAL internals : disable
