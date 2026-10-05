// Math and common functions of the Metal Shading Language.
// Floating point work is done in float via clang elementwise builtins (lowered natively by the AMDGPU
// backend, no device libraries); half and bfloat inputs are widened to float and narrowed back.
#pragma once

namespace metal {

// --- element type helpers (__vec_lane_info is defined in metal_compat_types.h)
template <typename T>
using __vec_info = __vec_lane_info<T>;

template <typename T>
using __element_t = typename __vec_info<remove_cv_t<T>>::element;

template <typename T, typename E>
using __rebind_t = conditional_t<__vec_info<T>::lanes == 1, E, vec<E, __vec_info<T>::lanes>>;

template <typename T>
using __float_t = __rebind_t<T, float>;

template <typename T>
METAL_FUNC __float_t<T> __widen(T x) {
  if constexpr (__vec_info<T>::lanes == 1) {
    return float(x);
  } else {
    return __builtin_convertvector(x, __float_t<T>);
  }
}

template <typename T>
METAL_FUNC T __narrow(__float_t<T> x) {
  if constexpr (__vec_info<T>::lanes == 1) {
    return T(x);
  } else {
    return __builtin_convertvector(x, T);
  }
}

template <typename T>
constexpr bool __is_float_like = is_floating_point_v<__element_t<T>>;

template <typename T>
constexpr bool __is_int_like = is_integral_v<__element_t<T>>;

#define METAL_COMPAT_FLOAT_UNARY(NAME, BUILTIN)                                                                       \
  template <typename T, enable_if_t<__is_float_like<T>, int> = 0>                                                    \
  METAL_FUNC T NAME(T x) {                                                                                            \
    return __narrow<T>(BUILTIN(__widen(x)));                                                                          \
  }

METAL_COMPAT_FLOAT_UNARY(sqrt, __builtin_elementwise_sqrt)
METAL_COMPAT_FLOAT_UNARY(exp, __builtin_elementwise_exp)
METAL_COMPAT_FLOAT_UNARY(exp2, __builtin_elementwise_exp2)
METAL_COMPAT_FLOAT_UNARY(log, __builtin_elementwise_log)
METAL_COMPAT_FLOAT_UNARY(log2, __builtin_elementwise_log2)
METAL_COMPAT_FLOAT_UNARY(log10, __builtin_elementwise_log10)
METAL_COMPAT_FLOAT_UNARY(floor, __builtin_elementwise_floor)
METAL_COMPAT_FLOAT_UNARY(ceil, __builtin_elementwise_ceil)
METAL_COMPAT_FLOAT_UNARY(trunc, __builtin_elementwise_trunc)
METAL_COMPAT_FLOAT_UNARY(round, __builtin_elementwise_round)
METAL_COMPAT_FLOAT_UNARY(rint, __builtin_elementwise_rint)
METAL_COMPAT_FLOAT_UNARY(sin, __builtin_elementwise_sin)
METAL_COMPAT_FLOAT_UNARY(cos, __builtin_elementwise_cos)
METAL_COMPAT_FLOAT_UNARY(fabs, __builtin_elementwise_abs)
#undef METAL_COMPAT_FLOAT_UNARY

template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T exp10(T x) {
  return __narrow<T>(__builtin_elementwise_exp2(__widen(x) * 3.32192809488736234787f));
}

template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T rsqrt(T x) {
  return __narrow<T>(1.0f / __builtin_elementwise_sqrt(__widen(x)));
}

template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T tan(T x) {
  const auto f = __widen(x);
  return __narrow<T>(__builtin_elementwise_sin(f) / __builtin_elementwise_cos(f));
}

template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T fract(T x) {
  const auto f = __widen(x);
  return __narrow<T>(__builtin_elementwise_min(f - __builtin_elementwise_floor(f), __float_t<T>(0x1.fffffep-1f)));
}

// tanh via exp with a series near zero (avoids cancellation in 1 - 2/(e^2x + 1)).
METAL_FUNC float __tanh_f32(float x) {
  const float a = __builtin_fabsf(x);
  float r;
  if (a < 0.0625f) {
    const float x2 = x * x;
    r = a * (1.0f + x2 * (-1.0f / 3.0f + x2 * (2.0f / 15.0f + x2 * (-17.0f / 315.0f))));
  } else if (a > 9.0f) {
    r = 1.0f;
  } else {
    r = 1.0f - 2.0f / (__builtin_expf(2.0f * a) + 1.0f);
  }
  return __builtin_copysignf(r, x);
}

// erf: Abramowitz & Stegun 7.1.26 refined (|error| < 1.5e-7).
METAL_FUNC float __erf_f32(float x) {
  const float a = __builtin_fabsf(x);
  const float t = 1.0f / (1.0f + 0.3275911f * a);
  const float poly =
      t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  const float r = 1.0f - poly * __builtin_expf(-a * a);
  return __builtin_copysignf(r, x);
}

#define METAL_COMPAT_FLOAT_SCALAR_FN(NAME, FN)                                                                        \
  template <typename T, enable_if_t<__is_float_like<T>, int> = 0>                                                    \
  METAL_FUNC T NAME(T x) {                                                                                            \
    auto f = __widen(x);                                                                                              \
    if constexpr (__vec_info<T>::lanes == 1) {                                                                        \
      return __narrow<T>(FN(f));                                                                                      \
    } else {                                                                                                          \
      _Pragma("unroll") for (int i = 0; i < __vec_info<T>::lanes; ++i) { f[i] = FN(f[i]); }                          \
      return __narrow<T>(f);                                                                                          \
    }                                                                                                                 \
  }

METAL_COMPAT_FLOAT_SCALAR_FN(tanh, __tanh_f32)
METAL_COMPAT_FLOAT_SCALAR_FN(erf, __erf_f32)
#undef METAL_COMPAT_FLOAT_SCALAR_FN

template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T erfc(T x) {
  return T(1) - erf(x);
}

template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T sinh(T x) {
  const auto f = __widen(x);
  return __narrow<T>(0.5f * (__builtin_elementwise_exp(f) - __builtin_elementwise_exp(-f)));
}

template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T cosh(T x) {
  const auto f = __widen(x);
  return __narrow<T>(0.5f * (__builtin_elementwise_exp(f) + __builtin_elementwise_exp(-f)));
}

// pow: MSL returns NaN for x < 0 with non-integral y; powr requires x >= 0.
template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T powr(T x, T y) {
  return __narrow<T>(__builtin_elementwise_exp2(__widen(y) * __builtin_elementwise_log2(__widen(x))));
}

template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T pow(T x, T y) {
  return __narrow<T>(__builtin_elementwise_pow(__widen(x), __widen(y)));
}

template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T fma(T a, T b, T c) {
  return __narrow<T>(__builtin_elementwise_fma(__widen(a), __widen(b), __widen(c)));
}

template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T copysign(T a, T b) {
  return __narrow<T>(__builtin_elementwise_copysign(__widen(a), __widen(b)));
}

template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T fmod(T a, T b) {
  const auto fa = __widen(a);
  const auto fb = __widen(b);
  return __narrow<T>(fa - fb * __builtin_elementwise_trunc(fa / fb));
}

template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T fmax(T a, T b) {
  return __narrow<T>(__builtin_elementwise_max(__widen(a), __widen(b)));
}

template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T fmin(T a, T b) {
  return __narrow<T>(__builtin_elementwise_min(__widen(a), __widen(b)));
}

template <typename T, enable_if_t<__is_float_like<T>, int> = 0>
METAL_FUNC T divide(T a, T b) {
  return __narrow<T>(__widen(a) / __widen(b));
}

// --- abs / min / max / clamp for floats and integers
template <typename T>
METAL_FUNC T abs(T x) {
  if constexpr (__is_float_like<T>) {
    return __narrow<T>(__builtin_elementwise_abs(__widen(x)));
  } else {
    return __builtin_elementwise_abs(x);
  }
}

template <typename T>
METAL_FUNC T max(T a, T b) {
  if constexpr (__is_float_like<T>) {
    return __narrow<T>(__builtin_elementwise_max(__widen(a), __widen(b)));
  } else {
    return __builtin_elementwise_max(a, b);
  }
}

template <typename T>
METAL_FUNC T min(T a, T b) {
  if constexpr (__is_float_like<T>) {
    return __narrow<T>(__builtin_elementwise_min(__widen(a), __widen(b)));
  } else {
    return __builtin_elementwise_min(a, b);
  }
}

template <typename T>
METAL_FUNC T clamp(T x, T lo, T hi) {
  return min(max(x, lo), hi);
}

template <typename T>
METAL_FUNC T saturate(T x) {
  return clamp(x, T(0), T(1));
}

template <typename T>
METAL_FUNC T mix(T x, T y, T a) {
  return x + (y - x) * a;
}

template <typename T>
METAL_FUNC T sign(T x) {
  return x > T(0) ? T(1) : (x < T(0) ? T(-1) : T(0));
}

template <typename T>
METAL_FUNC T step(T edge, T x) {
  return x < edge ? T(0) : T(1);
}

// select(a, b, c) = c ? b : a (elementwise for vectors)
template <typename T, typename C>
METAL_FUNC T select(T a, T b, C c) {
  if constexpr (__vec_info<T>::lanes == 1) {
    return c ? b : a;
  } else {
    T result;
#pragma unroll
    for (int i = 0; i < __vec_info<T>::lanes; ++i) {
      result[i] = c[i] ? b[i] : a[i];
    }
    return result;
  }
}

// --- classification
template <typename T>
METAL_FUNC bool isnan(T x) {
  return __builtin_isnan(float(x));
}

template <typename T>
METAL_FUNC bool isinf(T x) {
  return __builtin_isinf(float(x));
}

template <typename T>
METAL_FUNC bool isfinite(T x) {
  return __builtin_isfinite(float(x));
}

template <typename T>
METAL_FUNC bool signbit(T x) {
  return __builtin_signbit(float(x));
}

// --- integer bit operations
template <typename T>
METAL_FUNC T popcount(T x) {
  return __builtin_elementwise_popcount(x);
}

template <typename T>
METAL_FUNC T clz(T x) {
  if constexpr (__vec_info<T>::lanes == 1) {
    using U = make_unsigned_t<T>;
    return x == T(0) ? T(sizeof(T) * 8) : T(__builtin_clzg(U(x)));
  } else {
    T result;
#pragma unroll
    for (int i = 0; i < __vec_info<T>::lanes; ++i) {
      result[i] = clz(x[i]);
    }
    return result;
  }
}

template <typename T>
METAL_FUNC T ctz(T x) {
  if constexpr (__vec_info<T>::lanes == 1) {
    using U = make_unsigned_t<T>;
    return x == T(0) ? T(sizeof(T) * 8) : T(__builtin_ctzg(U(x)));
  } else {
    T result;
#pragma unroll
    for (int i = 0; i < __vec_info<T>::lanes; ++i) {
      result[i] = ctz(x[i]);
    }
    return result;
  }
}

template <typename T>
METAL_FUNC T reverse_bits(T x) {
  return __builtin_elementwise_bitreverse(x);
}

template <typename T>
METAL_FUNC T extract_bits(T x, uint offset, uint bits) {
  if (bits == 0) {
    return T(0);
  }
  using U = make_unsigned_t<__element_t<T>>;
  const U mask = bits >= sizeof(U) * 8 ? U(~U(0)) : U((U(1) << bits) - U(1));
  if constexpr (is_signed_v<__element_t<T>>) {
    const uint shift = uint(sizeof(U) * 8) - bits;
    return T(__element_t<T>(U(x >> offset) << shift) >> shift);
  } else {
    return T((x >> offset) & mask);
  }
}

template <typename T>
METAL_FUNC T insert_bits(T base, T insert, uint offset, uint bits) {
  using U = make_unsigned_t<__element_t<T>>;
  const U mask = bits >= sizeof(U) * 8 ? U(~U(0)) : U((U(1) << bits) - U(1));
  return T((base & ~(mask << offset)) | ((insert & mask) << offset));
}

template <typename T>
METAL_FUNC T mulhi(T a, T b) {
  if constexpr (sizeof(T) == 4) {
    if constexpr (is_signed_v<T>) {
      return T((long(a) * long(b)) >> 32);
    } else {
      return T((ulong(a) * ulong(b)) >> 32);
    }
  } else {
    static_assert(sizeof(T) == 4, "mulhi is implemented for 32-bit integers only");
  }
}

// --- geometric
template <typename T, int N>
METAL_FUNC T dot(vec<T, N> a, vec<T, N> b) {
  if constexpr (is_floating_point_v<T>) {
    const vec<float, N> p = __builtin_convertvector(a, vec<float, N>) * __builtin_convertvector(b, vec<float, N>);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      sum += p[i];
    }
    return T(sum);
  } else {
    const vec<T, N> p = a * b;
    T sum = T(0);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      sum += p[i];
    }
    return sum;
  }
}

template <typename T, int N>
METAL_FUNC T length(vec<T, N> a) {
  return sqrt(dot(a, a));
}

namespace fast {
using metal::abs;
using metal::clamp;
using metal::cos;
using metal::divide;
using metal::exp;
using metal::exp2;
using metal::fma;
using metal::fmax;
using metal::fmin;
using metal::log;
using metal::log2;
using metal::max;
using metal::min;
using metal::pow;
using metal::powr;
using metal::rsqrt;
using metal::sin;
using metal::sqrt;
using metal::tanh;
} // namespace fast

namespace precise {
using metal::abs;
using metal::clamp;
using metal::cos;
using metal::divide;
using metal::exp;
using metal::exp2;
using metal::fma;
using metal::fmax;
using metal::fmin;
using metal::log;
using metal::log2;
using metal::max;
using metal::min;
using metal::pow;
using metal::powr;
using metal::rsqrt;
using metal::sin;
using metal::sqrt;
using metal::tanh;
} // namespace precise

} // namespace metal
