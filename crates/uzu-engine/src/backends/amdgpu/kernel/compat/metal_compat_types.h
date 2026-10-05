// Types, address spaces and qualifiers of the Metal Shading Language on top of C++ for OpenCL.
#pragma once

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// --- address spaces: OpenCL C++ keeps MSL semantics (overloads and specializations by address space).
// `thread` maps to the generic address space rather than __private: in C++ for OpenCL `this` is __generic,
// and uzu returns `thread T&` to members from unqualified methods.
#define device __global
#define threadgroup __local
#define constant __constant
#define thread

// OpenCL C++ reserves `local` / `global` (aliases of __local / __global); uzu uses `local` as an identifier.
#define local __uzu_identifier_local
#define global __uzu_identifier_global

// --- floating point constants from <metal_math>
#define INFINITY __builtin_huge_valf()
#define NAN __builtin_nanf("")
#define HUGE_VALF __builtin_huge_valf()
#define MAXFLOAT 3.402823466e+38f
#define FLT_MAX 3.402823466e+38f
#define FLT_MIN 1.175494351e-38f
#define FLT_EPSILON 1.192092896e-07f
#define M_E_F 2.71828182845904523536f
#define M_LOG2E_F 1.44269504088896340736f
#define M_LOG10E_F 0.434294481903251827651f
#define M_LN2_F 0.693147180559945309417f
#define M_LN10_F 2.30258509299404568402f
#define M_PI_F 3.14159265358979323846f
#define M_PI_2_F 1.57079632679489661923f
#define M_PI_4_F 0.785398163397448309616f
#define M_1_PI_F 0.318309886183790671538f
#define M_2_PI_F 0.636619772367581343076f
#define M_2_SQRTPI_F 1.12837916709551257390f
#define M_SQRT2_F 1.41421356237309504880f
#define M_SQRT1_2_F 0.707106781186547524401f

#define METAL_FUNC inline __attribute__((always_inline))

// --- scalar types (normally provided by opencl-c-base.h, which is not included on purpose)
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef __bf16 bfloat;

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __INTPTR_TYPE__ intptr_t;
typedef __UINTPTR_TYPE__ uintptr_t;

// --- vector types
#define METAL_COMPAT_VECTORS(T)                                                                                       \
  typedef T T##2 __attribute__((ext_vector_type(2)));                                                                 \
  typedef T T##3 __attribute__((ext_vector_type(3)));                                                                 \
  typedef T T##4 __attribute__((ext_vector_type(4)));

METAL_COMPAT_VECTORS(char)
METAL_COMPAT_VECTORS(uchar)
METAL_COMPAT_VECTORS(short)
METAL_COMPAT_VECTORS(ushort)
METAL_COMPAT_VECTORS(int)
METAL_COMPAT_VECTORS(uint)
METAL_COMPAT_VECTORS(long)
METAL_COMPAT_VECTORS(ulong)
METAL_COMPAT_VECTORS(half)
METAL_COMPAT_VECTORS(bfloat)
METAL_COMPAT_VECTORS(float)
#undef METAL_COMPAT_VECTORS

// bool vectors are not valid OpenCL vector types: bool2..bool4 are structs in metal_compat_vector.h.
typedef signed char schar;

namespace metal {

template <typename T, int N>
using vec = T __attribute__((ext_vector_type(N)));

template <typename T>
struct __vec_lane_info {
  using element = T;
  static constexpr int lanes = 1;
};
template <typename T, int N>
struct __vec_lane_info<vec<T, N>> {
  using element = T;
  static constexpr int lanes = N;
};
// --- type traits (MSL exposes the <type_traits> subset under metal::)
template <typename T, T v>
struct integral_constant {
  static constexpr T value = v;
  using value_type = T;
  using type = integral_constant;
  constexpr operator value_type() const { return value; }
  constexpr value_type operator()() const { return value; }
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <bool B>
using bool_constant = integral_constant<bool, B>;

template <bool B, typename T = void>
struct enable_if {};
template <typename T>
struct enable_if<true, T> {
  using type = T;
};
template <bool B, typename T = void>
using enable_if_t = typename enable_if<B, T>::type;

template <bool B, typename T, typename F>
struct conditional {
  using type = T;
};
template <typename T, typename F>
struct conditional<false, T, F> {
  using type = F;
};
template <bool B, typename T, typename F>
using conditional_t = typename conditional<B, T, F>::type;

template <typename T, typename U>
struct is_same : false_type {};
template <typename T>
struct is_same<T, T> : true_type {};
template <typename T, typename U>
constexpr bool is_same_v = is_same<T, U>::value;

template <typename T>
struct remove_cv {
  using type = T;
};
template <typename T>
struct remove_cv<const T> {
  using type = T;
};
template <typename T>
struct remove_cv<volatile T> {
  using type = T;
};
template <typename T>
struct remove_cv<const volatile T> {
  using type = T;
};
template <typename T>
using remove_cv_t = typename remove_cv<T>::type;

template <typename T>
struct remove_reference {
  using type = T;
};
template <typename T>
struct remove_reference<T&> {
  using type = T;
};
template <typename T>
struct remove_reference<T&&> {
  using type = T;
};
template <typename T>
using remove_reference_t = typename remove_reference<T>::type;

template <typename T>
using remove_cvref_t = remove_cv_t<remove_reference_t<T>>;

template <typename T>
struct is_floating_point
    : bool_constant<is_same_v<remove_cv_t<T>, float> || is_same_v<remove_cv_t<T>, half> ||
                    is_same_v<remove_cv_t<T>, bfloat>> {};
template <typename T>
constexpr bool is_floating_point_v = is_floating_point<T>::value;

template <typename T>
struct is_integral
    : bool_constant<is_same_v<remove_cv_t<T>, bool> || is_same_v<remove_cv_t<T>, char> ||
                    is_same_v<remove_cv_t<T>, signed char> || is_same_v<remove_cv_t<T>, uchar> ||
                    is_same_v<remove_cv_t<T>, short> || is_same_v<remove_cv_t<T>, ushort> ||
                    is_same_v<remove_cv_t<T>, int> || is_same_v<remove_cv_t<T>, uint> ||
                    is_same_v<remove_cv_t<T>, long> || is_same_v<remove_cv_t<T>, ulong>> {};
template <typename T>
constexpr bool is_integral_v = is_integral<T>::value;

template <typename T>
struct is_signed : bool_constant<(T(-1) < T(0))> {};
template <typename T>
constexpr bool is_signed_v = is_signed<T>::value;

template <typename T>
struct is_unsigned : bool_constant<is_integral_v<T> && !(T(-1) < T(0))> {};
template <typename T>
constexpr bool is_unsigned_v = is_unsigned<T>::value;

template <typename T>
struct is_arithmetic : bool_constant<is_integral_v<T> || is_floating_point_v<T>> {};
template <typename T>
constexpr bool is_arithmetic_v = is_arithmetic<T>::value;

template <typename T>
struct make_unsigned;
template <>
struct make_unsigned<char> {
  using type = uchar;
};
template <>
struct make_unsigned<signed char> {
  using type = uchar;
};
template <>
struct make_unsigned<uchar> {
  using type = uchar;
};
template <>
struct make_unsigned<short> {
  using type = ushort;
};
template <>
struct make_unsigned<ushort> {
  using type = ushort;
};
template <>
struct make_unsigned<int> {
  using type = uint;
};
template <>
struct make_unsigned<uint> {
  using type = uint;
};
template <>
struct make_unsigned<long> {
  using type = ulong;
};
template <>
struct make_unsigned<ulong> {
  using type = ulong;
};
template <typename T>
using make_unsigned_t = typename make_unsigned<T>::type;

// --- integer_sequence
template <typename T, T... Is>
struct integer_sequence {
  using value_type = T;
  static constexpr size_t size() { return sizeof...(Is); }
};
template <size_t... Is>
using index_sequence = integer_sequence<size_t, Is...>;
template <typename T, T N>
using make_integer_sequence = __make_integer_seq<integer_sequence, T, N>;
template <size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

// --- array<T, N>
template <typename T, size_t N>
struct array {
  T elements_[N > 0 ? N : 1];
  // Unqualified methods are __generic: they serve thread, threadgroup and device objects alike.
  METAL_FUNC T& operator[](size_t i) { return elements_[i]; }
  METAL_FUNC const T& operator[](size_t i) const { return elements_[i]; }
  METAL_FUNC const constant T& operator[](size_t i) const constant { return elements_[i]; }
  METAL_FUNC static constexpr size_t size() { return N; }
};

// --- numeric_limits
template <typename T>
struct numeric_limits;

template <>
struct numeric_limits<float> {
  static constexpr bool has_infinity = true;
  static constexpr bool is_signed = true;
  static constexpr float infinity() { return __builtin_huge_valf(); }
  static constexpr float quiet_NaN() { return __builtin_nanf(""); }
  static constexpr float max() { return 3.402823466e+38f; }
  static constexpr float min() { return 1.175494351e-38f; }
  static constexpr float lowest() { return -3.402823466e+38f; }
  static constexpr float epsilon() { return 1.192092896e-07f; }
};

template <>
struct numeric_limits<half> {
  static constexpr bool has_infinity = true;
  static constexpr bool is_signed = true;
  static constexpr half infinity() { return __builtin_bit_cast(half, (ushort)0x7c00); }
  static constexpr half quiet_NaN() { return __builtin_bit_cast(half, (ushort)0x7e00); }
  static constexpr half max() { return __builtin_bit_cast(half, (ushort)0x7bff); }
  static constexpr half min() { return __builtin_bit_cast(half, (ushort)0x0400); }
  static constexpr half lowest() { return __builtin_bit_cast(half, (ushort)0xfbff); }
  static constexpr half epsilon() { return __builtin_bit_cast(half, (ushort)0x1400); }
};

template <>
struct numeric_limits<bfloat> {
  static constexpr bool has_infinity = true;
  static constexpr bool is_signed = true;
  static constexpr bfloat infinity() { return __builtin_bit_cast(bfloat, (ushort)0x7f80); }
  static constexpr bfloat quiet_NaN() { return __builtin_bit_cast(bfloat, (ushort)0x7fc0); }
  static constexpr bfloat max() { return __builtin_bit_cast(bfloat, (ushort)0x7f7f); }
  static constexpr bfloat min() { return __builtin_bit_cast(bfloat, (ushort)0x0080); }
  static constexpr bfloat lowest() { return __builtin_bit_cast(bfloat, (ushort)0xff7f); }
  static constexpr bfloat epsilon() { return __builtin_bit_cast(bfloat, (ushort)0x3c00); }
};

#define METAL_COMPAT_INT_LIMITS(T, MIN, MAX, SIGNED)                                                                  \
  template <>                                                                                                         \
  struct numeric_limits<T> {                                                                                          \
    static constexpr bool has_infinity = false;                                                                       \
    static constexpr bool is_signed = SIGNED;                                                                         \
    static constexpr T infinity() { return T(0); }                                                                   \
    static constexpr T max() { return MAX; }                                                                          \
    static constexpr T min() { return MIN; }                                                                          \
    static constexpr T lowest() { return MIN; }                                                                       \
  };

METAL_COMPAT_INT_LIMITS(bool, false, true, false)
METAL_COMPAT_INT_LIMITS(char, (char)-128, (char)127, true)
METAL_COMPAT_INT_LIMITS(schar, (schar)-128, (schar)127, true)
METAL_COMPAT_INT_LIMITS(uchar, (uchar)0, (uchar)255, false)
METAL_COMPAT_INT_LIMITS(short, (short)-32768, (short)32767, true)
METAL_COMPAT_INT_LIMITS(ushort, (ushort)0, (ushort)65535, false)
METAL_COMPAT_INT_LIMITS(int, -2147483647 - 1, 2147483647, true)
METAL_COMPAT_INT_LIMITS(uint, 0u, 4294967295u, false)
METAL_COMPAT_INT_LIMITS(long, -9223372036854775807L - 1, 9223372036854775807L, true)
METAL_COMPAT_INT_LIMITS(ulong, 0ul, 18446744073709551615ul, false)
#undef METAL_COMPAT_INT_LIMITS

// --- as_type: bit reinterpretation between same-sized types
template <typename To, typename From>
METAL_FUNC To as_type(From value) {
  static_assert(sizeof(To) == sizeof(From), "as_type requires types of equal size");
  return __builtin_bit_cast(To, value);
}

} // namespace metal
