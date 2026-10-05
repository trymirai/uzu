// MSL vector constructors on top of clang ext_vector_type.
//
// MSL vectors are classes: `float4(a, b, c, d)`, `float4(float2, z, w)` and `float4(bfloat4)` are
// constructor calls. Clang ext vectors only accept a single-scalar functional cast (splat) or a
// same-size bitcast. Each `<type>N(` call is therefore routed through a function-like macro to
// __make_vec, which implements the MSL semantics (splat, concatenation, element-wise conversion).
// Plain uses of the type names (declarations, template arguments) are not followed by `(`, so the
// function-like macros leave them alone.
#pragma once

namespace metal {

template <typename T>
struct __is_ext_vector : false_type {};
template <typename T, int N>
struct __is_ext_vector<vec<T, N>> : true_type {};

template <typename E, int N, typename A>
METAL_FUNC void __make_vec_append(thread vec<E, N>& result, thread int& index, A arg) {
  if constexpr (__is_ext_vector<A>::value) {
    using AE = typename __vec_lane_info<A>::element;
    constexpr int lanes = __vec_lane_info<A>::lanes;
#pragma unroll
    for (int i = 0; i < lanes; ++i) {
      result[index++] = E(AE(arg[i]));
    }
  } else {
    result[index++] = E(arg);
  }
}

template <typename E, int N, typename... Args>
METAL_FUNC vec<E, N> __make_vec(Args... args) {
  if constexpr (sizeof...(Args) == 0) {
    return vec<E, N>(E(0));
  } else if constexpr (sizeof...(Args) == 1 && !(__is_ext_vector<Args>::value || ...)) {
    // splat
    return vec<E, N>(E(args...));
  } else {
    vec<E, N> result;
    int index = 0;
    (__make_vec_append<E, N>(result, index, args), ...);
    return result;
  }
}

} // namespace metal

#define char2(...) ::metal::__make_vec<char, 2>(__VA_ARGS__)
#define char3(...) ::metal::__make_vec<char, 3>(__VA_ARGS__)
#define char4(...) ::metal::__make_vec<char, 4>(__VA_ARGS__)
#define uchar2(...) ::metal::__make_vec<uchar, 2>(__VA_ARGS__)
#define uchar3(...) ::metal::__make_vec<uchar, 3>(__VA_ARGS__)
#define uchar4(...) ::metal::__make_vec<uchar, 4>(__VA_ARGS__)
#define short2(...) ::metal::__make_vec<short, 2>(__VA_ARGS__)
#define short3(...) ::metal::__make_vec<short, 3>(__VA_ARGS__)
#define short4(...) ::metal::__make_vec<short, 4>(__VA_ARGS__)
#define ushort2(...) ::metal::__make_vec<ushort, 2>(__VA_ARGS__)
#define ushort3(...) ::metal::__make_vec<ushort, 3>(__VA_ARGS__)
#define ushort4(...) ::metal::__make_vec<ushort, 4>(__VA_ARGS__)
#define int2(...) ::metal::__make_vec<int, 2>(__VA_ARGS__)
#define int3(...) ::metal::__make_vec<int, 3>(__VA_ARGS__)
#define int4(...) ::metal::__make_vec<int, 4>(__VA_ARGS__)
#define uint2(...) ::metal::__make_vec<uint, 2>(__VA_ARGS__)
#define uint3(...) ::metal::__make_vec<uint, 3>(__VA_ARGS__)
#define uint4(...) ::metal::__make_vec<uint, 4>(__VA_ARGS__)
#define long2(...) ::metal::__make_vec<long, 2>(__VA_ARGS__)
#define long3(...) ::metal::__make_vec<long, 3>(__VA_ARGS__)
#define long4(...) ::metal::__make_vec<long, 4>(__VA_ARGS__)
#define ulong2(...) ::metal::__make_vec<ulong, 2>(__VA_ARGS__)
#define ulong3(...) ::metal::__make_vec<ulong, 3>(__VA_ARGS__)
#define ulong4(...) ::metal::__make_vec<ulong, 4>(__VA_ARGS__)
#define half2(...) ::metal::__make_vec<half, 2>(__VA_ARGS__)
#define half3(...) ::metal::__make_vec<half, 3>(__VA_ARGS__)
#define half4(...) ::metal::__make_vec<half, 4>(__VA_ARGS__)
#define bfloat2(...) ::metal::__make_vec<bfloat, 2>(__VA_ARGS__)
#define bfloat3(...) ::metal::__make_vec<bfloat, 3>(__VA_ARGS__)
#define bfloat4(...) ::metal::__make_vec<bfloat, 4>(__VA_ARGS__)
#define float2(...) ::metal::__make_vec<float, 2>(__VA_ARGS__)
#define float3(...) ::metal::__make_vec<float, 3>(__VA_ARGS__)
#define float4(...) ::metal::__make_vec<float, 4>(__VA_ARGS__)

// uzu's `vector_cast<To>(value)` (common/defines.h): element-wise conversion between vector types.
#define UZU_HAS_VECTOR_CAST
template <typename To, typename From>
METAL_FUNC To vector_cast(From value) {
  return __builtin_convertvector(value, To);
}

namespace metal {

// MSL bool vectors. OpenCL vector comparisons yield integer vectors (-1 / 0 per lane); a bool vector
// is built from such a mask (or splatted from a bool) and indexed by select / all / any.
template <int N>
struct __bool_vector {
  bool lanes_[N];

  __bool_vector() = default;

  METAL_FUNC __bool_vector(bool value) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      lanes_[i] = value;
    }
  }

  template <typename V, typename = enable_if_t<__is_ext_vector<V>::value>>
  METAL_FUNC __bool_vector(V mask) {
    static_assert(__vec_lane_info<V>::lanes == N, "bool vector width mismatch");
#pragma unroll
    for (int i = 0; i < N; ++i) {
      lanes_[i] = mask[i] != 0;
    }
  }

  METAL_FUNC bool operator[](int i) const { return lanes_[i]; }
  METAL_FUNC bool& operator[](int i) { return lanes_[i]; }
};

template <int N>
METAL_FUNC bool all(__bool_vector<N> x) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    if (!x[i]) {
      return false;
    }
  }
  return true;
}

template <int N>
METAL_FUNC bool any(__bool_vector<N> x) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    if (x[i]) {
      return true;
    }
  }
  return false;
}

// all / any over vector comparison results (OpenCL vector comparisons yield -1 / 0 per lane)
template <typename T>
METAL_FUNC bool all(T x) {
  if constexpr (__is_ext_vector<T>::value) {
#pragma unroll
    for (int i = 0; i < __vec_lane_info<T>::lanes; ++i) {
      if (!x[i]) {
        return false;
      }
    }
    return true;
  } else {
    return bool(x);
  }
}

template <typename T>
METAL_FUNC bool any(T x) {
  if constexpr (__is_ext_vector<T>::value) {
#pragma unroll
    for (int i = 0; i < __vec_lane_info<T>::lanes; ++i) {
      if (x[i]) {
        return true;
      }
    }
    return false;
  } else {
    return bool(x);
  }
}

} // namespace metal

typedef metal::__bool_vector<2> bool2;
typedef metal::__bool_vector<3> bool3;
typedef metal::__bool_vector<4> bool4;
