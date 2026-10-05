// SIMD-group (wave32) and threadgroup synchronization of the Metal Shading Language.
#pragma once

namespace metal {

// --- barriers
enum class mem_flags : uint {
  mem_none = 0,
  mem_device = 1,
  mem_threadgroup = 2,
  mem_texture = 4,
  mem_threadgroup_imageblock = 8,
  mem_object_data = 16,
};

METAL_FUNC constexpr mem_flags operator|(mem_flags a, mem_flags b) { return mem_flags(uint(a) | uint(b)); }

// With mem_device the fences are device (agent) scope: a workgroup-scope release does not wait for the other
// waves' global stores to reach L2, and uzu's "last threadgroup to arrive" kernels (radix_top_k_small.metal)
// write device memory in all threads, barrier with mem_device, then let one thread publish through a
// device-scope atomic; the last threadgroup, on another CU, could read keys still in flight (an intermittent
// radix_top_k_small_matches_cpu failure once memory was reused).
METAL_FUNC void threadgroup_barrier(mem_flags flags) {
  if (uint(flags) & uint(mem_flags::mem_device)) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  } else {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  }
}

METAL_FUNC void simdgroup_barrier(mem_flags) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// --- lane helpers
METAL_FUNC uint __simd_lane_id() { return __builtin_amdgcn_mbcnt_lo(~0u, 0u); }

// ds_bpermute moves 32-bit words between lanes; wider and narrower types are split / widened.
METAL_FUNC int __simd_permute_i32(int value, uint source_lane) {
  return __builtin_amdgcn_ds_bpermute(int(source_lane << 2), value);
}

template <typename T>
struct __simd_permute_impl {
  static METAL_FUNC T permute(T value, uint source_lane) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                  "unsupported SIMD shuffle type");
    if constexpr (sizeof(T) == 8) {
      int2 words = __builtin_bit_cast(int2, value);
      words.x = __simd_permute_i32(words.x, source_lane);
      words.y = __simd_permute_i32(words.y, source_lane);
      return __builtin_bit_cast(T, words);
    } else if constexpr (sizeof(T) == 4) {
      return __builtin_bit_cast(T, __simd_permute_i32(__builtin_bit_cast(int, value), source_lane));
    } else if constexpr (sizeof(T) == 2) {
      int word = int(__builtin_bit_cast(ushort, value));
      return __builtin_bit_cast(T, ushort(__simd_permute_i32(word, source_lane)));
    } else {
      int word = int(__builtin_bit_cast(uchar, value));
      return __builtin_bit_cast(T, uchar(__simd_permute_i32(word, source_lane)));
    }
  }
};

template <typename T, int N>
struct __simd_permute_impl<vec<T, N>> {
  static METAL_FUNC vec<T, N> permute(vec<T, N> value, uint source_lane) {
    vec<T, N> result;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = __simd_permute_impl<T>::permute(value[i], source_lane);
    }
    return result;
  }
};

template <>
struct __simd_permute_impl<bool> {
  static METAL_FUNC bool permute(bool value, uint source_lane) {
    return __simd_permute_i32(int(value), source_lane) != 0;
  }
};

template <typename T>
METAL_FUNC T simd_shuffle(T value, ushort source_lane) {
  return __simd_permute_impl<T>::permute(value, uint(source_lane) & 31u);
}

template <typename T>
METAL_FUNC T simd_shuffle_xor(T value, ushort mask) {
  return __simd_permute_impl<T>::permute(value, (__simd_lane_id() ^ uint(mask)) & 31u);
}

// Lanes outside [0, 32) keep their own value, as in MSL.
template <typename T>
METAL_FUNC T simd_shuffle_down(T value, ushort delta) {
  const uint lane = __simd_lane_id();
  const uint source = lane + uint(delta);
  return __simd_permute_impl<T>::permute(value, source < 32u ? source : lane);
}

template <typename T>
METAL_FUNC T simd_shuffle_up(T value, ushort delta) {
  const uint lane = __simd_lane_id();
  return __simd_permute_impl<T>::permute(value, lane >= uint(delta) ? lane - uint(delta) : lane);
}

template <typename T>
METAL_FUNC T simd_shuffle_rotate_down(T value, ushort delta) {
  return __simd_permute_impl<T>::permute(value, (__simd_lane_id() + uint(delta)) & 31u);
}

template <typename T>
METAL_FUNC T simd_shuffle_rotate_up(T value, ushort delta) {
  return __simd_permute_impl<T>::permute(value, (__simd_lane_id() - uint(delta)) & 31u);
}

template <typename T>
METAL_FUNC T simd_broadcast(T value, ushort broadcast_lane) {
  return __simd_permute_impl<T>::permute(value, uint(broadcast_lane) & 31u);
}

// --- votes
METAL_FUNC uint __simd_active_mask() { return __builtin_amdgcn_read_exec_lo(); }

METAL_FUNC uint __simd_first_active_lane() { return uint(__builtin_ctz(__simd_active_mask())); }

template <typename T>
METAL_FUNC T simd_broadcast_first(T value) {
  return __simd_permute_impl<T>::permute(value, __simd_first_active_lane());
}

METAL_FUNC bool simd_is_first() { return __simd_lane_id() == __simd_first_active_lane(); }

struct simd_vote {
  using vote_t = ulong;
  vote_t v;
  METAL_FUNC explicit constexpr simd_vote(vote_t value = 0) : v(value) {}
  METAL_FUNC explicit constexpr operator vote_t() const { return v; }
  METAL_FUNC constexpr bool all() const { return v == 0xffffffffull; }
  METAL_FUNC constexpr bool any() const { return v != 0; }
};

METAL_FUNC simd_vote simd_ballot(bool predicate) {
  return simd_vote(ulong(__builtin_amdgcn_ballot_w32(predicate)));
}

METAL_FUNC bool simd_all(bool predicate) {
  return __builtin_amdgcn_ballot_w32(predicate) == __simd_active_mask();
}

METAL_FUNC bool simd_any(bool predicate) { return __builtin_amdgcn_ballot_w32(predicate) != 0u; }

// --- reductions (butterfly over all 32 lanes; every lane receives the result)
template <typename T>
METAL_FUNC T simd_sum(T value) {
#pragma unroll
  for (ushort mask = 16; mask > 0; mask >>= 1) {
    value += simd_shuffle_xor(value, mask);
  }
  return value;
}

template <typename T>
METAL_FUNC T simd_product(T value) {
#pragma unroll
  for (ushort mask = 16; mask > 0; mask >>= 1) {
    value *= simd_shuffle_xor(value, mask);
  }
  return value;
}

template <typename T>
METAL_FUNC T __simd_max_scalar(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
METAL_FUNC T __simd_min_scalar(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
METAL_FUNC T simd_max(T value) {
#pragma unroll
  for (ushort mask = 16; mask > 0; mask >>= 1) {
    const T other = simd_shuffle_xor(value, mask);
    value = __simd_max_scalar(value, other);
  }
  return value;
}

template <typename T>
METAL_FUNC T simd_min(T value) {
#pragma unroll
  for (ushort mask = 16; mask > 0; mask >>= 1) {
    const T other = simd_shuffle_xor(value, mask);
    value = __simd_min_scalar(value, other);
  }
  return value;
}

template <typename T>
METAL_FUNC T simd_and(T value) {
#pragma unroll
  for (ushort mask = 16; mask > 0; mask >>= 1) {
    value &= simd_shuffle_xor(value, mask);
  }
  return value;
}

template <typename T>
METAL_FUNC T simd_or(T value) {
#pragma unroll
  for (ushort mask = 16; mask > 0; mask >>= 1) {
    value |= simd_shuffle_xor(value, mask);
  }
  return value;
}

template <typename T>
METAL_FUNC T simd_xor(T value) {
#pragma unroll
  for (ushort mask = 16; mask > 0; mask >>= 1) {
    value ^= simd_shuffle_xor(value, mask);
  }
  return value;
}

// --- prefix scans (Hillis-Steele over 32 lanes)
template <typename T>
METAL_FUNC T simd_prefix_inclusive_sum(T value) {
  const uint lane = __simd_lane_id();
#pragma unroll
  for (ushort delta = 1; delta < 32; delta <<= 1) {
    const T other = simd_shuffle_up(value, delta);
    if (lane >= delta) {
      value += other;
    }
  }
  return value;
}

template <typename T>
METAL_FUNC T simd_prefix_exclusive_sum(T value) {
  const T inclusive = simd_prefix_inclusive_sum(value);
  return inclusive - value;
}

template <typename T>
METAL_FUNC T simd_prefix_inclusive_product(T value) {
  const uint lane = __simd_lane_id();
#pragma unroll
  for (ushort delta = 1; delta < 32; delta <<= 1) {
    const T other = simd_shuffle_up(value, delta);
    if (lane >= delta) {
      value *= other;
    }
  }
  return value;
}

template <typename T>
METAL_FUNC T simd_prefix_exclusive_product(T value) {
  const uint lane = __simd_lane_id();
  T shifted = simd_shuffle_up(value, ushort(1));
  if (lane == 0) {
    shifted = T(1);
  }
  return simd_prefix_inclusive_product(shifted);
}

// --- quad (groups of 4 lanes)
template <typename T>
METAL_FUNC T quad_shuffle(T value, ushort quad_lane) {
  return __simd_permute_impl<T>::permute(value, (__simd_lane_id() & ~3u) | (uint(quad_lane) & 3u));
}

template <typename T>
METAL_FUNC T quad_shuffle_xor(T value, ushort mask) {
  return __simd_permute_impl<T>::permute(value, (__simd_lane_id() ^ (uint(mask) & 3u)) & 31u);
}

template <typename T>
METAL_FUNC T quad_broadcast(T value, ushort quad_lane) {
  return quad_shuffle(value, quad_lane);
}

template <typename T>
METAL_FUNC T quad_sum(T value) {
  value += quad_shuffle_xor(value, ushort(1));
  value += quad_shuffle_xor(value, ushort(2));
  return value;
}

template <typename T>
METAL_FUNC T quad_max(T value) {
  value = __simd_max_scalar(value, quad_shuffle_xor(value, ushort(1)));
  value = __simd_max_scalar(value, quad_shuffle_xor(value, ushort(2)));
  return value;
}

} // namespace metal
