#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

#include "mma.h"

// Avoids air.convert (SFU) for int → float which is slower on Apple GPU.
template <typename U>
METAL_FUNC U uint_to_fp(uint32_t x) {
  return static_cast<U>(as_type<float>(x | 0x4B000000u) - 8388608.0f);
}

template <>
METAL_FUNC bfloat uint_to_fp<bfloat>(uint32_t x) {
  return as_type<bfloat>(uint16_t(x | 0x4300u)) - bfloat(128.0f);
}

// Unpack 4 lanes of `BITS`-wide packed uints (low bits of each uint) into
// vec<U, 4> of floats. `BITS` in [1, 23] to fit the float23 mantissa trick.
template <int BITS>
METAL_FUNC float4 _uint4_to_fp4_float(uint4 n) {
  static_assert(BITS > 0 && BITS <= 23, "BITS must fit in float23 mantissa");
  constexpr uint mask = (1u << BITS) - 1u;
  n &= uint4(mask);
  return as_type<float4>(n | uint4(0x4B000000u)) - float4(8388608.0f);
}

template <typename U, int BITS>
METAL_FUNC vec<U, 4> uint4_to_fp4(uint4 n);

template <>
METAL_FUNC float4 uint4_to_fp4<float, 4>(uint4 n) {
  return _uint4_to_fp4_float<4>(n);
}

template <>
METAL_FUNC float4 uint4_to_fp4<float, 8>(uint4 n) {
  return _uint4_to_fp4_float<8>(n);
}

template <>
METAL_FUNC half4 uint4_to_fp4<half, 4>(uint4 n) {
  return half4(_uint4_to_fp4_float<4>(n));
}

template <>
METAL_FUNC half4 uint4_to_fp4<half, 8>(uint4 n) {
  return half4(_uint4_to_fp4_float<8>(n));
}

template <>
METAL_FUNC bfloat4 uint4_to_fp4<bfloat, 4>(uint4 n) {
  return bfloat4(_uint4_to_fp4_float<4>(n));
}

template <>
METAL_FUNC bfloat4 uint4_to_fp4<bfloat, 8>(uint4 n) {
  return bfloat4(_uint4_to_fp4_float<8>(n));
}

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  using U4 = vec<U, 4>;
  U sum = 0;
  thread U4* x4 = (thread U4*)x_thread;
  for (int i = 0; i < values_per_thread / 4; i++) {
    U4 v = U4(x[4 * i], x[4 * i + 1], x[4 * i + 2], x[4 * i + 3]);
    sum += v[0] + v[1] + v[2] + v[3];
    x4[i] = v;
  }
  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  U sum = 0;
  for (int i = 0; i < values_per_thread; ++i) {
    x_thread[i] = 0;
  }
  for (int i = 0; i < N; ++i) {
    U v = x[i];
    sum += v;
    x_thread[i] = v;
  }
  return sum;
}

template <typename U, int values_per_thread, int bits>
inline void qouter(
    const thread uint8_t* w,
    U x,
    U scale,
    U bias,
    thread U* result
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  if (bits == 4) {
    for (int i = 0; i < (values_per_thread / 2); i++) {
      result[2 * i] += x * (scale * uint_to_fp<U>(w[i] & 0x0fu) + bias);
      result[2 * i + 1] +=
          x * (scale * uint_to_fp<U>((w[i] >> 4) & 0x0fu) + bias);
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      result[i] += x * (scale * w[i] + bias);
    }
  }
}

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  U accum = 0;
  if (bits == 4) {
    using U4 = vec<U, 4>;
    const device ushort* ws = (const device ushort*)w;
    const thread U4* x4 = (const thread U4*)x_thread;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint wi = ws[i];
      U4 w_vec = uint4_to_fp4<U, 4>(uint4(wi, wi >> 4, wi >> 8, wi >> 12));
      accum += dot(x4[i], w_vec);
    }
  } else if (bits == 8) {
    using U4 = vec<U, 4>;
    const device uint* ws = (const device uint*)w;
    const thread U4* x4 = (const thread U4*)x_thread;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint wi = ws[i];
      U4 w_vec = uint4_to_fp4<U, 8>(uint4(wi, wi >> 8, wi >> 16, wi >> 24));
      accum += dot(x4[i], w_vec);
    }
  }
  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  U accum = 0;
  if (bits == 4) {
    using U4 = vec<U, 4>;
    const device uint16_t* ws = (const device uint16_t*)w;
    const thread U4* x4 = (const thread U4*)x_thread;

    int full = N / 4;
    for (int i = 0; i < full; i++) {
      uint16_t wi = ws[i];
      U4 w_vec = uint4_to_fp4<U, 4>(uint4(wi, wi >> 4, wi >> 8, wi >> 12));
      accum += dot(x4[i], w_vec);
    }

    int rem = N & 3;
    if (rem > 0) {
      uint16_t wv = ws[full];
      int base = 4 * full;
      if (rem > 0)
        accum += x_thread[base] * uint_to_fp<U>(wv & 0xf);
      if (rem > 1)
        accum += x_thread[base + 1] * uint_to_fp<U>((wv >> 4) & 0xf);
      if (rem > 2)
        accum += x_thread[base + 2] * uint_to_fp<U>((wv >> 8) & 0xf);
    }
  } else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int N, int bits>
inline void dequantize(
    const device uint8_t* w,
    U scale,
    U bias,
    threadgroup U* w_local
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  if (bits == 4) {
    U s0 = scale;
    U s1 = scale / static_cast<U>(16.0f);
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = s0 * (w[i] & 0x0f) + bias;
      w_local[2 * i + 1] = s1 * (w[i] & 0xf0) + bias;
    }
  } else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = scale * w[i] + bias;
    }
  }
}

template <>
inline void dequantize<bfloat, 8, 4>(
    const device uint8_t* w,
    bfloat scale,
    bfloat bias,
    threadgroup bfloat* w_local
) {
  const device uint32_t* w_ptr = (const device uint32_t*)w;
  uint32_t packed = *w_ptr;

  bfloat4 v0, v1;

  // Low 4 nibbles
  v0.x = static_cast<bfloat>(packed & 0xF);
  v0.y = static_cast<bfloat>((packed >> 4) & 0xF);
  v0.z = static_cast<bfloat>((packed >> 8) & 0xF);
  v0.w = static_cast<bfloat>((packed >> 12) & 0xF);

  // High 4 nibbles
  v1.x = static_cast<bfloat>((packed >> 16) & 0xF);
  v1.y = static_cast<bfloat>((packed >> 20) & 0xF);
  v1.z = static_cast<bfloat>((packed >> 24) & 0xF);
  v1.w = static_cast<bfloat>((packed >> 28) & 0xF);

  v0 = v0 * scale + bias;
  v1 = v1 * scale + bias;

  threadgroup bfloat4* out_ptr = (threadgroup bfloat4*)w_local;
  out_ptr[0] = v0;
  out_ptr[1] = v1;
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoaderMlx {
  static_assert(
      BCOLS <= group_size,
      "Group size should be larger than columns"
  );
  static_assert(
      group_size % BCOLS == 0,
      "Group size should be divisible by columns"
  );
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  METAL_CONST short pack_factor = get_pack_factor<bits, 8>();
  METAL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  METAL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  METAL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  METAL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoaderMlx(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  )
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor
        ),
        group_step_cnt(0), group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack,
          scale,
          bias,
          dst + i * pack_factor
      );
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor
      );
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits,
    bool per_output_layout = false>
struct QuantizedBlockLoaderZp {
  static_assert(
      BCOLS <= group_size,
      "Group size should be larger than columns"
  );
  static_assert(
      group_size % BCOLS == 0,
      "Group size should be divisible by columns"
  );
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  METAL_CONST short pack_factor = get_pack_factor<bits, 8>();
  METAL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  METAL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  METAL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  METAL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int groups_per_row;
  const int tile_stride;
  short group_step_cnt;
  int k_base;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* scales_row_start;
  const device uint8_t* zps_row_start;
  const int out_group_base;
  const int out_groups_total;
  const int zp_stride_total;

  QuantizedBlockLoaderZp(
      const device uint8_t* src_,
      const device T* scales_,
      const device uint8_t* zero_points_row_start_,
      const int src_ld_,
      const int groups_per_row_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      const int out_group_base_ = 0,
      const int out_groups_total_ = 0,
      const int zp_stride_total_ = 0
  )
      : src_ld(src_ld_), groups_per_row(groups_per_row_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor
        ),
        group_step_cnt(0), k_base(0), group_stride(BROWS * groups_per_row_),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(reduction_dim == 1 ? (scales_ + bi * groups_per_row_) : scales_),
        scales_row_start(
            reduction_dim == 1 ? (scales_ + bi * groups_per_row_) : scales_
        ),
        zps_row_start(
            reduction_dim == 1 ? (zero_points_row_start_ +
                                  bi * (bits == 4 ? ((groups_per_row_ + 1) / 2)
                                                  : groups_per_row_))
                               : zero_points_row_start_
        ),
        out_group_base(per_output_layout ? out_group_base_ : 0),
        out_groups_total(per_output_layout ? out_groups_total_ : 0),
        zp_stride_total(per_output_layout ? zp_stride_total_ : 0) {}

  inline void current_scale_bias(
      thread T& out_scale,
      thread T& out_bias
  ) const {
    uint zp_n;
    T scale_val;
    if (per_output_layout) {
      const int row_idx = k_base + bi;
      const int scale_index = row_idx * groups_per_row + out_group_base;
      scale_val = scales_row_start[scale_index];
      if (bits == 4) {
        const int byte_index =
            row_idx * zp_stride_total + (out_group_base >> 1);
        uint8_t zp_b = zps_row_start[byte_index];
        zp_n = (uint(zp_b) >> (uint(out_group_base & 1) * 4u)) & 0x0Fu;
      } else {
        const int zp_index = row_idx * zp_stride_total + out_group_base;
        zp_n = zps_row_start[zp_index];
      }
    } else {
      int g = reduction_dim == 0 ? (k_base / group_size)
                                 : (int)(scales - scales_row_start);
      scale_val = reduction_dim == 0 ? scales_row_start[g] : *scales;
      if (bits == 4) {
        const device uint8_t* zp_ptr = zps_row_start + (g >> 1);
        uint8_t zp_b = *zp_ptr;
        zp_n = (uint(zp_b) >> (uint(g & 1) * 4u)) & 0x0Fu;
      } else {
        zp_n = zps_row_start[g];
      }
    }
    out_scale = scale_val;
    out_bias = static_cast<T>(-scale_val * static_cast<T>(zp_n));
  }

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale;
    T bias;
    current_scale_bias(scale, bias);
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack,
          scale,
          bias,
          dst + i * pack_factor
      );
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1) {
      // N-tail: zero out rows beyond valid outputs
      if (bi >= src_tile_dim.x) {
        for (int i = 0; i < n_reads * pack_factor; i++) {
          dst[i] = T(0);
        }
        return;
      }

      int valid_cols = src_tile_dim.y; // 0..BK
      int valid_packs = (valid_cols + pack_factor - 1) / pack_factor;

      T scale;
      T bias;
      current_scale_bias(scale, bias);
      for (int i = 0; i < n_reads; i++) {
        int pack_idx = bj + i; // global pack index across the BK packs
        if (pack_idx < valid_packs) {
          dequantize<T, pack_factor, bits>(
              src + i * bytes_per_pack,
              scale,
              bias,
              dst + i * pack_factor
          );

          // Mask the last pack if needed
          if (pack_idx == valid_packs - 1) {
            int rem = valid_cols - pack_idx * pack_factor;
            if (rem < pack_factor) {
              for (int r = rem; r < pack_factor; ++r) {
                dst[i * pack_factor + r] = T(0);
              }
            }
          }
        } else {
          for (int j = 0; j < pack_factor; ++j) {
            dst[i * pack_factor + j] = T(0);
          }
        }
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale;
    T bias;
    current_scale_bias(scale, bias);
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack,
          scale,
          bias,
          dst + i * pack_factor
      );
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
        }
      } else {
        scales++;
      }
    } else {
      k_base += BROWS;
    }
  }
};

template <
    typename LoaderW,
    typename LoaderX,
    typename Mma,
    typename T,
    const bool aligned_N,
    const int BM,
    const int BK,
    const int BN>
inline void qmm_transposed_core(
    thread LoaderX& loader_x,
    thread LoaderW& loader_w,
    thread Mma& mma_op,
    const short num_els,
    const short num_outs,
    const int in_vec_size,
    device T* output,
    const int out_vec_size,
    threadgroup T* Xs,
    threadgroup T* Ws
) {
  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < in_vec_size; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < in_vec_size; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < in_vec_size; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < in_vec_size; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(output, out_vec_size, short2(num_outs, num_els));
  } else {
    mma_op.store_result(output, out_vec_size);
  }
}

template <
    typename T,
    const uint group_size,
    const uint bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32,
    const bool use_mlx_quant = false,
    const int WM = 2,
    const int WN = 2>
void qmm_transposed_impl(
    const device uint32_t* weights,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* input,
    device T* output,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const int in_vec_size,
    const int out_vec_size,
    const int batch_size,
    uint out_block_idx,
    uint batch_block_idx,
    uint simd_group,
    uint simd_lane
) {
  static_assert(BK >= 32, "BK should be larger than METAL_SIMD_SIZE");
  static_assert(BK % 32 == 0, "BK should be divisible by METAL_SIMD_SIZE");
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = matmul_utils::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      matmul_utils::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * 32>;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = (in_vec_size + group_size - 1) / group_size;
  const int out_row = batch_block_idx * BM;
  const int out_col = out_block_idx * BN;

  auto wl = (const device uint8_t*)weights;

  const device T* x_block = input + out_row * static_cast<int64_t>(in_vec_size);
  const device uint8_t* w_block = wl + out_col * in_vec_size_w;
  scales += out_col * in_vec_size_g;
  biases += out_col * in_vec_size_g;
  device T* y_block =
      output + out_row * static_cast<int64_t>(out_vec_size) + out_col;

  const short num_els = min(BM, batch_size - out_row);
  const short num_outs = min(BN, out_vec_size - out_col);
  loader_x_t loader_x(x_block, in_vec_size, Xs, simd_group, simd_lane);
  mma_t mma_op(simd_group, simd_lane);

  if (use_mlx_quant) {
    using loader_w_t = QuantizedBlockLoaderMlx<
        T,
        BN,
        BK,
        BK_padded,
        1,
        WM * WN * 32,
        group_size,
        bits>;
    loader_w_t loader_w(
        w_block,
        scales,
        biases,
        in_vec_size,
        Ws,
        simd_group,
        simd_lane
    );
    qmm_transposed_core<
        loader_w_t,
        loader_x_t,
        mma_t,
        T,
        aligned_N,
        BM,
        BK,
        BN>(
        loader_x,
        loader_w,
        mma_op,
        num_els,
        num_outs,
        in_vec_size,
        y_block,
        out_vec_size,
        Xs,
        Ws
    );
  } else {
    using loader_w_t = QuantizedBlockLoaderZp<
        T,
        BN,
        BK,
        BK_padded,
        1,
        WM * WN * 32,
        group_size,
        bits>;
    const device uint8_t* zero_points_row =
        zero_points +
        out_col * (bits == 4 ? ((in_vec_size_g + 1) / 2) : in_vec_size_g);
    loader_w_t loader_w(
        w_block,
        scales,
        zero_points_row,
        in_vec_size,
        in_vec_size_g,
        Ws,
        simd_group,
        simd_lane
    );
    qmm_transposed_core<
        loader_w_t,
        loader_x_t,
        mma_t,
        T,
        aligned_N,
        BM,
        BK,
        BN>(
        loader_x,
        loader_w,
        mma_op,
        num_els,
        num_outs,
        in_vec_size,
        y_block,
        out_vec_size,
        Xs,
        Ws
    );
  }
}
