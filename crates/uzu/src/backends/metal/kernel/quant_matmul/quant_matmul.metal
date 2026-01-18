#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

#include "mma.h"
#include "../common/mlp_epilogue.h"

// Function constant: true = MLX-style (pre-computed biases), false = AWQ-style
// (zero-points)
constant bool kUseMlxQuant [[function_constant(40)]];
constant bool kUseZeroPoints = !kUseMlxQuant;

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

  U sum = 0;
  if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  U sum = 0;
  if (bits == 4) {
    const U scale_lut[4] = {
        static_cast<U>(1.0f),
        static_cast<U>(1.0f / 16.0f),
        static_cast<U>(1.0f / 256.0f),
        static_cast<U>(1.0f / 4096.0f)
    };

    for (int i = 0; i < values_per_thread; ++i) {
      x_thread[i] = 0;
    }
    for (int i = 0; i < N; ++i) {
      U v = x[i];
      sum += v;
      x_thread[i] = v * scale_lut[i & 3];
    }
  } else if (bits == 8) {
    for (int i = 0; i < N; ++i) {
      U v = x[i];
      sum += v;
      x_thread[i] = v;
    }
    for (int i = N; i < values_per_thread; ++i) {
      x_thread[i] = 0;
    }
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
    U s0 = scale;
    U s1 = scale / 16.0f;
    for (int i = 0; i < (values_per_thread / 2); i++) {
      result[2 * i] += x * (s0 * (w[i] & 0x0f) + bias);
      result[2 * i + 1] += x * (s1 * (w[i] & 0xf0) + bias);
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
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }
  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U qdot_zero_point(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U zero_point
) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  U accum = 0;
  if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    const uint16_t zp0 = static_cast<uint16_t>(zero_point);
    const uint16_t zp1 = static_cast<uint16_t>(zero_point) << 4;
    const uint16_t zp2 = static_cast<uint16_t>(zero_point) << 8;
    const uint16_t zp3 = static_cast<uint16_t>(zero_point) << 12;

    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint16_t word = ws[i];
      accum += x_thread[4 * i] *
               static_cast<U>(
                   static_cast<int>(word & 0x000f) - static_cast<int>(zp0)
               );
      accum += x_thread[4 * i + 1] *
               static_cast<U>(
                   static_cast<int>(word & 0x00f0) - static_cast<int>(zp1)
               );
      accum += x_thread[4 * i + 2] *
               static_cast<U>(
                   static_cast<int>(word & 0x0f00) - static_cast<int>(zp2)
               );
      accum += x_thread[4 * i + 3] *
               static_cast<U>(
                   static_cast<int>(word & 0xf000) - static_cast<int>(zp3)
               );
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * (static_cast<U>(w[i]) - zero_point);
    }
  }
  return scale * accum;
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
    const device uint16_t* ws = (const device uint16_t*)w;

    int full = N / 4;
    for (int i = 0; i < full; i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }

    int rem = N & 3;
    if (rem > 0) {
      uint16_t wv = ws[full];
      int base = 4 * full;
      if (rem > 0)
        accum += x_thread[base] * (wv & 0x000f);
      if (rem > 1)
        accum += x_thread[base + 1] * (wv & 0x00f0);
      if (rem > 2)
        accum += x_thread[base + 2] * (wv & 0x0f00);
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

  UZU_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  UZU_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  UZU_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  UZU_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  UZU_MTL_CONST short group_steps = group_size / BCOLS;

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

  UZU_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  UZU_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  UZU_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  UZU_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  UZU_MTL_CONST short group_steps = group_size / BCOLS;

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
        zp_n = (out_group_base & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F);
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
        zp_n = (g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F);
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
    const bool aligned_K,
    const int BM,
    const int BK,
    const int BN>
inline void qmm_core(
    thread LoaderX& loader_x,
    thread LoaderW& loader_w,
    thread Mma& mma_op,
    const short num_els,
    const short num_outs,
    const int K,
    device T* y,
    const constant int& N,
    threadgroup T* Xs,
    threadgroup T* Ws,
    uint lid [[thread_index_in_threadgroup]]
) {
  if (num_els < BM) {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, num_els));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
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
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, BM));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
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
  if (num_els < BM) {
    mma_op.store_result_safe(y, N, short2(BN, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_K = false,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
void qmm_impl(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  static_assert(BK >= 32, "BK should be larger than SIMD_SIZE");
  static_assert(BK % 32 == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = matmul_utils::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, false, BK_padded, BN_padded>;
  using loader_x_t =
      matmul_utils::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * 32, 1, 4>;

  auto wl = (const device uint8_t*)w;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * bytes_per_pack / pack_factor;
  y += y_row * static_cast<int64_t>(N) + y_col;

  const device T* scales_base = scales;
  const device T* biases_base = biases;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (kUseMlxQuant) {
    using loader_w_t = QuantizedBlockLoaderMlx<
        T,
        BK,
        BN,
        BN_padded,
        0,
        WM * WN * 32,
        group_size,
        bits>;
    const device T* scales_mlx = scales_base + (y_col / group_size);
    const device T* biases_mlx = biases_base + (y_col / group_size);
    loader_w_t loader_w(wl, scales_mlx, biases_mlx, N, Ws, simd_gid, simd_lid);
    qmm_core<loader_w_t, loader_x_t, mma_t, T, aligned_K, BM, BK, BN>(
        loader_x,
        loader_w,
        mma_op,
        num_els,
        num_outs,
        K,
        y,
        N,
        Xs,
        Ws,
        lid
    );
  } else {
    const int out_groups_total = (N + group_size - 1) / group_size;
    const int groups_per_row = out_groups_total;
    const int out_group = y_col / group_size;
    const int zp_stride_out =
        (bits == 4) ? ((out_groups_total + 1) / 2) : out_groups_total;
    using loader_w_t = QuantizedBlockLoaderZp<
        T,
        BK,
        BN,
        BN_padded,
        0,
        WM * WN * 32,
        group_size,
        bits,
        true>;
    loader_w_t loader_w(
        wl,
        scales_base,
        zero_points,
        N,
        groups_per_row,
        Ws,
        simd_gid,
        simd_lid,
        out_group,
        out_groups_total,
        zp_stride_out
    );
    qmm_core<loader_w_t, loader_x_t, mma_t, T, aligned_K, BM, BK, BN>(
        loader_x,
        loader_w,
        mma_op,
        num_els,
        num_outs,
        K,
        y,
        N,
        Xs,
        Ws,
        lid
    );
  }
}

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
    const int K,
    device T* y,
    const constant int& N,
    threadgroup T* Xs,
    threadgroup T* Ws,
    uint lid [[thread_index_in_threadgroup]]
) {
  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K; k += BK) {
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
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K; k += BK) {
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
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
void qmm_transposed_impl(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  static_assert(BK >= 32, "BK should be larger than SIMD_SIZE");
  static_assert(BK % 32 == 0, "BK should be divisible by SIMD_SIZE");

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = matmul_utils::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      matmul_utils::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * 32>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = (K + group_size - 1) / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  const device T* x_block = x + y_row * static_cast<int64_t>(K);
  const device uint8_t* w_block = wl + y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  device T* y_block = y + y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x_block, K, Xs, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (kUseMlxQuant) {
    using loader_w_t = QuantizedBlockLoaderMlx<
        T,
        BN,
        BK,
        BK_padded,
        1,
        WM * WN * 32,
        group_size,
        bits>;
    loader_w_t loader_w(w_block, scales, biases, K, Ws, simd_gid, simd_lid);
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
        K,
        y_block,
        N,
        Xs,
        Ws,
        lid
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
        zero_points + y_col * (bits == 4 ? ((K_g + 1) / 2) : K_g);
    loader_w_t loader_w(
        w_block,
        scales,
        zero_points_row,
        K,
        K_g,
        Ws,
        simd_gid,
        simd_lid
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
        K,
        y_block,
        N,
        Xs,
        Ws,
        lid
    );
  }
}

template <typename T, int group_size, int bits, bool UseMlx>
void qmv_impl_dispatch(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& K, // in_vec_size
    const constant int& N, // out_vec_size
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = K * bytes_per_pack / pack_factor;
  const int in_vec_size_g =
      (K + group_size - 1) / group_size; // ceil(K / group_size)
  const device T* scales_base = scales;
  const device uint8_t* zero_points_base = zero_points;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                      simd_gid * results_per_simdgroup;
  const int used_out_row = min(N - results_per_simdgroup, out_row);

  if (out_row >= N) {
    return;
  }

  if (N < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

    const int zp_stride = bits == 4 ? ((in_vec_size_g + 1) / 2) : in_vec_size_g;
    const device uint8_t* zps_row_base = nullptr;
    const device T* biases_row_base = nullptr;
    if (UseMlx) {
      biases_row_base = biases + out_row * in_vec_size_g;
    } else {
      zps_row_base = zero_points + out_row * zp_stride;
    }

    x += tid.x * K + simd_lid * values_per_thread;
    y += tid.x * N + out_row;

    int k = 0;
    for (; k < K - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; out_row + row < N; row++) {
        if (row >= results_per_simdgroup)
          break;
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const int row_idx = out_row + row;
        const device T* sr = scales_base + row_idx * in_vec_size_g;

        int g = (k + simd_lid * values_per_thread) / group_size;
        U s = static_cast<U>(sr[g]);
        if (UseMlx) {
          const device T* bl = biases_row_base + row * in_vec_size_g;
          U b = static_cast<U>(bl[g]);
          result[row] +=
              qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
        } else {
          const device uint8_t* zl = zps_row_base + row * zp_stride;
          U zp;
          if (bits == 4) {
            uint8_t zp_b = zl[g >> 1];
            zp = static_cast<U>((g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
          } else {
            zp = static_cast<U>(zl[g]);
          }
          result[row] +=
              qdot_zero_point<U, values_per_thread, bits>(wl, x_thread, s, zp);
        }
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(K - k - simd_lid * values_per_thread),
        0,
        values_per_thread
    );
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x,
          x_thread,
          remaining
      );

      for (int row = 0; out_row + row < N; row++) {
        if (row >= results_per_simdgroup)
          break;
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const int row_idx = out_row + row;
        const device T* sr = scales_base + row_idx * in_vec_size_g;

        int g = (k + simd_lid * values_per_thread) / group_size;
        U s = static_cast<U>(sr[g]);
        if (UseMlx) {
          const device T* bl = biases_row_base + row * in_vec_size_g;
          U b = static_cast<U>(bl[g]);
          result[row] +=
              qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
        } else {
          const device uint8_t* zl = zps_row_base + row * zp_stride;
          U zp;
          if (bits == 4) {
            uint8_t zp_b = zl[g >> 1];
            zp = static_cast<U>((g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
          } else {
            zp = static_cast<U>(zl[g]);
          }
          result[row] +=
              qdot_zero_point<U, values_per_thread, bits>(wl, x_thread, s, zp);
        }
      }
    }

    for (int row = 0; out_row + row < N; row++) {
      if (row >= results_per_simdgroup)
        break;
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  } else {
    ws += used_out_row * in_vec_size_w +
          simd_lid * packs_per_thread * bytes_per_pack;
    scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

    const int zp_stride = bits == 4 ? ((in_vec_size_g + 1) / 2) : in_vec_size_g;
    const device uint8_t* zps_row_base = nullptr;
    const device T* biases_row_base = nullptr;
    if (UseMlx) {
      biases_row_base = biases + used_out_row * in_vec_size_g;
    } else {
      zps_row_base = zero_points + used_out_row * zp_stride;
    }

    x += tid.x * K + simd_lid * values_per_thread;
    y += tid.x * N + used_out_row;

    int k = 0;
    for (; k < K - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const int row_idx = used_out_row + row;
        const device T* sr = scales_base + row_idx * in_vec_size_g;

        int g = (k + simd_lid * values_per_thread) / group_size;
        U s = static_cast<U>(sr[g]);
        if (UseMlx) {
          const device T* bl = biases_row_base + row * in_vec_size_g;
          U b = static_cast<U>(bl[g]);
          result[row] +=
              qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
        } else {
          const device uint8_t* zl = zps_row_base + row * zp_stride;
          U zp;
          if (bits == 4) {
            uint8_t zp_b = zl[g >> 1];
            zp = static_cast<U>((g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
          } else {
            zp = static_cast<U>(zl[g]);
          }
          result[row] +=
              qdot_zero_point<U, values_per_thread, bits>(wl, x_thread, s, zp);
        }
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(K - k - simd_lid * values_per_thread),
        0,
        values_per_thread
    );

    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x,
          x_thread,
          remaining
      );

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const int row_idx = used_out_row + row;
        const device T* sr = scales_base + row_idx * in_vec_size_g;

        int g = (k + simd_lid * values_per_thread) / group_size;
        U s = static_cast<U>(sr[g]);
        if (UseMlx) {
          const device T* bl = biases_row_base + row * in_vec_size_g;
          U b = static_cast<U>(bl[g]);
          result[row] += qdot_safe<U, values_per_thread, bits>(
              wl,
              x_thread,
              s,
              b,
              sum,
              remaining
          );
        } else {
          const device uint8_t* zl = zps_row_base + row * zp_stride;
          U zp;
          if (bits == 4) {
            uint8_t zp_b = zl[g >> 1];
            zp = static_cast<U>((g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
          } else {
            zp = static_cast<U>(zl[g]);
          }
          result[row] +=
              qdot_zero_point<U, values_per_thread, bits>(wl, x_thread, s, zp);
        }
      }
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}

template <typename T, int group_size, int bits>
void qmv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  if (kUseMlxQuant) {
    qmv_impl_dispatch<T, group_size, bits, true>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        K,
        N,
        tid,
        simd_gid,
        simd_lid
    );
  } else {
    qmv_impl_dispatch<T, group_size, bits, false>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        K,
        N,
        tid,
        simd_gid,
        simd_lid
    );
  }
}

template <typename T, int group_size, int bits>
void qmv_fast_impl(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                      simd_gid * results_per_simdgroup;
  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  int zp_stride = 0;
  const device uint8_t* zps = nullptr;
  bool high_nibble = false;

  if (kUseMlxQuant) {
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  } else {
    if (bits == 4) {
      zp_stride = (in_vec_size_g + 1) / 2;
      zps = zero_points + out_row * zp_stride;
      int g_offset = simd_lid / scale_step_per_thread;
      zps += g_offset / 2;
      high_nibble = (g_offset & 1);
    } else {
      zp_stride = in_vec_size_g;
      zps = zero_points + out_row * zp_stride;
      zps += simd_lid / scale_step_per_thread;
    }
  }

  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    {
      auto wl0 = (const device uint8_t*)(ws);
      auto wl1 = (const device uint8_t*)(ws + in_vec_size_w);
      auto wl2 = (const device uint8_t*)(ws + 2 * in_vec_size_w);
      auto wl3 = (const device uint8_t*)(ws + 3 * in_vec_size_w);

      U s0 = static_cast<U>(scales[0]);
      U s1 = static_cast<U>(scales[in_vec_size_g]);
      U s2 = static_cast<U>(scales[2 * in_vec_size_g]);
      U s3 = static_cast<U>(scales[3 * in_vec_size_g]);

      if (kUseMlxQuant) {
        U b0 = static_cast<U>(biases[0]);
        U b1 = static_cast<U>(biases[in_vec_size_g]);
        U b2 = static_cast<U>(biases[2 * in_vec_size_g]);
        U b3 = static_cast<U>(biases[3 * in_vec_size_g]);
        result[0] +=
            qdot<U, values_per_thread, bits>(wl0, x_thread, s0, b0, sum);
        result[1] +=
            qdot<U, values_per_thread, bits>(wl1, x_thread, s1, b1, sum);
        result[2] +=
            qdot<U, values_per_thread, bits>(wl2, x_thread, s2, b2, sum);
        result[3] +=
            qdot<U, values_per_thread, bits>(wl3, x_thread, s3, b3, sum);
      } else {
        uint8_t zp_byte0 = zps[0];
        uint8_t zp_byte1 = zps[zp_stride];
        uint8_t zp_byte2 = zps[2 * zp_stride];
        uint8_t zp_byte3 = zps[3 * zp_stride];
        U zp0 = static_cast<U>(
            (bits == 4 && high_nibble) ? (zp_byte0 >> 4) : (zp_byte0 & 0x0F)
        );
        U zp1 = static_cast<U>(
            (bits == 4 && high_nibble) ? (zp_byte1 >> 4) : (zp_byte1 & 0x0F)
        );
        U zp2 = static_cast<U>(
            (bits == 4 && high_nibble) ? (zp_byte2 >> 4) : (zp_byte2 & 0x0F)
        );
        U zp3 = static_cast<U>(
            (bits == 4 && high_nibble) ? (zp_byte3 >> 4) : (zp_byte3 & 0x0F)
        );
        if (bits == 8) {
          zp0 = static_cast<U>(zp_byte0);
          zp1 = static_cast<U>(zp_byte1);
          zp2 = static_cast<U>(zp_byte2);
          zp3 = static_cast<U>(zp_byte3);
        }
        result[0] +=
            qdot_zero_point<U, values_per_thread, bits>(wl0, x_thread, s0, zp0);
        result[1] +=
            qdot_zero_point<U, values_per_thread, bits>(wl1, x_thread, s1, zp1);
        result[2] +=
            qdot_zero_point<U, values_per_thread, bits>(wl2, x_thread, s2, zp2);
        result[3] +=
            qdot_zero_point<U, values_per_thread, bits>(wl3, x_thread, s3, zp3);
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    if (kUseMlxQuant) {
      biases += block_size / group_size;
    } else {
      if (bits == 4) {
        zps += (block_size / group_size) / 2;
      } else {
        zps += block_size / group_size;
      }
    }
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits, bool use_zero_points>
void qvm_impl_core(
    const device uint32_t* ws,
    const device T* scales,
    const device T* biases,
    const device uint8_t* zero_points,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  constexpr int num_simdgroups = 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int tn = 32 / pack_factor;
  constexpr int block_size = 32;

  typedef float U;
  typedef struct {
    uint8_t wi[tn * bytes_per_pack];
  } vec_w;

  thread vec_w w_local;
  thread U result[tn * pack_factor] = {0};

  const int out_vec_size_w = out_vec_size * bytes_per_pack / pack_factor;
  const int out_vec_size_g = out_vec_size / group_size;
  const int out_col = pack_factor * tn * (tid.y * num_simdgroups + simd_gid);

  if (out_col >= out_vec_size) {
    return;
  }

  const int out_group = out_col / group_size;
  const int zp_row_stride =
      use_zero_points
          ? ((bits == 4) ? ((out_vec_size_g + 1) / 2) : out_vec_size_g)
          : 0;

  const device uint8_t* ws_ptr = (const device uint8_t*)ws +
                                 out_col * bytes_per_pack / pack_factor +
                                 simd_lid * out_vec_size_w;
  const device T* scales_base = scales;
  const device T* biases_base = biases;
  x += tid.x * in_vec_size + simd_lid;
  y += tid.x * out_vec_size + out_col;

  for (int k_base = 0; k_base < in_vec_size; k_base += block_size) {
    const int k_index = k_base + simd_lid;
    const bool active = k_index < in_vec_size;

    U x_local = active ? static_cast<U>(*x) : U(0);
    U scale =
        active
            ? static_cast<U>(scales_base[k_index * out_vec_size_g + out_group])
            : U(0);
    U bias;

    if (use_zero_points && active) {
      U zp;
      if (bits == 4) {
        const device uint8_t* row_ptr = zero_points + k_index * zp_row_stride;
        const device uint8_t* zp_ptr = row_ptr + (out_group >> 1);
        uint8_t zp_byte = *zp_ptr;
        bool high_nibble = (out_group & 1) != 0;
        zp = static_cast<U>(
            high_nibble ? ((zp_byte >> 4) & 0x0F) : (zp_byte & 0x0F)
        );
      } else {
        const device uint8_t* row_ptr = zero_points + k_index * zp_row_stride;
        const device uint8_t* zp_ptr = row_ptr + out_group;
        zp = static_cast<U>(*zp_ptr);
      }
      bias = -scale * zp;
    } else if (active) {
      bias = static_cast<U>(biases_base[k_index * out_vec_size_g + out_group]);
    } else {
      bias = U(0);
    }

    if (active) {
      w_local = *((device vec_w*)ws_ptr);
    }

    qouter<U, tn * pack_factor, bits>(
        (thread uint8_t*)&w_local,
        x_local,
        scale,
        bias,
        result
    );

    x += block_size;
    scales += block_size * out_vec_size_g;
    if (!use_zero_points) {
      biases += block_size * out_vec_size_g;
    }
    ws_ptr += block_size * out_vec_size_w;
  }

#pragma clang loop unroll(full)
  for (int k = 0; k < tn * pack_factor; k++) {
    result[k] = simd_sum(result[k]);
  }

  if (simd_lid == 0) {
#pragma clang loop unroll(full)
    for (int k = 0; k < tn * pack_factor; k++) {
      y[k] = static_cast<T>(result[k]);
    }
  }
}

template <typename T, int group_size, int bits>
void qvm_impl_mlx(
    const device uint32_t* ws,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  qvm_impl_core<T, group_size, bits, false>(
      ws,
      scales,
      biases,
      nullptr,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid
  );
}

template <typename T, int group_size, int bits>
void qvm_impl_zeropoint(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  qvm_impl_core<T, group_size, bits, true>(
      w,
      scales,
      nullptr,
      zero_points,
      x,
      y,
      in_vec_size,
      out_vec_size,
      tid,
      simd_gid,
      simd_lid
  );
}

template <typename T, int group_size, int bits>
void qvm_impl(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& K, // in_vec_size
    const constant int& N, // out_vec_size
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  if (kUseMlxQuant) {
    qvm_impl_mlx<T, group_size, bits>(
        w,
        scales,
        biases,
        x,
        y,
        K,
        N,
        tid,
        simd_gid,
        simd_lid
    );
  } else {
    qvm_impl_zeropoint<T, group_size, bits>(
        w,
        scales,
        zero_points,
        x,
        y,
        K,
        N,
        tid,
        simd_gid,
        simd_lid
    );
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_K,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel, max_total_threads_per_threadgroup(128)]] void qmm(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device T* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {

  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  qmm_impl<T, group_size, bits, aligned_K, BM, BK, BN>(
      w,
      scales,
      zero_points,
      biases,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      tid,
      lid,
      simd_gid,
      simd_lid
  );
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel, max_total_threads_per_threadgroup(128)]] void qmm_transposed(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device T* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {

  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  qmm_transposed_impl<T, group_size, bits, aligned_N, BM, BK, BN>(
      w,
      scales,
      zero_points,
      biases,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      tid,
      lid,
      simd_gid,
      simd_lid
  );
}

template <typename T, int group_size, int bits>
[[kernel, max_total_threads_per_threadgroup(64)]] void qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device T* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {

  qmv_impl<T, group_size, bits>(
      w,
      scales,
      zero_points,
      biases,
      x,
      y,
      K,
      N,
      tid,
      simd_gid,
      simd_lid
  );
}

template <typename T, int group_size, int bits>
[[kernel, max_total_threads_per_threadgroup(64)]] void qvm(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device T* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {

  qvm_impl<T, group_size, bits>(
      w,
      scales,
      zero_points,
      biases,
      x,
      y,
      K,
      N,
      tid,
      simd_gid,
      simd_lid
  );
}

template <typename T, int group_size, int bits>
[[kernel, max_total_threads_per_threadgroup(64)]] void qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device T* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  qmv_fast_impl<T, group_size, bits>(
      w,
      scales,
      zero_points,
      biases,
      x,
      y,
      K,
      N,
      tid,
      simd_gid,
      simd_lid
  );
}

template [[host_name("qmv_f16_g32_b4_fast")]] [[kernel]] void qmv_fast<
    half,
    32,
    4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f16_g64_b4_fast")]] [[kernel]] void qmv_fast<
    half,
    64,
    4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f16_g128_b4_fast")]] [[kernel]] void qmv_fast<
    half,
    128,
    4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_bf16_g32_b4_fast")]] [[kernel]] void qmv_fast<
    bfloat,
    32,
    4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_bf16_g64_b4_fast")]] [[kernel]] void qmv_fast<
    bfloat,
    64,
    4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_bf16_g128_b4_fast")]] [[kernel]] void qmv_fast<
    bfloat,
    128,
    4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f32_g32_b4_fast")]] [[kernel]] void qmv_fast<
    float,
    32,
    4>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f32_g64_b4_fast")]] [[kernel]] void qmv_fast<
    float,
    64,
    4>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f32_g128_b4_fast")]] [[kernel]] void qmv_fast<
    float,
    128,
    4>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f16_g32_b8_fast")]] [[kernel]] void qmv_fast<
    half,
    32,
    8>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f16_g64_b8_fast")]] [[kernel]] void qmv_fast<
    half,
    64,
    8>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f16_g128_b8_fast")]] [[kernel]] void qmv_fast<
    half,
    128,
    8>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_bf16_g32_b8_fast")]] [[kernel]] void qmv_fast<
    bfloat,
    32,
    8>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_bf16_g64_b8_fast")]] [[kernel]] void qmv_fast<
    bfloat,
    64,
    8>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_bf16_g128_b8_fast")]] [[kernel]] void qmv_fast<
    bfloat,
    128,
    8>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f32_g32_b8_fast")]] [[kernel]] void qmv_fast<
    float,
    32,
    8>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f32_g64_b8_fast")]] [[kernel]] void qmv_fast<
    float,
    64,
    8>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f32_g128_b8_fast")]] [[kernel]] void qmv_fast<
    float,
    128,
    8>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 32 (F16)
template [[host_name("qmm_f16_g32_b4")]] [[kernel]] void qmm<
    half,
    32,
    4,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f16_g32_b4")]] [[kernel]] void
qmm_transposed<half, 32, 4, true>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f16_g32_b4_unaligned")]] [[kernel]] void
qmm_transposed<half, 32, 4, false>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_f16_g32_b4_alignedk")]] [[kernel]] void qmm<
    half,
    32,
    4,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f16_g32_b4")]] [[kernel]] void qmv<half, 32, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_f16_g32_b4")]] [[kernel]] void qvm<half, 32, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device half* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 32 (F16) - 8 bit
template [[host_name("qmm_f16_g32_b8")]] [[kernel]] void qmm<
    half,
    32,
    8,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f16_g32_b8")]] [[kernel]] void
qmm_transposed<half, 32, 8, true>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f16_g32_b8_unaligned")]] [[kernel]] void
qmm_transposed<half, 32, 8, false>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_f16_g32_b8_alignedk")]] [[kernel]] void qmm<
    half,
    32,
    8,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f16_g32_b8")]] [[kernel]] void qmv<half, 32, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_f16_g32_b8")]] [[kernel]] void qvm<half, 32, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device half* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 64 (F16)
template [[host_name("qmm_f16_g64_b4")]] [[kernel]] void qmm<
    half,
    64,
    4,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_f16_g64_b4_alignedk")]] [[kernel]] void qmm<
    half,
    64,
    4,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f16_g64_b4")]] [[kernel]] void
qmm_transposed<half, 64, 4, true>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f16_g64_b4_unaligned")]] [[kernel]] void
qmm_transposed<half, 64, 4, false>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f16_g64_b4")]] [[kernel]] void qmv<half, 64, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_f16_g64_b4")]] [[kernel]] void qvm<half, 64, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device half* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 64 (F16) - 8 bit
template [[host_name("qmm_f16_g64_b8")]] [[kernel]] void qmm<
    half,
    64,
    8,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_f16_g64_b8_alignedk")]] [[kernel]] void qmm<
    half,
    64,
    8,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f16_g64_b8")]] [[kernel]] void
qmm_transposed<half, 64, 8, true>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f16_g64_b8_unaligned")]] [[kernel]] void
qmm_transposed<half, 64, 8, false>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f16_g64_b8")]] [[kernel]] void qmv<half, 64, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_f16_g64_b8")]] [[kernel]] void qvm<half, 64, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device half* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 128 (F16)
template [[host_name("qmm_f16_g128_b4")]] [[kernel]] void qmm<
    half,
    128,
    4,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_f16_g128_b4_alignedk")]] [[kernel]] void qmm<
    half,
    128,
    4,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f16_g128_b4")]] [[kernel]] void
qmm_transposed<half, 128, 4, true>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f16_g128_b4_unaligned")]] [[kernel]] void
qmm_transposed<half, 128, 4, false>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f16_g128_b4")]] [[kernel]] void qmv<half, 128, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_f16_g128_b4")]] [[kernel]] void qvm<half, 128, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device half* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 128 (F16) - 8 bit
template [[host_name("qmm_f16_g128_b8")]] [[kernel]] void qmm<
    half,
    128,
    8,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_f16_g128_b8_alignedk")]] [[kernel]] void qmm<
    half,
    128,
    8,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f16_g128_b8")]] [[kernel]] void
qmm_transposed<half, 128, 8, true>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f16_g128_b8_unaligned")]] [[kernel]] void
qmm_transposed<half, 128, 8, false>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f16_g128_b8")]] [[kernel]] void qmv<half, 128, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_f16_g128_b8")]] [[kernel]] void qvm<half, 128, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device half* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 32 (F32)
template [[host_name("qmm_f32_g32_b4")]] [[kernel]] void qmm<
    float,
    32,
    4,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f32_g32_b4")]] [[kernel]] void
qmm_transposed<float, 32, 4, true>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f32_g32_b4_unaligned")]] [[kernel]] void
qmm_transposed<float, 32, 4, false>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_f32_g32_b4_alignedk")]] [[kernel]] void qmm<
    float,
    32,
    4,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f32_g32_b4")]] [[kernel]] void qmv<float, 32, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_f32_g32_b4")]] [[kernel]] void qvm<float, 32, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device float* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 32 (F32) - 8 bit
template [[host_name("qmm_f32_g32_b8")]] [[kernel]] void qmm<
    float,
    32,
    8,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f32_g32_b8")]] [[kernel]] void
qmm_transposed<float, 32, 8, true>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f32_g32_b8_unaligned")]] [[kernel]] void
qmm_transposed<float, 32, 8, false>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_f32_g32_b8_alignedk")]] [[kernel]] void qmm<
    float,
    32,
    8,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f32_g32_b8")]] [[kernel]] void qmv<float, 32, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_f32_g32_b8")]] [[kernel]] void qvm<float, 32, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device float* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 64 (F32)
template [[host_name("qmm_f32_g64_b4")]] [[kernel]] void qmm<
    float,
    64,
    4,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_f32_g64_b4_alignedk")]] [[kernel]] void qmm<
    float,
    64,
    4,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f32_g64_b4")]] [[kernel]] void
qmm_transposed<float, 64, 4, true>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f32_g64_b4_unaligned")]] [[kernel]] void
qmm_transposed<float, 64, 4, false>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f32_g64_b4")]] [[kernel]] void qmv<float, 64, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_f32_g64_b4")]] [[kernel]] void qvm<float, 64, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device float* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 64 (F32) - 8 bit
template [[host_name("qmm_f32_g64_b8")]] [[kernel]] void qmm<
    float,
    64,
    8,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_f32_g64_b8_alignedk")]] [[kernel]] void qmm<
    float,
    64,
    8,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f32_g64_b8")]] [[kernel]] void
qmm_transposed<float, 64, 8, true>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f32_g64_b8_unaligned")]] [[kernel]] void
qmm_transposed<float, 64, 8, false>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f32_g64_b8")]] [[kernel]] void qmv<float, 64, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_f32_g64_b8")]] [[kernel]] void qvm<float, 64, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device float* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 128 (F32)
template [[host_name("qmm_f32_g128_b4")]] [[kernel]] void qmm<
    float,
    128,
    4,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_f32_g128_b4_alignedk")]] [[kernel]] void qmm<
    float,
    128,
    4,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f32_g128_b4")]] [[kernel]] void
qmm_transposed<float, 128, 4, true>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f32_g128_b4_unaligned")]] [[kernel]] void
qmm_transposed<float, 128, 4, false>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f32_g128_b4")]] [[kernel]] void qmv<float, 128, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_f32_g128_b4")]] [[kernel]] void qvm<float, 128, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device float* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 128 (F32) - 8 bit
template [[host_name("qmm_f32_g128_b8")]] [[kernel]] void qmm<
    float,
    128,
    8,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_f32_g128_b8_alignedk")]] [[kernel]] void qmm<
    float,
    128,
    8,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f32_g128_b8")]] [[kernel]] void
qmm_transposed<float, 128, 8, true>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_f32_g128_b8_unaligned")]] [[kernel]] void
qmm_transposed<float, 128, 8, false>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_f32_g128_b8")]] [[kernel]] void qmv<float, 128, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device float* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_f32_g128_b8")]] [[kernel]] void qvm<float, 128, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device float* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 32 (BF16)
template [[host_name("qmm_bf16_g32_b4")]] [[kernel]] void qmm<
    bfloat,
    32,
    4,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_bf16_g32_b4_alignedk")]] [[kernel]] void qmm<
    bfloat,
    32,
    4,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g32_b4")]] [[kernel]] void
qmm_transposed<bfloat, 32, 4, true>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g32_b4_unaligned")]] [[kernel]] void
qmm_transposed<bfloat, 32, 4, false>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_bf16_g32_b4")]] [[kernel]] void qmv<bfloat, 32, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_bf16_g32_b4")]] [[kernel]] void qvm<bfloat, 32, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device bfloat* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 32 (BF16) - 8 bit
template [[host_name("qmm_bf16_g32_b8")]] [[kernel]] void qmm<
    bfloat,
    32,
    8,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_bf16_g32_b8_alignedk")]] [[kernel]] void qmm<
    bfloat,
    32,
    8,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g32_b8")]] [[kernel]] void
qmm_transposed<bfloat, 32, 8, true>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g32_b8_unaligned")]] [[kernel]] void
qmm_transposed<bfloat, 32, 8, false>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_bf16_g32_b8")]] [[kernel]] void qmv<bfloat, 32, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_bf16_g32_b8")]] [[kernel]] void qvm<bfloat, 32, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device bfloat* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 64 (BF16)
template [[host_name("qmm_bf16_g64_b4")]] [[kernel]] void qmm<
    bfloat,
    64,
    4,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_bf16_g64_b4_alignedk")]] [[kernel]] void qmm<
    bfloat,
    64,
    4,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g64_b4")]] [[kernel]] void
qmm_transposed<bfloat, 64, 4, true>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g64_b4_unaligned")]] [[kernel]] void
qmm_transposed<bfloat, 64, 4, false>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_bf16_g64_b4")]] [[kernel]] void qmv<bfloat, 64, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_bf16_g64_b4")]] [[kernel]] void qvm<bfloat, 64, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device bfloat* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 64 (BF16) - 8 bit
template [[host_name("qmm_bf16_g64_b8")]] [[kernel]] void qmm<
    bfloat,
    64,
    8,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_bf16_g64_b8_alignedk")]] [[kernel]] void qmm<
    bfloat,
    64,
    8,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g64_b8")]] [[kernel]] void
qmm_transposed<bfloat, 64, 8, true>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g64_b8_unaligned")]] [[kernel]] void
qmm_transposed<bfloat, 64, 8, false>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_bf16_g64_b8")]] [[kernel]] void qmv<bfloat, 64, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_bf16_g64_b8")]] [[kernel]] void qvm<bfloat, 64, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device bfloat* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 128 (BF16)
template [[host_name("qmm_bf16_g128_b4")]] [[kernel]] void qmm<
    bfloat,
    128,
    4,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_bf16_g128_b4_alignedk")]] [[kernel]] void qmm<
    bfloat,
    128,
    4,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g128_b4")]] [[kernel]] void
qmm_transposed<bfloat, 128, 4, true>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g128_b4_unaligned")]] [[kernel]] void
qmm_transposed<bfloat, 128, 4, false>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_bf16_g128_b4")]] [[kernel]] void qmv<bfloat, 128, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_bf16_g128_b4")]] [[kernel]] void qvm<bfloat, 128, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device bfloat* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

// Group size 128 (BF16) - 8 bit
template [[host_name("qmm_bf16_g128_b8")]] [[kernel]] void qmm<
    bfloat,
    128,
    8,
    false>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_bf16_g128_b8_alignedk")]] [[kernel]] void qmm<
    bfloat,
    128,
    8,
    true>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g128_b8")]] [[kernel]] void
qmm_transposed<bfloat, 128, 8, true>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g128_b8_unaligned")]] [[kernel]] void
qmm_transposed<bfloat, 128, 8, false>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_bf16_g128_b8")]] [[kernel]] void qmv<bfloat, 128, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qvm_bf16_g128_b8")]] [[kernel]] void qvm<bfloat, 128, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device bfloat* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g128_b4_64x64")]] [[kernel]] void
qmm_transposed<bfloat, 128, 4, true, 64, 32, 64>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g128_b4_64x128")]] [[kernel]] void
qmm_transposed<bfloat, 128, 4, true, 64, 32, 128>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_transposed_bf16_g128_b4_128x64")]] [[kernel]] void
qmm_transposed<bfloat, 128, 4, true, 128, 32, 64>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

///////////////////////////////////////////////////////////////////////////////
/// MLP Fused QMV - Quantized Matrix-Vector with Fused Activation
///////////////////////////////////////////////////////////////////////////////

// MLP fused QMV kernel for decode path (M=1).
// Computes paired up and gate projections, then applies: out = up *
// activation(gate) Weight layout: [up_weights (hidden_dim rows), gate_weights
// (hidden_dim rows)] Output size is hidden_dim.
template <typename T, int group_size, int bits>
void qmv_mlp_fused_impl(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& hidden_dim, // Output size (half of weight rows)
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4; // Compute 4 paired rows at a time
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result_up[results_per_simdgroup] = {0};
  thread U result_gate[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;

  // Output row for 'up' projection
  const int out_row_up = tid.y * (num_simdgroups * results_per_simdgroup) +
                         simd_gid * results_per_simdgroup;

  // Corresponding 'gate' row is offset by hidden_dim
  const int out_row_gate = out_row_up + hidden_dim;

  // Check bounds - only process if up row is within hidden_dim
  if (out_row_up >= hidden_dim)
    return;

  // Pointer setup for up weights
  const device uint8_t* ws_up = ws + out_row_up * in_vec_size_w +
                                simd_lid * packs_per_thread * bytes_per_pack;
  const device T* scales_up =
      scales + out_row_up * in_vec_size_g + simd_lid / scale_step_per_thread;

  // Pointer setup for gate weights
  const device uint8_t* ws_gate = ws + out_row_gate * in_vec_size_w +
                                  simd_lid * packs_per_thread * bytes_per_pack;
  const device T* scales_gate =
      scales + out_row_gate * in_vec_size_g + simd_lid / scale_step_per_thread;

  const device T* biases_up = nullptr;
  const device T* biases_gate = nullptr;
  const device uint8_t* zps_up = nullptr;
  const device uint8_t* zps_gate = nullptr;
  int zp_stride = 0;
  bool high_nibble = false;

  if (kUseMlxQuant) {
    biases_up =
        biases + out_row_up * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases_gate = biases + out_row_gate * in_vec_size_g +
                  simd_lid / scale_step_per_thread;
  } else {
    if (bits == 4) {
      zp_stride = (in_vec_size_g + 1) / 2;
      zps_up = zero_points + out_row_up * zp_stride;
      zps_gate = zero_points + out_row_gate * zp_stride;
      int g_offset = simd_lid / scale_step_per_thread;
      zps_up += g_offset / 2;
      zps_gate += g_offset / 2;
      high_nibble = (g_offset & 1);
    } else {
      zp_stride = in_vec_size_g;
      zps_up = zero_points + out_row_up * zp_stride +
               simd_lid / scale_step_per_thread;
      zps_gate = zero_points + out_row_gate * zp_stride +
                 simd_lid / scale_step_per_thread;
    }
  }

  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * hidden_dim + out_row_up;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    // Compute both up and gate dot products for 4 rows each
    {
      auto wl_up0 = ws_up;
      auto wl_up1 = ws_up + in_vec_size_w;
      auto wl_up2 = ws_up + 2 * in_vec_size_w;
      auto wl_up3 = ws_up + 3 * in_vec_size_w;

      auto wl_gate0 = ws_gate;
      auto wl_gate1 = ws_gate + in_vec_size_w;
      auto wl_gate2 = ws_gate + 2 * in_vec_size_w;
      auto wl_gate3 = ws_gate + 3 * in_vec_size_w;

      U s_up0 = static_cast<U>(scales_up[0]);
      U s_up1 = static_cast<U>(scales_up[in_vec_size_g]);
      U s_up2 = static_cast<U>(scales_up[2 * in_vec_size_g]);
      U s_up3 = static_cast<U>(scales_up[3 * in_vec_size_g]);

      U s_gate0 = static_cast<U>(scales_gate[0]);
      U s_gate1 = static_cast<U>(scales_gate[in_vec_size_g]);
      U s_gate2 = static_cast<U>(scales_gate[2 * in_vec_size_g]);
      U s_gate3 = static_cast<U>(scales_gate[3 * in_vec_size_g]);

      if (kUseMlxQuant) {
        U b_up0 = static_cast<U>(biases_up[0]);
        U b_up1 = static_cast<U>(biases_up[in_vec_size_g]);
        U b_up2 = static_cast<U>(biases_up[2 * in_vec_size_g]);
        U b_up3 = static_cast<U>(biases_up[3 * in_vec_size_g]);

        U b_gate0 = static_cast<U>(biases_gate[0]);
        U b_gate1 = static_cast<U>(biases_gate[in_vec_size_g]);
        U b_gate2 = static_cast<U>(biases_gate[2 * in_vec_size_g]);
        U b_gate3 = static_cast<U>(biases_gate[3 * in_vec_size_g]);

        result_up[0] += qdot<U, values_per_thread, bits>(
            wl_up0,
            x_thread,
            s_up0,
            b_up0,
            sum
        );
        result_up[1] += qdot<U, values_per_thread, bits>(
            wl_up1,
            x_thread,
            s_up1,
            b_up1,
            sum
        );
        result_up[2] += qdot<U, values_per_thread, bits>(
            wl_up2,
            x_thread,
            s_up2,
            b_up2,
            sum
        );
        result_up[3] += qdot<U, values_per_thread, bits>(
            wl_up3,
            x_thread,
            s_up3,
            b_up3,
            sum
        );

        result_gate[0] += qdot<U, values_per_thread, bits>(
            wl_gate0,
            x_thread,
            s_gate0,
            b_gate0,
            sum
        );
        result_gate[1] += qdot<U, values_per_thread, bits>(
            wl_gate1,
            x_thread,
            s_gate1,
            b_gate1,
            sum
        );
        result_gate[2] += qdot<U, values_per_thread, bits>(
            wl_gate2,
            x_thread,
            s_gate2,
            b_gate2,
            sum
        );
        result_gate[3] += qdot<U, values_per_thread, bits>(
            wl_gate3,
            x_thread,
            s_gate3,
            b_gate3,
            sum
        );
      } else {
        auto extract_zp = [&](const device uint8_t* zps, int row_offset) -> U {
          uint8_t zp_byte = zps[row_offset * zp_stride];
          if (bits == 4) {
            return static_cast<U>(
                high_nibble ? (zp_byte >> 4) : (zp_byte & 0x0F)
            );
          } else {
            return static_cast<U>(zp_byte);
          }
        };

        U zp_up0 = extract_zp(zps_up, 0);
        U zp_up1 = extract_zp(zps_up, 1);
        U zp_up2 = extract_zp(zps_up, 2);
        U zp_up3 = extract_zp(zps_up, 3);

        U zp_gate0 = extract_zp(zps_gate, 0);
        U zp_gate1 = extract_zp(zps_gate, 1);
        U zp_gate2 = extract_zp(zps_gate, 2);
        U zp_gate3 = extract_zp(zps_gate, 3);

        result_up[0] += qdot_zero_point<U, values_per_thread, bits>(
            wl_up0,
            x_thread,
            s_up0,
            zp_up0
        );
        result_up[1] += qdot_zero_point<U, values_per_thread, bits>(
            wl_up1,
            x_thread,
            s_up1,
            zp_up1
        );
        result_up[2] += qdot_zero_point<U, values_per_thread, bits>(
            wl_up2,
            x_thread,
            s_up2,
            zp_up2
        );
        result_up[3] += qdot_zero_point<U, values_per_thread, bits>(
            wl_up3,
            x_thread,
            s_up3,
            zp_up3
        );

        result_gate[0] += qdot_zero_point<U, values_per_thread, bits>(
            wl_gate0,
            x_thread,
            s_gate0,
            zp_gate0
        );
        result_gate[1] += qdot_zero_point<U, values_per_thread, bits>(
            wl_gate1,
            x_thread,
            s_gate1,
            zp_gate1
        );
        result_gate[2] += qdot_zero_point<U, values_per_thread, bits>(
            wl_gate2,
            x_thread,
            s_gate2,
            zp_gate2
        );
        result_gate[3] += qdot_zero_point<U, values_per_thread, bits>(
            wl_gate3,
            x_thread,
            s_gate3,
            zp_gate3
        );
      }
    }

    ws_up += block_size * bytes_per_pack / pack_factor;
    ws_gate += block_size * bytes_per_pack / pack_factor;
    scales_up += block_size / group_size;
    scales_gate += block_size / group_size;
    if (kUseMlxQuant) {
      biases_up += block_size / group_size;
      biases_gate += block_size / group_size;
    } else {
      if (bits == 4) {
        zps_up += (block_size / group_size) / 2;
        zps_gate += (block_size / group_size) / 2;
      } else {
        zps_up += block_size / group_size;
        zps_gate += block_size / group_size;
      }
    }
    x += block_size;
  }

  // SIMD reduction and fused activation
  for (int row = 0; row < results_per_simdgroup; row++) {
    result_up[row] = simd_sum(result_up[row]);
    result_gate[row] = simd_sum(result_gate[row]);
    if (simd_lid == 0) {
      // Apply MLP fused epilogue: up * activation(gate)
      float fused = mlp_fused_epilogue_f32(result_up[row], result_gate[row]);
      y[row] = static_cast<T>(fused);
    }
  }
}

template <typename T, int group_size, int bits>
[[kernel, max_total_threads_per_threadgroup(64)]] void qmv_mlp_fused(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device T* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& hidden_dim [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  qmv_mlp_fused_impl<T, group_size, bits>(
      w,
      scales,
      zero_points,
      biases,
      x,
      y,
      K,
      hidden_dim,
      tid,
      simd_gid,
      simd_lid
  );
}

// Instantiate MLP fused QMV kernels for common configurations
template [[host_name("qmv_mlp_fused_f16_g32_b4")]] [[kernel]] void
qmv_mlp_fused<half, 32, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& hidden_dim [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_mlp_fused_f16_g64_b4")]] [[kernel]] void
qmv_mlp_fused<half, 64, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& hidden_dim [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_mlp_fused_f16_g128_b4")]] [[kernel]] void
qmv_mlp_fused<half, 128, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& hidden_dim [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_mlp_fused_bf16_g32_b4")]] [[kernel]] void
qmv_mlp_fused<bfloat, 32, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& hidden_dim [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_mlp_fused_bf16_g64_b4")]] [[kernel]] void
qmv_mlp_fused<bfloat, 64, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& hidden_dim [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmv_mlp_fused_bf16_g128_b4")]] [[kernel]] void
qmv_mlp_fused<bfloat, 128, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& hidden_dim [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

///////////////////////////////////////////////////////////////////////////////
/// MLP Fused QMM - Quantized Matrix-Matrix with Fused Activation
///////////////////////////////////////////////////////////////////////////////

// MLP fused QMM kernel for prefill path (M>1).
// Computes paired up and gate projections, then applies: out = up *
// activation(gate) Weight layout: [up_weights (hidden_dim cols), gate_weights
// (hidden_dim cols)] Output size is [M, hidden_dim].
template <
    typename T,
    int group_size,
    int bits,
    bool aligned_K,
    int BM = 32,
    int BK = 32,
    int BN = 32>
void qmm_mlp_fused_impl(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws_up,
    const constant int& K,
    const constant int&
        hidden_dim, // Output columns (half of total weight columns)
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  static_assert(BK >= 32, "BK should be larger than SIMD_SIZE");
  static_assert(BK % 32 == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = matmul_utils::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, false, BK_padded, BN_padded>;
  using loader_x_t =
      matmul_utils::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * 32, 1, 4>;

  auto wl_up = (const device uint8_t*)w;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN; // Column in output (within hidden_dim)

  // Only process tiles within hidden_dim
  if (y_col >= hidden_dim)
    return;

  x += y_row * static_cast<int64_t>(K);

  // Up weights at column y_col
  wl_up += y_col * bytes_per_pack / pack_factor;
  // Gate weights at column y_col + hidden_dim
  auto wl_gate = wl_up + hidden_dim * bytes_per_pack / pack_factor;

  y += y_row * static_cast<int64_t>(hidden_dim) + y_col;

  const device T* scales_base = scales;
  const device T* biases_base = biases;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, hidden_dim - y_col);

  // Threadgroup memory for gate weights (after Ws_up)
  threadgroup T* Ws_gate = Ws_up + BK * BN_padded;

  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  mma_t mma_up(simd_gid, simd_lid);
  mma_t mma_gate(simd_gid, simd_lid);

  const int k_iterations = (K + BK - 1) / BK;
  const int full_iterations = aligned_K ? k_iterations : k_iterations - 1;

  // Total output width for scale/bias strides
  const int N_total = hidden_dim * 2;

  if (kUseMlxQuant) {
    using loader_w_t = QuantizedBlockLoaderMlx<
        T,
        BK,
        BN,
        BN_padded,
        0,
        WM * WN * 32,
        group_size,
        bits>;

    const device T* scales_up = scales_base + (y_col / group_size);
    const device T* biases_up = biases_base + (y_col / group_size);
    const device T* scales_gate =
        scales_base + ((y_col + hidden_dim) / group_size);
    const device T* biases_gate =
        biases_base + ((y_col + hidden_dim) / group_size);

    loader_w_t loader_w_up(
        wl_up,
        scales_up,
        biases_up,
        N_total,
        Ws_up,
        simd_gid,
        simd_lid
    );
    loader_w_t loader_w_gate(
        wl_gate,
        scales_gate,
        biases_gate,
        N_total,
        Ws_gate,
        simd_gid,
        simd_lid
    );

    for (int k = 0; k < full_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w_up.load_unsafe();
      loader_w_gate.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);

      mma_up.mma(Xs, Ws_up);
      mma_gate.mma(Xs, Ws_gate);

      loader_x.next();
      loader_w_up.next();
      loader_w_gate.next();
    }

    if (!aligned_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BM, K - full_iterations * BK));
      loader_w_up.load_safe(short2(K - full_iterations * BK, num_outs));
      loader_w_gate.load_safe(short2(K - full_iterations * BK, num_outs));
      threadgroup_barrier(mem_flags::mem_threadgroup);

      mma_up.mma(Xs, Ws_up);
      mma_gate.mma(Xs, Ws_gate);
    }
  } else {
    const int out_groups_total = (N_total + group_size - 1) / group_size;
    const int groups_per_row = out_groups_total;
    const int out_group_up = y_col / group_size;
    const int out_group_gate = (y_col + hidden_dim) / group_size;
    const int zp_stride_out =
        (bits == 4) ? ((out_groups_total + 1) / 2) : out_groups_total;

    using loader_w_t = QuantizedBlockLoaderZp<
        T,
        BK,
        BN,
        BN_padded,
        0,
        WM * WN * 32,
        group_size,
        bits,
        true>;

    loader_w_t loader_w_up(
        wl_up,
        scales_base,
        zero_points,
        N_total,
        groups_per_row,
        Ws_up,
        simd_gid,
        simd_lid,
        out_group_up,
        out_groups_total,
        zp_stride_out
    );

    loader_w_t loader_w_gate(
        wl_gate,
        scales_base,
        zero_points,
        N_total,
        groups_per_row,
        Ws_gate,
        simd_gid,
        simd_lid,
        out_group_gate,
        out_groups_total,
        zp_stride_out
    );

    for (int k = 0; k < full_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w_up.load_unsafe();
      loader_w_gate.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);

      mma_up.mma(Xs, Ws_up);
      mma_gate.mma(Xs, Ws_gate);

      loader_x.next();
      loader_w_up.next();
      loader_w_gate.next();
    }

    if (!aligned_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BM, K - full_iterations * BK));
      loader_w_up.load_safe(short2(K - full_iterations * BK, num_outs));
      loader_w_gate.load_safe(short2(K - full_iterations * BK, num_outs));
      threadgroup_barrier(mem_flags::mem_threadgroup);

      mma_up.mma(Xs, Ws_up);
      mma_gate.mma(Xs, Ws_gate);
    }
  }

  // Apply MLP fused epilogue: out = up * activation(gate)
  // Access Ctile elements and apply fusion
  constexpr int kElemsPerTile = decltype(mma_up.Ctile)::kElemsPerTile;
#pragma unroll
  for (short i = 0; i < kElemsPerTile; i++) {
    float up_val = static_cast<float>(mma_up.Ctile.elems()[i]);
    float gate_val = static_cast<float>(mma_gate.Ctile.elems()[i]);
    float fused = mlp_fused_epilogue_f32(up_val, gate_val);
    mma_up.Ctile.elems()[i] = static_cast<T>(fused);
  }

  // Store result
  mma_up.store_result_safe(y, hidden_dim, short2(num_outs, num_els));
}

template <typename T, int group_size, int bits>
[[kernel, max_total_threads_per_threadgroup(128)]] void qmm_mlp_fused(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device uint8_t* zero_points
    [[buffer(2), function_constant(kUseZeroPoints)]],
    const device T* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& hidden_dim [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
  constexpr int BM = 32;
  constexpr int BK = 32;
  constexpr int BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[2 * BK * BN_padded]; // Double for up and gate

  bool aligned_K = (K % BK) == 0;
  if (aligned_K) {
    qmm_mlp_fused_impl<T, group_size, bits, true, BM, BK, BN>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        Xs,
        Ws,
        K,
        hidden_dim,
        M,
        tid,
        lid,
        simd_gid,
        simd_lid
    );
  } else {
    qmm_mlp_fused_impl<T, group_size, bits, false, BM, BK, BN>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        Xs,
        Ws,
        K,
        hidden_dim,
        M,
        tid,
        lid,
        simd_gid,
        simd_lid
    );
  }
}

// Instantiate MLP fused QMM kernels
template [[host_name("qmm_mlp_fused_f16_g32_b4")]] [[kernel]] void
qmm_mlp_fused<half, 32, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& hidden_dim [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_mlp_fused_f16_g64_b4")]] [[kernel]] void
qmm_mlp_fused<half, 64, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& hidden_dim [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_mlp_fused_f16_g128_b4")]] [[kernel]] void
qmm_mlp_fused<half, 128, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device half* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& hidden_dim [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_mlp_fused_bf16_g32_b4")]] [[kernel]] void
qmm_mlp_fused<bfloat, 32, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& hidden_dim [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_mlp_fused_bf16_g64_b4")]] [[kernel]] void
qmm_mlp_fused<bfloat, 64, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& hidden_dim [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);

template [[host_name("qmm_mlp_fused_bf16_g128_b4")]] [[kernel]] void
qmm_mlp_fused<bfloat, 128, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2)]],
    const device bfloat* biases [[buffer(2)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& hidden_dim [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
);
