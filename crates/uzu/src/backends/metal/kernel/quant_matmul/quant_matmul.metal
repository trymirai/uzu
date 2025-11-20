#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

#include "mma.h"

// Function constant: true = MLX-style (pre-computed biases), false = AWQ-style (zero-points)
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
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
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
    U zero_point) {
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
          static_cast<U>(static_cast<int>(word & 0x000f) - static_cast<int>(zp0));
      accum += x_thread[4 * i + 1] *
          static_cast<U>(static_cast<int>(word & 0x00f0) - static_cast<int>(zp1));
      accum += x_thread[4 * i + 2] *
          static_cast<U>(static_cast<int>(word & 0x0f00) - static_cast<int>(zp2));
      accum += x_thread[4 * i + 3] *
          static_cast<U>(static_cast<int>(word & 0xf000) - static_cast<int>(zp3));
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
    int N) {
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
      if (rem > 0) accum += x_thread[base] * (wv & 0x000f);
      if (rem > 1) accum += x_thread[base + 1] * (wv & 0x00f0);
      if (rem > 2) accum += x_thread[base + 2] * (wv & 0x0f00);
    }
  } else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int N, int bits>
inline void dequantize(const device uint8_t* w, U scale, U bias, threadgroup U* w_local) {
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  if (bits == 4) {
    U s[2] = {scale, scale / static_cast<U>(16.0f)};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = s[0] * (w[i] & 0x0f) + bias;
      w_local[2 * i + 1] = s[1] * (w[i] & 0xf0) + bias;
    }
  } else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = scale * w[i] + bias;
    }
  }
}

// Specialized vectorized dequantizer for bfloat, N=8 (pack_factor), bits=4
template <>
inline void dequantize<bfloat, 8, 4>(const device uint8_t* w, bfloat scale, bfloat bias, threadgroup bfloat* w_local) {
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
struct QuantizedBlockLoader {
  static_assert(BCOLS <= group_size, "Group size should be larger than columns");
  static_assert(group_size % BCOLS == 0, "Group size should be divisible by columns");
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  UZU_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  UZU_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  UZU_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  UZU_MTL_CONST short n_reads = (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  UZU_MTL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
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
  const device T* biases;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor + bj * bytes_per_pack),
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
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
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

      // K-tail: strictly limit to valid packs and zero trailing nibbles
      int valid_cols = src_tile_dim.y; // 0..BK
      int valid_packs = (valid_cols + pack_factor - 1) / pack_factor;

      T scale = *scales;
      T bias = *biases;
      for (int i = 0; i < n_reads; i++) {
        int pack_idx = bj + i; // global pack index across the BK packs
        if (pack_idx < valid_packs) {
          dequantize<T, pack_factor, bits>(
              src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
          if (pack_idx == valid_packs - 1) {
            int rem = valid_cols - (valid_packs - 1) * pack_factor;
            if (rem > 0 && rem < pack_factor) {
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

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
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

// Loader that computes bias from packed zero-points (U8) and scales on-GPU
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoaderZp {
  static_assert(BCOLS <= group_size, "Group size should be larger than columns");
  static_assert(group_size % BCOLS == 0, "Group size should be divisible by columns");
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  UZU_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  UZU_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  UZU_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  UZU_MTL_CONST short n_reads = (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
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

  QuantizedBlockLoaderZp(
      const device uint8_t* src_,
      const device T* scales_,
      const device uint8_t* zero_points_row_start_,
      const int src_ld_,
      const int groups_per_row_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        groups_per_row(groups_per_row_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        k_base(0),
        group_stride(BROWS * groups_per_row_),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor + bj * bytes_per_pack),
        scales(reduction_dim == 1 ? (scales_ + bi * groups_per_row_) : scales_),
        scales_row_start(reduction_dim == 1 ? (scales_ + bi * groups_per_row_) : scales_),
        zps_row_start(reduction_dim == 1 
            ? (zero_points_row_start_ + bi * (bits == 4 ? ((groups_per_row_ + 1) / 2) : groups_per_row_))
            : zero_points_row_start_) {}

  inline void current_scale_bias(thread T& out_scale, thread T& out_bias) const {
    int g = reduction_dim == 0 ? (k_base / group_size)
                               : (int)(scales - scales_row_start);
    uint zp_n;
    if (bits == 4) {
        const device uint8_t* zp_ptr = zps_row_start + (g >> 1);
        uint8_t zp_b = *zp_ptr;
        zp_n = (g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F);
    } else {
        zp_n = zps_row_start[g];
    }
    T scale_val = reduction_dim == 0 ? scales_row_start[g] : *scales;
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
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
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

      // K-tail: strictly limit to valid packs and zero trailing nibbles
      int valid_cols = src_tile_dim.y; // 0..BK
      int valid_packs = (valid_cols + pack_factor - 1) / pack_factor;

      T scale;
      T bias;
      current_scale_bias(scale, bias);
      for (int i = 0; i < n_reads; i++) {
        int pack_idx = bj + i; // global pack index across the BK packs
        if (pack_idx < valid_packs) {
          dequantize<T, pack_factor, bits>(
              src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
          if (pack_idx == valid_packs - 1) {
            int rem = valid_cols - (valid_packs - 1) * pack_factor;
            if (rem > 0 && rem < pack_factor) {
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
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
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
    uint lid [[thread_index_in_threadgroup]]) {
  (void)lid;
  const int full_blocks = K / BK;
  for (int kb = 0; kb < full_blocks; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (num_els < BM) {
      loader_x.load_safe(short2(BK, num_els));
    } else {
      loader_x.load_unsafe();
    }
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
  }

  if (!aligned_K) {
    const short tail_k = static_cast<short>(K - full_blocks * BK);
    if (tail_k > 0) {
      const short rows = (num_els < BM) ? num_els : BM;
      const short cols = (num_outs < BN) ? num_outs : BN;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(tail_k, rows));
      loader_w.load_safe(short2(cols, tail_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    }
  }

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
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= 32, "BK should be larger than SIMD_SIZE");
  static_assert(BK % 32 == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = matmul_utils::BlockMMA<T, T, BM, BN, BK, WM, WN, false, false, BK_padded, BN_padded>;
  using loader_x_t = matmul_utils::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * 32, 1, 4>;

  auto wl = (const device uint8_t*)w;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * bytes_per_pack / pack_factor;
  const int groups_per_row = (K + group_size - 1) / group_size;
  scales += y_col * groups_per_row;
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  // Create appropriate loader based on quantization type
  if (kUseMlxQuant) {
    // MLX quantization: uses pre-computed biases
    using loader_w_t = QuantizedBlockLoader<T, BK, BN, BN_padded, 0, WM * WN * 32, group_size, bits>;
    const device T* biases_row_start = biases + y_col * groups_per_row;
    loader_w_t loader_w(wl, scales, biases_row_start, N, Ws, simd_gid, simd_lid);
    qmm_core<loader_w_t, loader_x_t, mma_t, T, aligned_K, BM, BK, BN>(
        loader_x, loader_w, mma_op, num_els, num_outs, K, y, N, Xs, Ws, lid);
  } else {
    // GroupQuantized: uses zero-points
    using loader_w_t = QuantizedBlockLoaderZp<T, BK, BN, BN_padded, 0, WM * WN * 32, group_size, bits>;
    const device uint8_t* zero_points_row_start = zero_points + y_col * (bits == 4 ? ((groups_per_row + 1) / 2) : groups_per_row);
    loader_w_t loader_w(wl, scales, zero_points_row_start, N, groups_per_row, Ws, simd_gid, simd_lid);
    qmm_core<loader_w_t, loader_x_t, mma_t, T, aligned_K, BM, BK, BN>(
        loader_x, loader_w, mma_op, num_els, num_outs, K, y, N, Xs, Ws, lid);
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
    uint lid [[thread_index_in_threadgroup]]) {
  (void)lid;
  const int k_blocks = (K + BK - 1) / BK;
  for (int kb = 0; kb < k_blocks; kb++) {
    const short k_len = (kb < (K / BK)) ? BK : static_cast<short>(K - (K / BK) * BK);
    const bool needs_safe_x = (num_els < BM) || (k_len < BK);
    const bool needs_safe_w = (!aligned_N && num_outs < BN) || (k_len < BK);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (needs_safe_x) {
      loader_x.load_safe(short2(k_len, num_els));
    } else {
      loader_x.load_unsafe();
    }
    if (needs_safe_w) {
      loader_w.load_safe(short2(num_outs, k_len));
    } else {
      loader_w.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    if (kb + 1 < k_blocks) {
      loader_x.next();
      loader_w.next();
    }
  }
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
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= 32, "BK should be larger than SIMD_SIZE");
  static_assert(BK % 32 == 0, "BK should be divisible by SIMD_SIZE");

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = matmul_utils::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = matmul_utils::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * 32>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = (K + group_size - 1) / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  const device T* x_block = x + y_row * static_cast<int64_t>(K);
  const device uint8_t* w_block = wl + y_col * K_w;
  const device T* scales_row = scales + y_col * K_g;
  const device T* biases_row = biases + y_col * K_g;
  const device uint8_t* zero_points_row = zero_points + y_col * (bits == 4 ? ((K_g + 1) / 2) : K_g);
  device T* y_block = y + y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x_block, K, Xs, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (kUseMlxQuant) {
    using loader_w_t = QuantizedBlockLoader<
        T,
        BN,
        BK,
        BK_padded,
        1,
        WM * WN * 32,
        group_size,
        bits>;
    loader_w_t loader_w(w_block, scales_row, biases_row, K, Ws, simd_gid, simd_lid);
    qmm_transposed_core<loader_w_t, loader_x_t, mma_t, T, aligned_N, BM, BK, BN>(
        loader_x, loader_w, mma_op, num_els, num_outs, K, y_block, N, Xs, Ws, lid);
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
    loader_w_t loader_w(w_block, scales_row, zero_points_row, K, K_g, Ws, simd_gid, simd_lid);
    qmm_transposed_core<loader_w_t, loader_x_t, mma_t, T, aligned_N, BM, BK, BN>(
        loader_x, loader_w, mma_op, num_els, num_outs, K, y_block, N, Xs, Ws, lid);
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
    uint simd_lid [[thread_index_in_simdgroup]]) {
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
  const int in_vec_size_g = (K + group_size - 1) / group_size;  // ceil(K / group_size)
  const device T* scales_base = scales; // remember original base before pointer arithmetic
  const device uint8_t* zero_points_base = zero_points;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
  const int used_out_row = min(N - results_per_simdgroup, out_row);

  if (out_row >= N) {
    return;
  }

  if (N < (num_simdgroups * results_per_simdgroup)) {
    ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
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
        if (row >= results_per_simdgroup) break;
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const int row_idx = out_row + row;
        const device T* sr = scales_base + row_idx * in_vec_size_g;

        int g = (k + simd_lid * values_per_thread) / group_size;
        U s = static_cast<U>(sr[g]);
        if (UseMlx) {
          const device T* bl = biases_row_base + row * in_vec_size_g;
          U b = static_cast<U>(bl[g]);
          result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
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
        values_per_thread);
    if (remaining > 0) {
        U sum = load_vector_safe<T, U, values_per_thread, bits>(
            x, x_thread, remaining);

        for (int row = 0; out_row + row < N; row++) {
            if (row >= results_per_simdgroup) break;
            auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
            const int row_idx = out_row + row;
            const device T* sr = scales_base + row_idx * in_vec_size_g;

            int g = (k + simd_lid * values_per_thread) / group_size;
            U s = static_cast<U>(sr[g]);
            if (UseMlx) {
                const device T* bl = biases_row_base + row * in_vec_size_g;
                U b = static_cast<U>(bl[g]);
                result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
            } else {
                const device uint8_t* zl = zps_row_base + row * zp_stride;
                U zp;
                if (bits == 4) {
                    uint8_t zp_b = zl[g >> 1];
                    zp = static_cast<U>((g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
                } else {
                    zp = static_cast<U>(zl[g]);
                }
                result[row] += qdot_zero_point<U, values_per_thread, bits>(
                    wl, x_thread, s, zp);
            }
        }
    }

    for (int row = 0; out_row + row < N; row++) {
      if (row >= results_per_simdgroup) break;
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
  else {
    ws += used_out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
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
          result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
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
        values_per_thread);

    if (remaining > 0) {
        U sum = load_vector_safe<T, U, values_per_thread, bits>(
            x, x_thread, remaining);

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
                    wl, x_thread, s, b, sum, remaining);
            } else {
                const device uint8_t* zl = zps_row_base + row * zp_stride;
                U zp;
                if (bits == 4) {
                    uint8_t zp_b = zl[g >> 1];
                    zp = static_cast<U>((g & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
                } else {
                    zp = static_cast<U>(zl[g]);
                }
                result[row] += qdot_zero_point<U, values_per_thread, bits>(
                    wl, x_thread, s, zp);
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
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (kUseMlxQuant) {
    qmv_impl_dispatch<T, group_size, bits, true>(
        w, scales, zero_points, biases, x, y, K, N, tid, simd_gid, simd_lid);
  } else {
    qmv_impl_dispatch<T, group_size, bits, false>(
        w, scales, zero_points, biases, x, y, K, N, tid, simd_gid, simd_lid);
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
    uint simd_lid [[thread_index_in_simdgroup]]) {
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

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      U s = sl[0];
      
      if (kUseMlxQuant) {
        const device T* bl = biases + row * in_vec_size_g;
        U b = bl[0];
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      } else {
        const device uint8_t* zl = zps + row * zp_stride;
        uint8_t zp_byte = *zl;
        U zp = static_cast<U>((bits == 4 && high_nibble) ? (zp_byte >> 4) : (zp_byte & 0x0F));
        if (bits == 8) zp = static_cast<U>(zp_byte);
        result[row] += qdot_zero_point<U, values_per_thread, bits>(wl, x_thread, s, zp);
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

template <typename T, int group_size, int bits, bool UseMlx>
void qvm_impl_dispatch(
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
    uint simd_lid [[thread_index_in_simdgroup]]) {

  typedef float U;
  constexpr int BN = 64;
  constexpr int BK = 32;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();

  const int lidy = simd_gid;
  const int lidx = simd_lid;
  const int bn = BN / WN;

  const int out_col = tid.y * BN + lidy * bn;
  const int in_row = tid.x;

  if (out_col >= N) {
    return;
  }

  const device uint8_t* w_row = ((const device uint8_t*)w) + out_col * K * bytes_per_pack / pack_factor;
  const int in_vec_size_g = (K + group_size - 1) / group_size;
  const device T* scales_row = scales + out_col * in_vec_size_g;
  
  const device T* biases_row = nullptr;
  int zp_stride = 0;
  const device uint8_t* zps_row = nullptr;
  if (UseMlx) {
    biases_row = biases + out_col * in_vec_size_g;
  } else {
    if (bits == 4) {
        zp_stride = (in_vec_size_g + 1) / 2;
    } else {
        zp_stride = in_vec_size_g;
    }
    zps_row = zero_points + out_col * zp_stride;
  }

  const device T* x_row = x + in_row * K;
  device T* y_row = y + in_row * N + out_col;

  U result[BN / WN] = {0};
  thread U x_thread[pack_factor];

  for (int k = lidx * pack_factor; k < K; k += BK * pack_factor) {
    const int k_pack = k / pack_factor;
    const int remaining = min(pack_factor, K - k);
    
    U x_sum = load_vector_safe<T, U, pack_factor, bits>(
        x_row + k, x_thread, remaining);

    for (int n = 0; n < min(bn, N - out_col); n++) {
      const int w_offset = n * K * bytes_per_pack / pack_factor + k_pack * bytes_per_pack;
      const int group_idx = k / group_size;
      const int scale_idx = n * in_vec_size_g + group_idx;
      
      U scale = static_cast<U>(scales_row[scale_idx]);
      if (UseMlx) {
        U bias = static_cast<U>(biases_row[scale_idx]);
        result[n] += qdot_safe<U, pack_factor, bits>(
            w_row + w_offset, x_thread, scale, bias, x_sum, remaining);
      } else {
        const device uint8_t* zp_row = zps_row + n * zp_stride;
        U zp;
        if (bits == 4) {
            uint8_t zp_b = zp_row[group_idx >> 1];
            zp = static_cast<U>((group_idx & 1) ? ((zp_b >> 4) & 0x0F) : (zp_b & 0x0F));
        } else {
            zp = static_cast<U>(zp_row[group_idx]);
        }
        U bias = -scale * zp;
        
        result[n] += qdot_safe<U, pack_factor, bits>(
            w_row + w_offset, x_thread, scale, bias, x_sum, remaining);
      }
    }
  }

  for (int n = 0; n < min(bn, N - out_col); n++) {
    result[n] = simd_sum(result[n]);
    if (lidx == 0) {
      y_row[n] = static_cast<T>(result[n]);
    }
  }
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
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (kUseMlxQuant) {
    qvm_impl_dispatch<T, group_size, bits, true>(
        w, scales, zero_points, biases, x, y, K, N, tid, simd_gid, simd_lid);
  } else {
    qvm_impl_dispatch<T, group_size, bits, false>(
        w, scales, zero_points, biases, x, y, K, N, tid, simd_gid, simd_lid);
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
[[kernel]] void qmm(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device T* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  qmm_impl<T, group_size, bits, aligned_K, BM, BK, BN>(
      w, scales, zero_points, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
[[kernel]] void qmm_transposed(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device T* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  qmm_transposed_impl<T, group_size, bits, aligned_N, BM, BK, BN>(
      w, scales, zero_points, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device T* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  
  qmv_impl<T, group_size, bits>(w, scales, zero_points, biases, x, y, K, N, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void qvm(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device T* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  qvm_impl<T, group_size, bits>(
      w, scales, zero_points, biases, x, y, K, N, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device T* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  qmv_fast_impl<T, group_size, bits>(w, scales, zero_points, biases, x, y, K, N, tid, simd_gid, simd_lid);
}

template [[host_name("qmv_f16_g32_b4_fast")]]
[[kernel]] void qmv_fast<half, 32, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f16_g64_b4_fast")]]
[[kernel]] void qmv_fast<half, 64, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f16_g128_b4_fast")]]
[[kernel]] void qmv_fast<half, 128, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_bf16_g32_b4_fast")]]
[[kernel]] void qmv_fast<bfloat, 32, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_bf16_g64_b4_fast")]]
[[kernel]] void qmv_fast<bfloat, 64, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_bf16_g128_b4_fast")]]
[[kernel]] void qmv_fast<bfloat, 128, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f32_g32_b4_fast")]]
[[kernel]] void qmv_fast<float, 32, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f32_g64_b4_fast")]]
[[kernel]] void qmv_fast<float, 64, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f32_g128_b4_fast")]]
[[kernel]] void qmv_fast<float, 128, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f16_g32_b8_fast")]]
[[kernel]] void qmv_fast<half, 32, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f16_g64_b8_fast")]]
[[kernel]] void qmv_fast<half, 64, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f16_g128_b8_fast")]]
[[kernel]] void qmv_fast<half, 128, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_bf16_g32_b8_fast")]]
[[kernel]] void qmv_fast<bfloat, 32, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_bf16_g64_b8_fast")]]
[[kernel]] void qmv_fast<bfloat, 64, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_bf16_g128_b8_fast")]]
[[kernel]] void qmv_fast<bfloat, 128, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f32_g32_b8_fast")]]
[[kernel]] void qmv_fast<float, 32, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f32_g64_b8_fast")]]
[[kernel]] void qmv_fast<float, 64, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f32_g128_b8_fast")]]
[[kernel]] void qmv_fast<float, 128, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 32 (F16)
template [[host_name("qmm_f16_g32_b4")]]
[[kernel]] void qmm<half, 32, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f16_g32_b4")]]
[[kernel]] void qmm_transposed<half, 32, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f16_g32_b4_unaligned")]]
[[kernel]] void qmm_transposed<half, 32, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_f16_g32_b4_alignedk")]]
[[kernel]] void qmm<half, 32, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f16_g32_b4")]]
[[kernel]] void qmv<half, 32, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_f16_g32_b4")]]
[[kernel]] void qvm<half, 32, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device half* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 32 (F16) - 8 bit
template [[host_name("qmm_f16_g32_b8")]]
[[kernel]] void qmm<half, 32, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f16_g32_b8")]]
[[kernel]] void qmm_transposed<half, 32, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f16_g32_b8_unaligned")]]
[[kernel]] void qmm_transposed<half, 32, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_f16_g32_b8_alignedk")]]
[[kernel]] void qmm<half, 32, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f16_g32_b8")]]
[[kernel]] void qmv<half, 32, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_f16_g32_b8")]]
[[kernel]] void qvm<half, 32, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device half* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 64 (F16)
template [[host_name("qmm_f16_g64_b4")]]
[[kernel]] void qmm<half, 64, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_f16_g64_b4_alignedk")]]
[[kernel]] void qmm<half, 64, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f16_g64_b4")]]
[[kernel]] void qmm_transposed<half, 64, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f16_g64_b4_unaligned")]]
[[kernel]] void qmm_transposed<half, 64, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f16_g64_b4")]]
[[kernel]] void qmv<half, 64, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_f16_g64_b4")]]
[[kernel]] void qvm<half, 64, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device half* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 64 (F16) - 8 bit
template [[host_name("qmm_f16_g64_b8")]]
[[kernel]] void qmm<half, 64, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_f16_g64_b8_alignedk")]]
[[kernel]] void qmm<half, 64, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f16_g64_b8")]]
[[kernel]] void qmm_transposed<half, 64, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f16_g64_b8_unaligned")]]
[[kernel]] void qmm_transposed<half, 64, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f16_g64_b8")]]
[[kernel]] void qmv<half, 64, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_f16_g64_b8")]]
[[kernel]] void qvm<half, 64, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device half* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 128 (F16)
template [[host_name("qmm_f16_g128_b4")]]
[[kernel]] void qmm<half, 128, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_f16_g128_b4_alignedk")]]
[[kernel]] void qmm<half, 128, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f16_g128_b4")]]
[[kernel]] void qmm_transposed<half, 128, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f16_g128_b4_unaligned")]]
[[kernel]] void qmm_transposed<half, 128, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f16_g128_b4")]]
[[kernel]] void qmv<half, 128, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_f16_g128_b4")]]
[[kernel]] void qvm<half, 128, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device half* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 128 (F16) - 8 bit
template [[host_name("qmm_f16_g128_b8")]]
[[kernel]] void qmm<half, 128, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_f16_g128_b8_alignedk")]]
[[kernel]] void qmm<half, 128, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f16_g128_b8")]]
[[kernel]] void qmm_transposed<half, 128, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f16_g128_b8_unaligned")]]
[[kernel]] void qmm_transposed<half, 128, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f16_g128_b8")]]
[[kernel]] void qmv<half, 128, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_f16_g128_b8")]]
[[kernel]] void qvm<half, 128, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device half* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device half* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 32 (F32)
template [[host_name("qmm_f32_g32_b4")]]
[[kernel]] void qmm<float, 32, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f32_g32_b4")]]
[[kernel]] void qmm_transposed<float, 32, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f32_g32_b4_unaligned")]]
[[kernel]] void qmm_transposed<float, 32, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_f32_g32_b4_alignedk")]]
[[kernel]] void qmm<float, 32, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f32_g32_b4")]]
[[kernel]] void qmv<float, 32, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_f32_g32_b4")]]
[[kernel]] void qvm<float, 32, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device float* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 32 (F32) - 8 bit
template [[host_name("qmm_f32_g32_b8")]]
[[kernel]] void qmm<float, 32, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f32_g32_b8")]]
[[kernel]] void qmm_transposed<float, 32, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f32_g32_b8_unaligned")]]
[[kernel]] void qmm_transposed<float, 32, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_f32_g32_b8_alignedk")]]
[[kernel]] void qmm<float, 32, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f32_g32_b8")]]
[[kernel]] void qmv<float, 32, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_f32_g32_b8")]]
[[kernel]] void qvm<float, 32, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device float* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 64 (F32)
template [[host_name("qmm_f32_g64_b4")]]
[[kernel]] void qmm<float, 64, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_f32_g64_b4_alignedk")]]
[[kernel]] void qmm<float, 64, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f32_g64_b4")]]
[[kernel]] void qmm_transposed<float, 64, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f32_g64_b4_unaligned")]]
[[kernel]] void qmm_transposed<float, 64, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f32_g64_b4")]]
[[kernel]] void qmv<float, 64, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_f32_g64_b4")]]
[[kernel]] void qvm<float, 64, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device float* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 64 (F32) - 8 bit
template [[host_name("qmm_f32_g64_b8")]]
[[kernel]] void qmm<float, 64, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_f32_g64_b8_alignedk")]]
[[kernel]] void qmm<float, 64, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f32_g64_b8")]]
[[kernel]] void qmm_transposed<float, 64, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f32_g64_b8_unaligned")]]
[[kernel]] void qmm_transposed<float, 64, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f32_g64_b8")]]
[[kernel]] void qmv<float, 64, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_f32_g64_b8")]]
[[kernel]] void qvm<float, 64, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device float* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 128 (F32)
template [[host_name("qmm_f32_g128_b4")]]
[[kernel]] void qmm<float, 128, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_f32_g128_b4_alignedk")]]
[[kernel]] void qmm<float, 128, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f32_g128_b4")]]
[[kernel]] void qmm_transposed<float, 128, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f32_g128_b4_unaligned")]]
[[kernel]] void qmm_transposed<float, 128, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f32_g128_b4")]]
[[kernel]] void qmv<float, 128, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_f32_g128_b4")]]
[[kernel]] void qvm<float, 128, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device float* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 128 (F32) - 8 bit
template [[host_name("qmm_f32_g128_b8")]]
[[kernel]] void qmm<float, 128, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_f32_g128_b8_alignedk")]]
[[kernel]] void qmm<float, 128, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f32_g128_b8")]]
[[kernel]] void qmm_transposed<float, 128, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_f32_g128_b8_unaligned")]]
[[kernel]] void qmm_transposed<float, 128, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_f32_g128_b8")]]
[[kernel]] void qmv<float, 128, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_f32_g128_b8")]]
[[kernel]] void qvm<float, 128, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device float* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 32 (BF16)
template [[host_name("qmm_bf16_g32_b4")]]
[[kernel]] void qmm<bfloat, 32, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_bf16_g32_b4_alignedk")]]
[[kernel]] void qmm<bfloat, 32, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_bf16_g32_b4")]]
[[kernel]] void qmm_transposed<bfloat, 32, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_bf16_g32_b4_unaligned")]]
[[kernel]] void qmm_transposed<bfloat, 32, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_bf16_g32_b4")]]
[[kernel]] void qmv<bfloat, 32, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_bf16_g32_b4")]]
[[kernel]] void qvm<bfloat, 32, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device bfloat* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 32 (BF16) - 8 bit
template [[host_name("qmm_bf16_g32_b8")]]
[[kernel]] void qmm<bfloat, 32, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_bf16_g32_b8_alignedk")]]
[[kernel]] void qmm<bfloat, 32, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_bf16_g32_b8")]]
[[kernel]] void qmm_transposed<bfloat, 32, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_bf16_g32_b8_unaligned")]]
[[kernel]] void qmm_transposed<bfloat, 32, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_bf16_g32_b8")]]
[[kernel]] void qmv<bfloat, 32, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_bf16_g32_b8")]]
[[kernel]] void qvm<bfloat, 32, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device bfloat* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 64 (BF16)
template [[host_name("qmm_bf16_g64_b4")]]
[[kernel]] void qmm<bfloat, 64, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_bf16_g64_b4_alignedk")]]
[[kernel]] void qmm<bfloat, 64, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_bf16_g64_b4")]]
[[kernel]] void qmm_transposed<bfloat, 64, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_bf16_g64_b4_unaligned")]]
[[kernel]] void qmm_transposed<bfloat, 64, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_bf16_g64_b4")]]
[[kernel]] void qmv<bfloat, 64, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_bf16_g64_b4")]]
[[kernel]] void qvm<bfloat, 64, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device bfloat* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 64 (BF16) - 8 bit
template [[host_name("qmm_bf16_g64_b8")]]
[[kernel]] void qmm<bfloat, 64, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_bf16_g64_b8_alignedk")]]
[[kernel]] void qmm<bfloat, 64, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_bf16_g64_b8")]]
[[kernel]] void qmm_transposed<bfloat, 64, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_bf16_g64_b8_unaligned")]]
[[kernel]] void qmm_transposed<bfloat, 64, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_bf16_g64_b8")]]
[[kernel]] void qmv<bfloat, 64, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_bf16_g64_b8")]]
[[kernel]] void qvm<bfloat, 64, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device bfloat* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 128 (BF16)
template [[host_name("qmm_bf16_g128_b4")]]
[[kernel]] void qmm<bfloat, 128, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_bf16_g128_b4_alignedk")]]
[[kernel]] void qmm<bfloat, 128, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_bf16_g128_b4")]]
[[kernel]] void qmm_transposed<bfloat, 128, 4, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_bf16_g128_b4_unaligned")]]
[[kernel]] void qmm_transposed<bfloat, 128, 4, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_bf16_g128_b4")]]
[[kernel]] void qmv<bfloat, 128, 4>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_bf16_g128_b4")]]
[[kernel]] void qvm<bfloat, 128, 4>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device bfloat* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

// Group size 128 (BF16) - 8 bit
template [[host_name("qmm_bf16_g128_b8")]]
[[kernel]] void qmm<bfloat, 128, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_bf16_g128_b8_alignedk")]]
[[kernel]] void qmm<bfloat, 128, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_bf16_g128_b8")]]
[[kernel]] void qmm_transposed<bfloat, 128, 8, true>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_bf16_g128_b8_unaligned")]]
[[kernel]] void qmm_transposed<bfloat, 128, 8, false>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmv_bf16_g128_b8")]]
[[kernel]] void qmv<bfloat, 128, 8>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qvm_bf16_g128_b8")]]
[[kernel]] void qvm<bfloat, 128, 8>(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device uint8_t* zero_points [[buffer(2), function_constant(kUseZeroPoints)]],
    const device bfloat* biases [[buffer(2), function_constant(kUseMlxQuant)]],
    const device bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_bf16_g128_b4_64x64")]]
[[kernel]] void qmm_transposed<bfloat, 128, 4, true, 64, 32, 64>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);

template [[host_name("qmm_transposed_bf16_g128_b4_64x128")]]
[[kernel]] void qmm_transposed<bfloat, 128, 4, true, 64, 32, 128>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);


template [[host_name("qmm_transposed_bf16_g128_b4_128x64")]]
[[kernel]] void qmm_transposed<bfloat, 128, 4, true, 128, 32, 64>(
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
    uint simd_lid [[thread_index_in_simdgroup]]);
