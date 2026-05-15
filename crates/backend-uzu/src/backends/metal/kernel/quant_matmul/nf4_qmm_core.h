#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/dsl.h"
#include "nf4_common.h"
#include "quant_matmul.h"

using namespace metal;

// Lookup modes for the surviving NF4 QMM variants.
enum Nf4Mode {
  NF4_CONST = 1,
  NF4_TG = 3,
};

template <Nf4Mode MODE>
struct Nf4LookupCtx {
  threadgroup const half* tg_cb; // valid for TG; unused for CONST
};

template <Nf4Mode MODE>
inline half nf4_lookup(uint8_t nibble, thread const Nf4LookupCtx<MODE>& ctx) {
  if (MODE == NF4_CONST) {
    return nf4_codebook[nibble & 0x0f];
  } else { // NF4_TG
    return ctx.tg_cb[nibble & 0x0f];
  }
}

// NF4 dequantize: w[N/2] packed bytes -> w_local[N] half-floats in tg memory.
template <typename U, int N, Nf4Mode MODE>
inline void nf4_dequantize(
    const device uint8_t* w,
    U scale,
    threadgroup U* w_local,
    thread const Nf4LookupCtx<MODE>& ctx
) {
  for (int i = 0; i < (N / 2); i++) {
    uint8_t byte = w[i];
    half h0 = nf4_lookup<MODE>(byte & 0x0f, ctx);
    half h1 = nf4_lookup<MODE>((byte >> 4) & 0x0f, ctx);
    w_local[2 * i] = static_cast<U>(h0) * scale;
    w_local[2 * i + 1] = static_cast<U>(h1) * scale;
  }
}

// NF4 block loader: forks QuantizedBlockLoaderMlx (scale-only, no bias).
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    Nf4Mode MODE>
struct Nf4BlockLoader {
  static_assert(
      BCOLS <= group_size,
      "Group size should be larger than columns"
  );
  static_assert(
      group_size % BCOLS == 0,
      "Group size should be divisible by columns"
  );

  METAL_CONST short BITS = 4;
  METAL_CONST short pack_factor = get_pack_factor<BITS, 8>();
  METAL_CONST short bytes_per_pack = get_bytes_per_pack<BITS>();
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
  Nf4LookupCtx<MODE> ctx;

  Nf4BlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const int src_ld_,
      threadgroup T* dst_,
      Nf4LookupCtx<MODE> ctx_,
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
        scales(scales_ + bi * src_ld / group_size), ctx(ctx_) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }
    T scale = *scales;
    for (int i = 0; i < n_reads; i++) {
      nf4_dequantize<T, pack_factor, MODE>(
          src + i * bytes_per_pack,
          scale,
          dst + i * pack_factor,
          ctx
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
    for (int i = 0; i < n_reads; i++) {
      nf4_dequantize<T, pack_factor, MODE>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          dst + i * pack_factor,
          ctx
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
      scales += group_stride;
    }
  }
};

// NF4 block loader with E4M3 (1-byte FP8) per-group scales. Forks
// Nf4BlockLoader; the only differences are the `scales` buffer element type
// (uint8_t instead of T) and decoding each scale via e4m3_to_half() once per
// group before dequantizing.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    Nf4Mode MODE>
struct Nf4BlockLoaderE4m3 {
  static_assert(
      BCOLS <= group_size,
      "Group size should be larger than columns"
  );
  static_assert(
      group_size % BCOLS == 0,
      "Group size should be divisible by columns"
  );

  METAL_CONST short BITS = 4;
  METAL_CONST short pack_factor = get_pack_factor<BITS, 8>();
  METAL_CONST short bytes_per_pack = get_bytes_per_pack<BITS>();
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
  const device uint8_t* scales;
  Nf4LookupCtx<MODE> ctx;

  Nf4BlockLoaderE4m3(
      const device uint8_t* src_,
      const device uint8_t* scales_,
      const int src_ld_,
      threadgroup T* dst_,
      Nf4LookupCtx<MODE> ctx_,
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
        scales(scales_ + bi * src_ld / group_size), ctx(ctx_) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }
    T scale = static_cast<T>(e4m3_to_half(*scales));
    for (int i = 0; i < n_reads; i++) {
      nf4_dequantize<T, pack_factor, MODE>(
          src + i * bytes_per_pack,
          scale,
          dst + i * pack_factor,
          ctx
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
    T scale = static_cast<T>(e4m3_to_half(*scales));
    for (int i = 0; i < n_reads; i++) {
      nf4_dequantize<T, pack_factor, MODE>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          dst + i * pack_factor,
          ctx
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
      scales += group_stride;
    }
  }
};

template <
    typename T,
    uint GROUP_SIZE,
    Nf4Mode MODE,
    bool ALIGNED_N,
    int BM = 32,
    int BK = 32,
    int BN = 32,
    int WM = 2,
    int WN = 2>
inline void nf4_qmm_impl(
    const device uint32_t* weights,
    const device T* scales,
    const device T* input,
    device T* output,
    threadgroup T* Xs,
    threadgroup T* Ws,
    Nf4LookupCtx<MODE> ctx,
    const int in_vec_size,
    const int out_vec_size,
    const int batch_size,
    uint out_block_idx,
    uint batch_block_idx,
    uint simd_group,
    uint simd_lane
) {
  constexpr int BITS = 4;
  constexpr int pack_factor = get_pack_factor<BITS, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<BITS>();
  constexpr int BK_padded = (BK + 16 / int(sizeof(T)));

  using mma_t = matmul_utils::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      matmul_utils::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * 32>;
  using loader_w_t =
      Nf4BlockLoader<T, BN, BK, BK_padded, 1, WM * WN * 32, GROUP_SIZE, MODE>;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = (in_vec_size + GROUP_SIZE - 1) / int(GROUP_SIZE);
  const int out_row = batch_block_idx * BM;
  const int out_col = out_block_idx * BN;

  auto wl = (const device uint8_t*)weights;
  const device T* x_block = input + out_row * int64_t(in_vec_size);
  const device uint8_t* w_block = wl + out_col * in_vec_size_w;
  scales += out_col * in_vec_size_g;
  device T* y_block = output + out_row * int64_t(out_vec_size) + out_col;

  const short num_els = min(BM, batch_size - out_row);
  const short num_outs = min(BN, out_vec_size - out_col);
  loader_x_t loader_x(x_block, in_vec_size, Xs, simd_group, simd_lane);
  loader_w_t
      loader_w(w_block, scales, in_vec_size, Ws, ctx, simd_group, simd_lane);
  mma_t mma_op(simd_group, simd_lane);

  qmm_transposed_core<loader_w_t, loader_x_t, mma_t, T, ALIGNED_N, BM, BK, BN>(
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

// E4M3-scale fork of nf4_qmm_impl. Identical except `scales` is a uint8_t
// FP8 buffer decoded by Nf4BlockLoaderE4m3.
template <
    typename T,
    uint GROUP_SIZE,
    Nf4Mode MODE,
    bool ALIGNED_N,
    int BM = 32,
    int BK = 32,
    int BN = 32,
    int WM = 2,
    int WN = 2>
inline void nf4_qmm_impl_e4m3(
    const device uint32_t* weights,
    const device uint8_t* scales,
    const device T* input,
    device T* output,
    threadgroup T* Xs,
    threadgroup T* Ws,
    Nf4LookupCtx<MODE> ctx,
    const int in_vec_size,
    const int out_vec_size,
    const int batch_size,
    uint out_block_idx,
    uint batch_block_idx,
    uint simd_group,
    uint simd_lane
) {
  constexpr int BITS = 4;
  constexpr int pack_factor = get_pack_factor<BITS, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<BITS>();
  constexpr int BK_padded = (BK + 16 / int(sizeof(T)));

  using mma_t = matmul_utils::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      matmul_utils::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * 32>;
  using loader_w_t = Nf4BlockLoaderE4m3<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * 32,
      GROUP_SIZE,
      MODE>;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = (in_vec_size + GROUP_SIZE - 1) / int(GROUP_SIZE);
  const int out_row = batch_block_idx * BM;
  const int out_col = out_block_idx * BN;

  auto wl = (const device uint8_t*)weights;
  const device T* x_block = input + out_row * int64_t(in_vec_size);
  const device uint8_t* w_block = wl + out_col * in_vec_size_w;
  scales += out_col * in_vec_size_g;
  device T* y_block = output + out_row * int64_t(out_vec_size) + out_col;

  const short num_els = min(BM, batch_size - out_row);
  const short num_outs = min(BN, out_vec_size - out_col);
  loader_x_t loader_x(x_block, in_vec_size, Xs, simd_group, simd_lane);
  loader_w_t
      loader_w(w_block, scales, in_vec_size, Ws, ctx, simd_group, simd_lane);
  mma_t mma_op(simd_group, simd_lane);

  qmm_transposed_core<loader_w_t, loader_x_t, mma_t, T, ALIGNED_N, BM, BK, BN>(
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
