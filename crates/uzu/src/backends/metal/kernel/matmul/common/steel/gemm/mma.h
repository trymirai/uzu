

#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#include "../defines.h"
#include "../utils/type_traits.h"
#include "transforms.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

namespace steel {

// Max tile dimensions across all configurations.
// GEMM: TM up to 8 (64/(8*1)), TN up to 4 (64/(8*2))
// Split-K: TM up to 1, TN up to 2
// tile_matmad K dim is always 1
#define STEEL_MAX_TM 8
#define STEEL_MAX_TN 4
#define STEEL_MAX_CTILE_FRAGS (STEEL_MAX_TM * STEEL_MAX_TN)  // 32

template <typename T>
struct BaseMMAFrag {
  STEEL_CONST int kFragRows = 8;
  STEEL_CONST int kFragCols = 8;

  STEEL_CONST int kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST int kElemRows = 1;
  STEEL_CONST int kElemCols = 2;

  typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;
  typedef metal::vec<T, kElemsPerFrag> frag_type;

  METAL_FUNC static constexpr short2 get_coord(
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  ) {
    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    return short2{fn, fm};
  }

  template <typename SrcPtrType>
  METAL_FUNC static constexpr void load(
      thread frag_type& dst,
      SrcPtrType src,
      int str_x,
      int str_y
  ) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] = static_cast<T>(src[i * str_x + j * str_y]);
      }
    }
  }

  template <typename SrcPtrType>
  METAL_FUNC static constexpr void load_safe(
      thread frag_type& dst,
      SrcPtrType src,
      int str_x,
      int str_y,
      int lim_x,
      int lim_y,
      int off_x = 0,
      int off_y = 0
  ) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[(off_x + i) * str_x + (off_y + j) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <typename DstPtrType>
  METAL_FUNC static constexpr void store(
      const thread frag_type& src,
      DstPtrType dst,
      int str_x,
      int str_y
  ) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * str_x + j * str_y] = static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  template <typename DstPtrType>
  METAL_FUNC static constexpr void store_safe(
      const thread frag_type& src,
      DstPtrType dst,
      int str_x,
      int str_y,
      int lim_x,
      int lim_y,
      int off_x = 0,
      int off_y = 0
  ) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[(off_x + i) * str_x + (off_y + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <typename DstPtrType>
  METAL_FUNC static constexpr void store_slice(
      const thread frag_type& src,
      DstPtrType dst,
      int str_x,
      int str_y,
      int start_x,
      int stop_x,
      int start_y,
      int stop_y,
      int off_x = 0,
      int off_y = 0
  ) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < stop_x && (off_x + i) >= start_x &&
            (off_y + j) < stop_y && (off_y + j) >= start_y) {
          dst[(off_x + i) * str_x + (off_y + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  METAL_FUNC static constexpr void mma(
      thread frag_type& D,
      thread frag_type& A,
      thread frag_type& B,
      thread frag_type& C
  ) {
    mat_type D_mat;
    mat_type A_mat;
    mat_type B_mat;
    mat_type C_mat;

    reinterpret_cast<thread frag_type&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread frag_type&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread frag_type&>(C_mat.thread_elements()) = C;

    mma(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread frag_type&>(D_mat.thread_elements());
  }

  METAL_FUNC static constexpr void mma(
      thread mat_type& D,
      thread mat_type& A,
      thread mat_type& B,
      thread mat_type& C
  ) {
    simdgroup_multiply_accumulate(D, A, B, C);
  }
};

template <typename T>
struct MMATile {
  using MMAFrag_t = BaseMMAFrag<T>;
  using elem_type = T;
  STEEL_CONST int kFragRows = MMAFrag_t::kFragRows;
  STEEL_CONST int kFragCols = MMAFrag_t::kFragCols;
  STEEL_CONST int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;

  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;

  // Max-sized array — runtime kTileRows/kTileCols control how much is used
  frag_type val_frags[STEEL_MAX_CTILE_FRAGS] = {};

  short kTileRows;
  short kTileCols;
  short kNumFrags;

  METAL_FUNC MMATile() thread : kTileRows(0), kTileCols(0), kNumFrags(0) {}

  METAL_FUNC MMATile(short tile_rows, short tile_cols) thread
      : kTileRows(tile_rows), kTileCols(tile_cols),
        kNumFrags(tile_rows * tile_cols) {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < STEEL_MAX_CTILE_FRAGS; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j
  ) const {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  METAL_FUNC const thread elem_type* elems() const {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  METAL_FUNC short elems_per_tile() const {
    return kNumFrags * kElemsPerFrag;
  }

  template <typename U>
  METAL_FUNC void load(
      const threadgroup U* src,
      int w_x, int w_y, int str_x, int str_y
  ) {
    for (short i = 0; i < kTileRows; ++i) {
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(src[(i * kFragRows) * w_x * str_x +
                  (j * kFragCols) * w_y * str_y]),
            str_x, str_y
        );
      }
    }
  }

  template <typename U>
  METAL_FUNC void store(
      threadgroup U* dst,
      int w_x, int w_y, int str_x, int str_y
  ) const {
    for (short i = 0; i < kTileRows; ++i) {
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(dst[(i * kFragRows) * w_x * str_x +
                  (j * kFragCols) * w_y * str_y]),
            str_x, str_y
        );
      }
    }
  }

  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld, int w_x, int w_y) {
    for (short i = 0; i < kTileRows; ++i) {
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(src[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld, 1
        );
      }
    }
  }

  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld, int w_x, int w_y) const {
    for (short i = 0; i < kTileRows; ++i) {
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(dst[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld, 1
        );
      }
    }
  }

  template <typename U>
  METAL_FUNC void load_safe(
      const device U* src,
      const int ld,
      const short2 src_tile_dims,
      int w_x, int w_y
  ) {
    for (int i = 0; i < kTileRows; ++i) {
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load_safe(
            frag_at(i, j),
            src,
            ld, 1,
            src_tile_dims.y,
            src_tile_dims.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y
        );
      }
    }
  }

  template <typename U>
  METAL_FUNC void store_safe(
      device U* dst,
      const int ld,
      const short2 dst_tile_dims,
      int w_x, int w_y
  ) const {
    for (int i = 0; i < kTileRows; ++i) {
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store_safe(
            frag_at(i, j),
            dst,
            ld, 1,
            dst_tile_dims.y,
            dst_tile_dims.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y
        );
      }
    }
  }

  template <typename U>
  METAL_FUNC void store_slice(
      device U* dst,
      const int ld,
      const short2 start,
      const short2 stop,
      int w_x, int w_y
  ) const {
    for (int i = 0; i < kTileRows; ++i) {
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store_slice(
            frag_at(i, j),
            dst,
            ld, 1,
            start.y, stop.y,
            start.x, stop.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y
        );
      }
    }
  }
};

template <typename T, typename U>
METAL_FUNC void tile_matmad(
    thread MMATile<T>& D,
    thread MMATile<U>& A,
    thread MMATile<U>& B,
    thread MMATile<T>& C
) {
  for (short m = 0; m < A.kTileRows; ++m) {
    for (short n = 0; n < B.kTileCols; ++n) {
      short n_serp = (m % 2) ? (B.kTileCols - 1 - n) : n;
      for (short k = 0; k < A.kTileCols; ++k) {
        BaseMMAFrag<T>::mma(
            D.frag_at(m, n_serp),
            A.frag_at(m, k),
            B.frag_at(k, n_serp),
            C.frag_at(m, n_serp)
        );
      }
    }
  }
}

template <typename InT>
struct TransformNone<complex64_t, InT> {
  static METAL_FUNC complex64_t apply(complex64_t x) { return x; }
  static METAL_FUNC complex64_t apply(complex64_t x, complex64_t) { return x; }
};

template <
    typename T,
    typename U,
    typename AccumType = float,
    typename Epilogue = TransformNone<U, AccumType>>
struct BlockMMA {
  STEEL_CONST short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType>;

  // Tile params (set at construction)
  short BM, BN, BK, WM, WN;
  bool transpose_a, transpose_b;
  short lda_tgp, ldb_tgp;

  // Derived values
  short TM_stride, TN_stride;
  short TM, TN;
  short A_str_m, A_str_k;
  short B_str_k, B_str_n;
  short tile_stride_a, tile_stride_b;

  // Simdgroup matrices
  MMATile<AccumType> Atile;
  MMATile<AccumType> Btile;
  MMATile<AccumType> Ctile;

  // Offsets within threadgroup
  short sm;
  short sn;
  short As_offset;
  short Bs_offset;

  /* Constructor */
  METAL_FUNC BlockMMA(
      ushort simd_group_id,
      ushort simd_lane_id,
      short BM_, short BN_, short BK_,
      short WM_, short WN_,
      bool transpose_a_, bool transpose_b_,
      short lda_tgp_, short ldb_tgp_
  ) : BM(BM_), BN(BN_), BK(BK_), WM(WM_), WN(WN_),
      transpose_a(transpose_a_), transpose_b(transpose_b_),
      lda_tgp(lda_tgp_), ldb_tgp(ldb_tgp_),
      TM_stride(kFragSize * WM_),
      TN_stride(kFragSize * WN_),
      TM(BM_ / (kFragSize * WM_)),
      TN(BN_ / (kFragSize * WN_)),
      A_str_m(transpose_a_ ? 1 : lda_tgp_),
      A_str_k(transpose_a_ ? lda_tgp_ : 1),
      B_str_k(transpose_b_ ? 1 : ldb_tgp_),
      B_str_n(transpose_b_ ? ldb_tgp_ : 1),
      tile_stride_a(kFragSize * (transpose_a_ ? lda_tgp_ : 1)),
      tile_stride_b(kFragSize * (transpose_b_ ? 1 : ldb_tgp_)),
      Atile(TM, 1),
      Btile(1, TN),
      Ctile(TM, TN)
  {
    short tm = kFragSize * (simd_group_id / WN);
    short tn = kFragSize * (simd_group_id % WN);

    short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
    sm = simd_coord.y;
    sn = simd_coord.x;

    As_offset = (tm + sm) * A_str_m + (sn) * A_str_k;
    Bs_offset = (sm) * B_str_k + (tn + sn) * B_str_n;

    sm += tm;
    sn += tn;
  }

  /* (BM, BK) X (BK, BN) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
    As += As_offset;
    Bs += Bs_offset;

    for (short kk = 0; kk < BK; kk += kFragSize) {
      simdgroup_barrier(mem_flags::mem_none);
      Atile.load(As, WM, 1, A_str_m, A_str_k);
      simdgroup_barrier(mem_flags::mem_none);
      Btile.load(Bs, 1, WN, B_str_k, B_str_n);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Ctile, Atile, Btile, Ctile);
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U* D, const int ldd) {
    for (short i = 0; i < Ctile.elems_per_tile(); i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }
    D += sm * ldd + sn;
    Ctile.store(D, ldd, WM, WN);
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int ldd,
      short2 dst_tile_dims
  ) {
    for (short i = 0; i < Ctile.elems_per_tile(); i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }
    D += sm * ldd + sn;
    dst_tile_dims -= short2(sn, sm);
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;
    Ctile.store_safe(D, ldd, dst_tile_dims, WM, WN);
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue(
      const device U* C,
      const int ldc,
      const int fdc,
      thread const BinaryEpilogue& epilogue_op
  ) {
    C += (sm) * ldc + (sn) * fdc;

    for (short i = 0; i < TM; i++) {
      for (short j = 0; j < TN; j++) {
        thread auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kElemsPerFrag; k++) {
          accum[k] = epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
        }
      }
    }
  }

  STEEL_CONST short kElemsPerFrag = MMAFrag_acc_t::kElemsPerFrag;

  /* Apply epilogue safe */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue_safe(
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const BinaryEpilogue& epilogue_op
  ) {
    C += (sm) * ldc + (sn) * fdc;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    for (short i = 0; i < TM; i++) {
      for (short j = 0; j < TN; j++) {
        thread auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        U c_elems[2] = {0};

        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kElemsPerFrag; k++) {
          if ((j * TN_stride + k) < dst_tile_dims.x) {
            c_elems[k] = C[offset_c + k * fdc];
          }
        }

        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kElemsPerFrag; k++) {
          accum[k] = epilogue_op.apply(accum[k], c_elems[k]);
        }
      }
    }
  }

  /* Store results with epilogue from C source */
  METAL_FUNC void store_result(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      thread const Epilogue& epilogue_op
  ) const {
    C += (sm) * ldc + (sn) * fdc;
    D += (sm) * ldd + sn;

    for (short i = 0; i < TM; i++) {
      for (short j = 0; j < TN; j++) {
        thread const auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kElemsPerFrag; k++) {
          D[offset_d + k] = epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
        }
      }
    }
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const Epilogue& epilogue_op
  ) const {
    C += (sm) * ldc + (sn) * fdc;
    D += (sm) * ldd + sn;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        for (int j = 0; j < TN; j++) {
          thread const auto& accum = Ctile.frag_at(i, j);
          int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
          int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < kElemsPerFrag; k++) {
            if ((j * TN_stride + k) < dst_tile_dims.x) {
              D[offset_d + k] =
                  epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
            }
          }
        }
      }
    }
  }
};

} // namespace steel
