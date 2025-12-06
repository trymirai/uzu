#pragma once

#include "defines.h"

using namespace metal;

namespace uzu {
namespace matmul {

///////////////////////////////////////////////////////////////////////////////
// MMA Fragment - 8x8 simdgroup matrix fragment
///////////////////////////////////////////////////////////////////////////////

template <typename T, int kFragRows_, int kFragCols_>
struct BaseMMAFrag {
  static_assert(
      kFragRows_ == 8,
      "Only 8 x 8 fragment matrices are currently supported"
  );
  static_assert(
      kFragCols_ == 8,
      "Only 8 x 8 fragment matrices are currently supported"
  );
};

template <typename T>
struct BaseMMAFrag<T, 8, 8> {
  UZU_MTL_CONST int kFragRows = 8;
  UZU_MTL_CONST int kFragCols = 8;
  UZU_MTL_CONST int kElemsPerFrag = (kFragRows * kFragCols) / 32;
  UZU_MTL_CONST int kElemRows = 1;
  UZU_MTL_CONST int kElemCols = 2;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size"
  );

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

  template <typename SrcPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void load(
      thread frag_type& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y
  ) {
    UZU_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      UZU_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] = static_cast<T>(src[i * str_x + j * str_y]);
      }
    }
  }

  template <
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void load_safe(
      thread frag_type& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = 0,
      OffY off_y = 0
  ) {
    UZU_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      UZU_PRAGMA_UNROLL
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

  template <typename DstPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void store(
      const thread frag_type& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y
  ) {
    using U = pointer_element_t<DstPtrType>;

    UZU_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      UZU_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * str_x + j * str_y] = static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void store_safe(
      const thread frag_type& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = 0,
      OffY off_y = 0
  ) {
    using U = pointer_element_t<DstPtrType>;

    UZU_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      UZU_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
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

///////////////////////////////////////////////////////////////////////////////
// MMA Tile - collection of MMA fragments
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    int kTileRows_,
    int kTileCols_,
    class MMAFrag_ = BaseMMAFrag<T, 8, 8>>
struct MMATile {
  using MMAFrag_t = MMAFrag_;
  using elem_type = T;
  UZU_MTL_CONST int kFragRows = MMAFrag_t::kFragRows;
  UZU_MTL_CONST int kFragCols = MMAFrag_t::kFragCols;
  UZU_MTL_CONST int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;

  UZU_MTL_CONST int kTileRows = kTileRows_;
  UZU_MTL_CONST int kTileCols = kTileCols_;

  UZU_MTL_CONST int kRows = kTileRows * kFragRows;
  UZU_MTL_CONST int kCols = kTileCols * kFragCols;

  UZU_MTL_CONST int kNumFrags = kTileRows * kTileCols;
  UZU_MTL_CONST int kElemsPerTile = kNumFrags * kElemsPerFrag;

  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;

  frag_type val_frags[kNumFrags] = {frag_type(0)};

  METAL_FUNC MMATile() thread {}

  METAL_FUNC constexpr void clear() {
    UZU_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
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

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    UZU_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      UZU_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(
                src[(i * kFragRows) * w_x * str_x +
                    (j * kFragCols) * w_y * str_y]
            ),
            str_x,
            str_y
        );
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store(device U* dst, const int ld) const {
    UZU_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      UZU_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(dst[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld,
            1
        );
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store_safe(
      device U* dst,
      const int ld,
      const short2 dst_tile_dims
  ) const {
    UZU_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      UZU_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store_safe(
            frag_at(i, j),
            dst,
            ld,
            1,
            dst_tile_dims.y,
            dst_tile_dims.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y
        );
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// Tile matrix multiply-accumulate
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, int M, int N, int K>
METAL_FUNC void tile_matmad(
    thread MMATile<T, M, N>& D,
    thread MMATile<U, M, K>& A,
    thread MMATile<U, K, N>& B,
    thread MMATile<T, M, N>& C
) {
  UZU_PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    UZU_PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short n_serp = (m % 2) ? (N - 1 - n) : n;
      UZU_PRAGMA_UNROLL
      for (short k = 0; k < K; ++k) {
        MMATile<T, M, N>::MMAFrag_t::mma(
            D.frag_at(m, n_serp),
            A.frag_at(m, k),
            B.frag_at(k, n_serp),
            C.frag_at(m, n_serp)
        );
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Block MMA - manages the GEMM computation for a threadgroup
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    short lda_tgp,
    short ldb_tgp,
    typename AccumType = float,
    typename Epilogue = TransformNone<U, AccumType>>
struct BlockMMA {
  // MMAFrag size
  UZU_MTL_CONST short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  // Warp tile simdgroup matrix strides along M
  UZU_MTL_CONST short TM_stride = kFragSize * WM;
  // Warp tile simdgroup matrix strides along N
  UZU_MTL_CONST short TN_stride = kFragSize * WN;

  // Warp tile size along M
  UZU_MTL_CONST short TM = BM / (kFragSize * WM);
  // Warp tile size along N
  UZU_MTL_CONST short TN = BN / (kFragSize * WN);

  // Threadgroup A strides
  UZU_MTL_CONST short A_str_m = transpose_a ? 1 : lda_tgp; // M
  UZU_MTL_CONST short A_str_k = transpose_a ? lda_tgp : 1; // K

  // Threadgroup B strides
  UZU_MTL_CONST short B_str_k = transpose_b ? 1 : ldb_tgp; // K
  UZU_MTL_CONST short B_str_n = transpose_b ? ldb_tgp : 1; // N

  // Threadgroup strides along K
  UZU_MTL_CONST short tile_stride_a = kFragSize * A_str_k;
  UZU_MTL_CONST short tile_stride_b = kFragSize * B_str_k;

  // Simdgroup matrices
  MMATile<AccumType, TM, 1, MMAFrag_acc_t> Atile;
  MMATile<AccumType, 1, TN, MMAFrag_acc_t> Btile;
  MMATile<AccumType, TM, TN, MMAFrag_acc_t> Ctile;

  // Offsets within threadgroup
  short sm;
  short sn;

  short As_offset;
  short Bs_offset;

  /* Constructor */
  METAL_FUNC BlockMMA(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  ) {
    // Determine thread position in simdgroup matrix
    short tm = kFragSize * (simd_group_id / WN);
    short tn = kFragSize * (simd_group_id % WN);

    short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
    sm = simd_coord.y;
    sn = simd_coord.x;

    // Determine thread and simdgroup offset
    As_offset = (tm + sm) * A_str_m + (sn)*A_str_k; // M, K
    Bs_offset = (sm)*B_str_k + (tn + sn) * B_str_n; // K, N

    sm += tm;
    sn += tn;
  }

  /* (BM, BK) X (BK, BN) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
    // Adjust for simdgroup and thread location
    As += As_offset;
    Bs += Bs_offset;

    // Iterate over BK in blocks of kFragSize
    UZU_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += kFragSize) {
      simdgroup_barrier(mem_flags::mem_none);

      Atile.template load<T, WM, 1, A_str_m, A_str_k>(As);

      simdgroup_barrier(mem_flags::mem_none);

      Btile.template load<T, 1, WN, B_str_k, B_str_n>(Bs);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Ctile, Atile, Btile, Ctile);

      // Progress to next simdgroup tile
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U* D, const int ldd) {
    // Apply epilogue
    UZU_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;

    Ctile.template store<U, WM, WN>(D, ldd);
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int ldd,
      short2 dst_tile_dims
  ) {
    // Apply epilogue
    UZU_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Ctile.template store_safe<U, WM, WN>(D, ldd, dst_tile_dims);
  }

  /* Apply epilogue with C matrix */
  template <typename EpilogueOp>
  METAL_FUNC void apply_epilogue(
      const device U* C,
      const int ldc,
      const int fdc,
      thread const EpilogueOp& epilogue_op
  ) {
    const device U* C_ptr = C + sm * ldc + sn * fdc;

    UZU_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      UZU_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        thread auto& frag = Ctile.frag_at(i, j);
        UZU_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Ctile)::kElemsPerFrag; k++) {
          short row_offset = (i * kFragSize * WM);
          short col_offset = (j * kFragSize * WN);
          // Get the C element (accounting for fragment layout)
          U c_val = C_ptr[row_offset * ldc + col_offset * fdc + k];
          frag[k] = epilogue_op.apply(frag[k], static_cast<AccumType>(c_val));
        }
      }
    }
  }

  template <typename EpilogueOp>
  METAL_FUNC void apply_epilogue_safe(
      const device U* C,
      const int ldc,
      const int fdc,
      short2 tile_dims,
      thread const EpilogueOp& epilogue_op
  ) {
    const device U* C_ptr = C + sm * ldc + sn * fdc;
    tile_dims -= short2(sn, sm);

    UZU_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      UZU_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        thread auto& frag = Ctile.frag_at(i, j);
        short row_offset = (i * kFragSize * WM);
        short col_offset = (j * kFragSize * WN);
        UZU_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Ctile)::kElemsPerFrag; k++) {
          if (row_offset < tile_dims.y && col_offset + k < tile_dims.x) {
            U c_val = C_ptr[row_offset * ldc + col_offset * fdc + k];
            frag[k] = epilogue_op.apply(frag[k], static_cast<AccumType>(c_val));
          }
        }
      }
    }
  }
};

} // namespace matmul
} // namespace uzu
