#pragma once

#include "../common/defines.h"
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

namespace matmul_utils {

template <typename T>
struct pointer_element_t_impl {
  using type = T;
};
template <typename T>
struct pointer_element_t_impl<device T*> {
  using type = T;
};
template <typename T>
struct pointer_element_t_impl<threadgroup T*> {
  using type = T;
};
template <typename T>

using pointer_element_t = typename pointer_element_t_impl<T>::type;

template <typename T, typename U = T>
struct TransformNone {
  METAL_FUNC static U apply(U val) { return val; }
};

template <typename T, int FRAG_ROWS_, int FRAG_COLS_>
struct BaseMMAFrag {
  static_assert(
      FRAG_ROWS_ == 8,
      "Only 8 x 8 fragment matrices are currently supported"
  );
  static_assert(
      FRAG_COLS_ == 8,
      "Only 8 x 8 fragment matrices are currently supported"
  );
};

template <typename T>
struct BaseMMAFrag<T, 8, 8> {
  METAL_CONST int FRAG_ROWS = 8;
  METAL_CONST int FRAG_COLS = 8;
  METAL_CONST int ELEMS_PER_FRAG = (FRAG_ROWS * FRAG_COLS) / 32;
  METAL_CONST int ELEM_ROWS = 1;
  METAL_CONST int ELEM_COLS = 2;

  static_assert(
      ELEM_ROWS * ELEM_COLS == ELEMS_PER_FRAG,
      "MMAFrag shape is not consistent with MMAFrag size"
  );

  typedef metal::simdgroup_matrix<T, FRAG_ROWS, FRAG_COLS> mat_type;
  typedef metal::vec<T, ELEMS_PER_FRAG> frag_type;

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
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < ELEM_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < ELEM_COLS; j++) {
        dst[i * ELEM_COLS + j] = static_cast<T>(src[i * str_x + j * str_y]);
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

    METAL_PRAGMA_UNROLL
    for (short i = 0; i < ELEM_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < ELEM_COLS; j++) {
        dst[i * str_x + j * str_y] = static_cast<U>(src[i * ELEM_COLS + j]);
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

    METAL_PRAGMA_UNROLL
    for (short i = 0; i < ELEM_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < ELEM_COLS; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[(off_x + i) * str_x + (off_y + j) * str_y] =
              static_cast<U>(src[i * ELEM_COLS + j]);
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

template <
    typename T,
    int TILE_ROWS_,
    int TILE_COLS_,
    class MMAFrag_ = BaseMMAFrag<T, 8, 8>>
struct MMATile {
  using MMAFrag_t = MMAFrag_;
  using elem_type = T;
  METAL_CONST int FRAG_ROWS = MMAFrag_t::FRAG_ROWS;
  METAL_CONST int FRAG_COLS = MMAFrag_t::FRAG_COLS;
  METAL_CONST int ELEMS_PER_FRAG = MMAFrag_t::ELEMS_PER_FRAG;

  METAL_CONST int TILE_ROWS = TILE_ROWS_;
  METAL_CONST int TILE_COLS = TILE_COLS_;

  METAL_CONST int ROWS = TILE_ROWS * FRAG_ROWS;
  METAL_CONST int COLS = TILE_COLS * FRAG_COLS;

  METAL_CONST int NUM_FRAGS = TILE_ROWS * TILE_COLS;
  METAL_CONST int ELEMS_PER_TILE = NUM_FRAGS * ELEMS_PER_FRAG;

  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;

  frag_type val_frags[NUM_FRAGS] = {frag_type(0)};

  METAL_FUNC MMATile() thread {}

  METAL_FUNC constexpr void clear() {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < NUM_FRAGS; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * TILE_COLS + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j
  ) const {
    return val_frags[i * TILE_COLS + j];
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  METAL_FUNC const thread elem_type* elems() const {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < TILE_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < TILE_COLS; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(
                src[(i * FRAG_ROWS) * w_x * str_x +
                    (j * FRAG_COLS) * w_y * str_y]
            ),
            str_x,
            str_y
        );
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store(device U* dst, const int ld) const {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < TILE_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < TILE_COLS; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(dst[(i * FRAG_ROWS) * w_x * ld + (j * FRAG_COLS) * w_y]),
            ld,
            1
        );
      }
    }
  }

  // Store results into threadgroup memory (same layout as device variant)
  template <typename U, int w_x, int w_y>
  METAL_FUNC void store_tg(threadgroup U* dst, const int ld) const {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < TILE_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < TILE_COLS; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(dst[(i * FRAG_ROWS) * w_x * ld + (j * FRAG_COLS) * w_y]),
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
    METAL_PRAGMA_UNROLL
    for (int i = 0; i < TILE_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (int j = 0; j < TILE_COLS; ++j) {
        MMAFrag_t::store_safe(
            frag_at(i, j),
            dst,
            ld,
            1,
            dst_tile_dims.y,
            dst_tile_dims.x,
            (i * FRAG_ROWS) * w_x,
            (j * FRAG_COLS) * w_y
        );
      }
    }
  }
};

template <typename T, typename U, int M, int N, int K>
METAL_FUNC void tile_matmad(
    thread MMATile<T, M, N>& D,
    thread MMATile<U, M, K>& A,
    thread MMATile<U, K, N>& B,
    thread MMATile<T, M, N>& C
) {
  METAL_PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    METAL_PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short n_serp = (m % 2) ? (N - 1 - n) : n;
      METAL_PRAGMA_UNROLL
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
  METAL_CONST short FRAG_SIZE = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, FRAG_SIZE, FRAG_SIZE>;

  // Warp tile simdgroup matrix strides along M
  METAL_CONST short TM_stride = FRAG_SIZE * WM;
  // Warp tile simdgroup matrix strides along M
  METAL_CONST short TN_stride = FRAG_SIZE * WN;

  // Warp tile size along M
  METAL_CONST short TM = BM / (FRAG_SIZE * WM);
  // Warp tile size along N
  METAL_CONST short TN = BN / (FRAG_SIZE * WN);

  // Threadgroup A strides
  METAL_CONST short A_str_m = transpose_a ? 1 : lda_tgp; // M
  METAL_CONST short A_str_k = transpose_a ? lda_tgp : 1; // K

  // Threadgroup B strides
  METAL_CONST short B_str_k = transpose_b ? 1 : ldb_tgp; // K
  METAL_CONST short B_str_n = transpose_b ? ldb_tgp : 1; // N

  // Threadgroup strides along K
  METAL_CONST short tile_stride_a = FRAG_SIZE * A_str_k;
  METAL_CONST short tile_stride_b = FRAG_SIZE * B_str_k;

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
    short tm = FRAG_SIZE * (simd_group_id / WN);
    short tn = FRAG_SIZE * (simd_group_id % WN);

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

    // Iterate over BK in blocks of FRAG_SIZE
    METAL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += FRAG_SIZE) {
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
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::ELEMS_PER_TILE; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;

    Ctile.template store<U, WM, WN>(D, ldd);
  }

  /* Store results into threadgroup memory */
  METAL_FUNC void store_result_tg(threadgroup U* D, const int ldd) {
    // Apply epilogue
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::ELEMS_PER_TILE; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;

    Ctile.template store_tg<U, WM, WN>(D, ldd);
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int ldd,
      short2 dst_tile_dims
  ) {
    // Apply epilogue
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::ELEMS_PER_TILE; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Ctile.template store_safe<U, WM, WN>(D, ldd, dst_tile_dims);
  }
};

template <
    typename T,
    int BM,
    int BK,
    int lda_tgp,
    int reduction_dim,
    int tgp_size,
    int align_M = 1,
    int align_K = 4>
struct BlockLoader {
  METAL_CONST short n_reads = (BM * BK < tgp_size) ? 1 : (BM * BK) / tgp_size;

  const int src_ld;
  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device T* src;

  BlockLoader(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  )
      : src_ld(src_ld_), thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BK), bj((n_reads * thread_idx) % BK),
        dst(dst_ + bi * lda_tgp + bj), src(src_ + bi * src_ld + bj) {}

  void load_unsafe() const {
    if (BM * BK < tgp_size && bi >= BM) {
      return;
    }

    for (int i = 0; i < n_reads; i++) {
      dst[i] = src[i];
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BM * BK < tgp_size && bi >= BM) {
      return;
    }

    if (bi >= src_tile_dim.y || bj >= src_tile_dim.x) {
      for (int i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }

    for (int i = 0; i < n_reads; i++) {
      dst[i] = src[i];
    }
  }

  void next() {
    src += BK; // Advance by tile size
  }
};

} // namespace matmul_utils