#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

#include "steel/defines.h"
#include "steel/utils/type_traits.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

namespace uzu {
namespace matmul {

template <int start, int stop, int step, typename F>
constexpr void const_for_loop(F f) {
  if constexpr (start < stop) {
    constexpr auto idx = metal::integral_constant<int, start>{};
    f(idx);
    const_for_loop<start + step, stop, step, F>(f);
  }
}

///////////////////////////////////////////////////////////////////////////////
// MPP Fragment - 16x16 cooperative fragment
///////////////////////////////////////////////////////////////////////////////

struct BaseMppFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;

  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST short kElemRows = 2;
  STEEL_CONST short kElemCols = 4;

  STEEL_CONST short kElemRowsJump = 8;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MppFrag shape is not consistent with MppFrag size");

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;
    return short2{fn, fm};
  }

  METAL_FUNC static short2 get_coord(short idx) {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3)) + (idx >> 2) * 8;
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4 + idx % 4;
    return short2{fn, fm};
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = int,
      typename OffY = int>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = 0,
      OffY off_y = 0) {
    const short2 sc = get_coord();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump + sc.y;
      const auto c = off_y + sc.x;

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] =
            static_cast<T>(src[r * str_x + (c + j) * str_y]);
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = int,
      typename OffY = int>
  METAL_FUNC static constexpr void load_rows(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = 0,
      OffY off_y = 0) {
    const short2 sc = get_coord();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump + sc.y;
      const auto c = off_y + sc.x;

      if (r < lim_x) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        }
      } else {
        dst = dtype_frag_t<T>(0);
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = int,
      typename OffY = int>
  METAL_FUNC static constexpr void load_safe(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = 0,
      OffY off_y = 0) {
    const short2 sc = get_coord();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump + sc.y;
      const auto c = off_y + sc.x;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if (r < lim_x && (c + j) < lim_y) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename OffX = int,
      typename OffY = int>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      OffX off_x = 0,
      OffY off_y = 0) {
    using U = metal::pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump + sc.y;
      const auto c = off_y + sc.x;

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[r * str_x + (c + j) * str_y] =
            static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = int,
      typename OffY = int>
  METAL_FUNC static constexpr void store_safe(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = 0,
      OffY off_y = 0) {
    using U = metal::pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump + sc.y;
      const auto c = off_y + sc.x;

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if (r < lim_x && (c + j) < lim_y) {
          dst[r * str_x + (c + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_reduce(
      thread const dtype_frag_t<T>& inp_vals,
      thread T* reduced_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      T thr_reduce = Op::apply(
          Op::apply(inp_vals[i * kElemCols + 0], inp_vals[i * kElemCols + 1]),
          Op::apply(inp_vals[i * kElemCols + 2], inp_vals[i * kElemCols + 3]));

      T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

      T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

      reduced_vals[i] = Op::apply(reduced_vals[i], sgr_reduce);
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_bin_op(
      thread dtype_frag_t<T>& inp_vals,
      thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// MPP SubTile - grid of 16x16 fragments
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    short kRows_,
    short kCols_,
    typename MppFrag_t = BaseMppFrag>
struct MppSubTile {
  STEEL_CONST short kRows = kRows_;
  STEEL_CONST short kCols = kCols_;

  STEEL_CONST short kFragRows = MppFrag_t::kFragRows;
  STEEL_CONST short kFragCols = MppFrag_t::kFragCols;
  STEEL_CONST short kElemsPerFrag = MppFrag_t::kElemsPerFrag;

  STEEL_CONST short kSubTileRows = kRows / kFragRows;
  STEEL_CONST short kSubTileCols = kCols / kFragCols;

  STEEL_CONST short kNumFrags = kSubTileRows * kSubTileCols;
  STEEL_CONST short kElemsPerSubTile = kNumFrags * kElemsPerFrag;

  STEEL_CONST int kRowsPerThread = kSubTileRows * MppFrag_t::kElemRows;
  STEEL_CONST int kColsPerThread = kSubTileCols * MppFrag_t::kElemCols;

  STEEL_CONST short kFragThrRows = MppFrag_t::kElemRows;
  STEEL_CONST short kFragThrCols = MppFrag_t::kElemCols;
  STEEL_CONST short kFragRowsJump = MppFrag_t::kElemRowsJump;

  using frag_type = typename MppFrag_t::template dtype_frag_t<T>;

  frag_type val_frags[kNumFrags];

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kSubTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const {
    return val_frags[i * kSubTileCols + j];
  }

  METAL_FUNC thread T* elems() {
    return reinterpret_cast<thread T*>(val_frags);
  }

  METAL_FUNC const thread T* elems() const {
    return reinterpret_cast<const thread T*>(val_frags);
  }

  template <
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = int,
      typename OffY = int>
  METAL_FUNC constexpr void load(
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = 0,
      OffY off_y = 0) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kSubTileCols; ++j) {
        MppFrag_t::load(
            frag_at(i, j),
            src,
            str_x,
            str_y,
            off_x + i * kFragRows,
            off_y + j * kFragCols);
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename OffX = int,
      typename OffY = int>
  METAL_FUNC constexpr void store(
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      OffX off_x = 0,
      OffY off_y = 0) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kSubTileCols; ++j) {
        MppFrag_t::store(
            frag_at(i, j),
            dst,
            str_x,
            str_y,
            off_x + i * kFragRows,
            off_y + j * kFragCols);
      }
    }
  }

  template <
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = int,
      typename OffY = int>
  METAL_FUNC constexpr void load_safe(
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = 0,
      OffY off_y = 0) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kSubTileCols; ++j) {
        MppFrag_t::load_safe(
            frag_at(i, j),
            src,
            str_x,
            str_y,
            lim_x,
            lim_y,
            off_x + (i * kFragRows),
            off_y + (j * kFragCols));
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = int,
      typename OffY = int>
  METAL_FUNC constexpr void store_safe(
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = 0,
      OffY off_y = 0) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kSubTileCols; ++j) {
        MppFrag_t::store_safe(
            frag_at(i, j),
            dst,
            str_x,
            str_y,
            lim_x,
            lim_y,
            off_x + (i * kFragRows),
            off_y + (j * kFragCols));
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// MPP SubTile matmul via MPP matmul2d
///////////////////////////////////////////////////////////////////////////////

template <
    short RC,
    short CC,
    short RA,
    short CA,
    short RB,
    short CB,
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a,
    bool transpose_b,
    typename MppFrag_t = BaseMppFrag>
METAL_FUNC void subtile_matmad_mpp(
    thread MppSubTile<CType, RC, CC, MppFrag_t>& C,
    thread MppSubTile<AType, RA, CA, MppFrag_t>& A,
    metal::bool_constant<transpose_a>,
    thread MppSubTile<BType, RB, CB, MppFrag_t>& B,
    metal::bool_constant<transpose_b>) {

  constexpr short FMa = transpose_a ? CA : RA;
  constexpr short FMc = RC;
  static_assert(FMa == FMc, "MPP matmul: M dimensions do not match");

  constexpr short FNb = transpose_b ? RB : CB;
  constexpr short FNc = CC;
  static_assert(FNb == FNc, "MPP matmul: N dimensions do not match");

  constexpr short FKa = transpose_a ? RA : CA;
  constexpr short FKb = transpose_b ? CB : RB;
  static_assert(FKa == FKb, "MPP matmul: K dimensions do not match");

  constexpr short FM = FMc;
  constexpr short FN = FNc;
  constexpr short FK = FKa;

  constexpr int TM = FM / 16;
  constexpr int TN = FN / 16;
  constexpr int TK = FK / 16;

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      FM,
      FN,
      FK,
      transpose_a,
      transpose_b,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      CType>();

  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < TK; kk++) {
      const short fi = transpose_a ? kk : mm;
      const short fj = transpose_a ? mm : kk;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 8; i++) {
        ct_a[(TK * mm + kk) * 8 + i] = A.frag_at(fi, fj)[i];
      }
    }
  }

  STEEL_PRAGMA_UNROLL
  for (short nn = 0; nn < TN; nn++) {
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < TK; kk++) {
      const short fi = transpose_b ? nn : kk;
      const short fj = transpose_b ? kk : nn;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 8; i++) {
        ct_b[(TN * kk + nn) * 8 + i] = B.frag_at(fi, fj)[i];
      }
    }
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < ct_c.get_capacity(); i++) {
    ct_c[i] = C.elems()[i];
  }

  gemm_op.run(ct_a, ct_b, ct_c);

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < ct_c.get_capacity(); i++) {
    C.elems()[i] = ct_c[i];
  }
}

///////////////////////////////////////////////////////////////////////////////
// MPP SubTile full GEMM -- device-pointer version with proper coordinate mapping
// Handles the full K-dimension loop for one SM×SN output block
///////////////////////////////////////////////////////////////////////////////

template <
    short SM,
    short SN,
    short UK,
    typename AccumType,
    typename AType,
    typename BType,
    typename OutType,
    bool transpose_a,
    bool transpose_b>
METAL_FUNC void mpp_subtile_gemm_direct(
    const device AType* a_ptr, int lda,
    const device BType* b_ptr, int ldb,
    device OutType* d_ptr, int ldd,
    int K,
    short m_limit, short n_limit) {

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      SM, SN, UK, transpose_a, transpose_b, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<AType, BType, AccumType>();
  auto ct_b = gemm_op.template get_right_input_cooperative_tensor<AType, BType, AccumType>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      decltype(ct_a), decltype(ct_b), AccumType>();

  // Precompute coordinate mappings (constant per thread, reused across K iterations)
  const short A_CAP = ct_a.get_capacity();
  const short B_CAP = ct_b.get_capacity();
  const short C_CAP = ct_c.get_capacity();

  // Max capacity for cooperative tensors in MPP is small (typically 8-16 per operand)
  short a_col[16], a_row[16];
  short b_col[16], b_row[16];
  short c_col[16], c_row[16];
  bool c_valid[16];

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < A_CAP; i++) {
    auto coord = ct_a.get_multidimensional_index(i);
    a_col[i] = coord[0];
    a_row[i] = coord[1];
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < B_CAP; i++) {
    auto coord = ct_b.get_multidimensional_index(i);
    b_col[i] = coord[0];
    b_row[i] = coord[1];
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < C_CAP; i++) {
    auto coord = ct_c.get_multidimensional_index(i);
    c_col[i] = coord[0];
    c_row[i] = coord[1];
    c_valid[i] = ct_c.is_valid_element(i);
  }

  // Precompute A/B linear offsets for the non-K dimensions (constant across K)
  int a_base_offset[16];
  int b_base_offset[16];
  bool a_in_bounds[16];
  bool b_in_bounds[16];

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < A_CAP; i++) {
    // For non-transposed A: row=M, col=K -> base = row * lda, k offset added in loop
    // For transposed A: row=K, col=M -> base = col (M part), k offset = row
    if constexpr (!transpose_a) {
      a_base_offset[i] = a_row[i] * lda;
      a_in_bounds[i] = a_row[i] < m_limit;
    } else {
      a_base_offset[i] = a_col[i] * lda;  // M dim * lda
      a_in_bounds[i] = a_col[i] < m_limit;
    }
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < B_CAP; i++) {
    if constexpr (!transpose_b) {
      b_base_offset[i] = b_col[i];
      b_in_bounds[i] = b_col[i] < n_limit;
    } else {
      b_base_offset[i] = b_row[i] * ldb;  // N dim * ldb
      b_in_bounds[i] = b_row[i] < n_limit;
    }
  }

  // Initialize accumulator to zero
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < C_CAP; i++) {
    ct_c[i] = AccumType(0);
  }

  // Main K loop
  for (int k = 0; k < K; k += UK) {
    const short k_limit = short(min(int(UK), K - k));

    // Load A
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < A_CAP; i++) {
      const short k_coord = transpose_a ? a_row[i] : a_col[i];
      if (a_in_bounds[i] && k_coord < k_limit) {
        if constexpr (!transpose_a) {
          ct_a[i] = a_ptr[a_base_offset[i] + a_col[i]];
        } else {
          ct_a[i] = a_ptr[a_base_offset[i] + a_row[i]];
        }
      } else {
        ct_a[i] = AType(0);
      }
    }

    // Load B
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < B_CAP; i++) {
      const short k_coord = transpose_b ? b_col[i] : b_row[i];
      if (b_in_bounds[i] && k_coord < k_limit) {
        if constexpr (!transpose_b) {
          ct_b[i] = b_ptr[b_row[i] * ldb + b_col[i]];
        } else {
          ct_b[i] = b_ptr[b_base_offset[i] + b_col[i]];
        }
      } else {
        ct_b[i] = BType(0);
      }
    }

    gemm_op.run(ct_a, ct_b, ct_c);

    // Advance pointers along K
    a_ptr += transpose_a ? (UK * lda) : UK;
    b_ptr += transpose_b ? UK : (UK * ldb);
  }

  // Store result
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < C_CAP; i++) {
    if (c_valid[i] && c_row[i] < m_limit && c_col[i] < n_limit) {
      d_ptr[c_row[i] * ldd + c_col[i]] = OutType(ct_c[i]);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// MPP Tile - grid of subtiles
///////////////////////////////////////////////////////////////////////////////

template <typename T, short kTileRows_, short kTileCols_, class MppSubTile_>
struct MppTile {
  using MppSubTile_t = MppSubTile_;
  using elem_type = T;
  STEEL_CONST short kSubTileRows = MppSubTile_t::kRows;
  STEEL_CONST short kSubTileCols = MppSubTile_t::kCols;
  STEEL_CONST short kElemsPerSubTile = MppSubTile_t::kElemsPerSubTile;

  STEEL_CONST short kTileRows = kTileRows_;
  STEEL_CONST short kTileCols = kTileCols_;

  STEEL_CONST short kRows = kTileRows * kSubTileRows;
  STEEL_CONST short kCols = kTileCols * kSubTileCols;

  STEEL_CONST short kSubTiles = kTileRows * kTileCols;
  STEEL_CONST short kElemsPerTile = kSubTiles * kElemsPerSubTile;

  STEEL_CONST short kRowsPerThread = kTileRows * MppSubTile_t::kRowsPerThread;
  STEEL_CONST short kColsPerThread = kTileCols * MppSubTile_t::kColsPerThread;

  STEEL_CONST short kSubTileThrRows = MppSubTile_t::kRowsPerThread;
  STEEL_CONST short kSubTileThrCols = MppSubTile_t::kColsPerThread;

  MppSubTile_t val_subtiles[kSubTiles];

  METAL_FUNC MppTile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kSubTiles; ++i) {
      val_subtiles[i].clear();
    }
  }

  METAL_FUNC constexpr thread MppSubTile_t& subtile_at(
      const short i,
      const short j) {
    return val_subtiles[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread MppSubTile_t& subtile_at(
      const short i,
      const short j) const {
    return val_subtiles[i * kTileCols + j];
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_subtiles[0].elems());
  }

  METAL_FUNC const thread elem_type* elems() const {
    return reinterpret_cast<const thread elem_type*>(val_subtiles[0].elems());
  }

  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).load(
            &src[(i * kSubTileRows * ld + j * kSubTileCols)], ld, 1);
      }
    }
  }

  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).store(
            &dst[(i * kSubTileRows * ld + j * kSubTileCols)], ld, 1);
      }
    }
  }

  template <typename U>
  METAL_FUNC void
  load_safe(const device U* src, const int ld, const short2 src_tile_dims) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).load_safe(
            src,
            ld,
            1,
            src_tile_dims.y,
            src_tile_dims.x,
            i * kSubTileRows,
            j * kSubTileCols);
      }
    }
  }

  template <typename U>
  METAL_FUNC void
  store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).store_safe(
            dst,
            ld,
            1,
            dst_tile_dims.y,
            dst_tile_dims.x,
            i * kSubTileRows,
            j * kSubTileCols);
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// MPP Tile matmul - iterates subtiles
///////////////////////////////////////////////////////////////////////////////

template <
    class CTile,
    class ATile,
    class BTile,
    bool transpose_a,
    bool transpose_b>
METAL_FUNC void tile_matmad_mpp(
    thread CTile& C,
    thread ATile& A,
    metal::bool_constant<transpose_a>,
    thread BTile& B,
    metal::bool_constant<transpose_b>) {

  constexpr short TM = CTile::kTileRows;
  constexpr short TN = CTile::kTileCols;
  constexpr short TK = transpose_a ? ATile::kTileRows : ATile::kTileCols;

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < TM; ++i) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < TN; ++j) {
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < TK; ++k) {
        const short ra = transpose_a ? k : i;
        const short ca = transpose_a ? i : k;
        const short rb = transpose_b ? j : k;
        const short cb = transpose_b ? k : j;

        subtile_matmad_mpp(
            C.subtile_at(i, j),
            A.subtile_at(ra, ca),
            metal::bool_constant<transpose_a>{},
            B.subtile_at(rb, cb),
            metal::bool_constant<transpose_b>{});
      }
    }
  }
}

} // namespace matmul
} // namespace uzu
