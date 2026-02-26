// clang-format off
#include "../../../common/utils.h"
#include "../../../definitions.metal"

#include "../../common/mpp_gemm.h"

using namespace uzu::matmul;

namespace uzu {
namespace matmul {
using GEMMParams = steel::GEMMParams;
} // namespace matmul
} // namespace uzu

///////////////////////////////////////////////////////////////////////////////
// Supported MPP A/B/output combinations
///////////////////////////////////////////////////////////////////////////////

template <typename AType, typename BType, typename OutType>
struct MppCombo {
  using IsSupported = metal::false_type;
  using AccumType = float;
};

template <>
struct MppCombo<half, half, half> {
  using IsSupported = metal::true_type;
  using AccumType = float;
};

template <>
struct MppCombo<half, half, float> {
  using IsSupported = metal::true_type;
  using AccumType = float;
};

template <>
struct MppCombo<bfloat16_t, bfloat16_t, bfloat16_t> {
  using IsSupported = metal::true_type;
  using AccumType = float;
};

template <>
struct MppCombo<bfloat16_t, bfloat16_t, float> {
  using IsSupported = metal::true_type;
  using AccumType = float;
};

template <>
struct MppCombo<float, float, float> {
  using IsSupported = metal::true_type;
  using AccumType = float;
};

template <>
struct MppCombo<int8_t, int8_t, int> {
  using IsSupported = metal::true_type;
  using AccumType = int32_t;
};

template <>
struct MppCombo<int8_t, half, half> {
  using IsSupported = metal::true_type;
  using AccumType = float;
};

template <>
struct MppCombo<int8_t, half, float> {
  using IsSupported = metal::true_type;
  using AccumType = float;
};

template <>
struct MppCombo<int8_t, float, float> {
  using IsSupported = metal::true_type;
  using AccumType = float;
};

template <>
struct MppCombo<int8_t, bfloat16_t, bfloat16_t> {
  using IsSupported = metal::true_type;
  using AccumType = float;
};

template <>
struct MppCombo<int8_t, bfloat16_t, float> {
  using IsSupported = metal::true_type;
  using AccumType = float;
};

template <>
struct MppCombo<bfloat16_t, int8_t, bfloat16_t> {
  using IsSupported = metal::true_type;
  using AccumType = float;
};

template <>
struct MppCombo<bfloat16_t, int8_t, float> {
  using IsSupported = metal::true_type;
  using AccumType = float;
};

///////////////////////////////////////////////////////////////////////////////
// MPP GEMM implementation -- templated over tile config
///////////////////////////////////////////////////////////////////////////////

template <
    typename AType,
    typename BType,
    typename AccumType,
    typename OutType,
    short BM_,
    short BN_,
    short BK_,
    short WM_,
    short WN_>
METAL_FUNC void gemm_mpp_impl(
    const device AType* a,
    const device BType* b,
    device OutType* d,
    const constant GEMMParams* params,
    const bool align_m,
    const bool align_n,
    const bool align_k,
    uint simd_group_id,
    uint3 tid
) {
  constexpr short BM = BM_;
  constexpr short BN = BN_;
  constexpr short BK = BK_;
  constexpr short WM = WM_;
  constexpr short WN = WN_;

  const int tid_y = ((tid.y) << params->swizzle_log) +
                    ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  a += params->batch_stride_a * tid.z;
  b += params->batch_stride_b * tid.z;
  d += params->batch_stride_d * tid.z;

  threadgroup_barrier(mem_flags::mem_none);

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  a += c_row_long * params->lda;
  b += c_col_long * params->ldb;
  d += c_row_long * params->ldd + c_col_long;

  constexpr short UM = 16;
  constexpr short UN = 32;
  constexpr short UK = 16;
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / UM;
  constexpr short TN = SN / UN;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const int sgp_sm_int =
      align_m ? int(SM) : min(int(SM), params->M - (c_row + tm));
  const short sgp_sm = short(sgp_sm_int);

  const int sgp_sn_int =
      align_n ? int(SN) : min(int(SN), params->N - (c_col + tn));
  const short sgp_sn = short(sgp_sn_int);

  const bool is_unaligned_sm = align_m ? false : (sgp_sm != SM);
  const bool is_unaligned_sn = align_n ? false : (sgp_sn != SN);

  a += tm * params->lda;
  b += tn * params->ldb;
  d += tm * params->ldd + tn;

  using DSubTile = MppSubTile<AccumType, UM, UN>;
  MppTile<AccumType, TM, TN, DSubTile> Dtile;

  dispatch_bool(align_k, [&](auto kAlignedK) {
    dispatch_bool(align_m || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_n || !is_unaligned_sn, [&](auto kAlignedN) {
        Dtile = mpp_gemm_loop<
            AType,
            BType,
            SM,
            SN,
            SK,
            BK,
            false,
            true,
            kAlignedM.value,
            kAlignedN.value,
            kAlignedK.value,
            UM,
            UN,
            UK,
            AccumType>(
            a,
            b,
            params->lda,
            params->ldb,
            params->K,
            params->gemm_k_iterations_aligned,
            sgp_sm,
            sgp_sn);
        if (kAlignedM.value && kAlignedN.value) {
          Dtile.store(d, int(params->ldd));
        } else {
          Dtile.store_safe(d, int(params->ldd), short2(sgp_sn, sgp_sm));
        }
      });
    });
  });
}

///////////////////////////////////////////////////////////////////////////////
// DSL kernel entry point
///////////////////////////////////////////////////////////////////////////////

template <typename AType, typename BType, typename OutType>
VARIANTS(AType, half, bfloat, float, int8_t)
VARIANTS(BType, half, bfloat, float, int8_t)
VARIANTS(OutType, half, bfloat, float, int)
KERNEL(MatmulGemmMpp)(
    const device AType* a,
    const device BType* b,
    device OutType* d,
    const constant uzu::matmul::GEMMParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    const uint block_rows SPECIALIZE,
    const uint block_cols SPECIALIZE,
    const uint block_depth SPECIALIZE,
    const uint warps_per_row SPECIALIZE,
    const uint warps_per_col SPECIALIZE,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool align_k SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(4),
    const uint thread_z THREADS(4),
    const Simd simd
) {
  if (simd.group_idx >= warps_per_row * warps_per_col) {
    return;
  }

  IF_CONSTEXPR((MppCombo<AType, BType, OutType>::IsSupported::value)) {
    using AccumT = typename MppCombo<AType, BType, OutType>::AccumType;

    // Dispatch to the correct compile-time tile config
    if (block_rows == 128 && block_cols == 128 && block_depth == 512 &&
        warps_per_row == 4 && warps_per_col == 4) {
      gemm_mpp_impl<AType, BType, AccumT, OutType, 128, 128, 512, 4, 4>(
          a, b, d, params,
          align_m, align_n, align_k,
          simd.group_idx,
          uint3(group_x, group_y, group_z));
    } else {
      gemm_mpp_impl<AType, BType, AccumT, OutType, 64, 64, 256, 2, 2>(
          a, b, d, params,
          align_m, align_n, align_k,
          simd.group_idx,
          uint3(group_x, group_y, group_z));
    }
  }
}

// clang-format on
