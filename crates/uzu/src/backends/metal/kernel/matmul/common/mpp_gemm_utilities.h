#pragma once

#include "mpp_cooperative_matmul.h"

namespace uzu {
namespace matmul {

struct GEMMParams {
  const int M;
  const int N;
  const int K;

  const int leading_dim_a;
  const int leading_dim_b;
  const int leading_dim_d;

  const int tiles_n;
  const int tiles_m;

  const int64_t batch_stride_a;
  const int64_t batch_stride_b;
  const int64_t batch_stride_d;

  const int swizzle_log;
  const int gemm_k_iterations_aligned;

  const int batch_ndim;
};

} // namespace matmul
} // namespace uzu
