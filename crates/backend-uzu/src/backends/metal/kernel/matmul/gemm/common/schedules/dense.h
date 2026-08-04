#pragma once

#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../common/fragment.h"
#include "../../../common/mxu_gemm_loop.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace schedules {

struct DenseSchedule {
  template <typename Core, bool ALIGNED_M, bool ALIGNED_N, bool ALIGNED_K>
  static METAL_FUNC typename Core::AccumFragment run(
      const device typename Core::LeftType* a_simdgroup,
      const device typename Core::RightType* b_simdgroup,
      const int leading_dimension_a,
      const int leading_dimension_b,
      const int k_elements,
      const int aligned_k_iterations,
      const short simdgroup_limit_m,
      const short simdgroup_limit_n,
      const thread ThreadContext& thread_context
  ) {
    return uzu::matmul::gemm_loop<
        typename Core::LeftType,
        typename Core::RightType,
        Core::SIMDGROUP_BLOCK_M,
        Core::SIMDGROUP_BLOCK_N,
        Core::SIMDGROUP_BLOCK_K,
        Core::THREADGROUP_BLOCK_K_FP,
        false,
        Core::TRANSPOSE_RIGHT,
        ALIGNED_M,
        ALIGNED_N,
        ALIGNED_K,
        typename Core::AccumulatorType>(
        a_simdgroup,
        b_simdgroup,
        leading_dimension_a,
        leading_dimension_b,
        k_elements,
        aligned_k_iterations,
        simdgroup_limit_m,
        simdgroup_limit_n,
        thread_context
    );
  }
};

} // namespace schedules
} // namespace gemm
} // namespace uzu
