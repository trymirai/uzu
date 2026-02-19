

#include "../../../common/utils.h"
#include "../../../definitions.metal"
#include "../../common/steel/gemm/gemm.h"
#include "steel_gemm_splitk.h"

namespace uzu {
namespace matmul {
using GEMMSpiltKParams = steel::GEMMSpiltKParams;
} // namespace matmul
} // namespace uzu

KERNEL(MatmulSplitKPartialBfloat16)(
    const device bfloat16_t* a,
    const device bfloat16_t* b,
    device float* c,
    const constant uzu::matmul::GEMMSpiltKParams* params,
    const constant uint& partial_group_count_x,
    const constant uint& partial_group_count_y,
    const constant uint& partial_group_count_z,
    const uint group_x GROUPS(partial_group_count_x),
    const uint group_y GROUPS(partial_group_count_y),
    const uint group_z GROUPS(partial_group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(2),
    const uint thread_z THREADS(2),
    threadgroup bfloat16_t a_shared[64],
    threadgroup bfloat16_t b_shared[64],
    const Simd simd
) {
  gemm_splitk_impl<bfloat16_t, float, 16, 32, 16, 2, 2, false, true, false, true>(
      a,
      b,
      c,
      params,
      a_shared,
      b_shared,
      simd.lane_idx,
      simd.group_idx,
      uint3(group_x, group_y, group_z),
      uint3(thread_x, thread_y, thread_z)
  );
}

KERNEL(MatmulSplitKAccumBfloat16)(
    const device float* c_split,
    device bfloat16_t* d,
    const constant int& k_partitions,
    const constant int& partition_stride,
    const constant int& ldd,
    const constant uint& accum_total_threads_x,
    const constant uint& accum_total_threads_y,
    const uint gid_x AXIS(accum_total_threads_x, 16),
    const uint gid_y AXIS(accum_total_threads_y, 16)
) {
  gemm_splitk_accum_impl<float, bfloat16_t>(
      c_split,
      d,
      k_partitions,
      partition_stride,
      ldd,
      uint2(gid_x, gid_y)
  );
}
