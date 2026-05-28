#include "../../common/dsl.h"
#include "common/gemv_core.h"

using namespace metal;
using namespace uzu::gemm;
using namespace uzu::gemv;
using namespace uzu::matmul;

// Full-precision GEMV. The group-quantized path is the separate `qmv_fast`
// kernel; this entry point is a thin wrapper over the FP `GemvCore`.
template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(Gemv)(
    const device T* a,
    const device uint8_t* b_packed,
    device T* d,
    const device T* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* rht_factors
        OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const constant uzu::matmul::GemvParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const GemmDTransform output_transform SPECIALIZE,
    const GemvTiling gemv_tiling SPECIALIZE,
    threadgroup float partial_shared[GEMV_MAX_THREADGROUP_MEMORY],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(METAL_SIMD_SIZE),
    const uint thread_y THREADS(8),
    const uint thread_z THREADS(1),
    const ThreadContext thread_context
) {
  (void)group_x;
  (void)group_y;
  (void)thread_x;
  (void)thread_y;
  (void)thread_z;

  GemvCore<T>::run(
      a,
      b_packed,
      d,
      output_bias,
      rht_factors,
      params,
      output_transform,
      gemv_tiling,
      partial_shared,
      thread_context
  );
}
