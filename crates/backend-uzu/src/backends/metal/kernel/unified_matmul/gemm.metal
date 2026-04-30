#include "../common/dsl.h"
#include "../common/thread_context.h"

#include "common/pipeline.h"

using namespace metal;

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(UnifiedGemmSimdgroupFullPrecision)(
    const device T* a,
    const device T* b,
    device T* d,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(32),
    const ThreadContext thread_context
) {
  (void)a;
  (void)b;
  (void)d;
  (void)group_x;
  (void)group_y;
  (void)thread_x;
  (void)thread_context;
  uzu::unified_gemm::GemmPipeline<T>::run();
}

