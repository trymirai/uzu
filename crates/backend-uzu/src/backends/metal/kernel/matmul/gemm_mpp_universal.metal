#include "../common/dsl.h"

#include "common/gemm_mpp_universal.h"

using namespace uzu::matmul;

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemmMppUniversal)(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant float& ab_scale,
    const bool apply_ab_scale SPECIALIZE,
    const bool is_accumulate SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(32)) {
  (void)thread_x;
  uzu::dispatch_bool(apply_ab_scale, [&](auto scale_enabled) {
    uzu::dispatch_bool(is_accumulate, [&](auto accumulate_enabled) {
      GemmMppUniversal<T, scale_enabled.value, accumulate_enabled.value>::run(
          left_matrix,
          right_matrix,
          output_matrix,
          params,
          ab_scale,
          uint2(group_x, group_y));
    });
  });
}
