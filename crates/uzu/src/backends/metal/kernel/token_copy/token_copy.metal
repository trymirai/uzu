#include <metal_stdlib>
using namespace metal;

/// Copies a single sampled token (u32) to the token_ids buffer (u64).
/// Used in async pipeline to pass sampled token to next forward pass.
[[max_total_threads_per_threadgroup(1)]]
kernel void copy_sampled_token(
    device const uint32_t* src [[buffer(0)]],
    device uint64_t* dst [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
  if (idx == 0) {
    dst[0] = static_cast<uint64_t>(src[0]);
  }
}

/// Copies a single sampled token (u32) to an indexed results buffer (u32).
/// Used in async pipeline to store result for callback access.
[[max_total_threads_per_threadgroup(1)]]
kernel void copy_token_to_results(
    device const uint32_t* src [[buffer(0)]],
    device uint32_t* dst [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
  if (idx == 0) {
    dst[0] = src[0];
  }
}
