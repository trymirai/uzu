#include <metal_stdlib>
#include "../definitions.metal"

/// Copies a single sampled token (u32) to the token_ids buffer (u64).
/// Used in async pipeline to pass sampled token to next forward pass.
KERNEL(TokenCopySampled)(
    device const uint32_t* src,
    device uint64_t* dst,
    const uint idx AXIS(1, 1)
) {
  if (idx == 0) {
    dst[0] = static_cast<uint64_t>(src[0]);
  }
}

/// Copies a single sampled token (u32) to an indexed results buffer (u32).
/// Used in async pipeline to store result for callback access.
KERNEL(TokenCopyToResults)(
    device const uint32_t* src,
    device uint32_t* dst,
    const uint idx AXIS(1, 1)
) {
  if (idx == 0) {
    dst[0] = src[0];
  }
}
