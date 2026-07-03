#include <metal_stdlib>
#include "../common/dsl.h"

/// Copies a single sampled token (u32) to the token_ids buffer (u64).
/// Used in decoding loop to pass sampled token to next forward pass.
PUBLIC KERNEL(TokenCopySampled)(device const uint32_t* src, device uint64_t* dst) {
  dst[0] = static_cast<uint64_t>(src[0]);
}
