#include <metal_stdlib>
#include "../definitions.metal"

#include "fwht.h"

// Block-wise in-place Hadamard transform.
// Divides each row of length n into n/BLOCK_SIZE independent blocks,
// and applies a BLOCK_SIZE-point Hadamard transform to each block.
//
// For BLOCK_SIZE=32 (Apple Silicon simdgroup size), uses simd_shuffle_xor
// for a barrier-free, shared-memory-free implementation.
// For larger block sizes, uses threadgroup memory.

template <typename T, int BLOCK_SIZE>
VARIANTS(T, half, float, bfloat)
VARIANTS(BLOCK_SIZE, 32, 64, 128, 256)
KERNEL(FwhtBlock)(
    device T* data,
    constant uint& batch_size,
    constant uint& n,
    constant float& scale,
    threadgroup float shared_buf[BLOCK_SIZE],
    const uint group_idx GROUPS(batch_size),
    const uint tid THREADS(256)
) {
  if (tid >= BLOCK_SIZE) return;
  uint num_blocks = n / BLOCK_SIZE;
  device T* row = data + group_idx * n;

  for (uint block = 0; block < num_blocks; block++) {
    device T* block_ptr = row + block * BLOCK_SIZE;

    IF_CONSTEXPR(BLOCK_SIZE == 32) {
      // Simdgroup shuffle path — no barriers, no shared memory.
      // Each thread holds one element; butterfly partners communicate
      // via simd_shuffle_xor which swaps values between lanes that
      // differ by the XOR mask.
      float val = float(block_ptr[tid]);

      STEEL_PRAGMA_UNROLL
      for (ushort h = 1; h < 32; h <<= 1) {
        float other = simd_shuffle_xor(val, h);
        val = (tid & h) ? (other - val) : (val + other);
      }

      block_ptr[tid] = T(val * scale);
    } else {
      // Threadgroup memory path for larger block sizes.
      shared_buf[tid] = float(block_ptr[tid]);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_UNROLL
      for (ushort h = 1; h < BLOCK_SIZE; h <<= 1) {
        uint partner = tid ^ h;
        float a = shared_buf[tid];
        float b = shared_buf[partner];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared_buf[tid] = (tid & h) ? (b - a) : (a + b);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      block_ptr[tid] = T(shared_buf[tid] * scale);

      // Barrier before next block iteration to ensure shared_buf is free
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}
