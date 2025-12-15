#include <metal_stdlib>
#include "../definitions.metal"

#define GRAIN_SIZE 64
#define BLOCK_SIZE 1024

#define BITS_IN_U32 32

template <typename T>
void batched_bitmask(
    device const T* logits_data,
    device const uint32_t* bitmask_data,
    device T* processed_logits,
    constant uint& vocab_size,
    uint group_idx,
    uint batch_idx,
    uint thread_idx
) {
  uint bitmask_size = (vocab_size + (BITS_IN_U32 - 1)) / BITS_IN_U32;
  uint base_idx =
      batch_idx * vocab_size + group_idx * BLOCK_SIZE * GRAIN_SIZE + thread_idx;
  uint batch_end = batch_idx * vocab_size + vocab_size;

#pragma unroll(4)
  for (uint i = 0; i < GRAIN_SIZE; i++) {
    uint global_idx = base_idx + i * BLOCK_SIZE;
    if (global_idx < batch_end) {
      uint token_idx = global_idx - batch_idx * vocab_size;
      uint bitmask_idx = batch_idx * bitmask_size + (token_idx / BITS_IN_U32);
      bool mask = (bitmask_data[bitmask_idx] >>
                   (token_idx % BITS_IN_U32)) &
                  0b1;
      processed_logits[global_idx] =
          select(T(-INFINITY), logits_data[global_idx], mask);
    }
  }
}

#define outerArguments(T)                                                      \
  (device const T* logits_data [[buffer(0)]],                                  \
   device const uint32_t* bitmask_data [[buffer(1)]],                          \
   device T* processed_logits [[buffer(2)]],                                   \
   constant uint& vocab_size [[buffer(3)]],                                    \
   uint3 threadgroup_idx [[threadgroup_position_in_grid]],                     \
   uint3 thread_idx [[thread_position_in_threadgroup]])

#define innerArguments                                                         \
  (logits_data,                                                                \
   bitmask_data,                                                               \
   processed_logits,                                                           \
   vocab_size,                                                                 \
   threadgroup_idx.x,                                                          \
   threadgroup_idx.y,                                                          \
   thread_idx.x)

#define generateBitmaskKernel(functionName, scalarType, outerArgs, innerArgs)  \
  [[max_total_threads_per_threadgroup(                                         \
      1024                                                                     \
  )]] kernel void functionName##_##scalarType outerArgs {                      \
    functionName innerArgs;                                                    \
  }

#define generateBitmaskKernels(functionName)                                   \
  generateBitmaskKernel(                                                       \
      functionName,                                                            \
      float,                                                                   \
      outerArguments(float),                                                   \
      innerArguments                                                           \
  );                                                                           \
  generateBitmaskKernel(                                                       \
      functionName,                                                            \
      bfloat,                                                                  \
      outerArguments(bfloat),                                                  \
      innerArguments                                                           \
  );                                                                           \
  generateBitmaskKernel(                                                       \
      functionName,                                                            \
      half,                                                                    \
      outerArguments(half),                                                    \
      innerArguments                                                           \
  );

generateBitmaskKernels(batched_bitmask)

#undef outerArguments
#undef innerArguments
#undef generateBitmaskKernel
#undef generateBitmaskKernels
