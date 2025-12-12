#include <metal_stdlib>
#include "../definitions.metal"
#include "../rng.metal"

#define GRAIN_SIZE 64
#define BLOCK_SIZE 1024

#define WORDS_PER_OFFSET 4

template <typename T>
void batched_gumbel(
    device const T* logits_data,
    device const uint64_t* batch_seeds,
    device T* processed_logits,
    constant uint& vocab_size,
    uint group_idx,
    uint batch_idx,
    uint thread_idx
) {
  uint batch_start = vocab_size * batch_idx;
  uint batch_end = batch_start + vocab_size;

  uint grain_offset_in_batch = group_idx * BLOCK_SIZE * GRAIN_SIZE + thread_idx;

  uint grain_offset = batch_start + grain_offset_in_batch;

  uint64_t rng_seed = batch_seeds[batch_idx];
  uint64_t rng_offset = (group_idx * BLOCK_SIZE + thread_idx) *
                        (GRAIN_SIZE + WORDS_PER_OFFSET - 1) / WORDS_PER_OFFSET;
  PhiloxState rng;
  philox_init(&rng, rng_seed, rng_offset);

#pragma unroll(4)
  for (uint i = 0; i < GRAIN_SIZE; i++) {
    uint global_idx = grain_offset + i * BLOCK_SIZE;
    if (global_idx < batch_end) {
      processed_logits[global_idx] =
          logits_data[global_idx] +
          T(-fast::log(-fast::log(uniform_float(&rng))));
    }
  }
}

#define outerArguments(T)                                                      \
  (device const T* logits_data [[buffer(0)]],                                  \
   device const uint64_t* batch_seeds [[buffer(1)]],                           \
   device T* processed_logits [[buffer(2)]],                                   \
   constant uint& vocab_size [[buffer(3)]],                                    \
   uint3 threadgroup_idx [[threadgroup_position_in_grid]],                     \
   uint3 thread_idx [[thread_position_in_threadgroup]])

#define innerArguments                                                         \
  (logits_data,                                                                \
   batch_seeds,                                                                \
   processed_logits,                                                           \
   vocab_size,                                                                 \
   threadgroup_idx.x,                                                          \
   threadgroup_idx.y,                                                          \
   thread_idx.x)

#define generateGumbelKernel(functionName, scalarType, outerArgs, innerArgs)   \
  [[max_total_threads_per_threadgroup(                                         \
      1024                                                                     \
  )]] kernel void functionName##_##scalarType outerArgs {                      \
    functionName innerArgs;                                                    \
  }

#define generateGumbelKernels(functionName)                                    \
  generateGumbelKernel(                                                        \
      functionName,                                                            \
      float,                                                                   \
      outerArguments(float),                                                   \
      innerArguments                                                           \
  );                                                                           \
  generateGumbelKernel(                                                        \
      functionName,                                                            \
      bfloat,                                                                  \
      outerArguments(bfloat),                                                  \
      innerArguments                                                           \
  );                                                                           \
  generateGumbelKernel(                                                        \
      functionName,                                                            \
      half,                                                                    \
      outerArguments(half),                                                    \
      innerArguments                                                           \
  );

generateGumbelKernels(batched_gumbel)

#undef outerArguments
#undef innerArguments
#undef generateGumbelKernel
#undef generateGumbelKernels
