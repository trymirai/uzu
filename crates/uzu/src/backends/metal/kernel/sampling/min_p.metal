#include <metal_stdlib>
#include "../definitions.metal"

#define BLOCK_SIZE 1024
#define BLOCK_SIZE_IN_SIMDS (BLOCK_SIZE / 32)

template <typename T>
void batched_minp(
    device const T* logits_data,
    device T* processed_logits,
    threadgroup float shared_reduce_buffer[BLOCK_SIZE_IN_SIMDS],
    constant uint& vocab_size,
    constant float& min_p,
    uint batch_idx,
    uint thread_idx
) {
  uint batch_start = batch_idx * vocab_size;

  // Find maximum logit
  float local_max = -INFINITY;
#pragma unroll(4)
  for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
    float logit_value = float(logits_data[batch_start + i]);
    local_max = fmax(local_max, logit_value);
  }
  float max_logit = threadgroup_cooperative_reduce_max<BLOCK_SIZE>(
      local_max,
      shared_reduce_buffer,
      thread_idx
  );

  // Then the threshold is just max_logit + log(min_p), mask everything strictly below it
  T t_threshold = T(max_logit + log(min_p));
#pragma unroll(4)
  for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
    T logit_value = logits_data[batch_start + i];
    processed_logits[batch_start + i] =
        select(T(-INFINITY), logit_value, logit_value >= t_threshold);
  }
}

#define outerArguments(T)                                                      \
  (device const T* logits_data [[buffer(0)]],                                  \
   device T* processed_logits [[buffer(1)]],                                   \
   constant uint& vocab_size [[buffer(2)]],                                    \
   constant float& min_p [[buffer(3)]],                                        \
   uint3 threadgroup_idx [[threadgroup_position_in_grid]],                     \
   uint3 thread_idx [[thread_position_in_threadgroup]])

#define innerArguments                                                         \
  (logits_data,                                                                \
   processed_logits,                                                           \
   shared_reduce_buffer,                                                       \
   vocab_size,                                                                 \
   min_p,                                                                      \
   threadgroup_idx.x,                                                          \
   thread_idx.x)

#define generateMinpKernel(functionName, scalarType, outerArgs, innerArgs)     \
  [[max_total_threads_per_threadgroup(                                         \
      1024                                                                     \
  )]] kernel void functionName##_##scalarType outerArgs {                      \
    threadgroup float shared_reduce_buffer[BLOCK_SIZE_IN_SIMDS];               \
    functionName innerArgs;                                                    \
  }

#define generateMinpKernels(functionName)                                      \
  generateMinpKernel(                                                          \
      functionName,                                                            \
      float,                                                                   \
      outerArguments(float),                                                   \
      innerArguments                                                           \
  );                                                                           \
  generateMinpKernel(                                                          \
      functionName,                                                            \
      bfloat,                                                                  \
      outerArguments(bfloat),                                                  \
      innerArguments                                                           \
  );                                                                           \
  generateMinpKernel(functionName, half, outerArguments(half), innerArguments);

generateMinpKernels(batched_minp)

#undef outerArguments
#undef innerArguments
#undef generateMinpKernel
#undef generateMinpKernels
