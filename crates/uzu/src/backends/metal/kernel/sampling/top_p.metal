#include <metal_stdlib>
#include "../definitions.metal"

#define BLOCK_SIZE 1024
#define BLOCK_SIZE_IN_SIMDS (BLOCK_SIZE / 32)

#define MAX_ITERS 16

template <typename T>
void batched_topp(
    device const T* logits_data,
    device T* processed_logits,
    threadgroup float shared_reduce_buffer[BLOCK_SIZE_IN_SIMDS],
    constant uint& vocab_size,
    constant float& top_p,
    uint batch_idx,
    uint thread_idx
) {
  uint batch_start = batch_idx * vocab_size;

  // Find min (for binary search) and max (for binary search and softmax)
  float local_max = -INFINITY;
  float local_min = INFINITY;
#pragma unroll(4)
  for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
    float logit_value = float(logits_data[batch_start + i]);
    local_max = fmax(local_max, logit_value);
    local_min = select(
        local_min,
        fmin(local_min, logit_value),
        logit_value > -INFINITY
    );
  }
  float max_logit = threadgroup_cooperative_reduce_max<BLOCK_SIZE>(
      local_max,
      shared_reduce_buffer,
      thread_idx
  );
  float min_logit = threadgroup_cooperative_reduce_min<BLOCK_SIZE>(
      local_min,
      shared_reduce_buffer,
      thread_idx
  );

  // Find denominator for softmax
  float local_sum = 0.0f;
#pragma unroll(4)
  for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
    float logit_value = float(logits_data[batch_start + i]);
    local_sum += select(
        0.0,
        fast::exp(logit_value - max_logit),
        logit_value > -INFINITY
    );
  }
  float total_sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(
      local_sum,
      shared_reduce_buffer,
      thread_idx
  );

  // Do the binary search on the threshold
  float target_mass = top_p * total_sum;

  float low = min_logit;
  float high = max_logit;
  float threshold = (min_logit + max_logit) / 2.0;

  for (uint iter = 0; iter < MAX_ITERS; iter++) {
    // Find the mass above threshold
    float local_sum_above_threshold = 0.0f;
    float local_min_above_threshold = INFINITY;
#pragma unroll(4)
    for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
      float logit_value = float(logits_data[batch_start + i]);
      float logit_mass = fast::exp(logit_value - max_logit);
      local_sum_above_threshold +=
          select(0.0, logit_mass, logit_value >= threshold);
      local_min_above_threshold = fmin(
          local_min_above_threshold,
          select(INFINITY, logit_mass, logit_value >= threshold)
      );
    }
    float sum_above_threshold = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(
        local_sum_above_threshold,
        shared_reduce_buffer,
        thread_idx
    );
    float min_above_threshold = threadgroup_cooperative_reduce_min<BLOCK_SIZE>(
        local_min_above_threshold,
        shared_reduce_buffer,
        thread_idx
    );

    // Early exit
    if (sum_above_threshold >= target_mass &&
        sum_above_threshold - min_above_threshold < target_mass) {
      break;
    }
    // Update binary search
    if (sum_above_threshold >= target_mass) {
      low = threshold;
    } else {
      high = threshold;
    }
    threshold = (low + high) / 2.0;
  }

  T t_threshold = T(threshold);

// We know the threshold, just mask everything below it
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
   constant float& top_p [[buffer(3)]],                                        \
   uint3 threadgroup_idx [[threadgroup_position_in_grid]],                     \
   uint3 thread_idx [[thread_position_in_threadgroup]])

#define innerArguments                                                         \
  (logits_data,                                                                \
   processed_logits,                                                           \
   shared_reduce_buffer,                                                       \
   vocab_size,                                                                 \
   top_p,                                                                      \
   threadgroup_idx.x,                                                          \
   thread_idx.x)

#define generateToppKernel(functionName, scalarType, outerArgs, innerArgs)     \
  [[max_total_threads_per_threadgroup(                                         \
      1024                                                                     \
  )]] kernel void functionName##_##scalarType outerArgs {                      \
    threadgroup float shared_reduce_buffer[BLOCK_SIZE_IN_SIMDS];               \
    functionName innerArgs;                                                    \
  }

#define generateToppKernels(functionName)                                      \
  generateToppKernel(                                                          \
      functionName,                                                            \
      float,                                                                   \
      outerArguments(float),                                                   \
      innerArguments                                                           \
  );                                                                           \
  generateToppKernel(                                                          \
      functionName,                                                            \
      bfloat,                                                                  \
      outerArguments(bfloat),                                                  \
      innerArguments                                                           \
  );                                                                           \
  generateToppKernel(functionName, half, outerArguments(half), innerArguments);

generateToppKernels(batched_topp)

#undef outerArguments
#undef innerArguments
#undef generateToppKernel
#undef generateToppKernels
