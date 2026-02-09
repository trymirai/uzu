#include <metal_stdlib>
#include "../definitions.metal"

#define BLOCK_SIZE 1024
#define BLOCK_SIZE_IN_SIMDS (BLOCK_SIZE / 32)
#define MAX_ITERS 16

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(TopK) (
    device const T* logits,
    device T* processed_logits,
    threadgroup float shared_reduce_buffer[BLOCK_SIZE_IN_SIMDS],
    constant uint& batch_size,
    constant uint& vocab_size,
    constant uint& top_k,
    uint batch_idx GROUPS(batch_size),
    uint thread_idx THREADS(BLOCK_SIZE)
) {
  uint batch_start = batch_idx * vocab_size;

  // Find min and max logit for binary search
  float local_max = -INFINITY;
  float local_min = INFINITY;
#pragma unroll(4)
  for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
    float logit_value = float(logits[batch_start + i]);
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
  // Do the binary search on the threshold
  float low = min_logit;
  float high = max_logit;
  float threshold = (min_logit + max_logit) / 2.0;

  for (uint iter = 0; iter < MAX_ITERS; iter++) {
    // Find the mass above threshold
    uint local_num_above_threshold = 0;
#pragma unroll(4)
    for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
      float logit_value = float(logits[batch_start + i]);
      local_num_above_threshold += select(0, 1, logit_value >= threshold);
    }
    uint num_above_threshold = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(
        local_num_above_threshold,
        (threadgroup uint*)shared_reduce_buffer,
        thread_idx
    );

    // Update binary search
    if (num_above_threshold == top_k) {
      break;
    } else if (num_above_threshold > top_k) {
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
    T logit_value = logits[batch_start + i];
    processed_logits[batch_start + i] =
        select(T(-INFINITY), logit_value, logit_value >= t_threshold);
  }
}
