#include <metal_stdlib>
#include <metal_atomic>
#include "../definitions.metal"

#define BLOCK_SIZE 1024
#define GRAIN_SIZE 4

struct ArgmaxPair {
  float value;
  uint index;
};

constant ArgmaxPair ARGMAX_INIT = {-INFINITY, UINT_MAX};

bool argmax_is_better(ArgmaxPair a, ArgmaxPair b) {
  return a.value > b.value || (a.value == b.value && a.index < b.index);
}

ArgmaxPair argmax_combine(ArgmaxPair a, ArgmaxPair b) {
  return argmax_is_better(a, b) ? a : b;
}

//------------------------------------------------------------------------------------------------//
//  Cooperative threadgroup argmax reduction (2-level hierarchical)
template <ushort BLOCK_SIZE_PARAM>
static ArgmaxPair threadgroup_cooperative_argmax(
    ArgmaxPair value,
    threadgroup ArgmaxPair* shared,
    const ushort lid
) {
  const ushort simd_group_id = lid / 32;
  const ushort simd_lane_id = lid % 32;

  // Reduce within simdgroup using manual shuffle operations
  ArgmaxPair local_result = value;
  for (ushort offset = 16; offset > 0; offset /= 2) {
    ArgmaxPair other = {
        simd_shuffle_down(local_result.value, offset),
        simd_shuffle_down(local_result.index, offset)
    };
    local_result = argmax_combine(local_result, other);
  }

  // First thread in each simdgroup writes to shared memory
  if (simd_lane_id == 0) {
    shared[simd_group_id] = local_result;
  }

  // Synchronize across the threadgroup
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduce across simdgroups
  const ushort num_simd_groups = (BLOCK_SIZE_PARAM + 31) / 32;
  ArgmaxPair total_result = ARGMAX_INIT;
  if (lid < num_simd_groups) {
    total_result = shared[lid];
  }

  // Final simdgroup reduction
  for (ushort offset = 16; offset > 0; offset /= 2) {
    ArgmaxPair other = {
        simd_shuffle_down(total_result.value, offset),
        simd_shuffle_down(total_result.index, offset)
    };
    total_result = argmax_combine(total_result, other);
  }

  // Broadcast the result to all threads
  if (lid == 0) {
    shared[0] = total_result;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const ArgmaxPair result = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

template <ushort GRAIN_SIZE_PARAM>
static inline ArgmaxPair thread_argmax_reduce(
    thread ArgmaxPair (&values)[GRAIN_SIZE_PARAM]
) {
  ArgmaxPair result = values[0];
  for (ushort i = 1; i < GRAIN_SIZE_PARAM; i++) {
    result = argmax_combine(result, values[i]);
  }
  return result;
}

template <ushort GRAIN_SIZE_PARAM, typename T>
static void load_blocked_argmax_batched(
    thread ArgmaxPair (&values)[GRAIN_SIZE_PARAM],
    const device T* logits_data,
    const ushort local_id,
    const uint batch_idx,
    const uint vocab_offset,
    const uint vocab_size
) {
  for (ushort i = 0; i < GRAIN_SIZE_PARAM; i++) {
    uint vocab_idx = vocab_offset + local_id * GRAIN_SIZE_PARAM + i;
    if (vocab_idx < vocab_size) {
      uint global_idx = batch_idx * vocab_size + vocab_idx;
      float value;
      value = static_cast<float>(logits_data[global_idx]);
      values[i] = {value, vocab_idx};
    } else {
      values[i] = ARGMAX_INIT;
    }
  }
}

template <ushort BLOCK_SIZE_PARAM>
static ArgmaxPair threadgroup_raking_argmax(
    ArgmaxPair value,
    threadgroup ArgmaxPair* shared,
    const ushort lid
) {
  shared[lid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid < 32) {
    const ushort values_per_thread = BLOCK_SIZE_PARAM / 32;
    const ushort first_index = lid * values_per_thread;

    ArgmaxPair thread_result = shared[first_index];
    for (ushort i = first_index + 1; i < first_index + values_per_thread; i++) {
      thread_result = argmax_combine(thread_result, shared[i]);
    }

    ArgmaxPair simd_result = thread_result;

    for (ushort offset = 16; offset > 0; offset /= 2) {
      ArgmaxPair other = {
          simd_shuffle_down(simd_result.value, offset),
          simd_shuffle_down(simd_result.index, offset)
      };
      simd_result = argmax_combine(simd_result, other);
    }

    ArgmaxPair final_result = {
        simd_broadcast(simd_result.value, 0),
        simd_broadcast(simd_result.index, 0)
    };

    for (ushort i = first_index; i < first_index + values_per_thread; i++) {
      shared[i] = final_result;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ArgmaxPair result = shared[lid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

// Single-pass argmax

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(ArgmaxSingle)(
  const device T* logits_data,
  device uint* final_tokens,
  constant uint& batch_size,
  constant uint& vocab_size,
  threadgroup ArgmaxPair shared[BLOCK_SIZE],
  uint batch_idx GROUPS(batch_size),
  ushort local_id THREADS(BLOCK_SIZE)
) {
  if (batch_idx >= batch_size)
    return;

  const uint elements_per_tile = BLOCK_SIZE * GRAIN_SIZE;
  ArgmaxPair best = ARGMAX_INIT;

  for (uint vocab_offset = 0; vocab_offset < vocab_size;
       vocab_offset += elements_per_tile) {
    ArgmaxPair buf[GRAIN_SIZE];
    load_blocked_argmax_batched<GRAIN_SIZE>(
        buf,
        logits_data,
        local_id,
        batch_idx,
        vocab_offset,
        vocab_size
    );
    ArgmaxPair tile_best = thread_argmax_reduce<GRAIN_SIZE>(buf);
    best = argmax_combine(best, tile_best);
  }

  ArgmaxPair group_best =
      threadgroup_raking_argmax<BLOCK_SIZE>(best, shared, local_id);

  if (local_id == 0) {
    final_tokens[batch_idx] = group_best.index;
  }
}

// Two-pass argmax

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(ArgmaxMain)(
    const device T* logits_data,
    device ArgmaxPair* partial_results,
    constant uint& batch_size,
    constant uint& vocab_size,
    threadgroup ArgmaxPair shared[BLOCK_SIZE],
    uint batch_idx GROUPS(batch_size),
    uint vocab_group_idx GROUPS(vocab_size.div_ceil(BLOCK_SIZE * GRAIN_SIZE)),
    ushort local_id THREADS(BLOCK_SIZE)
) {

  if (batch_idx >= batch_size)
    return;

  const uint elements_per_group = BLOCK_SIZE * GRAIN_SIZE;
  const uint vocab_offset = vocab_group_idx * elements_per_group;

  ArgmaxPair values[GRAIN_SIZE];
  load_blocked_argmax_batched<GRAIN_SIZE>(
      values,
      logits_data,
      local_id,
      batch_idx,
      vocab_offset,
      vocab_size
  );

  ArgmaxPair thread_result = thread_argmax_reduce<GRAIN_SIZE>(values);
  ArgmaxPair group_result = threadgroup_cooperative_argmax<BLOCK_SIZE>(
      thread_result,
      shared,
      local_id
  );

  if (local_id == 0) {
    uint vocab_groups_per_batch =
        (vocab_size + elements_per_group - 1) / elements_per_group;
    uint partial_idx = batch_idx * vocab_groups_per_batch + vocab_group_idx;
    partial_results[partial_idx] = group_result;
  }
}

KERNEL(ArgmaxFinal)(
    const device ArgmaxPair* partial_results,
    device uint* final_tokens,
    constant uint& batch_size,
    constant uint& vocab_size,
    threadgroup ArgmaxPair shared[BLOCK_SIZE],
    uint batch_idx GROUPS(batch_size),
    ushort local_id THREADS(BLOCK_SIZE)
) {

  if (batch_idx >= batch_size)
    return;

  const uint elements_per_group = BLOCK_SIZE * GRAIN_SIZE;
  const uint vocab_groups_per_batch =
      (vocab_size + elements_per_group - 1) / elements_per_group;
  const uint partial_offset = batch_idx * vocab_groups_per_batch;

  ArgmaxPair thread_result = ARGMAX_INIT;

  for (uint i = local_id; i < vocab_groups_per_batch; i += BLOCK_SIZE) {
    thread_result =
        argmax_combine(thread_result, partial_results[partial_offset + i]);
  }

  ArgmaxPair result = threadgroup_cooperative_argmax<BLOCK_SIZE>(
      thread_result,
      shared,
      local_id
  );

  if (local_id == 0) {
    final_tokens[batch_idx] = result.index;
  }
}
