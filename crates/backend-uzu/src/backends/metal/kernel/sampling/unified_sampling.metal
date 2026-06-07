#include "../common/threadgroup_reduce.h"
#include "../common/thread_context.h"
#include "../common/defines.h"
#include "../common/dsl.h"

#include "../rng.h"

#include "../generated/argmax.h"

using namespace uzu::argmax;

#define THREADGROUP_SIZE 1024
#define THREADGROUP_SIZE_IN_SIMDS (THREADGROUP_SIZE / METAL_SIMD_SIZE)
#define BITS_IN_U32 32

#define WORDS_PER_OFFSET 4

#define MAX_ITERS 64

struct Logit {
  float value;
  uint32_t index;

  static const constant Logit LOWEST;

  template <typename T>
  static inline Logit load(const device T* logits, uint32_t index) {
    return {.value = float(logits[index]), .index = index};
  }

  inline bool operator>(Logit rhs) const { return value > rhs.value || (value == rhs.value && index < rhs.index); }
};

constexpr constant Logit
    Logit::LOWEST{.value = -numeric_limits<float>::infinity(), .index = numeric_limits<uint32_t>::max()};

struct SimdReduceMaxLogit {
  using value_type = Logit;
  static constant constexpr Logit identity = Logit::LOWEST;

  static Logit simd_reduce(Logit x) {
    METAL_PRAGMA_UNROLL
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
      Logit y = {
          .value = simd_shuffle_xor(x.value, offset),
          .index = simd_shuffle_xor(x.index, offset),
      };
      x = y > x ? y : x;
    }
    return x;
  }
};

// NOTE: top_k + top_p combination is not exactly matching lalamo ("parallel" here, should be top-k then top-p)
template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(UnifiedSampling) (
  const device T* logits,
  device uint32_t* output,
  const device uint64_t* seeds OPTIONAL(is_stochastic),
  const device uint32_t* bitmask OPTIONAL(has_bitmask),
  const constant float& temperature OPTIONAL(has_temperature),
  const constant uint32_t& top_k OPTIONAL(has_top_k),
  const constant float& top_p OPTIONAL(has_top_p),
  const constant float& min_p OPTIONAL(has_min_p),
  const constant uint32_t& vocab_size,
  const constant uint32_t& batch_size,
  const bool is_stochastic SPECIALIZE,
  const bool has_bitmask SPECIALIZE,
  const bool has_temperature SPECIALIZE,
  const bool temperature_after_filters SPECIALIZE,
  const bool has_top_k SPECIALIZE,
  const bool has_top_p SPECIALIZE,
  const bool has_min_p SPECIALIZE,
  threadgroup Logit shared[THREADGROUP_SIZE_IN_SIMDS],
  const ThreadContext thread_context,
  uint batch_idx GROUPS(batch_size),
  uint thread_idx THREADS(THREADGROUP_SIZE)
) {
  logits += vocab_size * batch_idx;
  output += batch_idx;
  uint64_t rng_seed, rng_offset;
  PhiloxState rng;
  if (is_stochastic) {
    rng_seed = seeds[batch_idx];
    rng_offset = div_ceil(vocab_size, THREADGROUP_SIZE * WORDS_PER_OFFSET) * thread_idx;
    philox_init(&rng, rng_seed, rng_offset);
  }
  if (has_bitmask) {
    bitmask += div_ceil(vocab_size, BITS_IN_U32) * batch_idx;
  }
  float recip_temperature;
  if (has_temperature) {
    recip_temperature = 1.0 / temperature;
  }
  float log_min_p;
  if (has_min_p) {
    log_min_p = log(min_p);
  }

  float thread_pre_filter_logit_max = -INFINITY;
  float thread_pre_filter_logit_norm = 0.0;
  Logit thread_post_gumbel_logit_max = Logit::LOWEST;
  for (uint32_t logit_index = thread_idx; logit_index < vocab_size; logit_index += THREADGROUP_SIZE) {
    Logit logit = Logit::load(logits, logit_index);

    if (has_bitmask) {
      bool mask = (bitmask[logit_index / BITS_IN_U32] >> (logit_index % BITS_IN_U32)) & 0b1;
      logit.value = mask ? logit.value : -INFINITY;
    }

    if (has_temperature && !temperature_after_filters) {
      logit.value *= recip_temperature;
    }

    if (has_top_p && logit.value != -INFINITY) {
      float new_thread_pre_filter_logit_max = max(thread_pre_filter_logit_max, logit.value);

      thread_pre_filter_logit_norm =
          thread_pre_filter_logit_norm * exp(thread_pre_filter_logit_max - new_thread_pre_filter_logit_max) +
          exp(logit.value - new_thread_pre_filter_logit_max);

      thread_pre_filter_logit_max = new_thread_pre_filter_logit_max;
    } else if (has_min_p) {
      thread_pre_filter_logit_max = max(thread_pre_filter_logit_max, logit.value);
    }

    if (has_temperature && temperature_after_filters) {
      logit.value *= recip_temperature;
    }

    if (is_stochastic) {
      logit.value += -log(-log(uniform_float(&rng)));
    }

    thread_post_gumbel_logit_max = logit > thread_post_gumbel_logit_max ? logit : thread_post_gumbel_logit_max;
  }
  float pre_filter_logit_max;
  if (has_top_p || has_min_p) {
    pre_filter_logit_max = threadgroup_cooperative_reduce<SimdReduceMax<float>, THREADGROUP_SIZE>(
        thread_pre_filter_logit_max,
        (threadgroup float*)shared,
        thread_context
    );
  }
  float pre_filter_logit_norm;
  if (has_top_p) {
    if (thread_pre_filter_logit_norm != 0.0) {
      thread_pre_filter_logit_norm *= exp(thread_pre_filter_logit_max - pre_filter_logit_max);
    }
    pre_filter_logit_norm = threadgroup_cooperative_reduce<SimdReduceSum<float>, THREADGROUP_SIZE>(
        thread_pre_filter_logit_norm,
        (threadgroup float*)shared,
        thread_context
    );
  }
  Logit candidate_logit_post_gumbel = threadgroup_cooperative_reduce<SimdReduceMaxLogit, THREADGROUP_SIZE>(
      thread_post_gumbel_logit_max,
      shared,
      thread_context
  );

  for (uint32_t iteration = 0; iteration < MAX_ITERS; iteration++) {
    Logit candidate_logit_pre_filter = Logit::load(logits, candidate_logit_post_gumbel.index);
    if (has_temperature && !temperature_after_filters) {
      candidate_logit_pre_filter.value *= recip_temperature;
    }
    if (is_stochastic) {
      philox_init(&rng, rng_seed, rng_offset);
    }

    uint32_t thread_num_above_candidate = 0;
    float thread_mass_above_candidate = 0.0;
    Logit thread_next_candidate_logit_post_gumbel = Logit::LOWEST;
    for (uint32_t logit_index = thread_idx; logit_index < vocab_size; logit_index += THREADGROUP_SIZE) {
      Logit logit = Logit::load(logits, logit_index);

      if (has_bitmask) {
        bool mask = (bitmask[logit_index / BITS_IN_U32] >> (logit_index % BITS_IN_U32)) & 0b1;
        logit.value = mask ? logit.value : -INFINITY;
      }

      if (has_temperature && !temperature_after_filters) {
        logit.value *= recip_temperature;
      }

      bool above_current_pre_filter = logit > candidate_logit_pre_filter;

      if (above_current_pre_filter) {
        if (has_top_k) {
          thread_num_above_candidate += 1;
        }
        if (has_top_p) {
          thread_mass_above_candidate += exp(logit.value - pre_filter_logit_max) / pre_filter_logit_norm;
        }
      }

      if (has_temperature && temperature_after_filters) {
        logit.value *= recip_temperature;
      }

      if (is_stochastic) {
        logit.value += -log(-log(uniform_float(&rng)));
      }

      if (above_current_pre_filter) {
        thread_next_candidate_logit_post_gumbel =
            logit > thread_next_candidate_logit_post_gumbel ? logit : thread_next_candidate_logit_post_gumbel;
      }
    }
    uint32_t num_above_candidate;
    if (has_top_k) {
      num_above_candidate = threadgroup_cooperative_reduce<SimdReduceSum<uint32_t>, THREADGROUP_SIZE>(
          thread_num_above_candidate,
          (threadgroup uint32_t*)shared,
          thread_context
      );
    }
    float mass_above_candidate;
    if (has_top_p) {
      mass_above_candidate = threadgroup_cooperative_reduce<SimdReduceSum<float>, THREADGROUP_SIZE>(
          thread_mass_above_candidate,
          (threadgroup float*)shared,
          thread_context
      );
    }
    Logit next_candidate_logit_post_gumbel = threadgroup_cooperative_reduce<SimdReduceMaxLogit, THREADGROUP_SIZE>(
        thread_next_candidate_logit_post_gumbel,
        shared,
        thread_context
    );

    bool filters_passed = true;

    if (has_top_k && num_above_candidate >= top_k) {
      filters_passed = false;
    }
    if (has_top_p && mass_above_candidate >= top_p) {
      filters_passed = false;
    }
    if (has_min_p && candidate_logit_pre_filter.value < pre_filter_logit_max + log_min_p) {
      filters_passed = false;
    }

    if (filters_passed || iteration == MAX_ITERS - 1) {
      *output = candidate_logit_post_gumbel.index;
      return;
    }

    candidate_logit_post_gumbel = next_candidate_logit_post_gumbel;
  }
}
