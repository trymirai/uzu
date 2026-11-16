#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;

#define BLOCK_SIZE 1024

UZU_CONST uint VALUES_PER_THREAD = 4;
static_assert(VALUES_PER_THREAD == HADAMARD_VECTOR_SIZE, "rotation operates on the same per-thread values");

template <typename T>
static METAL_FUNC vec<T, VALUES_PER_THREAD> load_values(const device T* source) {
  return *reinterpret_cast<const device packed_vec<T, VALUES_PER_THREAD>*>(source);
}

template <typename T, typename SourceT>
static METAL_FUNC void store_values(device T* destination, const vec<SourceT, VALUES_PER_THREAD> source) {
  *reinterpret_cast<device packed_vec<T, VALUES_PER_THREAD>*>(destination) = vec<T, VALUES_PER_THREAD>(source);
}

// TODO: Are numerics of subtract_mean fine?

template <typename InputT, typename AffineT, typename OutputT, typename AccumT>
VARIANTS(InputT, float, half, bfloat)
VARIANTS(AffineT, float, half, bfloat)
VARIANTS(OutputT, float, half, bfloat)
VARIANTS(AccumT, float)
PUBLIC KERNEL(Normalization)(
    const device InputT* input OPTIONAL(!in_place),
    const device AffineT* scales OPTIONAL(has_scales),
    const device AffineT* biases OPTIONAL(has_biases),
    device OutputT* output,
    device InputT* shortcut OPTIONAL(copy_to_shortcut),
    const device int32_t* hadamard_factors OPTIONAL(use_hadamard),
    constant uint& batch_size,
    constant uint& element_count,
    constant float& epsilon,
    constant float& scale_offset,
    constant uint& mixture_size,
    const bool in_place SPECIALIZE,
    const bool subtract_mean SPECIALIZE,
    const bool full_layer SPECIALIZE,
    const bool copy_to_shortcut SPECIALIZE,
    const bool residual_add SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    const bool has_biases SPECIALIZE,
    const bool has_scales SPECIALIZE,
    threadgroup AccumT shared_sum[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(BLOCK_SIZE)
) {
  if (in_place) {
    input = reinterpret_cast<const device InputT*>(output);
  }

  const uint batch_offset = batch_idx * element_count;
  const uint affine_offset = (batch_idx % mixture_size) * element_count;
  input += batch_offset;
  output += batch_offset;
  if (copy_to_shortcut) {
    shortcut += batch_offset;
  }

  // Step 1 - threads read from global and accumulate sum / sum of squares
  AccumT thread_sum = static_cast<AccumT>(0.0f);
  AccumT thread_sum_of_squares = static_cast<AccumT>(0.0f);

  for (uint first = thread_in_row * VALUES_PER_THREAD; first < element_count; first += BLOCK_SIZE * VALUES_PER_THREAD) {
    vec<InputT, VALUES_PER_THREAD> vals = load_values(input + first);
    // We can also fuse:
    // - TensorCopy (copy_to_shortcut)
    // - TensorAddSwap (copy_to_shortcut + residual_add)
    // Normalization in TensorAddSwap fusion mode operates on input + shortcut
    if (copy_to_shortcut) {
      if (residual_add) {
        vals += load_values(shortcut + first);
      }
      store_values(shortcut + first, vals);
    }
    METAL_PRAGMA_UNROLL
    for (uint k = 0; k < VALUES_PER_THREAD; ++k) {
      AccumT val_accum_t = static_cast<AccumT>(vals[k]);
      if (subtract_mean) {
        thread_sum += val_accum_t;
      }
      thread_sum_of_squares += val_accum_t * val_accum_t;
    }
  }

  // Step 2 - threads reduce their partial sums / sums of squares
  AccumT total_sum;
  if (subtract_mean) {
    total_sum =
        threadgroup_cooperative_reduce<SimdReduceSum<AccumT>, BLOCK_SIZE>(thread_sum, shared_sum, thread_context);
  }
  AccumT total_sum_of_squares = threadgroup_cooperative_reduce<SimdReduceSum<AccumT>, BLOCK_SIZE>(
      thread_sum_of_squares,
      shared_sum,
      thread_context
  );

  // And calculate mean/var/rms_inv from it
  AccumT mean = static_cast<AccumT>(0.0f);
  if (subtract_mean) {
    mean = total_sum / static_cast<AccumT>(element_count);
  }

  AccumT var = total_sum_of_squares / static_cast<AccumT>(element_count) - mean * mean;

  AccumT rms_inv = rsqrt(var + static_cast<AccumT>(epsilon));

  // Step 3 - elementwise normalization
  for (uint round_start = 0; round_start < element_count; round_start += BLOCK_SIZE * VALUES_PER_THREAD) {
    const uint first = round_start + thread_in_row * VALUES_PER_THREAD;
    const bool in_bounds = first < element_count;
    float4 values = 0.0f;
    vec<OutputT, VALUES_PER_THREAD> outs = 0;
    if (in_bounds) {
      // If we fuse TensorAddSwap, read shortcut (that now has input + shortcut)
      // No need for memory barrier because each thread only reads what it wrote
      vec<InputT, VALUES_PER_THREAD> xs = residual_add ? load_values(shortcut + first) : load_values(input + first);
      METAL_PRAGMA_UNROLL
      for (uint k = 0; k < VALUES_PER_THREAD; ++k) {
        const uint affine_index = affine_offset + first + k;
        const AccumT x = static_cast<AccumT>(xs[k]);
        // If full_layer, normalize and scale in AccumT, cast to OutputT at the end
        // If not, cast to OutputT after normalize, scale in OutputT
        OutputT val;
        if (has_scales) {
          AccumT scale = static_cast<AccumT>(scales[affine_index]) + static_cast<AccumT>(scale_offset);
          if (full_layer) {
            val = static_cast<OutputT>((x - mean) * rms_inv * scale);
          } else {
            val = static_cast<OutputT>((x - mean) * rms_inv) * static_cast<OutputT>(scale);
          }
        } else {
          val = static_cast<OutputT>((x - mean) * rms_inv);
        }

        if (has_biases) {
          val = static_cast<OutputT>(static_cast<AccumT>(val) + static_cast<AccumT>(biases[affine_index]));
        }
        values[k] = static_cast<float>(val);
        outs[k] = val;
      }
      if (use_hadamard) {
        values *= float4(load_hadamard_vector(hadamard_factors + first));
      }
    }
    if (use_hadamard) {
      values = simdgroup_hadamard_transform_vector(thread_context.simd_lane_id, values);
    }
    if (in_bounds) {
      if (use_hadamard) {
        store_hadamard_vector(output + first, values);
      } else {
        store_values(output + first, outs);
      }
    }
  }
}
