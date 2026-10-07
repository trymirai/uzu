#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;

#define BLOCK_SIZE 1024

// a - b evaluated as written: fast math would otherwise reassociate (x - pivot) - mean_delta into
// x - (pivot + mean_delta), rounding the shift at the row's magnitude.
template <typename T>
static METAL_FUNC T normalization_difference(T a, T b) {
#pragma clang fp reassociate(off)
  return a - b;
}

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
    constant float& post_layer_scalar,
    const bool in_place SPECIALIZE,
    const bool subtract_mean SPECIALIZE,
    const bool full_layer SPECIALIZE,
    const bool copy_to_shortcut SPECIALIZE,
    const bool residual_add SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    const bool scale_residual_sum SPECIALIZE,
    const bool scale_output SPECIALIZE,
    const bool has_biases SPECIALIZE,
    const bool has_scales SPECIALIZE,
    threadgroup AccumT shared_sum[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(BLOCK_SIZE)
) {
  // An empty row has no pivot and nothing to write; every thread returns before any barrier.
  if (element_count == 0) {
    return;
  }
  if (in_place) {
    input = reinterpret_cast<const device InputT*>(output);
  }

  const uint batch_offset = batch_idx * element_count;
  input += batch_offset;
  output += batch_offset;
  if (copy_to_shortcut) {
    shortcut += batch_offset;
  }

  // Step 1 - threads fuse the residual into the shortcut and accumulate the sum of squares of the RMS path
  AccumT thread_sum_of_squares = static_cast<AccumT>(0.0f);

  for (uint i = thread_in_row; i < element_count && (copy_to_shortcut || !subtract_mean); i += BLOCK_SIZE) {
    InputT val = input[i];
    // We can also fuse:
    // - TensorCopy (copy_to_shortcut)
    // - TensorAddSwap (copy_to_shortcut + residual_add)
    // Normalization in TensorAddSwap fusion mode operates on input + shortcut
    if (copy_to_shortcut) {
      if (residual_add) {
        val += shortcut[i];
        if (scale_residual_sum) {
          val = static_cast<InputT>(static_cast<float>(val) * post_layer_scalar);
        }
      }
      shortcut[i] = val;
    }
    if (!subtract_mean) {
      AccumT val_accum_t = static_cast<AccumT>(val);
      thread_sum_of_squares += val_accum_t * val_accum_t;
    }
  }

  // Step 2 - inverse RMS over the row. Each thread reads back only the elements it stored in step 1. With
  // subtract_mean, deltas from the first stored element are exact for near-constant rows, and squared deviations from
  // their mean avoid the cancellation of E[x^2] - mean^2.
  AccumT pivot = static_cast<AccumT>(0.0f);
  AccumT mean_delta = static_cast<AccumT>(0.0f);
  if (subtract_mean) {
    if (thread_in_row == 0 && element_count > 0) {
      shared_sum[0] = static_cast<AccumT>(residual_add ? shortcut[0] : input[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pivot = shared_sum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    AccumT thread_delta_sum = static_cast<AccumT>(0.0f);
    for (uint i = thread_in_row; i < element_count; i += BLOCK_SIZE) {
      thread_delta_sum += normalization_difference(static_cast<AccumT>(residual_add ? shortcut[i] : input[i]), pivot);
    }
    mean_delta =
        threadgroup_cooperative_reduce<SimdReduceSum<AccumT>, BLOCK_SIZE>(thread_delta_sum, shared_sum, thread_context) /
        static_cast<AccumT>(element_count);
    for (uint i = thread_in_row; i < element_count; i += BLOCK_SIZE) {
      AccumT deviation = normalization_difference(
          normalization_difference(static_cast<AccumT>(residual_add ? shortcut[i] : input[i]), pivot),
          mean_delta
      );
      thread_sum_of_squares += deviation * deviation;
    }
  }
  AccumT total_sum_of_squares = threadgroup_cooperative_reduce<SimdReduceSum<AccumT>, BLOCK_SIZE>(
      thread_sum_of_squares,
      shared_sum,
      thread_context
  );

  AccumT rms_inv =
      rsqrt(total_sum_of_squares / static_cast<AccumT>(element_count) + static_cast<AccumT>(epsilon));

  // Step 3 - elementwise normalization
  for (uint i = thread_in_row; i < element_count; i += BLOCK_SIZE) {
    AccumT x;
    // If we fuse TensorAddSwap, read shortcut (that now has input + shortcut)
    // No need for memory barrier because each thread only reads what it wrote
    if (residual_add) {
      x = static_cast<AccumT>(shortcut[i]);
    } else {
      x = static_cast<AccumT>(input[i]);
    }

    AccumT normalized = normalization_difference(normalization_difference(x, pivot), mean_delta) * rms_inv;

    // If full_layer, normalize and scale in AccumT, cast to OutputT at the end
    // If not, cast to OutputT after normalize, scale in OutputT
    OutputT val;
    if (has_scales) {
      AccumT scale = static_cast<AccumT>(scales[i]) + static_cast<AccumT>(scale_offset);
      if (full_layer) {
        val = static_cast<OutputT>(normalized * scale);
      } else {
        val = static_cast<OutputT>(normalized) * static_cast<OutputT>(scale);
      }
    } else {
      val = static_cast<OutputT>(normalized);
    }

    if (has_biases) {
      val = static_cast<OutputT>(static_cast<AccumT>(val) + static_cast<AccumT>(biases[i]));
    }

    if (use_hadamard) {
      val = static_cast<OutputT>(simdgroup_input_random_hadamard_transform(
          static_cast<ushort>(thread_in_row % METAL_SIMD_SIZE),
          val,
          hadamard_factors[i]
      ));
    }

    if (scale_output) {
      val = static_cast<OutputT>(static_cast<float>(val) * post_layer_scalar);
    }

    output[i] = val;
  }
}
