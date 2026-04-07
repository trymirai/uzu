#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"

using namespace metal;

#define BLOCK_SIZE 1024

template <typename InputT, typename ScaleT, typename OutputT, typename AccumT>
VARIANTS(InputT, float, half, bfloat)
VARIANTS(ScaleT, float, half, bfloat)
VARIANTS(OutputT, float, half, bfloat)
VARIANTS(AccumT, float, half)
PUBLIC KERNEL(RMSNorm)(
    const device InputT* input OPTIONAL(!in_place),
    const device ScaleT* scales,
    device OutputT* output,
    device InputT* shortcut OPTIONAL(copy_to_shortcut),
    constant uint& batch_size,
    constant uint& element_count,
    constant float& epsilon,
    constant float& scale_offset,
    const bool in_place SPECIALIZE,
    const bool full_layer SPECIALIZE,
    const bool copy_to_shortcut SPECIALIZE,
    const bool residual_add SPECIALIZE,
    threadgroup AccumT shared_sum[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(1024)
) {
  if (in_place) {
    input = reinterpret_cast<const device InputT*>(output);
  }

  const uint batch_offset = batch_idx * element_count;
  input += batch_offset;
  output += batch_offset;
  if (copy_to_shortcut) {
    shortcut += batch_offset;
  }

  // Step 1 - threads read from global and accumulate sum of squares
  AccumT thread_sum_of_squares = static_cast<AccumT>(0.0f);

  for (uint i = thread_in_row; i < element_count; i += BLOCK_SIZE) {
    InputT val = input[i];
    // We can also fuse:
    // - TensorCopy (copy_to_shortcut)
    // - TensorAddSwap (copy_to_shortcut + residual_add)
    // RMSNorm in TensorAddSwap fusion mode operates on input + shortcut
    if (copy_to_shortcut) {
      if (residual_add) {
        val += shortcut[i];
      }
      shortcut[i] = val;
    }
    AccumT val_accum_t = static_cast<AccumT>(val);
    thread_sum_of_squares += val_accum_t * val_accum_t;
  }

  // Step 2 - threads reduce their partial sums of squares
  AccumT total_sum_of_squares =
      threadgroup_cooperative_reduce<SimdReduceSum<AccumT>, BLOCK_SIZE>(
          thread_sum_of_squares,
          shared_sum,
          thread_context
      );

  // And pre-calculate rms_inv
  AccumT rms_inv = rsqrt(
      total_sum_of_squares / static_cast<AccumT>(element_count) +
      static_cast<AccumT>(epsilon)
  );

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

    AccumT scale =
        static_cast<AccumT>(scales[i]) + static_cast<AccumT>(scale_offset);

    // If full_layer, normalize and scale in AccumT, cast to OutputT at the end
    // If not, cast to OutputT after normalize, scale in OutputT
    if (full_layer) {
      output[i] = static_cast<OutputT>(x * rms_inv * scale);
    } else {
      output[i] =
          static_cast<OutputT>(x * rms_inv) * static_cast<OutputT>(scale);
    }
  }
}
