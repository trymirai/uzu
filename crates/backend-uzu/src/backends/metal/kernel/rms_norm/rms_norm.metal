#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;

#define BLOCK_SIZE 1024
// Maximum supported LoRA rank for the fused RMSNorm+A_down path.
#define LORA_RANK 16

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
    const device int32_t* hadamard_factors OPTIONAL(use_hadamard),
    const device OutputT* adapter_down_prime OPTIONAL(use_lora),
    device OutputT* h_output OPTIONAL(use_lora),
    constant uint& batch_size,
    constant uint& element_count,
    constant float& epsilon,
    constant float& scale_offset,
    const bool in_place SPECIALIZE,
    const bool full_layer SPECIALIZE,
    const bool copy_to_shortcut SPECIALIZE,
    const bool residual_add SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    const bool use_lora SPECIALIZE,
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

  // Step 3 - elementwise normalization (and optional LoRA h accumulation)
  //
  // When use_lora=true, also compute h[r] = A_down_prime[r, :] @ y_scaled for
  // r in [0, LORA_RANK).  A_down_prime = A_down @ H was pre-computed offline at
  // model-load time, so multiplying by the pre-Hadamard scaled-normalised value
  // (y_scaled = scale * normalised(input)) reproduces the same h that the
  // stand-alone A_down GEMV would have produced from y_rht.
  float h_partial[LORA_RANK];
  if (use_lora) {
    for (uint r = 0; r < LORA_RANK; r++)
      h_partial[r] = 0.0f;
  }

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
    OutputT val;
    if (full_layer) {
      val = static_cast<OutputT>(x * rms_inv * scale);
    } else {
      val = static_cast<OutputT>(x * rms_inv) * static_cast<OutputT>(scale);
    }

    // Accumulate partial dot products for h = A_down_prime @ y_scaled.
    // This uses the pre-Hadamard scaled value (val before the butterfly), which
    // is the correct operand given A_down_prime = A_down @ H.
    if (use_lora) {
      float y_i = float(val);
      for (uint r = 0; r < LORA_RANK; r++)
        h_partial[r] += float(adapter_down_prime[r * element_count + i]) * y_i;
    }

    if (use_hadamard) {
      val = static_cast<OutputT>(simdgroup_random_hadamard_transform(
          static_cast<ushort>(thread_in_row % METAL_SIMD_SIZE),
          val,
          hadamard_factors[i]
      ));
    }

    output[i] = val;
  }

  // Step 4 (only when use_lora) - cross-thread reduce h_partial[r] and write
  // h_output. Reuse shared_sum (size METAL_SIMD_SIZE = 32) sequentially for
  // each of the LORA_RANK=16 h rows.  shared_sum is idle after step 2.
  if (use_lora) {
    // h_output is indexed by [batch_idx * LORA_RANK + r].
    h_output += batch_idx * LORA_RANK;
    for (uint r = 0; r < LORA_RANK; r++) {
      float h_r =
          threadgroup_cooperative_reduce<SimdReduceSum<float>, BLOCK_SIZE>(
              h_partial[r],
              reinterpret_cast<threadgroup float*>(shared_sum),
              thread_context
          );
      if (thread_in_row == 0) {
        h_output[r] = static_cast<OutputT>(h_r);
      }
    }
  }
}
