#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;

#define BLOCK_SIZE 1024

UZU_CONST uint VALUES_PER_THREAD = 4;
static_assert(VALUES_PER_THREAD == hadamard::VECTOR_SIZE, "rotation needs 4 values per lane");

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
    const uint hadamard_size SPECIALIZE,
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
    vec<InputT, VALUES_PER_THREAD> vals = load_vec4(input + first);
    // We can also fuse:
    // - TensorCopy (copy_to_shortcut)
    // - TensorAddSwap (copy_to_shortcut + residual_add)
    // Normalization in TensorAddSwap fusion mode operates on input + shortcut
    if (copy_to_shortcut) {
      if (residual_add) {
        vals += load_vec4(shortcut + first);
      }
      store_vec4(shortcut + first, vals);
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
  for (uint base = 0; base < element_count; base += BLOCK_SIZE * VALUES_PER_THREAD) {
    const uint first = base + thread_in_row * VALUES_PER_THREAD;
    const bool in_bounds = first < element_count;
    float4 values = 0.0f;
    vec<OutputT, VALUES_PER_THREAD> outs = 0;
    if (in_bounds) {
      // If we fuse TensorAddSwap, read shortcut (that now has input + shortcut)
      // No need for memory barrier because each thread only reads what it wrote
      vec<InputT, VALUES_PER_THREAD> xs = residual_add ? load_vec4(shortcut + first) : load_vec4(input + first);
      METAL_PRAGMA_UNROLL
      for (uint k = 0; k < VALUES_PER_THREAD; ++k) {
        const uint affine_index = affine_offset + first + k;
        AccumT normalized = (static_cast<AccumT>(xs[k]) - mean) * rms_inv;
        // If full_layer, normalize and scale in AccumT, cast to OutputT at the end
        // If not, cast to OutputT after normalize, scale in OutputT
        if (has_scales) {
          AccumT scale = static_cast<AccumT>(scales[affine_index]) + static_cast<AccumT>(scale_offset);
          normalized = full_layer ? normalized * scale
                                  : static_cast<AccumT>(static_cast<OutputT>(normalized) * static_cast<OutputT>(scale));
        }
        if (!use_hadamard) {
          normalized = static_cast<AccumT>(static_cast<OutputT>(normalized));
        }
        if (has_biases) {
          normalized += static_cast<AccumT>(biases[affine_index]);
        }
        values[k] = normalized;
        outs[k] = static_cast<OutputT>(normalized);
      }
      if (use_hadamard) {
        values *= float4(load_vec4(hadamard_factors + first));
      }
    }
    if (use_hadamard) {
      outs = vec<OutputT, VALUES_PER_THREAD>(
          hadamard::simdgroup_transform_vector(thread_context.simd_lane_id, values, hadamard_size)
      );
    }
    if (in_bounds) {
      store_vec4(output + first, outs);
    }
  }
}
