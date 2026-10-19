#include <metal_stdlib>
#include "../common/dsl.h"
#include "../generated/trellis.h"
#include "../matmul/gemm/common/gemm_tiling.h"
#include "../matmul/gemm/common/mxu_mma_core.h"
#include "../matmul/gemm/common/quantized/trellis.h"

using namespace metal;
namespace tape = uzu::gemm::trellis;

#define THREADS_PER_THREADGROUP 256

// A tied trellis table stores each vocabulary row in the Kronecker basis of its readout. One threadgroup per token
// decodes the row, then undoes the rotation like lalamo's KroneckerRotation.unprocess_weights: view the row as
// [width, mixing], Hadamard over width, multiply by the mixing matrix, flip signs.
PUBLIC KERNEL(TrellisEmbeddingLookup)(
    const device uint* token_ids,
    const device uchar* codes,
    const device float* row_scales,
    const constant float codebook[5],
    const device float* signs,
    const device float* mixing,
    device bfloat* output,
    constant uint& batch_size,
    constant uint& vocab_size,
    constant uint& model_dim,
    constant float& input_scale,
    threadgroup float row[uzu::trellis::LOOKUP_MAX_COLUMNS],
    const uint vector_width SPECIALIZE,
    const uint transition_bits SPECIALIZE,
    const uint restart_columns SPECIALIZE,
    const uint batch_index GROUPS(batch_size),
    const uint thread_index THREADS(THREADS_PER_THREADGROUP)
) {
  device bfloat* output_row = output + batch_index * model_dim;
  const uint token_id = token_ids[batch_index];
  if (token_id >= vocab_size) {
    for (uint column = thread_index; column < model_dim; column += THREADS_PER_THREADGROUP) {
      output_row[column] = bfloat(0.0f);
    }
    return;
  }

  const tape::GemmTrellisFormat format = {vector_width, transition_bits, restart_columns};
  const device uchar* code_row = codes + token_id * tape::row_bytes(format, model_dim);
  const float row_scale = row_scales[token_id];
  for (uint column = 4 * thread_index; column < model_dim; column += 4 * THREADS_PER_THREADGROUP) {
    const uint biased_weights = tape::decode_biased_weights(format, code_row, column);
    for (uint lane = 0; lane < 4; ++lane) {
      const int level = int((biased_weights >> (8 * lane)) & 0xFF) - int(tape::WEIGHT_BIAS);
      // lalamo rounds the decoded row to bf16 before unrotating it.
      row[column + lane] = float(bfloat((codebook[0] * float(level) + codebook[1 + lane]) * row_scale));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint width = model_dim & (0u - model_dim);
  const uint mixing_dimension = model_dim / width;
  for (uint stride = 1; stride < width; stride <<= 1) {
    for (uint pair = thread_index; pair < model_dim / 2; pair += THREADS_PER_THREADGROUP) {
      const uint pair_row = pair / mixing_dimension;
      const uint low_row = pair_row / stride * 2 * stride + pair_row % stride;
      const uint low = low_row * mixing_dimension + pair % mixing_dimension;
      const uint high = low + stride * mixing_dimension;
      const float low_value = row[low];
      const float high_value = row[high];
      row[low] = low_value + high_value;
      row[high] = low_value - high_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float normalization = 1.0f / sqrt(float(width));
  for (uint column = thread_index; column < model_dim; column += THREADS_PER_THREADGROUP) {
    const uint row_start = column / mixing_dimension * mixing_dimension;
    const uint mixing_column = column % mixing_dimension;
    float value = 0.0f;
    for (uint source = 0; source < mixing_dimension; ++source) {
      value = fma(row[row_start + source], mixing[source * mixing_dimension + mixing_column], value);
    }
    output_row[column] = bfloat(value * normalization * signs[column] * input_scale);
  }
}
