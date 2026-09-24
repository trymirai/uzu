#include <metal_stdlib>
#include "../common/dsl.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;

// Mirai S input embedding (`D4S4Spec`): column c of a row is row_scale * ladder[index] * table[code[c / 4]][c % 4]
// with one 4-bit ladder index per 64 columns (low nibble first), times input_scale, rounded to bf16, then the
// 32-wide output Hadamard with the factors.
PUBLIC KERNEL(MiraiSEmbeddingLookup)(
    const device uint* token_ids,
    const device uchar* codes,
    const device bfloat* row_scales,
    const device uchar* ladder_indices,
    const device half* ladder,
    const device char4* table,
    const device int* output_hadamard_factors,
    device bfloat* output,
    constant uint& batch_size,
    constant uint& vocab_size,
    constant uint& model_dim,
    constant float& input_scale,
    const uint column AXIS(model_dim, 32),
    const uint batch_index AXIS(batch_size, 1)
) {
  const uint token = token_ids[batch_index];
  if (token >= vocab_size) {
    output[batch_index * model_dim + column] = bfloat(0.0f);
    return;
  }
  const uint group = column / 64;
  const uchar packed_index = ladder_indices[token * (model_dim / 128) + group / 2];
  const uint ladder_index = group % 2 == 0 ? packed_index & 15 : packed_index >> 4;
  const char4 point = table[codes[token * (model_dim / 4) + column / 4]];
  const float value = float(row_scales[token]) * float(ladder[ladder_index]) * float(point[column % 4]) * input_scale;
  output[batch_index * model_dim + column] = simdgroup_output_random_hadamard_transform(
      ushort(column % METAL_SIMD_SIZE),
      bfloat(value),
      output_hadamard_factors[column]
  );
}
