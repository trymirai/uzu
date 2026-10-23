import definitions;
import common.workgroup_reduction;
import generated.trellis;
import matmul.common.int8_operands;

// D[token, r] = bf16(((Σ a b - 54 Σ G_g) s + Σ_g G_g o_g) row_scales[r] activation_scales[token]) over TrellisTransform
// codes a, their column-group sums G and the codebook [s, o_0..o_3]. Row r of B is TrellisFormat::code_row_bytes(K)
// bytes of tape-2 blocks of block_columns columns (restart_columns, or K without restarts), packed bit after bit: each
// block holds a 16-bit state per VECTOR_WIDTH columns with transition_bits between states, its bits reversed. A state
// hashes to VECTOR_WIDTH bytes of biased weights b = 8 (sum of 2-bit fields) + (3 (byte & 15) & 15).

static const uint TRELLIS_THREADS = 256;
// Rows of B per workgroup, as the binding's group count n.div_ceil(8) assumes.
static const uint TRELLIS_OUTPUTS = 8;
static const uint TRELLIS_LANES = TRELLIS_THREADS / TRELLIS_OUTPUTS;
static const uint TRELLIS_WEIGHT_BIAS = 54;
static const uint TRELLIS_STATE_BITS = 16;
// Restart rows pad to this many bytes, rows without restarts to a byte.
static const uint TRELLIS_ROW_ALIGNMENT = 16;

uint trellis_block_bits(uint vector_width, uint transition_bits, uint block_columns) {
  return TRELLIS_STATE_BITS + (block_columns / vector_width - 1) * transition_bits;
}

uint trellis_row_bytes(uint vector_width, uint transition_bits, uint restart_columns, uint columns) {
  if (restart_columns == 0) {
    return (trellis_block_bits(vector_width, transition_bits, columns) + 7) / 8;
  }
  let bits = columns / restart_columns * trellis_block_bits(vector_width, transition_bits, restart_columns);
  return ((bits + 7) / 8 + TRELLIS_ROW_ALIGNMENT - 1) / TRELLIS_ROW_ALIGNMENT * TRELLIS_ROW_ALIGNMENT;
}

// The state of `column` in the code row at byte `row`: state i of a block of B bits is the little-endian 16-bit field at
// bit B - 16 - i transition_bits of the block, bit j of a row being bit j % 8 of its byte j / 8.
uint trellis_state(
  Ptr<uint8_t, Access.Read> codes,
  uint row,
  uint column,
  uint vector_width,
  uint transition_bits,
  uint block_columns
) {
  let block_bits = trellis_block_bits(vector_width, transition_bits, block_columns);
  let bit = (column / block_columns + 1) * block_bits - TRELLIS_STATE_BITS
          - column % block_columns / vector_width * transition_bits;
  let byte = row + bit / 8;
  let shift = bit % 8;
  // A field starting mid-byte ends in the third byte; one starting on a byte boundary must not read past the row.
  var third = 0u;
  if (shift != 0) {
    third = uint(codes[byte + 2]);
  }
  return ((uint(codes[byte]) | uint(codes[byte + 1]) << 8 | third << 16) >> shift) & 0xffffu;
}

uint trellis_hash(uint state) {
  var hash = state * 0xCFCCB83Fu + 0x584B4AA3u;
  hash ^= hash >> 16;
  return hash * 0x85EBCA6Bu;
}

uint trellis_biased_weights(uint hash) {
  let nibble_sums = hash - 3 * ((hash >> 2) & 0x33333333u);
  let byte_sums = (nibble_sums + (nibble_sums >> 4)) & 0x0F0F0F0Fu;
  let dither = ((hash & 0x0F0F0F0Fu) * 3) & 0x0F0F0F0Fu;
  return (byte_sums << 3) + dither;
}

// The biased weights of columns column to column + 3 of the code row at byte `row` of `codes`, one per byte from the
// lowest, each below 128; column is a multiple of 4 and the four columns lie in one block.
uint trellis_decode(
  Ptr<uint8_t, Access.Read> codes,
  uint row,
  uint column,
  uint vector_width,
  uint transition_bits,
  uint block_columns
) {
  let first = trellis_hash(trellis_state(codes, row, column, vector_width, transition_bits, block_columns));
  if (vector_width == 4) {
    return trellis_biased_weights(first ^ (first >> 16));
  }
  let second = trellis_hash(trellis_state(codes, row, column + 2, vector_width, transition_bits, block_columns));
  return trellis_biased_weights(((first & 0xffffu) | second << 16) ^ ((second & 0xffff0000u) | first >> 16));
}

[[Kernel]]
[[Precondition("new", "(vector_width == 2 || vector_width == 4) && (4..=8).contains(&transition_bits) && (restart_columns == 0 || (restart_columns >= 4 && restart_columns.is_multiple_of(4)))")]]
func TrellisGemv(
  Ptr<int8_t, Access.Read> activations,
  Ptr<float, Access.Read> column_group_sums,
  Ptr<float, Access.Read> activation_scales,
  Ptr<uint8_t, Access.Read> codes,
  Ptr<float, Access.Read> row_scales,
  Ptr<float, Access.Read> scale_and_offsets,
  Ptr<bf16, Access.ReadWrite> output,
  uint k,
  uint n,
  uint m,
  uint output_stride,
  [[Groups("n.div_ceil(8)")]] uint tile,
  [[Groups("m")]] uint token,
  [[Threads("256")]] uint tid,
  [[Specialize]] uint vector_width,
  [[Specialize]] uint transition_bits,
  [[Specialize]] uint restart_columns,
) {
  static_assert(COLUMN_GROUP_COUNT == 4, "one offset per column group");
  let invocation = subgroup_invocation();
  let lane = invocation % TRELLIS_LANES;
  let row = tile * TRELLIS_OUTPUTS + invocation / TRELLIS_LANES;
  var partial = 0;
  if (row < n) {
    let code_row = row * trellis_row_bytes(vector_width, transition_bits, restart_columns, k);
    let block_columns = restart_columns == 0 ? k : restart_columns;
    let a = token * k;
    for (uint column = lane * 4; column < k; column += TRELLIS_LANES * 4) {
      let weights = trellis_decode(codes, code_row, column, vector_width, transition_bits, block_columns);
      partial += dot_i8x4(load_aligned<uint, 4>(activations, a + column), weights);
    }
  }
  let level_dot = segment_reduce<Sum<int>, TRELLIS_THREADS>(partial, TRELLIS_LANES);
  if (lane == 0 && row < n) {
    let group = token * COLUMN_GROUP_COUNT;
    let sums = float4(
      column_group_sums[group], column_group_sums[group + 1], column_group_sums[group + 2], column_group_sums[group + 3]
    );
    let level = level_dot - int(TRELLIS_WEIGHT_BIAS) * int(sums.x + sums.y + sums.z + sums.w);
    let offsets = float4(scale_and_offsets[1], scale_and_offsets[2], scale_and_offsets[3], scale_and_offsets[4]);
    let value = float(level) * scale_and_offsets[0] + dot(sums, offsets);
    let scaled = value * row_scales[row] * activation_scales[token];
    output[token * output_stride + row] = bf16.fromFloat(scaled);
  }
}
