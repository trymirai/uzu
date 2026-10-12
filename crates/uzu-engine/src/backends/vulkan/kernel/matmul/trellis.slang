import definitions;
import common.activation;
import common.storage_address;
import generated.trellis;

// The Metal GemmTrellis product of TrellisTransform activations and trellis-coded B rows. Row r of B is
// row_bytes(K) bytes of MSB-first 16-bit states, state i at bit i transition_bits (vector width 2) or byte i (width 4,
// restarting every restart_columns columns when nonzero); each state's hash gives VECTOR_WIDTH bytes, four per 4
// columns from the lowest byte, of biased weights b = 8 (sum of the byte's four 2-bit fields) + (3 (byte & 15) & 15).
// D[token, r] = bf16(((Σ a b - 54 Σ G_g) s + Σ_g G_g o_g) row_scales[r] activation_scales[token]) with the INT8 codes a,
// their column-group sums G and the codebook [s, o_0, o_1, o_2, o_3]; the integer dot is exact in any order and the FP32
// steps are each one rounding, left to right. A workgroup computes TRELLIS_OUTPUTS rows of one token, TRELLIS_LANES
// invocations each striding K four columns at a time; their partial dots meet in shared memory.

static const uint TRELLIS_THREADS = 256;
// Rows of B per workgroup, as the binding's group count n.div_ceil(8) assumes.
static const uint TRELLIS_OUTPUTS = 8;
static const uint TRELLIS_LANES = TRELLIS_THREADS / TRELLIS_OUTPUTS;
static const uint TRELLIS_WEIGHT_BIAS = 54;
static const uint TRELLIS_STATE_BITS = 16;

groupshared int trellis_partials[TRELLIS_THREADS];

uint trellis_block_bytes(uint vector_width, uint transition_bits, uint block_columns) {
  let transitions = block_columns / vector_width - 1;
  return (TRELLIS_STATE_BITS + transitions * transition_bits + 7) / 8;
}

uint trellis_row_bytes(uint vector_width, uint transition_bits, uint restart_columns, uint columns) {
  let block_columns = restart_columns == 0 ? columns : restart_columns;
  return columns / block_columns * trellis_block_bytes(vector_width, transition_bits, block_columns);
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

// The biased weights of columns column to column + 3 of a code row, one per byte from the lowest; column is a multiple
// of 4. Width 2 reads states pairwise from one window, as Metal does, without restarts.
uint trellis_decode(
  Ptr<uint8_t, Access.Read> row,
  uint column,
  uint vector_width,
  uint transition_bits,
  uint restart_columns
) {
  if (vector_width == 4) {
    var byte = column / 4;
    if (restart_columns != 0) {
      byte = column / restart_columns * trellis_block_bytes(4, transition_bits, restart_columns)
           + column % restart_columns / 4;
    }
    let hash = trellis_hash(uint(row[byte]) << 8 | uint(row[byte + 1]));
    return trellis_biased_weights(hash ^ (hash >> 16));
  }
  let first_bit = column / vector_width * transition_bits;
  let byte = first_bit / 8;
  let shift = first_bit % 8;
  let fourth = transition_bits == 6 ? uint(row[byte + 3]) : 0u;
  let window = uint(row[byte]) << 24 | uint(row[byte + 1]) << 16 | uint(row[byte + 2]) << 8 | fourth;
  let first = trellis_hash((window >> (16 - shift)) & 0xffffu);
  let second = trellis_hash((window >> (16 - transition_bits - shift)) & 0xffffu);
  return trellis_biased_weights(((first & 0xffffu) | second << 16) ^ ((second & 0xffff0000u) | first >> 16));
}

[[Kernel]]
[[Precondition("new", "(vector_width == 4 && transition_bits == 8 && (restart_columns == 0 || (restart_columns >= 4 && restart_columns.is_multiple_of(4)))) || (vector_width == 2 && (transition_bits == 4 || transition_bits == 6) && restart_columns == 0)")]]
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
  let lane = tid % TRELLIS_LANES;
  let row = tile * TRELLIS_OUTPUTS + tid / TRELLIS_LANES;
  var partial = 0;
  if (row < n) {
    let code_row =
      element_pointer(codes, element_index(row, trellis_row_bytes(vector_width, transition_bits, restart_columns, k), 0));
    let a = element_pointer(activations, element_index(token, k, 0));
    for (uint column = lane * 4; column < k; column += TRELLIS_LANES * 4) {
      let weights = trellis_decode(code_row, column, vector_width, transition_bits, restart_columns);
      for (uint j = 0; j < 4; j++) {
        partial += int(a[column + j]) * int((weights >> (8 * j)) & 0xffu);
      }
    }
  }
  trellis_partials[tid] = partial;
  GroupMemoryBarrierWithGroupSync();

  if (lane == 0 && row < n) {
    var dot = 0;
    for (uint other = tid; other < tid + TRELLIS_LANES; other++) {
      dot += trellis_partials[other];
    }
    let sums = element_pointer(column_group_sums, element_index(token, COLUMN_GROUP_COUNT, 0));
    let level = dot - int(TRELLIS_WEIGHT_BIAS) * int(sums[0] + sums[1] + sums[2] + sums[3]);
    var offsets = exact_product(sums[0], scale_and_offsets[1]);
    for (uint group = 1; group < COLUMN_GROUP_COUNT; group++) {
      offsets = ordered_sum(offsets, exact_product(sums[group], scale_and_offsets[1 + group]));
    }
    let value = exact_fma(float(level), scale_and_offsets[0], offsets);
    let scaled = exact_product(exact_product(value, row_scales[row]), activation_scales[token]);
    element_pointer(output, element_index(token, output_stride, row))[0] = bf16.fromFloat(scaled);
  }
}
