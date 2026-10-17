import definitions;
import common.matmul;
import common.storage_address;
import generated.gemm_b_layout;
import generated.quantization_method;
import matmul.gemv.common.arguments;
import matmul.gemv.common.quant_chunk;
import matmul.gemv.common.quant_metadata;
import matmul.gemv.common.tile;

// Metal gemv/common/quant_slice.h, its compile-time constants computed from the specialized GROUP_SIZE, BITS and
// GROUP_LANES. Metal's static_asserts are the kernel's preconditions.

public struct QuantSlice {
  public uint GROUP_SIZE;
  public uint BITS;
  public uint VALUES_PER_LANE;
  uint CHUNK_VALUES;
  uint MAX_SLICE_VALUES;
  uint SLICE_VALUES;
  public uint SLICES_PER_LANE;
  uint SLICE_BYTES;
  uint SLICE_WORDS;
  uint CHUNKS_PER_SLICE;

  uint4 weights[GEMV_MAX_ROWS_PER_LANE][GEMV_MAX_SLICE_WORDS];

  public __init(GemvTile Tile, uint group_size, uint bits) {
    [unroll]
    for (uint r = 0; r < GEMV_MAX_ROWS_PER_LANE; r++) {
      [unroll]
      for (uint word = 0; word < GEMV_MAX_SLICE_WORDS; word++) { weights[r][word] = uint4(0); }
    }
    GROUP_SIZE = group_size;
    BITS = bits;
    VALUES_PER_LANE = GROUP_SIZE / Tile.GROUP_LANES;
    CHUNK_VALUES = QuantChunk.VALUES;
    MAX_SLICE_VALUES = 32;
    SLICE_VALUES = VALUES_PER_LANE > MAX_SLICE_VALUES ? MAX_SLICE_VALUES : VALUES_PER_LANE;
    SLICES_PER_LANE = VALUES_PER_LANE / SLICE_VALUES;
    SLICE_BYTES = SLICE_VALUES * BITS / QuantChunk.BITS_PER_BYTE;
    SLICE_WORDS = (SLICE_BYTES + QuantChunk.VECTOR_BYTES - 1) / QuantChunk.VECTOR_BYTES;
    CHUNKS_PER_SLICE = SLICE_VALUES / CHUNK_VALUES;
  }

  public uint row_stride(GemvParams params) {
    return params.in_vec_size * BITS / QuantChunk.BITS_PER_BYTE;
  }

  [mutating]
  public void load_weights(
    QuantPosition position,
    Ptr<uint8_t, Access.Read> weights_base,
    uint weight_row_indices[GEMV_MAX_ROWS_PER_LANE],
    uint row_stride,
    uint group_offset,
    GemvParams params,
    GemvTile Tile,
    bool INPUT_ALIGNED
  ) {
    let k = position.group * GROUP_SIZE + group_offset + position.slice * SLICE_VALUES;
    let offset_bytes = k * BITS / QuantChunk.BITS_PER_BYTE;
    [unroll]
    for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
      let row = weight_row_indices[R];
      let byte = gemm_b_code_byte(params.b_layout, row, k, params.in_vec_size, BITS == 4);
      let source = element_pointer(weights_base, byte);
      let valid_bytes = INPUT_ALIGNED ? SLICE_BYTES : (offset_bytes < row_stride ? min(row_stride - offset_bytes, SLICE_BYTES) : 0u);
      let LOAD_ALIGNMENT = min(SLICE_BYTES, QuantChunk.VECTOR_BYTES);
      let aligned = (address_words(source).x & (LOAD_ALIGNMENT - 1)) == 0;
      load_words(valid_bytes != SLICE_BYTES || !aligned, weights[R], source, valid_bytes);
    }
  }

  public void accumulate<A : IModelFloat, B : IModelFloat, D : IModelFloat>(
    inout uint fields,
    inout float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE],
    QuantPosition position,
    GemvOperands<A, B, D> ops,
    GemvParams params,
    GemvTile Tile,
    OutputTile tile,
    uint group_offset,
    uint batch_remaining,
    QuantMetadata metadata,
    QuantizationMethod B_PROLOGUE,
    bool INPUT_ALIGNED,
    bool FULL_TILE
  ) {
    // Nonzero centered codes have fields 127..126+BITS; include 1 for the ScaleBias input sum.
    fields = matmul_merge_fields(fields, 0xffu | (127u << 8) | ((126u + BITS) << 24));
    float partial[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE];
    float input_sum[GEMV_MAX_INPUT_ROWS];
    var smallest_minus_one = 0xffffffffu;
    var largest = 0u;
    [unroll]
    for (uint I = 0; I < Tile.INPUT_ROWS; I++) {
      input_sum[I] = 0.0;
      [unroll]
      for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
        partial[I][R] = 0.0;
      }
    }
    [unroll]
    for (uint chunk = 0; chunk < CHUNKS_PER_SLICE; chunk++) {
      float weight_values[GEMV_MAX_ROWS_PER_LANE][QuantChunk.VALUES];
      [unroll]
      for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
        metadata.decode(weights[R], chunk, R, weight_values[R]);
      }
      [unroll]
      for (uint I = 0; I < Tile.INPUT_ROWS; I++) {
        let source_row = FULL_TILE ? I : min(I, batch_remaining - 1);
        let k = position.group * GROUP_SIZE + group_offset + position.slice * SLICE_VALUES + chunk * CHUNK_VALUES;
        let input_row = tile.input_row + source_row;
        let input = element_pointer(ops.a, element_index(input_row, params.in_vec_size, k));
        float input_values[QuantChunk.VALUES];
        QuantChunk.load(INPUT_ALIGNED, input, input_values, k, params.in_vec_size, input_row);
        [unroll]
        for (uint i = 0; i < CHUNK_VALUES; i++) {
          let bits = asuint(input_values[i]) & 0x7fffffffu;
          smallest_minus_one = min(smallest_minus_one, bits - 1u);
          largest = max(largest, bits);
        }
        if (B_PROLOGUE == QuantizationMethod.ScaleBias) {
          [unroll]
          for (uint i = 0; i < CHUNK_VALUES; i++) {
            input_sum[I] += input_values[i];
          }
        }
        [unroll]
        for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
          [unroll]
          for (uint i = 0; i < CHUNK_VALUES; i++) {
            partial[I][R] = fma(input_values[i], weight_values[R][i], partial[I][R]);
          }
        }
      }
    }
    fields = matmul_merge_fields(fields, matmul_a_fields(smallest_minus_one, largest));
    metadata.fold(result, partial, input_sum, Tile);
  }

  void load_words(
    bool BOUNDED,
    out uint4 words[GEMV_MAX_SLICE_WORDS],
    Ptr<uint8_t, Access.Read> source,
    uint valid_bytes
  ) {
    if (!BOUNDED && SLICE_BYTES == QuantChunk.WORD_BYTES) {
      let packed = gemv_cast<uint, uint8_t>(source)[0];
      words[0] = uint4(packed, 0u, 0u, 0u);
    } else if (!BOUNDED && SLICE_BYTES == 2 * QuantChunk.WORD_BYTES) {
      let packed = gemv_cast<uint2, uint8_t>(source)[0];
      words[0] = uint4(packed[0], packed[1], 0u, 0u);
    } else if (!BOUNDED) {
      let packed = gemv_cast<uint4, uint8_t>(source);
      [unroll]
      for (uint word = 0; word < SLICE_WORDS; word++) {
        words[word] = packed[word];
      }
    } else {
      [unroll]
      for (uint word = 0; word < SLICE_WORDS; word++) {
        var packed = uint4(0u);
        [unroll]
        for (uint component = 0; component < QuantChunk.WORD_LANES; component++) {
          var value = 0u;
          [unroll]
          for (uint byte_index = 0; byte_index < QuantChunk.WORD_BYTES; byte_index++) {
            let index = QuantChunk.VECTOR_BYTES * word + QuantChunk.WORD_BYTES * component + byte_index;
            if (index < SLICE_BYTES && index < valid_bytes) {
              value |= uint(source[index]) << (QuantChunk.BITS_PER_BYTE * byte_index);
            }
          }
          packed[component] = value;
        }
        words[word] = packed;
      }
    }
  }
}
