import definitions;
import common.matmul;
import common.storage_address;
import matmul.gemv.common.arguments;
import matmul.gemv.common.tile;

// Metal gemv/common/full_precision_b_source.h. Metal advances its row pointers by k_stride; here the offset `step` past
// them does.

// Metal's vec<T, 4> load at `source` + `offset`, widened to FP32: BF16 as Metal widens it, onto the upper half. Vulkan
// loads a vector only at its natural alignment, which rows of a K that is not a multiple of 4 lack (Metal loads them
// unaligned); those load the same 4 values as scalars.
float4 gemv_load4<T : IModelFloat>(Ptr<T, Access.Read> source, uint offset, bool vector) {
  if (vector && ((address_words(source).x + offset * uint(__naturalStrideOf<T>())) & (4 * uint(__naturalStrideOf<T>()) - 1)) == 0) {
    if (__naturalStrideOf<T>() == 4) {
      return gemv_cast<float4, T>(source)[offset / 4];
    }
    let packed = gemv_cast<uint2, T>(source)[offset / 4];
    return float4(asfloat(packed.x << 16), asfloat(packed.x & 0xffff0000u), asfloat(packed.y << 16), asfloat(packed.y & 0xffff0000u));
  }
  return float4(source[offset].toFloat(), source[offset + 1].toFloat(), source[offset + 2].toFloat(), source[offset + 3].toFloat());
}

public struct FullPrecisionBSource {
  public static void accumulate<A : IModelFloat, B : IModelFloat, D : IModelFloat>(
    inout uint fields,
    inout float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE],
    GemvOperands<A, B, D> ops,
    GemvParams params,
    GemvTile Tile,
    OutputTile tile,
    bool INPUT_ALIGNED,
    bool FULL_TILE
  ) {
    let VALUES_PER_THREAD = 4u;
    let BLOCK_SIZE = VALUES_PER_THREAD * GEMV_SIMD_SIZE;
    let k_stride = Tile.K_SPLIT * BLOCK_SIZE;
    let k_start = tile.k_slice * BLOCK_SIZE;
    let thread_k = tile.reduction_lane * VALUES_PER_THREAD + k_start;
    let weights_row_stride = params.in_vec_size;
    let input_row = tile.input_row;
    let vector = params.in_vec_size % VALUES_PER_THREAD == 0;
    let input = element_pointer(ops.a, element_index(input_row, params.in_vec_size, thread_k));
    Ptr<B, Access.Read> weight_rows[GEMV_MAX_ROWS_PER_LANE];
    let last_row = params.out_vec_size > 0 ? params.out_vec_size - 1 : 0;

    [unroll]
    for (uint output_index = 0; output_index < Tile.ROWS_PER_LANE; output_index++) {
      let global_row = tile.row0 + output_index;
      let lookup_row = FULL_TILE ? global_row : min(global_row, last_row);
      let weight_row = params.gathered ? ops.gather_indices[input_row * params.out_vec_size + lookup_row] : lookup_row;
      weight_rows[output_index] =
        element_pointer(gemv_cast<B, uint>(ops.b), element_index(weight_row, weights_row_stride, thread_k));
    }

    uint k = k_start;
    uint step = 0;
    for (; k + BLOCK_SIZE <= params.in_vec_size; k += k_stride) {
      let input_values = gemv_load4(input, step, vector);
      [unroll] for (uint i = 0; i < 4; i++) { fields = matmul_merge_fields(fields, matmul_fields(input_values[i], 0.0)); }
      [unroll]
      for (uint output_index = 0; output_index < Tile.ROWS_PER_LANE; output_index++) {
        let global_row = tile.row0 + output_index;
        if (FULL_TILE || global_row < params.out_vec_size) {
          let weight_values = gemv_load4(weight_rows[output_index], step, vector);
          [unroll] for (uint i = 0; i < 4; i++) { fields = matmul_merge_fields(fields, matmul_fields(0.0, weight_values[i])); }
          result[0][output_index] += dot(weight_values, input_values);
        }
      }
      step += k_stride;
    }

    if (Tile.K_SPLIT == 1 && !INPUT_ALIGNED) {
      let thread_offset = tile.reduction_lane * VALUES_PER_THREAD;
      let remaining = k + thread_offset < params.in_vec_size
        ? min(int(params.in_vec_size - k - thread_offset), int(VALUES_PER_THREAD))
        : 0;
      if (remaining > 0) {
        [unroll]
        for (uint output_index = 0; output_index < Tile.ROWS_PER_LANE; output_index++) {
          let global_row = tile.row0 + output_index;
          if (OutputTile.row_in_range(FULL_TILE, global_row, params.out_vec_size)) {
            for (int index = 0; index < remaining; index++) {
              let x = input[step + uint(index)].toFloat();
              let w = weight_rows[output_index][step + uint(index)].toFloat();
              fields = matmul_merge_fields(fields, matmul_fields(x, w));
              result[0][output_index] += w * x;
            }
          }
        }
      }
    }
  }
}
