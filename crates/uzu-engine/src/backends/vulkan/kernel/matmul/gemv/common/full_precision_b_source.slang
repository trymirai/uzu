import definitions;
import matmul.gemv.common.arguments;
import matmul.gemv.common.tile;

// The 4 values at element `offset` of `source` in FP32, one load where `vector`: rows of a K that is not a multiple of
// 4 are not 4-value aligned.
float4 gemv_load4<T : IModelFloat>(Ptr<T, Access.Read> source, uint offset, bool vector) {
  if (vector) {
    if (__naturalStrideOf<T>() == 4) {
      return load_aligned<float4, 16>(source, offset);
    }
    let packed = load_aligned<uint2, 8>(source, offset);
    return float4(T.fromStorageBits(packed.x).toFloat(), T.fromStorageBits(packed.x >> 16).toFloat(),
                  T.fromStorageBits(packed.y).toFloat(), T.fromStorageBits(packed.y >> 16).toFloat());
  }
  return load_float4(source, offset);
}

public struct FullPrecisionBSource {
  public static void accumulate<A : IModelFloat, B : IModelFloat, D : IModelFloat>(
    inout float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE],
    GemvOperands<A, B, D> ops,
    GemvParams params,
    GemvTile Tile,
    OutputTile tile,
    bool INPUT_ALIGNED,
    bool FULL_TILE
  ) {
    let VALUES_PER_THREAD = 4u;
    let BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    let k_stride = Tile.K_SPLIT * BLOCK_SIZE;
    let k_start = tile.k_slice * BLOCK_SIZE;
    let thread_k = tile.reduction_lane * VALUES_PER_THREAD + k_start;
    let weights_row_stride = params.in_vec_size;
    let input_row = tile.input_row;
    let vector = params.in_vec_size % VALUES_PER_THREAD == 0;
    let weights = Ptr<B, Access.Read>(ops.b);
    var input = input_row * params.in_vec_size + thread_k;
    uint weight_rows[GEMV_MAX_ROWS_PER_LANE];
    let last_row = params.out_vec_size > 0 ? params.out_vec_size - 1 : 0;

    [unroll]
    for (uint output_index = 0; output_index < Tile.ROWS_PER_LANE; output_index++) {
      let global_row = tile.row0 + output_index;
      let lookup_row = FULL_TILE ? global_row : min(global_row, last_row);
      let weight_row = params.gathered ? ops.gather_indices[input_row * params.out_vec_size + lookup_row] : lookup_row;
      weight_rows[output_index] = weight_row * weights_row_stride + thread_k;
    }

    uint k = k_start;
    for (; k + BLOCK_SIZE <= params.in_vec_size; k += k_stride) {
      let input_values = gemv_load4(ops.a, input, vector);
      [unroll]
      for (uint output_index = 0; output_index < Tile.ROWS_PER_LANE; output_index++) {
        let global_row = tile.row0 + output_index;
        if (FULL_TILE || global_row < params.out_vec_size) {
          let weight_values = gemv_load4(weights, weight_rows[output_index], vector);
          result[0][output_index] += dot(weight_values, input_values);
        }
        weight_rows[output_index] += k_stride;
      }
      input += k_stride;
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
              let x = ops.a[input + uint(index)].toFloat();
              let w = weights[weight_rows[output_index] + uint(index)].toFloat();
              result[0][output_index] += w * x;
            }
          }
        }
      }
    }
  }
}
