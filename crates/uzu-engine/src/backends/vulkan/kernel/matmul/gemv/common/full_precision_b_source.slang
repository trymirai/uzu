import definitions;
import matmul.common.matmul;
import matmul.gemv.common.arguments;
import matmul.gemv.common.tile;

float4 unpack4<T : IStorageFloat>(uint2 words) {
  return float4(
    T.fromStorageBits(words.x).toFloat(), T.fromStorageBits(words.x >> 16).toFloat(),
    T.fromStorageBits(words.y).toFloat(), T.fromStorageBits(words.y >> 16).toFloat()
  );
}

float4 gemv_load4<T : IStorageFloat>(Ptr<T, Access.Read> source, uint offset, bool vector) {
  if (vector) {
    if (__naturalStrideOf<T>() == 4) {
      return load_aligned<float4, 16>(source, offset);
    }
    return unpack4<T>(load_aligned<uint2, 8>(source, offset));
  }
  return load_float4(source, offset);
}

public struct FullPrecisionBSource {
  public static void accumulate<A : IStorageFloat, B : IStorageFloat, D : IStorageFloat>(
    inout float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE],
    GemvOperands<A, B, D> ops,
    GemmParams params,
    GemvTile Tile,
    OutputTile tile,
    bool INPUT_ALIGNED,
    bool FULL_TILE
  ) {
    let VALUES_PER_THREAD = 4u;
    let BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    let thread_k = tile.reduction_lane * VALUES_PER_THREAD;
    let weights_row_stride = params.K;
    let input_row = tile.input_row;
    let vector = params.K % VALUES_PER_THREAD == 0;
    let weights = Ptr<B, Access.Read>(ops.b);
    var input = input_row * params.K + thread_k;
    uint weight_rows[GEMV_MAX_ROWS_PER_LANE];

    [unroll]
    for (uint output_index = 0; output_index < Tile.ROWS_PER_LANE; output_index++) {
      let global_row = tile.row0 + output_index;
      let lookup_row = FULL_TILE ? global_row : min(global_row, params.N - 1);
      let weight_row = params.gathered ? params.gathered_row(ops.gather_indices[input_row * params.N + lookup_row]) : lookup_row;
      weight_rows[output_index] = weight_row * weights_row_stride + thread_k;
    }

    uint k = 0;
    for (; k + BLOCK_SIZE <= params.K; k += BLOCK_SIZE) {
      let input_values = gemv_load4(ops.a, input, vector);
      [unroll]
      for (uint output_index = 0; output_index < Tile.ROWS_PER_LANE; output_index++) {
        let global_row = tile.row0 + output_index;
        if (FULL_TILE || global_row < params.N) {
          let weight_values = gemv_load4(weights, weight_rows[output_index], vector);
          result[0][output_index] += dot(weight_values, input_values);
        }
        weight_rows[output_index] += BLOCK_SIZE;
      }
      input += BLOCK_SIZE;
    }

    if (!INPUT_ALIGNED) {
      let remaining = k + thread_k < params.K
        ? min(int(params.K - k - thread_k), int(VALUES_PER_THREAD))
        : 0;
      if (remaining > 0) {
        [unroll]
        for (uint output_index = 0; output_index < Tile.ROWS_PER_LANE; output_index++) {
          let global_row = tile.row0 + output_index;
          if (OutputTile.row_in_range(FULL_TILE, global_row, params.N)) {
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
