import definitions;
import matmul.common.matmul;
import generated.quantization_method;
import matmul.common.simdgroup_multiply_accumulate;

// A tile's device-to-threadgroup staging into S: load_unsafe for a whole tile, load_safe zeroing outside
// source_tile_dimensions (columns, rows), next for the following tile.
public interface IThreadgroupLoader<T> {
  [mutating]
  public void load_unsafe<S : IThreadgroupStorage<T>>();
  [mutating]
  public void load_safe<S : IThreadgroupStorage<T>>(int2 source_tile_dimensions);
  [mutating]
  public void next();
}

// Its shape is constructor arguments, constants at every call site.
public struct ThreadgroupLoader<T : IModelFloat> : IThreadgroupLoader<T> {
  int threadgroup_tile_rows;
  int destination_leading_dimension;
  int reads_per_thread;
  int thread_rows;
  int source_leading_dimension;
  int tile_stride;
  int tile_row_index;
  int tile_col_index;
  uint destination;
  Ptr<T, Access.Read> source;
  uint offset;

  // The tile from element `source_offset` of `source_ptr`. REDUCTION_DIMENSION: tiles advance along columns (K) rather
  // than rows.
  public __init(
    Ptr<T, Access.Read> source_ptr,
    uint source_offset,
    int source_leading_dim,
    int tile_rows,
    int tile_cols,
    int destination_leading_dim,
    bool reduction_dimension,
    int threadgroup_size,
    ThreadContext thread_context,
  ) {
    threadgroup_tile_rows = tile_rows;
    destination_leading_dimension = destination_leading_dim;
    reads_per_thread = (tile_cols * tile_rows) / threadgroup_size;
    let thread_cols = tile_cols / reads_per_thread;
    thread_rows = threadgroup_size / thread_cols;
    source_leading_dimension = source_leading_dim;
    tile_stride = reduction_dimension ? tile_cols : tile_rows * source_leading_dim;
    let thread_index = int(thread_context.thread_index);
    tile_row_index = thread_index / thread_cols;
    tile_col_index = reads_per_thread * (thread_index % thread_cols);
    destination = uint(tile_row_index * destination_leading_dim + tile_col_index);
    source = source_ptr;
    offset = source_offset + uint(tile_row_index) * uint(source_leading_dim) + uint(tile_col_index);
  }

  // Each invocation copies reads_per_thread contiguous elements of every thread_rows-th row.
  [mutating]
  public void load_unsafe<S : IThreadgroupStorage<T>>() {
    [unroll]
    for (int i = 0; i < threadgroup_tile_rows; i += thread_rows) {
      [unroll]
      for (int j = 0; j < reads_per_thread; j++) {
        let value = source[offset + uint(i * source_leading_dimension + j)];
        S.write(destination + uint(i * destination_leading_dimension + j), value);
      }
    }
  }

  [mutating]
  public void load_safe<S : IThreadgroupStorage<T>>(int2 source_tile_dimensions) {
    let dimensions = source_tile_dimensions - int2(tile_col_index, tile_row_index);
    let zero = T.fromFloat(0.0);

    if (dimensions.x <= 0 || dimensions.y <= 0) {
      [unroll]
      for (int i = 0; i < threadgroup_tile_rows; i += thread_rows) {
        [unroll]
        for (int j = 0; j < reads_per_thread; j++) {
          S.write(destination + uint(i * destination_leading_dimension + j), zero);
        }
      }
      return;
    }

    [unroll]
    for (int i = 0; i < threadgroup_tile_rows; i += thread_rows) {
      [unroll]
      for (int j = 0; j < reads_per_thread; j++) {
        let valid = i < dimensions.y && j < dimensions.x;
        let loaded = source[offset + uint(valid ? i * source_leading_dimension + j : 0)];
        S.write(destination + uint(i * destination_leading_dimension + j), valid ? loaded : zero);
      }
    }
  }

  [mutating]
  public void next() {
    offset += uint(tile_stride);
  }
}

// value rounded to U, as every 16-bit operation of the staged loaders rounds its result.
public float round_to<U : IModelFloat>(float value) {
  return U.fromFloat(value).toFloat();
}

// The codes of code byte `byte` times scale plus bias in U arithmetic: two U4 values, the low nibble first and the high
// one read in place, 16 times its code, against the scale over 16; or one U8 value. Signed codes are stored with the top
// bit flipped.
public void dequantize_code<U : IModelFloat>(
  uint byte,
  float scale,
  float bias,
  bool u4,
  bool signed_codes,
  out U values[2],
) {
  values[1] = U.fromFloat(0.0);
  if (u4) {
    let code = byte ^ (signed_codes ? 0x88u : 0u);
    values[0] = staged_weight<U>(scale * float(code & 15u), bias);
    values[1] = staged_weight<U>(round_to<U>(scale / 16.0) * float(code & 0xf0u), bias);
  } else if (signed_codes) {
    values[0] = staged_weight<U>(scale * float(int(byte << 24) >> 24), round_to<U>(bias + round_to<U>(scale * 128.0)));
  } else {
    values[0] = staged_weight<U>(scale * float(byte), bias);
  }
}

// The staged weight of a product and bias, each rounded to U.
public U staged_weight<U : IModelFloat>(float product, float bias) {
  return U.fromFloat(round_to<U>(product) + bias);
}

// The bias a weight group's codes stage with in U: the ScaleBias bias, otherwise -scale zero_point rounded to U.
public float staged_bias<U : IModelFloat>(WeightGroup group, QuantizationMethod method) {
  return method == QuantizationMethod.ScaleBias ? group.bias : round_to<U>(-group.scale * float(group.zero_point));
}
