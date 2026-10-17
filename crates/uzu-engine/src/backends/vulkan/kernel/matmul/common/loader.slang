import definitions;
import common.matmul;
import common.storage_address;
import matmul.common.simdgroup_multiply_accumulate;

// Counterpart of Metal matmul/common/loader.h's ThreadgroupLoader and quant_unpack.h's dequantize, staging into a
// typed threadgroup array.

// A tile's device-to-threadgroup staging into S: load_unsafe for a whole tile, load_safe zeroing outside
// source_tile_dimensions (columns, rows), next for the following tile.
public interface IThreadgroupLoader<T> {
  [mutating]
  public void load_unsafe<S : IThreadgroupStorage<T>>();
  [mutating]
  public void load_safe<S : IThreadgroupStorage<T>>(int2 source_tile_dimensions);
  public uint exponent_fields();
  [mutating]
  public void next();
}

// Metal's template parameters are constructor arguments, constants at every call site.
public struct ThreadgroupLoader<T : IModelFloat, let RIGHT : bool> : IThreadgroupLoader<T> {
  uint2 observed_magnitudes;
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

  // REDUCTION_DIMENSION: tiles advance along columns (K) rather than rows.
  public __init(
    Ptr<T, Access.Read> source_ptr,
    int source_leading_dim,
    int tile_rows,
    int tile_cols,
    int destination_leading_dim,
    bool reduction_dimension,
    int threadgroup_size,
    ThreadContext thread_context,
  ) {
    observed_magnitudes = uint2(0xffffffffu, 0u);
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
    source = element_pointer(
      source_ptr, element_index(uint(tile_row_index), uint(source_leading_dim), uint(tile_col_index))
    );
  }

  public uint exponent_fields() {
    let fields = matmul_a_fields(observed_magnitudes.x, observed_magnitudes.y);
    return RIGHT ? 0xffu | ((fields & 0xffu) << 8) | ((fields & 0xff0000u) << 8) : fields;
  }

  // Each invocation copies reads_per_thread contiguous elements of every thread_rows-th row (Metal's ReadVector).
  [mutating]
  public void load_unsafe<S : IThreadgroupStorage<T>>() {
    [unroll]
    for (int i = 0; i < threadgroup_tile_rows; i += thread_rows) {
      [unroll]
      for (int j = 0; j < reads_per_thread; j++) {
        let stored = source[uint(i * source_leading_dimension + j)];
        let magnitude = asuint(T.widened(stored.storageBits())) & 0x7fffffffu;
        observed_magnitudes.x = min(observed_magnitudes.x, magnitude - 1u);
        observed_magnitudes.y = max(observed_magnitudes.y, magnitude);
        S.write(destination + uint(i * destination_leading_dimension + j), stored);
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
        let loaded = source[uint(valid ? i * source_leading_dimension + j : 0)];
        let stored = valid ? loaded : zero;
        let magnitude = asuint(T.widened(stored.storageBits())) & 0x7fffffffu;
        observed_magnitudes.x = min(observed_magnitudes.x, magnitude - 1u);
        observed_magnitudes.y = max(observed_magnitudes.y, magnitude);
        S.write(destination + uint(i * destination_leading_dimension + j), stored);
      }
    }
  }

  [mutating]
  public void next() {
    source = element_pointer(source, uint2(uint(tile_stride), 0));
  }
}

// value rounded to U: each Metal operation on bfloat operands rounds its result to bfloat (FP32 is unchanged).
public float round_to<U : IModelFloat>(float value) {
  return U.fromFloat(value).toFloat();
}

// quant_unpack.h's dequantize: the pack_factor codes of one pack at w[w_offset] times scale plus bias, in U arithmetic,
// into S from w_local. Signed codes are stored with the top bit flipped.
public uint dequantize<U : IModelFloat, S : IThreadgroupStorage<U>>(
  Ptr<uint8_t, Access.Read> w,
  uint w_offset,
  float scale,
  float bias,
  uint w_local,
  int pack_factor,
  int bits,
  bool signed_codes,
  uint fields,
) {
  if (bits == 4) {
    let s1 = round_to<U>(scale / 16.0);
    for (int i = 0; i < pack_factor / 2; i++) {
      let word = uint(w[w_offset + uint(i)]) ^ (signed_codes ? 0x88u : 0u);
      fields = stage<U, S>(w_local + uint(2 * i), scale * float(word & 15u), bias, fields);
      fields = stage<U, S>(w_local + uint(2 * i + 1), s1 * float(word & 0xf0u), bias, fields);
    }
  } else {
    if (signed_codes) {
      let adjusted_bias = round_to<U>(bias + round_to<U>(scale * 128.0));
      for (int i = 0; i < pack_factor; i++) {
        let signed_weight = (int(w[w_offset + uint(i)]) << 24) >> 24;
        fields = stage<U, S>(w_local + uint(i), scale * float(signed_weight), adjusted_bias, fields);
      }
    } else {
      for (int i = 0; i < pack_factor; i++) {
        let code = int(w[w_offset + uint(i)]);
        fields = stage<U, S>(w_local + uint(i), scale * float(code), bias, fields);
      }
    }
  }
  return fields;
}

// The staged weight of a product and bias: separate typed product and sum roundings, including F32 where round_to is
// the identity.
public U staged_weight<U : IModelFloat>(float product, float bias) {
  let rounded = round_to<U>(product);
  let value = spirv_asm {
    %sum:$$float = OpFAdd $rounded $bias;
    OpDecorate %sum NoContraction;
    result:$$float = OpCopyObject %sum;
  };
  return U.fromFloat(value);
}

uint stage<U : IModelFloat, S : IThreadgroupStorage<U>>(uint index, float product, float bias, uint fields) {
  let stored = staged_weight<U>(product, bias);
  S.write(index, stored);
  return matmul_merge_fields(fields, matmul_fields(0.0, stored.toFloat()));
}
