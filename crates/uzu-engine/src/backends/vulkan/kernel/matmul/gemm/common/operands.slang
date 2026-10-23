import definitions;
import common.matmul;
import common.quantized_value;
import generated.gemm;
import generated.quantization_method;
import matmul.common.loader;
import matmul.common.simdgroup_multiply_accumulate;

// The right operand B as stored, N rows of K, full precision or U4/U8 codes, and the loader that stages a
// block_n x block_k tile of it, rows shared_stride_b apart.
public interface IRightOperand<B> {
  associatedtype Loader : IThreadgroupLoader<B>;

  public Loader make_loader(
    uint block_col,
    int block_n,
    int block_k,
    int shared_stride_b,
    int threadgroup_threads,
    ThreadContext thread_context,
  );
}

// Full-precision B, element (n, k) at n leading_dimension_b + k.
public struct FullPrecisionRight<B : IModelFloat> : IRightOperand<B> {
  public Ptr<B, Access.Read> dense;
  public uint leading_dimension_b;

  public typealias Loader = ThreadgroupLoader<B>;

  public Loader make_loader(
    uint block_col,
    int block_n,
    int block_k,
    int shared_stride_b,
    int threadgroup_threads,
    ThreadContext thread_context,
  ) {
    return ThreadgroupLoader<B>(
      dense,
      block_col * leading_dimension_b,
      int(leading_dimension_b),
      block_n,
      block_k,
      shared_stride_b,
      true,
      threadgroup_threads,
      thread_context
    );
  }
}

// U4/U8 B of K-element rows under any scheme.
public struct QuantizedRight<B : IModelFloat> : IRightOperand<B> {
  public QuantizedWeights<B> weights;
  public uint k;

  public typealias Loader = QuantizedBlockLoader<B>;

  public Loader make_loader(
    uint block_col,
    int block_n,
    int block_k,
    int shared_stride_b,
    int threadgroup_threads,
    ThreadContext thread_context,
  ) {
    return QuantizedBlockLoader<B>(
      weights, k, block_col, block_n, block_k, shared_stride_b, threadgroup_threads, thread_context
    );
  }
}

// Stages a block_n x block_k tile of quantized B, each invocation dequantizing reads_per_thread consecutive packs of one
// row with its group's scale and bias in T arithmetic: ScaleBias reads the bias, ScaleZeroPoint forms -scale zero_point
// in T, ScaleSymmetric uses the midpoint 2^(bits - 1).
public struct QuantizedBlockLoader<T : IModelFloat> : IThreadgroupLoader<T> {
  int threadgroup_tile_rows;
  int bits;
  int pack_factor;
  int reads_per_thread;
  int group_steps_per_block;
  bool tile_has_idle_threads;
  QuantizationMethod method;

  int tile_stride;
  int group_step_counter;
  int group_stride;
  uint zero_point_group_stride;

  int tile_row_index;
  int tile_col_index;

  uint dst;
  Ptr<uint8_t, Access.Read> src;
  uint src_index;
  Ptr<T, Access.Read> scales;
  Ptr<T, Access.Read> biases;
  uint scale_index;
  Ptr<uint8_t, Access.Read> zero_points;
  uint zero_point_index;
  bool signed_codes;

  public __init(
    QuantizedWeights<T> right,
    uint k,
    uint block_col,
    int tile_rows,
    int tile_cols,
    int destination_leading_dimension,
    int threadgroup_size,
    ThreadContext thread_context,
  ) {
    threadgroup_tile_rows = tile_rows;
    bits = right.u4 ? 4 : 8;
    pack_factor = 8 / bits;
    method = right.method;
    let tile_cols_packed = tile_cols / pack_factor;
    tile_has_idle_threads = tile_cols_packed * tile_rows < threadgroup_size;
    reads_per_thread = tile_has_idle_threads ? 1 : (tile_cols_packed * tile_rows) / threadgroup_size;
    group_steps_per_block = int(right.group_size) / tile_cols;

    tile_stride = tile_cols_packed;
    group_step_counter = 0;
    group_stride = int(right.scale_group_stride);
    zero_point_group_stride = right.zero_point_group_stride;
    let thread_index = int(thread_context.thread_index);
    tile_row_index = reads_per_thread * thread_index / tile_cols_packed;
    tile_col_index = (reads_per_thread * thread_index) % tile_cols_packed;
    dst = uint(tile_row_index * destination_leading_dimension + tile_col_index * pack_factor);
    let row = block_col + uint(tile_row_index);
    src = right.codes;
    src_index = row * (k / uint(pack_factor)) + uint(tile_col_index);
    scales = right.scales;
    biases = right.biases;
    scale_index = row * right.scale_output_stride;
    zero_points = right.zero_points;
    zero_point_index = row * right.zero_point_output_stride;
    signed_codes = right.signed_codes;
  }

  void current_scale_bias(out float scale, out float bias) {
    scale = scales[scale_index].toFloat();
    if (method == QuantizationMethod.ScaleBias) {
      bias = biases[scale_index].toFloat();
      return;
    }
    var zero_point = 1u << uint(bits - 1);
    if (method == QuantizationMethod.ScaleZeroPoint) {
      zero_point = bits == 4 ? read_nibble(zero_points, zero_point_index) : uint(zero_points[zero_point_index]);
    }
    bias = round_to<T>(-scale * float(zero_point));
  }

  [mutating]
  public void load_unsafe<S : IThreadgroupStorage<T>>() {
    if (tile_has_idle_threads && tile_row_index >= threadgroup_tile_rows) {
      return;
    }
    float scale;
    float bias;
    current_scale_bias(scale, bias);
    for (int i = 0; i < reads_per_thread; i++) {
      dequantize<T, S>(
        src, src_index + uint(i), scale, bias, dst + uint(i * pack_factor), pack_factor, bits, signed_codes
      );
    }
  }

  [mutating]
  public void load_safe<S : IThreadgroupStorage<T>>(int2 src_tile_dim) {
    if (tile_has_idle_threads && tile_row_index >= threadgroup_tile_rows) {
      return;
    }
    let zero = T.fromFloat(0.0);
    if (tile_row_index >= src_tile_dim.y) {
      for (int i = 0; i < reads_per_thread * pack_factor; i++) {
        S.write(dst + uint(i), zero);
      }
      return;
    }
    let valid_cols = src_tile_dim.x;
    let valid_packs = (valid_cols + pack_factor - 1) / pack_factor;
    float scale;
    float bias;
    current_scale_bias(scale, bias);
    for (int i = 0; i < reads_per_thread; i++) {
      let pack_index = tile_col_index + i;
      if (pack_index < valid_packs) {
        dequantize<T, S>(
          src, src_index + uint(i), scale, bias, dst + uint(i * pack_factor), pack_factor, bits, signed_codes
        );
        if (pack_index == valid_packs - 1) {
          for (int lane = valid_cols - pack_index * pack_factor; lane < pack_factor; ++lane) {
            S.write(dst + uint(i * pack_factor + lane), zero);
          }
        }
      } else {
        for (int lane = 0; lane < pack_factor; ++lane) {
          S.write(dst + uint(i * pack_factor + lane), zero);
        }
      }
    }
  }

  [mutating]
  public void next() {
    src_index += uint(tile_stride);
    if (group_steps_per_block > 1) {
      group_step_counter++;
      if (group_step_counter != group_steps_per_block) {
        return;
      }
      group_step_counter = 0;
    }
    scale_index += uint(group_stride);
    zero_point_index += zero_point_group_stride;
  }
}
