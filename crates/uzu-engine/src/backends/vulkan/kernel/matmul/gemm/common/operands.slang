import definitions;
import common.matmul;
import common.quantized_value;
import common.storage_address;
import generated.gemm;
import generated.quantization_method;
import matmul.common.loader;
import matmul.common.simdgroup_multiply_accumulate;

// Counterpart of Metal gemm/common/operands.h and its staged B loaders: the right operand B as stored, N rows of K
// (Metal's TRANSPOSE_RIGHT), full precision or U4/U8 codes with their scheme; the loader that stages a block_n x block_k
// tile of it for the scalar core, rows shared_stride_b apart; and, as an IMatmulOperand, the CPU's reading of one
// element for matmul_replay. The left operand is always full precision, common.matmul's DenseOperand.
public interface IRightOperand<B> : IMatmulOperand {
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

  public typealias Loader = ThreadgroupLoader<B, true>;

  public float value(uint row, uint inner, uint base) {
    return dense[row * leading_dimension_b + inner].toFloat();
  }

  public Loader make_loader(
    uint block_col,
    int block_n,
    int block_k,
    int shared_stride_b,
    int threadgroup_threads,
    ThreadContext thread_context,
  ) {
    return ThreadgroupLoader<B, true>(
      element_pointer(dense, element_index(block_col, leading_dimension_b, 0)),
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

  public float value(uint row, uint inner, uint base) {
    return weights.value(row, inner, base);
  }

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

// Metal's quant_scale_bias.h and quant_scale_zero_point.h loaders as one: stages a block_n x block_k tile of B, each
// invocation dequantizing reads_per_thread consecutive packs of one row with that row's scale and bias for the current
// group, in T arithmetic. ScaleBias reads its bias; ScaleZeroPoint forms it per group as -scale zero_point in T, the
// zero point a nibble for U4; ScaleSymmetric uses the midpoint 2^(bits - 1). Metal's static_asserts (the tile's K
// divides the group, 4 or 8 bits) are the kernel's preconditions.
public struct QuantizedBlockLoader<T : IModelFloat> : IThreadgroupLoader<T> {
  uint observed_fields;
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
  Ptr<T, Access.Read> scales;
  Ptr<T, Access.Read> biases;
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
    observed_fields = 0x0000ffffu;
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
    src = element_pointer(right.codes, element_index(row, k / uint(pack_factor), uint(tile_col_index)));
    let params = element_index(row, right.scale_output_stride, 0);
    scales = element_pointer(right.scales, params);
    biases = element_pointer(right.biases, params);
    zero_points = right.zero_points;
    zero_point_index = row * right.zero_point_output_stride;
    signed_codes = right.signed_codes;
  }

  void current_scale_bias(out float scale, out float bias) {
    scale = scales[0].toFloat();
    if (method == QuantizationMethod.ScaleBias) {
      bias = biases[0].toFloat();
      return;
    }
    var zero_point = 1u << uint(bits - 1);
    if (method == QuantizationMethod.ScaleZeroPoint) {
      zero_point = bits == 4 ? read_nibble(zero_points, zero_point_index) : uint(zero_points[zero_point_index]);
    }
    bias = round_to<T>(-scale * float(zero_point));
  }

  public uint exponent_fields() { return observed_fields; }

  [mutating]
  public void load_unsafe<S : IThreadgroupStorage<T>>() {
    if (tile_has_idle_threads && tile_row_index >= threadgroup_tile_rows) {
      return;
    }
    float scale;
    float bias;
    current_scale_bias(scale, bias);
    observed_fields = matmul_merge_fields(observed_fields, matmul_quantized_fields<T>(scale, bias, uint(bits)));
    for (int i = 0; i < reads_per_thread; i++) {
      observed_fields = dequantize<T, S>(
        src, uint(i), scale, bias, dst + uint(i * pack_factor), pack_factor, bits, signed_codes, observed_fields
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
    observed_fields = matmul_merge_fields(observed_fields, matmul_quantized_fields<T>(scale, bias, uint(bits)));
    for (int i = 0; i < reads_per_thread; i++) {
      let pack_index = tile_col_index + i;
      if (pack_index < valid_packs) {
        observed_fields = dequantize<T, S>(
          src, uint(i), scale, bias, dst + uint(i * pack_factor), pack_factor, bits, signed_codes, observed_fields
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
    src = element_pointer(src, uint2(uint(tile_stride), 0));
    if (group_steps_per_block > 1) {
      group_step_counter++;
      if (group_step_counter != group_steps_per_block) {
        return;
      }
      group_step_counter = 0;
    }
    scales = element_pointer(scales, uint2(uint(group_stride), 0));
    biases = element_pointer(biases, uint2(uint(group_stride), 0));
    zero_point_index += zero_point_group_stride;
  }
}
