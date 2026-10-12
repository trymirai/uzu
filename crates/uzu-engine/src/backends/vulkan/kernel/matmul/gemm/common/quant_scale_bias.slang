import definitions;
import common.matmul;
import common.storage_address;
import matmul.common.loader;
import matmul.common.simdgroup_multiply_accumulate;

// Counterpart of Metal gemm/common/quant_scale_bias.h: stages a THREADGROUP_TILE_ROWS x THREADGROUP_TILE_COLS tile of
// transposed U4/U8 B (rows are output columns, columns K), each invocation dequantizing READS_PER_THREAD consecutive packs
// of one row with that row's scale and bias for the current group. Metal's template parameters are constructor
// arguments, constants at every call site; its static_asserts (columns divide the group, 4 or 8 bits) are the
// QuantizedGemm preconditions.
public struct QuantizedBlockLoaderScaleBias<T : IModelFloat> : IThreadgroupLoader<T> {
  uint observed_fields;
  int threadgroup_tile_rows;
  int bits;
  int pack_factor;
  int bytes_per_pack;
  int reads_per_thread;
  int group_steps_per_block;
  bool tile_has_idle_threads;

  int tile_stride;
  int group_step_counter;
  int group_stride;

  int tile_row_index;
  int tile_col_index;

  uint dst;
  Ptr<uint8_t, Access.Read> src;
  Ptr<T, Access.Read> scales;
  Ptr<T, Access.Read> biases;
  bool signed_codes;

  public __init(
    Ptr<uint8_t, Access.Read> src_,
    Ptr<T, Access.Read> scales_,
    Ptr<T, Access.Read> biases_,
    bool signed_codes_,
    int src_leading_dim_,
    int params_group_stride_,
    int params_output_stride_,
    int tile_rows,
    int tile_cols,
    int destination_leading_dimension,
    int threadgroup_size,
    int group_size,
    int bits_,
    ThreadContext thread_context,
  ) {
    observed_fields = 0x0000ffffu;
    threadgroup_tile_rows = tile_rows;
    bits = bits_;
    pack_factor = 8 / bits_;
    bytes_per_pack = 1;
    let tile_cols_packed = tile_cols / pack_factor;
    tile_has_idle_threads = tile_cols_packed * tile_rows < threadgroup_size;
    reads_per_thread = tile_has_idle_threads ? 1 : (tile_cols_packed * tile_rows) / threadgroup_size;
    group_steps_per_block = group_size / tile_cols;

    tile_stride = tile_cols_packed * bytes_per_pack;
    group_step_counter = 0;
    group_stride = params_group_stride_;
    let thread_index = int(thread_context.thread_index);
    tile_row_index = reads_per_thread * thread_index / tile_cols_packed;
    tile_col_index = (reads_per_thread * thread_index) % tile_cols_packed;
    dst = uint(tile_row_index * destination_leading_dimension + tile_col_index * pack_factor);
    src = element_pointer(
      src_,
      uint2(
        uint(tile_row_index) * uint(src_leading_dim_) * uint(bytes_per_pack) / uint(pack_factor)
          + uint(tile_col_index * bytes_per_pack),
        0
      )
    );
    scales = element_pointer(scales_, element_index(uint(tile_row_index), uint(params_output_stride_), 0));
    biases = element_pointer(biases_, element_index(uint(tile_row_index), uint(params_output_stride_), 0));
    signed_codes = signed_codes_;
  }

  public uint exponent_fields() { return observed_fields; }

  [mutating]
  public void load_unsafe<S : IThreadgroupStorage<T>>() {
    if (tile_has_idle_threads && tile_row_index >= threadgroup_tile_rows) {
      return;
    }

    let scale = scales[0].toFloat();
    let bias = biases[0].toFloat();
    observed_fields = matmul_merge_fields(observed_fields, matmul_quantized_fields<T>(scale, bias, uint(bits)));
    for (int i = 0; i < reads_per_thread; i++) {
      observed_fields = dequantize<T, S>(
        src, uint(i * bytes_per_pack), scale, bias, dst + uint(i * pack_factor), pack_factor, bits, signed_codes, observed_fields
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
    let scale = scales[0].toFloat();
    let bias = biases[0].toFloat();
    observed_fields = matmul_merge_fields(observed_fields, matmul_quantized_fields<T>(scale, bias, uint(bits)));
    for (int i = 0; i < reads_per_thread; i++) {
      let pack_index = tile_col_index + i;
      if (pack_index < valid_packs) {
        observed_fields = dequantize<T, S>(
          src, uint(i * bytes_per_pack), scale, bias, dst + uint(i * pack_factor), pack_factor, bits, signed_codes, observed_fields
        );
        if (pack_index == valid_packs - 1) {
          let remaining = valid_cols - pack_index * pack_factor;
          for (int lane = remaining; lane < pack_factor; ++lane) {
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
  }
}
