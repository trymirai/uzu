import definitions;
import common.matmul;
import common.quantized_value;
import common.storage_address;
import matmul.common.loader;
import matmul.common.simdgroup_multiply_accumulate;

// Counterpart of Metal gemm/common/quant_scale_zero_point.h: quant_scale_bias's staging with the bias formed per group
// as -scale zero_point in T arithmetic, the zero point read from its plane (a nibble for U4) or, for symmetric
// quantization, the midpoint 2^(bits - 1) without reading zero points. Metal's template parameters, SCALE_SYMMETRIC
// included, are constructor arguments, constants at every call site.
public struct QuantizedBlockLoaderScaleZeroPoint<T : IModelFloat> : IThreadgroupLoader<T> {
  uint observed_fields;
  int threadgroup_tile_rows;
  int bits;
  int pack_factor;
  int bytes_per_pack;
  int reads_per_thread;
  int group_steps_per_block;
  bool tile_has_idle_threads;
  bool scale_symmetric;

  int tile_stride;
  int group_step_counter;
  int group_stride;
  uint zero_point_group_stride;

  int tile_row_index;
  int tile_col_index;

  uint dst;
  Ptr<uint8_t, Access.Read> src;
  Ptr<T, Access.Read> scales;
  Ptr<uint8_t, Access.Read> zero_points_row_start;
  uint zero_point_index;
  bool signed_codes;

  public __init(
    Ptr<uint8_t, Access.Read> src_,
    Ptr<T, Access.Read> scales_,
    Ptr<uint8_t, Access.Read> zero_points_row_start_,
    bool signed_codes_,
    int src_leading_dim_,
    int params_group_stride_,
    int params_output_stride_,
    uint zero_point_output_stride_,
    uint zero_point_group_stride_,
    uint zero_point_index_,
    int tile_rows,
    int tile_cols,
    int destination_leading_dimension,
    int threadgroup_size,
    int group_size,
    int bits_,
    bool scale_symmetric_,
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
    scale_symmetric = scale_symmetric_;

    tile_stride = tile_cols_packed * bytes_per_pack;
    group_step_counter = 0;
    group_stride = params_group_stride_;
    zero_point_group_stride = zero_point_group_stride_;
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
    zero_points_row_start = zero_points_row_start_;
    zero_point_index = zero_point_index_ + uint(tile_row_index) * zero_point_output_stride_;
    signed_codes = signed_codes_;
  }

  void current_scale_bias(out float out_scale, out float out_bias) {
    uint zero_point_value;
    let scale_value = scales[0].toFloat();
    if (scale_symmetric) {
      zero_point_value = 1u << uint(bits - 1);
    } else if (bits == 4) {
      zero_point_value = read_nibble(zero_points_row_start, zero_point_index);
    } else {
      zero_point_value = uint(zero_points_row_start[zero_point_index]);
    }
    out_scale = scale_value;
    out_bias = round_to<T>(-scale_value * float(zero_point_value));
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
    float scale;
    float bias;
    current_scale_bias(scale, bias);
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
    if (!scale_symmetric) {
      zero_point_index += zero_point_group_stride;
    }
  }
}
