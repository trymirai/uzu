import definitions;
import common.activation;
import common.workgroup_reduction;

static const uint QKV_NORM_THREADS = 128;

// An input element, from the output in place; a branch, so an absent input is never read.
float qkv_norm_load<InputT: IStorageFloat, OutputT: IStorageFloat>(
  Ptr<InputT, Access.Read> input,
  Ptr<OutputT, Access.ReadWrite> output,
  uint index,
  bool in_place
) {
  if (in_place) {
    return output[index].toFloat();
  }
  return input[index].toFloat();
}

// RMS normalization of head_count heads of head_dim elements from head head_offset of each row of input_row_stride,
// one workgroup per head of a row, leaving every other element of the row untouched. The squares, their sum, mean plus
// epsilon and reciprocal square root follow the CPU's FP32 stages with subnormals kept; normalization is FP32. Without
// scales the normalized value is rounded to OutputT. With them the scale plus scale_offset multiplies it in FP32 for
// full_layer, otherwise both are rounded to OutputT and their product rounded again. Elementwise products and sums are
// exact FP32 roundings, subnormals kept. In place requires InputT == OutputT, the model's only use: the CPU reinterprets
// the output as InputT otherwise. Overlapping rows of active heads would race, so each row's stride must hold the heads
// up to the selection's end; the bound is in quotients, free of overflow, and empty selections pass as on the CPU.
[[Kernel]] [[Public]]
[[Precondition("new", "!in_place || InputT == OutputT")]]
[[Precondition("encode", "batch_size == 0 || head_count == 0 || head_dim == 0 || head_offset <= input_row_stride / head_dim && head_count <= input_row_stride / head_dim - head_offset")]]
func QKVNorm<InputT: IModelFloat, ScaleT: IModelFloat, OutputT: IModelFloat, AccumT: IFloat32>(
  [[Optional("!in_place")]] Ptr<InputT, Access.Read> qkvg_input,
  [[Optional("has_scales")]] Ptr<ScaleT, Access.Read> scales,
  Ptr<OutputT, Access.ReadWrite> qkvg_output,
  uint batch_size,
  uint input_row_stride,
  uint head_dim,
  float epsilon,
  float scale_offset,
  uint head_offset,
  uint head_count,
  bool full_layer,
  [[Groups("if head_dim == 0 { 0 } else { batch_size }")]] uint batch,
  [[Groups("head_count")]] uint head,
  [[Threads("128")]] uint tid,
  [[Specialize]] bool in_place,
  [[Specialize]] bool has_scales,
) {
  static_assert(QKV_NORM_THREADS == 128, "one workgroup of Threads(128) per head");
  let row = batch * input_row_stride + (head_offset + head) * head_dim;
  float squares = 0.0;
  for (uint i = tid; i < head_dim; i += QKV_NORM_THREADS) {
    let x = qkv_norm_load(qkvg_input, qkvg_output, row + i, in_place);
    squares = accumulate_squares(squares, x);
  }
  let rms_inv = rms_inverse<QKV_NORM_THREADS>(squares, head_dim, epsilon);
  // Each invocation reads back only the elements it read above, before it overwrites them.
  for (uint i = tid; i < head_dim; i += QKV_NORM_THREADS) {
    let x = qkv_norm_load(qkvg_input, qkvg_output, row + i, in_place);
    var value = exact_product(x, rms_inv);
    if (has_scales) {
      let scale = exact_sum(scales[i].toFloat(), scale_offset, false);
      if (full_layer) {
        value = exact_product(value, scale);
      } else {
        value = exact_product(OutputT.fromFloat(value).toFloat(), OutputT.fromFloat(scale).toFloat());
      }
    }
    qkvg_output[row + i] = OutputT.fromFloat(value);
  }
}
