import definitions;
import common.workgroup_reduction;

static const uint QKV_NORM_THREADS = 128;

// FP32 RMS normalization of head_count heads from head head_offset of each row, one workgroup per head, other elements
// untouched. Without scales the result rounds to OutputT; with them, full_layer multiplies by scale + scale_offset in
// FP32, otherwise both round to OutputT and so does their product.
[[Variants("InputT", "float, bf16")]]
[[Variants("ScaleT", "float, bf16")]]
[[Variants("OutputT", "float, bf16")]]
[[Variants("AccumT", "float")]]
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
    let x = in_place_load(qkvg_input, qkvg_output, row + i, in_place);
    squares += x * x;
  }
  let rms_inv = rms_inverse<QKV_NORM_THREADS>(squares, head_dim, epsilon);
  // Each invocation reads back only the elements it read above, before it overwrites them.
  for (uint i = tid; i < head_dim; i += QKV_NORM_THREADS) {
    let x = in_place_load(qkvg_input, qkvg_output, row + i, in_place);
    var value = x * rms_inv;
    if (has_scales) {
      let scale = scales[i].toFloat() + scale_offset;
      if (full_layer) {
        value *= scale;
      } else {
        value = OutputT.fromFloat(value).toFloat() * OutputT.fromFloat(scale).toFloat();
      }
    }
    qkvg_output[row + i] = OutputT.fromFloat(value);
  }
}
