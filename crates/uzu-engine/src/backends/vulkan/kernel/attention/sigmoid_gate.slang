import definitions;
import common.activation;

// Multiplies each output element in place by the sigmoid of its gate, read from rows of gate_row_stride, in the CPU's
// staging: the sigmoid 1 / (1 + e^-g) is an FP32 value of its own, SiLU of 1 with slope g, whose staged reciprocal keeps
// the subnormal sigmoids of the negative tail; its exact FP32 product with the output is rounded to T. The CPU's
// preconditions hold also for empty batches.
[[Kernel]] [[Public]]
[[Precondition("encode", "gate_dim > 0")]]
[[Precondition("encode", "gate_row_stride >= gate_dim")]]
func SigmoidGate<T: IModelFloat>(
  Ptr<T, Access.Read> gate,
  Ptr<T, Access.ReadWrite> output,
  uint gate_dim,
  uint batch_dim,
  uint gate_row_stride,
  [[Axis("gate_dim", "256")]] uint gate_idx,
  [[Axis("batch_dim", "1")]] uint batch_idx,
) {
  let index = batch_idx * gate_dim + gate_idx;
  let sigmoid = activate_silu_alpha<true>(1.0, gate[batch_idx * gate_row_stride + gate_idx].toFloat());
  output[index] = T.fromFloat(exact_product(output[index].toFloat(), sigmoid));
}
