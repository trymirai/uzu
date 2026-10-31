import definitions;
import common.activation;

// Multiplies each output element in place by the sigmoid of its gate, read from rows of gate_row_stride.
[[Variants("T", "float, bf16")]]
[[Kernel]] [[Public]]
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
  output[index] = T.fromFloat(output[index].toFloat() * sigmoid(gate[batch_idx * gate_row_stride + gate_idx].toFloat()));
}
