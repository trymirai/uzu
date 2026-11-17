import definitions;
import common.activation;

[[Variants("T", "bf16")]]
[[Kernel]] [[Public]]
func MoeCombine<T: IStorageFloat>(
  Ptr<T, Access.Read> expert_outputs,
  Ptr<float, Access.Read> expert_weights,
  Ptr<T, Access.Read> shared_output,
  Ptr<T, Access.Read> shared_gate,
  Ptr<T, Access.ReadWrite> output,
  uint model_dim,
  uint top_k,
  uint batch_size,
  [[Axis("model_dim", "BD")]] uint dim,
  [[Axis("batch_size", "1")]] uint token,
  [[Specialize]] uint BD,
) {
  let route = token * top_k;
  var sum = 0.0;
  for (uint pick = 0; pick < top_k; pick++) {
    sum = fma(expert_weights[route + pick], expert_outputs[(route + pick) * model_dim + dim].toFloat(), sum);
  }
  sum += shared_output[token * model_dim + dim].toFloat() * sigmoid(shared_gate[token].toFloat());
  output[token * model_dim + dim] = T.fromFloat(sum);
}
