import definitions;
import common.activation;
import generated.activation_type;

[[Variants("T", "float, half, bf16")]]
[[Kernel]] [[Public]] func Activation<T: IStorageFloat>(
  [[Optional("!in_place")]] Ptr<T, Access.Read> input,
  Ptr<T, Access.ReadWrite> output,
  uint n,
  [[PipelineVariants]] ActivationType act_type,
  float activation_alpha,
  [[Axis("n", "256")]] uint position,
  [[Specialize]] bool in_place,
) {
  output[position] = T.fromFloat(activate(in_place_load(input, output, position, in_place), act_type, activation_alpha));
}
