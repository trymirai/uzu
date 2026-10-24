import definitions;
import common.activation;
import generated.activation_type;

// FP32 math rounded to T at the store; IDENTITY copies the stored bits.
[[Kernel]] [[Public]] func Activation<T: IStorageFloat>(
  [[Optional("!in_place")]] Ptr<T, Access.Read> input,
  Ptr<T, Access.ReadWrite> output,
  uint n,
  [[PipelineVariants]] ActivationType act_type,
  [[Axis("n", "256")]] uint position,
  [[Specialize]] bool in_place,
) {
  T value;
  if (in_place) {
    value = output[position];
  } else {
    value = input[position];
  }
  if (act_type != ActivationType.IDENTITY) {
    value = T.fromFloat(activate(value.toFloat(), act_type));
  }
  output[position] = value;
}
