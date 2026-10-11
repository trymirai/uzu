import definitions;

[[Kernel]] [[Test]] func TestKernelAxisFloat(
  Ptr<float, Access.Read> input_0,
  Ptr<float, Access.Read> input_1,
  Ptr<float, Access.ReadWrite> output,
  uint size,
  [[Axis("size", "64")]] uint idx,
) {
  output[idx] = input_0[idx] + input_1[idx];
}
