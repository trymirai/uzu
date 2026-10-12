import definitions;

[[SpecializationConstant]] [[ThreadgroupSize]]
const uint threadgroup_size;

[[Kernel]] func test_kernel_group_thread<T: __BuiltinFloatingPointType>(
  Ptr<T, Access.Read> input_0,
  Ptr<T, Access.Read> input_1,
  Ptr<T, Access.ReadWrite> output,
  uint size,
  [[Groups("size.div_ceil(threadgroup_size)")]] uint group_idx,
  [[Threads("threadgroup_size")]] uint thread_idx,
) {
  uint idx = group_idx * threadgroup_size + thread_idx;

  if (idx < size) {
    output[idx] = input_0[idx] + input_1[idx];
  }
}

[[Kernel]] func test_kernel_axis<T: __BuiltinFloatingPointType>(
  Ptr<T, Access.Read> input_0,
  Ptr<T, Access.Read> input_1,
  Ptr<T, Access.ReadWrite> output,
  uint size,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  output[idx] = input_0[idx] + input_1[idx];
}

[[Kernel]] func test_kernel_axis_float(
  Ptr<float, Access.Read> input_0,
  Ptr<float, Access.Read> input_1,
  Ptr<float, Access.ReadWrite> output,
  uint size,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  output[idx] = input_0[idx] + input_1[idx];
}

[[Kernel]] func test_kernel_multispecialize<I: __BuiltinFloatingPointType, O: __BuiltinFloatingPointType>(
  Ptr<I, Access.Read> input,
  Ptr<O, Access.ReadWrite> output,
  uint size,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  output[idx] = O(input[idx].toFloat());
}
