import definitions;
import common.storage_address;

// Two kernels with a specialization of the same name and distinct effects, next to a plain one, in one linked module.

[[Kernel]] [[Test]] func TestSpecializationA(
  Ptr<uint, Access.Read> input,
  Ptr<uint, Access.ReadWrite> output,
  uint size,
  [[Axis("size", "64")]] uint idx,
  [[Specialize]] bool flag,
) {
  output[idx] = input[idx] + (flag ? 0x10u : 0x20u);
}

[[Kernel]] [[Test]] func TestSpecializationB(
  Ptr<uint, Access.Read> input,
  Ptr<uint, Access.ReadWrite> output,
  uint size,
  [[Axis("size", "64")]] uint idx,
  [[Specialize]] bool flag,
) {
  output[idx] = input[idx] + (flag ? 0x100u : 0x200u);
}

[[Kernel]] [[Test]] func TestSpecializationPlain(
  Ptr<uint, Access.Read> input,
  Ptr<uint, Access.ReadWrite> output,
  uint size,
  [[Axis("size", "64")]] uint idx,
) {
  output[idx] = input[idx] + 1u;
}

// The module owns TestSpecializationArray's state_size: it sizes a private FP32 row of max(state_size, 1) elements
// (zero-safe without select), which the wrapper passes instead of redeclaring. Each invocation widens its T row, each of
// q tokens updates the first state_size slots and stores their sum at the slot its bits select, and the row is stored
// back as T: a lowering guard, not a model kernel.
[[SpecializationConstant]] const uint __dsl_TestSpecializationArray_state_size = 64;

[[Kernel]] [[Test]] func TestSpecializationArray<T: IStorageFloat>(
  Ptr<T, Access.Read> input,
  Ptr<T, Access.ReadWrite> output,
  uint q,
  [[Specialize]] uint state_size,
  [[Groups("1")]] uint group,
  [[Threads("64")]] uint thread,
) {
  let invocation = group * 64 + thread;
  float row[(__dsl_TestSpecializationArray_state_size | (1u - ((__dsl_TestSpecializationArray_state_size | (0u - __dsl_TestSpecializationArray_state_size)) >> 31)))];
  let capacity = __dsl_TestSpecializationArray_state_size | (1u - ((__dsl_TestSpecializationArray_state_size | (0u - __dsl_TestSpecializationArray_state_size)) >> 31));
  for (uint i = 0; i < capacity; i++) {
    row[i] = element_pointer(input, element_index(invocation, capacity, i))[0].toFloat();
  }
  for (uint t = 0; t < q; t++) {
    let a = element_pointer(input, element_index(t, 1, 0))[0].toFloat();
    var acc = 0.0;
    for (uint i = 0; i < state_size; i++) {
      row[i] = row[i] * a + 1.0;
      acc += row[i];
    }
    row[asuint(acc) % capacity] = acc;
  }
  for (uint i = 0; i < capacity; i++) {
    element_pointer(output, element_index(invocation, capacity, i))[0] = T.fromFloat(row[i]);
  }
}
