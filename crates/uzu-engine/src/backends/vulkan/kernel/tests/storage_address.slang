import definitions;
import common.storage_address;

// Both kernels copy element idx of a block of `columns`-element rows with row stride `stride` (padding past `columns`
// untouched) through pointers rebuilt by element_pointer; size = rows * columns.

[[Kernel]] [[Test]] func TestStorageAddressCopy<T: IStorageFloat>(
  Ptr<T, Access.Read> input,
  Ptr<T, Access.ReadWrite> output,
  uint size,
  uint columns,
  uint stride,
  [[Axis("size", "64")]] uint idx,
) {
  let element = element_index(idx / columns, stride, idx % columns);
  element_pointer(output, element)[0] = element_pointer(input, element)[0];
}

// The same copy for uint, plus pure arithmetic on synthetic words that never form a pointer: for each of the first
// `count` (<= size) threads, cases holds row, stride, column, element bytes, base low and base high, and results gets
// the element index, byte offset and base + byte offset as (low, high) pairs. Thread 0 stores NumWorkgroups.x at
// results[6 * count].
[[Kernel]] [[Test]] func TestStorageAddressWords(
  Ptr<uint, Access.Read> input,
  Ptr<uint, Access.ReadWrite> output,
  Ptr<uint, Access.Read> cases,
  Ptr<uint, Access.ReadWrite> results,
  uint size,
  uint columns,
  uint stride,
  uint count,
  [[Axis("size", "64")]] uint idx,
) {
  let element = element_index(idx / columns, stride, idx % columns);
  element_pointer(output, element)[0] = element_pointer(input, element)[0];
  if (idx < count) {
    let c = 6 * idx;
    let index = element_index(cases[c], cases[c + 1], cases[c + 2]);
    let bytes = byte_offset(index, cases[c + 3]);
    let address = add_address(uint2(cases[c + 4], cases[c + 5]), bytes);
    results[c] = index.x;
    results[c + 1] = index.y;
    results[c + 2] = bytes.x;
    results[c + 3] = bytes.y;
    results[c + 4] = address.x;
    results[c + 5] = address.y;
  }
  if (idx == 0) {
    let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    results[6 * count] = groups.x;
  }
}
