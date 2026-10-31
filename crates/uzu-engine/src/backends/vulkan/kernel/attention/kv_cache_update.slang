import definitions;
import generated.kv_cache_update;

// Applies the first copy_count row copies, in order, to the keys and the values, bit for bit. One thread per column of
// one plane applies every copy, so chained copies, swaps and repeated destinations see the copies before them.
[[Variants("T", "float, bf16")]]
[[Kernel]] [[Public]]
[[Precondition("encode", "copy_count as usize <= copies.len()")]]
func KVCacheUpdate<T: IModelFloat>(
  Ptr<T, Access.ReadWrite> in_place_keys,
  Ptr<T, Access.ReadWrite> in_place_values,
  [[Constant]] Ptr<Copy, Access.Read> copies,
  uint copy_count,
  uint element_dim,
  [[Axis("element_dim", "256")]] uint element,
  [[Axis("2", "1")]] uint plane,
) {
  let cache = plane == 0 ? in_place_keys : in_place_values;
  for (uint index = 0; index < copy_count; index++) {
    let copy = copies[index];
    let source = copy.source * element_dim + element;
    let destination = copy.destination * element_dim + element;
    cache[destination] = cache[source];
  }
}
