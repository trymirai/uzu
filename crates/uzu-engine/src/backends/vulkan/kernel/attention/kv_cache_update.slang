import definitions;
import generated.kv_cache_update;

static const uint KV_CACHE_UPDATE_GROUP = 256;

// Copies, in the keys and in the values, rows of element_dim elements from row `source` to row `destination` of each of
// the first copy_count copies, as the CPU: elements of one index of one plane form an independent column, so one thread
// per column applies every copy in order, and chained copies, swaps and repeated destinations see the copies before them.
// Keys and values, written ranges that must not alias, take separate threads, halving each thread's chain of dependent
// copies. Elements move as stored, bit for bit. Without copies or elements nothing is dispatched.
[[Kernel]] [[Public]]
[[Precondition("encode", "copy_count as usize <= copies.len()")]]
func KVCacheUpdate<T: IModelFloat>(
  Ptr<T, Access.ReadWrite> in_place_keys,
  Ptr<T, Access.ReadWrite> in_place_values,
  [[HostSlice]] Ptr<Copy, Access.Read> copies,
  uint copy_count,
  uint element_dim,
  [[Groups("element_dim.div_ceil(256) * copy_count.min(1)")]] uint group,
  [[Groups("2")]] uint plane,
  [[Threads("256")]] uint thread,
) {
  static_assert(KV_CACHE_UPDATE_GROUP == 256, "one column per thread of Threads(256), Groups(div_ceil(256))");
  let element = group * KV_CACHE_UPDATE_GROUP + thread;
  if (element >= element_dim) {
    return;
  }
  let cache = plane == 0 ? in_place_keys : in_place_values;
  for (uint index = 0; index < copy_count; index++) {
    let copy = copies[index];
    let source = copy.source * element_dim + element;
    let destination = copy.destination * element_dim + element;
    cache[destination] = cache[source];
  }
}
