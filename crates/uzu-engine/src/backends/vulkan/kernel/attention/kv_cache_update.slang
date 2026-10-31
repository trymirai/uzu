import definitions;
import generated.kv_cache_update;

// Applies the first copy_count row copies, in order, to the key and value codes and their scales, bit for bit: rows of
// num_kv_heads heads of head_dim codes, read as words, and of num_kv_heads scales. One invocation per word column, the
// first num_kv_heads of them also per scale column, applies every copy, so chained copies, swaps and repeated
// destinations see the copies before them.
[[Kernel]] [[Public]]
[[Precondition("new", "head_dim.is_multiple_of(4) && backend_opaque_layout == 0")]]
[[Precondition("encode", "copy_count as usize <= copies.len()")]]
func KVCacheUpdate(
  Ptr<uint, Access.ReadWrite> in_place_k_codes,
  Ptr<uint, Access.ReadWrite> in_place_v_codes,
  Ptr<uint16_t, Access.ReadWrite> in_place_k_scales,
  Ptr<uint16_t, Access.ReadWrite> in_place_v_scales,
  [[Constant]] Ptr<Copy, Access.Read> copies,
  uint copy_count,
  [[Axis("num_kv_heads * head_dim / 4", "256")]] uint word,
  [[Specialize]] uint head_dim,
  [[Specialize]] uint num_kv_heads,
  [[Specialize]] uint backend_opaque_layout,
) {
  let row_words = num_kv_heads * head_dim / 4;
  for (uint index = 0; index < copy_count; index++) {
    let copy = copies[index];
    let source = copy.source * row_words + word;
    let destination = copy.destination * row_words + word;
    in_place_k_codes[destination] = in_place_k_codes[source];
    in_place_v_codes[destination] = in_place_v_codes[source];
    if (word < num_kv_heads) {
      let scale_source = copy.source * num_kv_heads + word;
      let scale_destination = copy.destination * num_kv_heads + word;
      in_place_k_scales[scale_destination] = in_place_k_scales[scale_source];
      in_place_v_scales[scale_destination] = in_place_v_scales[scale_source];
    }
  }
}
