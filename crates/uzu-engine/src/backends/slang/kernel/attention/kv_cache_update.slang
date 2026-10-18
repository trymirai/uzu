import definitions;
import storage;
import generated.kv_cache_update;

[[Public]] [[Kernel]] func KVCacheUpdate(
  RWStructuredBuffer<Atomic<uint32_t>> in_place_k_codes,
  RWStructuredBuffer<Atomic<uint32_t>> in_place_v_codes,
  RWStructuredBuffer<Atomic<uint32_t>> in_place_k_scales,
  RWStructuredBuffer<Atomic<uint32_t>> in_place_v_scales,
  [[Constant]] StructuredBuffer<Copy> copies,
  uint32_t copy_count,
  [[Specialize]] uint32_t head_dim,
  [[Specialize]] uint32_t num_kv_heads,
  [[Specialize]] uint32_t backend_opaque_layout,
  [[Threads("1")]] uint32_t lane,
) {
  // Copy sequentially so later copies can read rows written by earlier copies.
  uint32_t row_width = num_kv_heads * head_dim;
  for (uint32_t i = 0; i < copy_count; ++i) {
    for (uint32_t element = 0; element < row_width; ++element) {
      uint32_t source = copies[i].source * row_width + element;
      uint32_t destination = copies[i].destination * row_width + element;
      byteStore(in_place_k_codes, destination, byteLoad(in_place_k_codes, source));
      byteStore(in_place_v_codes, destination, byteLoad(in_place_v_codes, source));
    }
    for (uint32_t head = 0; head < num_kv_heads; ++head) {
      uint32_t source = copies[i].source * num_kv_heads + head;
      uint32_t destination = copies[i].destination * num_kv_heads + head;
      tensorStore<BFloat16>(in_place_k_scales, destination, tensorLoad<BFloat16>(in_place_k_scales, source));
      tensorStore<BFloat16>(in_place_v_scales, destination, tensorLoad<BFloat16>(in_place_v_scales, source));
    }
  }
}
