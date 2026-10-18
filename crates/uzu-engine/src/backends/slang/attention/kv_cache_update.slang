import definitions;
import storage;
import generated.kv_cache_update;

[[Variants("T", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func KVCacheUpdate<T : ITensorElement>(
    RWStructuredBuffer<Atomic<uint>> in_place_keys,
    RWStructuredBuffer<Atomic<uint>> in_place_values,
    [[Constant]] StructuredBuffer<Copy> copies,
    uint copy_count,
    uint element_dim,
    [[Axis("element_dim", "256")]] uint element_idx,
) {
  for (uint i = 0; i < copy_count; ++i) {
    uint source = copies[i].source * element_dim + element_idx;
    uint destination = copies[i].destination * element_dim + element_idx;
    tensorStore<T>(in_place_keys, destination, tensorLoad<T>(in_place_keys, source));
    tensorStore<T>(in_place_values, destination, tensorLoad<T>(in_place_values, source));
  }
}

[[Public]] [[Kernel]] func KVCacheUpdateInt8(
    RWStructuredBuffer<int8_t> in_place_k_codes,
    RWStructuredBuffer<uint8_t> in_place_v_codes,
    RWStructuredBuffer<uint16_t> in_place_k_scales,
    RWStructuredBuffer<uint16_t> in_place_v_scales,
    [[Constant]] StructuredBuffer<Copy> copies,
    uint copy_count,
    [[Specialize]] uint head_dim,
    [[Specialize]] uint num_kv_heads,
    [[Axis("num_kv_heads* head_dim", "256")]] uint element,
) {
  // TODO
}
