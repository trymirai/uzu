// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, input_length)
UZU_PARAM(uint, suffix_repetition_length)
UZU_BUFFER(context_ring)
UZU_BUFFER(input)

[shader("compute")][numthreads(256, 1, 1)] void main(uint lane : SV_GroupIndex) {
    let capacity = param_suffix_repetition_length();
    if (capacity == 0u) {
        return;
    }
    let offset = word_context_ring(0u);
    let length = word_context_ring(1u);
    GroupMemoryBarrierWithGroupSync();
    let skipped = param_input_length() - min(param_input_length(), capacity);
    for (var index = skipped + lane; index < param_input_length(); index += 256u) {
        put_context_ring(2u + (offset + length + index) % capacity, word_input(index));
    }
    DeviceMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        let total = length + param_input_length();
        put_context_ring(0u, (offset + total - min(total, capacity)) % capacity);
        put_context_ring(1u, min(total, capacity));
    }
}
