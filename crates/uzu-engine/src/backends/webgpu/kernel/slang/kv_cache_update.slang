// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_ARRAY(uint, copies)
#define copy_word(i) array_copies(i)
UZU_PARAM(uint, copy_count)
UZU_PARAM(uint, element_dim)
UZU_BUFFER(in_place_keys)
UZU_BUFFER(in_place_values)

[shader("compute")][numthreads(256, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    let element = gid.x;
    if (element >= param_element_dim()) {
        return;
    }
    for (var i = 0u; i < param_copy_count(); i++) {
        let source = copy_word(i * 2u) * param_element_dim() + element;
        let destination = copy_word(i * 2u + 1u) * param_element_dim() + element;
        store_in_place_keys(destination, load_in_place_keys(source));
        store_in_place_values(destination, load_in_place_values(source));
    }
}
