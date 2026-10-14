// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, act_type)
UZU_PARAM(uint, length_total)
UZU_BUFFER(input)
UZU_BUFFER(output)

[shader("compute")][numthreads(256, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    let i = gid.x;
    if (i >= param_length_total()) {
        return;
    }
    store_output(i, activate(load_input(i), param_act_type(), 1.0));
}
