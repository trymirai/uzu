// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(bool, has_soft_cap)
UZU_PARAM(uint, length_total)
UZU_PARAM(float, scale)
UZU_PARAM(float, soft_cap)
UZU_BUFFER(logits)

[shader("compute")][numthreads(256, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    let i = gid.x;
    if (i >= param_length_total()) {
        return;
    }
    var v = load_logits(i) * param_scale();
    if (param_has_soft_cap()) {
        v = tanh(v / param_soft_cap()) * param_soft_cap();
    }
    store_logits(i, v);
}
