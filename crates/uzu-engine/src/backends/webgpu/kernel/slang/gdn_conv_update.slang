// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, conv_dim)
UZU_PARAM(bool, has_bias)
UZU_PARAM(uint, kernel_size)
UZU_PARAM(uint, state_stride)
UZU_BUFFER(bias)
UZU_BUFFER(conv_weight)
UZU_BUFFER(in_out)
UZU_BUFFER(state)

[shader("compute")][numthreads(256, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    let channel = gid.x;
    if (channel >= param_conv_dim()) {
        return;
    }
    let base = channel * param_state_stride();
    let weight = channel * param_kernel_size();
    let x = load_in_out(channel);
    var sum = 0.0;
    if (param_has_bias()) {
        sum = load_bias(channel);
    }
    for (var tap = 0u; tap + 1u < param_kernel_size(); tap++) {
        sum += load_state(base + tap) * load_conv_weight(weight + tap);
    }
    sum += x * load_conv_weight(weight + param_kernel_size() - 1u);
    store_in_out(channel, activate(sum, 0u, 1.0));
    for (var tap = 1u; tap + 1u < param_kernel_size(); tap++) {
        store_state(base + tap - 1u, load_state(base + tap));
    }
    if (param_kernel_size() > 1u) {
        store_state(base + param_kernel_size() - 2u, x);
    }
}
