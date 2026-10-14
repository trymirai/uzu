// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, conv_dim)
UZU_PARAM(bool, has_bias)
UZU_PARAM(uint, kernel_size)
UZU_PARAM(uint, total_proj_dim)
UZU_BUFFER(base_state)
UZU_BUFFER(bias)
UZU_BUFFER(conv_weight)
UZU_BUFFER(in_proj)
UZU_BUFFER(out_proj)
UZU_BUFFER(parents)
UZU_BUFFER(suffix_state)

[shader("compute")][numthreads(128, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    let channel = gid.x;
    let node = gid.y;
    if (channel >= param_total_proj_dim()) {
        return;
    }
    let destination = node * param_total_proj_dim() + channel;
    if (channel >= param_conv_dim()) {
        store_out_proj(destination, load_in_proj(destination));
        return;
    }
    let stateStride = param_kernel_size() - 1u;
    var source = int(node);
    var sum = 0.0;
    if (param_has_bias()) {
        sum = load_bias(channel);
    }
    for (var history = 0u; history < param_kernel_size(); history++) {
        var sample = 0.0;
        if (source >= 0) {
            sample = load_in_proj(uint(source) * param_total_proj_dim() + channel);
        } else {
            sample = load_base_state(channel * stateStride + stateStride - uint(-source));
        }
        sum += sample * load_conv_weight(channel * param_kernel_size() + param_kernel_size() - 1u - history);
        if (history < stateStride) {
            store_suffix_state((node * param_conv_dim() + channel) * stateStride + stateStride - 1u - history,
                               sample);
        }
        if (source >= 0) {
            source = asint(word_parents(uint(source)));
        } else {
            source--;
        }
    }
    store_out_proj(destination, activate(sum, 0u, 1.0));
}
