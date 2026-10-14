// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(bool, has_bias)
UZU_PARAM(uint, in_proj_stride)
UZU_PARAM(uint, kernel_size)
UZU_PARAM(uint, model_dim)
UZU_PARAM(bool, state_in_place)
UZU_PARAM(uint, state_stride)
UZU_PARAM(uint, suffix_len)
UZU_BUFFER(b)
UZU_BUFFER(in_proj)
UZU_BUFFER(next_state)
UZU_BUFFER(out)
UZU_BUFFER(state)
UZU_BUFFER(w)

[shader("compute")][numthreads(128, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    let channel = gid.x;
    if (channel >= param_model_dim()) {
        return;
    }
    for (var token = 0u; token < param_suffix_len(); token++) {
        let offset = token * param_in_proj_stride() + channel;
        let x = load_in_proj(offset) * load_in_proj(offset + 2u * param_model_dim());
        var value = 0.0;
        if (param_has_bias()) {
            value = load_b(channel);
        }
        for (var tap = 0u; tap + 1u < param_kernel_size(); tap++) {
            var sample = load_state(channel * param_state_stride() + tap);
            if (token > 0u || param_state_in_place()) {
                sample = load_next_state(channel * param_state_stride() + tap);
            }
            value += load_w(channel * param_kernel_size() + tap) * sample;
        }
        value += load_w(channel * param_kernel_size() + param_kernel_size() - 1u) * x;
        store_out(token * param_model_dim() + channel, value * load_in_proj(offset + param_model_dim()));
        for (var tap = 0u; tap + 2u < param_kernel_size(); tap++) {
            var sample = load_state(channel * param_state_stride() + tap + 1u);
            if (token > 0u || param_state_in_place()) {
                sample = load_next_state(channel * param_state_stride() + tap + 1u);
            }
            store_next_state(channel * param_state_stride() + tap, sample);
        }
        if (param_kernel_size() > 1u) {
            store_next_state(channel * param_state_stride() + param_kernel_size() - 2u, x);
        }
    }
}
