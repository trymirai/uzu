// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(bool, has_bias)
UZU_PARAM(uint, in_proj_stride)
UZU_PARAM(uint, kernel_size)
UZU_PARAM(uint, model_dim)
UZU_PARAM(uint, state_stride)
UZU_PARAM(uint, suffix_len)
UZU_BUFFER(b)
UZU_BUFFER(in_proj)
UZU_BUFFER(out)
UZU_BUFFER(padded)
UZU_BUFFER(state_out)
UZU_BUFFER(w)

[shader("compute")][numthreads(128, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    let channel = gid.x;
    if (channel >= param_model_dim()) {
        return;
    }
    let tile = gid.y + gid.z * 65535u;
    // Tile zero also preserves the state-copy behavior for an empty suffix.
    if (tile != 0u && (param_suffix_len() == 0u || tile > (param_suffix_len() - 1u) / 4u)) return;
    let first = tile * 4u;
    [ForceUnroll]
    for (uint local = 0; local < 4u; local++) {
        if (local >= param_suffix_len() - first) break;
        let token = first + local;
        var value = 0.0;
        if (param_has_bias()) {
            value = load_b(channel);
        }
        for (var tap = 0u; tap < param_kernel_size(); tap++) {
            value += load_w(channel * param_kernel_size() + tap) *
                     load_padded((token + tap) * param_model_dim() + channel);
        }
        value *= load_in_proj(token * param_in_proj_stride() + param_model_dim() + channel);
        store_out(token * param_model_dim() + channel, value);
    }
    if (tile == 0u) for (var tap = 0u; tap + 1u < param_kernel_size(); tap++) {
        store_state_out(channel * param_state_stride() + tap,
                        load_padded((param_suffix_len() + tap) * param_model_dim() + channel));
    }
}
