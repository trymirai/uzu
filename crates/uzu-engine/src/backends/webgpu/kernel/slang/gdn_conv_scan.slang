// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, conv_dim)
UZU_PARAM(bool, has_bias)
UZU_PARAM(uint, kernel_size)
UZU_PARAM(uint, out_stride)
UZU_PARAM(uint, row_stride)
UZU_PARAM(uint, state_stride)
UZU_PARAM(uint, suffix_len)
UZU_BUFFER(bias)
UZU_BUFFER(conv_padded)
UZU_BUFFER(conv_weight)
UZU_BUFFER(in_proj)
UZU_BUFFER(state_out)

[shader("compute")][numthreads(128, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    let channel = gid.x;
    if (channel >= param_conv_dim()) {
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
        var sum = 0.0;
        if (param_has_bias()) {
            sum = load_bias(channel);
        }
        for (var tap = 0u; tap < param_kernel_size(); tap++) {
            sum += load_conv_weight(channel * param_kernel_size() + tap) *
                   load_conv_padded((token + tap) * param_row_stride() + channel);
        }
        store_in_proj(token * param_out_stride() + channel, activate(sum, 0u, 1.0));
    }
    if (tile == 0u) for (var tap = 0u; tap < param_state_stride(); tap++) {
        store_state_out(channel * param_state_stride() + tap,
                        load_conv_padded((param_suffix_len() + tap) * param_row_stride() + channel));
    }
}
