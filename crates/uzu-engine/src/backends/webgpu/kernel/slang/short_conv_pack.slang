// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, in_proj_stride)
UZU_PARAM(uint, model_dim)
UZU_PARAM(uint, state_stride)
UZU_PARAM(uint, suffix_len)
UZU_BUFFER(in_proj)
UZU_BUFFER(padded)
UZU_BUFFER(state_in)

[shader("compute")][numthreads(128, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    let channel = gid.x;
    if (channel >= param_model_dim()) {
        return;
    }
    let tile = gid.y + gid.z * 65535u;
    let rows = param_state_stride() + param_suffix_len();
    if (rows == 0u || tile > (rows - 1u) / 4u) return;
    let first = tile * 4u;
    [ForceUnroll]
    for (uint local = 0; local < 4u; local++) {
        if (local >= rows - first) break;
        let row = first + local;
        var value = 0.0;
        if (row < param_state_stride()) {
            value = load_state_in(channel * param_state_stride() + row);
        } else {
            let offset = (row - param_state_stride()) * param_in_proj_stride() + channel;
            value = load_in_proj(offset) * load_in_proj(offset + 2u * param_model_dim());
        }
        store_padded(row * param_model_dim() + channel, value);
    }
}
