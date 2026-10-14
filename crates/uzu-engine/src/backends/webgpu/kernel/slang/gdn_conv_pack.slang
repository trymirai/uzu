// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, num_channels)
UZU_PARAM(uint, row_stride)
UZU_PARAM(uint, state_stride)
UZU_PARAM(uint, suffix_len)
UZU_BUFFER(padded)
UZU_BUFFER(state_in)
UZU_BUFFER(x)

[shader("compute")][numthreads(256, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    let index = gid.x;
    if (index >= (param_state_stride() + param_suffix_len()) * param_num_channels()) {
        return;
    }
    let channel = index % param_num_channels();
    let row = index / param_num_channels();
    var value = 0.0;
    if (row < param_state_stride()) {
        value = load_state_in(channel * param_state_stride() + row);
    } else {
        value = load_x((row - param_state_stride()) * param_row_stride() + channel);
    }
    store_padded(row * param_row_stride() + channel, value);
}
