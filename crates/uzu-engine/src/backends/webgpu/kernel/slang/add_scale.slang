// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, length_total)
UZU_PARAM(uint, num_cols)
UZU_PARAM(float, scale)
UZU_BUFFER(bias)
UZU_BUFFER(input)
UZU_BUFFER(output)

[shader("compute")][numthreads(256, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    let i = gid.x;
    if (i >= param_length_total()) {
        return;
    }
    store_output(i, (load_input(i) + load_bias(i % param_num_cols())) * param_scale());
}
