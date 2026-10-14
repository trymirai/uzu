// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, gate_dim)
UZU_PARAM(uint, gate_row_stride)
UZU_PARAM(uint, length_total)
UZU_BUFFER(gate)
UZU_BUFFER(output)

[shader("compute")][numthreads(256, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    let i = gid.x;
    if (i >= param_length_total()) {
        return;
    }
    let j = (i / param_gate_dim()) * param_gate_row_stride() + i % param_gate_dim();
    store_output(i, load_output(i) / (1.0 + exp(-load_gate(j))));
}
