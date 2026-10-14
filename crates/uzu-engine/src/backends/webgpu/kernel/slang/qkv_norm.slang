// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(float, epsilon)
UZU_PARAM(bool, full_layer)
UZU_PARAM(bool, has_scales)
UZU_PARAM(uint, head_count)
UZU_PARAM(uint, head_dim)
UZU_PARAM(uint, head_offset)
UZU_PARAM(uint, input_row_stride)
UZU_PARAM(float, scale_offset)
UZU_BUFFER(qkvg_input)
UZU_BUFFER(qkvg_output)
UZU_BUFFER(scales)

groupshared float sums[256];
[shader("compute")][numthreads(256, 1, 1)] void main(uint lane : SV_GroupIndex, uint3 gid : SV_GroupID) {
    let base = (gid.x / param_head_count()) * param_input_row_stride() +
               (param_head_offset() + gid.x % param_head_count()) * param_head_dim();
    var sum = 0.0;
    for (var i = lane; i < param_head_dim(); i += 256u) {
        let v = load_qkvg_input(base + i);
        sum += v * v;
    }
    sums[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    let rms = rsqrt(sums[0] / float(param_head_dim()) + param_epsilon());
    for (var i = lane; i < param_head_dim(); i += 256u) {
        var v = load_qkvg_input(base + i) * rms;
        if (param_has_scales()) {
            let scale = load_scales(i) + param_scale_offset();
            if (param_full_layer()) {
                v *= scale;
            } else {
                v = round_qkvg_output(v) * round_qkvg_output(scale);
            }
        }
        store_qkvg_output(base + i, v);
    }
}
