// uzu-widths: 32, 64, 128
// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, HEAD_K_DIM)
UZU_PARAM(uint, head_v_dim)
UZU_PARAM(uint, key_dim)
UZU_PARAM(float, norm_epsilon)
UZU_PARAM(uint, num_k_heads)
UZU_PARAM(uint, num_v_heads)
UZU_PARAM(uint, value_dim)
UZU_PARAM(uint, width)
UZU_BUFFER(a_log)
UZU_BUFFER(dt_bias)
UZU_BUFFER(in_proj)
UZU_BUFFER(norm_weight)
UZU_BUFFER(out)
UZU_BUFFER(state)

// Each subgroup owns one value-state row at a time. Neighboring lanes access
// adjacent key components, matching the native Metal kernel's state layout.
groupshared float outputs[256];
groupshared float partialSquares[256 / WIDTH];
[shader("compute")][numthreads(256, 1, 1)] void main(uint lane : SV_WaveLaneIndex, uint wave : SV_WaveIndex,
                                                     uint3 group : SV_GroupID) {
    let hv = group.x;
    let hk = hv / (param_num_v_heads() / param_num_k_heads());
    let convDim = 2u * param_key_dim() + param_value_dim();
    float q[256 / WIDTH];
    float k[256 / WIDTH];
    float qSquares = 0.0;
    float kSquares = 0.0;
    [ForceUnroll] for (uint part = 0; part < 256 / WIDTH; part++) {
        let dk = lane + part * WIDTH;
        q[part] = dk < param_HEAD_K_DIM() ? load_in_proj(hk * param_HEAD_K_DIM() + dk) : 0.0;
        k[part] =
            dk < param_HEAD_K_DIM() ? load_in_proj(param_key_dim() + hk * param_HEAD_K_DIM() + dk) : 0.0;
        qSquares += q[part] * q[part];
        kSquares += k[part] * k[part];
    }
    let qScale = rsqrt(WaveActiveSum(qSquares) + 1e-6) * rsqrt(float(param_HEAD_K_DIM()));
    let kScale = rsqrt(WaveActiveSum(kSquares) + 1e-6);
    float kqPartial = 0.0;
    [ForceUnroll] for (uint part = 0; part < 256 / WIDTH; part++) {
        q[part] *= qScale;
        k[part] *= kScale;
        kqPartial += q[part] * k[part];
    }
    let kq = WaveActiveSum(kqPartial);
    let beta = 1.0 / (1.0 + exp(-load_in_proj(convDim + param_value_dim() + hv)));
    let a = load_in_proj(convDim + param_value_dim() + param_num_v_heads() + hv) + load_dt_bias(hv);
    let softplus = a <= 20.0 ? log(1.0 + exp(a)) : a;
    let decay = exp(-exp(load_a_log(hv)) * softplus);
    float outputSquares = 0.0;
    for (uint rowBase = 0; rowBase < param_head_v_dim(); rowBase += 256 / WIDTH) {
        let dv = rowBase + wave;
        let stateBase = (hv * param_head_v_dim() + dv) * param_HEAD_K_DIM();
        float rowState[256 / WIDTH];
        float sq = 0.0;
        float sk = 0.0;
        [ForceUnroll] for (uint part = 0; part < 256 / WIDTH; part++) {
            let dk = lane + part * WIDTH;
            rowState[part] =
                dv < param_head_v_dim() && dk < param_HEAD_K_DIM() ? load_state(stateBase + dk) : 0.0;
            sq += rowState[part] * q[part];
            sk += rowState[part] * k[part];
        }
        let retrieved = decay * WaveActiveSum(sk);
        let projected = decay * WaveActiveSum(sq);
        let value =
            dv < param_head_v_dim() ? load_in_proj(2u * param_key_dim() + hv * param_head_v_dim() + dv) : 0.0;
        let delta = beta * (value - retrieved);
        let output = projected + delta * kq;
        if (dv < param_head_v_dim()) {
            [ForceUnroll] for (uint part = 0; part < 256 / WIDTH; part++) {
                let dk = lane + part * WIDTH;
                if (dk < param_HEAD_K_DIM())
                    store_state(stateBase + dk, decay * rowState[part] + k[part] * delta);
            }
            outputSquares += output * output;
            if (lane == 0u)
                outputs[dv] = output;
        }
    }
    if (lane == 0u)
        partialSquares[wave] = outputSquares;
    GroupMemoryBarrierWithGroupSync();
    let totalSquares = WaveActiveSum(lane < 256 / WIDTH ? partialSquares[lane] : 0.0);
    let inv = rsqrt(totalSquares / float(param_head_v_dim()) + param_norm_epsilon());
    let dv = wave * WIDTH + lane;
    float result = 0.0;
    if (dv < param_head_v_dim()) {
        let z = load_in_proj(convDim + hv * param_head_v_dim() + dv);
        result = outputs[dv] * inv * load_norm_weight(dv) * activate(z, 0u, 1.0);
    }
    if (dv < param_head_v_dim()) store_out(hv * param_head_v_dim() + dv, result);
}
