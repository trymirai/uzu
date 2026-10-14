// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, HEAD_K_DIM)
UZU_PARAM(uint, key_dim)
UZU_PARAM(uint, num_k_heads)
UZU_PARAM(uint, num_v_heads)
UZU_PARAM(uint, value_dim)
UZU_PARAM(uint, width)
UZU_PARAM(bool, write_compact_v)
UZU_PARAM(bool, write_log_decay)
UZU_BUFFER(a_log)
UZU_BUFFER(beta_out)
UZU_BUFFER(compact_v_out)
UZU_BUFFER(decay_out)
UZU_BUFFER(dt_bias)
UZU_BUFFER(in_proj)
UZU_BUFFER(k_norm_out)
UZU_BUFFER(q_norm_out)

groupshared float qsum[WIDTH];
groupshared float ksum[WIDTH];
[shader("compute")][numthreads(WIDTH, 1, 1)] void main(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    let head = group.x;
    let token = group.y;
    let convDim = 2u * param_key_dim() + param_value_dim();
    let total = convDim + param_value_dim() + 2u * param_num_v_heads();
    let tokenBase = token * total;
    var q = 0.0;
    var k = 0.0;
    if (lane < param_HEAD_K_DIM()) {
        q = load_in_proj(tokenBase + head * param_HEAD_K_DIM() + lane);
        k = load_in_proj(tokenBase + param_key_dim() + head * param_HEAD_K_DIM() + lane);
    }
    qsum[lane] = q * q;
    ksum[lane] = k * k;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = param_width() / 2u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            qsum[lane] += qsum[lane + stride];
            ksum[lane] += ksum[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane < param_HEAD_K_DIM()) {
        let offset = token * param_key_dim() + head * param_HEAD_K_DIM() + lane;
        store_q_norm_out(offset, q * rsqrt(qsum[0] + 1e-6) * rsqrt(float(param_HEAD_K_DIM())));
        store_k_norm_out(offset, k * rsqrt(ksum[0] + 1e-6));
    }
    let perKey = param_num_v_heads() / param_num_k_heads();
    for (var index = lane; index < perKey; index += param_width()) {
        let hv = head * perKey + index;
        let betaRaw = load_in_proj(tokenBase + convDim + param_value_dim() + hv);
        let a = load_in_proj(tokenBase + convDim + param_value_dim() + param_num_v_heads() + hv) +
                load_dt_bias(hv);
        var softplus = a;
        if (a <= 20.0) {
            softplus = log(1.0 + exp(a));
        }
        let logDecay = -exp(load_a_log(hv)) * softplus;
        var decay = exp(logDecay);
        if (param_write_log_decay()) {
            decay = logDecay;
        }
        store_beta_out(token * param_num_v_heads() + hv, 1.0 / (1.0 + exp(-betaRaw)));
        store_decay_out(token * param_num_v_heads() + hv, decay);
    }
    if (param_write_compact_v()) {
        let valuesPerKey = param_value_dim() / param_num_k_heads();
        for (var i = lane; i < valuesPerKey; i += param_width()) {
            let column = head * valuesPerKey + i;
            store_compact_v_out(token * param_value_dim() + column,
                                load_in_proj(tokenBase + 2u * param_key_dim() + column));
        }
    }
}
