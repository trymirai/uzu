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

groupshared float q[WIDTH];
groupshared float k[WIDTH];
groupshared float reduction[WIDTH];
groupshared float kr[WIDTH];
[shader("compute")][numthreads(WIDTH, 1, 1)] void main(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    float rowState[WIDTH];

    let hv = group.x;
    let hk = hv / (param_num_v_heads() / param_num_k_heads());
    let convDim = 2u * param_key_dim() + param_value_dim();
    var qRaw = 0.0;
    var kRaw = 0.0;
    if (lane < param_HEAD_K_DIM()) {
        qRaw = load_in_proj(hk * param_HEAD_K_DIM() + lane);
        kRaw = load_in_proj(param_key_dim() + hk * param_HEAD_K_DIM() + lane);
    }
    reduction[lane] = qRaw * qRaw;
    kr[lane] = kRaw * kRaw;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = param_width() / 2u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
            kr[lane] += kr[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    let normalizedQ = qRaw * rsqrt(reduction[0] + 1e-6) * rsqrt(float(param_HEAD_K_DIM()));
    let normalizedK = kRaw * rsqrt(kr[0] + 1e-6);
    GroupMemoryBarrierWithGroupSync();
    q[lane] = normalizedQ;
    k[lane] = normalizedK;
    reduction[lane] = normalizedQ * normalizedK;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = param_width() / 2u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    let kq = reduction[0];
    let beta = 1.0 / (1.0 + exp(-load_in_proj(convDim + param_value_dim() + hv)));
    let a = load_in_proj(convDim + param_value_dim() + param_num_v_heads() + hv) + load_dt_bias(hv);
    var softplus = a;
    if (a <= 20.0) {
        softplus = log(1.0 + exp(a));
    }
    let decay = exp(-exp(load_a_log(hv)) * softplus);
    var output = 0.0;
    if (lane < param_head_v_dim()) {
        let stateBase = (hv * param_head_v_dim() + lane) * param_HEAD_K_DIM();
        var sq = 0.0;
        var sk = 0.0;
        for (var j = 0u; j < param_HEAD_K_DIM(); j++) {
            let value = load_state(stateBase + j);
            rowState[j] = value;
            sq += value * q[j];
            sk += value * k[j];
        }
        let delta = beta * (load_in_proj(2u * param_key_dim() + hv * param_head_v_dim() + lane) - decay * sk);
        output = decay * sq + delta * kq;
        for (var j = 0u; j < param_HEAD_K_DIM(); j++) {
            store_state(stateBase + j, decay * rowState[j] + k[j] * delta);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    reduction[lane] = output * output;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = param_width() / 2u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    let inv = rsqrt(reduction[0] / float(param_head_v_dim()) + param_norm_epsilon());
    if (lane < param_head_v_dim()) {
        let z = load_in_proj(convDim + hv * param_head_v_dim() + lane);
        store_out(hv * param_head_v_dim() + lane,
                  output * inv * load_norm_weight(lane) * activate(z, 0u, 1.0));
    }
}
