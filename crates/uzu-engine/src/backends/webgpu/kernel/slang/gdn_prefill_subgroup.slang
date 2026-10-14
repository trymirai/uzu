// uzu-widths: 32, 64, 128
// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, HEAD_K_DIM)
UZU_PARAM(uint, head_v_dim)
UZU_PARAM(uint, key_dim)
UZU_PARAM(uint, num_k_heads)
UZU_PARAM(uint, num_v_heads)
UZU_PARAM(uint, suffix_len)
UZU_PARAM(uint, value_dim)
UZU_BUFFER(beta_buf)
UZU_BUFFER(decay_buf)
UZU_BUFFER(in_proj)
UZU_BUFFER(k_norm)
UZU_BUFFER(out)
UZU_BUFFER(q_norm)
UZU_BUFFER(state)

// One subgroup per value row keeps the recurrent state in a few registers
// per lane while adjacent lanes load adjacent key components.
[shader("compute")][numthreads(256, 1, 1)] void main(uint lane : SV_WaveLaneIndex, uint wave : SV_WaveIndex,
                                                     uint3 group : SV_GroupID) {
    let dv = group.x * (256 / WIDTH) + wave;
    let hv = group.y;
    let hk = hv / (param_num_v_heads() / param_num_k_heads());
    let stateBase = (hv * param_head_v_dim() + dv) * param_HEAD_K_DIM();
    let total = 2u * param_key_dim() + 2u * param_value_dim() + 2u * param_num_v_heads();
    float rowState[256 / WIDTH];
    [ForceUnroll] for (uint part = 0; part < 256 / WIDTH; part++) {
        let dk = lane + part * WIDTH;
        rowState[part] =
            dv < param_head_v_dim() && dk < param_HEAD_K_DIM() ? load_state(stateBase + dk) : 0.0;
    }
    for (uint token = 0; token < param_suffix_len(); token++) {
        let offset = token * param_key_dim() + hk * param_HEAD_K_DIM();
        let decay = load_decay_buf(token * param_num_v_heads() + hv);
        let beta = load_beta_buf(token * param_num_v_heads() + hv);
        float key[256 / WIDTH];
        float retrievedPartial = 0.0;
        [ForceUnroll] for (uint part = 0; part < 256 / WIDTH; part++) {
            let dk = lane + part * WIDTH;
            key[part] = dk < param_HEAD_K_DIM() ? load_k_norm(offset + dk) : 0.0;
            retrievedPartial += (decay * rowState[part]) * key[part];
        }
        let retrieved = WaveActiveSum(retrievedPartial);
        let value = dv < param_head_v_dim()
                        ? load_in_proj(token * total + 2u * param_key_dim() + hv * param_head_v_dim() + dv)
                        : 0.0;
        let delta = beta * (value - retrieved);
        float outputPartial = 0.0;
        [ForceUnroll] for (uint part = 0; part < 256 / WIDTH; part++) {
            let dk = lane + part * WIDTH;
            rowState[part] = decay * rowState[part] + key[part] * delta;
            if (dk < param_HEAD_K_DIM())
                outputPartial += rowState[part] * load_q_norm(offset + dk);
        }
        let output = WaveActiveSum(outputPartial);
        if (lane == 0u && dv < param_head_v_dim())
            store_out(token * param_value_dim() + hv * param_head_v_dim() + dv, output);
    }
    if (dv < param_head_v_dim()) {
        [ForceUnroll] for (uint part = 0; part < 256 / WIDTH; part++) {
            let dk = lane + part * WIDTH;
            if (dk < param_HEAD_K_DIM())
                store_state(stateBase + dk, rowState[part]);
        }
    }
}
