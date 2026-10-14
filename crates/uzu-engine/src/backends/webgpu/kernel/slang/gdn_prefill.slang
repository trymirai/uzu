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

[shader("compute")][numthreads(32, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    float rowState[WIDTH];

    let dv = gid.x;
    let hv = gid.y;
    if (dv >= param_head_v_dim()) {
        return;
    }
    let hk = hv / (param_num_v_heads() / param_num_k_heads());
    let stateBase = (hv * param_head_v_dim() + dv) * param_HEAD_K_DIM();
    let total = 2u * param_key_dim() + 2u * param_value_dim() + 2u * param_num_v_heads();
    for (var j = 0u; j < param_HEAD_K_DIM(); j++) {
        rowState[j] = load_state(stateBase + j);
    }
    for (var token = 0u; token < param_suffix_len(); token++) {
        let offset = token * param_key_dim() + hk * param_HEAD_K_DIM();
        let decay = load_decay_buf(token * param_num_v_heads() + hv);
        let beta = load_beta_buf(token * param_num_v_heads() + hv);
        var retrieved = 0.0;
        for (var j = 0u; j < param_HEAD_K_DIM(); j++) {
            retrieved += (decay * rowState[j]) * load_k_norm(offset + j);
        }
        let v = load_in_proj(token * total + 2u * param_key_dim() + hv * param_head_v_dim() + dv);
        let delta = beta * (v - retrieved);
        var output = 0.0;
        for (var j = 0u; j < param_HEAD_K_DIM(); j++) {
            rowState[j] = decay * rowState[j] + load_k_norm(offset + j) * delta;
            output += rowState[j] * load_q_norm(offset + j);
        }
        store_out(token * param_value_dim() + hv * param_head_v_dim() + dv, output);
    }
    for (var j = 0u; j < param_HEAD_K_DIM(); j++) {
        store_state(stateBase + j, rowState[j]);
    }
}
