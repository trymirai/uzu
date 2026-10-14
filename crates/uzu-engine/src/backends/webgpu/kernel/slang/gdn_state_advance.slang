// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, HEAD_K_DIM)
UZU_PARAM(uint, accepted_len)
UZU_PARAM(uint, num_k_heads)
UZU_PARAM(uint, num_v_heads)
UZU_BUFFER(accepted_indices)
UZU_BUFFER(beta_buf)
UZU_BUFFER(k_norm)
UZU_BUFFER(log_decay_buf)
UZU_BUFFER(state)
UZU_BUFFER(v)

[shader("compute")][numthreads(32, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    float rowState[WIDTH];

    let dv = gid.x;
    let hv = gid.y;
    if (dv >= param_HEAD_K_DIM()) {
        return;
    }
    let hk = hv / (param_num_v_heads() / param_num_k_heads());
    let stateBase = (hv * param_HEAD_K_DIM() + dv) * param_HEAD_K_DIM();
    for (var j = 0u; j < param_HEAD_K_DIM(); j++) {
        rowState[j] = load_state(stateBase + j);
    }
    for (var accepted = 0u; accepted < param_accepted_len(); accepted++) {
        let node = word_accepted_indices(accepted);
        let offset = (node * param_num_k_heads() + hk) * param_HEAD_K_DIM();
        let decay = exp(load_log_decay_buf(node * param_num_v_heads() + hv));
        let beta = load_beta_buf(node * param_num_v_heads() + hv);
        var retrieved = 0.0;
        for (var j = 0u; j < param_HEAD_K_DIM(); j++) {
            rowState[j] *= decay;
            retrieved += rowState[j] * load_k_norm(offset + j);
        }
        let value = load_v((node * param_num_v_heads() + hv) * param_HEAD_K_DIM() + dv);
        let delta = beta * (value - retrieved);
        for (var j = 0u; j < param_HEAD_K_DIM(); j++) {
            rowState[j] += load_k_norm(offset + j) * delta;
        }
    }
    for (var j = 0u; j < param_HEAD_K_DIM(); j++) {
        store_state(stateBase + j, rowState[j]);
    }
}
