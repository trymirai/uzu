// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, batch_dim)
UZU_PARAM(bool, has_rope)
UZU_PARAM(uint, head_dim)
UZU_PARAM(uint, input_row_stride)
UZU_PARAM(uint, kv_token_offset)
UZU_PARAM(uint, num_kv_heads)
UZU_PARAM(uint, num_q_heads)
UZU_PARAM(uint, rope_dim)
UZU_PARAM(uint, total_heads)
UZU_BUFFER(cosines)
UZU_BUFFER(keys)
UZU_BUFFER(qkvg)
UZU_BUFFER(queries)
UZU_BUFFER(sines)
UZU_BUFFER(values)

[shader("compute")][numthreads(256, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    let index = gid.x;
    if (index >= param_batch_dim() * param_total_heads() * param_head_dim()) {
        return;
    }
    let dim = index % param_head_dim();
    let head = (index / param_head_dim()) % param_total_heads();
    let batch = index / (param_head_dim() * param_total_heads());
    let inputBase = batch * param_input_row_stride() + head * param_head_dim();
    let query = head < param_num_q_heads();
    let key = head >= param_num_q_heads() && head < param_num_q_heads() + param_num_kv_heads();
    var value = load_qkvg(inputBase + dim);
    if (param_has_rope() && dim < param_rope_dim() && (query || key)) {
        let halfDim = param_rope_dim() / 2u;
        let paired = choose(dim - halfDim, dim + halfDim, dim < halfDim);
        let signedPaired = load_qkvg(inputBase + paired) * choose(1.0, -1.0, dim < halfDim);
        value = value * load_cosines(batch * param_rope_dim() + dim) +
                signedPaired * load_sines(batch * param_rope_dim() + dim);
    }
    if (query) {
        store_queries((head * param_batch_dim() + batch) * param_head_dim() + dim, value);
    } else if (key) {
        store_keys(((param_kv_token_offset() + batch) * param_num_kv_heads() + head - param_num_q_heads()) *
                           param_head_dim() +
                       dim,
                   value);
    } else {
        store_values(((param_kv_token_offset() + batch) * param_num_kv_heads() + head - param_num_q_heads() -
                      param_num_kv_heads()) *
                             param_head_dim() +
                         dim,
                     value);
    }
}
