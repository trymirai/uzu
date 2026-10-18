import definitions;
import storage;

[[Variants("ElementT", "float, half, BFloat16")]]
[[Variants("RopeT", "float")]]
[[Public]] [[Kernel]] func AttentionPrepare<ElementT : ITensorElement, RopeT : ITensorElement>(
    StructuredBuffer<uint> qkvg,
    RWStructuredBuffer<Atomic<uint>> queries,
    [[Optional("has_kv")]] RWStructuredBuffer<Atomic<uint>> keys,
    [[Optional("has_kv")]] RWStructuredBuffer<Atomic<uint>> values,
    [[Optional("has_rope")]] StructuredBuffer<uint> cosines,
    [[Optional("has_rope")]] StructuredBuffer<uint> sines,
    uint num_q_heads,
    [[Optional("has_kv")]] uint num_kv_heads,
    uint head_dim,
    [[Optional("has_rope")]] uint rope_dim,
    [[Optional("has_kv")]] uint kv_token_offset,
    uint input_row_stride,
    uint batch_dim,
    [[Specialize]] bool has_kv,
    [[Specialize]] bool has_rope,
    [[Axis("head_dim", "128")]] uint head_dim_idx,
    [[Axis("num_q_heads + num_kv_heads.unwrap_or(0) * 2", "1")]] uint head_idx,
    [[Axis("batch_dim", "1")]] uint batch_idx,
) {
  uint total_heads = num_q_heads + (has_kv ? 2 * num_kv_heads : 0);
  uint3 gid = uint3((batch_idx * total_heads + head_idx) * head_dim + head_dim_idx, 0, 0);

  let index = gid.x;
  if (index >= batch_dim * total_heads * head_dim) {
    return;
  }
  let dim = index % head_dim;
  let head = (index / head_dim) % total_heads;
  let batch = index / (head_dim * total_heads);
  let inputBase = batch * input_row_stride + head * head_dim;
  let query = head < num_q_heads;
  let key = head >= num_q_heads && head < num_q_heads + num_kv_heads;
  var value = tensorLoad<ElementT>(qkvg, inputBase + dim);
  if (has_rope && dim < rope_dim && (query || key)) {
    let halfDim = rope_dim / 2u;
    let paired = choose(dim - halfDim, dim + halfDim, dim < halfDim);
    let signedPaired = tensorLoad<ElementT>(qkvg, inputBase + paired) * choose(1.0, -1.0, dim < halfDim);
    value = value * tensorLoad<RopeT>(cosines, batch * rope_dim + dim) +
            signedPaired * tensorLoad<RopeT>(sines, batch * rope_dim + dim);
  }
  if (query) {
    tensorStore<ElementT>(queries, (head * batch_dim + batch) * head_dim + dim, value);
  } else if (key) {
    tensorStore<ElementT>(
        keys,
        ((kv_token_offset + batch) * num_kv_heads + head - num_q_heads) * head_dim + dim,
        value
    );
  } else {
    tensorStore<ElementT>(
        values,
        ((kv_token_offset + batch) * num_kv_heads + head - num_q_heads - num_kv_heads) * head_dim + dim,
        value
    );
  }
}
