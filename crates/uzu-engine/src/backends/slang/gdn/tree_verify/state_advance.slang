import definitions;
import storage;

[[Variants("T", "float, half, BFloat16")]]
[[Variants("HEAD_K_DIM", "128")]]
[[Public]] [[Kernel]] func StateAdvance<
    T : ITensorElement,
    let HEAD_K_DIM : uint>(
    StructuredBuffer<uint> k_norm,
    StructuredBuffer<uint> v,
    StructuredBuffer<float> log_decay_buf,
    StructuredBuffer<float> beta_buf,
    StructuredBuffer<uint> accepted_indices,
    RWStructuredBuffer<float> state,
    uint accepted_len,
    [[Specialize]] uint num_v_heads,
    [[Specialize]] uint num_k_heads,
    ThreadContext thread_context,
    [[Groups("num_v_heads")]] uint hv_idx,
    [[Groups("(HEAD_K_DIM + 31) / 32")]] uint dv_group_idx,
    [[Threads("32")]] uint thread_idx,
) {
  uint3 gid = uint3(dv_group_idx * 32 + thread_idx, hv_idx, 0);

  float rowState[HEAD_K_DIM];

  let dv = gid.x;
  let hv = gid.y;
  if (dv >= HEAD_K_DIM) {
    return;
  }
  let hk = hv / (num_v_heads / num_k_heads);
  let stateBase = (hv * HEAD_K_DIM + dv) * HEAD_K_DIM;
  for (var j = 0u; j < HEAD_K_DIM; j++) {
    rowState[j] = state[stateBase + j];
  }
  for (var accepted = 0u; accepted < accepted_len; accepted++) {
    let node = accepted_indices[accepted];
    let offset = (node * num_k_heads + hk) * HEAD_K_DIM;
    let decay = exp(log_decay_buf[node * num_v_heads + hv]);
    let beta = beta_buf[node * num_v_heads + hv];
    var retrieved = 0.0;
    for (var j = 0u; j < HEAD_K_DIM; j++) {
      rowState[j] *= decay;
      retrieved += rowState[j] * tensorLoad<T>(k_norm, offset + j);
    }
    let value = tensorLoad<T>(v, (node * num_v_heads + hv) * HEAD_K_DIM + dv);
    let delta = beta * (value - retrieved);
    for (var j = 0u; j < HEAD_K_DIM; j++) {
      rowState[j] += tensorLoad<T>(k_norm, offset + j) * delta;
    }
  }
  for (var j = 0u; j < HEAD_K_DIM; j++) {
    state[stateBase + j] = rowState[j];
  }
}
