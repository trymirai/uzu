import definitions;
import storage;

[[Variants("T", "float, half, BFloat16")]]
[[Variants("HEAD_K_DIM", "128")]]
[[Public]] [[Kernel]] func DeltaNetPrefill<
    T : ITensorElement,
    let HEAD_K_DIM : uint32_t>(
    StructuredBuffer<float> q_norm,
    StructuredBuffer<float> k_norm,
    StructuredBuffer<float> beta_buf,
    StructuredBuffer<float> decay_buf,
    StructuredBuffer<uint32_t> in_proj,
    RWStructuredBuffer<float> state,
    RWStructuredBuffer<Atomic<uint32_t>> out,
    uint32_t num_v_heads,
    uint32_t num_k_heads,
    uint32_t head_v_dim,
    uint32_t key_dim,
    uint32_t value_dim,
    uint32_t suffix_len,
    uint32_t num_dv_groups,
    [[Groups("num_v_heads")]] uint32_t hv_idx,
    [[Groups("head_v_dim.div_ceil(32)")]] uint32_t dv_group,
    [[Threads("32")]] uint32_t tid,
) {
  uint3 gid = uint3(dv_group * 32 + tid, hv_idx, 0);

  float rowState[256];

  let dv = gid.x;
  let hv = gid.y;
  if (dv >= head_v_dim) {
    return;
  }
  let hk = hv / (num_v_heads / num_k_heads);
  let stateBase = (hv * head_v_dim + dv) * HEAD_K_DIM;
  let total = 2u * key_dim + 2u * value_dim + 2u * num_v_heads;
  for (var j = 0u; j < HEAD_K_DIM; j++) {
    rowState[j] = state[stateBase + j];
  }
  for (var token = 0u; token < suffix_len; token++) {
    let offset = token * key_dim + hk * HEAD_K_DIM;
    let decay = decay_buf[token * num_v_heads + hv];
    let beta = beta_buf[token * num_v_heads + hv];
    var retrieved = 0.0;
    for (var j = 0u; j < HEAD_K_DIM; j++) {
      retrieved += (decay * rowState[j]) * k_norm[offset + j];
    }
    let v = tensorLoad<T>(in_proj, token * total + 2u * key_dim + hv * head_v_dim + dv);
    let delta = beta * (v - retrieved);
    var output = 0.0;
    for (var j = 0u; j < HEAD_K_DIM; j++) {
      rowState[j] = decay * rowState[j] + k_norm[offset + j] * delta;
      output += rowState[j] * q_norm[offset + j];
    }
    tensorStore<T>(out, token * value_dim + hv * head_v_dim + dv, output);
  }
  for (var j = 0u; j < HEAD_K_DIM; j++) {
    state[stateBase + j] = rowState[j];
  }
}
