import definitions;
import common.activation;
import common.workgroup_reduction;
import generated.activation_type;

// One workgroup per (token, k head): the head's q and k rows are L2-normalized (delta_net_inverse) and stored scaled by
// 1 / sqrt(128) and unscaled; its v heads get beta = sigmoid(beta_raw) and the log decay, or its exponential. With
// write_compact_v the token's raw values are copied in 128-element blocks hk, hk + G, and so on.

static const uint DELTA_NET_PREP_THREADS = 128;

// The log decay -(e^a_log) softplus(a_raw + dt_bias).
public float delta_net_log_decay(float a_log, float a_raw, float dt_bias) {
  return -exp(a_log) * activate(a_raw + dt_bias, ActivationType.SOFTPLUS);
}

// The normalized query v inv qs, qs = 1 / sqrt(128).
public float delta_net_query(float v, float inv) {
  return v * inv * rsqrt(128.0);
}

// 1 / sqrt(sum + 1e-6) for the sum of squares of the 128 values at source[offset], one per invocation of the
// workgroup. Every invocation must call it.
float delta_net_inverse<T: IModelFloat>(Ptr<T, Access.Read> source, uint offset, uint tid) {
  let x = source[offset + tid].toFloat();
  return rms_inverse<DELTA_NET_PREP_THREADS>(x * x, 1, 1e-6);
}

[[Kernel]] [[Public]]
[[Precondition("new", "HEAD_K_DIM == 128")]]
[[Precondition("encode", "num_k_heads > 0")]]
[[Precondition("encode", "suffix_len <= 1 || num_k_heads <= key_dim / 128")]]
public func DeltaNetPrefillPrep<T: IModelFloat, QKT: IModelFloat>(
  Ptr<T, Access.Read> in_proj,
  Ptr<float, Access.Read> a_log,
  Ptr<float, Access.Read> dt_bias,
  Ptr<QKT, Access.ReadWrite> q_norm_out,
  Ptr<QKT, Access.ReadWrite> k_norm_out,
  [[Optional("write_compact_v")]] Ptr<T, Access.ReadWrite> compact_v_out,
  Ptr<float, Access.ReadWrite> beta_out,
  Ptr<float, Access.ReadWrite> decay_out,
  uint num_v_heads,
  uint num_k_heads,
  uint key_dim,
  uint value_dim,
  uint suffix_len,
  [[Groups("num_k_heads")]] uint head_group,
  [[Groups("suffix_len")]] uint token_group,
  [[Threads("128")]] uint tid,
  [[Specialize]] uint HEAD_K_DIM,
  [[Specialize]] bool write_log_decay,
  [[Specialize]] bool write_compact_v,
) {
  let hk = head_group;
  let token = token_group;
  let conv_dim = 2 * key_dim + value_dim;
  let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
  let heads_per_group = num_v_heads / num_k_heads;
  let tok_offset = token * total_proj_dim;
  if (write_compact_v) {
    for (uint v = hk * DELTA_NET_PREP_THREADS + tid; v < value_dim; v += num_k_heads * DELTA_NET_PREP_THREADS) {
      compact_v_out[token * value_dim + v] = in_proj[tok_offset + 2 * key_dim + v];
    }
  }
  let q_base = tok_offset + hk * DELTA_NET_PREP_THREADS;
  let k_base = tok_offset + key_dim + hk * DELTA_NET_PREP_THREADS;
  let q_inv = delta_net_inverse<T>(in_proj, q_base, tid);
  let k_inv = delta_net_inverse<T>(in_proj, k_base, tid);
  let out_base = token * key_dim + hk * DELTA_NET_PREP_THREADS;
  q_norm_out[out_base + tid] = QKT.fromFloat(delta_net_query(in_proj[q_base + tid].toFloat(), q_inv));
  k_norm_out[out_base + tid] = QKT.fromFloat(in_proj[k_base + tid].toFloat() * k_inv);
  for (uint group = tid; group < heads_per_group; group += DELTA_NET_PREP_THREADS) {
    let hv = hk * heads_per_group + group;
    let beta_raw = in_proj[tok_offset + conv_dim + value_dim + hv].toFloat();
    let a_raw = in_proj[tok_offset + conv_dim + value_dim + num_v_heads + hv].toFloat();
    let log_decay = delta_net_log_decay(a_log[hv], a_raw, dt_bias[hv]);
    beta_out[token * num_v_heads + hv] = sigmoid(beta_raw);
    decay_out[token * num_v_heads + hv] = write_log_decay ? log_decay : exp(log_decay);
  }
}
