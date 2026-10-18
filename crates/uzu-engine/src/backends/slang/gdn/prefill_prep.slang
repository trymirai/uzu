import definitions;
import storage;

groupshared float qsum[256];
groupshared float ksum[256];

[[Variants("T", "float, half, BFloat16")]]
[[Variants("QKT", "float, half, BFloat16")]]
[[Variants("HEAD_K_DIM", "128")]]
[[Public]] [[Kernel]] func DeltaNetPrefillPrep<T : ITensorElement, QKT : ITensorElement, let HEAD_K_DIM : uint>(
    StructuredBuffer<uint> in_proj,
    StructuredBuffer<float> a_log,
    StructuredBuffer<float> dt_bias,
    RWStructuredBuffer<Atomic<uint>> q_norm_out,
    RWStructuredBuffer<Atomic<uint>> k_norm_out,
    [[Optional("write_compact_v")]] RWStructuredBuffer<Atomic<uint>> compact_v_out,
    RWStructuredBuffer<float> beta_out,
    RWStructuredBuffer<float> decay_out,
    uint num_v_heads,
    uint num_k_heads,
    uint key_dim,
    uint value_dim,
    uint suffix_len,
    [[Groups("suffix_len")]] uint token_idx,
    [[Groups("num_k_heads")]] uint hk_idx,
    [[Specialize]] bool write_log_decay,
    [[Specialize]] bool write_compact_v,
    [[Threads("HEAD_K_DIM")]] uint lane,
) {
  uint3 group = uint3(hk_idx, token_idx, 0);

  let head = group.x;
  let token = group.y;
  let convDim = 2u * key_dim + value_dim;
  let total = convDim + value_dim + 2u * num_v_heads;
  let tokenBase = token * total;
  var q = 0.0;
  var k = 0.0;
  if (lane < HEAD_K_DIM) {
    q = tensorLoad<T>(in_proj, tokenBase + head * HEAD_K_DIM + lane);
    k = tensorLoad<T>(in_proj, tokenBase + key_dim + head * HEAD_K_DIM + lane);
  }
  qsum[lane] = q * q;
  ksum[lane] = k * k;
  GroupMemoryBarrierWithGroupSync();
  for (var stride = HEAD_K_DIM / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      qsum[lane] += qsum[lane + stride];
      ksum[lane] += ksum[lane + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (lane < HEAD_K_DIM) {
    let offset = token * key_dim + head * HEAD_K_DIM + lane;
    tensorStore<QKT>(q_norm_out, offset, q * rsqrt(qsum[0] + 1e-6) * rsqrt(float(HEAD_K_DIM)));
    tensorStore<QKT>(k_norm_out, offset, k * rsqrt(ksum[0] + 1e-6));
  }
  let perKey = num_v_heads / num_k_heads;
  for (var index = lane; index < perKey; index += HEAD_K_DIM) {
    let hv = head * perKey + index;
    let betaRaw = tensorLoad<T>(in_proj, tokenBase + convDim + value_dim + hv);
    let a = tensorLoad<T>(in_proj, tokenBase + convDim + value_dim + num_v_heads + hv) + dt_bias[hv];
    var softplus = a;
    if (a <= 20.0) {
      softplus = log(1.0 + exp(a));
    }
    let logDecay = -exp(a_log[hv]) * softplus;
    var decay = exp(logDecay);
    if (write_log_decay) {
      decay = logDecay;
    }
    beta_out[token * num_v_heads + hv] = 1.0 / (1.0 + exp(-betaRaw));
    decay_out[token * num_v_heads + hv] = decay;
  }
  if (write_compact_v) {
    let valuesPerKey = value_dim / num_k_heads;
    for (var i = lane; i < valuesPerKey; i += HEAD_K_DIM) {
      let column = head * valuesPerKey + i;
      tensorStore<T>(
          compact_v_out,
          token * value_dim + column,
          tensorLoad<T>(in_proj, tokenBase + 2u * key_dim + column)
      );
    }
  }
}
