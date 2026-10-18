import definitions;
import storage;

groupshared float q[256];
groupshared float k[256];
groupshared float reduction[256];
groupshared float kr[256];

[[Variants("T", "float, half, BFloat16")]]
[[Variants("HEAD_K_DIM", "128")]]
[[Public]] [[Kernel]] func DeltaNetUpdate<
    T : ITensorElement,
    let HEAD_K_DIM : uint>(
    StructuredBuffer<uint> in_proj,
    StructuredBuffer<float> a_log,
    StructuredBuffer<float> dt_bias,
    StructuredBuffer<float> norm_weight,
    RWStructuredBuffer<float> state,
    RWStructuredBuffer<Atomic<uint>> out,
    uint num_v_heads,
    uint num_k_heads,
    uint head_v_dim,
    uint key_dim,
    uint value_dim,
    float norm_epsilon,
    ThreadContext thread_context,
    [[Groups("num_v_heads")]] uint hv_idx,
    [[Threads("256")]] uint tid,
) {
  uint lane = tid;
  uint3 group = uint3(hv_idx, 0, 0);

  float rowState[256];

  let hv = group.x;
  let hk = hv / (num_v_heads / num_k_heads);
  let convDim = 2u * key_dim + value_dim;
  var qRaw = 0.0;
  var kRaw = 0.0;
  if (lane < HEAD_K_DIM) {
    qRaw = tensorLoad<T>(in_proj, hk * HEAD_K_DIM + lane);
    kRaw = tensorLoad<T>(in_proj, key_dim + hk * HEAD_K_DIM + lane);
  }
  reduction[lane] = qRaw * qRaw;
  kr[lane] = kRaw * kRaw;
  GroupMemoryBarrierWithGroupSync();
  for (var stride = 256 / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      reduction[lane] += reduction[lane + stride];
      kr[lane] += kr[lane + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  let normalizedQ = qRaw * rsqrt(reduction[0] + 1e-6) * rsqrt(float(HEAD_K_DIM));
  let normalizedK = kRaw * rsqrt(kr[0] + 1e-6);
  GroupMemoryBarrierWithGroupSync();
  q[lane] = normalizedQ;
  k[lane] = normalizedK;
  reduction[lane] = normalizedQ * normalizedK;
  GroupMemoryBarrierWithGroupSync();
  for (var stride = 256 / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      reduction[lane] += reduction[lane + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  let kq = reduction[0];
  let beta = 1.0 / (1.0 + exp(-tensorLoad<T>(in_proj, convDim + value_dim + hv)));
  let a = tensorLoad<T>(in_proj, convDim + value_dim + num_v_heads + hv) + dt_bias[hv];
  var softplus = a;
  if (a <= 20.0) {
    softplus = log(1.0 + exp(a));
  }
  let decay = exp(-exp(a_log[hv]) * softplus);
  var output = 0.0;
  if (lane < head_v_dim) {
    let stateBase = (hv * head_v_dim + lane) * HEAD_K_DIM;
    var sq = 0.0;
    var sk = 0.0;
    for (var j = 0u; j < HEAD_K_DIM; j++) {
      let value = state[stateBase + j];
      rowState[j] = value;
      sq += value * q[j];
      sk += value * k[j];
    }
    let delta = beta * (tensorLoad<T>(in_proj, 2u * key_dim + hv * head_v_dim + lane) - decay * sk);
    output = decay * sq + delta * kq;
    for (var j = 0u; j < HEAD_K_DIM; j++) {
      state[stateBase + j] = decay * rowState[j] + k[j] * delta;
    }
  }
  GroupMemoryBarrierWithGroupSync();
  reduction[lane] = output * output;
  GroupMemoryBarrierWithGroupSync();
  for (var stride = 256 / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      reduction[lane] += reduction[lane + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  let inv = rsqrt(reduction[0] / float(head_v_dim) + norm_epsilon);
  if (lane < head_v_dim) {
    let z = tensorLoad<T>(in_proj, convDim + hv * head_v_dim + lane);
    tensorStore<T>(out, hv * head_v_dim + lane, output * inv * norm_weight[lane] * activate(z, 0u, 1.0));
  }
}
