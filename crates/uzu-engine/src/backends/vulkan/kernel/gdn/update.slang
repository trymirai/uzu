import definitions;
import common.activation;
import gdn.prefill_prep;

// State [Hv, Dv, 128]: one workgroup per (v head, 16 rows), 8 invocations owning each row, lane l the 16 keys of
// float4s l + 8 c. Each row normalizes q and k itself, so its loads issue together and no sum crosses rows; the sums go
// through shared memory, which some wide-subgroup GPUs run several times faster than shuffles. Each row stores its
// updated state and its raw output o_i, and DeltaNetNormGate then applies the head-wide RMS and the gate.

static const uint DELTA_NET_UPDATE_KEYS = 128;
static const uint DELTA_NET_UPDATE_THREADS = 128;
static const uint DELTA_NET_UPDATE_LANES = 8;
static const uint DELTA_NET_UPDATE_CHUNKS = DELTA_NET_UPDATE_KEYS / (4 * DELTA_NET_UPDATE_LANES);
static const uint DELTA_NET_UPDATE_ROWS = DELTA_NET_UPDATE_THREADS / DELTA_NET_UPDATE_LANES;

groupshared float3 delta_net_partials[DELTA_NET_UPDATE_THREADS];

// The sum over the DELTA_NET_UPDATE_LANES invocations of tid's row, every one of them reading the row's partials.
float3 delta_net_row_sum(float3 value, uint tid) {
  delta_net_partials[tid] = value;
  GroupMemoryBarrierWithGroupSync();
  var total = float3(0.0);
  let first = tid / DELTA_NET_UPDATE_LANES * DELTA_NET_UPDATE_LANES;
  for (uint lane = 0; lane < DELTA_NET_UPDATE_LANES; lane++) {
    total += delta_net_partials[first + lane];
  }
  GroupMemoryBarrierWithGroupSync();
  return total;
}

[[Kernel]]
[[Precondition("new", "HEAD_K_DIM == 128")]]
[[Precondition("encode", "num_v_heads == 0 || num_k_heads > 0 && num_v_heads.is_multiple_of(num_k_heads)")]]
func DeltaNetUpdate<T: IModelFloat>(
  Ptr<T, Access.Read> in_proj,
  Ptr<float, Access.Read> a_log,
  Ptr<float, Access.Read> dt_bias,
  Ptr<float, Access.ReadWrite> state,
  Ptr<T, Access.ReadWrite> out,
  uint num_v_heads,
  uint num_k_heads,
  uint head_v_dim,
  uint key_dim,
  uint value_dim,
  [[Groups("num_v_heads")]] uint head_group,
  [[Groups("head_v_dim.div_ceil(16)")]] uint row_group,
  [[Threads("128")]] uint tid,
  [[Specialize]] uint HEAD_K_DIM,
) {
  let lane = tid % DELTA_NET_UPDATE_LANES;
  let i = row_group * DELTA_NET_UPDATE_ROWS + tid / DELTA_NET_UPDATE_LANES;
  let valid = i < head_v_dim;
  let hv = head_group;
  let conv_dim = 2 * key_dim + value_dim;
  let hk = hv / (num_v_heads / num_k_heads);
  let q_base = hk * DELTA_NET_UPDATE_KEYS;
  let k_base = key_dim + hk * DELTA_NET_UPDATE_KEYS;
  let row = (hv * head_v_dim + i) * DELTA_NET_UPDATE_KEYS;
  // The lane's keys of q, k and the state row; a masked row reads no state and adds zeros.
  float4 q[DELTA_NET_UPDATE_CHUNKS];
  float4 k[DELTA_NET_UPDATE_CHUNKS];
  float4 s[DELTA_NET_UPDATE_CHUNKS];
  var q_squares = 0.0;
  var k_squares = 0.0;
  [unroll]
  for (uint c = 0; c < DELTA_NET_UPDATE_CHUNKS; c++) {
    let key = 4 * (lane + DELTA_NET_UPDATE_LANES * c);
    q[c] = load_float4(in_proj, q_base + key);
    k[c] = load_float4(in_proj, k_base + key);
    s[c] = valid ? load_aligned<float4, 16>(state, row + key) : float4(0.0);
    q_squares += dot(q[c], q[c]);
    k_squares += dot(k[c], k[c]);
  }
  let squares = delta_net_row_sum(float3(q_squares, k_squares, 0.0), tid);
  // As delta_net_inverse: 1 / sqrt(sum of squares + 1e-6).
  let q_inv = rsqrt(squares.x + 1e-6);
  let k_inv = rsqrt(squares.y + 1e-6);
  var partial_dots = float3(0.0);
  [unroll]
  for (uint c = 0; c < DELTA_NET_UPDATE_CHUNKS; c++) {
    q[c] = float4(delta_net_query(q[c].x, q_inv), delta_net_query(q[c].y, q_inv), delta_net_query(q[c].z, q_inv),
      delta_net_query(q[c].w, q_inv));
    k[c] *= k_inv;
    partial_dots += float3(dot(s[c], q[c]), dot(s[c], k[c]), dot(k[c], q[c]));
  }
  // s . q, s . k and k . q.
  let dots = delta_net_row_sum(partial_dots, tid);
  let beta = sigmoid(in_proj[conv_dim + value_dim + hv].toFloat());
  let a_raw = in_proj[conv_dim + value_dim + num_v_heads + hv].toFloat();
  let decay = exp(delta_net_log_decay(a_log[hv], a_raw, dt_bias[hv]));
  let v = in_proj[2 * key_dim + hv * head_v_dim + min(i, head_v_dim - 1)].toFloat();
  let delta = beta * (v - decay * dots.y);
  if (valid) {
    [unroll]
    for (uint c = 0; c < DELTA_NET_UPDATE_CHUNKS; c++) {
      store_aligned<16>(state, row + 4 * (lane + DELTA_NET_UPDATE_LANES * c), decay * s[c] + k[c] * delta);
    }
    if (lane == 0) {
      out[hv * head_v_dim + i] = T.fromFloat(decay * dots.x + delta * dots.z);
    }
  }
}
