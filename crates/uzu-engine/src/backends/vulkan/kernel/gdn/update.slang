import definitions;
import common.activation;
import gdn.prefill_prep;

// State [Hv, Dv, 128]: one workgroup per (v head, 4 rows), 32 invocations owning each row, lane l keys l + 32 i. Each
// row normalizes q and k itself, so its loads issue together and no sum crosses rows; the sums go through shared memory,
// which some wide-subgroup GPUs run several times faster than shuffles. Each row stores its updated state and its raw
// output o_i, and DeltaNetNormGate then applies the head-wide RMS and the gate.

static const uint DELTA_NET_UPDATE_KEYS = 128;
static const uint DELTA_NET_UPDATE_THREADS = 128;
static const uint DELTA_NET_UPDATE_ROWS = DELTA_NET_UPDATE_THREADS / SIMD_SIZE;

groupshared float3 delta_net_partials[DELTA_NET_UPDATE_THREADS];

// The sum over the 32 invocations of tid's row, every one of them reading the row's partials.
float3 delta_net_row_sum(float3 value, uint tid) {
  delta_net_partials[tid] = value;
  GroupMemoryBarrierWithGroupSync();
  var total = float3(0.0);
  let first = tid / SIMD_SIZE * SIMD_SIZE;
  for (uint lane = 0; lane < SIMD_SIZE; lane++) {
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
  [[Groups("head_v_dim.div_ceil(4)")]] uint row_group,
  [[Threads("128")]] uint tid,
  [[Specialize]] uint HEAD_K_DIM,
) {
  let lane = tid % SIMD_SIZE;
  let i = row_group * DELTA_NET_UPDATE_ROWS + tid / SIMD_SIZE;
  let valid = i < head_v_dim;
  let hv = head_group;
  let conv_dim = 2 * key_dim + value_dim;
  let hk = hv / (num_v_heads / num_k_heads);
  let q_base = hk * DELTA_NET_UPDATE_KEYS;
  let k_base = key_dim + hk * DELTA_NET_UPDATE_KEYS;
  let row = (hv * head_v_dim + i) * DELTA_NET_UPDATE_KEYS;
  // The lane's keys of q, k and the state row; a masked row reads no state and adds zeros.
  var q = float4(0.0);
  var k = float4(0.0);
  var s = float4(0.0);
  for (uint j = 0; j < 4; j++) {
    let key = lane + SIMD_SIZE * j;
    q[j] = in_proj[q_base + key].toFloat();
    k[j] = in_proj[k_base + key].toFloat();
    if (valid) {
      s[j] = state[row + key];
    }
  }
  let squares = delta_net_row_sum(float3(dot(q, q), dot(k, k), 0.0), tid);
  // As delta_net_inverse: 1 / sqrt(sum of squares + 1e-6).
  let q_inv = rsqrt(squares.x + 1e-6);
  let k_inv = rsqrt(squares.y + 1e-6);
  for (uint j = 0; j < 4; j++) {
    q[j] = delta_net_query(q[j], q_inv);
    k[j] *= k_inv;
  }
  // s . q, s . k and k . q.
  let dots = delta_net_row_sum(float3(dot(s, q), dot(s, k), dot(k, q)), tid);
  let beta = sigmoid(in_proj[conv_dim + value_dim + hv].toFloat());
  let a_raw = in_proj[conv_dim + value_dim + num_v_heads + hv].toFloat();
  let decay = exp(delta_net_log_decay(a_log[hv], a_raw, dt_bias[hv]));
  let v = in_proj[2 * key_dim + hv * head_v_dim + min(i, head_v_dim - 1)].toFloat();
  let delta = beta * (v - decay * dots.y);
  if (valid) {
    for (uint j = 0; j < 4; j++) {
      state[row + lane + SIMD_SIZE * j] = decay * s[j] + k[j] * delta;
    }
    if (lane == 0) {
      out[hv * head_v_dim + i] = T.fromFloat(decay * dots.x + delta * dots.z);
    }
  }
}
