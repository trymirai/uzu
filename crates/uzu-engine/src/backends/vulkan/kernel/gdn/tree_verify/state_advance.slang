import definitions;
import common.workgroup_reduction;
import generated.delta_net;

// Advances the [num_v_heads, 128, 128] FP32 state along the accepted path, k head hv / floor(num_v_heads / num_k_heads)
// as Metal maps it, also for nondivisible ratios. Each simdgroup holds STATE_ADVANCE_ROWS rows privately, lane l keys
// 4l..4l + 3; per accepted token s <- s e^log_decay, then s += k beta (v - s . k).

static const uint STATE_ADVANCE_THREADS = 128;
// The state is square, as Metal's StateAdvance.
static const uint STATE_ADVANCE_HEAD_V_DIM = DELTA_NET_HEAD_K_DIM;
public static const uint STATE_ADVANCE_ROWS = 4;

// Rows first.. of value head hv, by a workgroup of THREADS invocations that all call it: simdgroup s advances the
// STATE_ADVANCE_ROWS from first + s STATE_ADVANCE_ROWS, along a path of accepted_len >= 1 tokens.
public void advance_state<T : IStorageFloat, let THREADS : uint>(
  Ptr<T, Access.Read> k_norm,
  Ptr<T, Access.Read> v,
  Ptr<float, Access.Read> log_decay_buf,
  Ptr<float, Access.Read> beta_buf,
  Ptr<uint, Access.Read> accepted_indices,
  Ptr<float, Access.ReadWrite> state,
  uint accepted_len,
  uint num_v_heads,
  uint num_k_heads,
  uint hv,
  uint first,
) {
  let invocation = subgroup_invocation();
  let hk = hv / (num_v_heads / num_k_heads);
  let dv = first + invocation / SIMD_SIZE * STATE_ADVANCE_ROWS;
  let dk = invocation % SIMD_SIZE * 4;
  let key_dim = num_k_heads * DELTA_NET_HEAD_K_DIM;
  let value_dim = num_v_heads * STATE_ADVANCE_HEAD_V_DIM;
  let tile = (hv * STATE_ADVANCE_HEAD_V_DIM + dv) * DELTA_NET_HEAD_K_DIM + dk;
  float4 rows[STATE_ADVANCE_ROWS];
  for (uint r = 0; r < STATE_ADVANCE_ROWS; r++) {
    rows[r] = load_aligned<float4, 16>(state, tile + r * DELTA_NET_HEAD_K_DIM);
  }
  // The exit test follows the body, as accepted_len >= 1 allows: lavapipe runs a body whose head test failed with its
  // uniform loads unmasked, which would index the tree through a token past accepted_indices.
  uint i = 0;
  do {
    let t = accepted_indices[i];
    let lane = t * num_v_heads + hv;
    let decay = exp(log_decay_buf[lane]);
    let k_offset = t * key_dim + hk * DELTA_NET_HEAD_K_DIM + dk;
    let k = load_float4(k_norm, k_offset);
    var kv = float4(0.0);
    for (uint r = 0; r < STATE_ADVANCE_ROWS; r++) {
      rows[r] *= decay;
      kv[r] = dot(rows[r], k);
    }
    kv = segment_reduce<VectorSum<4>, THREADS>(kv, SIMD_SIZE);
    let v_row = t * value_dim + hv * STATE_ADVANCE_HEAD_V_DIM + dv;
    let value = load_float4(v, v_row);
    let delta = beta_buf[lane] * (value - kv);
    for (uint r = 0; r < STATE_ADVANCE_ROWS; r++) {
      rows[r] += k * delta[r];
    }
    i++;
  } while (i < accepted_len);
  for (uint r = 0; r < STATE_ADVANCE_ROWS; r++) {
    store_aligned<16>(state, tile + r * DELTA_NET_HEAD_K_DIM, rows[r]);
  }
}

[[Variants("T", "float, bf16")]]
[[Kernel]] [[Public]]
[[Precondition("new", "HEAD_K_DIM == crate::backends::vulkan::gpu_types::delta_net::DELTA_NET_HEAD_K_DIM")]]
[[Precondition("new", "num_k_heads != 0 && (num_v_heads == 0 || num_v_heads >= num_k_heads)")]]
public func StateAdvance<T: IStorageFloat>(
  Ptr<T, Access.Read> k_norm,
  Ptr<T, Access.Read> v,
  Ptr<float, Access.Read> log_decay_buf,
  Ptr<float, Access.Read> beta_buf,
  Ptr<uint, Access.Read> accepted_indices,
  Ptr<float, Access.ReadWrite> state,
  uint accepted_len,
  [[Groups("if accepted_len == 0 { 0 } else { num_v_heads }")]] uint head_group,
  [[Groups("8")]] uint row_group,
  [[Threads("128")]] uint thread_idx,
  [[Specialize]] uint HEAD_K_DIM,
  [[Specialize]] uint num_v_heads,
  [[Specialize]] uint num_k_heads,
) {
  // The empty grid for accepted_len 0 leaves a path of at least one token.
  advance_state<T, STATE_ADVANCE_THREADS>(
    k_norm, v, log_decay_buf, beta_buf, accepted_indices, state, accepted_len, num_v_heads, num_k_heads, head_group,
    row_group * (STATE_ADVANCE_THREADS / SIMD_SIZE) * STATE_ADVANCE_ROWS
  );
}
