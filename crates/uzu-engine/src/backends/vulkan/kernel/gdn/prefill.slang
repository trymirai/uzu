import definitions;
import common.workgroup_reduction;
import generated.delta_net;

static const uint DELTA_NET_PREFILL_THREADS = 128;
static const uint DELTA_NET_PREFILL_DV_PER_SIMDGROUP = 4;
static const uint DELTA_NET_PREFILL_SIMDGROUPS = DELTA_NET_PREFILL_THREADS / SIMD_SIZE;

[[Variants("T", "float, bf16")]]
[[Kernel]] [[Public]]
[[Precondition("new", "HEAD_K_DIM == crate::backends::vulkan::gpu_types::delta_net::DELTA_NET_HEAD_K_DIM")]]
[[Precondition("encode", "num_k_heads > 0")]]
[[Precondition("encode", "num_v_heads >= num_k_heads")]]
[[Precondition("encode", "num_v_heads * head_v_dim <= value_dim")]]
[[Precondition("encode", "key_dim.is_multiple_of(4)")]]
func DeltaNetPrefill<T: IStorageFloat>(
  Ptr<float, Access.Read> q_norm,
  Ptr<float, Access.Read> k_norm,
  Ptr<float, Access.Read> beta_buf,
  Ptr<float, Access.Read> decay_buf,
  Ptr<T, Access.Read> in_proj,
  Ptr<float, Access.ReadWrite> state,
  Ptr<T, Access.ReadWrite> out,
  uint num_v_heads,
  uint num_k_heads,
  uint head_v_dim,
  uint key_dim,
  uint value_dim,
  uint suffix_len,
  uint num_dv_groups,
  [[Groups("num_dv_groups")]] uint row_group,
  [[Groups("num_v_heads")]] uint hv,
  [[Threads("128")]] uint thread_idx,
  [[Specialize]] uint HEAD_K_DIM,
) {
  static_assert(
    DELTA_NET_PREFILL_THREADS == 128 && DELTA_NET_PREFILL_SIMDGROUPS * DELTA_NET_PREFILL_DV_PER_SIMDGROUP == 16,
    "Threads(128), num_dv_groups is head_v_dim.div_ceil(16)"
  );
  let invocation = subgroup_invocation();
  let simd = invocation / SIMD_SIZE;
  let lane = invocation % SIMD_SIZE;
  let dv_idx = (row_group * DELTA_NET_PREFILL_SIMDGROUPS + simd) * DELTA_NET_PREFILL_DV_PER_SIMDGROUP;
  let rows = dv_idx < head_v_dim ? min(head_v_dim - dv_idx, DELTA_NET_PREFILL_DV_PER_SIMDGROUP) : 0u;

  let hk = hv / (num_v_heads / num_k_heads);
  let total_proj_dim = 2 * key_dim + 2 * value_dim + 2 * num_v_heads;
  let dk_base = lane * 4;

  var q_offset = hk * DELTA_NET_HEAD_K_DIM + dk_base;
  var k_offset = hk * DELTA_NET_HEAD_K_DIM + dk_base;
  var beta_offset = hv;
  var decay_offset = hv;
  var v_row = 2 * key_dim + hv * head_v_dim + dv_idx;
  var out_row = hv * head_v_dim + dv_idx;

  let s_in = (hv * head_v_dim + dv_idx) * DELTA_NET_HEAD_K_DIM + dk_base;
  float4 s[DELTA_NET_PREFILL_DV_PER_SIMDGROUP];
  [ForceUnroll]
  for (uint r = 0; r < DELTA_NET_PREFILL_DV_PER_SIMDGROUP; r++) {
    s[r] = r < rows ? load_aligned<float4, 16>(state, s_in + r * DELTA_NET_HEAD_K_DIM) : float4(0.0);
  }

  for (uint token = 0; token < suffix_len; token++) {
    let decay = decay_buf[decay_offset];
    let beta = beta_buf[beta_offset];
    let k = load_aligned<float4, 16>(k_norm, k_offset);
    let q = load_aligned<float4, 16>(q_norm, q_offset);

    float4 kv_mem;
    float4 v_val;
    [ForceUnroll]
    for (uint r = 0; r < DELTA_NET_PREFILL_DV_PER_SIMDGROUP; r++) {
      s[r] *= decay;
      kv_mem[r] = dot(s[r], k);
      v_val[r] = r < rows ? in_proj[v_row + r].toFloat() : 0.0;
    }
    kv_mem = segment_reduce<VectorSum<4>, DELTA_NET_PREFILL_THREADS>(kv_mem, SIMD_SIZE);
    let delta = beta * (v_val - kv_mem);
    float4 o;
    [ForceUnroll]
    for (uint r = 0; r < DELTA_NET_PREFILL_DV_PER_SIMDGROUP; r++) {
      s[r] += k * delta[r];
      o[r] = dot(s[r], q);
    }
    o = segment_reduce<VectorSum<4>, DELTA_NET_PREFILL_THREADS>(o, SIMD_SIZE);

    [ForceUnroll]
    for (uint r = 0; r < DELTA_NET_PREFILL_DV_PER_SIMDGROUP; r++) {
      if (lane == 0 && r < rows) {
        out[out_row + r] = T.fromFloat(o[r]);
      }
    }

    q_offset += key_dim;
    k_offset += key_dim;
    beta_offset += num_v_heads;
    decay_offset += num_v_heads;
    v_row += total_proj_dim;
    out_row += value_dim;
  }

  [ForceUnroll]
  for (uint r = 0; r < DELTA_NET_PREFILL_DV_PER_SIMDGROUP; r++) {
    if (r < rows) {
      store_aligned<16>(state, s_in + r * DELTA_NET_HEAD_K_DIM, s[r]);
    }
  }
}
