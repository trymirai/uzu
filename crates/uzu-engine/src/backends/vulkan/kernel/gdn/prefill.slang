import definitions;
import common.workgroup_reduction;
import generated.delta_net;

// Each simdgroup owns four value rows of v head hv (k head hv / floor(H / G)), lane l keys 4l..4l + 3, held in FP32
// across all tokens: per token s decays, kv = s . k, s += k beta (v - kv), o = s . q, both summed over the simdgroup.
// Rows from head_v_dim on stay zero and touch no memory but join every reduction. num_dv_groups is ignored.

static const uint DELTA_NET_PREFILL_THREADS = 128;
static const uint DELTA_NET_PREFILL_DV_PER_SIMDGROUP = 4;
static const uint DELTA_NET_PREFILL_SIMDGROUPS = DELTA_NET_PREFILL_THREADS / SIMD_SIZE;

[[Kernel]] [[Public]]
[[Precondition("new", "HEAD_K_DIM == crate::backends::vulkan::gpu_types::delta_net::DELTA_NET_HEAD_K_DIM")]]
[[Precondition("encode", "num_k_heads > 0")]]
[[Precondition("encode", "num_v_heads == 0 || num_v_heads >= num_k_heads")]]
[[Precondition("encode", "suffix_len <= 1 || num_v_heads == 0 || head_v_dim == 0 || num_v_heads <= value_dim / head_v_dim")]]
[[Precondition("encode", "suffix_len <= 1 || key_dim.is_multiple_of(4)")]]
func DeltaNetPrefill<T: IModelFloat>(
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
  [[Groups("if suffix_len == 0 { 0 } else { head_v_dim.div_ceil(16) }")]] uint row_group,
  [[Groups("num_v_heads")]] uint head_group,
  [[Threads("128")]] uint thread_idx,
  [[Specialize]] uint HEAD_K_DIM,
) {
  let invocation = subgroup_invocation();
  let simd = invocation / SIMD_SIZE;
  let lane = invocation % SIMD_SIZE;
  let hv = head_group;
  let dv_idx = (row_group * DELTA_NET_PREFILL_SIMDGROUPS + simd) * DELTA_NET_PREFILL_DV_PER_SIMDGROUP;
  // The simdgroup's rows below head_v_dim.
  let rows = dv_idx < head_v_dim ? min(head_v_dim - dv_idx, DELTA_NET_PREFILL_DV_PER_SIMDGROUP) : 0u;

  let hk = hv / (num_v_heads / num_k_heads);
  let conv_dim = 2 * key_dim + value_dim;
  let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
  let dk_base = lane * 4;

  var q_offset = hk * DELTA_NET_HEAD_K_DIM + dk_base;
  var k_offset = hk * DELTA_NET_HEAD_K_DIM + dk_base;
  var beta_offset = hv;
  var decay_offset = hv;
  var v_row = 2 * key_dim + hv * head_v_dim + dv_idx;
  var out_row = hv * head_v_dim + dv_idx;

  // State layout: [Hv, Dv, Dk], contiguous along Dk (DELTA_NET_HEAD_K_DIM floats per row).
  let s_in = (hv * head_v_dim + dv_idx) * DELTA_NET_HEAD_K_DIM + dk_base;
  var s0 = float4(0.0);
  var s1 = float4(0.0);
  var s2 = float4(0.0);
  var s3 = float4(0.0);
  if (rows > 0) { s0 = load_aligned<float4, 16>(state, s_in + 0 * DELTA_NET_HEAD_K_DIM); }
  if (rows > 1) { s1 = load_aligned<float4, 16>(state, s_in + 1 * DELTA_NET_HEAD_K_DIM); }
  if (rows > 2) { s2 = load_aligned<float4, 16>(state, s_in + 2 * DELTA_NET_HEAD_K_DIM); }
  if (rows > 3) { s3 = load_aligned<float4, 16>(state, s_in + 3 * DELTA_NET_HEAD_K_DIM); }

  for (uint token = 0; token < suffix_len; token++) {
    let decay = decay_buf[decay_offset];
    let beta = beta_buf[beta_offset];
    let k = load_aligned<float4, 16>(k_norm, k_offset);
    let q = load_aligned<float4, 16>(q_norm, q_offset);

    s0 *= decay;
    s1 *= decay;
    s2 *= decay;
    s3 *= decay;
    var kv_mem = float4(dot(s0, k), dot(s1, k), dot(s2, k), dot(s3, k));
    kv_mem = segment_reduce<VectorSum<4>, DELTA_NET_PREFILL_THREADS>(kv_mem, SIMD_SIZE);

    var v_val = float4(0.0);
    if (rows > 0) { v_val.x = in_proj[v_row + 0].toFloat(); }
    if (rows > 1) { v_val.y = in_proj[v_row + 1].toFloat(); }
    if (rows > 2) { v_val.z = in_proj[v_row + 2].toFloat(); }
    if (rows > 3) { v_val.w = in_proj[v_row + 3].toFloat(); }
    let delta = beta * (v_val - kv_mem);
    s0 += k * delta.x;
    s1 += k * delta.y;
    s2 += k * delta.z;
    s3 += k * delta.w;
    var o = float4(dot(s0, q), dot(s1, q), dot(s2, q), dot(s3, q));
    o = segment_reduce<VectorSum<4>, DELTA_NET_PREFILL_THREADS>(o, SIMD_SIZE);

    if (lane == 0) {
      if (rows > 0) { out[out_row + 0] = T.fromFloat(o.x); }
      if (rows > 1) { out[out_row + 1] = T.fromFloat(o.y); }
      if (rows > 2) { out[out_row + 2] = T.fromFloat(o.z); }
      if (rows > 3) { out[out_row + 3] = T.fromFloat(o.w); }
    }

    q_offset += key_dim;
    k_offset += key_dim;
    beta_offset += num_v_heads;
    decay_offset += num_v_heads;
    v_row += total_proj_dim;
    out_row += value_dim;
  }

  if (rows > 0) { store_aligned<16>(state, s_in + 0 * DELTA_NET_HEAD_K_DIM, s0); }
  if (rows > 1) { store_aligned<16>(state, s_in + 1 * DELTA_NET_HEAD_K_DIM, s1); }
  if (rows > 2) { store_aligned<16>(state, s_in + 2 * DELTA_NET_HEAD_K_DIM, s2); }
  if (rows > 3) { store_aligned<16>(state, s_in + 3 * DELTA_NET_HEAD_K_DIM, s3); }
}
