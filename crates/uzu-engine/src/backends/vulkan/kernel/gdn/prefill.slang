import definitions;
import common.storage_address;
import gdn.prefill_prep;

// Metal DeltaNetPrefill: in a 128-invocation workgroup, each of four logical 32-lane simdgroups owns four value rows of
// v head hv, k head hk = hv / floor(H / G), and lane l the keys 4l..4l + 3 of each row, held as four FP32 vectors from
// before the first token until after the last. Per token the vectors are decayed, kv = s . k summed over the simdgroup,
// delta = beta (v - kv), s += k delta, and o = s . q summed likewise, which lane 0 writes as T; the state is stored once
// after the last token. Rows from head_v_dim on are masked: their vectors stay zero and nothing of theirs is read or
// written, but their lanes join every reduction. num_dv_groups is ignored; the value groups are head_v_dim.div_ceil(16).
//
// Vulkan differences: element offsets are 64-bit words added one term at a time, so total_proj_dim is never formed, and
// the vector accesses promise only float's alignment. Full 32-lane subgroups reduce with WaveActiveSum; other widths keep
// the same logical lanes and reduce through shared memory. Workgroups stride over value groups and heads by the actual
// NumWorkgroups, which the host clamps to the device limits, so the loops holding barriers are uniform, advancing only
// while more than one step remains. Tokens' output rows are disjoint only when value_dim holds the heads, which the last
// precondition requires whenever several tokens exist; the bound is a quotient, free of overflow.

static const uint DELTA_NET_PREFILL_THREADS = 128;
static const uint DELTA_NET_PREFILL_SIMD_SIZE = 32;
static const uint DELTA_NET_PREFILL_DV_PER_SIMDGROUP = 4;
static const uint DELTA_NET_PREFILL_SIMDGROUPS = DELTA_NET_PREFILL_THREADS / DELTA_NET_PREFILL_SIMD_SIZE;
static const uint DELTA_NET_PREFILL_ROWS = DELTA_NET_PREFILL_SIMDGROUPS * DELTA_NET_PREFILL_DV_PER_SIMDGROUP;

// Vulkan only: the logical simdgroups' lanes when they are not subgroup lanes, [simd * 32 + lane].
groupshared float4 delta_net_prefill_lanes[DELTA_NET_PREFILL_THREADS];

// Metal's simd_sum over the 32 lanes of logical simdgroup `simd`; without native subgroups, the lanes' values summed in
// ascending lane order through shared memory, every invocation at both barriers.
float4 delta_net_prefill_sum(float4 value, bool native, uint simd, uint lane) {
  if (native) {
    return WaveActiveSum(value);
  }
  let base = simd * DELTA_NET_PREFILL_SIMD_SIZE;
  delta_net_prefill_lanes[base + lane] = value;
  GroupMemoryBarrierWithGroupSync();
  var sum = float4(0.0);
  for (uint other = 0; other < DELTA_NET_PREFILL_SIMD_SIZE; other++) {
    sum += delta_net_prefill_lanes[base + other];
  }
  GroupMemoryBarrierWithGroupSync();
  return sum;
}

[[Kernel]] [[Public]]
[[Precondition("encode", "HEAD_K_DIM == 128")]]
[[Precondition("encode", "num_k_heads > 0")]]
[[Precondition("encode", "num_v_heads == 0 || num_v_heads >= num_k_heads")]]
[[Precondition("encode", "suffix_len <= 1 || num_v_heads == 0 || head_v_dim == 0 || num_v_heads <= value_dim / head_v_dim")]]
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
  [[Groups("if suffix_len == 0 || num_v_heads == 0 || head_v_dim == 0 { 0 } else { head_v_dim.div_ceil(16).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint row_group,
  [[Groups("num_v_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint head_group,
  [[Threads("128")]] uint thread_idx,
  [[Specialize]] uint HEAD_K_DIM,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let native = subgroups_full32(DELTA_NET_PREFILL_SIMDGROUPS);
  let simd = native ? subgroup_id() : thread_idx / DELTA_NET_PREFILL_SIMD_SIZE;
  let lane = native ? WaveGetLaneIndex() : thread_idx % DELTA_NET_PREFILL_SIMD_SIZE;
  let dk_base = lane * 4;
  let heads_per_group = num_v_heads / num_k_heads;
  let blocks = head_v_dim / DELTA_NET_PREFILL_ROWS + (head_v_dim % DELTA_NET_PREFILL_ROWS != 0 ? 1u : 0u);
  var hv = head_group;
  while (hv < num_v_heads) {
    let hk = hv / heads_per_group;
    var dv_group = row_group;
    while (dv_group < blocks) {
      // The simdgroup's rows below head_v_dim; with none, row 0 of the head stands in for addresses never accessed.
      let remaining = head_v_dim - dv_group * DELTA_NET_PREFILL_ROWS;
      let first = simd * DELTA_NET_PREFILL_DV_PER_SIMDGROUP;
      let rows = first < remaining ? min(remaining - first, DELTA_NET_PREFILL_DV_PER_SIMDGROUP) : 0u;
      let dv_idx = rows > 0 ? dv_group * DELTA_NET_PREFILL_ROWS + first : 0u;

      var q_ptr = element_pointer(q_norm, element_index(hk, 128, dk_base));
      var k_ptr = element_pointer(k_norm, element_index(hk, 128, dk_base));
      var beta_ptr = element_pointer(beta_buf, uint2(hv, 0));
      var decay_ptr = element_pointer(decay_buf, uint2(hv, 0));
      var v_row = element_pointer(delta_net_rows(in_proj, 1, uint3(key_dim, 0, 0)), element_index(hv, head_v_dim, dv_idx));
      var out_row = element_pointer(out, element_index(hv, head_v_dim, dv_idx));

      // State layout: [Hv, Dv, Dk], contiguous along Dk (128 floats per row). Unrolled by hand as Metal's.
      let s_in = element_pointer(delta_net_state_row(state, hv, head_v_dim, dv_idx), uint2(dk_base, 0));
      var s0 = float4(0.0);
      var s1 = float4(0.0);
      var s2 = float4(0.0);
      var s3 = float4(0.0);
      if (rows > 0) { s0 = load_float4(s_in); }
      if (rows > 1) { s1 = load_float4(element_pointer(s_in, uint2(1 * 128, 0))); }
      if (rows > 2) { s2 = load_float4(element_pointer(s_in, uint2(2 * 128, 0))); }
      if (rows > 3) { s3 = load_float4(element_pointer(s_in, uint2(3 * 128, 0))); }

      for (uint token = 0; token < suffix_len; token++) {
        let decay = decay_ptr[0];
        let beta = beta_ptr[0];
        let k = load_float4(k_ptr);
        let q = load_float4(q_ptr);

        s0 *= decay;
        s1 *= decay;
        s2 *= decay;
        s3 *= decay;
        var kv_mem = float4(dot(s0, k), dot(s1, k), dot(s2, k), dot(s3, k));
        kv_mem = delta_net_prefill_sum(kv_mem, native, simd, lane);

        var v_val = float4(0.0);
        if (rows > 0) { v_val.x = v_row[0].toFloat(); }
        if (rows > 1) { v_val.y = element_pointer(v_row, uint2(1, 0))[0].toFloat(); }
        if (rows > 2) { v_val.z = element_pointer(v_row, uint2(2, 0))[0].toFloat(); }
        if (rows > 3) { v_val.w = element_pointer(v_row, uint2(3, 0))[0].toFloat(); }
        let delta = beta * (v_val - kv_mem);
        s0 += k * delta.x;
        s1 += k * delta.y;
        s2 += k * delta.z;
        s3 += k * delta.w;
        var o = float4(dot(s0, q), dot(s1, q), dot(s2, q), dot(s3, q));
        o = delta_net_prefill_sum(o, native, simd, lane);

        if (lane == 0) {
          if (rows > 0) { out_row[0] = T.fromFloat(o.x); }
          if (rows > 1) { element_pointer(out_row, uint2(1, 0))[0] = T.fromFloat(o.y); }
          if (rows > 2) { element_pointer(out_row, uint2(2, 0))[0] = T.fromFloat(o.z); }
          if (rows > 3) { element_pointer(out_row, uint2(3, 0))[0] = T.fromFloat(o.w); }
        }

        q_ptr = element_pointer(q_ptr, uint2(key_dim, 0));
        k_ptr = element_pointer(k_ptr, uint2(key_dim, 0));
        beta_ptr = element_pointer(beta_ptr, uint2(num_v_heads, 0));
        decay_ptr = element_pointer(decay_ptr, uint2(num_v_heads, 0));
        v_row = delta_net_rows(v_row, 1, uint3(key_dim, value_dim, num_v_heads));
        out_row = element_pointer(out_row, uint2(value_dim, 0));
      }

      if (rows > 0) { store_float4(s_in, s0); }
      if (rows > 1) { store_float4(element_pointer(s_in, uint2(1 * 128, 0)), s1); }
      if (rows > 2) { store_float4(element_pointer(s_in, uint2(2 * 128, 0)), s2); }
      if (rows > 3) { store_float4(element_pointer(s_in, uint2(3 * 128, 0)), s3); }

      if (blocks - dv_group <= groups.x) {
        break;
      }
      dv_group += groups.x;
    }
    if (num_v_heads - hv <= groups.y) {
      break;
    }
    hv += groups.y;
  }
}
