import definitions;
import common.activation;
import common.storage_address;
import common.workgroup_reduction;
import generated.activation_type;
import gdn.prefill_prep;

// Metal's coalesced state layout [Hv, Dv, 128]: a 32-lane simdgroup owns each row, lane l holding keys l + 32*i.
// Rows below 128 are read and updated once, with their output saved for the head-wide RMS.
//
// Vulkan uses 512 threads where supported, otherwise 128. Non-32 subgroups reduce logical lanes through shared
// memory; masked rows join every barrier. Normalized q/k and exact scalar stages reuse PrefillPrep's helpers.
// Rows from 128 on defer state writes and are recomputed after RMS. Heads stride by actual NumWorkgroups,
// with 64-bit element offsets and a final barrier before the next head overwrites shared memory.

static const uint DELTA_NET_UPDATE_KEYS = 128;
static const uint DELTA_NET_UPDATE_SIMD_SIZE = 32;
static const uint DELTA_NET_UPDATE_MAX_THREADS = 512;
// Metal's UPDATE_MAX_HEAD_V_DIM, a multiple of every simdgroup count.
static const uint DELTA_NET_UPDATE_SHARED_ROWS = 128;

groupshared float delta_net_q[DELTA_NET_UPDATE_KEYS];
groupshared float delta_net_k[DELTA_NET_UPDATE_KEYS];
groupshared float delta_net_kq;
groupshared float delta_net_o[DELTA_NET_UPDATE_SHARED_ROWS];
// Vulkan only: the logical simdgroups' lanes when they are not subgroup lanes, [simd * 32 + lane].
groupshared float2 delta_net_update_lanes[DELTA_NET_UPDATE_MAX_THREADS];

// Metal's simd_sum of (sq, sk) over the 32 lanes of logical simdgroup `simd`; without native subgroups, the lanes'
// values summed in ascending lane order through shared memory, every invocation at both barriers.
float2 delta_net_update_sum(float2 value, bool native, uint simd, uint lane) {
  if (native) {
    return WaveActiveSum(value);
  }
  let base = simd * DELTA_NET_UPDATE_SIMD_SIZE;
  delta_net_update_lanes[base + lane] = value;
  GroupMemoryBarrierWithGroupSync();
  var sum = float2(0.0);
  for (uint other = 0; other < DELTA_NET_UPDATE_SIMD_SIZE; other++) {
    sum += delta_net_update_lanes[base + other];
  }
  GroupMemoryBarrierWithGroupSync();
  return sum;
}

// o_i of the state row `row` from the lane's keys q and k, storing the updated row with `write`. Every invocation must
// call it; a masked one (`valid` false) accesses nothing and adds zeros.
float delta_net_update_row(
  Ptr<float, Access.ReadWrite> row,
  bool valid,
  bool write,
  float v,
  float beta,
  float decay,
  float kq,
  float4 q,
  float4 k,
  bool native,
  uint simd,
  uint lane,
) {
  var s = float4(0.0);
  var sq = 0.0;
  var sk = 0.0;
  for (uint i = 0; i < 4; i++) {
    if (valid) {
      s[i] = element_pointer(row, uint2(lane + DELTA_NET_UPDATE_SIMD_SIZE * i, 0))[0];
    }
    sq += s[i] * q[i];
    sk += s[i] * k[i];
  }
  let dots = delta_net_update_sum(float2(sq, sk), native, simd, lane);
  let delta = exact_product(beta, ordered_sum(v, -exact_product(decay, dots.y)));
  if (valid && write) {
    for (uint i = 0; i < 4; i++) {
      element_pointer(row, uint2(lane + DELTA_NET_UPDATE_SIMD_SIZE * i, 0))[0] = decay * s[i] + k[i] * delta;
    }
  }
  return ordered_sum(exact_product(decay, dots.x), exact_product(delta, kq));
}

[[Kernel]]
[[Precondition("new", "UPDATE_THREADS == 128 || UPDATE_THREADS == 512")]]
[[Precondition("encode", "HEAD_K_DIM == 128")]]
[[Precondition("encode", "num_v_heads == 0 || num_k_heads > 0 && num_v_heads.is_multiple_of(num_k_heads)")]]
func DeltaNetUpdate<T: IModelFloat>(
  Ptr<T, Access.Read> in_proj,
  Ptr<float, Access.Read> a_log,
  Ptr<float, Access.Read> dt_bias,
  Ptr<float, Access.Read> norm_weight,
  Ptr<float, Access.ReadWrite> state,
  Ptr<T, Access.ReadWrite> out,
  uint num_v_heads,
  uint num_k_heads,
  uint head_v_dim,
  uint key_dim,
  uint value_dim,
  float norm_epsilon,
  [[Groups("if num_v_heads == 0 || head_v_dim == 0 { 0 } else { num_v_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint head_group,
  [[Threads("UPDATE_THREADS")]] uint tid,
  [[Specialize]] uint HEAD_K_DIM,
  [[Specialize]] uint UPDATE_THREADS,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let simdgroups = UPDATE_THREADS / DELTA_NET_UPDATE_SIMD_SIZE;
  let native = subgroups_full32(simdgroups);
  let simd = native ? subgroup_id() : tid / DELTA_NET_UPDATE_SIMD_SIZE;
  let lane = native ? WaveGetLaneIndex() : tid % DELTA_NET_UPDATE_SIMD_SIZE;
  let batches = head_v_dim / simdgroups + (head_v_dim % simdgroups != 0 ? 1u : 0u);
  let values = delta_net_rows(in_proj, 1, uint3(key_dim, 0, 0));
  let gates = element_pointer(values, uint2(value_dim, 0));
  let raw = element_pointer(gates, uint2(value_dim, 0));
  var hv = head_group;
  while (hv < num_v_heads) {
    let hk = hv / (num_v_heads / num_k_heads);
    let q_row = element_pointer(in_proj, element_index(hk, DELTA_NET_UPDATE_KEYS, 0));
    let k_row = element_pointer(element_pointer(in_proj, uint2(key_dim, 0)), element_index(hk, DELTA_NET_UPDATE_KEYS, 0));
    let q_inv = delta_net_inverse(q_row, tid, UPDATE_THREADS);
    let k_inv = delta_net_inverse(k_row, tid, UPDATE_THREADS);
    if (tid < DELTA_NET_UPDATE_KEYS) {
      delta_net_q[tid] = delta_net_query(element_pointer(q_row, uint2(tid, 0))[0].toFloat(), q_inv);
      delta_net_k[tid] = exact_product(element_pointer(k_row, uint2(tid, 0))[0].toFloat(), k_inv);
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) {
      var kq = asfloat(0x80000000u);
      for (uint j = 0; j < DELTA_NET_UPDATE_KEYS; j++) {
        kq = ordered_sum(kq, exact_product(delta_net_k[j], delta_net_q[j]));
      }
      delta_net_kq = kq;
    }
    GroupMemoryBarrierWithGroupSync();
    let kq = delta_net_kq;
    var q = float4(0.0);
    var k = float4(0.0);
    for (uint i = 0; i < 4; i++) {
      q[i] = delta_net_q[lane + DELTA_NET_UPDATE_SIMD_SIZE * i];
      k[i] = delta_net_k[lane + DELTA_NET_UPDATE_SIMD_SIZE * i];
    }
    let beta = activate_silu_alpha<true>(1.0, element_pointer(raw, uint2(hv, 0))[0].toFloat());
    let a_raw = element_pointer(raw, element_index(1, num_v_heads, hv))[0].toFloat();
    let decay = delta_net_decay(
      -delta_net_log_decay(element_pointer(a_log, uint2(hv, 0))[0], a_raw, element_pointer(dt_bias, uint2(hv, 0))[0])
    );
    // Row i = batch simdgroups + simd, valid below head_v_dim; a masked row stands in row 0 for addresses never accessed.
    // Batches below 128 / simdgroups hold only saved rows, the others none.
    float squares = 0.0;
    for (uint batch = 0; batch < batches; batch++) {
      let valid = simd < head_v_dim - batch * simdgroups;
      let i = valid ? batch * simdgroups + simd : 0u;
      let saved = batch < DELTA_NET_UPDATE_SHARED_ROWS / simdgroups;
      var v = 0.0;
      if (valid) {
        v = element_pointer(values, element_index(hv, head_v_dim, i))[0].toFloat();
      }
      let o = delta_net_update_row(
        delta_net_state_row(state, hv, head_v_dim, i), valid, saved, v, beta, decay, kq, q, k, native, simd, lane
      );
      if (valid && lane == 0) {
        if (saved) {
          delta_net_o[i] = o;
        }
        squares = accumulate_squares(squares, o);
      }
    }
    let inv = delta_net_rms_inverse(squares, head_v_dim, norm_epsilon, UPDATE_THREADS);
    for (uint batch = 0; batch < batches; batch++) {
      let valid = simd < head_v_dim - batch * simdgroups;
      let i = valid ? batch * simdgroups + simd : 0u;
      var o = 0.0;
      if (batch < DELTA_NET_UPDATE_SHARED_ROWS / simdgroups) {
        if (valid) {
          o = delta_net_o[i];
        }
      } else {
        var v = 0.0;
        if (valid) {
          v = element_pointer(values, element_index(hv, head_v_dim, i))[0].toFloat();
        }
        o = delta_net_update_row(
          delta_net_state_row(state, hv, head_v_dim, i), valid, true, v, beta, decay, kq, q, k, native, simd, lane
        );
      }
      if (valid && lane == 0) {
        let element = element_index(hv, head_v_dim, i);
        let gate = activate(element_pointer(gates, element)[0].toFloat(), ActivationType.SILU);
        let scaled = exact_product(exact_product(o, inv), element_pointer(norm_weight, uint2(i, 0))[0]);
        element_pointer(out, element)[0] = T.fromFloat(exact_product(scaled, gate));
      }
    }
    GroupMemoryBarrierWithGroupSync();
    if (num_v_heads - hv <= groups.x) {
      break;
    }
    hv += groups.x;
  }
}
