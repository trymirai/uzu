import definitions;
import common.activation;
import common.storage_address;
import common.workgroup_reduction;
import generated.activation_type;
import gdn.prefill_prep;
import ssm.ssd_update;

// The CPU DeltaNetUpdate of one token, one workgroup per v head hv of k head hk = hv / (H / G). Its q and k rows, at
// in_proj[hk 128] and key_dim past it, are L2-normalized as DeltaNetPrefillPrep's (q to (v inv) qs, k to v inv) into
// shared memory, and invocation 0 forms kq = k . q from -0, the CPU sum's start. beta is sigmoid(beta_raw) and the
// decay e^(-(e^a_log) softplus(a_raw + dt_bias)). State row i of head hv first takes the dots sq = s . q and sk = s . k
// from +0, then retrieved = decay sk, delta = beta (v_i - retrieved) and o_i = decay sq + delta kq, then
// s <- decay s + k delta; the output is T(((o_i inv) norm_weight[i]) silu(z_i)), inv the reciprocal RMS of the FP32 o
// (rms_inverse). Each product and sum is one IEEE FP32 operation, never fused. A first pass computes o and its squares
// from the unmodified rows; the second recomputes the same o and delta, writes the rows and the output. One invocation
// owns each row, striding by the workgroup and advancing only while more than one step remains. Element offsets are
// 64-bit words added one term at a time. Workgroups stride over heads by the actual NumWorkgroups, which the host clamps
// to the device limit, so the loop holding the barriers is uniform; its last barrier ends every read of the shared q, k
// and kq before the next head overwrites them.

static const uint DELTA_NET_UPDATE_THREADS = 128;

groupshared float delta_net_q[DELTA_NET_UPDATE_THREADS];
groupshared float delta_net_k[DELTA_NET_UPDATE_THREADS];
groupshared float delta_net_kq;

// o_i and delta_i of the state row `row`, in the CPU's order.
float2 delta_net_update_row(Ptr<float, Access.ReadWrite> row, float v, float beta, float decay, float kq) {
  var sq = 0.0;
  var sk = 0.0;
  for (uint j = 0; j < DELTA_NET_UPDATE_THREADS; j++) {
    let s = element_pointer(row, uint2(j, 0))[0];
    sq = ordered_sum(sq, exact_product(s, delta_net_q[j]));
    sk = ordered_sum(sk, exact_product(s, delta_net_k[j]));
  }
  let delta = exact_product(beta, ordered_sum(v, -exact_product(decay, sk)));
  return float2(ordered_sum(exact_product(decay, sq), exact_product(delta, kq)), delta);
}

[[Kernel]] [[Public]]
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
  [[Threads("128")]] uint tid,
  [[Specialize]] uint HEAD_K_DIM,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let values = delta_net_rows(in_proj, 1, uint3(key_dim, 0, 0));
  let gates = element_pointer(values, uint2(value_dim, 0));
  let raw = element_pointer(gates, uint2(value_dim, 0));
  var hv = head_group;
  while (hv < num_v_heads) {
    let hk = hv / (num_v_heads / num_k_heads);
    let q_row = element_pointer(in_proj, element_index(hk, DELTA_NET_UPDATE_THREADS, 0));
    let k_row = element_pointer(element_pointer(in_proj, uint2(key_dim, 0)), element_index(hk, DELTA_NET_UPDATE_THREADS, 0));
    let q_inv = delta_net_inverse(q_row, tid);
    let k_inv = delta_net_inverse(k_row, tid);
    delta_net_q[tid] = delta_net_query(element_pointer(q_row, uint2(tid, 0))[0].toFloat(), q_inv);
    delta_net_k[tid] = exact_product(element_pointer(k_row, uint2(tid, 0))[0].toFloat(), k_inv);
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) {
      var kq = asfloat(0x80000000u);
      for (uint j = 0; j < DELTA_NET_UPDATE_THREADS; j++) {
        kq = ordered_sum(kq, exact_product(delta_net_k[j], delta_net_q[j]));
      }
      delta_net_kq = kq;
    }
    GroupMemoryBarrierWithGroupSync();
    let kq = delta_net_kq;
    let beta = activate_silu_alpha<true>(1.0, element_pointer(raw, uint2(hv, 0))[0].toFloat());
    let a_raw = element_pointer(raw, element_index(1, num_v_heads, hv))[0].toFloat();
    let decay = ssd_decay(
      -delta_net_log_decay(element_pointer(a_log, uint2(hv, 0))[0], a_raw, element_pointer(dt_bias, uint2(hv, 0))[0])
    );
    float squares = 0.0;
    if (tid < head_v_dim) {
      var i = tid;
      while (true) {
        let v = element_pointer(values, element_index(hv, head_v_dim, i))[0].toFloat();
        let o = delta_net_update_row(delta_net_state_row(state, hv, head_v_dim, i), v, beta, decay, kq).x;
        squares = accumulate_squares(squares, o);
        if (head_v_dim - i <= DELTA_NET_UPDATE_THREADS) {
          break;
        }
        i += DELTA_NET_UPDATE_THREADS;
      }
    }
    let inv = rms_inverse<DELTA_NET_UPDATE_THREADS>(squares, head_v_dim, norm_epsilon);
    if (tid < head_v_dim) {
      var i = tid;
      while (true) {
        let element = element_index(hv, head_v_dim, i);
        let row = delta_net_state_row(state, hv, head_v_dim, i);
        let update = delta_net_update_row(row, element_pointer(values, element)[0].toFloat(), beta, decay, kq);
        for (uint j = 0; j < DELTA_NET_UPDATE_THREADS; j++) {
          let s = element_pointer(row, uint2(j, 0));
          s[0] = ordered_sum(exact_product(decay, s[0]), exact_product(delta_net_k[j], update.y));
        }
        let gate = activate(element_pointer(gates, element)[0].toFloat(), ActivationType.SILU);
        let scaled = exact_product(exact_product(update.x, inv), element_pointer(norm_weight, uint2(i, 0))[0]);
        element_pointer(out, element)[0] = T.fromFloat(exact_product(scaled, gate));
        if (head_v_dim - i <= DELTA_NET_UPDATE_THREADS) {
          break;
        }
        i += DELTA_NET_UPDATE_THREADS;
      }
    }
    GroupMemoryBarrierWithGroupSync();
    if (num_v_heads - hv <= groups.x) {
      break;
    }
    hv += groups.x;
  }
}
