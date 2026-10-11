import definitions;
import common.activation;
import common.storage_address;
import gdn.prefill_prep;

// The CPU StateAdvance along the accepted path. One invocation owns state row (hv, dv) of the [num_v_heads, 128, 128]
// FP32 state, held privately from one load to one store, for k head hk = hv / floor(num_v_heads / num_k_heads) (the
// CPU's mapping, also for nondivisible ratios). For each accepted token t in order: d = e^log_decay[t, hv]
// (delta_net_exp, NaN and infinities decided on the bits); kv from +0 over dk ascending of s_dk <- s_dk d then
// s_dk k[t, hk, dk]; delta = beta[t, hv] (v[t, hv, dv] - kv); then over dk ascending s_dk <- s_dk + k[t, hk, dk] delta.
// Each product and sum is one IEEE FP32 operation, never fused. Rows are disjoint and every other buffer is read-only,
// so no barrier is needed. Element offsets are 64-bit words formed one term at a time. Workgroups stride over the two
// row blocks and the heads by the actual NumWorkgroups, which the host clamps to the device limits, advancing only
// while more than one step remains.

static const uint STATE_ADVANCE_THREADS = 64;
static const uint STATE_ADVANCE_DIM = 128;

[[Kernel]] [[Public]]
[[Precondition("new", "HEAD_K_DIM == 128")]]
[[Precondition("new", "num_k_heads != 0 && (num_v_heads == 0 || num_v_heads >= num_k_heads)")]]
func StateAdvance<T: IModelFloat>(
  Ptr<T, Access.Read> k_norm,
  Ptr<T, Access.Read> v,
  Ptr<float, Access.Read> log_decay_buf,
  Ptr<float, Access.Read> beta_buf,
  Ptr<uint, Access.Read> accepted_indices,
  Ptr<float, Access.ReadWrite> state,
  uint accepted_len,
  [[Groups("if accepted_len == 0 || num_v_heads == 0 { 0 } else { 2u32.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint row_group,
  [[Groups("num_v_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint head_group,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] uint HEAD_K_DIM,
  [[Specialize]] uint num_v_heads,
  [[Specialize]] uint num_k_heads,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let blocks = STATE_ADVANCE_DIM / STATE_ADVANCE_THREADS;
  let heads_per_group = num_v_heads / num_k_heads;
  var hv = head_group;
  while (hv < num_v_heads) {
    let hk = hv / heads_per_group;
    var block = row_group;
    while (block < blocks) {
      let dv = block * STATE_ADVANCE_THREADS + thread_idx;
      let state_row = delta_net_state_row(state, hv, STATE_ADVANCE_DIM, dv);
      float row[STATE_ADVANCE_DIM];
      for (uint dk = 0; dk < STATE_ADVANCE_DIM; dk++) {
        row[dk] = element_pointer(state_row, uint2(dk, 0))[0];
      }
      for (uint i = 0; i < accepted_len; i++) {
        let t = element_pointer(accepted_indices, uint2(i, 0))[0];
        let lane = element_index(t, num_v_heads, hv);
        let d = delta_net_exp(element_pointer(log_decay_buf, lane)[0]);
        let k = element_pointer(k_norm, byte_offset(element_index(t, num_k_heads, hk), STATE_ADVANCE_DIM));
        var kv = 0.0;
        for (uint dk = 0; dk < STATE_ADVANCE_DIM; dk++) {
          row[dk] = exact_product(row[dk], d);
          kv = ordered_sum(kv, exact_product(row[dk], element_pointer(k, uint2(dk, 0))[0].toFloat()));
        }
        let value = element_pointer(v, add_address(byte_offset(lane, STATE_ADVANCE_DIM), uint2(dv, 0)))[0].toFloat();
        let delta = exact_product(element_pointer(beta_buf, lane)[0], ordered_sum(value, -kv));
        for (uint dk = 0; dk < STATE_ADVANCE_DIM; dk++) {
          row[dk] = ordered_sum(row[dk], exact_product(element_pointer(k, uint2(dk, 0))[0].toFloat(), delta));
        }
        if (accepted_len - i <= 1) {
          break;
        }
      }
      for (uint dk = 0; dk < STATE_ADVANCE_DIM; dk++) {
        element_pointer(state_row, uint2(dk, 0))[0] = row[dk];
      }
      if (blocks - block <= groups.x) {
        break;
      }
      block += groups.x;
    }
    if (num_v_heads - hv <= groups.y) {
      break;
    }
    hv += groups.y;
  }
}
