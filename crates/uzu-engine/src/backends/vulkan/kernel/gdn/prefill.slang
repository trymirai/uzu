import definitions;
import common.activation;
import common.storage_address;
import gdn.prefill_prep;

// The CPU DeltaNetPrefill, one invocation per (v head hv, row i) of k head hk = hv / floor(H / G), over the tokens in
// order. For token t, with decay and beta at t num_v_heads + hv and q and k at t key_dim + hk 128, the state row
// s = state[hv, i, :] gives kv = sum_j (decay s_j) k_j from +0, delta = beta (v - kv) with v at
// in_proj[t total_proj_dim + 2 key_dim + hv head_v_dim + i], then in ascending j s_j <- decay s_j + k_j delta and
// o += s_j q_j from +0 with the new s_j, and out[t value_dim + hv head_v_dim + i] = T(o). Each product and sum is one
// IEEE FP32 operation, never fused; the invocation reads each state element before writing it. num_dv_groups is
// unused, as on the CPU. Element offsets are 64-bit words added one term at a time, so total_proj_dim is never formed.
// Workgroups stride over row blocks and heads by the actual NumWorkgroups, which the host clamps to the device limits,
// advancing only while more than one step remains. Tokens' output rows are disjoint only when value_dim holds the heads,
// which the last precondition requires whenever several tokens exist; the bound is a quotient, free of overflow.

static const uint DELTA_NET_PREFILL_THREADS = 64;

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
  [[Groups("if suffix_len == 0 || num_v_heads == 0 || head_v_dim == 0 { 0 } else { head_v_dim.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint row_group,
  [[Groups("num_v_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint head_group,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] uint HEAD_K_DIM,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let heads_per_group = num_v_heads / num_k_heads;
  let blocks = head_v_dim / DELTA_NET_PREFILL_THREADS + (head_v_dim % DELTA_NET_PREFILL_THREADS != 0 ? 1u : 0u);
  var hv = head_group;
  while (hv < num_v_heads) {
    let hk = hv / heads_per_group;
    var block = row_group;
    while (block < blocks) {
      if (thread_idx < head_v_dim - block * DELTA_NET_PREFILL_THREADS) {
        let i = block * DELTA_NET_PREFILL_THREADS + thread_idx;
        let row = delta_net_state_row(state, hv, head_v_dim, i);
        let element = element_index(hv, head_v_dim, i);
        for (uint token = 0; token < suffix_len; token++) {
          let lane = element_index(token, num_v_heads, hv);
          let d = element_pointer(decay_buf, lane)[0];
          let b = element_pointer(beta_buf, lane)[0];
          let head = element_index(hk, 128, 0);
          let q = element_pointer(element_pointer(q_norm, element_index(token, key_dim, 0)), head);
          let k = element_pointer(element_pointer(k_norm, element_index(token, key_dim, 0)), head);
          var kv = 0.0;
          for (uint j = 0; j < 128; j++) {
            let s = element_pointer(row, uint2(j, 0))[0];
            kv = ordered_sum(kv, exact_product(exact_product(d, s), element_pointer(k, uint2(j, 0))[0]));
          }
          let values = delta_net_rows(delta_net_rows(in_proj, token, uint3(key_dim, value_dim, num_v_heads)), 1, uint3(key_dim, 0, 0));
          let delta = exact_product(b, ordered_sum(element_pointer(values, element)[0].toFloat(), -kv));
          var o = 0.0;
          for (uint j = 0; j < 128; j++) {
            let s = element_pointer(row, uint2(j, 0));
            let next = ordered_sum(exact_product(d, s[0]), exact_product(element_pointer(k, uint2(j, 0))[0], delta));
            s[0] = next;
            o = ordered_sum(o, exact_product(next, element_pointer(q, uint2(j, 0))[0]));
          }
          element_pointer(element_pointer(out, element_index(token, value_dim, 0)), element)[0] = T.fromFloat(o);
        }
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
