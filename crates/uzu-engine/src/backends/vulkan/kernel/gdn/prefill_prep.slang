import definitions;
import common.activation;
import common.storage_address;
import common.workgroup_reduction;
import generated.activation_type;
import ssm.ssd_update;

// The CPU DeltaNetPrefillPrep, one workgroup per (token, k head). The q and k rows of the head, at
// in_proj[t total_proj_dim + hk 128] and key_dim past it, are L2-normalized by 1 / sqrt(sum + 1e-6), the sum of squares
// sequential from +0 in the CPU's order (delta_net_inverse), and stored as QKT((v inv) qs) and QKT(v inv) at
// t key_dim + hk 128. Its v heads hv = hk floor(H / G) + group get beta = sigmoid(beta_raw) and the log decay
// -(e^a_log) softplus(a_raw + dt_bias), stored as it is or as its exponential, at t num_v_heads + hv; the lanes from
// G floor(H / G) on stay untouched. With write_compact_v, the token's value_dim raw T values are copied to
// t value_dim, in 128-element blocks hk, hk + G, and so on. Each product and sum is one IEEE FP32 operation, never
// fused. total_proj_dim = 2 key_dim + 2 value_dim + 2 num_v_heads is never formed: element offsets are 64-bit words
// added one term at a time. Workgroups stride over heads and tokens by the actual NumWorkgroups, which the host clamps
// to the device limits, so the loops holding the reductions' barriers are uniform; the others stride by the workgroup,
// advancing only while more than one step remains. Tokens' q and k rows are disjoint only when key_dim holds the G heads,
// which the last precondition requires whenever several tokens exist.

static const uint DELTA_NET_PREP_THREADS = 128;

// e^a in the CPU's single expf step: NaN stays NaN, +inf gives +inf, otherwise ssd_decay's e^(-(-a)), so -inf gives +0
// and finite a the builtin exponential, whose subnormal results may flush to +0.
public float delta_net_exp(float a) {
  return asuint(a) == 0x7f800000u ? a : ssd_decay(-a);
}

// The CPU's -(e^a_log) softplus(a_raw + dt_bias), each step one FP32 rounding; the negation is exact.
public float delta_net_log_decay(float a_log, float a_raw, float dt_bias) {
  return exact_product(-delta_net_exp(a_log), activate(ordered_sum(a_raw, dt_bias), ActivationType.SOFTPLUS));
}

// The CPU's (v inv) qs, with qs its FP32 1 / sqrt(128) (bits 0x3db504f3), each product one rounding.
public float delta_net_query(float v, float inv) {
  return exact_product(exact_product(v, inv), asfloat(0x3db504f3u));
}

// The element 2 row (terms.x + terms.y + terms.z) past `base`, one 64-bit term at a time.
public Ptr<T, access> delta_net_rows<T, Access access>(Ptr<T, access> base, uint row, uint3 terms) {
  var pointer = base;
  for (uint term = 0; term < 3; term++) {
    let stride = element_index(row, terms[term], 0);
    pointer = element_pointer(element_pointer(pointer, stride), stride);
  }
  return pointer;
}

// State row (head, i) of the [num_v_heads, head_v_dim, 128] FP32 state: the 64-bit row index times 128.
public Ptr<float, access> delta_net_state_row<Access access>(Ptr<float, access> state, uint head, uint head_v_dim, uint i) {
  return element_pointer(state, byte_offset(element_index(head, head_v_dim, i), 128));
}

// 1 / sqrt(sum + 1e-6) for the sum of squares of the 128 values at `row`, formed by invocation 0 in order from +0 as
// the CPU's; every other partial is +0, so the workgroup sum is exact. Every invocation must call it.
public float delta_net_inverse<T: IModelFloat>(Ptr<T, Access.Read> row, uint tid) {
  float squares = 0.0;
  if (tid == 0) {
    for (uint j = 0; j < DELTA_NET_PREP_THREADS; j++) {
      squares = accumulate_squares(squares, element_pointer(row, uint2(j, 0))[0].toFloat());
    }
  }
  return rms_inverse<DELTA_NET_PREP_THREADS>(squares, 1, 1e-6);
}

[[Kernel]] [[Public]]
[[Precondition("encode", "HEAD_K_DIM == 128")]]
[[Precondition("encode", "num_k_heads > 0")]]
[[Precondition("encode", "suffix_len <= 1 || num_k_heads <= key_dim / 128")]]
public func DeltaNetPrefillPrep<T: IModelFloat, QKT: IModelFloat>(
  Ptr<T, Access.Read> in_proj,
  Ptr<float, Access.Read> a_log,
  Ptr<float, Access.Read> dt_bias,
  Ptr<QKT, Access.ReadWrite> q_norm_out,
  Ptr<QKT, Access.ReadWrite> k_norm_out,
  [[Optional("write_compact_v")]] Ptr<T, Access.ReadWrite> compact_v_out,
  Ptr<float, Access.ReadWrite> beta_out,
  Ptr<float, Access.ReadWrite> decay_out,
  uint num_v_heads,
  uint num_k_heads,
  uint key_dim,
  uint value_dim,
  uint suffix_len,
  [[Groups("if suffix_len == 0 { 0 } else { num_k_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint head_group,
  [[Groups("suffix_len.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint token_group,
  [[Threads("128")]] uint tid,
  [[Specialize]] uint HEAD_K_DIM,
  [[Specialize]] bool write_log_decay,
  [[Specialize]] bool write_compact_v,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let heads_per_group = num_v_heads / num_k_heads;
  let blocks = value_dim / DELTA_NET_PREP_THREADS + (value_dim % DELTA_NET_PREP_THREADS != 0 ? 1u : 0u);
  var token = token_group;
  while (token < suffix_len) {
    let row = delta_net_rows(in_proj, token, uint3(key_dim, value_dim, num_v_heads));
    let gates = delta_net_rows(row, 1, uint3(key_dim, value_dim, 0));
    var hk = head_group;
    while (hk < num_k_heads) {
      if (write_compact_v && hk < blocks) {
        let values = delta_net_rows(row, 1, uint3(key_dim, 0, 0));
        let compact = element_pointer(compact_v_out, element_index(token, value_dim, 0));
        var block = hk;
        while (true) {
          if (tid < value_dim - block * DELTA_NET_PREP_THREADS) {
            let lane = element_index(block, DELTA_NET_PREP_THREADS, tid);
            element_pointer(compact, lane)[0] = element_pointer(values, lane)[0];
          }
          if (blocks - block <= num_k_heads) {
            break;
          }
          block += num_k_heads;
        }
      }
      let q_row = element_pointer(row, element_index(hk, DELTA_NET_PREP_THREADS, 0));
      let k_row = element_pointer(element_pointer(row, uint2(key_dim, 0)), element_index(hk, DELTA_NET_PREP_THREADS, 0));
      let q_inv = delta_net_inverse(q_row, tid);
      let k_inv = delta_net_inverse(k_row, tid);
      let lane = element_index(hk, DELTA_NET_PREP_THREADS, tid);
      element_pointer(element_pointer(q_norm_out, element_index(token, key_dim, 0)), lane)[0] =
        QKT.fromFloat(delta_net_query(element_pointer(q_row, uint2(tid, 0))[0].toFloat(), q_inv));
      element_pointer(element_pointer(k_norm_out, element_index(token, key_dim, 0)), lane)[0] =
        QKT.fromFloat(exact_product(element_pointer(k_row, uint2(tid, 0))[0].toFloat(), k_inv));
      if (tid < heads_per_group) {
        var group = tid;
        while (true) {
          let hv = hk * heads_per_group + group;
          let beta_raw = element_pointer(gates, uint2(hv, 0))[0].toFloat();
          let a_raw = element_pointer(gates, element_index(1, num_v_heads, hv))[0].toFloat();
          let log_decay = delta_net_log_decay(
            element_pointer(a_log, uint2(hv, 0))[0], a_raw, element_pointer(dt_bias, uint2(hv, 0))[0]
          );
          let head = element_index(token, num_v_heads, hv);
          element_pointer(beta_out, head)[0] = activate_silu_alpha<true>(1.0, beta_raw);
          element_pointer(decay_out, head)[0] = write_log_decay ? log_decay : ssd_decay(-log_decay);
          if (heads_per_group - group <= DELTA_NET_PREP_THREADS) {
            break;
          }
          group += DELTA_NET_PREP_THREADS;
        }
      }
      if (num_k_heads - hk <= groups.x) {
        break;
      }
      hk += groups.x;
    }
    if (suffix_len - token <= groups.y) {
      break;
    }
    token += groups.y;
  }
}
