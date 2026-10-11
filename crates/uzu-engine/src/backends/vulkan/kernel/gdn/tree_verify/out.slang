import definitions;
import common.activation;
import common.ordered_dot;
import common.storage_address;
import gdn.prefill_prep;

// CPU-ordered FP32 o[b, row, hv, dv]: start at +0, add the optional h0 dot term, then every qkd * u product ascending.
// One invocation owns each output; zero products and Dk=0 work are retained. CPU use_mxu/transposed_h0 are ignored.
// Addresses are termwise 64-bit; clamped workgroups stride over value-dim blocks, rows and heads without uint wrap.

static const uint BUILD_TREE_OUT_THREADS = 64;

[[Kernel]] [[Public]]
[[Precondition("encode", "qk_heads != 0 && (batch_size == 0 || value_heads == 0 || value_heads >= qk_heads)")]]
func BuildTreeOut<QKT: IModelFloat, OutputT: IModelFloat>(
  Ptr<QKT, Access.Read> q,
  Ptr<float, Access.Read> prefix,
  Ptr<float, Access.Read> qkd,
  Ptr<float, Access.Read> u,
  [[Optional("use_h0")]] Ptr<float, Access.Read> h0,
  [[Optional("use_h0")]] Ptr<int, Access.Read> h0_indices,
  Ptr<OutputT, Access.ReadWrite> o,
  float scale,
  uint batch_size,
  uint tree_size,
  uint qk_heads,
  uint value_heads,
  uint head_k_dim,
  uint head_v_dim,
  [[Groups("if batch_size == 0 || tree_size == 0 || value_heads == 0 || head_v_dim == 0 { 0 } else { head_v_dim.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint dv_group,
  [[Groups("if batch_size == 0 || tree_size == 0 || value_heads == 0 || head_v_dim == 0 { 0 } else { tree_size.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1]) }")]] uint row_group,
  [[Groups("if batch_size == 0 || tree_size == 0 || value_heads == 0 || head_v_dim == 0 { 0 } else { value_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[2]) }")]] uint head_group,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] bool use_mxu,
  [[Specialize]] bool transposed_h0,
  [[Specialize]] bool use_h0,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let dv_blocks = head_v_dim / BUILD_TREE_OUT_THREADS + (head_v_dim % BUILD_TREE_OUT_THREADS != 0 ? 1u : 0u);
  let heads_per_group = value_heads / qk_heads;
  var hv = head_group;
  while (hv < value_heads) {
    let hk = hv / heads_per_group;
    var row = row_group;
    while (row < tree_size) {
      var dv_block = dv_group;
      while (dv_block < dv_blocks) {
        if (thread_idx < head_v_dim - dv_block * BUILD_TREE_OUT_THREADS) {
          let dv = dv_block * BUILD_TREE_OUT_THREADS + thread_idx;
          for (uint b = 0; b < batch_size; b++) {
            let lane = add_address(byte_offset(element_index(b, tree_size, row), value_heads), uint2(hv, 0));
            var acc = 0.0;
            if (use_h0) {
              let slot = element_pointer(h0_indices, uint2(b, 0))[0];
              if (slot >= 0) {
                let q_lane = add_address(byte_offset(element_index(b, tree_size, row), qk_heads), uint2(hk, 0));
                let h0_head = byte_offset(element_index(uint(slot), value_heads, hv), head_v_dim);
                let h0_row = element_pointer(h0, byte_offset(add_address(h0_head, uint2(dv, 0)), head_k_dim));
                let h0_dot = ordered_dot(element_pointer(q, byte_offset(q_lane, head_k_dim)), h0_row, head_k_dim);
                let decay = delta_net_exp(element_pointer(prefix, lane)[0]);
                acc = ordered_sum(acc, exact_product(exact_product(decay, scale), h0_dot));
              }
            }
            let head = element_index(b, value_heads, hv);
            let qkd_head = element_pointer(qkd, byte_offset(byte_offset(head, tree_size), tree_size));
            let u_head = element_pointer(u, byte_offset(byte_offset(head, tree_size), head_v_dim));
            for (uint col = 0; col < tree_size; col++) {
              let qkd_value = element_pointer(qkd_head, element_index(row, tree_size, col))[0];
              let u_value = element_pointer(u_head, element_index(col, head_v_dim, dv))[0];
              acc = ordered_sum(acc, exact_product(qkd_value, u_value));
            }
            element_pointer(o, add_address(byte_offset(lane, head_v_dim), uint2(dv, 0)))[0] = OutputT.fromFloat(acc);
          }
        }
        if (dv_blocks - dv_block <= groups.x) {
          break;
        }
        dv_block += groups.x;
      }
      if (tree_size - row <= groups.y) {
        break;
      }
      row += groups.y;
    }
    if (value_heads - hv <= groups.z) {
      break;
    }
    hv += groups.z;
  }
}
