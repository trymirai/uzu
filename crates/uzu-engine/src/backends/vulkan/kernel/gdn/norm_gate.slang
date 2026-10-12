import definitions;
import common.activation;
import common.storage_address;
import common.workgroup_reduction;
import generated.activation_type;

// The CPU DeltaNetNormGate in place, one workgroup per (token, head): element i of the head row at
// in_out[t value_dim + h head_v_dim] becomes T(((o_i inv) norm_weight[i]) silu(z_i)), with z_i at
// in_proj[t total_proj_dim + conv_dim + h head_v_dim + i] and inv the reciprocal RMS of the row. The squares, their sum,
// mean plus epsilon and reciprocal square root follow the CPU's FP32 stages with subnormals kept (accumulate_squares,
// rms_inverse); the products are IEEE FP32 roundings in the CPU's order and only the result converts to T. Each
// invocation rereads only the elements it squared, after the reduction's barriers. Element offsets are 64-bit words
// formed one term at a time. Workgroups stride over heads and tokens by the actual NumWorkgroups, which the host clamps
// to the device limits, so every loop holding the reduction's barriers is uniform; elements stride by the workgroup,
// advancing only while more than one step remains. Tokens' head rows are disjoint only when value_dim holds the heads,
// which the precondition requires whenever several tokens exist; the bound is a quotient, free of overflow.

static const uint DELTA_NET_NORM_GATE_THREADS = 128;

[[Kernel]] [[Public]]
[[Precondition("encode", "suffix_len <= 1 || num_v_heads == 0 || head_v_dim == 0 || num_v_heads <= value_dim / head_v_dim")]]
func DeltaNetNormGate<T: IModelFloat>(
  Ptr<T, Access.ReadWrite> in_out,
  Ptr<T, Access.Read> in_proj,
  Ptr<float, Access.Read> norm_weight,
  uint num_v_heads,
  uint head_v_dim,
  uint value_dim,
  uint conv_dim,
  uint total_proj_dim,
  float norm_epsilon,
  uint suffix_len,
  [[Groups("if suffix_len == 0 || num_v_heads == 0 || head_v_dim == 0 { 0 } else { num_v_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint head_group,
  [[Groups("suffix_len.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint token_group,
  [[Threads("128")]] uint tid,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  var token = token_group;
  while (token < suffix_len) {
    let values = element_pointer(in_out, element_index(token, value_dim, 0));
    let gates = element_pointer(in_proj, element_index(token, total_proj_dim, conv_dim));
    var head = head_group;
    while (head < num_v_heads) {
      float squares = 0.0;
      if (tid < head_v_dim) {
        var i = tid;
        while (true) {
          squares = accumulate_squares(squares, element_pointer(values, element_index(head, head_v_dim, i))[0].toFloat());
          if (head_v_dim - i <= DELTA_NET_NORM_GATE_THREADS) {
            break;
          }
          i += DELTA_NET_NORM_GATE_THREADS;
        }
      }
      let inv = rms_inverse<DELTA_NET_NORM_GATE_THREADS>(squares, head_v_dim, norm_epsilon);
      if (tid < head_v_dim) {
        var i = tid;
        while (true) {
          let value = element_pointer(values, element_index(head, head_v_dim, i));
          let gate = activate(element_pointer(gates, element_index(head, head_v_dim, i))[0].toFloat(), ActivationType.SILU);
          let scaled = exact_product(exact_product(value[0].toFloat(), inv), element_pointer(norm_weight, uint2(i, 0))[0]);
          value[0] = T.fromFloat(exact_product(scaled, gate));
          if (head_v_dim - i <= DELTA_NET_NORM_GATE_THREADS) {
            break;
          }
          i += DELTA_NET_NORM_GATE_THREADS;
        }
      }
      if (num_v_heads - head <= groups.x) {
        break;
      }
      head += groups.x;
    }
    if (suffix_len - token <= groups.y) {
      break;
    }
    token += groups.y;
  }
}
