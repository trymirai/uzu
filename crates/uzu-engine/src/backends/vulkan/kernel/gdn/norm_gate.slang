import definitions;
import common.activation;
import common.workgroup_reduction;
import generated.activation_type;

static const uint DELTA_NET_NORM_GATE_THREADS = 128;

[[Variants("T", "float, bf16")]]
[[Kernel]] [[Public]]
[[Precondition("encode", "suffix_len <= 1 || num_v_heads * head_v_dim <= value_dim")]]
func DeltaNetNormGate<T: IStorageFloat>(
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
  [[Groups("num_v_heads")]] uint head_group,
  [[Groups("suffix_len")]] uint token_group,
  [[Threads("128")]] uint tid,
) {
  let base = token_group * value_dim + head_group * head_v_dim;
  let gates = token_group * total_proj_dim + conv_dim + head_group * head_v_dim;
  float squares = 0.0;
  for (uint i = tid; i < head_v_dim; i += DELTA_NET_NORM_GATE_THREADS) {
    let x = in_out[base + i].toFloat();
    squares += x * x;
  }
  let inv = rms_inverse<DELTA_NET_NORM_GATE_THREADS>(squares, head_v_dim, norm_epsilon);
  for (uint i = tid; i < head_v_dim; i += DELTA_NET_NORM_GATE_THREADS) {
    let gate = activate(in_proj[gates + i].toFloat(), ActivationType.SILU);
    in_out[base + i] = T.fromFloat(in_out[base + i].toFloat() * inv * norm_weight[i] * gate);
  }
}
