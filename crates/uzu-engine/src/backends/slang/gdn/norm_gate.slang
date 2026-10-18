import definitions;
import storage;

groupshared float sums[256];

[[Variants("T", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func DeltaNetNormGate<T : ITensorElement>(
    RWStructuredBuffer<Atomic<uint>> in_out,
    StructuredBuffer<uint> in_proj,
    StructuredBuffer<float> norm_weight,
    uint num_v_heads,
    uint head_v_dim,
    uint value_dim,
    uint conv_dim,
    uint total_proj_dim,
    float norm_epsilon,
    uint suffix_len,
    ThreadContext thread_context,
    [[Groups("suffix_len")]] uint token_idx,
    [[Groups("num_v_heads")]] uint hv_idx,
    [[Threads("256")]] uint tid,
) {
  uint lane = tid;
  uint3 group = uint3(hv_idx, token_idx, 0);

  let base = group.y * value_dim + group.x * head_v_dim;
  var sum = 0.0;
  for (var i = lane; i < head_v_dim; i += 256u) {
    let value = tensorLoad<T>(in_out, base + i);
    sum += value * value;
  }
  sums[lane] = sum;
  GroupMemoryBarrierWithGroupSync();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      sums[lane] += sums[lane + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  let inv = rsqrt(sums[0] / float(head_v_dim) + norm_epsilon);
  for (var i = lane; i < head_v_dim; i += 256u) {
    let z = tensorLoad<T>(in_proj, group.y * total_proj_dim + conv_dim + group.x * head_v_dim + i);
    tensorStore<T>(in_out, base + i, tensorLoad<T>(in_out, base + i) * inv * norm_weight[i] * activate(z, 0u, 1.0));
  }
}
