import definitions;
import storage;

groupshared float sums[256];

[[Variants("InputT", "float, half, BFloat16")]]
[[Variants("ScaleT", "float")]]
[[Variants("OutputT", "float, half, BFloat16")]]
[[Variants("AccumT", "float")]]
[[Public]] [[Kernel]] func QKVNorm<InputT : ITensorElement, ScaleT : ITensorElement, OutputT : ITensorElement, AccumT : ITensorElement>(
    [[Optional("!in_place")]] StructuredBuffer<uint> qkvg_input,
    [[Optional("has_scales")]] StructuredBuffer<uint> scales,
    RWStructuredBuffer<Atomic<uint>> qkvg_output,
    uint batch_size,
    uint input_row_stride,
    uint head_dim,
    float epsilon,
    float scale_offset,
    uint head_offset,
    uint head_count,
    bool full_layer,
    [[Groups("batch_size")]] uint batch_idx,
    [[Groups("head_count")]] uint head_idx,
    [[Threads("256")]] uint lane_id,
    [[Specialize]] bool in_place,
    [[Specialize]] bool has_scales,
) {
  uint lane = lane_id;
  uint3 gid = uint3(batch_idx * head_count + head_idx, 0, 0);

  let base = (gid.x / head_count) * input_row_stride + (head_offset + gid.x % head_count) * head_dim;
  var sum = 0.0;
  for (var i = lane; i < head_dim; i += 256u) {
    let v = (in_place ? tensorLoad<OutputT>(qkvg_output, base + i) : tensorLoad<InputT>(qkvg_input, base + i));
    sum += v * v;
  }
  sums[lane] = sum;
  GroupMemoryBarrierWithGroupSync();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      sums[lane] += sums[lane + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  let rms = rsqrt(sums[0] / float(head_dim) + epsilon);
  for (var i = lane; i < head_dim; i += 256u) {
    var v = (in_place ? tensorLoad<OutputT>(qkvg_output, base + i) : tensorLoad<InputT>(qkvg_input, base + i)) * rms;
    if (has_scales) {
      let scale = tensorLoad<ScaleT>(scales, i) + scale_offset;
      if (full_layer) {
        v *= scale;
      } else {
        v = tensorRound<OutputT>(v) * tensorRound<OutputT>(scale);
      }
    }
    tensorStore<OutputT>(qkvg_output, base + i, v);
  }
}
