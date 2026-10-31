import definitions;
import attention.attention;
import common.workgroup_reduction;
import generated.kv_layout;
import generated.qkv_norm;

static const uint ATTENTION_PREPARE_THREADS = 32;

// Element `channel` of a head under its RMS norm: ElementT-rounded, with scales multiplied in FP32 for full_layer and
// otherwise as ElementT, as Metal's prepare_norm.
float prepare_norm<ElementT : IStorageFloat>(
  ElementT value,
  uint channel,
  float inverse_rms,
  Ptr<float, Access.Read> scales,
  bool has_scales,
  bool enabled,
  bool full_layer,
  float scale_offset
) {
  if (!enabled) {
    return value.toFloat();
  }
  let normalized = value.toFloat() * inverse_rms;
  if (!has_scales) {
    return ElementT.fromFloat(normalized).toFloat();
  }
  let scale = scales[channel] + scale_offset;
  if (full_layer) {
    return ElementT.fromFloat(normalized * scale).toFloat();
  }
  return ElementT.fromFloat(ElementT.fromFloat(normalized).toFloat() * ElementT.fromFloat(scale).toFloat()).toFloat();
}

// Packed qkvg rows to INT8 codes, one workgroup of 32 invocations per token and head, lane l holding dimensions
// [l HEAD_DIM / 32, (l + 1) HEAD_DIM / 32): the head's norm, RoPE of the first rope_dim dimensions of queries and keys,
// attention_rotate, then attention_quantize. Query rows are [kv head][query head in its group][token]; key and value rows
// [kv_token_offset + token][kv head], with the value codes biased by V_CODE_BIAS. Any trailing gate is unread.
[[Variants("ElementT", "float, bf16")]]
[[Variants("RopeT", "float")]]
[[Kernel]] [[Public]]
[[Precondition("new", "HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256 || HEAD_DIM == 512")]]
[[Precondition("new", "num_kv_heads > 0 && backend_opaque_layout == 0")]]
[[Precondition("encode", "num_q_heads.is_multiple_of(num_kv_heads)")]]
func AttentionPrepare<ElementT : IStorageFloat, RopeT : IStorageFloat>(
  Ptr<ElementT, Access.Read> qkvg,
  Ptr<int8_t, Access.ReadWrite> q_codes,
  Ptr<bf16, Access.ReadWrite> q_scales,
  [[Optional("append_kv")]] Ptr<int8_t, Access.ReadWrite> k_codes,
  [[Optional("append_kv")]] Ptr<bf16, Access.ReadWrite> k_scales,
  [[Optional("append_kv")]] Ptr<uint8_t, Access.ReadWrite> v_codes,
  [[Optional("append_kv")]] Ptr<bf16, Access.ReadWrite> v_scales,
  [[Optional("has_rope")]] Ptr<RopeT, Access.Read> cosines,
  [[Optional("has_rope")]] Ptr<RopeT, Access.Read> sines,
  [[Optional("has_query_norm_scales")]] Ptr<float, Access.Read> query_norm_scales,
  [[Optional("has_key_norm_scales")]] Ptr<float, Access.Read> key_norm_scales,
  [[Optional("has_value_norm_scales")]] Ptr<float, Access.Read> value_norm_scales,
  QKVNormParams norm_params,
  uint num_q_heads,
  [[Optional("has_rope")]] uint rope_dim,
  uint kv_token_offset,
  uint input_row_stride,
  uint suffix_length,
  uint query_rows,
  [[Groups("suffix_length")]] uint token,
  [[Groups("num_q_heads + num_kv_heads * 2")]] uint head,
  [[Threads("32")]] uint tid,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] bool has_rope,
  [[Specialize]] uint num_kv_heads,
  [[Specialize]] uint backend_opaque_layout,
  [[Specialize]] bool append_kv,
  [[Specialize]] bool has_query_norm_scales,
  [[Specialize]] bool has_key_norm_scales,
  [[Specialize]] bool has_value_norm_scales,
) {
  static_assert(ATTENTION_PREPARE_THREADS == SIMD_SIZE, "one group of Threads(32) per token and head");
  let is_query = head < num_q_heads;
  // Uniform across the workgroup, which the rotation's and quantization's reductions need.
  if (!append_kv && !is_query) {
    return;
  }
  let is_key = !is_query && head < num_q_heads + num_kv_heads;
  let lane = subgroup_invocation();
  let dims = HEAD_DIM / SIMD_SIZE;
  let first = lane * dims;
  let norm = is_query ? 0u : is_key ? 1u : 2u;
  let enabled = norm_params.enabled[norm] != 0;
  let full_layer = norm_params.full_layer[norm] != 0;
  let scale_offset = norm_params.scale_offset[norm];
  let has_scales = is_query ? has_query_norm_scales : is_key ? has_key_norm_scales : has_value_norm_scales;
  let scales = is_query ? query_norm_scales : is_key ? key_norm_scales : value_norm_scales;
  let row = token * input_row_stride + head * HEAD_DIM;

  var inverse_rms = 1.0;
  if (enabled) {
    var squares = 0.0;
    for (uint base = 4 * lane; base < HEAD_DIM; base += 4 * SIMD_SIZE) {
      [unroll]
      for (uint i = 0; i < 4; i++) {
        let value = qkvg[row + base + i].toFloat();
        squares += value * value;
      }
    }
    inverse_rms = rms_inverse<ATTENTION_PREPARE_THREADS>(squares, HEAD_DIM, norm_params.epsilon[norm]);
  }
  float values[ATTENTION_DIMS];
  for (uint a = 0; a < dims; a++) {
    let channel = first + a;
    var element =
      prepare_norm(qkvg[row + channel], channel, inverse_rms, scales, has_scales, enabled, full_layer, scale_offset);
    if (has_rope && (is_query || is_key) && channel < rope_dim) {
      let half = rope_dim / 2;
      let paired_channel = channel < half ? channel + half : channel - half;
      let paired = prepare_norm(
        qkvg[row + paired_channel], paired_channel, inverse_rms, scales, has_scales, enabled, full_layer, scale_offset
      );
      let table = token * rope_dim + channel;
      let rotated = rope_rotated(element, paired, channel < half, cosines[table].toFloat(), sines[table].toFloat());
      element = ElementT.fromFloat(rotated).toFloat();
    }
    values[a] = element;
  }
  attention_rotate<ATTENTION_PREPARE_THREADS>(values, dims, lane);
  int codes[ATTENTION_DIMS];
  let scale = attention_quantize<ATTENTION_PREPARE_THREADS>(values, dims, codes);

  if (is_query) {
    let gqa_factor = num_q_heads / num_kv_heads;
    let query_row = head / gqa_factor * query_rows + head % gqa_factor * suffix_length + token;
    for (uint a = 0; a < dims; a++) {
      q_codes[query_row * HEAD_DIM + first + a] = int8_t(codes[a]);
    }
    if (lane == 0) {
      q_scales[query_row] = scale;
    }
    return;
  }
  let kv_row = (kv_token_offset + token) * num_kv_heads + head - num_q_heads - (is_key ? 0 : num_kv_heads);
  for (uint a = 0; a < dims; a++) {
    if (is_key) {
      k_codes[kv_row * HEAD_DIM + first + a] = int8_t(codes[a]);
    } else {
      v_codes[kv_row * HEAD_DIM + first + a] = uint8_t(codes[a] + int(V_CODE_BIAS));
    }
  }
  if (lane == 0 && is_key) {
    k_scales[kv_row] = scale;
  } else if (lane == 0) {
    v_scales[kv_row] = scale;
  }
}
