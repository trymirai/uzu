import definitions;
import attention.attention;
import common.workgroup_reduction;
import generated.kv_layout;
import generated.qkv_norm;
import generated.rope;
import generated.trie;
import vision.patch_grid;

groupshared float prepare_block_mean[ATTENTION_DIMS * SIMD_SIZE];

// A head's input channel: the block mean when the kernel pools blocks of keys, the projected row otherwise.
ElementT prepare_input<ElementT : IStorageFloat>(
  Ptr<ElementT, Access.Read> qkvg,
  uint row,
  uint channel,
  bool block_mean
) {
  if (block_mean) {
    return ElementT.fromFloat(prepare_block_mean[channel]);
  }
  return qkvg[row + channel];
}

float prepare_norm<ElementT : IStorageFloat>(
  ElementT value,
  uint channel,
  float row_scale,
  Ptr<float, Access.Read> scales,
  bool has_scales,
  bool enabled,
  bool full_layer,
  float scale_offset
) {
  if (!enabled) {
    return value.toFloat();
  }
  let normalized = value.toFloat() * row_scale;
  if (!has_scales) {
    return ElementT.fromFloat(normalized).toFloat();
  }
  let scale = scales[channel] + scale_offset;
  if (full_layer) {
    return ElementT.fromFloat(normalized * scale).toFloat();
  }
  return ElementT.fromFloat(ElementT.fromFloat(normalized).toFloat() * ElementT.fromFloat(scale).toFloat()).toFloat();
}

[[Variants("ElementT", "float, bf16")]]
[[Kernel]] [[Public]]
[[Precondition("new", "HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256 || HEAD_DIM == 512")]]
[[Precondition("new", "num_kv_heads > 0 && backend_opaque_layout == 0")]]
[[Precondition("encode", "num_q_heads.is_multiple_of(num_kv_heads)")]]
func AttentionPrepare<ElementT : IStorageFloat>(
  Ptr<ElementT, Access.Read> qkvg,
  Ptr<int8_t, Access.ReadWrite> q_codes,
  Ptr<bf16, Access.ReadWrite> q_scales,
  [[Optional("append_kv")]] Ptr<int8_t, Access.ReadWrite> k_codes,
  [[Optional("append_kv")]] Ptr<bf16, Access.ReadWrite> k_scales,
  [[Optional("append_kv")]] Ptr<uint8_t, Access.ReadWrite> v_codes,
  [[Optional("append_kv")]] Ptr<bf16, Access.ReadWrite> v_scales,
  [[Optional("has_rope")]] Ptr<float, Access.Read> rope_inverse_frequencies,
  [[Optional("has_query_norm_scales")]] Ptr<float, Access.Read> query_norm_scales,
  [[Optional("has_key_norm_scales")]] Ptr<float, Access.Read> key_norm_scales,
  [[Optional("has_value_norm_scales")]] Ptr<float, Access.Read> value_norm_scales,
  [[Optional("block_mean")]] Ptr<TrieNode, Access.Read> pool_nodes,
  [[Optional("block_mean")]] Ptr<int, Access.Read> pool_windows,
  [[Optional("block_mean")]] Ptr<ElementT, Access.ReadWrite> raw_keys,
  QKVNormParams norm_params,
  uint num_q_heads,
  [[Optional("has_rope")]] uint rope_dim,
  [[Optional("has_rope")]] float rope_scale,
  [[Optional("has_rope")]] uint rope_position_offset,
  [[Optional("has_mrope")]] RoPEGridParams rope_grid,
  uint kv_token_offset,
  uint input_row_stride,
  uint suffix_length,
  uint query_rows,
  [[Optional("block_mean")]] uint prefix_tokens,
  [[Optional("has_trie")]] Ptr<TrieNode, Access.Read> trie,
  [[Groups("suffix_length")]] uint token,
  [[Groups("num_q_heads + num_kv_heads * 2")]] uint head,
  [[Threads("32")]] uint tid,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] bool has_rope,
  [[Specialize]] bool has_mrope,
  [[Specialize]] uint num_kv_heads,
  [[Specialize]] uint backend_opaque_layout,
  [[Specialize]] bool append_kv,
  [[Specialize]] bool has_query_norm_scales,
  [[Specialize]] bool has_key_norm_scales,
  [[Specialize]] bool has_value_norm_scales,
  [[Specialize]] bool block_mean,
  [[Specialize]] uint compression_ratio,
  [[Specialize]] bool has_trie,
) {
  static_assert(SIMD_SIZE == 32, "Threads(32)");
  let is_query = head < num_q_heads;
  if (!append_kv && !is_query) {
    return;
  }
  let is_key = !is_query && head < num_q_heads + num_kv_heads;
  let lane = subgroup_invocation();
  let dims = HEAD_DIM / SIMD_SIZE;
  let first = lane * dims;
  let norm = is_query ? 0u : is_key ? 1u : 2u;
  let op = norm_params.op[norm];
  let enabled = op != uint(QKVL2Op::None);
  let full_layer = norm_params.full_layer[norm] != 0;
  let scale_offset = norm_params.scale_offset[norm];
  let has_scales = is_query ? has_query_norm_scales : is_key ? has_key_norm_scales : has_value_norm_scales;
  let scales = is_query ? query_norm_scales : is_key ? key_norm_scales : value_norm_scales;
  let row = token * input_row_stride + head * HEAD_DIM;
  // Trie nodes sit at their depth below the root, not at their row in the suffix.
  var rope_position = has_rope ? rope_position_offset + (has_trie ? trie[token].height : token) : 0;
  var grid_row = 0u;
  var grid_column = 0u;
  if (has_mrope) {
    // Rows are cells of grids in raster order: a text token is a 1x1 grid.
    let cell = rope_grid.first_cell + (has_trie ? trie[token].height : token);
    let grid_size = rope_grid.grid_width * rope_grid.grid_height;
    rope_position = rope_position_offset + cell / grid_size * max(rope_grid.grid_width, rope_grid.grid_height);
    let patch = vision_patch_coordinates(cell % grid_size, rope_grid.grid_width, rope_grid.merge_size);
    grid_row = patch.y;
    grid_column = patch.x;
  }
  if (block_mean) {
    // raw_keys: the open block's cached rows, then one row per suffix node.
    let block_start = prefix_tokens - prefix_tokens % compression_ratio;
    let completes_block = (prefix_tokens + pool_nodes[token].height + 1) % compression_ratio == 0;
    for (uint a = 0; a < dims; a++) {
      let channel = first + a;
      var sum = 0.0;
      for (uint i = 0; completes_block && i < compression_ratio; i++) {
        let slot = uint(pool_windows[token * compression_ratio + i]);
        if (slot >= prefix_tokens) {
          sum += qkvg[(slot - prefix_tokens) * input_row_stride + channel].toFloat();
        } else {
          sum += raw_keys[(slot - block_start) * HEAD_DIM + channel].toFloat();
        }
      }
      raw_keys[(compression_ratio - 1 + token) * HEAD_DIM + channel] = qkvg[row + channel];
      prepare_block_mean[channel] = ElementT.fromFloat(sum / float(compression_ratio)).toFloat();
    }
    GroupMemoryBarrierWithGroupSync();
  }

  var row_scale = 1.0;
  if (enabled) {
    var squares = 0.0;
    for (uint base = 4 * lane; base < HEAD_DIM; base += 4 * SIMD_SIZE) {
      [unroll]
      for (uint i = 0; i < 4; i++) {
        let value = prepare_input(qkvg, row, base + i, block_mean).toFloat();
        squares += value * value;
      }
    }
    let sum_squares = workgroup_reduce<Sum<float>, SIMD_SIZE>(squares);
    let epsilon = norm_params.epsilon[norm];
    row_scale = op == uint(QKVL2Op::Clip) ? min(1.0, norm_params.clip_threshold[norm] / (sqrt(sum_squares) + epsilon))
                                          : rsqrt(sum_squares / float(HEAD_DIM) + epsilon);
  }
  float values[ATTENTION_DIMS];
  for (uint a = 0; a < dims; a++) {
    let channel = first + a;
    let input = prepare_input(qkvg, row, channel, block_mean);
    var element = prepare_norm(input, channel, row_scale, scales, has_scales, enabled, full_layer, scale_offset);
    if (has_rope && (is_query || is_key) && channel < rope_dim) {
      let half = rope_dim / 2;
      let paired_channel = channel < half ? channel + half : channel - half;
      let paired_input = prepare_input(qkvg, row, paired_channel, block_mean);
      let paired =
        prepare_norm(paired_input, paired_channel, row_scale, scales, has_scales, enabled, full_layer, scale_offset);
      var position = rope_position;
      if (has_mrope) {
        position += mrope_axis_offset(
          channel % half,
          half,
          grid_row,
          grid_column,
          rope_grid.height_pairs,
          rope_grid.width_pairs,
          rope_grid.interleaved_axes != 0
        );
      }
      let rotated = rope_rotate(element, paired, rope_inverse_frequencies, rope_scale, position, channel, rope_dim);
      element = ElementT.fromFloat(rotated).toFloat();
    }
    values[a] = element;
  }
  attention_rotate<SIMD_SIZE>(values, dims, lane);
  int codes[ATTENTION_DIMS];
  let scale = attention_quantize<SIMD_SIZE>(values, dims, codes);

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
