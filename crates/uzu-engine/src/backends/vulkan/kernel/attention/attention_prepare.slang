import definitions;
import common.activation;

static const uint ATTENTION_PREPARE_TILE = 128;

// Splits packed qkvg rows into queries [head, token, head_dim] and, with has_kv, keys and values
// [kv_token_offset + token, head, head_dim], leaving any trailing gate unread. One workgroup per tile of one head of one
// token. Elements past rope_dim, value heads and every element without has_rope are copied bit for bit; the first rope_dim
// elements of query and key heads are rotated as the CPU apply_rope: the input times the cosine plus the signed paired
// element times the sine, each product and the sum an FP32 rounding of its own with subnormals kept, then rounded to
// ElementT. The paired element of the first half is negated through its sign bit, which the device may flush otherwise.
// The CPU's preconditions hold also for empty batches; the row stride bound is a quotient, free of overflow.
[[Kernel]] [[Public]]
[[Precondition("encode", "num_q_heads > 0 || has_kv")]]
[[Precondition("encode", "head_dim > 0")]]
[[Precondition("encode", "num_kv_heads.unwrap_or(1) > 0")]]
[[Precondition("encode", "rope_dim.unwrap_or(2) > 0")]]
[[Precondition("encode", "rope_dim.unwrap_or(0) <= head_dim")]]
[[Precondition("encode", "rope_dim.unwrap_or(0).is_multiple_of(2)")]]
[[Precondition("encode", "num_q_heads <= input_row_stride / head_dim && num_kv_heads.unwrap_or(0) <= (input_row_stride / head_dim - num_q_heads) / 2")]]
func AttentionPrepare<ElementT: IBFloat16, RopeT: IFloat32>(
  Ptr<ElementT, Access.Read> qkvg,
  Ptr<ElementT, Access.ReadWrite> queries,
  [[Optional("has_kv")]] Ptr<ElementT, Access.ReadWrite> keys,
  [[Optional("has_kv")]] Ptr<ElementT, Access.ReadWrite> values,
  [[Optional("has_rope")]] Ptr<RopeT, Access.Read> cosines,
  [[Optional("has_rope")]] Ptr<RopeT, Access.Read> sines,
  uint num_q_heads,
  [[Optional("has_kv")]] uint num_kv_heads,
  uint head_dim,
  [[Optional("has_rope")]] uint rope_dim,
  [[Optional("has_kv")]] uint kv_token_offset,
  uint input_row_stride,
  uint batch_dim,
  [[Groups("head_dim.div_ceil(128)")]] uint tile,
  [[Groups("num_q_heads + 2 * num_kv_heads.unwrap_or(0)")]] uint head,
  [[Groups("batch_dim")]] uint batch,
  [[Threads("128")]] uint tid,
  [[Specialize]] bool has_kv,
  [[Specialize]] bool has_rope,
) {
  static_assert(ATTENTION_PREPARE_TILE == 128, "one tile of Threads(128) per workgroup, Groups(div_ceil(128))");
  let element = tile * ATTENTION_PREPARE_TILE + tid;
  if (element >= head_dim) {
    return;
  }
  let row = batch * input_row_stride + head * head_dim;
  let is_query = !has_kv || head < num_q_heads;
  let is_key = !is_query && head < num_q_heads + num_kv_heads;
  var value = qkvg[row + element];
  if (has_rope && element < rope_dim && (is_query || is_key)) {
    let half = rope_dim / 2;
    let first = element < half;
    let paired = qkvg[row + (first ? element + half : element - half)].toFloat();
    let signed_paired = asfloat(asuint(paired) ^ (first ? 0x80000000u : 0u));
    let table = batch * rope_dim + element;
    let rotated = exact_sum(
      exact_product(value.toFloat(), cosines[table].toFloat()),
      exact_product(signed_paired, sines[table].toFloat()),
      false
    );
    value = ElementT.fromFloat(rotated);
  }
  if (is_query) {
    queries[(head * batch_dim + batch) * head_dim + element] = value;
    return;
  }
  let kv_head = head - num_q_heads - (is_key ? 0 : num_kv_heads);
  let index = ((kv_token_offset + batch) * num_kv_heads + kv_head) * head_dim + element;
  if (is_key) {
    keys[index] = value;
  } else {
    values[index] = value;
  }
}
