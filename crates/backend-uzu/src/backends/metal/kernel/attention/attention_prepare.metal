#include "../common/dsl.h"

// TODO: very ugly and wasteful, clean up and optimize

template <typename ElementT, typename RopeT>
inline ElementT apply_rope(
    const device ElementT* head,
    const device RopeT* cosines,
    const device RopeT* sines,
    uint32_t batch_idx,
    uint32_t head_dim_idx,
    uint32_t rope_dim
) {
  const uint32_t half_rope_dim = rope_dim / 2;
  const uint32_t paired_idx =
      head_dim_idx < half_rope_dim ? head_dim_idx + half_rope_dim : head_dim_idx - half_rope_dim;
  const float input = float(head[head_dim_idx]);
  const float paired = float(head[paired_idx]);
  const float signed_paired = head_dim_idx < half_rope_dim ? -paired : paired;
  const float cos_val = float(cosines[batch_idx * rope_dim + head_dim_idx]);
  const float sin_val = float(sines[batch_idx * rope_dim + head_dim_idx]);

  return static_cast<ElementT>(input * cos_val + signed_paired * sin_val);
}

template <typename ElementT, typename RopeT>
VARIANTS(ElementT, bfloat)
VARIANTS(RopeT, float)
PUBLIC KERNEL(AttentionPrepare) (
  const device ElementT* qkv, // [token, (q, k, v), head_dim]
  device ElementT* queries, // [head_idx, token, head_dim]
  device ElementT* keys OPTIONAL(has_kv), // [(kv_token_offset + token), head_idx, head_dim]
  device ElementT* values OPTIONAL(has_kv), // [(kv_token_offset + token), head_idx, head_dim]
  const device RopeT* cosines OPTIONAL(has_rope),
  const device RopeT* sines OPTIONAL(has_rope),
  const constant uint32_t& num_q_heads,
  const constant uint32_t& num_kv_heads OPTIONAL(has_kv),
  const constant uint32_t& head_dim,
  const constant uint32_t& rope_dim OPTIONAL(has_rope),
  const constant uint32_t& kv_token_offset OPTIONAL(has_kv),
  const constant uint32_t& batch_dim,
  const bool has_kv SPECIALIZE,
  const bool has_rope SPECIALIZE,
  const uint32_t head_dim_idx AXIS(head_dim, 128),
  const uint32_t head_idx AXIS(num_q_heads + num_kv_heads.unwrap_or(0) * 2, 1),
  const uint32_t batch_idx AXIS(batch_dim, 1)
) {
  const uint32_t total_heads = has_kv ? num_q_heads + num_kv_heads * 2 : num_q_heads;
  const uint32_t qkv_head_idx = batch_idx * total_heads * head_dim + head_idx * head_dim;
  const device ElementT* qkv_head = qkv + qkv_head_idx;
  const bool is_query = !has_kv || head_idx < num_q_heads;
  const bool is_key = has_kv && head_idx >= num_q_heads && head_idx < num_q_heads + num_kv_heads;

  ElementT element = qkv_head[head_dim_idx];
  if (has_rope && head_dim_idx < rope_dim && (is_query || is_key)) {
    element = apply_rope(qkv_head, cosines, sines, batch_idx, head_dim_idx, rope_dim);
  }

  if (is_query) {
    const uint32_t q_idx = head_idx * batch_dim * head_dim + batch_idx * head_dim + head_dim_idx;
    queries[q_idx] = element;
  } else if (is_key) {
    const uint32_t k_idx =
        (kv_token_offset + batch_idx) * num_kv_heads * head_dim + (head_idx - num_q_heads) * head_dim + head_dim_idx;

    keys[k_idx] = element;
  } else if (has_kv) {
    const uint32_t v_idx = (kv_token_offset + batch_idx) * num_kv_heads * head_dim +
                           (head_idx - num_q_heads - num_kv_heads) * head_dim + head_dim_idx;

    values[v_idx] = element;
  }
}
