#include "../common/dsl.h"
#include "rope.h"

template <typename ElementT, typename RopeT>
VARIANTS(ElementT, bfloat)
VARIANTS(RopeT, float)
PUBLIC KERNEL(AttentionPrepare) (
  const device ElementT* qkvg, // [token, input_row_stride], with an optional trailing gate
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
  const constant uint32_t& input_row_stride,
  const constant uint32_t& batch_dim,
  const bool has_kv SPECIALIZE,
  const bool has_rope SPECIALIZE,
  const uint32_t head_dim_idx AXIS(head_dim, 128),
  const uint32_t head_idx AXIS(num_q_heads + num_kv_heads.unwrap_or(0) * 2, 1),
  const uint32_t batch_idx AXIS(batch_dim, 1)
) {
  const uint32_t qkvg_head_idx = batch_idx * input_row_stride + head_idx * head_dim;
  const device ElementT* qkvg_head = qkvg + qkvg_head_idx;
  const bool is_query = !has_kv || head_idx < num_q_heads;
  const bool is_key = has_kv && head_idx >= num_q_heads && head_idx < num_q_heads + num_kv_heads;

  ElementT element = qkvg_head[head_dim_idx];
  if (has_rope && head_dim_idx < rope_dim && (is_query || is_key)) {
    element = apply_rope(qkvg_head, cosines, sines, batch_idx, head_dim_idx, rope_dim);
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
