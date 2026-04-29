struct AttnParams {
  uint64_t q_strides[3];
  uint64_t k_strides[3];
  uint64_t v_strides[3];
  uint64_t o_strides[3];
  uint gqa_factor;
  float scale;
  uint q_len;
  uint k_len;
  uint q_off;
  uint nq_aligned;
  uint q_rem;
  uint nk;
  uint nk_aligned;
  uint k_rem;
};