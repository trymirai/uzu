struct AttnParams {
  int64_t q_strides[3];
  int64_t k_strides[3];
  int64_t v_strides[3];
  int64_t o_strides[3];
  int gqa_factor;
  float scale;
  int q_len;
  int k_len;
  int q_off;
  int nq_aligned;
  int q_rem;
  int nk;
  int nk_aligned;
  int k_rem;
};