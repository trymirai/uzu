import definitions;

// Mean over the sequence of [batch, seq_len, hidden_dim] into [batch, hidden_dim], summed in sequence order in FP32.
[[Kernel]] [[Public]] func PoolingMean<T: IStorageFloat>(
  Ptr<T, Access.Read> input,
  Ptr<T, Access.ReadWrite> output,
  uint seq_len,
  uint hidden_dim,
  uint batch_size,
  [[Axis("hidden_dim", "16")]] uint dim_idx,
  [[Axis("batch_size", "16")]] uint batch_idx,
) {
  float sum = 0.0;
  for (uint seq_idx = 0; seq_idx < seq_len; seq_idx++) {
    sum += input[(batch_idx * seq_len + seq_idx) * hidden_dim + dim_idx].toFloat();
  }
  output[batch_idx * hidden_dim + dim_idx] = T.fromFloat(sum / float(seq_len));
}
