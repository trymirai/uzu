import definitions;
import common.activation;

[[Variants("T", "float, half, bf16")]]
[[Kernel]] [[Public]] func GatedResidualExpand<T: IStorageFloat>(
  Ptr<T, Access.Read> input,
  Ptr<T, Access.ReadWrite> output,
  uint model_dim,
  uint stream_count,
  uint length,
  [[Axis("length", "256")]] uint position,
) {
  let row = position / (stream_count * model_dim);
  output[position] = input[row * model_dim + position % model_dim];
}

[[Variants("T", "float, half, bf16")]]
[[Kernel]] [[Public]] func GatedResidualLowRankActivation<T: IStorageFloat>(
  Ptr<T, Access.ReadWrite> values,
  uint stream_count,
  uint length,
  [[Axis("length", "256")]] uint position,
) {
  let x = values[position].toFloat() / float(stream_count);
  values[position] = T.fromFloat(x * sigmoid(x));
}

[[Variants("T", "float, half, bf16")]]
[[Kernel]] [[Public]] func GatedResidualMix<T: IStorageFloat>(
  Ptr<T, Access.Read> streams,
  Ptr<T, Access.Read> logits,
  Ptr<T, Access.ReadWrite> output,
  uint model_dim,
  uint stream_count,
  uint length,
  [[Axis("length", "256")]] uint position,
) {
  let row_offset = position / model_dim * stream_count * model_dim + position % model_dim;
  var sum = 0.0;
  for (uint stream = 0; stream < stream_count; stream++) {
    let index = row_offset + stream * model_dim;
    sum += streams[index].toFloat() * sigmoid(logits[index].toFloat());
  }
  output[position] = T.fromFloat(sum / float(stream_count));
}

[[Variants("T", "float, half, bf16")]]
[[Kernel]] [[Public]] func GatedResidualUpdate<T: IStorageFloat>(
  Ptr<T, Access.Read> streams,
  Ptr<T, Access.Read> injection_logits,
  Ptr<T, Access.Read> update,
  Ptr<T, Access.ReadWrite> output,
  uint model_dim,
  uint stream_count,
  uint length,
  [[Axis("length", "256")]] uint position,
) {
  let row_stream = position / model_dim;
  let weight = 2.0 * sigmoid(injection_logits[row_stream].toFloat() / float(stream_count));
  let update_value = update[row_stream / stream_count * model_dim + position % model_dim].toFloat();
  output[position] = T.fromFloat(streams[position].toFloat() + weight * update_value);
}
