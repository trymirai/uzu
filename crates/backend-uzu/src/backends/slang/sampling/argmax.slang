import definitions;
import generated.argmax;

static const uint BLOCK_SIZE = 1024;

[[Variants("T", "float, half")]]
[[GroupShared("shared", "BLOCK_SIZE")]]
[[Public]] [[Kernel]] func ArgmaxSingle<T: __BuiltinFloatingPointType, shared: IGroupShared<ArgmaxPair>>(
  StructuredBuffer<T> logits_data,
  RWStructuredBuffer<uint> final_tokens,
  uint batch_size,
  uint vocab_size,
  [[Groups("batch_size")]] uint batch_idx,
  [[Threads("1024")]] uint local_id,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[GroupShared("shared", "BLOCK_SIZE")]]
[[Public]] [[Kernel]] func ArgmaxMain<T: __BuiltinFloatingPointType, shared: IGroupShared<ArgmaxPair>>(
  StructuredBuffer<T> logits_data,
  RWStructuredBuffer<ArgmaxPair> partial_results,
  uint batch_size,
  uint vocab_size,
  ThreadContext thread_context,
  [[Groups("batch_size")]] uint batch_idx,
  [[Groups("vocab_size.div_ceil(1024 * 4)")]] uint vocab_group_idx,
  [[Threads("1024")]] uint local_id,
) {
  // TODO
}

[[GroupShared("shared", "BLOCK_SIZE")]]
[[Public]] [[Kernel]] func ArgmaxFinal<shared: IGroupShared<ArgmaxPair>>(
  StructuredBuffer<ArgmaxPair> partial_results,
  RWStructuredBuffer<uint> final_tokens,
  uint batch_size,
  uint vocab_size,
  ThreadContext thread_context,
  [[Groups("batch_size")]] uint batch_idx,
  [[Threads("1024")]] uint local_id,
) {
  // TODO
}