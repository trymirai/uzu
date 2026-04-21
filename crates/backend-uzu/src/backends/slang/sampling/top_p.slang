import definitions;

static const uint BLOCK_SIZE = 1024;
static const uint BLOCK_SIZE_IN_SIMDS = BLOCK_SIZE / 32;

[[Variants("T", "float, half")]]
[[GroupShared("shared_reduce_buffer", "BLOCK_SIZE_IN_SIMDS")]]
[[Public]] [[Kernel]] func TopP<T: __BuiltinFloatingPointType, shared_reduce_buffer: IGroupShared<float>>(
  [[Optional("!in_place")]] StructuredBuffer<T> logits,
  RWStructuredBuffer<T> processed_logits,
  uint batch_size,
  uint vocab_size,
  float top_p,
  ThreadContext thread_context,
  [[Groups("batch_size")]] uint batch_idx,
  [[Threads("1024")]] uint thread_idx,
  [[Specialize]] bool in_place,
) {
  // TODO
}