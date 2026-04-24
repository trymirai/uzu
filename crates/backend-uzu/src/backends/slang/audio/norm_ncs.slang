import definitions;

static const uint AUDIO_NORM_NCS_MAX_SIMDS = 32;

[[Variants("T", "float, half")]]
[[GroupShared("shared_mean", "AUDIO_NORM_NCS_MAX_SIMDS")]]
[[GroupShared("shared_variance", "AUDIO_NORM_NCS_MAX_SIMDS")]]
[[Public]] [[Kernel]] func AudioNormNcs<T: __BuiltinFloatingPointType, shared_mean: IGroupShared<float>, shared_variance: IGroupShared<float>>(
  StructuredBuffer<T> input,
  StructuredBuffer<T> scales,
  StructuredBuffer<T> bias,
  RWStructuredBuffer<T> output,
  StructuredBuffer<int> lengths,
  int channels,
  int seq_len,
  float epsilon,
  int subtract_mean,
  int batch_size,
  ThreadContext thread_context,
  [[Groups("batch_size")]] uint b,
  [[Groups("seq_len")]] uint t,
  [[Threads("256")]] uint lid,
) {
  // TODO
}