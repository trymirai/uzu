import definitions;
import common.workgroup_reduction;

// Online row softmax: each invocation keeps a running (max, normalizer) over strided chunks, merged across the
// workgroup; `has_sinks` seeds invocation 0 with (sinks[outer_index], 1). Masked tail elements contribute nothing.

static const uint SOFTMAX_THREADS = 256;
static const uint SOFTMAX_ELEMENTS_PER_THREAD = 4;
static const float SOFTMAX_LOWEST = asfloat(0xff7fffffu);  // -FLT_MAX

// Merges (max, normalizer) pairs.
struct SoftmaxCombine : IReduction {
  typealias Value = float2;
  static float2 combine(float2 a, float2 b) {
    float max_value = max(a.x, b.x);
    return float2(max_value, a.y * exp(a.x - max_value) + b.y * exp(b.x - max_value));
  }
  static float2 subgroup(float2 value) {
    float max_value = WaveActiveMax(value.x);
    return float2(max_value, WaveActiveSum(value.y * exp(value.x - max_value)));
  }
}

[[Variants("T", "float, half, bf16")]]
[[Kernel]] [[Public]] func Softmax<T: IStorageFloat>(
  Ptr<T, Access.ReadWrite> values,
  [[Optional("has_sinks")]] Ptr<T, Access.Read> sinks,
  uint row_length,
  uint outer_dim,
  uint batch_dim,
  [[Groups("if row_length == 0 { 0 } else { outer_dim }")]] uint outer_index,
  [[Groups("batch_dim")]] uint batch_index,
  [[Threads("256")]] uint tid,
  [[Specialize]] bool has_sinks,
) {
  let row = (outer_index * batch_dim + batch_index) * row_length;
  let stride = SOFTMAX_THREADS * SOFTMAX_ELEMENTS_PER_THREAD;

  float2 state = float2(SOFTMAX_LOWEST, 0.0);
  if (has_sinks && tid == 0) {
    state = float2(sinks[outer_index].toFloat(), 1.0);
  }

  // Pass 1: online (max, normalizer) over this thread's chunks.
  for (uint base = tid * SOFTMAX_ELEMENTS_PER_THREAD; base < row_length; base += stride) {
    float chunk[SOFTMAX_ELEMENTS_PER_THREAD];
    float chunk_max = state.x;
    for (uint i = 0; i < SOFTMAX_ELEMENTS_PER_THREAD; i++) {
      chunk[i] = SOFTMAX_LOWEST;
      if (base + i < row_length) {
        chunk[i] = values[row + base + i].toFloat();
      }
      chunk_max = max(chunk_max, chunk[i]);
    }
    float norm = state.y * exp(state.x - chunk_max);
    for (uint i = 0; i < SOFTMAX_ELEMENTS_PER_THREAD; i++) {
      if (base + i < row_length) {
        norm += exp(chunk[i] - chunk_max);
      }
    }
    state = float2(chunk_max, norm);
  }

  state = workgroup_reduce<SoftmaxCombine, SOFTMAX_THREADS>(state);
  let final_max = state.x;
  let inverse_norm = 1.0 / state.y;

  // Pass 2: normalize.
  for (uint base = tid * SOFTMAX_ELEMENTS_PER_THREAD; base < row_length; base += stride) {
    for (uint i = 0; i < SOFTMAX_ELEMENTS_PER_THREAD; i++) {
      if (base + i < row_length) {
        let index = row + base + i;
        values[index] = T.fromFloat(exp(values[index].toFloat() - final_max) * inverse_norm);
      }
    }
  }
}
