import definitions;
import generated.activation_type;
import generated.argmax;
import generated.kv_cache_update;
import generated.quantization;
import generated.ring;

static const uint threadgroup_size = 256;

[[Variants("T", "float, half")]]
[[Kernel]] func test_kernel_group_thread<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input_0,
  StructuredBuffer<T> input_1,
  RWStructuredBuffer<T> output,
  uint size,
  [[Groups("size.div_ceil(threadgroup_size)")]] uint group_idx,
  [[Threads("threadgroup_size")]] uint thread_idx,
) {
  uint idx = group_idx * threadgroup_size + thread_idx;

  if (idx < size) {
    output[idx] = input_0[idx] + input_1[idx];
  }
}

[[Variants("T", "float, half")]]
[[Variants("BS", "2, 4, 8")]]
[[Kernel]] func test_kernel_axis<T: __BuiltinFloatingPointType, let BS: uint32_t>(
  StructuredBuffer<T> input_0,
  StructuredBuffer<T> input_1,
  RWStructuredBuffer<T> output,
  uint size,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  output[idx] = input_0[idx] + input_1[idx];
}

[[Kernel]] func test_kernel_axis_float(
  StructuredBuffer<float> input_0,
  StructuredBuffer<float> input_1,
  RWStructuredBuffer<float> output,
  uint size,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  output[idx] = input_0[idx] + input_1[idx];
}

[[Variants("I", "float, half")]]
[[Variants("O", "float, half")]]
[[Kernel]] func test_kernel_multispecialize<I: __BuiltinFloatingPointType, O: __BuiltinFloatingPointType>(
  StructuredBuffer<I> input,
  RWStructuredBuffer<O> output,
  uint size,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  output[idx] = O(input[idx].toFloat());
}

[[Variants("T", "float, half")]]
[[Variants("BM", "16, 32, 64")]]
[[Variants("BN", "16, 32, 64")]]
[[Constraint("BM * BN <= 1024")]]
[[Kernel]] func test_kernel_constraint<T: __BuiltinFloatingPointType, let BM: uint32_t, let BN: uint32_t>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  uint rows,
  uint cols,
  [[Axis("rows", "BM")]] uint row,
  [[Axis("cols", "BN")]] uint col,
) {
  output[row * cols + col] = input[row * cols + col];
}

[[Variants("T", "float, half")]]
[[Kernel]] func test_kernel_specialize<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  uint size,
  [[Specialize]] bool in_place,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  if (in_place) {
    output[idx] = output[idx] + T(1.0);
  } else {
    output[idx] = input[idx];
  }
}

[[Variants("T", "float, half")]]
[[Kernel]] func test_kernel_optional<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  [[Optional("use_bias")]] StructuredBuffer<T> bias,
  RWStructuredBuffer<T> output,
  uint size,
  [[Specialize]] bool use_bias,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  T v = input[idx];
  if (use_bias) {
    v = v + bias[idx];
  }
  output[idx] = v;
}

[[Variants("T", "float, half")]]
[[Kernel]] func test_kernel_indirect<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  StructuredBuffer<uint> tile_map,
  RWStructuredBuffer<T> output,
  [[Groups("INDIRECT")]] uint tile_idx,
  [[Threads("64")]] uint lane,
) {
  uint element = tile_map[tile_idx] * 64 + lane;
  output[element] = input[element];
}

[[Variants("T", "float, half")]]
[[Kernel]] func test_kernel_constant_array<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  int factors[16],
  RWStructuredBuffer<T> output,
  uint size,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  output[idx] = input[idx] * T(factors[idx % 16]);
}

[[Variants("T", "float, half")]]
[[GroupShared("Scratch", "128")]]
[[Kernel]] func test_kernel_groupshared_sized<T: __BuiltinFloatingPointType, Scratch: IGroupShared<T>>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  uint batch_size,
  uint element_count,
  [[Groups("batch_size")]] uint batch_idx,
  [[Threads("128")]] uint lane,
) {
  T sum = T(0.0);
  for (uint i = lane; i < element_count; i += 128) {
    sum += input[batch_idx * element_count + i];
  }
  Scratch.store(lane, sum);
  GroupMemoryBarrierWithGroupSync();

  if (lane == 0) {
    T total = T(0.0);
    for (uint i = 0; i < 128; ++i) {
      total += Scratch.load(i);
    }
    output[batch_idx] = total;
  }
}

[[Variants("T", "float, half")]]
[[GroupShared("Carry", "1")]]
[[Kernel]] func test_kernel_groupshared_ref<T: __BuiltinFloatingPointType, Carry: IGroupShared<T>>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  uint batch_size,
  uint element_count,
  [[Groups("batch_size")]] uint batch_idx,
  [[Threads("128")]] uint lane,
) {
  if (lane == 0) {
    Carry.store(0, input[batch_idx]);
  }
  GroupMemoryBarrierWithGroupSync();
  if (lane < element_count) {
    output[batch_idx * element_count + lane] = Carry.load(0);
  }
}

groupshared float carry;

struct Carry: IGroupShared<float> {
  static func load(uint idx) -> float {
    return carry;
  }
  static func store(uint idx, float value) {
    carry = value;
  }
}

[shader("compute")]
[numthreads(128,1,1)]
func test_kernel_groupshared_ref_entry_float(
  StructuredBuffer<float> input,
  RWStructuredBuffer<float> output,
  uniform uint batch_size,
  uniform uint element_count,
  uint3 groupId : SV_GroupID,
  uint3 threadId : SV_GroupThreadID
) {
  test_kernel_groupshared_ref<float, Carry>(input, output, batch_size, element_count, groupId.x, threadId.x);
}

[[Variants("T", "float, half")]]
[[Kernel]] func test_kernel_axis2d<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  uint rows,
  uint cols,
  [[Axis("rows", "8")]] uint row,
  [[Axis("cols", "32")]] uint col,
) {
  output[row * cols + col] = input[row * cols + col];
}

[[Variants("T", "float, half")]]
[[Kernel]] func test_kernel_groups2d<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  uint rows,
  uint cols,
  [[Groups("rows.div_ceil(32)")]] uint row_block,
  [[Groups("cols.div_ceil(32)")]] uint col_block,
  [[Threads("32")]] uint lane_row,
  [[Threads("32")]] uint lane_col,
) {
  uint row = row_block * 32 + lane_row;
  uint col = col_block * 32 + lane_col;
  if (row < rows && col < cols) {
    output[row * cols + col] = input[row * cols + col];
  }
}

[[Variants("T", "float, half")]]
[[Kernel]] func test_kernel_enum_specialize<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  uint size,
  [[Specialize]] ActivationType act,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  float x = input[idx].toFloat();
  float y;
  if (act == ActivationType::SILU) {
    y = x / (1.0 + exp(-x));
  } else if (act == ActivationType::GELU) {
    y = 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
  } else if (act == ActivationType::TANH) {
    y = tanh(x);
  } else if (act == ActivationType::SOFTPLUS) {
    y = log(1.0 + exp(x));
  } else {
    y = x;
  }
  output[idx] = T(y);
}

[[Kernel]] func test_kernel_struct_buffer(
  StructuredBuffer<ArgmaxPair> pairs,
  RWStructuredBuffer<ArgmaxPair> out_pairs,
  RWStructuredBuffer<uint> indices,
  RingParams ring,
  uint size,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  ArgmaxPair p = pairs[idx];
  uint dst = (idx + ring.ring_offset) % ring.ring_length;
  out_pairs[dst] = p;
  indices[dst] = p.index;
}

[[Variants("T", "float, half")]]
[[Kernel]] func test_kernel_enum_runtime<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  uint size,
  QuantizationMode mode,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  float scale;
  if (mode == QuantizationMode::UINT4) {
    scale = 15.0;
  } else if (mode == QuantizationMode::INT8) {
    scale = 127.0;
  } else {
    scale = 255.0;
  }
  output[idx] = T(input[idx].toFloat() * scale);
}

[[Variants("T", "float, half")]]
[[Kernel]] func test_kernel_struct_scalar<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  Swap swap,
  uint stride,
  [[Axis("stride", "threadgroup_size")]] uint idx,
) {
  output[swap.destination * stride + idx] = input[swap.source * stride + idx];
}

[[Variants("T", "float, half")]]
[[Kernel]] func test_kernel_struct_constant<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  Swap[16] swap,
  uint stride,
  [[Axis("stride", "threadgroup_size")]] uint idx,
) {
  output[swap[8].destination * stride + idx] = input[swap[8].source * stride + idx];
}

[shader("compute")]
[numthreads(threadgroup_size,1,1)]
func test_kernel_struct_constant_entry_float(
  StructuredBuffer<float> input,
  RWStructuredBuffer<float> output,
  uniform Swap[16] swap,
  uniform uint stride,
  uint3 groupId : SV_GroupID,
  uint3 threadId : SV_GroupThreadID
) {
  test_kernel_struct_constant<float>(input, output, swap, stride, groupId.x * threadgroup_size + threadId.x);
}

[[Kernel]] func test_kernel_thread_context(
  RWStructuredBuffer<ThreadContext> output,
  uint size,
  [[Axis("size", "threadgroup_size")]] uint idx,
  ThreadContext thread_context
) {
  output[idx] = thread_context;
}
