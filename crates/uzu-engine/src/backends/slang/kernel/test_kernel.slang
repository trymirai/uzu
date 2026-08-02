import definitions;
import generated.activation_type;
import generated.kv_cache_update;
import generated.quantization;
import generated.ring;

// TODO: remove after real kernels working

static const uint threadgroup_size = 256;

[[Variants("T", "float, half")]]
[[Kernel]] func TestKernelGroupThread<T: __BuiltinFloatingPointType>(
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
[[Kernel]] func TestKernelAxis<T: __BuiltinFloatingPointType, let BS: uint32_t>(
  StructuredBuffer<T> input_0,
  StructuredBuffer<T> input_1,
  RWStructuredBuffer<T> output,
  uint size,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  output[idx] = input_0[idx] + input_1[idx];
}

[[Kernel]] func TestKernelAxisFloat(
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
[[Kernel]] func TestKernelMultispecialize<I: __BuiltinFloatingPointType, O: __BuiltinFloatingPointType>(
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
[[Kernel]] func TestKernelConstraint<T: __BuiltinFloatingPointType, let BM: uint32_t, let BN: uint32_t>(
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
[[Kernel]] func TestKernelSpecialize<T: __BuiltinFloatingPointType>(
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
[[Kernel]] func TestKernelOptional<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  [[Optional("use_bias")]] StructuredBuffer<T> bias,
  RWStructuredBuffer<T> output,
  [[Optional("use_bias")]] uint size,
  [[Specialize]] bool use_bias,
  [[Axis("size.unwrap_or(257)", "threadgroup_size")]] uint idx,
) {
  T v = input[idx];
  if (use_bias) {
    v = v + bias[idx];
  }
  output[idx] = v;
}

[[Variants("T", "float, half")]]
[[Kernel]] func TestKernelIndirect<T: __BuiltinFloatingPointType>(
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
[[Kernel]] func TestKernelConstantArray<T: __BuiltinFloatingPointType>(
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
[[Kernel]] func TestKernelGroupsharedSized<T: __BuiltinFloatingPointType, Scratch: IGroupShared<T>>(
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
[[Kernel]] func TestKernelGroupsharedRef<T: __BuiltinFloatingPointType, Carry: IGroupShared<T>>(
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
func TestKernelGroupsharedRefEntryFloat(
  StructuredBuffer<float> input,
  RWStructuredBuffer<float> output,
  uniform uint batch_size,
  uniform uint element_count,
  uint3 groupId : SV_GroupID,
  uint3 threadId : SV_GroupThreadID
) {
  TestKernelGroupsharedRef<float, Carry>(input, output, batch_size, element_count, groupId.x, threadId.x);
}

[[Variants("T", "float, half")]]
[[Kernel]] func TestKernelAxis2d<T: __BuiltinFloatingPointType>(
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
[[Kernel]] func TestKernelGroups2d<T: __BuiltinFloatingPointType>(
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
[[Kernel]] func TestKernelEnumSpecialize<T: __BuiltinFloatingPointType>(
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
  } else if (act == ActivationType::GELUApprox) {
    y = 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
  } else if (act == ActivationType::SOFTPLUS) {
    y = log(1.0 + exp(x));
  } else {
    y = x;
  }
  output[idx] = T(y);
}

[[Kernel]] func TestKernelStructBuffer(
  StructuredBuffer<Copy> pairs,
  RWStructuredBuffer<Copy> out_pairs,
  RWStructuredBuffer<uint> indices,
  RingParams ring,
  uint size,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  Copy p = pairs[idx];
  uint dst = (idx + ring.ring_offset) % ring.ring_length;
  out_pairs[dst] = p;
  indices[dst] = p.source;
}

[[Variants("T", "float, half")]]
[[Kernel]] func TestKernelEnumRuntime<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  uint size,
  QuantizationMode mode,
  [[Axis("size", "threadgroup_size")]] uint idx,
) {
  float scale;
  if (mode == QuantizationMode::U4) {
    scale = 15.0;
  } else if (mode == QuantizationMode::I8) {
    scale = 127.0;
  } else {
    scale = 255.0;
  }
  output[idx] = T(input[idx].toFloat() * scale);
}

[[Variants("T", "float, half")]]
[[Kernel]] func TestKernelStructScalar<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  Copy copy,
  uint stride,
  [[Axis("stride", "threadgroup_size")]] uint idx,
) {
  output[copy.destination * stride + idx] = input[copy.source * stride + idx];
}

[[Variants("T", "float, half")]]
[[Kernel]] func TestKernelStructConstant<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  Copy[16] copy,
  uint stride,
  [[Axis("stride", "threadgroup_size")]] uint idx,
) {
  output[copy[8].destination * stride + idx] = input[copy[8].source * stride + idx];
}

[shader("compute")]
[numthreads(threadgroup_size,1,1)]
func TestKernelStructConstantEntryFloat(
  StructuredBuffer<float> input,
  RWStructuredBuffer<float> output,
  uniform Copy[16] copy,
  uniform uint stride,
  uint3 groupId : SV_GroupID,
  uint3 threadId : SV_GroupThreadID
) {
  TestKernelStructConstant<float>(input, output, copy, stride, groupId.x * threadgroup_size + threadId.x);
}

[[Kernel]] func TestKernelThreadContext(
  RWStructuredBuffer<ThreadContext> output,
  uint size,
  [[Axis("size", "threadgroup_size")]] uint idx,
  ThreadContext thread_context
) {
  output[idx] = thread_context;
}
