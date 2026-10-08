import definitions;
import generated.activation_type;

// Echoes every typed uniform and specialization into eight words per element, reading optional values only under
// their presence conditions, in one or two rows chosen by the specialized enum.

static const uint TYPED_CONSTANTS_ABSENT = 0xa5a5a5a5u;

[[Kernel]] [[Test]] func TestTypedConstants(
  Ptr<uint, Access.ReadWrite> output,
  uint size,
  ActivationType activation,
  [[Optional("has_offset")]] uint offset,
  [[Optional("mode == ActivationType::IDENTITY")]] float scale,
  [[Groups("size.div_ceil(32)")]] uint group_x,
  [[Groups("mode as u32 % 2 + 1")]] uint row,
  [[Threads("32")]] uint lane,
  [[Specialize]] ActivationType mode,
  [[Specialize]] uint word,
  [[Specialize]] bool has_offset,
  [[Specialize]] [[Optional("has_offset")]] uint group,
  [[Specialize]] [[Optional("mode != ActivationType::SILU")]] ActivationType secondary,
) {
  let index = group_x * 32 + lane;
  if (index >= size) {
    return;
  }
  let base = (row * size + index) * 8;
  output[base] = uint(activation);
  output[base + 1] = uint(mode);
  output[base + 2] = word;
  output[base + 3] = has_offset ? 1u : 0u;
  output[base + 4] = has_offset ? offset + index : TYPED_CONSTANTS_ABSENT;
  output[base + 5] = mode == ActivationType.IDENTITY ? asuint(scale) : TYPED_CONSTANTS_ABSENT;
  output[base + 6] = has_offset ? group : TYPED_CONSTANTS_ABSENT;
  output[base + 7] = mode != ActivationType.SILU ? uint(secondary) : TYPED_CONSTANTS_ABSENT;
}

// Echoes the pipeline's activation, both regular specializations and the uniform after it into four words per element.
[[Kernel]] [[Test]] func TestPipelineVariants(
  Ptr<uint, Access.ReadWrite> output,
  uint size,
  [[PipelineVariants]] ActivationType activation,
  uint word,
  [[Axis("size", "32")]] uint index,
  [[Specialize]] bool flag,
  [[Specialize]] uint group,
) {
  output[index * 4] = uint(activation);
  output[index * 4 + 1] = flag ? 1u : 0u;
  output[index * 4 + 2] = group;
  output[index * 4 + 3] = word + index;
}

// Wrapping signed and unsigned byte arithmetic on scalar byte storage; the odd term for negative values exposes a load
// that zero-extends signed bytes.
[[Kernel]] [[Test]] func TestByteStorage(
  Ptr<int8_t, Access.Read> signed_input,
  Ptr<uint8_t, Access.Read> unsigned_input,
  Ptr<int8_t, Access.ReadWrite> signed_output,
  Ptr<uint8_t, Access.ReadWrite> unsigned_output,
  uint size,
  [[Axis("size", "32")]] uint index,
) {
  let signed_value = int(signed_input[index]);
  let unsigned_value = uint(unsigned_input[index]);
  signed_output[index] = int8_t(signed_value * 3 + int(unsigned_value) + (signed_value < 0 ? 1 : 0));
  unsigned_output[index] = uint8_t(unsigned_value * 7u + uint(signed_value));
}
