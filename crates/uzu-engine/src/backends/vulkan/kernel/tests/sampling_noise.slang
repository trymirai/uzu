import definitions;
import common.sampling;
import common.storage_address;

// Each input row holds seed low/high, token, vocabulary size and an independently chosen raw word.
[[Kernel]] [[Test]] func TestSamplingNoise(
  Ptr<uint, Access.Read> inputs,
  Ptr<uint, Access.ReadWrite> outputs,
  uint count,
  [[Axis("count", "64")]] uint index,
) {
  if (index < count) {
    let input = element_pointer(inputs, element_index(index, 5, 0));
    let output = element_pointer(outputs, element_index(index, 10, 0));
    let key = uint2(input[0], input[1]);
    let slot = sampling_index(input[2], input[3]);
    let words = sampling_words(key, slot.x);
    output[0] = slot.x;
    output[1] = slot.y;
    for (uint word = 0; word < 4; word++) {
      output[2 + word] = words[word];
    }
    output[6] = asuint(sampling_uniform(words[slot.y]));
    output[7] = asuint(sampling_gumbel(key, slot));
    output[8] = asuint(sampling_uniform(input[4]));
    output[9] = asuint(sampling_noise(sampling_uniform(input[4])));
  }
}

[[Kernel]] [[Test]] func TestSamplingUniform(
  Ptr<uint, Access.ReadWrite> outputs,
  uint start,
  uint count,
  [[Axis("count", "64")]] uint index,
) {
  if (index < count) {
    let output = element_pointer(outputs, element_index(index, 2, 0));
    let uniform = sampling_uniform((start + index) << 8);
    output[0] = asuint(uniform);
    output[1] = asuint(sampling_noise(uniform));
  }
}
