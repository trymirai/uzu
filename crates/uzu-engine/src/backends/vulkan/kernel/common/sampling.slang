uint2 multiply_extended(uint a, uint b) {
  return spirv_asm {
    %pair_type = OpTypeStruct $$uint $$uint;
    %pair:%pair_type = OpUMulExtended $a $b;
    %low:$$uint = OpCompositeExtract %pair 0;
    %high:$$uint = OpCompositeExtract %pair 1;
    result:$$uint2 = OpCompositeConstruct %low %high;
  };
}

public uint4 sampling_words(uint2 key, uint offset) {
  var counter = uint4(offset, 0, 0, 0);
  for (uint round = 0; round < 10; round++) {
    let p0 = multiply_extended(0xd2511f53u, counter.x);
    let p1 = multiply_extended(0xcd9e8d57u, counter.z);
    counter = uint4(p1.y ^ counter.y ^ key.x, p1.x, p0.y ^ counter.w ^ key.y, p0.x);
    key += uint2(0x9e3779b9u, 0xbb67ae85u);
  }
  return counter;
}

public uint2 sampling_index(uint token, uint vocab_size) {
  let stride = vocab_size / 4096 + (vocab_size % 4096 != 0 ? 1u : 0u);
  return uint2(stride * (token % 1024) + token / 4096, (token / 1024) % 4);
}

public float sampling_uniform(uint word) {
  return float(max(word >> 8, 1u)) * (1.0 / 16777216.0);
}

public float sampling_noise(float uniform) {
  let x = 1.0 - uniform;
  let inner = x <= (1.0 / 1024.0) ? x * (1.0 + x * (0.5 + x / 3.0)) : -log(uniform);
  return -log(inner);
}
