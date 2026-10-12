import common.activation;

// Philox4x32-10, in the CPU sampler's counter layout: offset in word 0, seed low/high as the key. UMulExtended
// preserves both halves of the product without requiring shader Int64.
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

// Same token-to-counter mapping as sampling::revidx. The quotient/remainder form of ceil avoids uint overflow.
public uint2 sampling_index(uint token, uint vocab_size) {
  let stride = vocab_size / 4096 + (vocab_size % 4096 != 0 ? 1u : 0u);
  return uint2(stride * (token % 1024) + token / 4096, (token / 1024) % 4);
}

// Exact uniform in [2^-24, 1 - 2^-24]. Converting the full 32-bit word instead can round to 1 and yield infinite noise.
public float sampling_uniform(uint word) {
  return float(max(word >> 8, 1u)) * (1.0 / 16777216.0);
}

public float sampling_noise(float uniform) {
  return -staged_log(-staged_log(uniform));
}

public float sampling_gumbel(uint2 key, uint2 index) {
  return sampling_noise(sampling_uniform(sampling_words(key, index.x)[index.y]));
}
