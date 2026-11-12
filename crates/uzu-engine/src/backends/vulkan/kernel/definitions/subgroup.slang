implementing definitions;

public static const uint SIMD_SIZE = 32;

public uint subgroup_id() {
  return WaveGetWaveIndex();
}

public uint subgroup_invocation() {
  return subgroup_id() * WaveGetLaneCount() + WaveGetLaneIndex();
}

public T shuffle_xor<T>(T value, uint mask) {
  return spirv_asm {
    OpCapability GroupNonUniformShuffle;
    result:$$T = OpGroupNonUniformShuffleXor Subgroup $value $mask;
  };
}
