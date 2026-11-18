implementing definitions;

public static const uint SIMD_SIZE = 32;
// Subgroup operations are slow on subgroups wider than this, so those use shared memory.
public static const uint MAX_FAST_SUBGROUP_LANES = 64;

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
