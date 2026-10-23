implementing definitions;

// A simdgroup: SIMD_SIZE invocations in subgroup_invocation order, reduced by segment_reduce at any subgroup width.
public static const uint SIMD_SIZE = 32;

public uint subgroup_id() {
  return WaveGetWaveIndex();
}

// This invocation's index with the workgroup numbered subgroup by subgroup, lane by lane: aligned power-of-two runs no
// wider than the subgroup are runs of its lanes.
public uint subgroup_invocation() {
  return subgroup_id() * WaveGetLaneCount() + WaveGetLaneIndex();
}

// `value` of lane WaveGetLaneIndex() ^ mask, an active lane of this subgroup. Plain Slang has no ShuffleXor intrinsic.
public T shuffle_xor<T>(T value, uint mask) {
  return spirv_asm {
    OpCapability GroupNonUniformShuffle;
    result:$$T = OpGroupNonUniformShuffleXor Subgroup $value $mask;
  };
}
