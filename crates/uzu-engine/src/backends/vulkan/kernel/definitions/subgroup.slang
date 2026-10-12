implementing definitions;

// The pinned Slang 2026.2 SDK has no wave-index/wave-count intrinsics; load the SPIR-V SubgroupId and NumSubgroups
// builtins directly.
public uint subgroup_id() {
  return spirv_asm { result:$$uint = OpLoad builtin(SubgroupId:uint); };
}

public uint subgroup_count() {
  return spirv_asm { result:$$uint = OpLoad builtin(NumSubgroups:uint); };
}

// Whether a workgroup of `subgroups` x 32 invocations runs as exactly `subgroups` subgroups of 32 lanes, so none is
// partial. Uniform across the workgroup.
public bool subgroups_full32(uint subgroups) {
  return WaveGetLaneCount() == 32 && subgroup_count() == subgroups;
}
