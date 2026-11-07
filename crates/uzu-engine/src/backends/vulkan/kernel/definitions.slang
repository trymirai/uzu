module definitions;

// The kernel DSL is the one the WebGPU kernels use.
__include "../../slang/kernel/definitions/dsl.slang";
__include definitions.storage_float;
__include definitions.bf16;
__include definitions.subgroup;

// One access of a U at `element` of `pointer`, whose ALIGNMENT-byte alignment callers guarantee.
public U load_aligned<U, let ALIGNMENT : int, T, Access access>(Ptr<T, access> pointer, uint element) {
  return loadAligned<ALIGNMENT>(Ptr<U>(pointer + element));
}

// The four values of `pointer` from `element`, widened.
public float4 load_float4<T : IStorageFloat, Access access>(Ptr<T, access> pointer, uint element) {
  return float4(
    pointer[element].toFloat(), pointer[element + 1].toFloat(), pointer[element + 2].toFloat(),
    pointer[element + 3].toFloat()
  );
}

// The four 16-bit values packed in `words`, low halves first, widened.
public float4 unpack4<T : IStorageFloat>(uint2 words) {
  return float4(
    T.fromStorageBits(words.x).toFloat(), T.fromStorageBits(words.x >> 16).toFloat(),
    T.fromStorageBits(words.y).toFloat(), T.fromStorageBits(words.y >> 16).toFloat()
  );
}

public void store_aligned<let ALIGNMENT : int, U, T>(Ptr<T, Access.ReadWrite> pointer, uint element, U value) {
  storeAligned<ALIGNMENT>(Ptr<U>(pointer + element), value);
}

public static const float NEGATIVE_INFINITY = asfloat(0xff800000u);

// The integer dot of the 4 signed bytes of `a` and `b`, never saturating.
public int dot_i8x4(uint a, uint b) {
  return spirv_asm {
    OpCapability DotProduct;
    OpCapability DotProductInput4x8BitPacked;
    result:$$int = OpSDot $a $b PackedVectorFormat4x8Bit;
  };
}

// `acc` plus the integer dot of the 4 unsigned bytes of `a` and `b`; callers' sums never saturate.
public uint udot_u8x4(uint a, uint b, uint acc) {
  return spirv_asm {
    OpCapability DotProduct;
    OpCapability DotProductInput4x8BitPacked;
    result:$$uint = OpUDotAccSat $a $b $acc PackedVectorFormat4x8Bit;
  };
}

// Element `index` of `output` in place, otherwise of `input`; a branch, so an absent input is never read.
public float in_place_load<I : IStorageFloat, O : IStorageFloat>(
  Ptr<I, Access.Read> input,
  Ptr<O, Access.ReadWrite> output,
  uint index,
  bool in_place
) {
  if (in_place) {
    return output[index].toFloat();
  }
  return input[index].toFloat();
}
