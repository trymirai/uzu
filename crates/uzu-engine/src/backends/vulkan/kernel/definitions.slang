module definitions;

__include "../../slang/kernel/definitions/dsl.slang";
__include definitions.storage_float;
__include definitions.bf16;
__include definitions.subgroup;

public U load_aligned<U, let ALIGNMENT : int, T, Access access>(Ptr<T, access> pointer, uint element) {
  return loadAligned<ALIGNMENT>(Ptr<U>(pointer + element));
}

public float4 load_float4<T : IStorageFloat, Access access>(Ptr<T, access> pointer, uint element) {
  return float4(
    pointer[element].toFloat(), pointer[element + 1].toFloat(), pointer[element + 2].toFloat(),
    pointer[element + 3].toFloat()
  );
}

public void store_aligned<let ALIGNMENT : int, U, T>(Ptr<T, Access.ReadWrite> pointer, uint element, U value) {
  storeAligned<ALIGNMENT>(Ptr<U>(pointer + element), value);
}

public static const float NEGATIVE_INFINITY = asfloat(0xff800000u);

public int dot_i8x4(uint a, uint b) {
  return spirv_asm {
    OpCapability DotProduct;
    OpCapability DotProductInput4x8BitPacked;
    result:$$int = OpSDot $a $b PackedVectorFormat4x8Bit;
  };
}

public uint udot_u8x4(uint a, uint b, uint acc) {
  return spirv_asm {
    OpCapability DotProduct;
    OpCapability DotProductInput4x8BitPacked;
    result:$$uint = OpUDotAccSat $a $b $acc PackedVectorFormat4x8Bit;
  };
}

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
