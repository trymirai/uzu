module definitions;

__include definitions.dsl;
__include definitions.storage_float;
__include definitions.bf16;
__include definitions.float_variants;
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

public void store_aligned<let ALIGNMENT : int, U, T>(Ptr<T, Access.ReadWrite> pointer, uint element, U value) {
  storeAligned<ALIGNMENT>(Ptr<U>(pointer + element), value);
}

public static const float NEGATIVE_INFINITY = asfloat(0xff800000u);

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

// A kernel precondition the binding checks, a host expression: phase "new" over specializations, checked by the
// constructor before creating anything; phase "encode" also over uniforms, checked before recording.
[[__AttributeUsage(_AttributeTargets.Function)]]
struct PreconditionAttribute {
  string phase;
  string condition;
};
