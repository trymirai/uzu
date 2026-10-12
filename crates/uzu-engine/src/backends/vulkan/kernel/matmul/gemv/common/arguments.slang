import definitions;

// Metal gemv/common/arguments.h. As in Metal, b is a word pointer each B source reinterprets; Vulkan's
// GemmDTransform flags are the kernels' has_scale, accumulate, has_bias and has_soft_cap specializations (no RHT).

public struct GemvOperands<A : IModelFloat, B : IModelFloat, D : IModelFloat> {
  public Ptr<uint, Access.Read> b;
  public Ptr<B, Access.Read> scales;
  public Ptr<uint8_t, Access.Read> zero_points;
  public Ptr<B, Access.Read> biases;
  public Ptr<A, Access.Read> a;
  public Ptr<D, Access.ReadWrite> d;
  public Ptr<B, Access.Read> output_bias;
  public Ptr<uint, Access.Read> gather_indices;
}

public struct GemvParams {
  public uint in_vec_size;
  public uint out_vec_size;
  public uint scale_output_stride;
  public uint scale_group_stride;
  public uint zero_point_output_stride;
  public uint zero_point_group_stride;
  public uint batch_size;
  public float ab_scale;
  public float soft_cap;
  public bool has_scale;
  public bool accumulate;
  public bool has_bias;
  public bool has_soft_cap;
  public bool gathered;
  public bool signed_codes;
}

// Metal's reinterpret_cast of a device pointer: the same address as another element type. Loads through the result
// carry its type's natural alignment, which callers guarantee as Metal's do.
public Ptr<U, Access.Read> gemv_cast<U, T>(Ptr<T, Access.Read> pointer) {
  return spirv_asm {
    result:$$Ptr<U, Access.Read> = OpBitcast $pointer;
  };
}
