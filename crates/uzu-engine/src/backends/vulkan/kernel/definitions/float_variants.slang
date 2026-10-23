implementing definitions;

// Type-parameter constraints that restrict a kernel's storage types to an exact variant set; conversions are the
// inherited IStorageFloat ones.

// FP32 only: accumulation types.
public interface IFloat32 : IStorageFloat {}

// F32, F16 or BF16: model storage types.
public interface IModelFloat : IStorageFloat {}

// Native 16-bit tensor operands.
public interface IModelFloat16 : IModelFloat {}

public extension float : IFloat32, IModelFloat {}
public extension half : IModelFloat16 {}
