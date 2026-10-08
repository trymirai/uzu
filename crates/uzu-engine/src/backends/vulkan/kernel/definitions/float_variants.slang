implementing definitions;

// Type-parameter constraints that restrict a kernel's storage types to an exact variant set; conversions are the
// inherited IStorageFloat ones.

// FP32 only: accumulation types.
public interface IFloat32 : IStorageFloat {}

// F32 or BF16: model storage types.
public interface IModelFloat : IStorageFloat {}

// BF16 only: storage of kernels whose CPU kernel admits nothing else.
public interface IBFloat16 : IStorageFloat {}

public extension float : IFloat32, IModelFloat {}
