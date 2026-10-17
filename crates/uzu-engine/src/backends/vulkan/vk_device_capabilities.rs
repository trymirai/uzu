use bitflags::bitflags;

bitflags! {
    /// Optional device capabilities that select kernels or that kernels require.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VkDeviceCapabilities: u32 {
        /// Round-to-nearest-even and preserved signed zeros, infinities and NaNs for 16- and 32-bit floats, which
        /// every generated kernel declares.
        const FLOAT_CONTROLS = 1 << 0;
        /// Compute pipelines may require full subgroups.
        const FULL_SUBGROUPS = 1 << 1;
        const MEMORY_BUDGET = 1 << 2;
        /// Subgroup-scope 16 x 16 x 16 BF16 cooperative matrices with FP32 accumulation on 32-lane subgroups.
        const BF16_TENSOR_CORES = 1 << 3;
        /// The FP16 counterpart of `BF16_TENSOR_CORES`.
        const FP16_TENSOR_CORES = 1 << 4;
        /// Subgroup-scope 16 x 16 x 32 INT8 cooperative matrices with INT32 accumulation on 32-lane subgroups.
        const INT8_TENSOR_CORES_K32 = 1 << 5;
        /// 16 x 16 x 16 INT8 cooperative matrices, where 16 x 16 x 32 ones are missing.
        const INT8_TENSOR_CORES_K16 = 1 << 6;
        /// Either INT8 depth, so test it with `intersects`.
        const INT8_TENSOR_CORES = Self::INT8_TENSOR_CORES_K32.bits() | Self::INT8_TENSOR_CORES_K16.bits();
        /// VK_NV_cooperative_matrix2 per-element operations on INT8 tensor cores.
        const INT8_ELEMENT_FOLD = 1 << 7;
        /// Packed 4 x 8-bit integer dot products on 32-lane subgroups.
        const INTEGER_DOT_PRODUCT = 1 << 8;
        /// Only unsigned packed dot products are accelerated.
        const UNSIGNED_DOT_PRODUCT = 1 << 9;
        /// Unsigned dot products on full native 128-lane subgroups, two per workgroup, with clustered reductions.
        const GEMV_INT8_NATIVE_128 = 1 << 10;
    }
}
