use bitflags::bitflags;

use crate::data_type::DataType;

bitflags! {
    /// Optional device capabilities that select kernels or that kernels require.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VkDeviceCapabilities: u32 {
        /// Round-to-nearest-even and preserved signed zeros, infinities and NaNs for 16- and 32-bit floats, which
        /// every generated kernel declares.
        const FLOAT_CONTROLS = 1 << 0;
        /// Compute pipelines may require full subgroups of 32 invocations, which kernels whose algorithm is 32 lanes
        /// wide pin.
        const SUBGROUP_SIZE_32 = 1 << 1;
        const MEMORY_BUDGET = 1 << 2;
        /// Subgroup-scope 16 x 16 x 16 BF16 cooperative matrices with FP32 accumulation.
        const BF16_TENSOR_CORES = 1 << 3;
        /// The FP16 counterpart of `BF16_TENSOR_CORES`, as reported; defaults also need exact products.
        const FP16_TENSOR_CORES = 1 << 4;
        /// Subgroup-scope 16 x 16 x 32 INT8 cooperative matrices with INT32 accumulation.
        const INT8_TENSOR_CORES_K32 = 1 << 5;
        /// Subgroup-scope 16 x 16 x 16 INT8 cooperative matrices with INT32 accumulation, reported independently of the
        /// 16 x 16 x 32 ones; the baseline prefers those where both exist.
        const INT8_TENSOR_CORES_K16 = 1 << 6;
        /// Either INT8 depth, so test it with `intersects`.
        const INT8_TENSOR_CORES = Self::INT8_TENSOR_CORES_K32.bits() | Self::INT8_TENSOR_CORES_K16.bits();
        /// VK_NV_cooperative_matrix2 per-element operations on INT8 tensor cores.
        const INT8_ELEMENT_FOLD = 1 << 7;
        /// Only unsigned packed dot products are accelerated.
        const UNSIGNED_DOT_PRODUCT = 1 << 8;
        /// Unsigned dot products on full native 128-lane subgroups, two per workgroup, reduced through shared memory.
        const GEMV_INT8_NATIVE_128 = 1 << 9;
        /// VK_KHR_shader_bfloat16's BF16 type, which kernels that store BF16 load, store and convert natively.
        const BF16_TYPE = 1 << 10;
        /// VK_KHR_calibrated_timestamps with the device time domain, which reads the clock of recorded timestamps.
        const CALIBRATED_TIMESTAMPS = 1 << 11;
    }
}

impl VkDeviceCapabilities {
    /// 16 x 16 cooperative matrices of `data_type` on pinned 32-wide subgroups, as the float tensor-core kernels take.
    pub fn float_tensor_cores(
        self,
        data_type: DataType,
    ) -> bool {
        self.contains(Self::SUBGROUP_SIZE_32)
            && match data_type {
                DataType::BF16 => self.contains(Self::BF16_TENSOR_CORES),
                DataType::F16 => self.contains(Self::FP16_TENSOR_CORES),
                _ => false,
            }
    }

    /// The default activation-format and weight-layout preference for INT8 GEMM: INT8 matrices on pinned 32-wide
    /// subgroups, unless only unsigned dot products are accelerated, where GEMV keeps the rows. A policy, not support:
    /// the GEMM cores' legality is their raw matrix shape and subgroup bits.
    pub fn prefers_int8_gemm(self) -> bool {
        self.contains(Self::SUBGROUP_SIZE_32)
            && self.intersects(Self::INT8_TENSOR_CORES)
            && !self.contains(Self::UNSIGNED_DOT_PRODUCT)
    }
}
