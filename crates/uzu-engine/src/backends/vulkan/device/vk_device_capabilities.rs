use bitflags::bitflags;

use crate::data_type::DataType;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct VkDeviceCapabilities: u32 {
        const SUBGROUP_SIZE_32 = 1 << 0;
        const MEMORY_BUDGET = 1 << 1;
        const BF16_TENSOR_CORES = 1 << 2;
        const INT8_TENSOR_CORES_K32 = 1 << 3;
        const INT8_TENSOR_CORES_K16 = 1 << 4;
        const INT8_TENSOR_CORES = Self::INT8_TENSOR_CORES_K32.bits() | Self::INT8_TENSOR_CORES_K16.bits();
        const INT8_ELEMENT_FOLD = 1 << 5;
        const UNSIGNED_DOT_PRODUCT = 1 << 6;
        const BF16_TYPE = 1 << 7;
        const CALIBRATED_TIMESTAMPS = 1 << 8;
        const CALIBRATED_TIMESTAMPS_EXT = 1 << 9;
    }
}

impl VkDeviceCapabilities {
    pub fn float_tensor_cores(
        self,
        data_type: DataType,
    ) -> bool {
        self.contains(Self::SUBGROUP_SIZE_32) && data_type == DataType::BF16 && self.contains(Self::BF16_TENSOR_CORES)
    }

    pub fn prefers_int8_gemm(self) -> bool {
        self.contains(Self::SUBGROUP_SIZE_32)
            && self.intersects(Self::INT8_TENSOR_CORES)
            && !self.contains(Self::UNSIGNED_DOT_PRODUCT)
    }
}
