use crate::backends::{
    common::gpu_types::unified_gemm::QuantizedMetadataKind,
    metal::kernel::unified_matmul::gemm::UnifiedGemmSpecializationError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct QuantizedStorageFormat {
    pub(crate) bits_per_weight: u8,
    pub(crate) group_size: u32,
    pub(crate) metadata_kind: QuantizedMetadataKind,
}

impl QuantizedStorageFormat {
    pub(crate) fn validate(&self) -> Result<(), UnifiedGemmSpecializationError> {
        if !(1..=8).contains(&self.bits_per_weight) {
            return Err(UnifiedGemmSpecializationError::UnsupportedBitsPerWeight(self.bits_per_weight));
        }
        if self.group_size == 0 {
            return Err(UnifiedGemmSpecializationError::ZeroGroupSize);
        }
        Ok(())
    }
}
