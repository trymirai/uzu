use std::num::NonZeroU32;

use crate::backends::metal::kernel::unified_matmul::gemm::UnifiedGemmSpecializationError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GroupSize(NonZeroU32);

impl GroupSize {
    pub(crate) fn try_new(value: u32) -> Result<Self, UnifiedGemmSpecializationError> {
        NonZeroU32::new(value).map(Self).ok_or(UnifiedGemmSpecializationError::ZeroGroupSize)
    }

    pub(crate) const fn get(self) -> u32 {
        self.0.get()
    }
}
