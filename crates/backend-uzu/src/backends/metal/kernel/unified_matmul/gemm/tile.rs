use crate::backends::metal::kernel::unified_matmul::gemm::UnifiedGemmSpecializationError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GemmTile {
    pub(crate) threadgroup_m: u32,
    pub(crate) threadgroup_n: u32,
    pub(crate) threadgroup_k: u32,
    pub(crate) simdgroup_m: u32,
    pub(crate) simdgroup_n: u32,
    pub(crate) simdgroup_k: u32,
    pub(crate) fragment_m: u32,
    pub(crate) fragment_n: u32,
    pub(crate) fragment_k: u32,
    pub(crate) simdgroups_m: u32,
    pub(crate) simdgroups_n: u32,
}

impl GemmTile {
    pub(crate) fn validate(&self) -> Result<(), UnifiedGemmSpecializationError> {
        if self.threadgroup_m == 0
            || self.threadgroup_n == 0
            || self.threadgroup_k == 0
            || self.simdgroup_m == 0
            || self.simdgroup_n == 0
            || self.simdgroup_k == 0
            || self.fragment_m == 0
            || self.fragment_n == 0
            || self.fragment_k == 0
            || self.simdgroups_m == 0
            || self.simdgroups_n == 0
        {
            return Err(UnifiedGemmSpecializationError::ZeroTileDimension);
        }

        Ok(())
    }
}
