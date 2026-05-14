use crate::backends::common::gpu_types::unified_gemm::{
    GemmAlignment, GemmComputeKind, GemmInputPrologueKind, GemmOutputTransformKind, GemmTilingConfig,
    GemmWeightPrologueKind,
};

use super::UnifiedGemmSpecializationError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct UnifiedGemmSpecialization {
    pub(crate) tiling_config: GemmTilingConfig,
    pub(crate) input_prologue: GemmInputPrologueKind,
    pub(crate) compute: GemmComputeKind,
    pub(crate) output_transform: GemmOutputTransformKind,
    pub(crate) alignment: GemmAlignment,
    pub(crate) weight_prologue: GemmWeightPrologueKind,
    pub(crate) bits_per_weight: u32,
    pub(crate) group_size: u32,
}

impl UnifiedGemmSpecialization {
    pub(crate) fn try_validate(self) -> Result<Self, UnifiedGemmSpecializationError> {
        let tile = &self.tiling_config;
        if [
            tile.threadgroup_m,
            tile.threadgroup_n,
            tile.threadgroup_k,
            tile.simdgroup_m,
            tile.simdgroup_n,
            tile.simdgroup_k,
            tile.fragment_m,
            tile.fragment_n,
            tile.fragment_k,
            tile.simdgroups_m,
            tile.simdgroups_n,
        ]
        .contains(&0)
        {
            return Err(UnifiedGemmSpecializationError::ZeroTileDimension);
        }

        if self.group_size != 0 && tile.threadgroup_k > self.group_size {
            return Err(UnifiedGemmSpecializationError::ThreadgroupKExceedsGroupSize {
                threadgroup_k: tile.threadgroup_k,
                group_size: self.group_size,
            });
        }

        Ok(self)
    }
}
