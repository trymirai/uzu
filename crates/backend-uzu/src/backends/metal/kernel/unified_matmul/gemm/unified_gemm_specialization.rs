use crate::backends::metal::kernel::unified_matmul::gemm::{
    GemmAlignment, GemmComputeKind, GemmInputPrologueKind, GemmOutputTransformKind, GemmTilingConfig,
    UnifiedGemmSpecializationError, WeightsStorageFormat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct UnifiedGemmSpecialization {
    pub(crate) tiling_config: GemmTilingConfig,
    pub(crate) input_prologue: GemmInputPrologueKind,
    pub(crate) compute: GemmComputeKind,
    pub(crate) output: GemmOutputTransformKind,
    pub(crate) alignment: GemmAlignment,
    pub(crate) weights_storage: WeightsStorageFormat,
}

impl UnifiedGemmSpecialization {
    pub(crate) const fn new(
        tiling_config: GemmTilingConfig,
        input_prologue: GemmInputPrologueKind,
        compute: GemmComputeKind,
        output: GemmOutputTransformKind,
        alignment: GemmAlignment,
        weights_storage: WeightsStorageFormat,
    ) -> Self {
        Self {
            tiling_config,
            input_prologue,
            compute,
            output,
            alignment,
            weights_storage,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), UnifiedGemmSpecializationError> {
        let tile = &self.tiling_config;
        if tile.threadgroup_m == 0
            || tile.threadgroup_n == 0
            || tile.threadgroup_k == 0
            || tile.simdgroup_m == 0
            || tile.simdgroup_n == 0
            || tile.simdgroup_k == 0
            || tile.fragment_m == 0
            || tile.fragment_n == 0
            || tile.fragment_k == 0
            || tile.simdgroups_m == 0
            || tile.simdgroups_n == 0
        {
            return Err(UnifiedGemmSpecializationError::ZeroTileDimension);
        }

        let group_size = self.weights_storage.group_size();
        if group_size != 0 && tile.threadgroup_k > group_size {
            return Err(UnifiedGemmSpecializationError::ThreadgroupKExceedsGroupSize {
                threadgroup_k: tile.threadgroup_k,
                group_size,
            });
        }

        Ok(())
    }
}
